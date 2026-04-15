"""
library_ai.py — Core AI engine for LibraryAI.

Refactored from the original js_ai.py CLI tool. This module provides the
LibraryAI class which implements:
  - Retrieval-Augmented Generation (RAG) over ChromaDB document store
  - Per-user chat memory with vector similarity search
  - Multi-format document ingestion (PDF, TXT, DOCX)
  - Live web search fallback via DuckDuckGo
  - Streaming token generation via LangChain + Ollama

This module is consumed by main.py (FastAPI) and should not be run directly.
"""

import os
import re
import logging
from typing import Tuple, List, Dict, Generator, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import httpx
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.tools import DuckDuckGoSearchRun

logger = logging.getLogger(__name__)


class LibraryAI:
    def __init__(self, chroma_dir: str, ollama_base_url: str = "http://localhost:11434"):
        """Initialize the AI engine with connections to ChromaDB, Ollama, and HuggingFace.

        Args:
            chroma_dir: Filesystem path for ChromaDB persistence. Two collections
                        are created: the default collection for documents and
                        'chat_memory' for user conversation history.
            ollama_base_url: Base URL of the Ollama server (e.g. http://localhost:11434).
        """
        embed_device = os.getenv("EMBED_DEVICE", "cuda")
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-en-v1.5",
            model_kwargs={"device": embed_device},
        )

        self.vector_db = Chroma(
            persist_directory=chroma_dir, embedding_function=self.embedding_model
        )
        self.memory_db = Chroma(
            collection_name="chat_memory",
            persist_directory=chroma_dir,
            embedding_function=self.embedding_model,
        )

        self.llm = OllamaLLM(model="llama3:8b", base_url=ollama_base_url)
        self.web_search = DuckDuckGoSearchRun()

        self.recent_chat_buffers: Dict[str, List[str]] = {}
        self._setup_chains()

    def _setup_chains(self):
        """Configure LangChain prompt templates and chain pipelines.

        Creates two chains:
        - rephrase_chain: Rewrites a follow-up question into a standalone search query
          by resolving pronouns and references from the recent chat buffer.
        - qa_chain: Generates an answer given context (documents or web results)
          and past conversation history.
        """
        rephrase_template = """Given the following conversation and a follow-up question, rephrase the follow-up question to be a highly specific standalone search query.

        CRITICAL INSTRUCTIONS:
        - If the user uses pronouns like "I", "me", or "my", rewrite them as "the user" or "the user's".
        - Resolve any vague words like "it" or "that" based on the Recent Chat History.
        - ONLY output the standalone question. Do not answer it. Do not add conversational filler.

        Recent Chat History:
        {recent_history}

        Follow Up Input: {question}
        Standalone Question:"""

        qa_template = """You are a highly intelligent AI assistant.
        You have access to two equal sources of truth:
        1. Past Conversation Context (memories and facts the user explicitly told you).
        2. Library Context (documents retrieved from the database or live web search).

        Answer the user's question using information from EITHER of these sources.
        If the user asks about a personal fact they previously shared, retrieve it from the Past Conversation Context.
        If you cannot find the answer in either source, simply say "I cannot find the answer." Do not invent or hallucinate information.

        Past Conversation Context:
        {chat_history}

        Library Context:
        {context}

        Question: {question}
        Helpful Answer:"""

        self.rephrase_chain = PromptTemplate.from_template(rephrase_template) | self.llm
        self.qa_chain = PromptTemplate.from_template(qa_template) | self.llm

        # --- Research: clean query extraction ---
        research_query_template = """You are a research librarian. Convert this user request into 2-3 clean academic search queries.

RULES:
- Strip meta-instructions like "find papers about", "get me sources on", "I need research on"
- Extract the CORE scientific/technical topic
- Generate variations: one specific, one moderately broad, one broad
- Keep each query between 2 and 8 words
- Do NOT number them, do NOT add bullets or explanation
- Output ONLY the queries, one per line

Recent context: {recent_history}
User request: {question}
Search queries:"""

        self.research_query_chain = PromptTemplate.from_template(research_query_template) | self.llm

        # --- Research: synthesis from papers ---
        research_template = """You are an expert academic research analyst synthesizing findings from scientific papers.

RECENT CONVERSATION:
{recent_history}

PAPERS FOUND:
{papers_context}

INSTRUCTIONS:
- Write a comprehensive, well-structured research overview that answers the user's question.
- Use markdown formatting with clear section headers (##).
- Organize into sections like: ## Overview, ## Key Findings, ## Technical Approaches, ## Challenges & Open Problems, ## Future Directions
- Reference specific papers using their number in square brackets, e.g. [1], [2], [3]. These numbers correspond to the paper numbers listed above.
- Include specific numbers, metrics, benchmarks, and technical details from paper abstracts.
- Compare and contrast different approaches across papers.
- If papers only partially cover the topic, clearly note what gaps remain.
- Use the recent conversation to understand context, follow-up references, and pronouns.
- Be thorough but concise.
- Do NOT invent facts not present in the papers.

User's Research Question: {question}

## Research Overview"""

        self.research_chain = PromptTemplate.from_template(research_template) | self.llm

    # ==========================================
    # PER-USER SHORT-TERM BUFFER
    # ==========================================

    def _get_buffer(self, user_id: str) -> List[str]:
        if user_id not in self.recent_chat_buffers:
            self.recent_chat_buffers[user_id] = []
        return self.recent_chat_buffers[user_id]

    def _update_buffer(self, user_id: str, entry: str):
        """Append an entry to the user's buffer, keeping only the last 6 entries."""
        buf = self._get_buffer(user_id)
        buf.append(entry)
        if len(buf) > 6:
            self.recent_chat_buffers[user_id] = buf[-6:]

    # ==========================================
    # MEMORY COMMANDS (User Facts)
    # ==========================================

    def save_fact(self, user_id: str, fact: str) -> str:
        """Store an explicit user fact in the vector memory database.

        Args:
            user_id: Unique identifier for the user (from JWT).
            fact: The fact text to save (e.g. "My favorite color is blue").

        Returns:
            Confirmation message string.
        """
        self.memory_db.add_texts(
            texts=[f"User provided an explicit fact to remember: {fact}"],
            metadatas=[{"role": "user", "type": "explicit_fact", "user_id": user_id}],
        )
        self._update_buffer(user_id, f"User explicitly stated a fact: {fact}")
        return f"I will remember that: '{fact}'"

    def list_facts(self, user_id: str) -> List[str]:
        """Retrieve all explicit facts saved by this user."""
        all_facts = self.memory_db.get(
            where={"$and": [{"type": "explicit_fact"}, {"user_id": user_id}]}
        )
        return [
            doc.replace("User provided an explicit fact to remember: ", "")
            for doc in all_facts["documents"]
        ]

    def delete_fact(self, user_id: str, keyword: str) -> int:
        """Delete all facts containing the keyword (case-insensitive). Returns count deleted."""
        all_facts = self.memory_db.get(
            where={"$and": [{"type": "explicit_fact"}, {"user_id": user_id}]}
        )
        ids_to_delete = [
            doc_id
            for doc_id, doc_text in zip(all_facts["ids"], all_facts["documents"])
            if keyword.lower() in doc_text.lower()
        ]
        if ids_to_delete:
            self.memory_db._collection.delete(ids=ids_to_delete)
        return len(ids_to_delete)

    def wipe_memory(self, user_id: str):
        """Irreversibly delete ALL memory entries (facts + chat history) for this user."""
        all_mems = self.memory_db.get(where={"user_id": user_id})
        if all_mems["ids"]:
            self.memory_db._collection.delete(ids=all_mems["ids"])
        self.recent_chat_buffers.pop(user_id, None)

    def clear_buffer(self, user_id: str):
        """Clear only the in-memory short-term chat buffer (does not affect persisted memory)."""
        self.recent_chat_buffers.pop(user_id, None)

    # ==========================================
    # DOCUMENT COMMANDS (Library CRUD)
    # ==========================================

    def ingest_document(self, file_path: str, original_filename: str) -> dict:
        """Load a document, split it into chunks, embed, and store in ChromaDB.

        Args:
            file_path: Absolute path to the file on disk.
            original_filename: The user-facing filename (stored in metadata).

        Returns:
            Dict with 'success', 'chunks', 'filename' on success,
            or 'success': False and 'error' message on failure.
        """
        if not os.path.exists(file_path):
            return {"success": False, "error": f"File not found: {file_path}"}

        ext = os.path.splitext(original_filename)[1].lower()
        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
        elif ext == ".txt":
            loader = TextLoader(file_path, autodetect_encoding=True)
        elif ext in [".docx", ".doc"]:
            loader = Docx2txtLoader(file_path)
        else:
            return {"success": False, "error": f"Unsupported file type: {ext}"}

        raw_docs = loader.load()
        if not raw_docs:
            return {"success": False, "error": "File is empty."}

        chunked_docs = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        ).split_documents(raw_docs)

        for doc in chunked_docs:
            doc.metadata["document_name"] = original_filename

        self.vector_db.add_documents(chunked_docs)
        return {"success": True, "chunks": len(chunked_docs), "filename": original_filename}

    def ingest_document_stream(self, file_path: str, original_filename: str) -> Generator[dict, None, None]:
        """Load, chunk, and embed a document in batches, yielding progress events.

        Yields event dicts:
          {"type": "progress", "message": "...", "percent": 0-100}
          {"type": "done", "success": True, "chunks": N, "filename": "..."}
          {"type": "error", "error": "..."}
        """
        if not os.path.exists(file_path):
            yield {"type": "error", "error": f"File not found: {file_path}"}
            return

        ext = os.path.splitext(original_filename)[1].lower()
        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
        elif ext == ".txt":
            loader = TextLoader(file_path, autodetect_encoding=True)
        elif ext in [".docx", ".doc"]:
            loader = Docx2txtLoader(file_path)
        else:
            yield {"type": "error", "error": f"Unsupported file type: {ext}"}
            return

        yield {"type": "progress", "message": "Parsing document...", "percent": 5}

        try:
            raw_docs = loader.load()
        except Exception as e:
            yield {"type": "error", "error": f"Failed to parse document: {e}"}
            return

        if not raw_docs:
            yield {"type": "error", "error": "File is empty."}
            return

        yield {"type": "progress", "message": f"Parsed {len(raw_docs)} pages. Chunking...", "percent": 15}

        chunked_docs = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        ).split_documents(raw_docs)

        for doc in chunked_docs:
            doc.metadata["document_name"] = original_filename

        total = len(chunked_docs)
        batch_size = 50
        yield {"type": "progress", "message": f"Embedding {total} chunks...", "percent": 20}

        for i in range(0, total, batch_size):
            batch = chunked_docs[i : i + batch_size]
            self.vector_db.add_documents(batch)
            done_count = min(i + batch_size, total)
            pct = 20 + int(80 * done_count / total)
            yield {
                "type": "progress",
                "message": f"Embedded {done_count}/{total} chunks...",
                "percent": pct,
            }

        yield {"type": "done", "success": True, "chunks": total, "filename": original_filename}

    def remove_document(self, filename: str) -> bool:
        """Remove all chunks belonging to a document. Returns True if found and deleted."""
        existing = self.vector_db.get(where={"document_name": filename})
        if not existing["ids"]:
            return False
        self.vector_db._collection.delete(where={"document_name": filename})
        return True

    def list_documents(self) -> List[str]:
        """Return a sorted list of unique document filenames in the library."""
        unique_docs = set()
        offset, batch_size = 0, 1000
        while True:
            batch = self.vector_db.get(limit=batch_size, offset=offset)
            if not batch["ids"]:
                break
            for meta in batch["metadatas"]:
                if meta and "document_name" in meta:
                    unique_docs.add(meta["document_name"])
                elif meta and "source" in meta:
                    unique_docs.add(os.path.basename(meta["source"]))
            offset += batch_size
        return sorted(unique_docs)

    # ==========================================
    # ACADEMIC RESEARCH PIPELINE
    # ==========================================

    def _build_research_queries(self, user_input: str, recent_history: str) -> dict:
        """Extract clean academic search queries and optional year filter from user input."""
        # Extract year constraint from the raw user request
        year_match = re.search(r'\b(20[0-9]{2})\b', user_input)
        year = int(year_match.group(1)) if year_match else None

        # Use LLM to produce clean search queries
        try:
            raw = self.research_query_chain.invoke({
                "recent_history": recent_history,
                "question": user_input,
            }).strip()
            queries = []
            for line in raw.split("\n"):
                q = line.strip().strip("\"'").lstrip("-•*").lstrip("0123456789.)").strip()
                if 3 < len(q) < 120:
                    queries.append(q)
            queries = queries[:3]
        except Exception as e:
            logger.error("Research query extraction failed: %s", e)
            queries = []

        # Fallback: regex-based cleanup of the original query
        if not queries:
            clean = re.sub(
                r"^(find|get|search|look\s?up|give\s+me|show\s+me|pull|I\s+need)\s+"
                r"(\d+\s+)?(full\s+)?(sources?|papers?|articles?|research|docs?)\s+"
                r"(on|about|for|regarding|related\s+to)\s+",
                "", user_input, flags=re.IGNORECASE,
            ).strip()
            clean = re.sub(r"\(?\b20[0-9]{2}\b\)?", "", clean).strip()
            queries = [clean] if clean else [user_input]

        return {"queries": queries, "year": year}

    def _search_semantic_scholar(
        self, query: str, limit: int = 10, year_range: Optional[str] = None
    ) -> List[dict]:
        """Search Semantic Scholar with optional year filtering and proper error logging."""
        papers = []
        try:
            params: dict = {
                "query": query,
                "limit": limit,
                "fields": "title,abstract,authors,year,url,openAccessPdf,externalIds,citationCount,venue",
            }
            if year_range:
                params["year"] = year_range

            resp = httpx.get(
                "https://api.semanticscholar.org/graph/v1/paper/search",
                params=params,
                timeout=15.0,
            )
            logger.info("Semantic Scholar [%s] year=%s: %d (%d bytes)",
                        query, year_range, resp.status_code, len(resp.content))

            if resp.status_code == 200:
                for p in resp.json().get("data", []):
                    paper = {
                        "title": p.get("title", "Untitled"),
                        "authors": [a.get("name", "") for a in (p.get("authors") or [])[:5]],
                        "year": p.get("year"),
                        "abstract": p.get("abstract", "") or "",
                        "url": p.get("url", ""),
                        "venue": p.get("venue", "") or "",
                        "citationCount": p.get("citationCount", 0) or 0,
                        "downloadUrl": None,
                    }
                    oa = p.get("openAccessPdf")
                    if oa and oa.get("url"):
                        paper["downloadUrl"] = oa["url"]
                    elif p.get("externalIds", {}).get("ArXiv"):
                        arxiv_id = p["externalIds"]["ArXiv"]
                        paper["downloadUrl"] = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
                        if not paper["url"]:
                            paper["url"] = f"https://arxiv.org/abs/{arxiv_id}"
                    papers.append(paper)
            else:
                logger.warning("Semantic Scholar %d: %s", resp.status_code, resp.text[:300])
        except Exception as e:
            logger.error("Semantic Scholar search failed: %s", e)
        return papers

    def _search_arxiv(self, query: str, limit: int = 8) -> List[dict]:
        """Search arXiv with improved query construction and error logging."""
        import xml.etree.ElementTree as ET

        papers = []
        try:
            # Build AND query from significant words for better matching
            words = [w for w in query.split() if len(w) > 2]
            if words:
                search_query = "+AND+".join(f"all:{w}" for w in words)
            else:
                search_query = f"all:{query}"

            resp = httpx.get(
                "http://export.arxiv.org/api/query",
                params={
                    "search_query": search_query,
                    "start": 0,
                    "max_results": limit,
                    "sortBy": "relevance",
                    "sortOrder": "descending",
                },
                timeout=15.0,
            )
            logger.info("arXiv [%s]: %d (%d bytes)", query, resp.status_code, len(resp.content))

            if resp.status_code == 200:
                ns = {"atom": "http://www.w3.org/2005/Atom"}
                root = ET.fromstring(resp.text)
                for entry in root.findall("atom:entry", ns):
                    title = entry.findtext("atom:title", "", ns).strip().replace("\n", " ")
                    if not title or title == "Error":
                        continue
                    abstract = entry.findtext("atom:summary", "", ns).strip().replace("\n", " ")
                    authors = [
                        a.findtext("atom:name", "", ns)
                        for a in entry.findall("atom:author", ns)
                    ][:5]
                    published = entry.findtext("atom:published", "", ns)[:4]
                    link = ""
                    pdf_link = ""
                    for lnk in entry.findall("atom:link", ns):
                        if lnk.get("type") == "text/html":
                            link = lnk.get("href", "")
                        if lnk.get("title") == "pdf":
                            pdf_link = lnk.get("href", "")
                    if not link:
                        link_el = entry.find("atom:id", ns)
                        link = link_el.text if link_el is not None else ""

                    papers.append({
                        "title": title,
                        "authors": authors,
                        "year": int(published) if published.isdigit() else None,
                        "abstract": abstract[:500],
                        "url": link,
                        "venue": "arXiv",
                        "citationCount": 0,
                        "downloadUrl": pdf_link or (
                            link.replace("/abs/", "/pdf/") + ".pdf" if "/abs/" in link else None
                        ),
                    })
            else:
                logger.warning("arXiv %d: %s", resp.status_code, resp.text[:300])
        except Exception as e:
            logger.error("arXiv search failed: %s", e)
        return papers

    def _search_openalex(
        self, query: str, limit: int = 10, year: Optional[int] = None
    ) -> List[dict]:
        """Search OpenAlex for open-access academic papers."""
        papers = []
        try:
            params: dict = {
                "search": query,
                "per_page": limit,
                "sort": "relevance_score:desc",
                "select": "title,authorships,publication_year,primary_location,"
                          "cited_by_count,doi,open_access,id",
            }
            if year:
                params["filter"] = (
                    f"from_publication_date:{year - 1}-01-01,"
                    f"to_publication_date:{year}-12-31"
                )

            resp = httpx.get(
                "https://api.openalex.org/works",
                params=params,
                headers={"User-Agent": "LibraryAI/1.0 (mailto:research@libraryai.dev)"},
                timeout=15.0,
            )
            logger.info("OpenAlex [%s] year=%s: %d", query, year, resp.status_code)

            if resp.status_code == 200:
                for w in resp.json().get("results", []):
                    authors = [
                        a.get("author", {}).get("display_name", "")
                        for a in (w.get("authorships") or [])[:5]
                        if a.get("author", {}).get("display_name")
                    ]
                    pl = w.get("primary_location") or {}
                    venue = (pl.get("source") or {}).get("display_name", "")
                    doi = w.get("doi", "") or ""
                    url = doi if doi else (w.get("id", "") or "")
                    oa = w.get("open_access") or {}

                    papers.append({
                        "title": w.get("title", "Untitled") or "Untitled",
                        "authors": authors,
                        "year": w.get("publication_year"),
                        "abstract": "",  # OpenAlex search doesn't return abstracts
                        "url": url,
                        "venue": venue,
                        "citationCount": w.get("cited_by_count", 0) or 0,
                        "downloadUrl": oa.get("oa_url"),
                    })
            else:
                logger.warning("OpenAlex %d: %s", resp.status_code, resp.text[:300])
        except Exception as e:
            logger.error("OpenAlex search failed: %s", e)
        return papers

    # ── PubMed (NCBI E-utilities) ────────────────────────────────────
    def _search_pubmed(self, query: str, limit: int = 8) -> List[dict]:
        """Search PubMed for biomedical literature."""
        import xml.etree.ElementTree as ET

        papers = []
        try:
            # Step 1: search for IDs
            search_resp = httpx.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                params={"db": "pubmed", "term": query, "retmax": limit, "sort": "relevance"},
                timeout=15.0,
            )
            if search_resp.status_code != 200:
                logger.warning("PubMed search %d", search_resp.status_code)
                return papers

            root = ET.fromstring(search_resp.text)
            ids = [id_el.text for id_el in root.findall(".//Id") if id_el.text]
            if not ids:
                return papers

            # Step 2: fetch summaries
            fetch_resp = httpx.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
                params={"db": "pubmed", "id": ",".join(ids), "retmode": "json"},
                timeout=15.0,
            )
            logger.info("PubMed [%s]: %d ids, fetch %d", query, len(ids), fetch_resp.status_code)

            if fetch_resp.status_code == 200:
                result = fetch_resp.json().get("result", {})
                for pmid in ids:
                    doc = result.get(pmid)
                    if not doc or not isinstance(doc, dict):
                        continue
                    authors = [
                        a.get("name", "")
                        for a in (doc.get("authors") or [])[:5]
                    ]
                    pub_date = doc.get("pubdate", "")
                    year = None
                    if pub_date and pub_date[:4].isdigit():
                        year = int(pub_date[:4])
                    papers.append({
                        "title": doc.get("title", "Untitled"),
                        "authors": authors,
                        "year": year,
                        "abstract": "",
                        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                        "venue": doc.get("fulljournalname", "") or doc.get("source", ""),
                        "citationCount": 0,
                        "downloadUrl": None,
                    })
        except Exception as e:
            logger.error("PubMed search failed: %s", e)
        return papers

    # ── CORE (core.ac.uk) ────────────────────────────────────────────
    def _search_core(self, query: str, limit: int = 8) -> List[dict]:
        """Search CORE for open-access research outputs."""
        papers = []
        try:
            resp = httpx.get(
                "https://api.core.ac.uk/v3/search/works",
                params={"q": query, "limit": limit},
                headers={"User-Agent": "LibraryAI/1.0"},
                timeout=15.0,
            )
            logger.info("CORE [%s]: %d", query, resp.status_code)

            if resp.status_code == 200:
                for item in resp.json().get("results", []):
                    authors = [
                        a.get("name", "") for a in (item.get("authors") or [])[:5]
                    ]
                    year = item.get("yearPublished")
                    dl = None
                    for link in (item.get("links") or []):
                        if link.get("type") == "download":
                            dl = link.get("url")
                            break
                    if not dl:
                        dl = item.get("downloadUrl")
                    papers.append({
                        "title": item.get("title", "Untitled") or "Untitled",
                        "authors": authors,
                        "year": year,
                        "abstract": (item.get("abstract") or "")[:500],
                        "url": item.get("sourceFulltextUrls", [None])[0]
                              or item.get("identifiers", [None])[0]
                              or "",
                        "venue": item.get("publisher", "") or "",
                        "citationCount": item.get("citationCount", 0) or 0,
                        "downloadUrl": dl,
                    })
            else:
                logger.warning("CORE %d: %s", resp.status_code, resp.text[:300])
        except Exception as e:
            logger.error("CORE search failed: %s", e)
        return papers

    # ── CrossRef ─────────────────────────────────────────────────────
    def _search_crossref(self, query: str, limit: int = 8) -> List[dict]:
        """Search CrossRef for DOI metadata."""
        papers = []
        try:
            resp = httpx.get(
                "https://api.crossref.org/works",
                params={
                    "query": query,
                    "rows": limit,
                    "sort": "relevance",
                    "select": "title,author,published-print,published-online,"
                              "container-title,is-referenced-by-count,DOI,link",
                },
                headers={
                    "User-Agent": "LibraryAI/1.0 (mailto:research@libraryai.dev)",
                },
                timeout=15.0,
            )
            logger.info("CrossRef [%s]: %d", query, resp.status_code)

            if resp.status_code == 200:
                for item in resp.json().get("message", {}).get("items", []):
                    title_list = item.get("title", [])
                    title = title_list[0] if title_list else "Untitled"
                    authors = [
                        f"{a.get('given', '')} {a.get('family', '')}".strip()
                        for a in (item.get("author") or [])[:5]
                    ]
                    date_parts = (
                        item.get("published-print", {}).get("date-parts", [[]])
                        or item.get("published-online", {}).get("date-parts", [[]])
                    )
                    year = date_parts[0][0] if date_parts and date_parts[0] else None
                    venue_list = item.get("container-title", [])
                    venue = venue_list[0] if venue_list else ""
                    doi = item.get("DOI", "")
                    url = f"https://doi.org/{doi}" if doi else ""

                    papers.append({
                        "title": title,
                        "authors": authors,
                        "year": year,
                        "abstract": "",
                        "url": url,
                        "venue": venue,
                        "citationCount": item.get("is-referenced-by-count", 0) or 0,
                        "downloadUrl": None,
                    })
            else:
                logger.warning("CrossRef %d: %s", resp.status_code, resp.text[:300])
        except Exception as e:
            logger.error("CrossRef search failed: %s", e)
        return papers

    # ── Europe PMC ───────────────────────────────────────────────────
    def _search_europe_pmc(self, query: str, limit: int = 8) -> List[dict]:
        """Search Europe PMC for biomedical and life sciences literature."""
        papers = []
        try:
            resp = httpx.get(
                "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
                params={
                    "query": query,
                    "format": "json",
                    "pageSize": limit,
                    "resultType": "core",
                },
                timeout=15.0,
            )
            logger.info("Europe PMC [%s]: %d", query, resp.status_code)

            if resp.status_code == 200:
                for item in resp.json().get("resultList", {}).get("result", []):
                    authors_str = item.get("authorString", "")
                    authors = [a.strip() for a in authors_str.split(",")[:5]] if authors_str else []
                    year_str = item.get("pubYear")
                    year = int(year_str) if year_str and str(year_str).isdigit() else None
                    pmcid = item.get("pmcid", "")
                    pmid = item.get("pmid", "")
                    doi = item.get("doi", "")
                    url = (
                        f"https://europepmc.org/article/PMC/{pmcid}" if pmcid
                        else f"https://doi.org/{doi}" if doi
                        else f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid
                        else ""
                    )
                    dl_url = (
                        f"https://europepmc.org/backend/ptpmcrender.fcgi?accid={pmcid}&blobtype=pdf"
                        if pmcid else None
                    )
                    papers.append({
                        "title": item.get("title", "Untitled") or "Untitled",
                        "authors": authors,
                        "year": year,
                        "abstract": (item.get("abstractText") or "")[:500],
                        "url": url,
                        "venue": item.get("journalTitle", "") or "",
                        "citationCount": item.get("citedByCount", 0) or 0,
                        "downloadUrl": dl_url,
                    })
            else:
                logger.warning("Europe PMC %d: %s", resp.status_code, resp.text[:300])
        except Exception as e:
            logger.error("Europe PMC search failed: %s", e)
        return papers

    # ── Papers With Code ─────────────────────────────────────────────
    def _search_papers_with_code(self, query: str, limit: int = 8) -> List[dict]:
        """Search Papers With Code for ML/AI papers with implementations."""
        papers = []
        try:
            resp = httpx.get(
                "https://paperswithcode.com/api/v1/search/",
                params={"q": query, "page": 1, "items_per_page": limit},
                timeout=15.0,
            )
            logger.info("Papers With Code [%s]: %d", query, resp.status_code)

            if resp.status_code == 200:
                data = resp.json()
                results = data.get("results", data) if isinstance(data, dict) else data
                if isinstance(results, list):
                    for item in results[:limit]:
                        paper = item.get("paper", item) if isinstance(item, dict) else {}
                        if not isinstance(paper, dict):
                            continue
                        authors_raw = paper.get("authors", [])
                        if isinstance(authors_raw, list):
                            authors = [str(a) for a in authors_raw[:5]]
                        else:
                            authors = []
                        url_slug = paper.get("url_abs") or paper.get("url", "")
                        arxiv_id = paper.get("arxiv_id", "")
                        dl = f"https://arxiv.org/pdf/{arxiv_id}.pdf" if arxiv_id else None
                        if not url_slug and arxiv_id:
                            url_slug = f"https://arxiv.org/abs/{arxiv_id}"
                        papers.append({
                            "title": paper.get("title", "Untitled") or "Untitled",
                            "authors": authors,
                            "year": None,
                            "abstract": (paper.get("abstract") or "")[:500],
                            "url": url_slug,
                            "venue": "Papers With Code",
                            "citationCount": 0,
                            "downloadUrl": dl,
                        })
        except Exception as e:
            logger.error("Papers With Code search failed: %s", e)
        return papers

    # ── DuckDuckGo (web results as source) ───────────────────────────
    def _search_duckduckgo(self, query: str, limit: int = 8) -> List[dict]:
        """Use DuckDuckGo web search and return results as paper-like entries."""
        papers = []
        try:
            raw = self.web_search.invoke(query + " research paper")
            # DuckDuckGo returns a string of concatenated snippets
            # Parse what we can
            if raw:
                papers.append({
                    "title": f"Web results: {query}",
                    "authors": [],
                    "year": None,
                    "abstract": raw[:500],
                    "url": "",
                    "venue": "DuckDuckGo Web Search",
                    "citationCount": 0,
                    "downloadUrl": None,
                })
        except Exception as e:
            logger.error("DuckDuckGo search failed: %s", e)
        return papers

    # ── Wikipedia ────────────────────────────────────────────────────
    def _search_wikipedia(self, query: str, limit: int = 5) -> List[dict]:
        """Search Wikipedia for background/context articles."""
        papers = []
        try:
            resp = httpx.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": query,
                    "srlimit": limit,
                    "format": "json",
                    "utf8": 1,
                },
                timeout=10.0,
            )
            logger.info("Wikipedia [%s]: %d", query, resp.status_code)

            if resp.status_code == 200:
                for item in resp.json().get("query", {}).get("search", []):
                    title = item.get("title", "")
                    snippet = re.sub(r"<[^>]+>", "", item.get("snippet", ""))
                    page_id = item.get("pageid", "")
                    papers.append({
                        "title": title,
                        "authors": ["Wikipedia"],
                        "year": None,
                        "abstract": snippet[:500],
                        "url": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                        "venue": "Wikipedia",
                        "citationCount": 0,
                        "downloadUrl": None,
                    })
        except Exception as e:
            logger.error("Wikipedia search failed: %s", e)
        return papers

    # ── Source registry (maps source id → search method) ────────────
    _SOURCE_METHODS: Dict[str, str] = {
        "semantic_scholar": "_search_semantic_scholar",
        "arxiv": "_search_arxiv",
        "openalex": "_search_openalex",
        "pubmed": "_search_pubmed",
        "core": "_search_core",
        "crossref": "_search_crossref",
        "europe_pmc": "_search_europe_pmc",
        "papers_with_code": "_search_papers_with_code",
        "duckduckgo": "_search_duckduckgo",
        "wikipedia": "_search_wikipedia",
    }

    _DEFAULT_SOURCES = ["semantic_scholar", "arxiv", "openalex"]

    def _search_all_apis(
        self, queries: List[str], year: Optional[int] = None,
        sources: Optional[List[str]] = None,
    ) -> List[dict]:
        """Fire concurrent searches across selected APIs."""
        active = sources if sources else self._DEFAULT_SOURCES
        all_papers: List[dict] = []
        year_range = f"{year - 1}-{year}" if year else None

        with ThreadPoolExecutor(max_workers=12) as pool:
            futures = []
            for i, q in enumerate(queries[:3]):
                for src in active:
                    method_name = self._SOURCE_METHODS.get(src)
                    if not method_name:
                        continue
                    method = getattr(self, method_name, None)
                    if not method:
                        continue

                    # Source-specific argument handling
                    if src == "semantic_scholar":
                        if i == 0 and year_range:
                            futures.append(pool.submit(method, q, 10, year_range))
                            futures.append(pool.submit(method, q, 8, None))
                        else:
                            futures.append(pool.submit(method, q, 8, year_range))
                    elif src == "openalex":
                        futures.append(pool.submit(method, q, 8, year))
                    else:
                        futures.append(pool.submit(method, q, 6))

            for f in as_completed(futures):
                try:
                    all_papers.extend(f.result(timeout=20))
                except Exception as e:
                    logger.error("Search future failed: %s", e)

        return all_papers

    def _deduplicate_papers(self, papers: List[dict]) -> List[dict]:
        """Deduplicate by normalised title and rank by quality signals."""
        seen: set = set()
        unique: List[dict] = []
        for p in papers:
            key = re.sub(r"[^a-z0-9\s]", "", p["title"].lower()).strip()[:80]
            if key and key not in seen:
                seen.add(key)
                unique.append(p)

        def _score(p: dict) -> float:
            s = 0.0
            if p.get("abstract"):
                s += 1000
            if p.get("downloadUrl"):
                s += 500
            s += min(p.get("citationCount", 0), 500)
            if p.get("year"):
                s += (p["year"] - 2000) * 2
            return s

        unique.sort(key=_score, reverse=True)
        return unique

    def execute_research_stream(
        self, user_id: str, query: str,
        sources: Optional[List[str]] = None,
    ) -> Generator[dict, None, None]:
        """Enhanced research pipeline: multi-API concurrent search → LLM synthesis.

        Searches Semantic Scholar, arXiv, and OpenAlex concurrently with multiple
        query variations, deduplicates and ranks results, then streams an LLM
        synthesis referencing the papers found.

        Yields event dicts compatible with the SSE streaming protocol.
        """
        buffer = self._get_buffer(user_id)
        recent_history_str = "\n".join(buffer) if buffer else "No recent conversation."

        # 1. Extract clean search queries + year filter
        yield {"type": "status", "message": "Analyzing research request..."}
        params = self._build_research_queries(query, recent_history_str)
        queries = params["queries"]
        year = params["year"]
        yield {"type": "debug", "standalone_query": " | ".join(queries)}

        # 2. Concurrent search across selected sources
        active = sources if sources else self._DEFAULT_SOURCES
        source_names = [sid.replace("_", " ").title() for sid in active]
        yield {"type": "status", "message": f"Searching {', '.join(source_names[:4])}{'…' if len(source_names) > 4 else ''}"}
        all_papers = self._search_all_apis(queries, year, sources=sources)
        all_papers = self._deduplicate_papers(all_papers)

        # 3. If too few results, broaden search without year filter
        if len(all_papers) < 3:
            yield {"type": "status", "message": "Broadening search..."}
            broadest = queries[-1] if len(queries) > 1 else queries[0]
            broader = self._search_all_apis([broadest], year=None, sources=sources)
            all_papers = self._deduplicate_papers(all_papers + broader)

        # 4. Fallback to web search if still nothing
        if not all_papers:
            yield {"type": "status", "message": "No academic papers found. Searching the web..."}
            try:
                web_results = self.web_search.invoke(queries[0] + " research paper 2024 2025")
                yield {"type": "status", "message": "Synthesizing web findings..."}
                for chunk in self.research_chain.stream({
                    "question": query,
                    "recent_history": recent_history_str,
                    "papers_context": (
                        "No academic database papers were found. "
                        "Synthesize from these web search results instead:\n\n"
                        + web_results
                    ),
                }):
                    yield {"type": "token", "content": chunk}
                yield {"type": "sources", "sources": ["Web Search (DuckDuckGo)"]}
            except Exception as e:
                logger.error("Web fallback failed: %s", e)
                yield {
                    "type": "token",
                    "content": (
                        "I couldn't find academic papers or web results for this query. Try:\n\n"
                        "- Using more specific technical terms\n"
                        "- Removing year constraints\n"
                        "- Broadening the topic"
                    ),
                }
            yield {"type": "done"}
            return

        yield {
            "type": "status",
            "message": f"Found {len(all_papers)} papers. Synthesizing research overview...",
        }

        # 5. Emit paper metadata to frontend
        yield {"type": "papers", "papers": all_papers}

        # 6. Build rich context for LLM synthesis
        papers_context = ""
        for i, p in enumerate(all_papers[:12], 1):
            authors_str = ", ".join(p["authors"][:3])
            if len(p["authors"]) > 3:
                authors_str += " et al."
            papers_context += f"\n---\nPaper {i}: \"{p['title']}\"\n"
            papers_context += f"Authors: {authors_str}\n"
            if p["year"]:
                papers_context += f"Year: {p['year']}\n"
            if p["venue"]:
                papers_context += f"Venue: {p['venue']}\n"
            if p.get("citationCount"):
                papers_context += f"Citations: {p['citationCount']}\n"
            if p["abstract"]:
                papers_context += f"Abstract: {p['abstract'][:500]}\n"

        # 7. Stream LLM synthesis
        ai_answer = ""
        for chunk in self.research_chain.stream(
            {"question": query, "recent_history": recent_history_str, "papers_context": papers_context}
        ):
            ai_answer += chunk
            yield {"type": "token", "content": chunk}

        # 8. Sources
        sources = []
        for p in all_papers:
            src = p["title"]
            if p["year"]:
                src += f" ({p['year']})"
            sources.append(src)
        yield {"type": "sources", "sources": sources}

        # 9. Update buffer
        self._update_buffer(user_id, f"User: {query}")
        self._update_buffer(user_id, f"AI: {ai_answer[:200]}...")

        yield {"type": "done"}

    # ==========================================
    # RAG PIPELINE
    # ==========================================

    def parse_mode(self, mode: str) -> Tuple[bool, bool, bool]:
        """Returns (use_library, use_memory, force_web) based on mode string."""
        modes = {
            "strict": (True, False, False),
            "chat": (False, True, False),
            "web": (False, False, True),
            "default": (True, True, False),
        }
        return modes.get(mode, modes["default"])

    def execute_query_stream(
        self, user_id: str, query: str, use_library: bool, use_memory: bool, force_web: bool
    ) -> Generator[dict, None, None]:
        """
        Streaming RAG pipeline. Yields dicts:
          {"type": "status", "message": "..."} - status updates
          {"type": "debug", "standalone_query": "..."} - rephrased query
          {"type": "token", "content": "..."} - streamed answer tokens
          {"type": "sources", "sources": [...]} - sources used
          {"type": "done"} - completion signal
        """
        buffer = self._get_buffer(user_id)
        recent_history_str = "\n".join(buffer) if buffer else "No recent conversation."

        # 1. Rephrase
        standalone_query = self.rephrase_chain.invoke(
            {"recent_history": recent_history_str, "question": query}
        ).strip()
        yield {"type": "debug", "standalone_query": standalone_query}

        # 2. Retrieve memory
        chat_history_str = "Memory search disabled."
        if use_memory:
            memory_docs = self.memory_db.similarity_search(standalone_query, k=2)
            if memory_docs:
                chat_history_str = "\n\n".join([d.page_content for d in memory_docs])

        # 3. Retrieve library / web
        context_str = "Library search disabled."
        library_docs = []
        if force_web:
            yield {"type": "status", "message": "Searching the web..."}
            try:
                context_str = f"LIVE WEB RESULTS:\n{self.web_search.invoke(standalone_query)}"
            except Exception as e:
                context_str = f"Web search failed: {e}"
        elif use_library:
            library_docs = self.vector_db.similarity_search(standalone_query, k=3)
            if library_docs:
                context_str = "\n\n".join([d.page_content for d in library_docs])

        # 4. Stream answer
        ai_answer = ""
        for chunk in self.qa_chain.stream(
            {"question": standalone_query, "context": context_str, "chat_history": chat_history_str}
        ):
            ai_answer += chunk
            yield {"type": "token", "content": chunk}

        # 5. Fallback to web if AI can't answer
        if "I cannot find the answer" in ai_answer and not force_web:
            yield {"type": "status", "message": "Falling back to web search..."}
            try:
                context_str = f"LIVE WEB RESULTS:\n{self.web_search.invoke(standalone_query)}"
                ai_answer = ""
                for chunk in self.qa_chain.stream(
                    {"question": standalone_query, "context": context_str, "chat_history": chat_history_str}
                ):
                    ai_answer += chunk
                    yield {"type": "token", "content": chunk}
                force_web = True
            except Exception as e:
                yield {"type": "status", "message": f"Web fallback failed: {e}"}

        # 6. Build sources
        sources = []
        if force_web:
            sources.append("Live Internet Search (DuckDuckGo)")
        elif library_docs and use_library:
            for doc in library_docs:
                sources.append(
                    f"{doc.metadata.get('source', doc.metadata.get('document_name', 'Unknown'))} "
                    f"(Page {doc.metadata.get('page', 'N/A')})"
                )

        yield {"type": "sources", "sources": sources}

        # 7. Save to memory
        if use_library and use_memory and not force_web:
            self.memory_db.add_texts(
                texts=[f"User asked: {query}", f"AI answered: {ai_answer}"],
                metadatas=[
                    {"role": "user", "user_id": user_id},
                    {"role": "assistant", "user_id": user_id},
                ],
            )

        self._update_buffer(user_id, f"User: {query}")
        self._update_buffer(user_id, f"AI: {ai_answer}")

        yield {"type": "done"}
