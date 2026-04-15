import os
from typing import Tuple, List, Dict
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.tools import DuckDuckGoSearchRun

class LibraryAI:
    def __init__(self, chroma_dir: str):
        """Initializes the models, databases, and chains."""
        print("Connecting to local databases...")
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-en-v1.5",
            model_kwargs={'device': 'cuda'}
        )
        
        self.vector_db = Chroma(persist_directory=chroma_dir, embedding_function=self.embedding_model)
        self.memory_db = Chroma(collection_name="chat_memory", persist_directory=chroma_dir, embedding_function=self.embedding_model)
        
        print("Initializing qwen3-coder:30b model via Ollama...")
        self.llm = OllamaLLM(model="qwen3-coder:30b")
        self.web_search = DuckDuckGoSearchRun()
        
        self.recent_chat_buffer: List[str] = []
        self._setup_chains()

    def _setup_chains(self):
        """Sets up the prompt templates and LangChain pipelines."""
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

    # ==========================================
    # COMMAND ROUTING (CRUD Operations)
    # ==========================================

    def process_system_command(self, user_input: str) -> bool:
        """
        Catches direct database commands. 
        Returns True if a command was executed (meaning skip the RAG pipeline).
        """
        command = user_input.lower()

        if command.startswith("-info "):
            self._save_explicit_fact(user_input[6:].strip())
            return True
        elif command == "-facts":
            self._list_facts()
            return True
        elif command.startswith("-forget "):
            self._delete_fact(user_input[8:].strip())
            return True
        elif command.startswith("-upload "):
            self._ingest_document(user_input[8:].strip())
            return True
        elif command.startswith("-remove "):
            self._remove_document(user_input[8:].strip())
            return True
        elif command == "-docs":
            self._list_documents()
            return True
        elif command == "-clear":
            self.recent_chat_buffer.clear()
            print("🧹 [Buffer Cleared]: Short-term RAM wiped.")
            return True
        elif command == "-wipe":
            self._wipe_all_memory()
            return True

        return False # Not a system command, proceed to chat

    # --- Helper Methods for Commands ---
    def _save_explicit_fact(self, fact: str):
        self.memory_db.add_texts(
            texts=[f"User provided an explicit fact to remember: {fact}"],
            metadatas=[{"role": "user", "type": "explicit_fact"}]
        )
        print(f"🧠 [Memory Saved]: I will remember that: '{fact}'")
        self._update_short_term_buffer(f"User explicitly stated a fact: {fact}")

    def _list_facts(self):
        all_facts = self.memory_db.get(where={"type": "explicit_fact"})
        print("\n🧠 [Explicit Facts Remembered]:")
        if not all_facts['documents']:
            print("  - No facts saved yet.")
        else:
            for doc in all_facts['documents']:
                print(f"  - {doc.replace('User provided an explicit fact to remember: ', '')}")

    def _delete_fact(self, keyword: str):
        all_facts = self.memory_db.get(where={"type": "explicit_fact"})
        ids_to_delete = [
            doc_id for doc_id, doc_text in zip(all_facts['ids'], all_facts['documents'])
            if keyword in doc_text.lower()
        ]
        
        if ids_to_delete:
            self.memory_db._collection.delete(ids=ids_to_delete)
            print(f"🗑️ [Memory Deleted]: Forgot {len(ids_to_delete)} fact(s) containing '{keyword}'.")
        else:
            print(f"⚠️ [Not Found]: No explicit facts found containing '{keyword}'.")

    def _ingest_document(self, file_path: str):
        if not os.path.exists(file_path):
            print(f"❌ [Error]: Could not find file at '{file_path}'")
            return
        
        print(f"📖 [Reading]: Loading '{file_path}'...")
        try:
            ext = os.path.splitext(file_path)[1].lower()
            if ext == '.pdf':
                loader = PyPDFLoader(file_path)
            elif ext == '.txt':
                loader = TextLoader(file_path, autodetect_encoding=True)
            elif ext in ['.docx', '.doc']:
                loader = Docx2txtLoader(file_path)
            else:
                print(f"❌ [Error]: Unsupported file type '{ext}'.")
                return

            raw_docs = loader.load()
            if not raw_docs:
                print(f"⚠️ [Warning]: File is empty.")
                return
            
            chunked_docs = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(raw_docs)
            filename = os.path.basename(file_path)
            for doc in chunked_docs:
                doc.metadata["document_name"] = filename
            
            print(f"✂️ [Chunking]: {len(chunked_docs)} pieces. 🧠 [Embedding]...")
            self.vector_db.add_documents(chunked_docs)
            print(f"✅ [Success]: '{filename}' memorized.")
            self._update_short_term_buffer(f"System: User uploaded {filename}")
        except Exception as e:
            print(f"❌ [Error]: Failed to process document. {e}")

    def _remove_document(self, target_file: str):
        existing_docs = self.vector_db.get(where={"document_name": target_file})
        if not existing_docs['ids']:
            print(f"⚠️ [Not Found]: '{target_file}' not in library.")
        else:
            self.vector_db._collection.delete(where={"document_name": target_file})
            print(f"🗑️ [Document Removed]: Deleted '{target_file}'.")

    def _list_documents(self):
        unique_docs = set()
        offset, batch_size = 0, 1000
        print("📚 [Scanning Library]: Fetching documents...")
        while True:
            batch = self.vector_db.get(limit=batch_size, offset=offset)
            if not batch['ids']:
                break
            for meta in batch['metadatas']:
                if meta and 'document_name' in meta:
                    unique_docs.add(meta['document_name'])
                elif meta and 'source' in meta:
                    unique_docs.add(os.path.basename(meta['source']))
            offset += batch_size
        print("\n📚 [Library Documents]:")
        for doc in sorted(unique_docs):
            print(f"  - {doc}")
        if not unique_docs:
            print("  - Empty.")

    def _wipe_all_memory(self):
        all_mems = self.memory_db.get()
        if all_mems['ids']:
            self.memory_db._collection.delete(ids=all_mems['ids'])
        self.recent_chat_buffer.clear()
        print("☢️ [Nuclear Wipe]: All chat memories deleted.")

    # ==========================================
    # RAG PIPELINE
    # ==========================================

    def parse_search_flags(self, user_input: str) -> Tuple[str, bool, bool, bool]:
        """Parses RAG routing commands (-strict, -chat, -web) and returns search parameters."""
        use_library, use_memory, force_web = True, True, False
        query_text = user_input
        command = user_input.lower()

        if command.startswith("-strict "):
            use_memory, query_text = False, user_input[8:].strip()
            print("📚 [Strict Mode]: Searching ONLY the library.")
        elif command.startswith("-chat "):
            use_library, query_text = False, user_input[6:].strip()
            print("💬 [Chat Mode]: Searching ONLY past conversations.")
        elif command.startswith("-web "):
            use_library, use_memory, force_web = False, False, True
            query_text = user_input[5:].strip()
            print("🌐 [Web Mode]: Searching the live internet.")

        return query_text, use_library, use_memory, force_web

    def execute_query(self, query_text: str, use_library: bool, use_memory: bool, force_web: bool):
        """Runs the core RAG pipeline: Rephrase -> Retrieve -> Generate -> Fallback."""
        print("Thinking...\n")
        
        # 1. Rephrase
        recent_history_str = "\n".join(self.recent_chat_buffer) if self.recent_chat_buffer else "No recent conversation."
        standalone_query = self.rephrase_chain.invoke({"recent_history": recent_history_str, "question": query_text}).strip()
        print(f"🔍 [Debug - Searching for]: {standalone_query}\n")
        
        # 2. Retrieve Memory
        chat_history_str = "Memory search disabled."
        if use_memory:
            memory_docs = self.memory_db.similarity_search(standalone_query, k=2)
            if memory_docs:
                chat_history_str = "\n\n".join([doc.page_content for doc in memory_docs])

        # 3. Retrieve Library/Web
        context_str, library_docs = "Library search disabled.", []
        if force_web:
            try:
                context_str = f"LIVE WEB RESULTS:\n{self.web_search.invoke(standalone_query)}"
            except Exception as e:
                context_str = f"Web search failed: {e}"
        elif use_library:
            library_docs = self.vector_db.similarity_search(standalone_query, k=3)
            if library_docs:
                context_str = "\n\n".join([doc.page_content for doc in library_docs])

        # 4. Generate (Streaming to console)
        print("🤖 ANSWER:\n", end="")
        ai_answer = ""
        for chunk in self.qa_chain.stream({"question": standalone_query, "context": context_str, "chat_history": chat_history_str}):
            print(chunk, end="", flush=True)
            ai_answer += chunk
        
        # 5. Interceptor Fallback
        if "I cannot find the answer" in ai_answer and not force_web:
            print("\n\n🌐 [Fallback Triggered]: Searching the live web...")
            try:
                context_str = f"LIVE WEB RESULTS:\n{self.web_search.invoke(standalone_query)}"
                print("🤖 WEB ANSWER:\n", end="")
                ai_answer = ""  # Reset answer for the fallback
                
                # Stream the fallback response
                for chunk in self.qa_chain.stream({"question": standalone_query, "context": context_str, "chat_history": chat_history_str}):
                    print(chunk, end="", flush=True)
                    ai_answer += chunk
                force_web = True
                
            except Exception as e:
                print(f"\n❌ [Web Error]: {e}")

        # 6. Output & Save
        self._print_results(ai_answer, force_web, library_docs, use_library)
        if use_library and use_memory and not force_web:
            self.memory_db.add_texts(texts=[f"User asked: {query_text}", f"AI answered: {ai_answer}"], metadatas=[{"role": "user"}, {"role": "assistant"}])
            
        self._update_short_term_buffer(f"User: {query_text}")
        self._update_short_term_buffer(f"AI: {ai_answer}")

    # --- Utility Methods ---
    def _print_results(self, answer: str, forced_web: bool, docs: list, used_lib: bool):
        print("\n\n📚 SOURCES USED:")
        if forced_web:
            print("- Live Internet Search (DuckDuckGo)")
        elif docs and used_lib:
            for doc in docs:
                print(f"- {doc.metadata.get('source', doc.metadata.get('document_name', 'Unknown'))} (Page {doc.metadata.get('page', 'Unknown')})")
        elif used_lib:
            print("- No documents retrieved.")
        else:
            print("- Library ignored.")
        print("-" * 70)

    def _update_short_term_buffer(self, entry: str):
        self.recent_chat_buffer.append(entry)
        if len(self.recent_chat_buffer) > 6:
            self.recent_chat_buffer = self.recent_chat_buffer[-6:]

    def print_welcome_menu(self):
        print("\n✅ AI is ready to chat!\n--- Memory Commands ---\n -info [fact] : Save a personal fact\n -facts : List all saved facts\n -forget [word] : Delete facts containing a keyword\n--- Library Commands ---\n -upload [path] : Ingest a .pdf, .txt, or .docx file\n -docs : List all files in the library\n -remove [name] : Delete a file from the library\n--- Chat Commands ---\n -strict [query] : Search ONLY the library\n -chat [query] : Search ONLY past conversations\n -web [query] : Search ONLY the live internet\n -clear : Wipe short-term RAM\n -wipe : NUCLEAR wipe of all chat memory\n" + "="*70)

def main():
    CHROMA_DIRECTORY = "./chroma_db"
    app = LibraryAI(CHROMA_DIRECTORY)
    app.print_welcome_menu()

    while True:
        user_input = input("\nAsk your library a question (or type 'exit'): ").strip()
        if user_input.lower() in ['exit', 'quit']:
            print("Shutting down...")
            break
        
        # 1. Check if it's a direct DB command (Skip RAG if true)
        if app.process_system_command(user_input):
            continue
        
        # 2. Parse search routing flags
        clean_query, use_lib, use_mem, force_web = app.parse_search_flags(user_input)
        
        # 3. Execute the RAG pipeline
        app.execute_query(clean_query, use_lib, use_mem, force_web)

if __name__ == "__main__":
    main()