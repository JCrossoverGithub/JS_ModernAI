"""
steps/evaluation/gen_eval.py — ZenML step: generation quality with ROUGE + BERTScore.

Phase 6 (Chapter 7 — Evaluation):
  Samples stored documents from MongoDB, generates reference answers with Ollama
  (using the full document snippet as context — the "gold" answer), generates a
  second answer *without context* (the "system" answer), and computes:

  - ROUGE-1 / ROUGE-2 / ROUGE-L   : n-gram overlap (fast, no GPU needed)
  - BERTScore F1                   : semantic similarity via BERT embeddings

  This lets you compare fine-tuned vs. base model output quality without needing
  a human-labelled test set.

Returns a dict with mean scores across all pairs.
"""

import json
import os
from loguru import logger
from zenml import step


@step
def evaluate_generation(
    model_id: str = "",
    sample_size: int = 0,
) -> dict:
    """Score generated answers against Ollama-produced reference answers.

    Args:
        model_id: HuggingFace model ID or local adapter path.  If empty, uses
                  Ollama (`settings.eval_llm`) for both reference and hypothesis
                  so you can run evaluation without a trained adapter.
        sample_size: number of QA pairs.  0 → settings.eval_sample_size.

    Returns:
        dict: rouge1, rouge2, rougeL (precision/recall/fmeasure)
              and bertscore_f1 (mean).
    """
    from rouge_score import rouge_scorer
    from langchain_ollama import OllamaLLM

    from config import get_settings
    from db.mongo import get_database

    settings = get_settings()
    n = sample_size or settings.eval_sample_size

    # ── 1. Sample documents ───────────────────────────────────────────────────
    db = get_database()
    cursor = db["raw_documents"].find({}, {"content": 1, "_id": 0}).limit(n)
    docs = [d["content"] for d in cursor if d.get("content")]

    if not docs:
        raise ValueError("No documents in MongoDB.")

    logger.info("Evaluating generation over %d documents...", len(docs))

    # ── 2. Load hypothesis generator ─────────────────────────────────────────
    use_hf = bool(model_id)
    if use_hf:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
        from peft import PeftModel

        is_adapter = os.path.isdir(model_id)
        hf_base = settings.base_model_id if is_adapter else model_id
        hf_token = settings.huggingface_access_token or None

        logger.info("Loading HF model %s for hypothesis generation...", hf_base)
        tokenizer = AutoTokenizer.from_pretrained(hf_base, token=hf_token, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            hf_base, quantization_config=bnb, device_map="auto", token=hf_token,
            trust_remote_code=True,
        )
        if is_adapter:
            model = PeftModel.from_pretrained(model, model_id)
            model = model.merge_and_unload()

        gen_pipe = pipeline(
            "text-generation", model=model, tokenizer=tokenizer,
            max_new_tokens=256, do_sample=False,
        )

        def generate_hyp(prompt: str) -> str:
            out = gen_pipe(prompt, return_full_text=False)
            return out[0]["generated_text"].strip()

    else:
        # No adapter — use Ollama (useful to baseline the stock model)
        llm = OllamaLLM(model=settings.eval_llm, base_url=settings.ollama_base_url)

        def generate_hyp(prompt: str) -> str:
            return llm.invoke(prompt).strip()

    ref_llm = OllamaLLM(model=settings.eval_llm, base_url=settings.ollama_base_url)

    # ── 3. Generate reference + hypothesis pairs ──────────────────────────────
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    rouge_totals = {"rouge1": {"p": 0, "r": 0, "f": 0},
                    "rouge2": {"p": 0, "r": 0, "f": 0},
                    "rougeL": {"p": 0, "r": 0, "f": 0}}
    references, hypotheses = [], []

    for i, content in enumerate(docs):
        snippet = content[:600].strip()
        if len(snippet) < 50:
            continue

        # Question
        q_prompt = (
            "Generate one clear factual question whose answer is in this text. "
            "Return ONLY the question.\n\nTEXT:\n" + snippet
        )
        try:
            question = ref_llm.invoke(q_prompt).strip()
        except Exception as exc:
            logger.warning("Q gen failed [%d]: %s", i, exc)
            continue

        # Reference: Ollama with full context (gold answer)
        ref_prompt = (
            f"Context:\n{snippet}\n\nQuestion: {question}\n\n"
            "Answer using the context. Be concise and accurate."
        )
        try:
            reference = ref_llm.invoke(ref_prompt).strip()
        except Exception as exc:
            logger.warning("Reference gen failed [%d]: %s", i, exc)
            continue

        # Hypothesis: fine-tuned model / Ollama without context
        hyp_prompt = f"Question: {question}\n\nAnswer:"
        try:
            hypothesis = generate_hyp(hyp_prompt)
        except Exception as exc:
            logger.warning("Hypothesis gen failed [%d]: %s", i, exc)
            hypothesis = ""

        # ROUGE
        r = scorer.score(reference, hypothesis)
        for key in rouge_totals:
            rouge_totals[key]["p"] += r[key].precision
            rouge_totals[key]["r"] += r[key].recall
            rouge_totals[key]["f"] += r[key].fmeasure

        references.append(reference)
        hypotheses.append(hypothesis)

        if (i + 1) % 20 == 0:
            logger.info("  %d / %d pairs scored", i + 1, len(docs))

    if not references:
        raise ValueError("No valid pairs could be generated for evaluation.")

    k = len(references)
    scores: dict = {}
    for key in rouge_totals:
        scores[f"{key}_precision"] = rouge_totals[key]["p"] / k
        scores[f"{key}_recall"] = rouge_totals[key]["r"] / k
        scores[f"{key}_fmeasure"] = rouge_totals[key]["f"] / k

    # ── 4. BERTScore (semantic similarity) ───────────────────────────────────
    logger.info("Computing BERTScore for %d pairs...", k)
    try:
        from bert_score import score as bert_score

        P, R, F = bert_score(
            hypotheses,
            references,
            lang="en",
            model_type="microsoft/deberta-xlarge-mnli",
            device=settings.embed_device,
            batch_size=8,
            verbose=False,
        )
        scores["bertscore_precision"] = float(P.mean())
        scores["bertscore_recall"] = float(R.mean())
        scores["bertscore_f1"] = float(F.mean())
    except Exception as exc:
        logger.warning("BERTScore failed (is bert-score installed?): %s", exc)
        scores["bertscore_f1"] = -1.0

    # ── 5. Persist results ────────────────────────────────────────────────────
    os.makedirs(settings.eval_output_dir, exist_ok=True)
    out_path = os.path.join(settings.eval_output_dir, "gen_eval.json")
    with open(out_path, "w") as f:
        json.dump({"n_pairs": k, "model_id": model_id or settings.eval_llm, "scores": scores}, f, indent=2)

    logger.info("Generation eval results → %s", out_path)
    for metric, val in scores.items():
        logger.info("  %-30s %.4f", metric, val)

    return scores
