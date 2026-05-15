"""
steps/training/dpo.py — ZenML steps: preference dataset + DPO trainer.

Phase 5 (Chapter 6 — Preference Alignment).
  - build_preference_dataset: sample prompts from MongoDB, generate chosen
    (helpful, detailed) and rejected (vague, unhelpful) responses via Ollama,
    return a HuggingFace Dataset in DPO triplet format.
  - run_dpo: load the SFT LoRA adapter in 4-bit, train with TRL DPOTrainer
    (ref_model=None → implicit PEFT reference, memory-efficient on 8 GB VRAM).
"""

import os
from loguru import logger
from zenml import step

# ── Prompt templates for chosen / rejected generation ─────────────────────────
_CHOSEN_SYSTEM = (
    "You are a knowledgeable, thorough research assistant. "
    "Provide a detailed, well-structured answer with relevant context and examples."
)
_REJECTED_SYSTEM = (
    "Answer in at most two short sentences. Do not elaborate."
)

_PROMPT_TEMPLATE = (
    "<|begin_of_text|>"
    "<|start_header_id|>system<|end_header_id|>\n\n"
    "{system}"
    "<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n"
    "{question}"
    "<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
)


@step
def build_preference_dataset(
    model_id: str,
    max_samples: int = 500,
):
    """Build a DPO preference Dataset from MongoDB documents.

    For each sampled document snippet:
      - Generates a question (same Ollama teacher used in SFT).
      - Generates a *chosen* response: Ollama with a detailed, helpful system prompt.
      - Generates a *rejected* response: Ollama with a deliberately terse prompt.

    This avoids loading the 8B model on GPU during dataset construction, keeping
    the full VRAM budget available for the training step.

    Returns:
        datasets.Dataset with columns ['prompt', 'chosen', 'rejected'].
    """
    import datasets as hf_datasets
    from langchain_ollama import OllamaLLM
    from config import get_settings
    from db.mongo import get_database

    settings = get_settings()
    db = get_database()

    cursor = db["raw_documents"].find({}, {"content": 1, "_id": 0}).limit(max_samples)
    docs = [d["content"] for d in cursor if d.get("content")]

    if not docs:
        raise ValueError(
            "No documents found in MongoDB. Upload documents and run the SFT pipeline first."
        )

    logger.info("Generating preference pairs from %d documents...", len(docs))

    llm = OllamaLLM(
        model="mannix/llama3.1-8b-abliterated",
        base_url=settings.ollama_base_url,
    )

    samples = []
    for i, content in enumerate(docs):
        snippet = content[:600].strip()
        if len(snippet) < 50:
            continue

        # 1. Generate question
        q_prompt = (
            "Given the following text, generate a single clear question whose answer "
            "is contained within the text. Return ONLY the question.\n\n"
            f"TEXT:\n{snippet}"
        )
        try:
            question = llm.invoke(q_prompt).strip()
            if len(question) < 10:
                question = f"What does the following describe? {snippet[:80]}..."
        except Exception as exc:
            logger.warning("Q generation failed for doc %d: %s", i, exc)
            question = f"Summarize: {snippet[:80]}..."

        # 2. Chosen response — detailed, using the snippet as context
        chosen_prompt = (
            f"Context:\n{snippet}\n\n"
            f"Question: {question}\n\n"
            "Provide a thorough, well-structured answer based on the context above."
        )
        try:
            chosen = llm.invoke(chosen_prompt).strip()
        except Exception as exc:
            logger.warning("Chosen generation failed for doc %d: %s", i, exc)
            chosen = snippet  # fallback: use snippet as the answer

        # 3. Rejected response — terse, no context, degraded quality
        rejected_prompt = f"Answer briefly in one sentence: {question}"
        try:
            rejected = llm.invoke(rejected_prompt).strip()
        except Exception as exc:
            logger.warning("Rejected generation failed for doc %d: %s", i, exc)
            rejected = "I don't know."

        # Skip pairs where chosen and rejected are identical or too similar
        if rejected.lower().strip() == chosen.lower().strip():
            continue

        samples.append({
            "prompt": _PROMPT_TEMPLATE.format(
                system=_CHOSEN_SYSTEM,
                question=question,
            ),
            "chosen": f"{chosen}<|eot_id|>",
            "rejected": f"{rejected}<|eot_id|>",
        })

        if (i + 1) % 50 == 0:
            logger.info("  %d / %d preference pairs generated", i + 1, len(docs))

    if not samples:
        raise ValueError("No preference pairs could be generated.")

    dataset = hf_datasets.Dataset.from_list(samples)
    logger.info("Preference dataset ready — %d pairs.", len(dataset))
    return dataset


@step
def run_dpo(base_model_id: str, dataset) -> str:
    """Fine-tune *base_model_id* adapter with DPO (Direct Preference Optimization).

    Memory strategy for RTX 3070 Ti 8 GB:
      - Load the SFT LoRA adapter in 4-bit NF4.
      - ref_model=None: TRL uses the frozen PEFT base as the implicit reference
        (avoids loading a second full model copy).
      - gradient_checkpointing + paged AdamW 8-bit for optimizer memory.

    Args:
        base_model_id: Either a HuggingFace model ID or a local path to a PEFT
                       adapter directory (e.g. ./models/sft/final_adapter).
                       If a local adapter path is given, the base model is loaded
                       from settings.base_model_id and the adapter is merged on top.

    Returns:
        Local path to the saved DPO LoRA adapter directory.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from trl import DPOConfig, DPOTrainer
    from config import get_settings

    settings = get_settings()
    output_dir = settings.dpo_output_dir
    os.makedirs(output_dir, exist_ok=True)

    hf_token = settings.huggingface_access_token or None
    is_local_adapter = os.path.isdir(base_model_id)

    # If base_model_id is a local PEFT adapter path, load true base model + adapter
    hf_model_id = settings.base_model_id if is_local_adapter else base_model_id
    adapter_path = base_model_id if is_local_adapter else None

    # ── Tokenizer ──────────────────────────────────────────────────────────────
    logger.info("Loading tokenizer: %s", hf_model_id)
    tokenizer = AutoTokenizer.from_pretrained(
        hf_model_id,
        token=hf_token,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── 4-bit model ────────────────────────────────────────────────────────────
    logger.info("Loading model in 4-bit NF4%s...",
                f" + SFT adapter from {adapter_path}" if adapter_path else "")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        hf_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        token=hf_token,
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    if adapter_path:
        # Load existing SFT LoRA weights as starting point for DPO
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
        logger.info("SFT LoRA adapter loaded from %s", adapter_path)
    else:
        # No prior adapter — apply fresh LoRA (training from base model directly)
        lora_cfg = LoraConfig(
            r=settings.lora_r,
            lora_alpha=settings.lora_alpha,
            lora_dropout=settings.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_cfg)

    model.print_trainable_parameters()

    # ── DPOTrainer ─────────────────────────────────────────────────────────────
    # ref_model=None → TRL uses the frozen PEFT base (implicit reference).
    # This is the recommended memory-efficient setting for PEFT-based DPO.
    dpo_args = DPOConfig(
        beta=settings.dpo_beta,
        output_dir=output_dir,
        num_train_epochs=settings.dpo_epochs,
        per_device_train_batch_size=settings.sft_batch_size,
        gradient_accumulation_steps=settings.sft_grad_accum,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,
        max_grad_norm=0.3,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
        max_length=settings.sft_max_seq_length,
        max_prompt_length=settings.sft_max_seq_length // 2,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    logger.info(
        "Starting DPO — %d pairs, %d epoch(s), beta=%.2f",
        len(dataset), settings.dpo_epochs, settings.dpo_beta,
    )
    trainer.train()

    # ── Save DPO adapter ───────────────────────────────────────────────────────
    final_adapter = os.path.join(output_dir, "final_adapter")
    trainer.model.save_pretrained(final_adapter)
    tokenizer.save_pretrained(final_adapter)
    logger.info("DPO adapter saved → %s", final_adapter)

    return final_adapter
