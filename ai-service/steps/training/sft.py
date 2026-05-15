"""
steps/training/sft.py — ZenML steps: dataset construction + SFT trainer.

Phase 4 (Chapter 5 — Supervised Fine-Tuning).
  - build_instruction_dataset: Pull raw docs from MongoDB, generate Q/A pairs via
    Ollama teacher, return a HuggingFace Dataset in Llama 3.1 chat format.
  - run_sft: 4-bit QLoRA fine-tuning with PEFT + TRL SFTTrainer, optimised for
    RTX 3070 Ti (8 GB VRAM): batch=1, grad_accum=4, gradient_checkpointing.
"""

import os
from loguru import logger
from zenml import step

# ── Llama 3.1 chat template ────────────────────────────────────────────────────
_SYSTEM_PROMPT = (
    "You are a knowledgeable research assistant. "
    "Answer questions concisely and accurately based on your training."
)


def _format_llama3(system: str, user: str, assistant: str) -> str:
    """Format a single turn using the Llama 3.1 instruction template."""
    return (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\n"
        f"{system}"
        "<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user}"
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{assistant}"
        "<|eot_id|>"
    )


@step
def build_instruction_dataset(
    model_id: str,
    max_samples: int = 1000,
):
    """Build an instruction-tuning Dataset from MongoDB documents.

    For each raw document:
      1. Truncate content to a readable snippet (~800 chars).
      2. Ask the Ollama teacher LLM to generate a question for that snippet.
      3. Wrap (question, snippet) in the Llama 3.1 chat template.
      4. Return a HuggingFace Dataset with a single 'text' column.

    Args:
        model_id: Unused here (kept for ZenML pipeline signature compatibility).
        max_samples: Maximum number of documents to pull from MongoDB.

    Returns:
        datasets.Dataset with columns ['text'].
    """
    import datasets as hf_datasets
    from langchain_ollama import OllamaLLM
    from config import get_settings
    from db.mongo import get_database

    settings = get_settings()
    db = get_database()

    # Pull raw document content from MongoDB
    cursor = db["raw_documents"].find({}, {"content": 1, "_id": 0}).limit(max_samples)
    docs = [d["content"] for d in cursor if d.get("content")]

    if not docs:
        raise ValueError(
            "No documents found in MongoDB. Upload documents via the /documents/upload "
            "endpoint before running the training pipeline."
        )

    logger.info("Generating Q/A pairs from %d documents via Ollama teacher...", len(docs))

    llm = OllamaLLM(
        model="mannix/llama3.1-8b-abliterated",
        base_url=settings.ollama_base_url,
    )

    samples = []
    for i, content in enumerate(docs):
        snippet = content[:800].strip()
        if len(snippet) < 50:
            continue

        prompt = (
            "Given the following text, generate a single clear question whose answer "
            "is contained within the text. Return ONLY the question.\n\n"
            f"TEXT:\n{snippet}"
        )
        try:
            question = llm.invoke(prompt).strip()
            # Sanity-check: must look like an actual question
            if len(question) < 10:
                question = f"What does the following describe? {snippet[:80]}..."
        except Exception as exc:
            logger.warning("Q generation failed for doc %d: %s", i, exc)
            question = f"Summarize the following text: {snippet[:80]}..."

        samples.append({"text": _format_llama3(_SYSTEM_PROMPT, question, snippet)})

        if (i + 1) % 50 == 0:
            logger.info("  %d / %d pairs generated", i + 1, len(docs))

    if not samples:
        raise ValueError("No Q/A pairs could be generated from the available documents.")

    dataset = hf_datasets.Dataset.from_list(samples)
    logger.info("Instruction dataset ready — %d samples.", len(dataset))
    return dataset


@step
def run_sft(base_model_id: str, dataset) -> str:
    """Fine-tune *base_model_id* with 4-bit QLoRA (PEFT) + TRL SFTTrainer.

    Memory budget for RTX 3070 Ti 8 GB:
      - 4-bit NF4 quantization   → ~5.5 GB
      - LoRA adapters (r=16)     → ~0.2 GB
      - Activations + optimizer  → ~1.5 GB  (gradient_checkpointing + paged AdamW)
      Total                      → ~7.2 GB  (fits with 0.8 GB headroom)

    Training config (overridable via env / .env):
      LR=2e-4, cosine schedule, warmup_ratio=0.03, bf16=True

    Returns:
        Local path to the saved LoRA adapter directory.
    """
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTConfig, SFTTrainer
    from config import get_settings

    settings = get_settings()
    output_dir = settings.sft_output_dir
    os.makedirs(output_dir, exist_ok=True)

    hf_token = settings.huggingface_access_token or None

    # ── Tokenizer ──────────────────────────────────────────────────────────────
    logger.info("Loading tokenizer: %s", base_model_id)
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        token=hf_token,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── 4-bit model ────────────────────────────────────────────────────────────
    logger.info("Loading model in 4-bit NF4 (bitsandbytes)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        token=hf_token,
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    # ── LoRA adapter ───────────────────────────────────────────────────────────
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

    # ── SFTTrainer ─────────────────────────────────────────────────────────────
    sft_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=settings.sft_epochs,
        per_device_train_batch_size=settings.sft_batch_size,
        gradient_accumulation_steps=settings.sft_grad_accum,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        tf32=True,
        max_grad_norm=0.3,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
        max_seq_length=settings.sft_max_seq_length,
        dataset_text_field="text",
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    logger.info(
        "Starting SFT — %d samples, %d epochs, batch=%d×%d",
        len(dataset), settings.sft_epochs,
        settings.sft_batch_size, settings.sft_grad_accum,
    )
    trainer.train()

    # ── Save LoRA adapter ──────────────────────────────────────────────────────
    adapter_path = os.path.join(output_dir, "final_adapter")
    trainer.model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    logger.info("LoRA adapter saved → %s", adapter_path)

    return adapter_path
