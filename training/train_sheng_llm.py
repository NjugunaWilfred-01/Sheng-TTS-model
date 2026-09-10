"""
Production-Ready HuggingFace PEFT LoRA Fine-Tuning Pipeline for Sheng Conversational LLM.
Fine-tunes Qwen2.5-Instruct on dataset/train_sheng_sft.jsonl with ChatML formatting.
"""
import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TrainShengLLM")


def load_sft_dataset(jsonl_path: str, tokenizer: Any, max_seq_length: int = 256):
    """Loads and tokenizes ChatML dialogue turns for causal LM training."""
    raw_data = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                raw_data.append(json.loads(line.strip()))

    logger.info(f"Loaded {len(raw_data)} conversational records from {jsonl_path}")

    tokenized_samples = []
    for item in raw_data:
        dialogue = item["conversations"]
        # Format into ChatML prompt string
        formatted_text = tokenizer.apply_chat_template(
            dialogue,
            tokenize=False,
            add_generation_prompt=False
        )
        tokenized = tokenizer(
            formatted_text,
            max_length=max_seq_length,
            truncation=True,
            padding=False
        )
        tokenized_samples.append({
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": tokenized["input_ids"].copy()
        })

    logger.info(f"Successfully tokenized {len(tokenized_samples)} samples.")
    return Dataset.from_list(tokenized_samples)


def train_sheng_llm(
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    train_jsonl: str = "dataset/train_sheng_sft.jsonl",
    output_dir: str = "llm_sheng_lora_output",
    num_epochs: int = 3,
    learning_rate: float = 2e-4,
    batch_size: int = 4,
    lora_r: int = 16,
    lora_alpha: int = 32
):
    logger.info(f"🚀 Initializing Sheng LLM LoRA Fine-Tuning on '{model_name}'...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Compute Device: {device.upper()}")

    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load Base Model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True
    )
    model.config.use_cache = False

    # 3. Configure PEFT LoRA
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 4. Prepare Dataset
    dataset = load_sft_dataset(train_jsonl, tokenizer)
    split_dataset = dataset.train_test_split(test_size=0.1, seed=42) if len(dataset) > 10 else {"train": dataset, "test": dataset}
    train_set = split_dataset["train"]
    eval_set = split_dataset["test"]

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
        return_tensors="pt"
    )

    # 5. Training Arguments
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(out_path),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=2,
        learning_rate=learning_rate,
        warmup_steps=10,
        num_train_epochs=num_epochs,
        logging_steps=5,
        save_strategy="epoch",
        eval_strategy="epoch" if len(eval_set) > 0 else "no",
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to="none",
        dataloader_num_workers=0
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=eval_set,
        data_collator=data_collator,
        processing_class=tokenizer
    )

    logger.info("🏋️ Starting LLM Training Loop...")
    trainer.train()

    # 6. Save Adapter & Tokenizer
    adapter_save_dir = out_path / "final_adapter"
    adapter_save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_save_dir))
    tokenizer.save_pretrained(str(adapter_save_dir))
    logger.info(f"✅ Sheng LLM LoRA Fine-Tuning Complete! Adapter saved to: {adapter_save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sheng LLM Fine-Tuning Recipe")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--train_jsonl", type=str, default="dataset/train_sheng_sft.jsonl")
    parser.add_argument("--output_dir", type=str, default="llm_sheng_lora_output")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)

    args = parser.parse_args()
    train_sheng_llm(
        model_name=args.model_name,
        train_jsonl=args.train_jsonl,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )
