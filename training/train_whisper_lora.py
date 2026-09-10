"""
Production-Ready HuggingFace PEFT LoRA Fine-Tuning Pipeline for Whisper on Sheng/Swahili.
Loads dataset/train_sheng_asr.jsonl, extracts audio mel features, trains LoRA adapters,
and saves the fine-tuned adapter weights.
"""
import os
import sys
import json
import logging
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import torch
import soundfile as sf
import numpy as np

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TrainWhisperLoRA")


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Split inputs and labels since they have different lengths and need different padding methods
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # If bos token is appended in previous tokenization step, cut bos token
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def prepare_dataset_from_jsonl(jsonl_path: str, processor: Any):
    """Loads and formats the JSONL dataset for Whisper training."""
    from datasets import Dataset

    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line.strip()))

    logger.info(f"Loaded {len(records)} training samples from {jsonl_path}")

    processed_data = []
    for item in records:
        audio_path = item["audio_filepath"]
        text = item["normalized_text"]

        if not Path(audio_path).exists() or not text:
            continue

        try:
            # Read 16kHz audio
            speech_array, sampling_rate = sf.read(audio_path)
            if len(speech_array.shape) > 1:
                speech_array = speech_array.mean(axis=1)  # Convert to mono

            # Whisper expects 16kHz audio
            if sampling_rate != 16000:
                import librosa
                speech_array = librosa.resample(speech_array, orig_sr=sampling_rate, target_sr=16000)

            # Cap long audio chunks to 30 seconds (Whisper standard window)
            max_samples = 16000 * 30
            if len(speech_array) > max_samples:
                speech_array = speech_array[:max_samples]

            # Extract Mel Spectrogram input features
            input_features = processor.feature_extractor(speech_array, sampling_rate=16000).input_features[0]

            # Tokenize target text with Swahili language tags (capped to max Whisper decoder length 440)
            labels = processor.tokenizer(text, max_length=440, truncation=True).input_ids

            processed_data.append({
                "input_features": input_features,
                "labels": labels
            })
        except Exception as e:
            logger.warning(f"Skipping corrupt audio {audio_path}: {e}")

    logger.info(f"Successfully processed {len(processed_data)} feature-extracted samples.")
    return Dataset.from_list(processed_data)


def run_training(
    model_name: str = "openai/whisper-tiny",
    train_jsonl: str = "dataset/train_sheng_asr.jsonl",
    output_dir: str = "whisper_sheng_lora_output",
    num_epochs: int = 5,
    learning_rate: float = 1e-3,
    batch_size: int = 4,
    lora_r: int = 16,
    lora_alpha: int = 32
):
    from transformers import (
        WhisperForConditionalGeneration,
        WhisperProcessor,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments
    )
    from peft import LoraConfig, get_peft_model

    logger.info(f"🚀 Initializing Whisper PEFT LoRA Training on '{model_name}'...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Compute Device: {device.upper()}")

    # 1. Load Processor & Tokenizer
    processor = WhisperProcessor.from_pretrained(
        model_name,
        language="swahili",
        task="transcribe"
    )

    # 2. Load Base Model
    model = WhisperForConditionalGeneration.from_pretrained(
        model_name,
        device_map="auto" if torch.cuda.is_available() else None
    )

    # Freeze base model parameters
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False

    # 3. Configure LoRA
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 4. Prepare Dataset
    dataset = prepare_dataset_from_jsonl(train_jsonl, processor)
    if len(dataset) == 0:
        logger.error("Dataset is empty. Aborting training.")
        return

    # Train / Validation Split
    split_dataset = dataset.train_test_split(test_size=0.15, seed=42) if len(dataset) > 5 else {"train": dataset, "test": dataset}
    train_set = split_dataset["train"]
    eval_set = split_dataset["test"]

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # 5. Training Arguments
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(out_path),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=2,
        learning_rate=learning_rate,
        warmup_steps=5,
        num_train_epochs=num_epochs,
        logging_steps=2,
        save_strategy="epoch",
        eval_strategy="epoch" if len(eval_set) > 0 else "no",
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        predict_with_generate=True,
        report_to="none",
        dataloader_num_workers=0
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_set,
        eval_dataset=eval_set,
        data_collator=data_collator,
        processing_class=processor.feature_extractor
    )

    logger.info("🏋️ Starting Training Loop...")
    trainer.train()

    # 6. Save Adapter Weights & Processor
    adapter_save_dir = out_path / "final_adapter"
    adapter_save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_save_dir))
    processor.save_pretrained(str(adapter_save_dir))
    logger.info(f"✅ LoRA Fine-Tuning Complete! Adapter saved to: {adapter_save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Whisper PEFT LoRA Fine-Tuning for Sheng")
    parser.add_argument("--model_name", type=str, default="openai/whisper-tiny")
    parser.add_argument("--train_jsonl", type=str, default="dataset/train_sheng_asr.jsonl")
    parser.add_argument("--output_dir", type=str, default="whisper_sheng_lora_output")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)

    args = parser.parse_args()
    run_training(
        model_name=args.model_name,
        train_jsonl=args.train_jsonl,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )
