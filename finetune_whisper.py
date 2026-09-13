"""
finetune_whisper.py - LoRA fine-tuning of Whisper for Sheng (Swahili) ASR.

Designed for the zoza_transcripts/mapped_data dataset (123 audio/transcript pairs).
Internally splits into train/eval (e.g. 100/23) so evaluation is leakage-free.

Recommended: run on a GPU (Google Colab free T4 works well). On CPU this will
be extremely slow.

Steps:
  1. pip install datasets transformers accelerate peft evaluate jiwer ctranslate2 soundfile librosa
  2. python finetune_whisper.py --data_dir zoza_transcripts/mapped_data --output_dir whisper-sheng-lora
  3. Convert to faster-whisper format (see printout at the end of training).
  4. Benchmark: python eval_asr.py --model ./whisper-sheng-ct2 ... (the held-out
     eval split is printed at the start; use it for BOTH baseline and fine-tuned runs).
"""
import argparse
import random
import re
from pathlib import Path

import torch
from datasets import Audio, Dataset
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

MODEL_ID = "openai/whisper-small"   # use "openai/whisper-base" if VRAM is tight


def load_pairs(data_dir: Path):
    audio_dir = data_dir / "audios"
    transcript_dir = data_dir / "transcripts"
    rows = []
    for audio_path in sorted(audio_dir.glob("audio_*_for_script_*.mp3")):
        idx = int(re.search(r"audio_(\d+)", audio_path.name).group(1))
        transcript_path = transcript_dir / f"transcript_{idx}_for_audio_{idx}.txt"
        if transcript_path.exists():
            rows.append({"audio": str(audio_path), "text": transcript_path.read_text(encoding="utf-8").strip()})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="zoza_transcripts/mapped_data")
    ap.add_argument("--output_dir", default="whisper-sheng-lora")
    ap.add_argument("--eval_fraction", type=float, default=0.2, help="held-out fraction (default 0.2 -> ~23 of 123)")
    ap.add_argument("--epochs", type=float, default=15)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3, help="LoRA typically uses a higher LR than full FT")
    args = ap.parse_args()

    from peft import LoraConfig, get_peft_model

    rows = load_pairs(Path(args.data_dir))
    print(f"Loaded {len(rows)} pairs from {args.data_dir}")
    random.Random(42).shuffle(rows)
    n_eval = max(1, int(len(rows) * args.eval_fraction))
    eval_rows, train_rows = rows[:n_eval], rows[n_eval:]
    print(f"Split: {len(train_rows)} train / {n_eval} held-out eval")
    print("HELD-OUT IDS (use these for eval_asr.py):",
          sorted(int(re.search(r'audio_(\d+)', r['audio']).group(1)) for r in eval_rows))

    processor = WhisperProcessor.from_pretrained(MODEL_ID, language="swahili", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
    model.config.forced_decoder_ids = None          # let training learn freely; language set via processor
    model.config.suppress_tokens = []

    lora = LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.05, bias="none",
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"],
        # task_type="SEQ_2_SEQ_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    def preprocess(batch):
        audio = batch["audio"]  # already resampled by cast
        batch["input_features"] = processor.feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
        batch["labels"] = processor.tokenizer(batch["text"]).input_ids
        return batch

    train_ds = Dataset.from_list(train_rows).cast_column("audio", Audio(sampling_rate=16000))
    eval_ds = Dataset.from_list(eval_rows).cast_column("audio", Audio(sampling_rate=16000))
    train_ds = train_ds.map(preprocess, remove_columns=["audio"], desc="featurizing train")
    eval_ds = eval_ds.map(preprocess, remove_columns=["audio"], desc="featurizing eval")

    class Collator:
        def __call__(self, features):
            batch = processor.feature_extractor.pad(
                [{"input_features": f["input_features"]} for f in features], return_tensors="pt")
            labels = processor.tokenizer.pad(
                [{"input_ids": f["labels"]} for f in features], return_tensors="pt")
            batch["labels"] = labels["input_ids"].masked_fill(labels.attention_mask.ne(1), -100)
            return batch

    import evaluate
    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        text_pred = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        text_ref = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        return {"wer": wer_metric.compute(predictions=text_pred, references=text_ref)}

    targs = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=2,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        warmup_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        fp16=torch.cuda.is_available(),
        logging_steps=5,
        report_to="none",
        seed=42,
    )

    trainer = Seq2SeqTrainer(
        model=model, args=targs, train_dataset=train_ds, eval_dataset=eval_ds,
        data_collator=Collator(), compute_metrics=compute_metrics,
    )
    trainer.train()

    # Merge LoRA adapters into the base model and save in HF format
    final_dir = Path(args.output_dir) / "final"
    merged = model.merge_and_unload()
    merged.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)
    print(f"\nSaved merged model to {final_dir}")

    print("\nNext steps:")
    print(f"  1. Convert for faster-whisper:")
    print(f"     ct2-transformers-converter --model {final_dir} \\")
    print(f"         --output_dir {final_dir}-ct2 --copy_files tokenizer.json preprocessor_config.json config.json \\")
    print(f"         --quantization int8")
    print(f"  2. Benchmark the fine-tuned model on the HELD-OUT files listed above:")
    print(f"     python eval_asr.py --model {final_dir}-ct2 --device cpu \\")
    print(f"         --data_dir <heldout_subset> --out results_experiment.csv")
    print(f"  3. Benchmark the baseline (whisper small) on the SAME held-out files:")
    print(f"     python eval_asr.py --model small --device cpu \\")
    print(f"         --data_dir <heldout_subset> --out results_baseline.csv")


if __name__ == "__main__":
    main()
