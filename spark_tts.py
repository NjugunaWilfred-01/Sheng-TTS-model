# -*- coding: utf-8 -*-
"""
Spark-TTS (0.5B) - Local Fine-Tuning Script
"""

import os
import sys
import re
import locale
import torch
import numpy as np
import torchaudio.transforms as T
from datasets import load_dataset, Audio
from huggingface_hub import snapshot_download
from unsloth import FastModel
from trl import SFTConfig, SFTTrainer

# ---------------------------------------------------------
# 0. SETUP & PATHS
# ---------------------------------------------------------
# NOTE: Ensure you have manually cloned Spark-TTS and your dataset repo into your working directory
sys.path.append('Spark-TTS')
from sparktts.models.audio_tokenizer import BiCodecTokenizer
from sparktts.utils.audio import audio_volume_normalize

# Local dataset paths
dataset_dir = "/home/ray/Desktop/mkuru/dataset"
segmented_dir = os.path.join(dataset_dir, "segmented")
manifest_path = os.path.join(dataset_dir, "clean_manifest.csv")

# ---------------------------------------------------------
# 1. LOAD BASE MODEL & LORA CONFIGURATION
# ---------------------------------------------------------
max_seq_length = 2048

# Download model and code
snapshot_download("unsloth/Spark-TTS-0.5B", local_dir="Spark-TTS-0.5B")

model, tokenizer = FastModel.from_pretrained(
    model_name=f"Spark-TTS-0.5B/LLM",
    max_seq_length=max_seq_length,
    dtype=torch.float32,
    full_finetuning=True,
    load_in_4bit=False,
)

# Add LoRA adapters
model = FastModel.get_peft_model(
    model,
    r=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=128,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

# ---------------------------------------------------------
# 2. TOKENIZER INITIALIZATION
# ---------------------------------------------------------
audio_tokenizer = BiCodecTokenizer("Spark-TTS-0.5B", "cuda")

def extract_wav2vec2_features(wavs: torch.Tensor) -> torch.Tensor:
    if wavs.shape[0] != 1:
         raise ValueError(f"Expected batch size 1, but got shape {wavs.shape}")
    wav_np = wavs.squeeze(0).cpu().numpy()

    processed = audio_tokenizer.processor(
        wav_np,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True,
    )
    input_values = processed.input_values
    input_values = input_values.to(audio_tokenizer.feature_extractor.device)

    model_output = audio_tokenizer.feature_extractor(input_values)

    if model_output.hidden_states is None:
         raise ValueError("Wav2Vec2Model did not return hidden states.")

    num_layers = len(model_output.hidden_states)
    required_layers = [11, 14, 16]

    feats_mix = (
        model_output.hidden_states[11] + model_output.hidden_states[14] + model_output.hidden_states[16]
    ) / 3

    return feats_mix

def formatting_audio_func(example):
    text = f"{example['source']}: {example['text']}" if "source" in example else example["text"]
    audio_array = example["audio"]["array"]
    sampling_rate = example["audio"]["sampling_rate"]

    target_sr = audio_tokenizer.config['sample_rate']

    if sampling_rate != target_sr:
        resampler = T.Resample(orig_freq=sampling_rate, new_freq=target_sr)
        audio_tensor_temp = torch.from_numpy(audio_array).float()
        audio_array = resampler(audio_tensor_temp).numpy()

    if audio_tokenizer.config["volume_normalize"]:
        audio_array = audio_volume_normalize(audio_array)

    ref_wav_np = audio_tokenizer.get_ref_clip(audio_array)

    audio_tensor = torch.from_numpy(audio_array).unsqueeze(0).float().to(audio_tokenizer.device)
    ref_wav_tensor = torch.from_numpy(ref_wav_np).unsqueeze(0).float().to(audio_tokenizer.device)

    feat = extract_wav2vec2_features(audio_tensor)

    batch = {
        "wav": audio_tensor,
        "ref_wav": ref_wav_tensor,
        "feat": feat.to(audio_tokenizer.device),
    }

    semantic_token_ids, global_token_ids = audio_tokenizer.model.tokenize(batch)

    global_tokens = "".join([f"<|bicodec_global_{i}|>" for i in global_token_ids.squeeze().cpu().numpy()])
    semantic_tokens = "".join([f"<|bicodec_semantic_{i}|>" for i in semantic_token_ids.squeeze().cpu().numpy()])

    inputs = [
        "<|task_tts|>",
        "<|start_content|>",
        text,
        "<|end_content|>",
        "<|start_global_token|>",
        global_tokens,
        "<|end_global_token|>",
        "<|start_semantic_token|>",
        semantic_tokens,
        "<|end_semantic_token|>",
        "<|im_end|>"
    ]
    inputs = "".join(inputs)
    return {"text": inputs}

# ---------------------------------------------------------
# 3. DATA PIPELINE
# ---------------------------------------------------------
print("Loading dataset...")

# Load completely fresh from CSV
dataset = load_dataset("csv", data_files=manifest_path, split="train")

# Map out exactly where the files are
file_locator = {}
for root, dirs, files in os.walk(segmented_dir):
    for file in files:
        if file.endswith(".wav"):
            file_locator[file] = os.path.join(root, file)

# Rename safely
dataset = dataset.rename_column("clip_filename", "audio")
dataset = dataset.rename_column("clean_sheng_text", "text")

# Drop EVERYTHING else to prevent conflicts
dataset = dataset.remove_columns([col for col in dataset.column_names if col not in ["audio", "text"]])

# Replace paths using local directory mapping
def apply_local_paths(row):
    filename = os.path.basename(row["audio"])
    row["audio"] = file_locator.get(filename, row["audio"])
    return row

dataset = dataset.map(apply_local_paths)

# Cast to Audio
dataset = dataset.cast_column("audio", Audio())

# Tokenize audio
print("Tokenizing audio (this may take a moment)...")
dataset = dataset.map(formatting_audio_func, remove_columns=["audio"])

print("Moving Bicodec model and Wav2Vec2Model to cpu.")
audio_tokenizer.model.cpu()
audio_tokenizer.feature_extractor.cpu()
torch.cuda.empty_cache()

# ---------------------------------------------------------
# 4. TRAINING
# ---------------------------------------------------------
print("Starting training...")
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    packing=False,
    args=SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,
        learning_rate=2e-4,
        fp16=False,
        bf16=False,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.001,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="none",
    ),
)

gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer.train()

used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"Peak reserved memory = {used_memory} GB.")

# ---------------------------------------------------------
# 5. INFERENCE & SAVING
# ---------------------------------------------------------
input_text = "Oyaaa unaingia lab tufike choche za madhe tushike mutura"
chosen_voice = None 

FastModel.for_inference(model)

@torch.inference_mode()
def generate_speech_from_text(text: str, device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")) -> np.ndarray:
    torch.compiler.reset()
    prompt = "".join(["<|task_tts|>", "<|start_content|>", text, "<|end_content|>", "<|start_global_token|>"])
    model_inputs = tokenizer([prompt], return_tensors="pt").to(device)
    
    max_generation_attempts = 5
    for attempt in range(1, max_generation_attempts + 1):
        generated_ids = model.generate(
            **model_inputs, max_new_tokens=2048, do_sample=True, temperature=0.8, top_k=50, top_p=1,
            eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id
        )
        generated_ids_trimmed = generated_ids[:, model_inputs.input_ids.shape[1]:]
        predicts_text = tokenizer.batch_decode(generated_ids_trimmed, skip_special_tokens=False)[0]

        semantic_matches = re.findall(r"<\|bicodec_semantic_(\d+)\|>", predicts_text)
        if not semantic_matches: return np.array([], dtype=np.float32)
        pred_semantic_ids = torch.tensor([int(token) for token in semantic_matches]).long().unsqueeze(0)

        global_matches = re.findall(r"<\|bicodec_global_(\d+)\|>", predicts_text)
        pred_global_ids = torch.tensor([int(token) for token in global_matches]).long().unsqueeze(0).unsqueeze(0) if global_matches else torch.zeros((1, 1, 1), dtype=torch.long)

        audio_tokenizer.device = device
        audio_tokenizer.model.to(device)
        try:
            wav_np = audio_tokenizer.detokenize(pred_global_ids.to(device).squeeze(0), pred_semantic_ids.to(device))
            return wav_np
        except RuntimeError as e:
            if "mat1 and mat2 shapes cannot be multiplied" not in str(e): raise
            continue
    raise RuntimeError("Failed to generate valid tokens.")

if __name__ == "__main__":
    print(f"Generating speech for: '{input_text}'")
    text = f"{chosen_voice}: " + input_text if chosen_voice else input_text
    generated_waveform = generate_speech_from_text(input_text)

    if generated_waveform.size > 0:
        import soundfile as sf
        output_filename = "generated_speech_controllable.wav"
        sample_rate = audio_tokenizer.config.get("sample_rate", 16000)
        sf.write(output_filename, generated_waveform, sample_rate)
        print(f"Audio saved to {output_filename}")

print("Saving local LoRA adapters...")
model.save_pretrained("spark_tts_lora")
tokenizer.save_pretrained("spark_tts_lora")
print("Process completed successfully!")
