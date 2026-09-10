"""
Inference script to test the fine-tuned Whisper PEFT LoRA adapter on Sheng speech audio.
"""
import sys
import torch
import soundfile as sf
from pathlib import Path
from peft import PeftModel, PeftConfig
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from sheng_lexicon import ShengNormalizer

ADAPTER_DIR = BASE_DIR / "whisper_sheng_lora_output" / "final_adapter"


def transcribe_with_lora(audio_path: str) -> str:
    print(f"Loading base model and fine-tuned LoRA adapter from {ADAPTER_DIR}...")
    
    # 1. Load config & processor
    peft_config = PeftConfig.from_pretrained(str(ADAPTER_DIR))
    processor = WhisperProcessor.from_pretrained(str(ADAPTER_DIR), language="swahili", task="transcribe")

    # 2. Load Base Model + LoRA Adapter
    base_model = WhisperForConditionalGeneration.from_pretrained(peft_config.base_model_name_or_path)
    model = PeftModel.from_pretrained(base_model, str(ADAPTER_DIR))
    model.eval()

    # 3. Read Audio & Extract Features
    speech_array, sampling_rate = sf.read(audio_path)
    if len(speech_array.shape) > 1:
        speech_array = speech_array.mean(axis=1)

    if sampling_rate != 16000:
        import librosa
        speech_array = librosa.resample(speech_array, orig_sr=sampling_rate, target_sr=16000)

    # Cap to 30s
    if len(speech_array) > 16000 * 30:
        speech_array = speech_array[:16000 * 30]

    inputs = processor.feature_extractor(speech_array, sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features

    # 4. Generate Transcription with LoRA
    with torch.no_grad():
        predicted_ids = model.generate(
            input_features,
            language="swahili",
            task="transcribe",
            max_new_tokens=128
        )

    transcription = processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
    normalized = ShengNormalizer.normalize(transcription)

    print("\n=======================================================")
    print(f"🎙️  Fine-Tuned LoRA Transcription: \"{transcription}\"")
    print(f"✨ Normalized Sheng Output:        \"{normalized}\"")
    print("=======================================================\n")
    return normalized


if __name__ == "__main__":
    test_audio = sys.argv[1] if len(sys.argv) > 1 else str(BASE_DIR / "temp_audio" / "demo_input_1.mp3")
    transcribe_with_lora(test_audio)
