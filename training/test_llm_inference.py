"""
Inference tester for the fine-tuned Sheng Conversational LLM LoRA Adapter.
"""
import sys
import torch
import argparse
from pathlib import Path
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_DIR = Path(__file__).resolve().parent.parent
ADAPTER_DIR = BASE_DIR / "llm_sheng_lora_output" / "final_adapter"


def generate_sheng_response(user_prompt: str, adapter_path: str = None) -> str:
    path_to_use = adapter_path or str(ADAPTER_DIR)
    print(f"Loading base model and Sheng LoRA adapter from {path_to_use}...")

    # 1. Load Config & Tokenizer
    peft_config = PeftConfig.from_pretrained(path_to_use)
    tokenizer = AutoTokenizer.from_pretrained(path_to_use, trust_remote_code=True)

    # 2. Load Base Model + LoRA Adapter
    base_model = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path,
        torch_dtype=torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base_model, path_to_use)
    model.eval()

    # 3. Format with ChatML Template
    messages = [
        {"role": "system", "content": "Wewe ni msee mjanja wa Nairobi anayeongea Sheng safi na Kiswahili ya mtaani. Jibu kwa kifupi na uchangamfu wa kishikaji."},
        {"role": "user", "content": user_prompt}
    ]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors="pt")

    # 4. Generate Output
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=80,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode only the generated response
    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    print("\n=======================================================")
    print(f"🗣️  User Prompt:   \"{user_prompt}\"")
    print(f"🤖 Fine-Tuned Sheng Reply: \"{response}\"")
    print("=======================================================\n")
    return response


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test fine-tuned Sheng LLM")
    parser.add_argument("--prompt", type=str, default="Niaje bazenga, form ni gani leo mtaani?")
    args = parser.parse_args()
    generate_sheng_response(args.prompt)
