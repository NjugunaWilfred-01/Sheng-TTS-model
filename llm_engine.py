"""
LLM Conversational Brain: Sheng Persona with Multi-Backend Support.
"""
import time
import random
import logging
import requests
from typing import List, Dict, Tuple, Any

from config import (
    LLM_BACKEND, OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL,
    GEMINI_API_KEY, OLLAMA_BASE_URL, OLLAMA_MODEL
)
from sheng_lexicon import SHENG_SYSTEM_PROMPT, FEW_SHOT_CONVERSATIONS, HEURISTIC_INTENTS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ShengLLMEngine")


class ShengLLMEngine:
    def __init__(self, backend: str = LLM_BACKEND):
        self.backend = backend
        self.conversation_history: List[Dict[str, str]] = []

    def reset_history(self):
        """Clears conversation history."""
        self.conversation_history = []

    def generate_response(self, user_message: str) -> Tuple[str, Dict[str, Any]]:
        """
        Generates a natural Sheng / Swahili conversational reply.
        """
        start_time = time.time()
        backend_used = self.backend

        # Try chosen backend with automatic fallback to heuristic
        try:
            if self.backend == "lora":
                reply = self._generate_lora(user_message)
            elif self.backend == "openai" and OPENAI_API_KEY:
                reply = self._generate_openai(user_message)
            elif self.backend == "ollama":
                reply = self._generate_ollama(user_message)
            elif self.backend == "gemini" and GEMINI_API_KEY:
                reply = self._generate_gemini(user_message)
            else:
                backend_used = "heuristic"
                reply = self._generate_heuristic(user_message)
        except Exception as e:
            logger.warning(f"Backend '{self.backend}' failed ({e}), falling back to heuristic engine.")
            backend_used = "heuristic (fallback)"
            reply = self._generate_heuristic(user_message)

        # Update conversation history
        self.conversation_history.append({"role": "user", "content": user_message})
        self.conversation_history.append({"role": "assistant", "content": reply})

        duration = time.time() - start_time
        metadata = {
            "backend": backend_used,
            "llm_time_ms": round(duration * 1000, 2),
            "response_length": len(reply)
        }
        logger.info(f"LLM generated ({backend_used}) in {metadata['llm_time_ms']}ms: '{reply}'")
        return reply, metadata

    def _generate_heuristic(self, text: str) -> str:
        """Fast offline rule-based Sheng dialogue generator."""
        text_lower = text.lower()
        for pattern, responses in HEURISTIC_INTENTS:
            if pattern.search(text_lower):
                return random.choice(responses)

        # Default witty Sheng fallback responses
        fallbacks = [
            "Wazi chief! Sijakupata fiti but ebu jaribu kuieka na njia ingine."
        ]
        return random.choice(fallbacks)

    def _generate_openai(self, text: str) -> str:
        """Calls OpenAI-compatible API."""
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        messages = [{"role": "system", "content": SHENG_SYSTEM_PROMPT}]
        for turn in FEW_SHOT_CONVERSATIONS:
            messages.append({"role": "user", "content": turn["user"]})
            messages.append({"role": "assistant", "content": turn["bot"]})
        for msg in self.conversation_history[-6:]:
            messages.append(msg)
        messages.append({"role": "user", "content": text})

        payload = {
            "model": OPENAI_MODEL,
            "messages": messages,
            "max_tokens": 120,
            "temperature": 0.7
        }
        resp = requests.post(f"{OPENAI_BASE_URL}/chat/completions", headers=headers, json=payload, timeout=8)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()

    def _generate_ollama(self, text: str) -> str:
        """Calls local Ollama instance."""
        messages = [{"role": "system", "content": SHENG_SYSTEM_PROMPT}]
        for turn in FEW_SHOT_CONVERSATIONS:
            messages.append({"role": "user", "content": turn["user"]})
            messages.append({"role": "assistant", "content": turn["bot"]})
        for msg in self.conversation_history[-4:]:
            messages.append(msg)
        messages.append({"role": "user", "content": text})

        payload = {
            "model": OLLAMA_MODEL,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.7, "num_predict": 100}
        }
        resp = requests.post(f"{OLLAMA_BASE_URL}/chat", json=payload, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return data["message"]["content"].strip()

    def _generate_gemini(self, text: str) -> str:
        """Calls Gemini API with Sheng prompt."""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
        system_instruction = {"parts": [{"text": SHENG_SYSTEM_PROMPT}]}
        contents = []
        for turn in FEW_SHOT_CONVERSATIONS:
            contents.append({"role": "user", "parts": [{"text": turn["user"]}]})
            contents.append({"role": "model", "parts": [{"text": turn["bot"]}]})
        contents.append({"role": "user", "parts": [{"text": text}]})

        payload = {
            "system_instruction": system_instruction,
            "contents": contents,
            "generationConfig": {"maxOutputTokens": 100, "temperature": 0.7}
        }
        resp = requests.post(url, json=payload, timeout=8)
        resp.raise_for_status()
        data = resp.json()
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()

    def _generate_lora(self, text: str) -> str:
        """Generates Sheng conversational reply using local fine-tuned LoRA adapter."""
        from pathlib import Path
        import torch
        from peft import PeftModel, PeftConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from config import BASE_DIR

        adapter_path = str(BASE_DIR / "llm_sheng_lora_output" / "final_adapter")
        if not hasattr(self, "_lora_model") or self._lora_model is None:
            logger.info(f"Lazy loading fine-tuned Sheng LoRA model from {adapter_path}...")
            peft_config = PeftConfig.from_pretrained(adapter_path)
            self._lora_tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
            base_model = AutoModelForCausalLM.from_pretrained(
                peft_config.base_model_name_or_path,
                torch_dtype=torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            self._lora_model = PeftModel.from_pretrained(base_model, adapter_path)
            self._lora_model.eval()

        messages = [
            {"role": "system", "content": SHENG_SYSTEM_PROMPT},
        ]
        for turn in FEW_SHOT_CONVERSATIONS:
            messages.append({"role": "user", "content": turn["user"]})
            messages.append({"role": "assistant", "content": turn["bot"]})
        for msg in self.conversation_history[-4:]:
            messages.append(msg)
        messages.append({"role": "user", "content": text})

        prompt_text = self._lora_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._lora_tokenizer(prompt_text, return_tensors="pt")

        with torch.no_grad():
            output_ids = self._lora_model.generate(
                **inputs,
                max_new_tokens=60,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self._lora_tokenizer.eos_token_id
            )

        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        return self._lora_tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
