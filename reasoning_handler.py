"""VnexAI reasoning handler — run CoT inference on exported .bin models.

Place this file next to:
  - model.bin
  - tokenizer.bin
  - transformer_model.py  (from VnexAI panel)
  - chatbot_tokenizer.py  (from VnexAI panel)

Usage:
    python reasoning_handler.py "What is 14 plus 27?"
    python reasoning_handler.py "your question" --json
    python reasoning_handler.py "your question" --no-self-check

For agent pipelines:
    from reasoning_handler import ReasoningEngine
    engine = ReasoningEngine()
    result = engine.ask("What is 14 plus 27?")
    print(result["reasoning"], result["answer"])
"""
from __future__ import annotations

import argparse
import json
import os
import sys

MODEL_PATH = os.environ.get("VNEXAI_MODEL_PATH", "model.bin")
TOKENIZER_PATH = os.environ.get("VNEXAI_TOKENIZER_PATH", "tokenizer.bin")
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


def split_reasoning(raw_text: str):
    if THINK_CLOSE not in raw_text:
        return None, raw_text.strip()
    reasoning, _, answer = raw_text.partition(THINK_CLOSE)
    reasoning = reasoning.strip()
    if reasoning.lower().startswith(THINK_OPEN.lower()):
        reasoning = reasoning[len(THINK_OPEN):].strip()
    return (reasoning or None), answer.strip()


class ReasoningEngine:
    def __init__(self, model_path: str = MODEL_PATH, tokenizer_path: str = TOKENIZER_PATH):
        from transformer_model import TransformerChatbot
        from chatbot_tokenizer import ChatbotTokenizer

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")

        self.model = TransformerChatbot.load(model_path)
        self.tokenizer = ChatbotTokenizer()
        self.tokenizer.load(tokenizer_path)
        if not getattr(self.model, "decoder_only", False):
            raise ValueError("Reasoning requires a decoder-only transformer model.")

    def ask(
        self,
        message: str,
        *,
        temperature: float = 0.8,
        self_check: bool = True,
        max_length: int = 120,
    ) -> dict:
        prompt1 = f"{message} {THINK_OPEN}"
        raw1 = self.model.generate(prompt1, self.tokenizer, max_length=max_length, temperature=temperature)
        reasoning1, answer1 = split_reasoning(THINK_OPEN + raw1)

        if not self_check:
            return {"reasoning": reasoning1, "answer": answer1, "self_checked": False}

        prompt2 = (
            f"{message} {THINK_OPEN} {reasoning1 or ''} {THINK_CLOSE} {answer1} "
            f"Let me double-check this. {THINK_OPEN}"
        )
        raw2 = self.model.generate(prompt2, self.tokenizer, max_length=max_length, temperature=temperature)
        reasoning2, answer2 = split_reasoning(THINK_OPEN + raw2)
        return {
            "reasoning": reasoning2 or reasoning1,
            "answer": answer2 or answer1,
            "self_checked": True,
        }


def main():
    parser = argparse.ArgumentParser(description="VnexAI chain-of-thought reasoning")
    parser.add_argument("message")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-length", type=int, default=120)
    parser.add_argument("--no-self-check", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print JSON for agents")
    parser.add_argument("--hide-think", action="store_true")
    args = parser.parse_args()

    engine = ReasoningEngine()
    result = engine.ask(
        args.message,
        temperature=args.temperature,
        self_check=not args.no_self_check,
        max_length=args.max_length,
    )

    if args.json:
        print(json.dumps(result, ensure_ascii=False))
        return

    if result.get("reasoning") and not args.hide_think:
        print("--- thinking ---")
        print(result["reasoning"])
        print("--- answer ---")
    print(result["answer"])


if __name__ == "__main__":
    main()
