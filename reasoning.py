"""Built-in chain-of-thought reasoning for VnexAI decoder-only models."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
COT_STATE_FILE = os.path.join(PROJECT_ROOT, "cot_reasoning_data.json")
DATASET_FILE = os.path.join(PROJECT_ROOT, "cot_reasoning_dataset.py")
HANDLER_FILE = os.path.join(PROJECT_ROOT, "reasoning_handler.py")


def _think(user: str, steps: List[str], answer: str) -> Dict[str, str]:
    reasoning = "\n".join(f"Step {i + 1}: {s}" for i, s in enumerate(steps))
    bot = f"{THINK_OPEN}\n{reasoning}\n{THINK_CLOSE}\n{answer}"
    return {"user": user.strip(), "bot": bot}


CORE_COT_PAIRS: List[Dict[str, str]] = [
    _think("What is 14 plus 27?", ["Identify the operands: 14 and 27.", "Add: 14 + 27 = 41."], "14 plus 27 is 41."),
    _think("What is 9 times 7?", ["Multiply 9 by 7.", "9 × 7 = 63."], "9 times 7 is 63."),
    _think("What is 144 divided by 12?", ["Division: 144 ÷ 12.", "12 × 12 = 144, so the quotient is 12."], "144 divided by 12 is 12."),
    _think("What is 15 minus 8?", ["Subtract 8 from 15.", "15 - 8 = 7."], "15 minus 8 is 7."),
    _think("What is 25 squared?", ["Square means multiply a number by itself.", "25 × 25 = 625."], "25 squared is 625."),
    _think("If I have 3 apples and buy 5 more, how many do I have?", ["Start with 3.", "Add 5: 3 + 5 = 8."], "You would have 8 apples."),
    _think("A train travels 60 miles in 1.5 hours. What is its speed?", ["Speed = distance ÷ time.", "60 ÷ 1.5 = 40 miles per hour."], "The train's speed is 40 miles per hour."),
    _think("If a shirt costs $20 and is 25% off, what is the sale price?", ["Discount = 25% of 20 = 5.", "Sale price = 20 - 5 = 15."], "The shirt costs $15 after the discount."),
    _think("I have $50 and spend $18 on lunch and $12 on a book. How much is left?", ["Total spent = 18 + 12 = 30.", "Remaining = 50 - 30 = 20."], "You have $20 left."),
    _think("How many minutes are in 2.5 hours?", ["1 hour = 60 minutes.", "2.5 × 60 = 150 minutes."], "There are 150 minutes in 2.5 hours."),
    _think("Which is bigger, 0.5 or 1/3?", ["1/3 ≈ 0.333.", "Compare 0.5 and 0.333; 0.5 is larger."], "0.5 is bigger than 1/3."),
    _think("What is the area of a rectangle 4 m by 9 m?", ["Area = length × width.", "4 × 9 = 36 square meters."], "The area is 36 square meters."),
    _think("If a car uses 1 gallon every 25 miles, how many gallons for 100 miles?", ["Gallons = distance ÷ mpg.", "100 ÷ 25 = 4 gallons."], "The car would use 4 gallons."),
    _think("Is 17 a prime number?", ["Test divisors up to √17 ≈ 4.1.", "17 is not divisible by 2, 3, or any smaller prime."], "Yes, 17 is a prime number."),
    _think("What is half of 86?", ["Half means divide by 2.", "86 ÷ 2 = 43."], "Half of 86 is 43."),
    _think("If 3 workers finish a wall in 12 days, how long for 6 workers?", ["Total work = 3 × 12 = 36 worker-days.", "6 workers → 36 ÷ 6 = 6 days."], "6 workers would take 6 days."),
    _think("What is the next number in 2, 4, 8, 16?", ["Each term doubles.", "16 × 2 = 32."], "The next number is 32."),
    _think("If today is Wednesday, what day is it in 10 days?", ["10 days = 1 week + 3 days.", "Wednesday + 3 days = Saturday."], "In 10 days it will be Saturday."),
    _think("A recipe needs 2 cups of flour for 12 cookies. How much for 36 cookies?", ["Scale factor = 36 ÷ 12 = 3.", "Flour = 2 × 3 = 6 cups."], "You need 6 cups of flour."),
    _think("What is 20% of 150?", ["20% = 0.20.", "0.20 × 150 = 30."], "20% of 150 is 30."),
    _think("Convert 3.5 kilometers to meters.", ["1 km = 1000 m.", "3.5 × 1000 = 3500 meters."], "3.5 kilometers is 3500 meters."),
    _think("If x + 7 = 19, what is x?", ["Isolate x by subtracting 7.", "x = 19 - 7 = 12."], "x equals 12."),
    _think("Which is heavier: 1 kg or 950 g?", ["Convert: 1 kg = 1000 g.", "1000 g > 950 g."], "1 kg is heavier than 950 g."),
    _think("A box has 4 rows of 6 eggs. How many eggs total?", ["Multiply rows by columns.", "4 × 6 = 24 eggs."], "There are 24 eggs in the box."),
    _think("If you flip a fair coin twice, how many possible outcomes?", ["Each flip has 2 outcomes.", "2 × 2 = 4 total outcomes."], "There are 4 possible outcomes."),
    _think("What is the average of 10, 20, and 30?", ["Sum = 10 + 20 + 30 = 60.", "Average = 60 ÷ 3 = 20."], "The average is 20."),
    _think("How many sides does a hexagon have?", ["Recall polygon names.", "Hex means six."], "A hexagon has 6 sides."),
    _think("If a laptop costs $800 plus 8% tax, what is the total?", ["Tax = 0.08 × 800 = 64.", "Total = 800 + 64 = 864."], "The total is $864."),
    _think("What is 0.75 as a fraction in lowest terms?", ["0.75 = 75/100.", "Reduce: 75/100 = 3/4."], "0.75 is 3/4."),
    _think("A pool fills at 5 liters per minute. How long to fill 200 liters?", ["Time = volume ÷ rate.", "200 ÷ 5 = 40 minutes."], "It takes 40 minutes."),
    _think("Is 91 divisible by 7?", ["Divide 91 by 7.", "7 × 13 = 91."], "Yes, 91 is divisible by 7."),
    _think("What is the perimeter of a square with side 9 cm?", ["Perimeter = 4 × side.", "4 × 9 = 36 cm."], "The perimeter is 36 cm."),
    _think("If you save $15 per week, how much in 8 weeks?", ["Multiply weekly savings by weeks.", "15 × 8 = 120."], "You would save $120."),
    _think("What comes next: 5, 10, 15, 20?", ["Add 5 each time.", "Next term = 25."], "The next number is 25."),
    _think("How many hours are in 3 days?", ["1 day = 24 hours.", "3 × 24 = 72 hours."], "There are 72 hours in 3 days."),
    _think("If 4 pencils cost $2, how much for 10 pencils?", ["Unit price = 2 ÷ 4 = $0.50.", "10 × 0.50 = $5."], "10 pencils cost $5."),
    _think("What is the greatest common factor of 18 and 24?", ["Factors of 18: 1,2,3,6,9,18.", "Largest shared factor is 6."], "The GCF is 6."),
    _think("A triangle has angles 45° and 90°. What is the third angle?", ["Triangle angles sum to 180°.", "180 - 45 - 90 = 45°."], "The third angle is 45°."),
    _think("If you drive 180 miles at 60 mph, how long does it take?", ["Time = distance ÷ speed.", "180 ÷ 60 = 3 hours."], "The trip takes 3 hours."),
    _think("What is 1/4 plus 1/2?", ["Convert to common denominator: 1/4 + 2/4.", "Sum = 3/4."], "1/4 plus 1/2 equals 3/4."),
    _think("How many weeks are in 84 days?", ["1 week = 7 days.", "84 ÷ 7 = 12 weeks."], "84 days is 12 weeks."),
    _think("If a number is doubled and then 6 is added, the result is 26. What is the number?", ["Let n be the number: 2n + 6 = 26.", "2n = 20, so n = 10."], "The number is 10."),
    _think("What is 7 factorial divided by 6 factorial?", ["7! / 6! = 7.", "Because 7! = 7 × 6!."], "The result is 7."),
    _think("A bag has 3 red and 7 blue marbles. What fraction are red?", ["Total marbles = 10.", "Red fraction = 3/10."], "3/10 of the marbles are red."),
    _think("Convert 2500 milliliters to liters.", ["1000 mL = 1 L.", "2500 ÷ 1000 = 2.5 liters."], "2500 milliliters is 2.5 liters."),
    _think("What is the square root of 81?", ["Find n where n × n = 81.", "9 × 9 = 81."], "The square root of 81 is 9."),
    _think("If rent is $1200 split equally among 4 roommates, how much each?", ["Divide total by 4.", "1200 ÷ 4 = 300."], "Each roommate pays $300."),
    _think("What is 11 mod 4?", ["11 ÷ 4 = 2 remainder 3.", "The remainder is 3."], "11 mod 4 is 3."),
    _think("A store marks up cost by 50%. If cost is $40, what is the price?", ["Markup = 0.5 × 40 = 20.", "Price = 40 + 20 = 60."], "The selling price is $60."),
    _think("How many bytes are in 2 kilobytes?", ["1 KB = 1024 bytes.", "2 × 1024 = 2048 bytes."], "2 kilobytes is 2048 bytes."),
    _think("If all cats are mammals and all mammals breathe air, do cats breathe air?", ["Cats are mammals.", "Mammals breathe air, so cats breathe air."], "Yes, cats breathe air."),
    _think("What is 100 minus 37?", ["Subtract 37 from 100.", "100 - 37 = 63."], "100 minus 37 is 63."),
]


def get_cot_pairs(repeat_with_shuffle: int = 1, seed: int = 7) -> List[Dict[str, str]]:
    import random

    rng = random.Random(seed)
    out: List[Dict[str, str]] = []
    for _ in range(max(1, repeat_with_shuffle)):
        block = list(CORE_COT_PAIRS)
        rng.shuffle(block)
        out.extend(block)
    return out


def load_custom_cot_jsonl(path: str) -> List[Dict[str, str]]:
    pairs: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            prompt = (row.get("prompt") or row.get("user") or "").strip()
            reasoning = (row.get("reasoning") or "").strip()
            answer = (row.get("answer") or "").strip()
            if not prompt or not reasoning or not answer:
                continue
            pairs.append({
                "user": prompt,
                "bot": f"{THINK_OPEN}\n{reasoning}\n{THINK_CLOSE}\n{answer}",
            })
    return pairs


def is_decoder_only(model) -> bool:
    if model is None:
        return False
    if hasattr(model, "hidden_dim") and not hasattr(model, "decoder_only"):
        return False
    return bool(getattr(model, "decoder_only", False))


def split_reasoning(raw_text: str):
    if THINK_CLOSE not in raw_text:
        return None, raw_text.strip()
    reasoning, _, answer = raw_text.partition(THINK_CLOSE)
    reasoning = reasoning.strip()
    if reasoning.lower().startswith(THINK_OPEN.lower()):
        reasoning = reasoning[len(THINK_OPEN):].strip()
    return (reasoning or None), answer.strip()


def merge_into_pretrain(state: Dict[str, Any], pairs: List[Dict[str, str]]) -> None:
    existing = list(state.get("pretrain_data") or [])
    old_cot = state.get("cot_data") or []
    if old_cot:
        old_keys = {(p.get("user"), p.get("bot")) for p in old_cot}
        existing = [p for p in existing if (p.get("user"), p.get("bot")) not in old_keys]
    new_keys = {(p.get("user"), p.get("bot")) for p in pairs}
    existing = [p for p in existing if (p.get("user"), p.get("bot")) not in new_keys]
    state["pretrain_data"] = existing + list(pairs)


def generate_with_reasoning(
    model,
    tokenizer,
    message: str,
    temperature: float = 0.8,
    self_check: bool = True,
    max_length: int = 120,
) -> Dict[str, Any]:
    """Two-pass reasoning: draft thinking + answer, then self-verification."""
    pass1_prompt = f"{message} {THINK_OPEN}"
    raw1 = model.generate(pass1_prompt, tokenizer, max_length=max_length, temperature=temperature)
    reasoning1, answer1 = split_reasoning(THINK_OPEN + raw1)

    if not self_check:
        return {
            "reasoning_pass1": reasoning1,
            "reasoning_final": reasoning1,
            "answer": answer1,
            "self_checked": False,
        }

    pass2_prompt = (
        f"{message} {THINK_OPEN} {reasoning1 or ''} {THINK_CLOSE} {answer1} "
        f"Let me double-check this. {THINK_OPEN}"
    )
    raw2 = model.generate(pass2_prompt, tokenizer, max_length=max_length, temperature=temperature)
    reasoning2, answer2 = split_reasoning(THINK_OPEN + raw2)

    return {
        "reasoning_pass1": reasoning1,
        "reasoning_final": reasoning2 or reasoning1,
        "answer": answer2 or answer1,
        "self_checked": True,
    }


def reasoning_handler_code(model_filename: str = "model.bin", tokenizer_filename: str = "tokenizer.bin") -> str:
    return f'''"""VnexAI reasoning handler — run CoT inference on exported .bin models.

Place this file next to:
  - {model_filename}
  - {tokenizer_filename}
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

MODEL_PATH = os.environ.get("VNEXAI_MODEL_PATH", "{model_filename}")
TOKENIZER_PATH = os.environ.get("VNEXAI_TOKENIZER_PATH", "{tokenizer_filename}")
THINK_OPEN = "{THINK_OPEN}"
THINK_CLOSE = "{THINK_CLOSE}"


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
            raise FileNotFoundError(f"Model not found: {{model_path}}")
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer not found: {{tokenizer_path}}")

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
        prompt1 = f"{{message}} {{THINK_OPEN}}"
        raw1 = self.model.generate(prompt1, self.tokenizer, max_length=max_length, temperature=temperature)
        reasoning1, answer1 = split_reasoning(THINK_OPEN + raw1)

        if not self_check:
            return {{"reasoning": reasoning1, "answer": answer1, "self_checked": False}}

        prompt2 = (
            f"{{message}} {{THINK_OPEN}} {{reasoning1 or ''}} {{THINK_CLOSE}} {{answer1}} "
            f"Let me double-check this. {{THINK_OPEN}}"
        )
        raw2 = self.model.generate(prompt2, self.tokenizer, max_length=max_length, temperature=temperature)
        reasoning2, answer2 = split_reasoning(THINK_OPEN + raw2)
        return {{
            "reasoning": reasoning2 or reasoning1,
            "answer": answer2 or answer1,
            "self_checked": True,
        }}


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
'''


def export_readme(model_name: str) -> str:
    return f"""# {model_name} — VnexAI export bundle

This package contains a trained VnexAI model with built-in chain-of-thought reasoning support.

## Files

- `{model_name}.bin` — trained model weights
- `{model_name}_tokenizer.bin` — vocabulary / tokenizer
- `reasoning_handler.py` — CoT inference script for agents
- `transformer_model.py` — model runtime (copy from VnexAI)
- `chatbot_tokenizer.py` — tokenizer runtime (copy from VnexAI)

## Quick start

```bash
pip install numpy
python reasoning_handler.py "What is 14 plus 27?"
```

## Agent integration

```python
from reasoning_handler import ReasoningEngine

engine = ReasoningEngine("{model_name}.bin", "{model_name}_tokenizer.bin")
result = engine.ask("Explain your reasoning step by step for 8 × 6")
print(result["reasoning"])
print(result["answer"])
```

## How thinking works

The handler forces the model to open a `{THINK_OPEN}` block, generates step-by-step reasoning,
then runs a second self-check pass before returning the final answer — the same loop used inside
the VnexAI training panel.
"""


def _write_dataset_file() -> None:
    pairs_json = json.dumps(CORE_COT_PAIRS)
    content = (
        '"""Auto-generated from reasoning.py — edit CORE_COT_PAIRS there instead."""\n'
        "import json\n\n"
        f"_PAIRS_JSON = {pairs_json!r}\n\n"
        "def get_cot_pairs():\n"
        "    return json.loads(_PAIRS_JSON)\n"
    )
    with open(DATASET_FILE, "w", encoding="utf-8") as f:
        f.write(content)


def _write_handler_file() -> None:
    with open(HANDLER_FILE, "w", encoding="utf-8") as f:
        f.write(reasoning_handler_code())


def startup(state: Dict[str, Any]) -> None:
    state.setdefault("cot_reasoning_enabled", False)
    state.setdefault("training_mode", "scratch")

    try:
        _write_dataset_file()
    except OSError:
        pass
    try:
        _write_handler_file()
    except OSError:
        pass

    if os.path.exists(COT_STATE_FILE):
        try:
            with open(COT_STATE_FILE, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, list) and loaded:
                state["cot_data"] = loaded
                merge_into_pretrain(state, loaded)
        except (json.JSONDecodeError, OSError):
            pass


def decoder_only_error() -> str:
    return (
        "CoT reasoning requires a Decoder-Only Transformer model. "
        "Create or load a model with architecture 'transformer_decoder'."
    )


def register_routes(app, state: Dict[str, Any], _training_state: Dict[str, Any]) -> None:
    from fastapi import File, HTTPException, Request, UploadFile
    from fastapi.responses import StreamingResponse
    import io

    @app.get("/api/reasoning/status")
    @app.get("/api/plugins/cot-reasoning/status")
    async def reasoning_status():
        model = state.get("model")
        data = state.get("cot_data")
        return {
            "loaded": data is not None,
            "count": len(data) if data else 0,
            "reasoning_enabled": bool(state.get("cot_reasoning_enabled", False)),
            "decoder_only": is_decoder_only(model) if model else None,
            "requires_decoder_only": True,
            "dataset_file_generated": os.path.exists(DATASET_FILE),
            "handler_file_generated": os.path.exists(HANDLER_FILE),
        }

    @app.post("/api/reasoning/load-builtin")
    @app.post("/api/plugins/cot-reasoning/load-builtin")
    async def reasoning_load_builtin(request: Request):
        raw_body = await request.body()
        body = json.loads(raw_body) if raw_body else {}
        repeat = int(body.get("repeat_with_shuffle", 1))
        pairs = get_cot_pairs(repeat_with_shuffle=repeat)
        merge_into_pretrain(state, pairs)
        state["cot_data"] = pairs
        with open(COT_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(pairs, f, ensure_ascii=False)
        return {"count": len(pairs), "message": f"Loaded {len(pairs)} CoT pairs into pre-train data."}

    @app.post("/api/reasoning/upload")
    @app.post("/api/plugins/cot-reasoning/upload")
    async def reasoning_upload(file: UploadFile = File(...)):
        tmp_path = os.path.join(PROJECT_ROOT, "_cot_upload_tmp.jsonl")
        contents = await file.read()
        with open(tmp_path, "wb") as f:
            f.write(contents)
        try:
            pairs = load_custom_cot_jsonl(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        if not pairs:
            raise HTTPException(
                status_code=400,
                detail="No valid rows. Each JSONL line needs prompt, reasoning, and answer.",
            )
        existing = state.get("cot_data") or []
        combined = existing + pairs
        merge_into_pretrain(state, pairs)
        state["cot_data"] = combined
        with open(COT_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(combined, f, ensure_ascii=False)
        return {"count": len(pairs), "total": len(combined), "message": f"Added {len(pairs)} custom CoT pairs."}

    @app.delete("/api/reasoning")
    @app.delete("/api/plugins/cot-reasoning")
    async def reasoning_delete():
        old_cot = state.get("cot_data") or []
        if old_cot and state.get("pretrain_data"):
            old_keys = {(p.get("user"), p.get("bot")) for p in old_cot}
            state["pretrain_data"] = [
                p for p in state["pretrain_data"]
                if (p.get("user"), p.get("bot")) not in old_keys
            ] or None
        state["cot_data"] = None
        if os.path.exists(COT_STATE_FILE):
            os.remove(COT_STATE_FILE)
        return {"message": "CoT data deleted."}

    @app.get("/api/reasoning/toggle")
    @app.get("/api/plugins/cot-reasoning/toggle")
    async def reasoning_get_toggle():
        return {"reasoning_enabled": bool(state.get("cot_reasoning_enabled", False))}

    @app.post("/api/reasoning/toggle")
    @app.post("/api/plugins/cot-reasoning/toggle")
    async def reasoning_set_toggle(request: Request):
        body = await request.json()
        enabled = bool(body.get("enabled", False))
        if enabled:
            model = state.get("model")
            if model is not None and not is_decoder_only(model):
                raise HTTPException(status_code=400, detail=decoder_only_error())
        state["cot_reasoning_enabled"] = enabled
        return {"reasoning_enabled": enabled}

    @app.post("/api/reasoning/export/prepare")
    @app.post("/api/plugins/cot-reasoning/export/prepare")
    async def reasoning_export_prepare(request: Request):
        body = await request.json()
        model_name = body.get("model_name", "vnexai_model")
        model = state.get("model")
        if not model:
            raise HTTPException(status_code=400, detail="No model.")
        if not is_decoder_only(model):
            raise HTTPException(status_code=400, detail=decoder_only_error())
        handler_code = reasoning_handler_code(f"{model_name}.bin", f"{model_name}_tokenizer.bin")
        state["reasoning_handler_bytes"] = handler_code.encode("utf-8")
        return {"size_bytes": len(state["reasoning_handler_bytes"]), "ready": True}

    @app.get("/api/reasoning/export/download")
    @app.get("/api/plugins/cot-reasoning/export/download")
    async def reasoning_export_download():
        handler_bytes = state.get("reasoning_handler_bytes")
        if not handler_bytes:
            raise HTTPException(status_code=404, detail="Handler not prepared.")
        return StreamingResponse(
            io.BytesIO(handler_bytes),
            media_type="text/x-python",
            headers={"Content-Disposition": 'attachment; filename="reasoning_handler.py"'},
        )
