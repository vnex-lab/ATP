"""Built-in Chain-of-Thought (CoT) dataset for the cot_reasoning plugin.

Every pair here teaches the model to *generate* its own reasoning before
answering, in the form:

    <think>
    step 1 ...
    step 2 ...
    </think>
    final answer

This is plain training data -- no fine-tuning step involved. These pairs get
merged into state["training_data"] / state["pretrain_data"] the same way
builtin_pretrain_dataset.py pairs do, so the existing from-scratch training
loop in api.py (run_training_thread) learns the <think>/</think> pattern as
part of normal next-token prediction.

NOTE: decoder_only models only. In decoder_only mode the user text and bot
text live in one continuous token stream (see api.py:_encode_pairs), which is
what lets the model actually *generate* the reasoning step itself instead of
having it handed to it. Encoder-decoder (RNN) models can still train on this
data but won't get the "thinks before answering" behavior since user/bot are
encoded as separate sequences.

Keep entries varied (many phrasings of the same underlying problem type) so
the model generalises the *pattern* <think>...reasoning...</think>answer
rather than memorising exact strings -- same philosophy as builtin_sft_dataset.py.
"""

from __future__ import annotations
import random
from typing import Dict, List


def _think(user: str, reasoning_steps: List[str], answer: str) -> Dict[str, str]:
    reasoning = "\n".join(f"Step {i+1}: {s}" for i, s in enumerate(reasoning_steps))
    bot = f"<think>\n{reasoning}\n</think>\n{answer}"
    return {"user": user.strip(), "bot": bot}


# -- Core CoT pairs -----------------------------------------------------------
# Covers: arithmetic, simple word problems, comparisons, unit conversion,
# basic logic. Each shows the full step -> step -> answer shape so the model
# learns to imitate the *format*, which is the realistic ceiling for a
# from-scratch model trained on consumer hardware.

_CORE_COT: List[Dict[str, str]] = [
    _think(
        "What is 14 plus 27?",
        ["Identify the two numbers: 14 and 27.", "Add them together: 14 + 27 = 41."],
        "14 plus 27 is 41.",
    ),
    _think(
        "If I have 3 apples and buy 5 more, how many do I have?",
        ["Start with 3 apples.", "Add the 5 new apples: 3 + 5 = 8."],
        "You'd have 8 apples.",
    ),
    _think(
        "A train travels 60 miles in 1.5 hours. What is its speed?",
        ["Speed is distance divided by time.", "Distance = 60 miles, time = 1.5 hours.", "60 / 1.5 = 40."],
        "The train's speed is 40 miles per hour.",
    ),
    _think(
        "What is 9 times 7?",
        ["Multiply 9 by 7.", "9 * 7 = 63."],
        "9 times 7 is 63.",
    ),
    _think(
        "If a shirt costs $20 and is 25% off, what is the price?",
        ["25% of $20 is 0.25 * 20 = 5.", "Subtract the discount from the price: 20 - 5 = 15."],
        "The shirt costs $15 after the discount.",
    ),
    _think(
        "Which is bigger, 0.5 or 1/3?",
        ["Convert 1/3 to a decimal: 1/3 is about 0.333.", "Compare 0.5 and 0.333.", "0.5 is larger."],
        "0.5 is bigger than 1/3.",
    ),
    _think(
        "How many minutes are in 2.5 hours?",
        ["There are 60 minutes in 1 hour.", "Multiply: 2.5 * 60 = 150."],
        "There are 150 minutes in 2.5 hours.",
    ),
    _think(
        "If today is Wednesday, what day is it in 10 days?",
        ["10 days is 1 week (7 days) plus 3 extra days.", "One week from Wednesday is Wednesday again.", "3 more days after Wednesday is Saturday."],
        "In 10 days it will be Saturday.",
    ),
    _think(
        "I have $50 and spend $18 on lunch and $12 on a book. How much is left?",
        ["Total spent: 18 + 12 = 30.", "Subtract from starting amount: 50 - 30 = 20."],
        "You have $20 left.",
    ),
    _think(
        "What is the area of a rectangle 4m by 9m?",
        ["Area of a rectangle is length times width.", "4 * 9 = 36."],
        "The area is 36 square meters.",
    ),
    _think(
        "If a car uses 1 gallon every 25 miles, how many gallons for 100 miles?",
        ["Divide total distance by miles per gallon.", "100 / 25 = 4."],
        "The car would use 4 gallons.",
    ),
    _think(
        "Is 17 a prime number?",
        ["Check if 17 is divisible by any number other than 1 and itself.", "It is not divisible by 2, 3, 5 or anything up to its square root (about 4.1).", "No divisors found other than 1 and 17."],
        "Yes, 17 is a prime number.",
    ),
    _think(
        "What is half of 86?",
        ["Half means dividing by 2.", "86 / 2 = 43."],
        "Half of 86 is 43.",
    ),
    _think(
        "If 3 workers can build a wall in 12 days, how long would 6 workers take?",
        ["More workers means less time, so this is inverse proportion.", "3 workers times 12 days = 36 worker-days of total work needed.", "36 worker-days / 6 workers = 6 days."],
        "6 workers would take 6 days.",
    ),
    _think(
        "What's the next number in the sequence 2, 4, 8, 16?",
        ["Look at the relationship between consecutive numbers.", "Each number is double the previous one.", "16 * 2 = 32."],
        "The next number is 32.",
    ),
]


def get_cot_pairs(repeat_with_shuffle: int = 1, seed: int = 7) -> List[Dict[str, str]]:
    """Return the built-in CoT pairs.

    Args:
        repeat_with_shuffle: how many shuffled copies of the core set to
            return concatenated (use >1 to give the pattern more weight in
            a training run without writing more examples by hand).
        seed: RNG seed for reproducibility.
    """
    rng = random.Random(seed)
    out: List[Dict[str, str]] = []
    for _ in range(max(1, repeat_with_shuffle)):
        block = list(_CORE_COT)
        rng.shuffle(block)
        out.extend(block)
    return out


def load_custom_cot_jsonl(path: str) -> List[Dict[str, str]]:
    """Load user-supplied CoT data from a JSONL file.

    Each line must be JSON with fields: prompt (or user), reasoning, answer.
    Rows missing required fields are skipped. Returns {user, bot} pairs in
    the same <think>...</think>answer shape as the built-in set.
    """
    import json as _json

    pairs: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = _json.loads(line)
            except _json.JSONDecodeError:
                continue
            prompt = (row.get("prompt") or row.get("user") or "").strip()
            reasoning = (row.get("reasoning") or "").strip()
            answer = (row.get("answer") or "").strip()
            if not prompt or not reasoning or not answer:
                continue
            bot = f"<think>\n{reasoning}\n</think>\n{answer}"
            pairs.append({"user": prompt, "bot": bot})
    return pairs
