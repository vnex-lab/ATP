"""
Common English tokens for vocabulary padding (word-level tokenizer).

When your training text has few unique words, the tokenizer only keeps types seen
in that text — so vocab can sit around ~1k even if max_vocab is 10k. Padding adds
frequent English words (from bundled data/common_words_en.txt) so inference hits
<UNK> less often.

Word list derived from the public "google-10000-english" frequency-style corpus
(first20hours/google-10000-english on GitHub), filtered to alphabetic tokens length >= 2.
"""

from __future__ import annotations

import pathlib
from typing import List, Optional

_PADDING_CACHE: Optional[List[str]] = None


def get_padding_word_list() -> List[str]:
    global _PADDING_CACHE
    if _PADDING_CACHE is not None:
        return list(_PADDING_CACHE)

    root = pathlib.Path(__file__).resolve().parent
    path = root / "data" / "common_words_en.txt"
    words: List[str] = []
    if path.is_file():
        text = path.read_text(encoding="utf-8")
        seen: set[str] = set()
        for line in text.splitlines():
            w = line.strip().lower()
            if len(w) < 2 or not w.isalpha() or w in seen:
                continue
            seen.add(w)
            words.append(w)
    _PADDING_CACHE = words
    return list(_PADDING_CACHE)
