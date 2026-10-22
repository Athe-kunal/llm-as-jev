"""Text normalization for duplicate and benchmark-overlap checks.

Words are NFKC-normalized, case-folded and split on non-word characters.
Sentences are split after . ! ? and at line breaks.
"""

import hashlib
import re
import unicodedata
from collections.abc import Iterable

from llm_jev.dataloader import schema

NGRAM = 13
MIN_SENTENCE_WORDS = 6

_WORD = re.compile(r"\w+")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n+")
_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})


def words(text: str) -> list[str]:
    """The normalized words of a text."""
    text = unicodedata.normalize("NFKC", text).translate(_QUOTES).casefold()
    return _WORD.findall(text)


def sentences(text: str) -> Iterable[str]:
    """Normalized sentences of at least MIN_SENTENCE_WORDS words."""
    for sentence in _SENTENCE_END.split(text):
        sentence_words = words(sentence)
        if len(sentence_words) >= MIN_SENTENCE_WORDS:
            yield " ".join(sentence_words)


def ngrams(text: str) -> Iterable[str]:
    """Every run of NGRAM consecutive normalized words."""
    text_words = words(text)
    for i in range(len(text_words) - NGRAM + 1):
        yield " ".join(text_words[i:i + NGRAM])


def input_key(example: schema.Example) -> str:
    """Hashes an example's input (parts, question, options) for duplicate checks.

    Copies that differ only in case, spacing, punctuation or option order get the same key.
    """
    texts = [part.text for part in example.parts] + [example.question]
    texts += sorted(" ".join(words(option)) for option in example.options)
    return hashlib.sha256(
        "\x1f".join(" ".join(words(t)) for t in texts).encode()).hexdigest()
