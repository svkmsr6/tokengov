"""Two-tier response cache: exact match + semantic similarity.

Tier 1 is a hash lookup on (model, prompt). Tier 2 compares hashing-vector
embeddings of the prompt against cached entries for the same model and
returns the closest answer when cosine similarity meets the threshold.

The embedding is a deterministic hash-trick vector (tokens hashed into a
fixed-dimension space, L2-normalized). No network, no model weights — which
keeps the cache fully offline and makes its behavior trivially testable.
Swap in a real embedding function via the ``embed_fn`` constructor argument.
"""

from __future__ import annotations

import hashlib
import math
import re
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

__all__ = ["CacheEntry", "CacheStats", "TwoTierCache", "hashing_embedding"]

_WORD = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _h(token: str, dim: int) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % dim


def hashing_embedding(text: str, dim: int = 256) -> tuple[float, ...]:
    """Deterministic hashing-trick embedding, L2-normalized.

    This is the default offline embedder. It is not semantically clever — it
    captures lexical overlap, which is exactly what a similarity cache wants
    for near-duplicate prompts (retries, reworded requests, template drift).
    """
    vec = [0.0] * dim
    for tok in _tokenize(text):
        vec[_h(tok, dim)] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return tuple(v / norm for v in vec)


def cosine(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """Cosine similarity of two equal-length vectors."""
    return sum(x * y for x, y in zip(a, b, strict=True))


@dataclass(slots=True)
class CacheEntry:
    prompt: str
    model: str
    response: Any
    embedding: tuple[float, ...]
    created_at: float


@dataclass(slots=True)
class CacheStats:
    hits: int = 0
    misses: int = 0
    exact_hits: int = 0
    semantic_hits: int = 0
    evictions: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


class TwoTierCache:
    """Exact-match + semantic-similarity cache with TTL and LRU eviction.

    Args:
        ttl_seconds: Entries older than this are treated as misses and
            evicted lazily on access.
        max_entries: LRU bound across both tiers combined.
        similarity_threshold: Minimum cosine similarity for a tier-2 hit.
        embed_fn: ``text -> vector`` used for tier-2 comparison. Defaults to
            :func:`hashing_embedding`.

    Example:
        >>> cache = TwoTierCache(ttl_seconds=300, max_entries=1000)
        >>> cache.set("What is 2+2?", "gpt-4o-mini", "4")
        >>> cache.get("What is 2+2?", "gpt-4o-mini")
        '4'
    """

    def __init__(
        self,
        ttl_seconds: float = 600.0,
        max_entries: int = 256,
        similarity_threshold: float = 0.95,
        embed_fn: Callable[[str], tuple[float, ...]] | None = None,
    ) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self.similarity_threshold = similarity_threshold
        self._embed = embed_fn or (lambda text: hashing_embedding(text))
        self._exact: OrderedDict[str, CacheEntry] = OrderedDict()
        self._semantic: OrderedDict[int, CacheEntry] = OrderedDict()
        self._seq = 0
        self.stats = CacheStats()

    # -- public API ---------------------------------------------------------

    def get(self, prompt: str, model: str) -> Any | None:
        """Return the cached response for ``(prompt, model)`` or ``None``.

        A tier-1 hit is free. A tier-2 (semantic) hit also promotes the entry
        to tier 1 under the exact new prompt, so the retry becomes free.
        """
        now = time.monotonic()
        key = self._exact_key(prompt, model)

        entry = self._exact.get(key)
        if entry is not None:
            if self._fresh(entry, now):
                self._exact.move_to_end(key)
                self.stats.hits += 1
                self.stats.exact_hits += 1
                return entry.response
            self._expire(self._exact, key)

        best_key, best_entry, best_sim = self._best_semantic(prompt, model, now)
        if best_entry is not None and best_sim >= self.similarity_threshold:
            self._semantic.move_to_end(best_key)
            self.stats.hits += 1
            self.stats.semantic_hits += 1
            # Promote so the next identical request is an exact hit.
            self._exact[key] = CacheEntry(
                prompt=prompt,
                model=model,
                response=best_entry.response,
                embedding=best_entry.embedding,
                created_at=now,
            )
            self._exact.move_to_end(key)
            self._evict_if_needed()
            return best_entry.response

        self.stats.misses += 1
        return None

    def set(self, prompt: str, model: str, response: Any) -> None:
        """Cache ``response`` for ``(prompt, model)``."""
        now = time.monotonic()
        emb = self._embed(prompt)
        entry = CacheEntry(
            prompt=prompt, model=model, response=response, embedding=emb, created_at=now
        )
        key = self._exact_key(prompt, model)
        self._exact[key] = entry
        self._exact.move_to_end(key)
        self._semantic[self._seq] = entry
        self._seq += 1
        self._evict_if_needed()

    def clear(self) -> None:
        """Drop all entries and reset hit/miss counters."""
        self._exact.clear()
        self._semantic.clear()
        self.stats = CacheStats()

    # -- internals ----------------------------------------------------------

    @staticmethod
    def _exact_key(prompt: str, model: str) -> str:
        return hashlib.sha256(f"{model}\x00{prompt}".encode()).hexdigest()

    def _fresh(self, entry: CacheEntry, now: float) -> bool:
        return (now - entry.created_at) < self.ttl_seconds

    def _best_semantic(
        self, prompt: str, model: str, now: float
    ) -> tuple[int | None, CacheEntry | None, float]:
        emb = self._embed(prompt)
        best_key, best_entry, best_sim = None, None, 0.0
        for key, entry in list(self._semantic.items()):
            if entry.model != model:
                continue
            if not self._fresh(entry, now):
                self._expire(self._semantic, key)
                continue
            sim = cosine(emb, entry.embedding)
            if sim > best_sim:
                best_key, best_entry, best_sim = key, entry, sim
        return best_key, best_entry, best_sim

    def _evict_if_needed(self) -> None:
        # Each live prompt occupies one slot in each tier.
        while len(self._exact) > self.max_entries:
            self._exact.popitem(last=False)
            self.stats.evictions += 1
        while len(self._semantic) > self.max_entries:
            self._semantic.popitem(last=False)
            self.stats.evictions += 1

    @staticmethod
    def _expire(ordered: OrderedDict, key) -> None:
        ordered.pop(key, None)

    def __len__(self) -> int:
        return len(self._exact)
