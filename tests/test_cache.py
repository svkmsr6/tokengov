"""Two-tier cache tests."""

from tokengov.cache import TwoTierCache, hashing_embedding


def test_exact_hit():
    cache = TwoTierCache()
    cache.set("What is 2+2?", "m", "4")
    assert cache.get("What is 2+2?", "m") == "4"
    assert cache.stats.exact_hits == 1
    assert cache.stats.hit_rate == 1.0


def test_miss_on_unknown_prompt():
    cache = TwoTierCache()
    cache.set("What is 2+2?", "m", "4")
    assert cache.get("What is 3+3?", "m") is None
    assert cache.stats.misses == 1


def test_semantic_hit_on_near_duplicate():
    cache = TwoTierCache(similarity_threshold=0.7)
    cache.set("Summarize this paragraph in one sentence.", "m", "SUMMARY")
    result = cache.get("Summarize this paragraph in a single sentence.", "m")
    assert result == "SUMMARY"
    assert cache.stats.semantic_hits == 1
    # Promotion: the new phrasing is now an exact hit.
    assert cache.get("Summarize this paragraph in a single sentence.", "m") == "SUMMARY"
    assert cache.stats.exact_hits == 1


def test_semantic_miss_on_unrelated_prompt():
    cache = TwoTierCache(similarity_threshold=0.9)
    cache.set("Summarize this paragraph in one sentence.", "m", "SUMMARY")
    assert cache.get("Explain Kubernetes pod scheduling to a beginner", "m") is None
    assert cache.stats.misses == 1


def test_no_cross_model_contamination():
    cache = TwoTierCache(similarity_threshold=0.0)  # any similarity would match
    cache.set("same exact prompt text", "cheap", "A")
    assert cache.get("same exact prompt text", "frontier") is None


def test_ttl_expiry(monkeypatch):
    # Deterministic clock: no real sleeping, no CI timing flakes.
    now = [1000.0]

    def fake_monotonic() -> float:
        return now[0]

    monkeypatch.setattr("time.monotonic", fake_monotonic)
    cache = TwoTierCache(ttl_seconds=0.05)
    cache.set("prompt", "m", "answer")
    now[0] += 0.01
    assert cache.get("prompt", "m") == "answer"
    now[0] += 0.06  # past the 0.05s TTL
    assert cache.get("prompt", "m") is None
    assert cache.stats.misses == 1


def test_lru_eviction():
    cache = TwoTierCache(max_entries=2)
    cache.set("p1", "m", "r1")
    cache.set("p2", "m", "r2")
    cache.set("p3", "m", "r3")  # evicts p1
    assert cache.get("p1", "m") is None
    assert cache.get("p2", "m") == "r2"
    assert cache.stats.evictions >= 1


def test_clear_resets_stats():
    cache = TwoTierCache()
    cache.set("p", "m", "r")
    cache.get("p", "m")
    cache.clear()
    assert len(cache) == 0
    assert cache.stats.hits == 0 and cache.stats.misses == 0


def test_hashing_embedding_properties():
    a = hashing_embedding("the quick brown fox")
    b = hashing_embedding("the quick brown fox")
    c = hashing_embedding("completely different words entirely")
    assert a == b  # deterministic
    assert abs(sum(x * x for x in a) - 1.0) < 1e-9  # unit norm
    from tokengov.cache import cosine

    assert cosine(a, hashing_embedding(a and "the quick brown fox")) == 1.0
    assert cosine(a, c) < 0.9


def test_custom_embed_fn():
    seen = []

    def embed(text):
        seen.append(text)
        return (1.0, 0.0)

    cache = TwoTierCache(embed_fn=embed)
    cache.set("hello", "m", "world")
    cache.get("hello there", "m")
    assert seen  # custom embedder was used
