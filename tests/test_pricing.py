"""Pricing table and cost calculation tests."""

import pytest

from tokengov.pricing import (
    ModelPricing,
    calc_cost,
    count_tokens,
    get_pricing,
    register_pricing,
)


def test_calc_cost_simple():
    # gpt-4o-mini: $0.15/1M in, $0.60/1M out
    cost = calc_cost("gpt-4o-mini", 1_000_000, 1_000_000)
    assert cost == pytest.approx(0.75)


def test_calc_cost_scales_with_tokens():
    cost = calc_cost("gpt-4o", 100_000, 50_000)
    # 100k * $2.50/1M + 50k * $10/1M = 0.25 + 0.50
    assert cost == pytest.approx(0.75)


def test_cached_input_billed_at_reduced_rate():
    # gpt-4o: $2.50 fresh in, $1.25 cached in
    fresh = calc_cost("gpt-4o", 100_000, 0, cached_input_tokens=0)
    cached = calc_cost("gpt-4o", 100_000, 0, cached_input_tokens=100_000)
    assert fresh == pytest.approx(0.25)
    assert cached == pytest.approx(0.125)


def test_cached_never_exceeds_input():
    cost = calc_cost("gpt-4o", 10, 0, cached_input_tokens=999)
    assert cost >= 0.0


def test_unknown_model_raises():
    with pytest.raises(KeyError, match="No pricing registered"):
        calc_cost("does-not-exist", 10, 10)


def test_register_pricing_extends_table():
    register_pricing(
        ModelPricing(model="test-internal", input_per_million=1.0, output_per_million=2.0)
    )
    try:
        p = get_pricing("test-internal")
        assert p.output_per_million == 2.0
        assert calc_cost("test-internal", 1_000_000, 0) == pytest.approx(1.0)
    finally:
        from tokengov.pricing import PRICING

        PRICING.pop("test-internal", None)


def test_count_tokens_is_sane():
    assert count_tokens("hello world") > 0
    assert count_tokens("") == 0
    assert count_tokens("a b c") == 3
