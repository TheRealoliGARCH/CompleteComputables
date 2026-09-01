import pytest

from complete_computables import (
    Indeterminate,
    NegativeInfinity,
    Null,
    PositiveInfinity,
    Zero,
)
from financial_economics import (
    DecisionState,
    SecurityValuation,
    ValuationEvidence,
    ValuationObservation,
)


def test_security_valuation_preserves_cost_and_price_states():
    valuation = SecurityValuation.from_values(None, 10)
    assert valuation.cost.state is Null
    assert valuation.price.value == pytest.approx(10.0)
    assert not valuation.is_observable
    assert valuation.classify() is DecisionState.UNAVAILABLE


def test_indeterminate_valuation_is_not_forced_to_a_number():
    valuation = SecurityValuation.from_values(Indeterminate, 10)
    assert valuation.is_observable
    assert not valuation.is_decidable
    assert valuation.classify() is DecisionState.UNDECIDABLE


def test_finite_valuation_classification():
    assert SecurityValuation.from_values(10, 12).classify() is DecisionState.UNDERPRICED
    assert SecurityValuation.from_values(12, 10).classify() is DecisionState.OVERPRICED
    assert SecurityValuation.from_values(10, 10).classify() is DecisionState.FAIR


def test_unbounded_price_states_remain_explicit():
    assert (
        SecurityValuation.from_values(10, PositiveInfinity).classify()
        is DecisionState.UNBOUNDED_UPSIDE
    )
    assert (
        SecurityValuation.from_values(10, NegativeInfinity).classify()
        is DecisionState.UNBOUNDED_DOWNSIDE
    )


def test_price_to_cost_uses_complete_computable_ratio():
    assert SecurityValuation.from_values(0, 0).price_to_cost().state is Indeterminate
    assert SecurityValuation.from_values(0, 5).price_to_cost().state is PositiveInfinity
    assert SecurityValuation.from_values(5, 0).price_to_cost().state is Zero


def test_observation_requires_name():
    with pytest.raises(ValueError):
        ValuationObservation.from_value("", 1)


def test_evidence_preserves_missing_and_indeterminate_inputs():
    evidence = ValuationEvidence.from_pairs(
        [
            ("spot", 100),
            ("liquidity", None),
            ("model_disagreement", Indeterminate),
        ]
    )
    assert evidence.has_null
    assert evidence.has_indeterminate
    assert evidence.get("liquidity").state is Null
    assert evidence.get("model_disagreement").state is Indeterminate
    assert evidence.information_floor == 0


def test_evidence_rejects_duplicate_names():
    with pytest.raises(ValueError):
        ValuationEvidence.from_pairs([("spot", 100), ("spot", 101)])


def test_empty_evidence_has_null_information_floor():
    evidence = ValuationEvidence.from_pairs([])
    assert evidence.information_floor == 0
