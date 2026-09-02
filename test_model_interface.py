import pytest

from complete_computables import Indeterminate, Null, Positive, PositiveInfinity
from financial_economics import DecisionState, SecurityValuation, ValuationEvidence
from model_interface import DirectValuationModel, Scenario, evaluate_scenarios


def test_direct_model_preserves_observed_valuation():
    evidence = ValuationEvidence.from_pairs([("cost", 10), ("price", 12)])
    output = DirectValuationModel().evaluate(evidence)
    assert output.valuation.cost.value == pytest.approx(10)
    assert output.valuation.price.value == pytest.approx(12)
    assert output.valuation.classify() is DecisionState.UNDERPRICED


def test_missing_information_is_not_converted_to_a_number():
    valuation = SecurityValuation.from_values(None, 12)
    assert valuation.classify() is DecisionState.UNAVAILABLE


def test_indeterminate_information_is_preserved():
    valuation = SecurityValuation.from_values(10, Indeterminate)
    assert valuation.classify() is DecisionState.UNDECIDABLE


def test_extended_price_state_is_preserved():
    valuation = SecurityValuation.from_values(10, PositiveInfinity)
    assert valuation.classify() is DecisionState.UNBOUNDED_UPSIDE


def test_scenario_collection_does_not_force_aggregation():
    scenarios = {
        "known": Scenario.from_values(10, 12),
        "missing": Scenario.from_values(None, 12),
        "unknown": Scenario.from_values(10, Indeterminate),
    }
    result = evaluate_scenarios(scenarios)
    assert result["known"].classify() is DecisionState.UNDERPRICED
    assert result["missing"].classify() is DecisionState.UNAVAILABLE
    assert result["unknown"].classify() is DecisionState.UNDECIDABLE


def test_evidence_requires_unique_names():
    with pytest.raises(ValueError):
        ValuationEvidence.from_pairs([("x", 1), ("x", 2)])
