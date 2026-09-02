import pytest

from complete_computables import Indeterminate, Null, Positive, PositiveInfinity
from financial_economics import DecisionState, SecurityValuation, ValuationEvidence
from model_interface import (
    DirectValuationModel,
    ModelAssumptions,
    ModelDiagnostics,
    Scenario,
    evaluate_scenarios,
)


def test_direct_model_preserves_observed_valuation():
    evidence = ValuationEvidence.from_pairs([("cost", 10), ("price", 12)])
    output = DirectValuationModel().evaluate(evidence)
    assert output.valuation.cost.value == pytest.approx(10)
    assert output.valuation.price.value == pytest.approx(12)
    assert output.valuation.classify() is DecisionState.UNDERPRICED
    assert output.diagnostics is not None
    assert output.diagnostics.warnings == ()


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


def test_assumptions_are_explicit_and_inspectable():
    assumptions = ModelAssumptions(
        uses_risk_neutral_measure=True,
        assumes_continuous_paths=True,
        assumes_frictionless_market=True,
        assumes_constant_volatility=True,
        allows_jumps=False,
        allows_transaction_costs=False,
    )
    assert assumptions.restrictions == (
        "risk-neutral measure",
        "continuous paths",
        "frictionless market",
        "constant volatility",
        "no jumps",
        "no transaction costs",
    )


def test_default_assumptions_do_not_impose_bsm_style_restrictions():
    assumptions = DirectValuationModel.assumptions
    assert assumptions.restrictions == ()
    assert assumptions.allows_jumps
    assert assumptions.allows_transaction_costs
    assert assumptions.allows_discrete_observation


def test_diagnostics_surface_null_and_indeterminate_evidence():
    evidence = ValuationEvidence.from_pairs([("cost", None), ("price", Indeterminate)])
    valuation = SecurityValuation.from_values(None, Indeterminate)
    diagnostics = ModelDiagnostics.from_result(evidence, valuation)
    assert diagnostics.evidence_information_floor == 0
    assert diagnostics.valuation_information_level == 0
    assert diagnostics.warnings == (
        "evidence contains Null state",
        "evidence contains Indeterminate state",
    )


def test_model_output_automatically_builds_diagnostics():
    evidence = ValuationEvidence.from_pairs([("cost", 10), ("price", Positive)])
    output = DirectValuationModel().evaluate(evidence)
    assert output.diagnostics is not None
    assert output.diagnostics.evidence_information_floor == 2
    assert output.diagnostics.valuation_information_level == 2
