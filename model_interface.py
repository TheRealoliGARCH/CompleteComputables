"""Model interfaces for Complete Computables.

Models consume Complete Computable evidence and may impose their own
assumptions. Those assumptions are metadata at the model boundary, never
hidden requirements of the Complete Computables representation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Mapping

from complete_computables import CompleteComputable, StateLike, coerce
from financial_economics import SecurityValuation, ValuationEvidence


@dataclass(frozen=True)
class ModelAssumptions:
    """Explicit registry of assumptions that a valuation model may impose."""

    uses_risk_neutral_measure: bool = False
    assumes_continuous_paths: bool = False
    assumes_frictionless_market: bool = False
    assumes_constant_volatility: bool = False
    allows_jumps: bool = True
    allows_transaction_costs: bool = True
    allows_discrete_observation: bool = True
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        flags = (
            self.uses_risk_neutral_measure,
            self.assumes_continuous_paths,
            self.assumes_frictionless_market,
            self.assumes_constant_volatility,
            self.allows_jumps,
            self.allows_transaction_costs,
            self.allows_discrete_observation,
        )
        if any(not isinstance(value, bool) for value in flags):
            raise TypeError("model assumption flags must be booleans")
        if any(not isinstance(note, str) or not note.strip() for note in self.notes):
            raise ValueError("assumption notes must be non-empty strings")

    @property
    def restrictions(self) -> tuple[str, ...]:
        """Return restrictive assumptions declared by the model."""
        restrictions: list[str] = []
        if self.uses_risk_neutral_measure:
            restrictions.append("risk-neutral measure")
        if self.assumes_continuous_paths:
            restrictions.append("continuous paths")
        if self.assumes_frictionless_market:
            restrictions.append("frictionless market")
        if self.assumes_constant_volatility:
            restrictions.append("constant volatility")
        if not self.allows_jumps:
            restrictions.append("no jumps")
        if not self.allows_transaction_costs:
            restrictions.append("no transaction costs")
        if not self.allows_discrete_observation:
            restrictions.append("no discrete observation")
        return tuple(restrictions)


@dataclass(frozen=True)
class ModelDiagnostics:
    """Diagnostics for evidence coverage and explicit information loss."""

    evidence_information_floor: int
    valuation_information_level: int
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_result(
        cls, evidence: ValuationEvidence, valuation: SecurityValuation
    ) -> "ModelDiagnostics":
        warnings: list[str] = []
        if evidence.has_null:
            warnings.append("evidence contains Null state")
        if evidence.has_indeterminate:
            warnings.append("evidence contains Indeterminate state")
        if valuation.information_level < evidence.information_floor:
            warnings.append("valuation output loses information relative to evidence")
        return cls(
            evidence_information_floor=evidence.information_floor,
            valuation_information_level=valuation.information_level,
            warnings=tuple(warnings),
        )


@dataclass(frozen=True)
class ModelOutput:
    """A valuation result with evidence, assumptions, and diagnostics."""

    valuation: SecurityValuation
    evidence: ValuationEvidence
    model_name: str
    assumptions: ModelAssumptions = field(default_factory=ModelAssumptions)
    diagnostics: ModelDiagnostics | None = None

    def __post_init__(self) -> None:
        if not self.model_name.strip():
            raise ValueError("model_name must be non-empty")
        if self.diagnostics is None:
            object.__setattr__(
                self,
                "diagnostics",
                ModelDiagnostics.from_result(self.evidence, self.valuation),
            )


class ValuationModel(ABC):
    """Minimal interface for any Complete Computables valuation model."""

    name = "Unnamed valuation model"
    assumptions = ModelAssumptions()

    @abstractmethod
    def evaluate(self, evidence: ValuationEvidence) -> ModelOutput:
        """Evaluate evidence without changing Complete Computable semantics."""
        raise NotImplementedError


class DirectValuationModel(ValuationModel):
    """Reference model returning observed cost and price unchanged."""

    name = "Direct valuation"
    assumptions = ModelAssumptions()

    def __init__(self, cost_name: str = "cost", price_name: str = "price") -> None:
        if not cost_name.strip() or not price_name.strip():
            raise ValueError("cost_name and price_name must be non-empty")
        self.cost_name = cost_name
        self.price_name = price_name

    def evaluate(self, evidence: ValuationEvidence) -> ModelOutput:
        valuation = SecurityValuation.from_values(
            evidence.get(self.cost_name),
            evidence.get(self.price_name),
        )
        return ModelOutput(valuation, evidence, self.name, self.assumptions)


@dataclass(frozen=True)
class Scenario:
    """One model-free valuation scenario."""

    cost: CompleteComputable
    price: CompleteComputable
    label: str = "scenario"

    @classmethod
    def from_values(
        cls, cost: StateLike, price: StateLike, label: str = "scenario"
    ) -> "Scenario":
        if not label.strip():
            raise ValueError("scenario label must be non-empty")
        return cls(coerce(cost), coerce(price), label)

    def valuation(self) -> SecurityValuation:
        return SecurityValuation(self.cost, self.price)


def evaluate_scenarios(
    scenarios: Mapping[str, Scenario],
) -> dict[str, SecurityValuation]:
    """Evaluate scenarios independently without forced aggregation."""
    return {name: scenario.valuation() for name, scenario in scenarios.items()}


__all__ = [
    "ModelAssumptions",
    "ModelDiagnostics",
    "ModelOutput",
    "ValuationModel",
    "DirectValuationModel",
    "Scenario",
    "evaluate_scenarios",
]
