"""Model interfaces for Complete Computables.

Models consume information represented by Complete Computables and return
valuation observations.  No particular stochastic process is required.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Mapping

from complete_computables import CompleteComputable, StateLike, coerce
from financial_economics import SecurityValuation, ValuationEvidence


@dataclass(frozen=True)
class ModelOutput:
    """A model result together with the evidence used to obtain it."""

    valuation: SecurityValuation
    evidence: ValuationEvidence
    model_name: str


class ValuationModel(ABC):
    """Minimal interface for any Complete Computable valuation model."""

    name = "Unnamed valuation model"

    @abstractmethod
    def evaluate(self, evidence: ValuationEvidence) -> ModelOutput:
        """Evaluate a security without changing the underlying state semantics."""
        raise NotImplementedError


class DirectValuationModel(ValuationModel):
    """Reference model that directly consumes observed cost and price.

    This is deliberately not a pricing theory.  It is a baseline proving that
    the interface can carry an empirical valuation without imposing dynamics.
    """

    name = "Direct valuation"

    def __init__(self, cost_name: str = "cost", price_name: str = "price") -> None:
        self.cost_name = cost_name
        self.price_name = price_name

    def evaluate(self, evidence: ValuationEvidence) -> ModelOutput:
        valuation = SecurityValuation.from_values(
            evidence.get(self.cost_name),
            evidence.get(self.price_name),
        )
        return ModelOutput(valuation, evidence, self.name)


@dataclass(frozen=True)
class Scenario:
    """One model-free valuation scenario."""

    cost: CompleteComputable
    price: CompleteComputable
    label: str = "scenario"

    @classmethod
    def from_values(cls, cost: StateLike, price: StateLike, label: str = "scenario") -> "Scenario":
        if not label:
            raise ValueError("scenario label must be non-empty")
        return cls(coerce(cost), coerce(price), label)

    def valuation(self) -> SecurityValuation:
        return SecurityValuation(self.cost, self.price)


def evaluate_scenarios(scenarios: Mapping[str, Scenario]) -> dict[str, SecurityValuation]:
    """Evaluate a collection of scenarios without aggregating away states."""
    return {name: scenario.valuation() for name, scenario in scenarios.items()}


__all__ = [
    "ModelOutput",
    "ValuationModel",
    "DirectValuationModel",
    "Scenario",
    "evaluate_scenarios",
]
