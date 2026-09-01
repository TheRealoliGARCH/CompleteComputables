"""Model-agnostic financial-economic primitives for Complete Computables.

This layer deliberately avoids embedding any specific stochastic process,
risk-neutral measure, frictionless-market assumption, or continuous-hedging
requirement.  It represents valuation states first and leaves model choice to
higher layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from complete_computables import (
    CompleteComputable,
    ComputableState,
    Indeterminate,
    Negative,
    NegativeInfinity,
    Null,
    Positive,
    PositiveInfinity,
    StateLike,
    Zero,
    coerce,
    information_rank,
    ratio,
)


class DecisionState(str, Enum):
    """Decision-safe interpretation of a valuation comparison."""

    UNAVAILABLE = "Unavailable"
    UNDECIDABLE = "Undecidable"
    FAIR = "Fair"
    UNDERPRICED = "Underpriced"
    OVERPRICED = "Overpriced"
    UNBOUNDED_UPSIDE = "UnboundedUpside"
    UNBOUNDED_DOWNSIDE = "UnboundedDownside"


@dataclass(frozen=True)
class SecurityValuation:
    """A security valuation represented by (cost, price).

    The paper defines every security valuation by a pair (C, P).  This class
    preserves each component as a Complete Computable rather than coercing the
    pair into a conventional real-valued price prematurely.
    """

    cost: CompleteComputable
    price: CompleteComputable

    @classmethod
    def from_values(cls, cost: StateLike, price: StateLike) -> "SecurityValuation":
        return cls(cost=coerce(cost), price=coerce(price))

    @property
    def is_observable(self) -> bool:
        return self.cost.state != Null and self.price.state != Null

    @property
    def is_decidable(self) -> bool:
        return self.is_observable and self.cost.state != Indeterminate and self.price.state != Indeterminate

    @property
    def information_level(self) -> int:
        """Return the weaker information level of the two valuation components."""
        return min(information_rank(self.cost), information_rank(self.price))

    def price_to_cost(self) -> CompleteComputable:
        """Return P/C without imposing any stochastic pricing model."""
        return ratio(self.price, self.cost)

    def classify(self) -> DecisionState:
        """Classify the valuation only when the representation supports it.

        Finite numeric values are compared directly.  Distinguished states are
        never silently converted to arbitrary finite numbers.
        """
        if not self.is_observable:
            return DecisionState.UNAVAILABLE
        if not self.is_decidable:
            return DecisionState.UNDECIDABLE

        c, p = self.cost, self.price

        if p.state == PositiveInfinity:
            return DecisionState.UNBOUNDED_UPSIDE
        if p.state == NegativeInfinity:
            return DecisionState.UNBOUNDED_DOWNSIDE
        if c.state == PositiveInfinity:
            return DecisionState.OVERPRICED
        if c.state == NegativeInfinity:
            return DecisionState.UNDERPRICED

        if c.is_finite and p.is_finite:
            if c.value is None or p.value is None:
                return DecisionState.UNDECIDABLE
            if p.value > c.value:
                return DecisionState.UNDERPRICED
            if p.value < c.value:
                return DecisionState.OVERPRICED
            return DecisionState.FAIR

        return DecisionState.UNDECIDABLE


@dataclass(frozen=True)
class ValuationObservation:
    """One empirical observation used by a valuation model or decision rule."""

    name: str
    value: CompleteComputable

    @classmethod
    def from_value(cls, name: str, value: StateLike) -> "ValuationObservation":
        if not name:
            raise ValueError("observation name must be non-empty")
        return cls(name=name, value=coerce(value))


@dataclass(frozen=True)
class ValuationEvidence:
    """A model-independent collection of observed valuation inputs.

    Evidence is deliberately separated from a stochastic-model specification.
    This allows empirical, structural, simulation-based, or learned models to
    consume the same information-preserving representation.
    """

    observations: tuple[ValuationObservation, ...]

    @classmethod
    def from_pairs(cls, observations: Iterable[tuple[str, StateLike]]) -> "ValuationEvidence":
        items = tuple(ValuationObservation.from_value(name, value) for name, value in observations)
        names = [item.name for item in items]
        if len(names) != len(set(names)):
            raise ValueError("observation names must be unique")
        return cls(items)

    def get(self, name: str) -> CompleteComputable:
        for item in self.observations:
            if item.name == name:
                return item.value
        raise KeyError(name)

    @property
    def information_floor(self) -> int:
        if not self.observations:
            return information_rank(Null)
        return min(information_rank(item.value) for item in self.observations)

    @property
    def has_null(self) -> bool:
        return any(item.value.state == Null for item in self.observations)

    @property
    def has_indeterminate(self) -> bool:
        return any(item.value.state == Indeterminate for item in self.observations)


__all__ = [
    "DecisionState",
    "SecurityValuation",
    "ValuationObservation",
    "ValuationEvidence",
]
