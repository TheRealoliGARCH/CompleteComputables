"""Complete Computables: a seven-state ratio valuation system.

The implementation follows the specification in
"The Complete Computables as the Minimal System for Financial Economics":

    C = {Null, Indeterminate, 0, +, -, +Infinity, -Infinity}

The module provides explicit state objects, ratio evaluation, symbolic and
naive projections, and a small validation surface suitable for use by
financial-economic models.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import inf
from numbers import Real
from typing import Any, Union


class ComputableState(str, Enum):
    """The seven Complete Computable states."""

    NULL = "Null"
    INDETERMINATE = "Indeterminate"
    ZERO = "0"
    POSITIVE = "+"
    NEGATIVE = "-"
    POSITIVE_INFINITY = "+Infinity"
    NEGATIVE_INFINITY = "-Infinity"


# Public aliases matching the terminology of the paper.
Null = ComputableState.NULL
Indeterminate = ComputableState.INDETERMINATE
Zero = ComputableState.ZERO
Positive = ComputableState.POSITIVE
Negative = ComputableState.NEGATIVE
PositiveInfinity = ComputableState.POSITIVE_INFINITY
NegativeInfinity = ComputableState.NEGATIVE_INFINITY


@dataclass(frozen=True)
class CompleteComputable:
    """A value represented by a Complete Computable state.

    For finite real values, ``value`` is retained so that the seven-state
    abstraction can still be embedded in ordinary arithmetic.  For the
    distinguished non-real outcomes, ``value`` is ``None``.
    """

    state: ComputableState
    value: float | None = None

    def __post_init__(self) -> None:
        if self.state in {
            ComputableState.NULL,
            ComputableState.INDETERMINATE,
            ComputableState.POSITIVE_INFINITY,
            ComputableState.NEGATIVE_INFINITY,
        }:
            if self.value is not None:
                raise ValueError(f"{self.state.value} cannot carry a finite value")
        elif self.state == ComputableState.ZERO:
            if self.value is not None and self.value != 0:
                raise ValueError("Zero state must have value 0")
        elif self.state == ComputableState.POSITIVE:
            if self.value is not None and self.value <= 0:
                raise ValueError("Positive state must have a positive value")
        elif self.state == ComputableState.NEGATIVE:
            if self.value is not None and self.value >= 0:
                raise ValueError("Negative state must have a negative value")

    @classmethod
    def from_value(cls, value: Any) -> "CompleteComputable":
        """Construct a Complete Computable from a supported numeric value.

        ``None`` maps to Null.  Naive IEEE ``nan`` is intentionally rejected:
        indeterminacy is represented explicitly by ``Indeterminate``.
        """
        if value is None:
            return cls(ComputableState.NULL)
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError("value must be a real number or None")
        value = float(value)
        if value != value:  # NaN
            raise ValueError("NaN is not a Complete Computable state")
        if value == inf:
            return cls(ComputableState.POSITIVE_INFINITY)
        if value == -inf:
            return cls(ComputableState.NEGATIVE_INFINITY)
        if value == 0:
            return cls(ComputableState.ZERO, 0.0)
        if value > 0:
            return cls(ComputableState.POSITIVE, value)
        return cls(ComputableState.NEGATIVE, value)

    @property
    def is_finite(self) -> bool:
        return self.state in {
            ComputableState.ZERO,
            ComputableState.POSITIVE,
            ComputableState.NEGATIVE,
        }

    def sign(self) -> ComputableState:
        """Return the finite sign state; reject non-finite/non-ordinary states."""
        if self.state not in {
            ComputableState.ZERO,
            ComputableState.POSITIVE,
            ComputableState.NEGATIVE,
        }:
            raise ValueError(f"{self.state.value} has no ordinary finite sign")
        return self.state

    def __str__(self) -> str:
        return self.state.value


StateLike = Union[CompleteComputable, ComputableState, Real, None]


def coerce(value: StateLike) -> CompleteComputable:
    """Normalize Python numeric values and Complete Computables."""
    if isinstance(value, CompleteComputable):
        return value
    if isinstance(value, ComputableState):
        return CompleteComputable(value)
    return CompleteComputable.from_value(value)


def ratio(numerator: StateLike, denominator: StateLike) -> CompleteComputable:
    """Evaluate the Complete Computable ratio ``numerator / denominator``.

    This implements the paper's decision structure:

    * Null in either argument -> Null
    * 0 / 0 -> Indeterminate
    * nonzero / 0 -> signed infinity
    * otherwise -> ordinary sign/finite ratio

    Explicit infinities are also handled so that the state space remains
    closed for computational use.
    """
    a = coerce(numerator)
    b = coerce(denominator)

    if a.state == ComputableState.NULL or b.state == ComputableState.NULL:
        return CompleteComputable(ComputableState.NULL)

    if a.state == ComputableState.INDETERMINATE or b.state == ComputableState.INDETERMINATE:
        return CompleteComputable(ComputableState.INDETERMINATE)

    # Finite denominator equal to zero.
    if b.state == ComputableState.ZERO:
        if a.state == ComputableState.ZERO:
            return CompleteComputable(ComputableState.INDETERMINATE)
        if a.state in {ComputableState.POSITIVE, ComputableState.POSITIVE_INFINITY}:
            return CompleteComputable(ComputableState.POSITIVE_INFINITY)
        if a.state in {ComputableState.NEGATIVE, ComputableState.NEGATIVE_INFINITY}:
            return CompleteComputable(ComputableState.NEGATIVE_INFINITY)

    # Explicit infinity divided by infinity is left indeterminate.
    if a.state in {ComputableState.POSITIVE_INFINITY, ComputableState.NEGATIVE_INFINITY} and b.state in {
        ComputableState.POSITIVE_INFINITY,
        ComputableState.NEGATIVE_INFINITY,
    }:
        return CompleteComputable(ComputableState.INDETERMINATE)

    # Finite / finite can preserve the actual value.
    if a.is_finite and b.is_finite:
        assert a.value is not None and b.value is not None
        return CompleteComputable.from_value(a.value / b.value)

    # Finite / infinity -> zero with the sign determined by the quotient.
    if a.is_finite and b.state in {
        ComputableState.POSITIVE_INFINITY,
        ComputableState.NEGATIVE_INFINITY,
    }:
        assert a.value is not None
        if a.value == 0:
            return CompleteComputable(ComputableState.ZERO, 0.0)
        return CompleteComputable(ComputableState.ZERO, 0.0)

    # Infinity / finite nonzero -> signed infinity.
    if a.state in {ComputableState.POSITIVE_INFINITY, ComputableState.NEGATIVE_INFINITY} and b.is_finite:
        assert b.value is not None
        positive = (a.state == ComputableState.POSITIVE_INFINITY) == (b.value > 0)
        return CompleteComputable(
            ComputableState.POSITIVE_INFINITY if positive else ComputableState.NEGATIVE_INFINITY
        )

    raise ValueError(f"Unsupported Complete Computable ratio: {a.state.value}/{b.state.value}")


def symbolic(value: StateLike) -> ComputableState:
    """Project a Complete Computable to the five Symbolic Computables."""
    state = coerce(value).state
    if state in {ComputableState.POSITIVE_INFINITY, ComputableState.NEGATIVE_INFINITY}:
        raise ValueError("Symbolic Computables cannot preserve the sign of infinity")
    return state


def naive(value: StateLike) -> ComputableState:
    """Project a Complete Computable to the four Naive Computables."""
    state = coerce(value).state
    if state in {
        ComputableState.INDETERMINATE,
        ComputableState.POSITIVE_INFINITY,
        ComputableState.NEGATIVE_INFINITY,
    }:
        return ComputableState.NULL
    return state


COMPLETE_COMPUTABLES = tuple(ComputableState)
SYMBOLIC_COMPUTABLES = (
    ComputableState.NULL,
    ComputableState.INDETERMINATE,
    ComputableState.ZERO,
    ComputableState.POSITIVE,
    ComputableState.NEGATIVE,
)
NAIVE_COMPUTABLES = (
    ComputableState.NULL,
    ComputableState.ZERO,
    ComputableState.POSITIVE,
    ComputableState.NEGATIVE,
)


__all__ = [
    "ComputableState",
    "CompleteComputable",
    "COMPLETE_COMPUTABLES",
    "SYMBOLIC_COMPUTABLES",
    "NAIVE_COMPUTABLES",
    "Null",
    "Indeterminate",
    "Zero",
    "Positive",
    "Negative",
    "PositiveInfinity",
    "NegativeInfinity",
    "coerce",
    "ratio",
    "symbolic",
    "naive",
]
