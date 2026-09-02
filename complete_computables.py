"""Complete Computables: a seven-state computational system."""

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


Null = ComputableState.NULL
Indeterminate = ComputableState.INDETERMINATE
Zero = ComputableState.ZERO
Positive = ComputableState.POSITIVE
Negative = ComputableState.NEGATIVE
PositiveInfinity = ComputableState.POSITIVE_INFINITY
NegativeInfinity = ComputableState.NEGATIVE_INFINITY


@dataclass(frozen=True)
class CompleteComputable:
    """A Complete Computable state with an optional finite real value."""

    state: ComputableState
    value: float | None = None

    def __post_init__(self) -> None:
        if self.state in {Null, Indeterminate, PositiveInfinity, NegativeInfinity} and self.value is not None:
            raise ValueError(f"{self.state.value} cannot carry a finite value")
        if self.state == Zero and self.value not in (None, 0, 0.0):
            raise ValueError("Zero state must have value 0")
        if self.state == Positive and self.value is not None and self.value <= 0:
            raise ValueError("Positive state must have a positive value")
        if self.state == Negative and self.value is not None and self.value >= 0:
            raise ValueError("Negative state must have a negative value")

    @classmethod
    def from_value(cls, value: Any) -> "CompleteComputable":
        if value is None:
            return cls(Null)
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError("value must be a real number or None")
        value = float(value)
        if value != value:
            raise ValueError("NaN is represented by Indeterminate, not a numeric value")
        if value == inf:
            return cls(PositiveInfinity)
        if value == -inf:
            return cls(NegativeInfinity)
        if value == 0:
            return cls(Zero, 0.0)
        return cls(Positive if value > 0 else Negative, value)

    @property
    def is_finite(self) -> bool:
        return self.state in {Zero, Positive, Negative}

    def sign(self) -> ComputableState:
        if not self.is_finite:
            raise ValueError(f"{self.state.value} has no finite sign")
        return self.state

    def __str__(self) -> str:
        return self.state.value


StateLike = Union[CompleteComputable, ComputableState, Real, None]


def coerce(value: StateLike) -> CompleteComputable:
    if isinstance(value, CompleteComputable):
        return value
    if isinstance(value, ComputableState):
        return CompleteComputable(value)
    return CompleteComputable.from_value(value)


def _finite_ratio(a: CompleteComputable, b: CompleteComputable) -> CompleteComputable:
    assert a.value is not None and b.value is not None
    return CompleteComputable.from_value(a.value / b.value)


def ratio(numerator: StateLike, denominator: StateLike) -> CompleteComputable:
    """Evaluate a Complete Computable ratio using explicit state rules."""
    a, b = coerce(numerator), coerce(denominator)
    if a.state == Null or b.state == Null:
        return CompleteComputable(Null)
    if a.state == Indeterminate or b.state == Indeterminate:
        return CompleteComputable(Indeterminate)
    if b.state == Zero:
        if a.state == Zero:
            return CompleteComputable(Indeterminate)
        if a.state in {Positive, PositiveInfinity}:
            return CompleteComputable(PositiveInfinity)
        if a.state in {Negative, NegativeInfinity}:
            return CompleteComputable(NegativeInfinity)
    if a.is_finite and b.is_finite:
        return _finite_ratio(a, b)
    if a.is_finite and b.state in {PositiveInfinity, NegativeInfinity}:
        return CompleteComputable(Zero, 0.0)
    if a.state in {PositiveInfinity, NegativeInfinity} and b.state in {PositiveInfinity, NegativeInfinity}:
        return CompleteComputable(Indeterminate)
    if a.state in {PositiveInfinity, NegativeInfinity} and b.is_finite:
        assert b.value is not None
        positive = (a.state == PositiveInfinity) == (b.value > 0)
        return CompleteComputable(PositiveInfinity if positive else NegativeInfinity)
    raise ValueError(f"Unsupported ratio: {a.state.value}/{b.state.value}")


def multiply(left: StateLike, right: StateLike) -> CompleteComputable:
    """Multiply Complete Computables as a conservative arithmetic extension."""
    a, b = coerce(left), coerce(right)
    if a.state == Null or b.state == Null:
        return CompleteComputable(Null)
    if a.state == Indeterminate or b.state == Indeterminate:
        return CompleteComputable(Indeterminate)
    if a.is_finite and b.is_finite:
        assert a.value is not None and b.value is not None
        return CompleteComputable.from_value(a.value * b.value)
    if (a.state == Zero and b.state in {PositiveInfinity, NegativeInfinity}) or (b.state == Zero and a.state in {PositiveInfinity, NegativeInfinity}):
        return CompleteComputable(Indeterminate)
    if a.state in {PositiveInfinity, NegativeInfinity} and b.state in {PositiveInfinity, NegativeInfinity}:
        positive = (a.state == PositiveInfinity) == (b.state == PositiveInfinity)
        return CompleteComputable(PositiveInfinity if positive else NegativeInfinity)
    if a.state in {PositiveInfinity, NegativeInfinity} and b.is_finite:
        assert b.value is not None
        if b.value == 0:
            return CompleteComputable(Indeterminate)
        positive = (a.state == PositiveInfinity) == (b.value > 0)
        return CompleteComputable(PositiveInfinity if positive else NegativeInfinity)
    if b.state in {PositiveInfinity, NegativeInfinity} and a.is_finite:
        assert a.value is not None
        if a.value == 0:
            return CompleteComputable(Indeterminate)
        positive = (b.state == PositiveInfinity) == (a.value > 0)
        return CompleteComputable(PositiveInfinity if positive else NegativeInfinity)
    raise ValueError(f"Unsupported product: {a.state.value}*{b.state.value}")


def symbolic(value: StateLike) -> ComputableState:
    """Project to the five-state symbolic subset without hiding infinity."""
    state = coerce(value).state
    if state in {PositiveInfinity, NegativeInfinity}:
        raise ValueError("Symbolic projection is undefined for infinity states")
    return state


def naive(value: StateLike) -> ComputableState:
    """Project to the four-state naive subset."""
    state = coerce(value).state
    if state in {Indeterminate, PositiveInfinity, NegativeInfinity}:
        return Null
    return state


# Rank is deliberately only a coarse hierarchy level; it is not the order.
_INFORMATION_RANK = {
    Null: 0,
    Indeterminate: 1,
    Zero: 2,
    Positive: 2,
    Negative: 2,
    PositiveInfinity: 3,
    NegativeInfinity: 3,
}


def information_rank(value: StateLike) -> int:
    """Return the coarse level of a state in the information hierarchy."""
    return _INFORMATION_RANK[coerce(value).state]


def at_least_as_informative(left: StateLike, right: StateLike) -> bool:
    """Return whether ``left`` is at least as informative as ``right``.

    The relation is a partial order: 0, +, and - are mutually incomparable,
    and +Infinity and -Infinity are distinct maximal states.
    """
    a, b = coerce(left).state, coerce(right).state
    if a == b or b == Null:
        return True
    if a == Null:
        return False
    if b == Indeterminate:
        return a != Null
    if a == Indeterminate:
        return False
    if b in {Zero, Positive, Negative}:
        return a == b or a in {PositiveInfinity, NegativeInfinity}
    if b == PositiveInfinity:
        return a == PositiveInfinity
    if b == NegativeInfinity:
        return a == NegativeInfinity
    return False


def information_comparable(left: StateLike, right: StateLike) -> bool:
    """Return whether two states are comparable in the information order."""
    return at_least_as_informative(left, right) or at_least_as_informative(right, left)


COMPLETE_COMPUTABLES = tuple(ComputableState)
SYMBOLIC_COMPUTABLES = (Null, Indeterminate, Zero, Positive, Negative)
NAIVE_COMPUTABLES = (Null, Zero, Positive, Negative)


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
    "multiply",
    "symbolic",
    "naive",
    "information_rank",
    "at_least_as_informative",
    "information_comparable",
]
