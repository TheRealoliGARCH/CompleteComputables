import pytest

from complete_computables import (
    ComputableState,
    CompleteComputable,
    Indeterminate,
    Negative,
    NegativeInfinity,
    Null,
    Positive,
    PositiveInfinity,
    Zero,
    at_least_as_informative,
    information_comparable,
    information_rank,
    multiply,
    naive,
    ratio,
    symbolic,
)


def test_state_space_is_complete():
    assert set(ComputableState) == {
        Null,
        Indeterminate,
        Zero,
        Positive,
        Negative,
        PositiveInfinity,
        NegativeInfinity,
    }


def test_ratio_table_from_paper():
    assert ratio(None, None).state is Null
    assert ratio(None, 2).state is Null
    assert ratio(0, None).state is Null
    assert ratio(0, 0).state is Indeterminate
    assert ratio(2, 0).state is PositiveInfinity
    assert ratio(-2, 0).state is NegativeInfinity
    assert ratio(2, 3).state is Positive
    assert ratio(-2, 3).state is Negative
    assert ratio(-2, -3).state is Positive


def test_finite_values_are_preserved():
    result = ratio(6, 3)
    assert result.state is Positive
    assert result.value == pytest.approx(2.0)


def test_constructor_rejects_inconsistent_states():
    with pytest.raises(ValueError):
        CompleteComputable(Positive, -1)
    with pytest.raises(ValueError):
        CompleteComputable(Zero, 1)


def test_symbolic_and_naive_projections():
    assert symbolic(0) is Zero
    assert naive(Indeterminate) is Null
    assert naive(PositiveInfinity) is Null
    assert naive(NegativeInfinity) is Null
    assert symbolic(Positive) is Positive
    assert symbolic(Negative) is Negative


def test_symbolic_projection_rejects_infinity_information_loss():
    with pytest.raises(ValueError):
        symbolic(PositiveInfinity)
    with pytest.raises(ValueError):
        symbolic(NegativeInfinity)


def test_from_value_and_infinity_states():
    assert CompleteComputable.from_value(0).state is Zero
    assert CompleteComputable.from_value(2.5).state is Positive
    assert CompleteComputable.from_value(-2.5).state is Negative
    assert CompleteComputable.from_value(float("inf")).state is PositiveInfinity
    assert CompleteComputable.from_value(float("-inf")).state is NegativeInfinity
    assert CompleteComputable.from_value(None).state is Null
    with pytest.raises(ValueError):
        CompleteComputable.from_value(float("nan"))


def test_multiplication_preserves_sign_and_finite_value():
    result = multiply(-2, -3)
    assert result.state is Positive
    assert result.value == pytest.approx(6.0)
    assert multiply(2, -3).state is Negative
    assert multiply(0, 8).state is Zero


def test_multiplication_handles_extended_states():
    assert multiply(PositiveInfinity, 2).state is PositiveInfinity
    assert multiply(NegativeInfinity, 2).state is NegativeInfinity
    assert multiply(PositiveInfinity, -2).state is NegativeInfinity
    assert multiply(PositiveInfinity, PositiveInfinity).state is PositiveInfinity
    assert multiply(PositiveInfinity, NegativeInfinity).state is NegativeInfinity
    assert multiply(0, PositiveInfinity).state is Indeterminate


def test_extended_ratios():
    assert ratio(6, PositiveInfinity).state is Zero
    assert ratio(PositiveInfinity, 2).state is PositiveInfinity
    assert ratio(PositiveInfinity, -2).state is NegativeInfinity
    assert ratio(PositiveInfinity, PositiveInfinity).state is Indeterminate
    assert ratio(NegativeInfinity, NegativeInfinity).state is Indeterminate


def test_information_hierarchy_is_a_partial_order():
    assert information_rank(Null) == 0
    assert information_rank(Indeterminate) == 1
    assert information_rank(Zero) == 2
    assert information_rank(PositiveInfinity) == 3
    assert at_least_as_informative(Indeterminate, Null)
    assert at_least_as_informative(Positive, Indeterminate)
    assert at_least_as_informative(PositiveInfinity, Positive)
    assert at_least_as_informative(NegativeInfinity, Negative)
    assert not at_least_as_informative(Positive, Negative)
    assert not at_least_as_informative(Negative, Positive)
    assert not at_least_as_informative(PositiveInfinity, NegativeInfinity)
    assert not at_least_as_informative(NegativeInfinity, PositiveInfinity)
    assert not at_least_as_informative(Indeterminate, Positive)
    assert at_least_as_informative(PositiveInfinity, Null)


def test_information_comparability_distinguishes_incomparable_states():
    assert information_comparable(Null, Positive)
    assert information_comparable(Indeterminate, NegativeInfinity)
    assert information_comparable(Positive, PositiveInfinity)
    assert not information_comparable(Zero, Positive)
    assert not information_comparable(Positive, Negative)
    assert not information_comparable(PositiveInfinity, NegativeInfinity)
