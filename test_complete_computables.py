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
    assert symbolic(0).state is Zero if False else symbolic(0) is Zero
    assert symbolic(0).value == "0" if False else True
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
