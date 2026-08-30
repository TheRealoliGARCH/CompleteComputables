# Complete Computables

Implementation of the seven-state **Complete Computables** framework described in *The Complete Computables as the Minimal System for Financial Economics*.

The state space is

$$
\mathcal C = \{\mathrm{Null},\mathrm{Indeterminate},0,+,-,+\infty,-\infty\}.
$$

## Core behavior

The `ratio()` function implements the paper's ratio decision rules:

- Null in either argument returns `Null`.
- `0 / 0` returns `Indeterminate`.
- A nonzero finite numerator divided by zero returns signed infinity.
- Ordinary finite ratios preserve their numerical value and sign.

The module also exposes the five-state Symbolic Computables and four-state Naive Computables as explicit projections.

## Usage

```python
from complete_computables import ratio

ratio(None, 5)   # Null
ratio(0, 0)      # Indeterminate
ratio(2, 0)      # +Infinity
ratio(-2, 0)     # -Infinity
ratio(6, 3)      # + with value 2.0
```

## Validation

The repository includes `test_complete_computables.py`, covering the seven-state space, the ratio table, finite-value preservation, constructor validation, and lower-information projections.

## Status

The implementation is continuously validated by GitHub Actions on pushes and pull requests to `main`.
