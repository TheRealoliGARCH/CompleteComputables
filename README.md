# Complete Computables

Implementation of the seven-state **Complete Computables** framework described in *The Complete Computables as the Minimal System for Financial Economics*.

## State space

$$
\mathcal C = \{\mathrm{Null},\mathrm{Indeterminate},0,+,-,+\infty,-\infty\}.
$$

The implementation keeps the distinguished states explicit rather than collapsing them into ordinary IEEE floating-point values.

## Core operations

- `ratio(a, b)` implements the specified ratio decision rules.
- `multiply(a, b)` provides sign-preserving multiplication over the extended state space.
- `CompleteComputable.from_value(x)` classifies real numeric inputs.
- `symbolic(x)` projects to the five Symbolic Computables.
- `naive(x)` projects to the four Naive Computables.
- `information_rank(x)` exposes the hierarchy level.
- `at_least_as_informative(a, b)` implements the information comparison while retaining incomparability among `0`, `+`, and `-`.

## Ratio examples

```python
from complete_computables import ratio

ratio(None, 5)   # Null
ratio(0, 0)      # Indeterminate
ratio(2, 0)      # +Infinity
ratio(-2, 0)     # -Infinity
ratio(6, 3)      # + with value 2.0
```

## Information hierarchy

The paper distinguishes seven Complete Computables from the lower-information Symbolic and Naive systems. The implementation therefore treats the finite states `0`, `+`, and `-` as distinct and does not impose an artificial ordering between them.

## Validation

`test_complete_computables.py` covers the state space, ratio rules, finite-value preservation, constructor invariants, projections, extended-state arithmetic, and information ordering.

GitHub Actions validates pushes and pull requests to `main`. If the account-level Actions spending limit prevents runner allocation, CI remains an infrastructure constraint rather than a test result.
