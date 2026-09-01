# Complete Computables

Implementation of the seven-state **Complete Computables** framework described in *The Complete Computables as the Minimal System for Financial Economics*.

## State space

$$
\mathcal C = \{\mathrm{Null},\mathrm{Indeterminate},0,+,-,+\infty,-\infty\}.
$$

The implementation keeps the distinguished states explicit rather than collapsing them into ordinary IEEE floating-point values.

## Architecture

The repository separates representation from valuation models:

```text
Complete Computable states
        -> information-preserving evidence
        -> security valuation (cost, price)
        -> model-specific valuation logic
        -> decision systems
```

No stochastic process, probability measure, continuous-hedging assumption, or frictionless-market assumption is embedded in the representation layer.

## Core operations

- `ratio(a, b)` implements the specified ratio decision rules.
- `multiply(a, b)` provides sign-preserving multiplication over the extended state space.
- `CompleteComputable.from_value(x)` classifies real numeric inputs.
- `symbolic(x)` projects to the five Symbolic Computables.
- `naive(x)` projects to the four Naive Computables.
- `information_rank(x)` exposes the hierarchy level.
- `at_least_as_informative(a, b)` implements information comparison while retaining state distinctions.

## Financial-economics layer

`financial_economics.py` implements model-agnostic primitives:

- `SecurityValuation(cost, price)` represents the paper's `(C, P)` security valuation pair.
- `ValuationObservation` stores one observed valuation input without collapsing its state.
- `ValuationEvidence` stores collections of observations independently of any valuation model.
- `DecisionState` permits `Unavailable` and `Undecidable` outcomes instead of forcing every case into a numerical recommendation.

For finite cost and price values, valuation comparisons can be made directly. `Null`, `Indeterminate`, and unbounded states remain explicit and are never silently converted to arbitrary finite numbers.

## Ratio examples

```python
from complete_computables import ratio

ratio(None, 5)   # Null
ratio(0, 0)      # Indeterminate
ratio(2, 0)      # +Infinity
ratio(-2, 0)     # -Infinity
ratio(6, 3)      # + with value 2.0
```

## Design constraint

Complete Computables is a computational substrate, not a Black-Scholes-style closed-form pricing model. Higher layers may use empirical models, stochastic-volatility models, jump processes, simulations, machine learning, structural models, or other valuation methods without redefining the underlying seven-state semantics.

## Validation

- `test_complete_computables.py` covers the state space, ratio rules, finite-value preservation, constructor invariants, projections, extended-state arithmetic, and information ordering.
- `test_financial_economics.py` covers `(C, P)` representation, unavailable and undecidable valuations, finite valuation classification, unbounded states, ratios, and information-preserving evidence.

GitHub Actions validates pushes and pull requests to `main`. If the account-level Actions spending limit prevents runner allocation, CI remains an infrastructure constraint rather than a test result.
