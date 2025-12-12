# Mini MLTL SAT Solver (Python + Z3)

This project is a small SAT solver for a **bounded fragment of Metric Linear Temporal Logic (MLTL)** implemented in Python using the Z3 SMT solver.

Given:
- an MLTL formula φ, and 
- a finite trace length `N`,

the solver checks whether there exists a trace of length `N` over propositional atoms such that **φ holds at time 0**. 
If the formula is satisfiable, the solver prints one satisfying trace and can optionally export it as a CSV file.

---

## 1. Supported Logic

The current implementation supports the following MLTL fragment over a finite, discrete-time trace `0 .. N-1`.

### 1.1 Atomic Propositions

- Examples: `p`, `q`, `r1`, `flag_ok`, ...
- Syntax rule: identifiers start with a letter or underscore, followed by letters, digits, or underscores.

### 1.2 Boolean Connectives

- `!φ` 
  Logical negation (not).
- `φ & ψ` 
  Conjunction (and).
- `φ | ψ` 
  Disjunction (or).
- `φ -> ψ` 
  Implication (implies, right-associative).

**Precedence (from lowest to highest):**

1. `->`
2. `|`
3. `&`
4. `!`, temporal operators, parentheses

Parentheses can always be used to override precedence, e.g. `(p & q) -> r`.

### 1.3 Temporal Operators (bounded)

All temporal operators are interpreted over a finite trace of length `N`, with time indices `0 .. N-1`.

- `F[a,b](φ)` 
  “Eventually within a bounded window” 
  At time `i`, `F[a,b](φ)` is true iff 
  there exists `k ∈ [a, b]` such that `i + k < N` and `φ` is true at time `i + k`.

- `G[a,b](φ)` 
  “Always within a bounded window”: 
  At time `i`, `G[a,b](φ)` is true iff 
  for all `k ∈ [a, b]` with `i + k < N`, `φ` is true at time `i + k`. 
  If there is no such valid `k` (the window is entirely beyond the end of the trace), this is vacuously true.

- `U[a,b](φ, ψ)` 
  Bounded **until**: 
  At time `i`, `φ U[a,b] ψ` is true iff there exists `k ∈ [a, b]` such that:
  - `i + k < N`, 
  - `ψ` is true at time `i + k`, and 
  - for all `j` with `0 ≤ j < k`, `φ` is true at time `i + j`. 
  Intuitively: starting at `i`, `φ` holds continuously until some time in the bounded window where `ψ` becomes true.

### 1.4 Parentheses

- `(φ)` 
  Standard parentheses for grouping and controlling precedence.

---

## 2. SAT Semantics

The solver checks **satisfiability at time 0**:

> Given a formula φ and trace length `N`, 
> does there exist a trace of length `N` over the atomic propositions 
> such that φ is true at **time 0**?

Internally, the encoding produces a Boolean vector `bits_φ[0..N-1]` for each subformula φ. 
For the root formula, `root_bits[0]` corresponds to φ being true at time 0.
The solver enforces:

```text
root_bits[0] == True

##3. Installation
     python -m venv venv
     source venv/bin/activate
 
     Run the solver: 
     python3 miltl_solver.py
