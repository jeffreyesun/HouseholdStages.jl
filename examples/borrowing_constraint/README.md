# Aiyagari (1994) — Natural vs Ad-hoc Borrowing Limits

The two classical borrowing limits of the income-fluctuation problem, on an
asset grid spanning negative wealth. The limit lives in the **grid floor**, not
in a separate stage — so the household block is just the canonical spine.

## Household block

```
IncomeShock ∘ IncomeReceipt ∘ ConsumptionSavingsStage
```

(`MarkovStage(:income) ∘ IncomeStage ∘ ConsumptionSavingsStage`, the same spine
as Aiyagari/Huggett.) The asset grid is Huggett's: a linear borrowing segment
`[a_min, 0)` glued to a log-spaced positive segment.

## The two limits

- **Ad-hoc limit** `a ≥ φ`: an exogenous, state-independent floor — just the
  lower end of the asset grid, `a_min`.
- **Natural limit** `a ≥ −y_min/r`: the deepest **sustainable** debt. At
  `a = −y_min/r` rolling the debt over leaves exactly `c = 0` in the worst income
  state (`r·a + y_min = 0`); borrowing past it forces `c < 0` in some history,
  which CRRA (`u(0) = −∞`) rules out. With CRRA the natural limit is therefore
  enforced **automatically** by the `c ≥ 0` feasibility already inside
  `ConsumptionSavingsStage` — set `a_min = −y_min/r` and **no extra stage is
  needed**. (A grid floor BELOW `−y_min/r` would create infeasible-to-sustain
  cells that trap mass; the floor must sit at or just inside the natural limit.)

## What it shows

The same spine is solved at two grid floors (`steady_state.jl → compare_limits`):

```
β = 0.96, σ = 1.5, r = 0.0300;  natural limit −y_min/r = −3.333
  a_min = −3.003 (near-natural): A_mean = +0.31, frac borrowing = 0.36, ΣΛ = 1.0
  a_min = −1.000 (tighter ad-hoc): A_mean = +1.98, frac borrowing = 0.04, ΣΛ = 1.0
```

The tighter ad-hoc floor binds sooner: far fewer households in debt, a larger
precautionary buffer. Both VFI = 416 (genuine convergence).

## Dogfooding finding — `BorrowingConstraintStage` is unusable in the VFI loop

The natural API for a **state-dependent** limit (the per-income natural limit
`a ≥ −y/r`, or the solvency bounds in `examples/limited_commitment`) is
`BorrowingConstraintStage(infeasible = (; wealth, income, env) -> …)`, which
masks infeasible cells with `−Inf`. It does not work here:

1. **The `−Inf` breaks VFI convergence.** The fixed-point loop
   (`src/outer_loop/outer_loop_internal.jl:65`) measures
   `diff = maximum(abs, V_new .- V)`. Once a cell holds a steady `−Inf`,
   `V_new − V = −Inf − (−Inf) = NaN`, so `maximum(abs, …) = NaN` and
   `while diff > tol` is `NaN > tol = false`: iteration **terminates after 2
   passes** with an unconverged V, a garbage policy, and a Λ that collapses onto
   the masked cells. Confirmed directly: with the stage, VFI = 2 and all mass
   piles on the masked floor; without it, VFI = 416 and the distribution is
   well-behaved.
2. **CRRA traps.** Masking exactly at the natural limit `−y/r` on a grid that
   extends below it leaves cells that are infeasible-to-sustain regardless of the
   mask (the `c → 0` boundary), which absorb mass.

Both are `src/`-level issues (off-limits in this exercise). The fix would be to
make the VFI norm ignore `NaN`/`−Inf` cells (e.g. a finite-cell sup-norm). The
within-constraints workaround — a `UtilityStage` with a **finite** penalty placed
after the savings choice — is demonstrated in `examples/limited_commitment`.

## Run

```julia
# from HouseholdStages/
julia --project=. examples/borrowing_constraint/steady_state.jl
```
