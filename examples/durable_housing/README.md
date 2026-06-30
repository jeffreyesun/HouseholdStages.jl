# Durable / housing (S,s) adjustment

A heterogeneous-agent household with a lumpy housing durable, in the (S,s)
tradition of **Berger–Vavra (2015)** and **Díaz–Luengo-Prado (2010)**. The
point of this example is that the entire household block is a composition of
**existing library stages** — no bespoke household stage is rolled here.

## Household block

The within-period problem, in time order:

```
Move ∘ IncomeShock ∘ BuyHome ∘ Receipt ∘ UserCost ∘ ConsumptionSavings
```

| Stage | Library stage | What it does |
|---|---|---|
| `Move` | `MarkovStage` (`:h`) | Owner→renter moving shock: owners are ejected to the renter level w.p. `π_move`, keeping the buy choice live in the stationary distribution. |
| `IncomeShock` | `MarkovStage` (`:income`) | 3-state labour-income Markov. |
| `BuyHome` | `BuyHomeStage` (`:h`) | Gated `ArgmaxStage`. A renter (h-index 1) may move to any owned size or stay renting; an owner is **gated to keep its own size**. That gate *is* the (S,s) inaction region — only renters re-optimise the stock. |
| `Receipt` | `WealthChangeStage` (`:wealth`) | `b ↦ (1+r)·b + w·y` (cash-on-hand). Placed after the buy so every cell entering savings has income credited — the wealth-grid floor is then strictly feasible (`c > 0`), which the gated-owner branch requires. |
| `UserCost` | `WealthChangeStage` (`:wealth`) | `b ↦ b − u·h` with the Jorgenson user cost `u = (r+δ)·q` (foregone return + depreciation/maintenance), charged to everyone on the owned slice. |
| `ConsumptionSavings` | `ConsumptionSavingsStage` (`:wealth`) | Picks next-period financial wealth `b'`; `c = b − b'`. Flow utility is a Cobb–Douglas consumption–housing composite `c^{1−ξ}·s(h)^ξ` put through CRRA, with the housing service `s(h)` folded in via `utility_axes = (:h,)`. |

Moments attached to the chain (`define_moments!`): `mean_wealth`, `mean_house`,
and the homeownership rate `own_rate = ∫ 1{h>0} dΛ`.

## A note on the price timing

A one-time stock-price purchase `q·h` is **not** expressible from existing
stages here: after the buy choice, a fresh buyer of size `h` and a continuing
owner of size `h` sit on the *identical* `(wealth, income, h)` cell, so any
`wealth_post(cell)` closure following the choice charges them identically.
Distinguishing them would need a pre-buy-size axis that no stage populates.
This example therefore uses the **user-cost** formulation — a per-period
housing cost `(r+δ)·q·h` charged uniformly to the owned slice — which *is*
expressible, sidesteps the distinguishability problem, and is the standard
Díaz–Luengo-Prado durable timing (per-period maintenance + depreciation, with
lumpiness coming from the buy gate rather than a one-shot transaction cost).
The catalog's other listed expression, `DurableAdjustmentStage` (a
`ContinuousArgmaxStage` whose reward charges `adjustment_cost(d' − d)` at the
choice), is the route to a one-time convex adjustment cost without any
distinguishability issue.

## Equilibrium

Returns, wage, and the house price are **exogenous** (partial equilibrium): there
is no market to clear, so the "outer loop" is a single inner V/Λ fixed-point
solve (`solve_steady_state_given_env!`). Impatience (`β(1+r) < 1`) plus the
wealth-grid floor deliver a stationary distribution; the moving shock keeps the
renter level populated so the buy choice stays live.

## Parameters and expected output

`N_w = 160` log-spaced wealth grid on `[0, 40]`; 3-state income; housing sizes
`[0, 1, 2]` (index 1 = renter); `β = 0.93`, `σ = 2`, `ξ = 0.15`, `q = 3.0`,
`δ = 0.03`, `π_move = 0.06`. At the baseline calibration the steady state is:

```
mean wealth    ≈ 0.94
mean house     ≈ 0.77
ownership rate ≈ 0.77
size policy    = [1, 3]   (renters who buy span the full size range)
```

A 77% ownership rate with a persistent renter tail (refreshed each period by the
moving shock) and a non-trivial buy policy.

## How to run

From the `HouseholdStages` directory:

```julia
# steady state + printed report
julia --project=. examples/durable_housing/steady_state.jl

# regression test (existing stages only)
julia --project=. -e 'using HouseholdStages; include("test/test_example_durable_housing.jl")'
```
