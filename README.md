# HouseholdStages

A Julia library for the **household block** of heterogeneous-agent macro
models when The within-period problem decomposes into stages.

The library provides:

1. Optimized primitive stages:
  - Markov shock
  - Deterministic update of wealth or other continuous variable
  - Discrete or continuous choice
  - Continuous choice over Markov transitions, a la rational expectations, search-and-matching, and more!

2. "Segmentation" machinery to allow stages to depend on idiosyncratic states *arbitrarily*. For example:
  - Migration costs can depend on an individual's age, income, or anything.
  - Income transition probabilities can depend on age, race, wealth, or anything.
  - Any stage parameter can depend on any idiosyncratic state.

3. A mean-field machinery to allow stages to depend on ambient variables like prices and parameters to enable efficient market-clearing and calibration.

4. A moment-attaching machinery to compute aggregate moments like aggregate demand.

5. Two combinators which allow stages to be combined into composite stages (The Stage category is closed under these combinators):
- Temporal composition `∘`
- Direct sum or parallel composition `⊕`

6. Two lifts:
- To Dual for forward-mode AD
- To optimized GPU kernels

7. Machinery to compute sequence-space-Jacobians (SSJs).

## Mathematical Framework

This package implements "stages", which are *segmented, reward-tagged, mean-field Markov kernels*.

Every stage exposes the same **K-operator** signature: `backward!`
pushes a value function `V` through the adjoint `Kᵀ`, `forward!`
pushes a distribution `Λ` through `K`. Adjointness of `K` and `Kᵀ`
gives the duality identity

```
⟨V_in, Λ_in⟩ = ⟨V_out, Λ_out⟩ + ⟨r, Λ_in⟩
```

(where `r` is the stage's flow payoff). That identity is the
correctness test every stage gets for free.

The library is the operational companion to a separate
categorical-foundations paper. The paper proves the structural facts
the library uses — composition is associative, the per-stage lifts are
functorial, the fake-news algorithm is a forward-mode Jacobian functor —
but the library stands on its own.

## Install

```julia
julia> ]activate path/to/HouseholdStages
julia> ]instantiate
julia> using HouseholdStages
```

Dependencies: `ForwardDiff` (for `lift_jacobian`), `Adapt` (for
`to_device` / `lift_gpu`), `NNlib`, and `SpecialFunctions`, plus stdlib
`LinearAlgebra` and `Printf`. `CUDA` is an optional weak dependency that
loads the GPU extension. Julia 1.10 or later.

## Worked example — Aiyagari steady state

A three-stage chain (income shock, wealth receipt, consumption-savings
argmax) with the moment `K_supplied = ∫ wealth dΛ` attached, solved by
damped tatonnement on aggregate capital `K`.

```julia
using HouseholdStages

@kwdef struct AiyagariParams
    β :: Float64 = 0.96; σ :: Float64 = 1.5
    α :: Float64 = 0.36; δ :: Float64 = 0.08; L :: Float64 = 1.0
    y_grid :: Vector{Float64} = [0.6, 1.0, 1.4]
    P_y    :: Matrix{Float64} = [0.7 0.2 0.1;
                                 0.2 0.6 0.2;
                                 0.1 0.2 0.7]
    N_w :: Int = 400; w_max :: Float64 = 100.0
end
Base.Broadcast.broadcastable(p::AiyagariParams) = Ref(p)

function aiyagari_household(p = AiyagariParams())
    layout = GriddedLayout(
        :wealth => GriddedContinuous(0.0, p.w_max, p.N_w; spacing = :log),
        :income => Discrete(p.y_grid),
    )

    shock   = MarkovStage(layout; axis = :income, transition_matrix = p.P_y)
    receipt = WealthChangeStage(layout;
        wealth_post = (; wealth, income, env) -> (1 + env.r) * wealth + env.w * income,
        axis        = :wealth,
    )
    savings = ConsumptionSavingsStage(layout;
        β       = p.β,
        utility = (cell, c) -> u_crra(c, p.σ),
    )

    hh = shock ∘ receipt ∘ savings
    return define_moments!(hh;
        K_supplied = at_end(integrand = :wealth, reduce = sum),
    )
end

function aiyagari_prices(K, p = AiyagariParams())
    r = p.α * (K / p.L)^(p.α - 1) - p.δ
    w = (1 - p.α) * (K / p.L)^p.α
    return (; r, w)
end

function aiyagari_steady_state(; K_init = 5.0, update_speed = 0.05,
                                 rtol = 2e-2, max_iter = 500)
    p  = AiyagariParams()
    hh = aiyagari_household(p)
    K, V, Λ, err = K_init, nothing, nothing, Inf
    for iter in 1:max_iter
        env = (; K, aiyagari_prices(K, p)...)
        res = isnothing(V) ?
            solve_steady_state_given_env!(hh, env) :
            solve_steady_state_given_env!(hh, env; V_init = V, Λ_init = Λ)
        V, Λ = res.V, res.Λ
        K_sup = res.moments.K_supplied
        err   = abs(K_sup - K) / K
        err <= rtol && break
        K += update_speed * (K_sup - K)
    end
    return (; K, aiyagari_prices(K)..., V, Λ, err)
end
```

`aiyagari_steady_state()` returns `K = 5.679`, `r = 3.85%`, `w = 1.20`
in 18 outer iterations.

The example folder `examples/aiyagari/` ships a polished version with
verbose printing and CLI driver; `examples/notebooks/aiyagari.jl` is a
Pluto-style walkthrough.

## The stage catalog

A stage is a Julia struct whose K-operator is determined by its
configuration. Backward populates a per-call **kernel** (the K-operator's
runtime data — integer policy, choice-probability tensor, materialised
wealth-post array) and writes the new value function; forward consumes
the kernel and pushes the distribution. The library is organised
around the trichotomy `<X>StageSpec` (immutable configuration),
`<X>StageBuffer` (per-call state, including the kernel and a per-call
cache — a `NamedTuple` from `allocate_cache`, or `nothing`), `<X>Stage`
(the bundled user-facing wrapper). Only the bundled `<X>Stage` form is
exported.

### Exogenous transitions

- **`MarkovStage(layout; axis, transition_matrix)`** — Markov draw along a
  named axis. K is the transition matrix itself, V/θ-independent;
  kernel = `nothing`.

### Discrete choice

- **`ArgmaxStage(layout; reward, axis)`** — hard choice along a named
  `axis`: each origin cell takes the destination maximising
  `reward[dest, origin] + V_end[dest]`. `reward` is a matrix source
  (a `Matrix`, a `FromEnv`, or a `(; dep…[, env]) -> Matrix` closure).
  K is a sparse scatter; kernel = integer policy.
- **`LogitChoiceStage(layout; axis, cost_matrix, ε)`** —
  Gumbel-smoothed choice over the choice axis, with an origin→destination
  `cost_matrix[i, j]`. K is a stochastic kernel; kernel = action
  probability tensor. A destination payoff (state-dependent or a static
  amenity shifter) is V-additive, so it composes in as a `UtilityStage`
  before the logit (`LogitChoiceStage ∘ UtilityStage(u)`), or use
  `LogitUtilityStage` for the packaged composition.
- **`MigrationStage(layout; axis, migration_cost, ε)`** —
  cost-matrix logit on a location-style categorical axis.
  `migration_cost` is `(n_loc, n_loc)` (a `Matrix` or `FromEnv(:C)`),
  `migration_cost[i, j]` the cost of moving `i → j`. A destination
  amenity is a `UtilityStage` composed before the move.

### Wealth dynamics

- **`WealthChangeStage(layout; wealth_post, axis = :wealth)`** —
  deterministic wealth update via a `wealth_post` closure of the layout
  axes it names as keyword arguments, plus `env`
  (`(; wealth, income, env) -> …`). Backward interpolates `V_end` along
  the wealth axis at each cell's post-stage wealth; forward redistributes
  mass to the wealth grid via share-weighted accumulation (Young's
  method). A post-stage wealth past either end of the grid clamps to the
  endpoint, on both the V and the Λ side.
- **`AssetPriceChangeStage(layout; holdings_axis, ...)`** — sugar over
  `WealthChangeStage` for the asset-revaluation pattern `b_post =
  b_pre + (env.q − env.q_last) · cell.holdings_axis`. Returns a
  bundled `WealthChangeStage`; all `WealthChangeStage` machinery
  applies.
- **`ConsumptionSavingsStage(layout; β, utility, axis = :wealth, utility_axes)`** —
  pick next-period wealth `b_end`; implicit budget `c = b_in − b_end`,
  with `c ≤ 0` masked out. `utility` takes `(cell, c)` positionally
  (optionally `; env`) and must be supermodular; name any state beyond
  `axis` that it reads in `utility_axes`. The choice is solved by a
  monotone divide-and-conquer node walk, `O(n_w log n_w)` per slice.
- **`BorrowingConstraintStage(layout; infeasible)`** — mask infeasible
  cells with `-Inf` on V; identity on Λ. `infeasible` is either a
  precomputed `AbstractArray{Bool}` of layout shape or a
  `(; ax…[, env]) -> Bool` predicate (re-evaluated each backward pass).

### Glue

- **`UtilityStage(layout; utility)`** — additive flow utility on V;
  identity on Λ. K is the identity on measures. Also serves as a
  terminal / bequest stage: pass the bequest function as `utility`
  and seed the chain with `V_end = 0`.
- **`IdentityStage(layout)`** — no-op (K = I). Useful as a component
  inside `product` when one branch performs no within-period action.
- **`ForgetfulSumStage(layout; axis)`** — drop one axis of the
  state space. Backward broadcasts V along the dropped axis; forward
  sums Λ along it. The canonical layout-changing stage.

## Composition, product, replication

```julia
chain = s1 ∘ s2 ∘ s3                                  # time-ordered
prod  = s1 ⊕ s2                                       # parallel
cohort_chain = replicate_age(chain, N; axis = :age)   # N uniform copies
```

- `∘` is `Base.:∘` overloaded on stages — **time-ordered**, opposite
  of Julia's function composition. `s1 ∘ s2` runs `s1` first.
- `⊕` builds a `ProductStage` along a new axis (default `:group`).
  The K-operator is the block-diagonal direct sum of the components';
  forward and backward operate per-component on slices of a fused
  tensor. Each factor must carry the product axis at size 1 and span the
  same two layouts as the others; their specs may otherwise differ.
- `replicate_age(stage, N)` is sugar for `product(stage, stage, …,
  stage; axis = :age)`. Cross-cohort threading (bequest, birth,
  mortality) is the caller's responsibility.

`ChainStage` and `ProductStage` are themselves stages — closure under
`∘` and `⊕` makes compounds usable anywhere a primitive stage is.

## Moments

```julia
hh = chain
define_moments!(hh;
    K_supplied = at_end(integrand = :wealth,                       reduce = sum),
    L_supplied = at_end(integrand = (; income) -> income,          reduce = sum),
)
m = compute_moments(hh, Λ, env)   # m.K_supplied, m.L_supplied
```

`define_moment!` / `define_moments!` are append-only by default; pass
`overwrite_existing_moment_definitions = true` to override.
`at_end(; integrand, reduce)`'s `integrand` is either a `Symbol`
(axis-field shortcut: `:wealth` ≡ `(; wealth) -> wealth`) or a closure
following the standard `(; ax…[, env])` convention. `reduce` is
typically `sum` or `mean`.

`compute_moments(hh, Λ, env)` is non-mutating and takes `Λ` explicitly
— it doesn't read the chain's buffer.

## User-facing solvers

Three helpers absorb the per-env inner work (V backward to a fixed
point, Λ forward to its stationary distribution, plus moment readout)
so the consumer can focus on the outer loop:

- `solve_vfi_steady_state_given_env!(hh, env; V_init, tol, maxiter)`
  — repeated `backward!` to V's fixed point. Returns `(; V, iters,
  converged)`.
- `solve_lambda_steady_state_given_env!(hh; Λ_init, tol, maxiter)` —
  repeated `forward!` to Λ's stationary distribution. Kernels must
  have been seated by a prior `backward!` at the same env. Returns
  `(; Λ, iters, converged)`.
- `solve_steady_state_given_env!(hh, env; V_init, Λ_init, ...)` —
  bundles the above and also evaluates moments. Returns
  `(; V, Λ, moments, history, iters)`. The bundled chain warm-starts
  from buffer state across calls, so a sequence of perturbed-env
  solves runs in a few VFI iterations per call.

For transition paths:

- `solve_transition_given_env_path!(hh, env_path; Λ_0, V_T)` —
  allocates `T` per-period chain copies (sharing the Spec, each with
  its own Buffer), runs a backward sweep `t = T:-1:1` then a forward
  sweep `t = 1:T`, returns `(; V_path, Λ_path, moments_path)`. The
  per-period buffer separation eliminates a class of stale-kernel bugs
  that hand-rolled transition drivers tend to hit.

Closing the model — tatonnement on `K` or `r`, AR(1) shock processes,
Anderson acceleration, calibration outer loops — is the consumer's
job. The library handles the household chain at a given env.

## Sequence-space utilities (fake-news algorithm)

`compute_fake_news_ssj(hh, env_ss, T; inputs, outputs, mode = :fd)` is the
one-call driver. It solves the steady state at `env_ss`, runs the fake-news
algorithm (Auclert-Bardóczy-Rognlie-Straub 2021) over a `T`-period horizon,
and returns the sequence-space Jacobians `J[t, s] = ∂y_t/∂env_s` as a `Dict`
keyed by `(input, output)` — bare for a single pair — with output dates on
the rows and shock dates on the columns. Each moment must reduce with `sum`.
`mode = :fd` (default) differences the chain at `env_ss ± h`; `mode = :dual`
carries `ForwardDiff.Dual` seeds, `n_dual` inputs per pass.

The steps it runs are exposed individually for hand assembly:

- `expectation_vectors(hh, integrand, T)` — Step 2. Iterates the
  chain's `forward_adjoint!` (K-transpose action on a per-cell
  integrand) to produce expectation arrays for `t = 0, 1, …, T − 1`.
  The chain's kernels must have been seated by a prior `backward!`
  at the steady-state env; no env argument here.
- `build_F(curlyY, curlyD, curlyE)` — Step 3. Assembles the fake-news
  matrix from per-period direct effects (`curlyY`, `curlyD`, from
  Step 1) and the expectation vectors.
- `J_from_F(F)` — Step 4. Anti-diagonal cumulation into the
  sequence-space Jacobian.

Step 1 (per-period direct effects of a shock) is the env perturbation the
driver's `mode` selects; on the hand-assembled path it is model-specific,
typically a finite-difference perturbation of env around the steady state.
`examples/aiyagari_mit_shock/ssj.jl` runs the full pipeline end-to-end on
the Aiyagari chain.

## Lifts

- `lift_jacobian(stage; n_dual = 1, tag, primal_eltype)` — rebuild with
  `ForwardDiff.Dual`-typed buffers for forward-mode AD. The rebuild
  flows through `with_eltype`, which is keyed on each Spec type — every
  concrete stage in the library has a `with_eltype` method. Reverse mode
  is the per-stage adjoint surface: call `backward_adjoint!` /
  `forward_adjoint!` on a stage directly.
- `lift_gpu(stage, to)` — relocate a built stage's arrays onto a device
  through `Adapt` (`lift_gpu(stage, CuArray)`, aliased by
  `to_device(stage, CuArray)`); `to_host(stage)` brings it back. The
  CUDA extension supplies the on-device stratified driver.

The per-stage reverse-mode adjoints exist for every choice stage
(`ArgmaxStage`, `LogitChoiceStage`, `MigrationStage`,
`ConsumptionSavingsStage`) via the envelope theorem at the materialised
K. `WealthChangeStage` has both `forward_adjoint!` and `backward_adjoint!`
wired through its interpolation kernel — the stored K applied the other
way — with `forward_adjoint!` the direction `expectation_vectors` uses.

## Worked examples

Four self-contained examples under `examples/`. Each owns its outer
loop end-to-end; the library supplies stages, lifts, and the per-env
inner solvers.

| Example | Chain | Outer loop | Result |
|---|---|---|---|
| `aiyagari/` | `MarkovStage ∘ WealthChangeStage ∘ ConsumptionSavingsStage` | tatonnement on `K` | `K = 5.679`, 18 iters |
| `krusell_smith/` | same chain, K-S calibration | tatonnement on `K` | `K = 12.88`, 24 iters |
| `aiyagari_mit_shock/` | same chain, permanent TFP step | damped Picard on `{K_t}` (`transition.jl`); SSJ pipeline (`ssj.jl`) | `K_ss^pre = 5.679` → `K_ss^post = 6.135`, peak `K[20] ≈ 6.05`, 15 transition iters |
| `spatial/` | adds `MigrationStage` between income shock and receipt | damped Picard on `(K_home, K_abroad)` | `K_home = K_abroad = 2.83`, symmetric pop split, 3 iters |

All four run at `N_w = 400`.

`examples/notebooks/` contains Pluto-style four-section walkthroughs
for each model — closer to a tutorial than a CLI driver.

## Status

Things that work but have rough edges, or are scaffolded for future
work:

- **`lift_gpu` relocates a stage to a device, with rough edges.** A
  built stage moves through `Adapt` (`lift_gpu(stage, CuArray)` /
  `to_device` / `to_host`), and loading the CUDA extension supplies the
  on-device stratified driver. Ops ride into the kernel by value, so a
  stage whose closures capture non-`isbits` data (an array-carrying cost
  closure or `env`) stays on host arrays.
- **Differentiating through a `FromEnv` field.** `FromEnv(:key)` reads
  an env field at evaluation time — several example models use it to
  sweep a parameter without rebuilding the chain — but no shipping
  example feeds a `FromEnv`-sourced field as a differentiation input to
  the SSJ / gradient drivers, so that path is unproven on a real
  workload.
- **`ProductStage` requires components sharing the same two layouts** —
  each factor carries the product axis at size 1 and spans the same
  start and end layout; their specs may differ in type. Mismatched
  layouts raise at construction. Heterogeneous-shape products (e.g.,
  working vs. retired cohorts with different state spaces) would need a
  separate dispatch path.
## Comparison to existing toolkits

Closest in spirit to **SSJ** (Auclert, Bardóczy, Rognlie, Straub 2021),
whose `HetBlock` family exposes the same household-layer
decomposition. SSJ's primitives (`Continuous1D / 2D`, `Exogenous`,
discrete-choice helpers) instantiate the same kernel-producing
signature, the V/Λ duality is implicit in the back / forward-pass
structure, and the fake-news algorithm is the same machinery as
`expectation_vectors + build_F + J_from_F`. The difference: in
`HouseholdStages`, composition under `∘` and the per-stage
functorial lifts (`define_moment!`, `lift_jacobian`, `lift_gpu`,
`replicate_age`) are first-class operations on stages. Chains compose
by Julia operator, moments and Jacobians arise as lifts of the chain,
and the same interface carries forward through AD and GPU.

**HARK** (Carroll, Palmer, White et al.) decomposes period solvers
through class inheritance and `solveOnePeriod` chains, but stages are
implicit in the solver architecture rather than the unit of code.
**Reiter's** perturbation framework operates on the linearised
backward operator and is conceptually stage-shaped (steady state first,
then perturb), but the operator is constructed by hand rather than
composed. **Maliar-Maliar-Winant's** intratemporal / intertemporal
decomposition expresses the same algebraic split in a non-stage-shaped
(iteration-on-allocation) solver.

## Where to read more

- **`examples/`** — the four headline worked examples (the Aiyagari
  one is the smallest entry point), plus many more model directories,
  each with a short `README.md`. `examples/notebooks/` has Pluto-style
  walkthroughs of the headline models.
- **The source** — every stage, kernel, and solver carries a
  docstring; `?ArgmaxStage` (and any other exported name) in the REPL
  pulls up its contract. The `src/` tree is organised by subsystem —
  `stages/`, `kernels/`, `layouts/`, `combinators/`, `outer_loop/`.

## License

MIT.
