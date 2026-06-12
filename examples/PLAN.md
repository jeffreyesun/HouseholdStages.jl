# HouseholdStages Examples — Living Plan

> **Location.** `HouseholdStages/examples/` (moved 2026-05-17 from the
> repo-root `CVIAYN_examples/`; see the Decisions log in
> `PROJECT_PLAN.md`).
>
> **Purpose.** Concrete model implementations that consume the
> `HouseholdStages/` library. Each example exercises a different model
> feature and serves double duty as (a) a regression test of the core
> library on a real workload and (b) a source of figures/numbers for
> the paper.
>
> **Per-example self-contained discipline (locked 2026-05-13;
> revised 2026-05-18 second pass).** Each example folder is
> self-contained at the *model* layer — params, layout, household
> chain, prices, utility — with **no shared model code across
> examples**. The *outer-loop* layer (tatonnement on $\bar K$ or a
> $K$-path, AR(1) shock generators) is also example-specific and
> stays in `<example>/{steady_state,transition}.jl`; only the inner
> V/Λ fixed-point solves at a given env come from
> `HouseholdStages/src/outer_loop.jl` (three helpers, see
> `HouseholdStages/PLAN.md` Cell 4). The four `examples/notebooks/*.jl`
> driver scripts depend only on `HouseholdStages` and **never**
> `include` sibling example folders — each notebook inlines its own
> model + outer loop. See `PROJECT_PLAN.md` Decisions log 2026-05-18
> for the principle.
>
> **One evidence-driven question still open, scheduled ~2–4 weeks
> out:** What's in `HouseholdStages` that no example actually uses?
> prune.
>
> **Intentional duplication across examples (do not factor).** A
> casual reader of the four example folders will notice three
> patterns recurring with byte-identical bodies. They are
> deliberately duplicated, not accidentally:
>
> 1. **`u_crra` CRRA utility.** Each example folder defines its own
>    `u_crra` (and L05 inlines it too). They are byte-identical. Per
>    the self-contained discipline above, each example owns its own
>    economic primitives; factoring `u_crra` into the library or a
>    shared helper would force every example to gain a dependency on
>    the helper file and remove the most-pedagogically-clear single-
>    file model statement. Keep the duplication.
> 2. **`Base.Broadcast.broadcastable(p::ParamsStruct) = Ref(p)`.** Each
>    example's `Params` struct opts into broadcast-as-scalar so that
>    closures captured by `env` can broadcast over cell arrays.
>    Identical body per example, intentional. (The library cannot
>    define this on the user's struct without owning the struct.)
> 3. **Tatonnement-loop scaffold.** Each example's
>    `steady_state.jl` and (where present) `transition.jl` rolls its
>    own damped tatonnement (init $\bar K$, iterate prices →
>    `solve_steady_state_given_env!` → moment readout → damped
>    update → residual check). The skeleton is structurally
>    identical across the four examples and the L05 tutorial
>    notebook, but each example wants different update rules,
>    verbose-print formats, and residual semantics — and per the
>    "close-the-model outer loops are example-specific" principle
>    of `HouseholdStages/PLAN.md` Cell 4, this duplication is the
>    cost of keeping the library out of model-specific business.

---

## Phase 0 — Steady-state examples

- **Cell 0 — Aiyagari (1994)**
  - **Plan.** Bare-bones heterogeneous-agent steady state, no
    aggregate shock. Smallest non-trivial exercise of the new
    `HouseholdStages` library.
  - **Specs.** N_w = 400 wealth-grid points (exponentially spaced)
    over [0, 100]; 3-state productivity Markov; tatonnement on
    aggregate K; inner VFI to 1e-7 and inner Λ fixed point to 1e-6
    per outer iteration. The exponential grid is required by the new
    3-stage decomposition's `WealthChangeStage.backward`, which linearly
    interpolates V past the top of the grid for cells where
    `(1+r) b + w y > w_max`; on a uniform grid the extrapolation
    amplifies V each pass and breaks the Bellman contraction.
    `ConsumptionSavingsStage` uses `monotone_search = :divide_conquer`
    (O(n_w log n_w) per slice — safe under concave u + linear
    budget). Inner VFI is plain backward iteration (Howard's policy
    iteration was removed library-wide on 2026-05-17 — see
    `PROJECT_PLAN.md` Decisions log).
  - **Status.** **Updated 2026-05-18 (second-pass refactor).**
    `aiyagari/{model.jl, steady_state.jl, README.md}` compose
    `IncomeShock ∘ IncomeReceipt ∘ ConsumptionSavingsStage` (canonical
    L03/L04 decomposition). At N_w = 400 converges to K = 5.6847
    in 18 outer iterations. `steady_state.jl` now rolls its own
    damped tatonnement loop on $\bar K$ (using
    `solve_steady_state_given_env!` for the per-K inner work);
    the brief same-day stint with `solve_picard_steady_state` was
    reverted on the user's principle that closing-the-model loops
    belong with the consumer.

- **Cell 1 — Krusell-Smith (deterministic aggregate)**
  - **Plan.** K-S household block with constant TFP `A = 1.0`. Same
    chain shape as Aiyagari but with the K-S employed/unemployed
    income process.
  - **Specs.** β = 0.96 (annual-style), log utility, `P_y` with ≈11%
    stationary unemployment, `y_unemp = 0.07` (canonical K-S — a
    strictly positive unemployed income is needed for the new chain
    so the b = 0 corner stays feasible). 100-point exponential wealth
    grid on [0, 200] (wider/finer than Aiyagari's because K-S sits at
    `β(1+r) ≈ 1` where small changes in r flip the argmax policy).
    Tatonnement on K with `update_speed = 0.01`, `rtol = 0.05`.
    (More aggressive outer drivers are not portable to the 3-stage
    chain — extreme-K probes push `r > 0.3` and the linear
    V-extrapolation in `WealthChangeStage.backward` amplifies V faster
    than `1/β`, breaking the Bellman contraction.)
  - **Status.** **Updated 2026-05-18 (second-pass refactor).**
    `krusell_smith/{model.jl, steady_state.jl, README.md}` now compose
    `IncomeShock ∘ IncomeReceipt ∘ ConsumptionSavingsStage`. Converges to
    K = 12.8791, r = 0.0404, w = 1.6703 in 24 outer iterations
    (β(1+r) = 0.99880). `steady_state.jl` rolls its own damped
    tatonnement on $\bar K$ (using `solve_steady_state_given_env!`
    for the per-K inner work). The OLD
    "K ≈ 11.4" claim in this slot was never verified: the previous
    `ks_savings_stage` had closure signatures that didn't match the
    new `(cell, c; env)` / `(cell, a_next; env)` API, so running the
    pre-refactor `krusell_smith/steady_state.jl` raised
    `MethodError` (verified by stashing and rerunning). The new
    chain's K_supplied(K) is a step function with a particularly
    sharp policy switch at K-S's near-watershed equilibrium — the
    rtol = 5% floor is set by that switch, not by anything we can
    tighten without smoothing the savings policy.

## Phase 1 — Transition path

- **Cell 2 — Aiyagari MIT shock**
  - **Plan.** Aiyagari household + one-time unanticipated TFP shock
    with AR(1) decay back to steady state. Two artifacts:
    (i) `transition.jl` solves the deterministic perfect-foresight
    transition by damped Picard on the path of K_t;
    (ii) `ssj.jl` demonstrates the household-layer sequence-space
    utilities (`expectation_vectors`, `build_F`, `J_from_F`).
  - **Specs.** T = 100 periods; A_0 = 1.05; ρ = 0.85; damping = 0.2
    (lowered from 0.5; the new 3-stage chain diffuses mass more
    slowly than the old `GridSavings`-only chain, and damping ≥ 0.3
    oscillates and diverges into a multi-period cycle). SSJ horizon
    30. Hard-argmax `ConsumptionSavingsStage` makes finite-difference
    cross-check at small ε zero; the example documents this and stops
    short of FD validation. Steady-state warm start is tatonnement —
    aggressive extreme-K probes break the Bellman contraction
    (same finding as Aiyagari and K-S).
  - **Status.** **Updated 2026-05-18 (second-pass refactor).**
    `aiyagari_mit_shock/{model.jl, transition.jl, ssj.jl, README.md}`
    now compose `IncomeShock ∘ IncomeReceipt ∘ ConsumptionSavingsStage`.
    Transition lands `K[1] = 5.74` (impact), peaks at `K[5] = 5.87`,
    decays to `K[100] ≈ 5.69` at horizon end. Residual hits the
    hard-argmax `ConsumptionSavingsStage` discretization floor around
    2.6e-3 by ~iter 20 and oscillates there (the earlier PLAN
    status's "converges in 26 iters to tol = 1e-3" claim was
    incorrect both before and after the refactor). `transition.jl`
    rolls its own SS-warm-start tatonnement loop AND its own
    transition driver, using `solve_steady_state_given_env!` for
    per-env inner work. The local `tfp_path` is defined in
    `model.jl` (was briefly promoted to `src/` mid-day on
    2026-05-18, then put back here). SSJ utilities run end-to-end:
    `expectation_vectors` produces 30 arrays of shape (80, 3);
    `build_F` + `J_from_F` assemble a 30×30 sequence-space Jacobian.
    The SS-correctness check `⟨𝓔[t], Λ_ss⟩ ≈ K_ss` holds to 4
    decimals for all t. **Library additions:** SSJ required
    `forward_adjoint!` methods on `WealthChangeStage` and
    `ConsumptionSavingsStage` (both previously hit the error fallback);
    these were added in `HouseholdStages/src/lifts/jacobian.jl`
    (`_share_gather!` helper for WealthChangeStage; sparse-policy gather
    for ConsumptionSavingsStage). `WealthChangeStage.backward_adjoint!` left
    stubbed — not needed by SSJ. Notebook driver created at
    `notebooks/aiyagari_mit_shock.jl`. **Previously** (2026-05-13):
    combined `shock ∘ GridSavings`, transition in ~11 iters at
    damping = 0.5.

## Phase 2 — Spatial

- **Cell 3 — Spatial (two locations)**
  - **Plan.** Smallest spatial extension. State space includes a
    `:location` axis (categorical over `[:home, :abroad]`). Stages:
    `IncomeShock ∘ MigrationStage ∘ IncomeReceipt ∘
    ConsumptionSavingsStage` — the **dedicated `MigrationStage` stage** (added
    2026-05-16; cost matrix + ε, no user closure) sits between the
    income shock and the L03/L04 savings decomposition. Per-location
    moments via integrand closures reading `cell.location`. Outer
    loop: damped Picard on `(K_home, K_abroad)`.
  - **Specs.** N_w = 60 (exponential wealth grid on [0, 30]); 3-state
    income; baseline calibration uses equal TFP across locations (a
    non-zero productivity gap induces period-3 oscillations under
    damped Picard; documented in the example). High logit ε = 5.0 to
    stabilize the iteration. Inner V tol 1e-7, Λ tol 1e-6 (matching
    the other examples after the refactor); outer damped Picard at
    damping = 0.1, rtol = 0.25 absolute (≈8% of K_eq) — accepts the
    hard-argmax `ConsumptionSavingsStage` step-function noise floor.
  - **Status.** **Updated 2026-05-18 (second-pass refactor).**
    `spatial/{model.jl, steady_state.jl, README.md}` compose
    `IncomeShock ∘ MigrationStage ∘ IncomeReceipt ∘
    ConsumptionSavingsStage`. The migration stage is the
    `MigrationStage(layout; migration_cost, ε)` (cost matrix + Gumbel
    dispersion; no `flow_payoff` closure). At N_w = 400 converges in
    3 outer iterations to K_home = K_abroad = 2.8305; pop = 0.5/0.5
    symmetric; r ≈ 0.039, w ≈ 1.195. `steady_state.jl` rolls its
    own absolute-damped tatonnement on the pair `(K_home, K_abroad)`,
    using `solve_steady_state_given_env!` for the per-pair inner
    work. The
    `migration_cost` field is no longer carried on `env` (it's static
    on the stage); the env now has just the four prices. Inner VFI is
    plain backward iteration (Howard removed library-wide 2026-05-17;
    MigrationStage's smooth choice would have made Howard a no-op here
    anyway). Total K is about 5% below the
    pre-refactor "K ≈ 3.0 per location" claim (the new chain's
    `WealthChangeStage` + `ConsumptionSavingsStage` decomposition uses linear V
    interpolation rather than the hard-policy push, and the active
    wealth region's tighter resolution under the exponential grid
    shifts K down — same direction as Aiyagari 5.87 → 5.62 under
    the same refactor). Notebook driver created at
    `notebooks/spatial.jl`. **Previously** (2026-05-13): combined
    `MarkovStage ∘ LogitChoiceStage ∘ GridSavings`, K_home = K_abroad ≈
    3.0.

## Phase 3+ — Deferred / future

The original `INITIAL_PLAN.md` lists future examples: search,
housing, sector switching, trade, OLG, spatial+housing. None are
in scope tonight. The "what's worth extracting from the four
existing examples?" question is the next-look decision (~2-4 weeks
out per the methodology).

---

## Cross-cutting

- **Layout convention.** Each example is its own subfolder:
  - `model.jl` — stage and price definitions, plus any
    example-specific transition utilities (`tfp_path` etc.).
  - `steady_state.jl` / `transition.jl` / `ssj.jl` — example-side
    outer-loop drivers (tatonnement on $\bar K$, MIT-shock
    transition, SSJ pipeline). These call into
    `HouseholdStages.solve_steady_state_given_env!` (and the V-only
    / Λ-only variants) for the per-env inner work, but the loops
    themselves are written here.
  - `README.md` — model description, parameters, expected outputs.
  - No `Project.toml` per example — they all run from the workspace
    root `Project.toml` which develops `HouseholdStages/`.
- **`notebooks/` subfolder.** Per-example pedagogical drivers
  (`aiyagari.jl`, `krusell_smith.jl`, `aiyagari_mit_shock.jl`,
  `spatial.jl`) that inline the *model* and call the
  `HouseholdStages` helpers for the outer loop. Self-contained;
  depend only on `HouseholdStages`. See `tutorial_notebooks/L05_*`
  for the slide-companion notebook on the MIT-shock case.
- **Dropped from the previous plan.** Cells 4–8 from the prior
  PLAN.md (search, housing, sector switching, trade, OLG,
  spatial+housing) are deferred to when the methodology says the
  next batch is informative.
- **Calibration_audit.jl** moved to `_attic/`. The audit notes
  remain available for users investigating K-S calibration choices.
