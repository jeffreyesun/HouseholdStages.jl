# Aiyagari MIT shock — transition path + sequence-space utilities

The Aiyagari household block under a one-time unanticipated permanent
TFP step. Two distinct uses of the same chain:

- **`transition.jl`** — solves the deterministic perfect-foresight
  transition by damped Picard on the path of aggregate capital
  `{K_t}` using `solve_transition_given_env_path!`.
- **`ssj.jl`** — runs the household layer's two env-derivative
  services at the steady state, `compute_fake_news_ssj` and
  `compute_steady_state_gradient`, plus `expectation_vectors` on its
  own.

The chain is the same three-stage decomposition as `../aiyagari/`:

```
IncomeShock ∘ IncomeReceipt ∘ ConsumptionSavingsStage
```

TFP `A_t` is carried as an explicit argument to `aiyagari_prices` (not
baked into the chain) so the transition driver can sweep it period by
period; the household chain's env stays minimal — `(; K, r, w)`.

## The shock

A **permanent step**: `A_t = A_0` for all `t ≥ 1` with default
`A_0 = 1.05`. The economy starts in the deterministic steady state at
`A = 1.0` and transitions to a new steady state at the higher TFP
level. `model.jl`'s `tfp_path` builds the path.

## Transition algorithm (`transition.jl`)

```julia
tr = mit_shock_transition(; T = 100, A_0 = 1.05,
                            damping = 0.2, tol = 1e-3)
```

1. Solve the pre-shock steady state at `A = 1`. Pins `K_ss^pre`,
   `V_ss^pre`, `Λ_ss^pre`.
2. Solve the post-shock steady state at `A = A_0`. Pins `K_ss^post`,
   `V_ss^post`.
3. Initialise `{K_t}` as a linear interpolation `K_ss^pre →
   K_ss^post`.
4. Iterate the damped Picard step:
   - Build `env_path[t] = (; K = K_t, aiyagari_prices(K_t, A_t)...)`.
   - One call to `solve_transition_given_env_path!(hh, env_path;
     Λ_0 = Λ_ss^pre, V_T = V_ss^post)` runs the per-period backward
     sweep then forward sweep with per-period buffers, and returns
     `tr.moments_path[t].K_supplied` for each `t`.
   - Damped update: `K_t ← (1 − d) K_t + d K_supplied_t`.
5. Stop when `‖K_supplied − K‖∞ < tol`.

The per-period buffer separation inside
`solve_transition_given_env_path!` is what makes step 4 safe:
each forward pass reads the kernel materialised by *its own* backward
pass at the period-specific env, not the stale kernel from a
neighbouring period.

## Transition result

```
K_ss^pre     = 5.6847
K_ss^post    = 6.1352
K[1]   (impact)  = 5.7111
K[5]             = 5.8348
K[20]            = 6.0538
K[50]            = 6.1284
K[100] (≈end)    = 6.1340
```

Converges in 15 outer iterations at `damping = 0.2`, `tol = 1e-3`.
Larger damping (≥ 0.3) oscillates near the impact period: the chain
redistributes mass by shares across three stages, so `K_supplied`
trails a `K` update by more than one period and the outer loop needs
the heavier damping to stay stable.

## Sequence-space derivatives (`ssj.jl`)

The env-derivative services on this chain, with `:r` as the input and
the moment `K_supplied` as the output:

```julia
ssj_demo(; T_horizon = 30)
```

1. Solve the pre-shock steady state.
2. Re-seed the chain's kernels at the steady-state env via one
   `backward!` / `forward!` pair (so the K used in the K-transpose
   iteration is the SS K).
3. `𝓔 = expectation_vectors(hh, cell -> cell.wealth, T_horizon)`
   iterates the chain's `forward_adjoint!` to produce the
   K-transpose-propagated expectation arrays — step 2 of the
   fake-news algorithm (Auclert-Bardóczy-Rognlie-Straub 2021), shown
   standalone here and run internally by the driver below.
4. `compute_fake_news_ssj(hh, env_ss, T_horizon; …)` assembles
   the whole `J[t, s] = ∂K_supplied_t/∂r_s`, once at `mode = :dual`
   (one tangent-seated Dual chain, exact at every anticipation
   distance) and once at `mode = :fd` (two primal lanes at
   `env_ss ± h`, `O(h²)`), and prints `max|J_dual − J_fd|` as a
   self-check. `ssj.jl`'s header says what limits that number, and
   why the steady-state solve tolerance has to be tight before an
   `h`-scan of it means anything.
5. `compute_steady_state_gradient(hh, env_ss; …)` gives the
   permanent-shock comparative static `∂K_supplied/∂r`. It re-solves
   the steady state at `env_ss ± h` rather than differentiating it —
   the steady state is a fixed point, and this package does not
   differentiate fixed points — so `:fd` is its only mode and its
   accuracy floor is `lambda_tol`, which the call passes explicitly.

The SS-correctness check `⟨𝓔[t], Λ_ss⟩ ≈ K_ss` holds for all `t` (to
~4 decimals at `N_w = 400`) — `𝓔[t]` is the K-transpose-propagated
expectation of the wealth integrand `t` periods out, which against
the stationary distribution returns the same steady-state aggregate
the Λ-side `compute_moments` does.

### What `expectation_vectors` requires

`forward_adjoint!` methods on every component of the chain. These
come generically from each stage's seated kernel
(`src/lifts/jacobian.jl`); only stages whose forward is not a plain
`K·Λ` (`PointwiseScaleStage`, `EntryStage`) carry overrides.

## Finite-difference cross-checks

The `:fd` lane of step 4 is meaningful here because the savings
policy is a continuous off-grid position (`ContinuousArgmaxStage`
seating an `InterpKernel`), so it moves smoothly with sub-grid price
perturbations. The independent oracle — the direct method, FD on
whole transition paths — is heavy at `N_w = 400` and lives in
`test/test_sequence_space.jl` on a smaller fixture, alongside the
fake-news identity `F[t,s] = J[t,s] − J[t−1,s−1]`.

## Run

```bash
julia --project=. examples/aiyagari_mit_shock/transition.jl
julia --project=. examples/aiyagari_mit_shock/ssj.jl
```

The transition takes ~30 s at `T = 100`, `N_w = 400`; the derivatives
demo takes ~14 s at `T_horizon = 30`, most of it compilation.

## Files

- `model.jl` — params, layout, stage constructors,
  `aiyagari_prices(K, A, p)`, `tfp_path`.
- `transition.jl` — pre- and post-shock SS warm-starts plus the
  damped Picard transition loop.
- `ssj.jl` — sequence-space and steady-state derivatives demo.
- `../notebooks/aiyagari_mit_shock.jl` — self-contained walkthrough,
  model through sequence-space derivatives.
