# Indivisible labor (Rogerson)

An incomplete-markets economy with **extensive-margin** labor: a household either works
full-time hours `n̄` or not at all — a discrete `{work, not-work}` participation choice —
then chooses savings. Working earns `w·ε·n̄` and costs a fixed disutility `v(n̄)`.
References: Rogerson (1988), Prescott–Rogerson–Wallenius (2009), Chang–Kim (2007).

## The ✅ build: discrete participation as an `ArgmaxStage` (sub-approach (a))

The participation choice is a `(max, +)` over a 2-level `:participation` axis, with the
chosen indicator feeding the budget — the auxiliary-axis pattern (as in `two_asset_hank`).
Household block, **existing stages only**:

```
EmploymentShock ∘ Participate ∘ Budget(reads participation) ∘ ConsumptionSavings
```

```julia
part_reward = [ 0.0    0.0;          # row 1 = not-work: reward 0
               -vbar  -vbar]         # row 2 = work:     reward −v(n̄)   (columns identical)

shock   = MarkovStage(layout; axis = :income, transition_matrix = P_y)
choose  = ArgmaxStage(layout; axis = :participation, reward = part_reward, search = :brute)
budget  = WealthChangeStage(layout; axis = :wealth,            # cash = (1+r)b + w·ε·n̄·1{work}
    wealth_post = (; wealth, income, participation, env) ->
        (1 + env.r) * wealth + env.w * income * nbar * participation)
savings = ConsumptionSavingsStage(layout; β, utility = (cell, c; env) -> u_crra(c, Val(σ)))

hh = shock ∘ choose ∘ budget ∘ savings
```

The period value is

```
V_start = max_p [ −v(n̄)·1{p=work} + max_{b'} ( u(c_p) + β·E V' ) ],   c_p = (1+r)b + w·ε·n̄·1{p=work} − b'.
```

The outer `max_p` is `choose`; its continuation `V_end[p]` is the consumption-savings value
under participation `p`. Preferences are separable `u(c) − v(n)`, so the work disutility is a
flat reward `−v(n̄)` in the participation argmax and the savings stage carries plain CRRA `u(c)`.

**Why the persistent `:participation` axis is still a within-period choice.** The argmax
reward is independent of the *incoming* participation state (the two columns of
`part_reward` are identical, and the continuation does not depend on last period's
participation either). So every cell re-selects the same optimal participation regardless of
its prior value — the choice is genuinely re-made each period. Keeping the axis (rather than
collapsing it with a `ForgetfulSumStage`) is harmless and lets the chosen participation be
read directly as an end-of-period moment (`∫ 1{work} dΛ`).

## Why NOT the Rogerson lottery via `MixingStage` (sub-approach (b))

The catalog also lists the convexifying employment lottery as a `MixingStage` blending a
work-kernel `K_work` and non-work-kernel `K_not` (`K_θ = θ·K_work + (1−θ)·K_not`). This does
**not** compose faithfully here, for two compounding reasons:

1. **The work/not difference is not a participation-axis transition.** `MixingStage` blends
   two row-stochastic transitions *on a single axis* at a convex cost. But "work" vs "not"
   differs in the **budget** — a move on the *wealth* axis (`+w·ε·n̄` of cash) plus a flat
   utility cost — not in a transition on the participation axis. Encoding the work/not
   wealth consequence as a participation-axis kernel is exactly the cross-axis coupling the
   auxiliary-axis pattern (a) exists to handle; `MixingStage`'s single-axis `c(θ)` blend
   cannot express it.

2. **The Rogerson lottery is valued linearly in θ, with no convex cost.** The household
   values the lottery as `θ·V_work + (1−θ)·V_not` — linear in `θ`. `MixingStage`'s closed
   form `V = b + c*(a−b)` is the conjugate of a *convex* cost `c(θ)`; with `c ≡ 0` the
   conjugate is `max(a, b)`, i.e. the **corner** — which is precisely the discrete choice (a).
   So even setting the budget issue aside, the faithful Rogerson individual problem reduces
   to the discrete `ArgmaxStage`. The lottery's convexification is a representative-agent /
   aggregate device; the incomplete-markets object (Chang–Kim) is the discrete choice, with
   the participation **rate** emerging from the cross-section — which is exactly what (a)
   delivers.

So (a) is built and (b) is recorded as a non-fit: `MixingStage` would need the blend to live
on the wealth/budget margin (it does not) and a convex cost the Rogerson lottery does not have.

## Run

```
julia --startup-file=no --project=. examples/indivisible_labor/steady_state.jl
```

Partial equilibrium at fixed `(r, w)` (~9 s, CPU, `N_w = 150`). Representative output:

```
  r                 = 0.0300
  w                 = 1.0000
  participation rate = 0.6960
  mean hours         = 0.6960
  mean wealth        = 3.5928
  ΣΛ (mass)          = 1.000000
  V finite           = true
  participation rate by productivity state ε:
    ε = 0.60  →  P(work) = 0.3062
    ε = 1.00  →  P(work) = 0.7868
    ε = 1.40  →  P(work) = 0.9951
  participation by wealth tercile: low = 0.7576, mid = 0.6548, high = 0.6731
```

V is finite everywhere, Λ is a conserved stationary distribution (mass 1). The **extensive
margin responds to productivity**: participation rises sharply with `ε` (0.31 → 0.79 → 0.995)
— productive agents work, low-`ε` agents mostly opt out. The wealth gradient shows the
reservation-wealth (income) effect from low to mid terciles (0.76 → 0.65); the high tercile
ticks back up because high-`ε` agents (who almost always work) accumulate the most wealth, so
the unconditional wealth gradient mixes the income effect with this composition.
