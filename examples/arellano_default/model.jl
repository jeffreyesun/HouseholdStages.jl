##############################################################
# Arellano (2008) sovereign default with priced bond spreads #
##############################################################

# A small open economy with a stochastic endowment `y` that issues one-period,
# non-contingent discount bonds `a` (`a < 0` is debt). Each period in good
# standing it chooses REPAY vs DEFAULT; defaulting discharges the debt but
# triggers exclusion — autarky with the Arellano asymmetric output cost
# `min(y, ŷ)` — from which it is readmitted with probability `ψ`. The piece that distinguishes this from
# `examples/default` is the BOND PRICE: risk-neutral foreign lenders price the
# bond at
#
#     q(a', y) = (1/(1+rf)) · (1 − E[default(a', y') | y]),
#
# so the budget when issuing debt is `c = x − q(a', y)·a'` — the price depends on
# the CHOSEN next assets `a'` AND the current income `y`. Default risk raises
# spreads on high debt and endogenously caps borrowing. `q` is itself a fixed
# point: it depends on next period's default policy, which depends on `q`. That
# outer fixed point is rolled in `steady_state.jl`; the WITHIN-period block here
# is library stages only.
#
# The block is the `examples/default` chain with ONE change at the savings step:
#
#     IncomeShock ∘ DefaultChoice ∘ DebtReset ∘ Receipt ∘ Savings ∘ Readmission
#
# `IncomeShock`   — `MarkovStage` on `:income` (the endowment AR(1), Rouwenhorst).
# `DefaultChoice` — `DefaultStage`: gated repay/default argmax on `:status`
#                   (1 = good standing, 2 = excluded). Good standing chooses
#                   repay (stay 1, score 0) or default (→2, score −χ); an
#                   excluded agent is gated to stay excluded.
# `DebtReset`     — `WealthChangeStage`: an excluded/defaulting agent carries
#                   zero assets (debt discharged); a repayer keeps `a`.
# `Receipt`       — `WealthChangeStage`: cash-on-hand. Good standing matures its
#                   bonds at FACE value (one-period discount bonds — no `(1+r)`,
#                   the return is in the price `q < 1`): `x = a + y`. Excluded:
#                   `x = min(y, ŷ)` (the Arellano output cost, debt discharged).
# `Savings`       — a raw `ArgmaxStage` on `:wealth` whose reward is a
#                   destination-AND-income-priced budget `c = x − q(a', y)·a'`.
#                   `ConsumptionSavingsStage` cannot express this: it sees the
#                   chosen `a'` only through `c = before − after` at UNIT price,
#                   so it cannot apply `q(a', y)·a'`. We build the
#                   `M[after, before]` reward matrix ourselves (a plain CLOSURE,
#                   no reward struct) and compose `∘ TimeDiscountingStage(β)` for
#                   the discount the argmax carries no `β` of its own.
# `Readmission`   — `MarkovStage` on `:status`: an excluded agent regains good
#                   standing w.p. `ψ` (geometric exclusion spell). This
#                   persistent cost — not the one-shot penalty — is what deters
#                   always-default and delivers a non-degenerate default rate.
#
# The price schedule `q` rides in `env.q` (an `(N_a, N_y)` matrix over the
# `a'`-grid × current income); the savings reward closure reads it. Everything
# else (the `q ⇄ household` fixed-point loop, the default-policy read, the
# spread report) is custom driver code in `steady_state.jl`.

using HouseholdStages
using LinearAlgebra: dot


# Parameters #
#------------#

@kwdef struct ArellanoParams
    β  :: Float64 = 0.915                # impatience (β < 1/(1+rf)) drives the debt motive
    σ  :: Float64 = 2.0                  # CRRA
    rf :: Float64 = 0.017                # risk-free rate of the foreign lenders
    ŷ_frac :: Float64 = 0.95             # Arellano asymmetric default cost: excluded output = min(y, ŷ), ŷ = ŷ_frac·E[y]
    χ  :: Float64 = 0.0                  # extra flat per-period utility cost of default (0 ⇒ pure Arellano)
    ψ  :: Float64 = 0.28                 # readmission prob (mean exclusion spell 1/ψ ≈ 3.6 periods)

    ρ_y  :: Float64 = 0.85               # endowment AR(1) persistence (in logs)
    σ_y  :: Float64 = 0.10               # endowment unconditional std (in logs)
    N_y  :: Int     = 5                  # income states (Rouwenhorst)

    N_a   :: Int     = 120               # asset grid points
    a_min :: Float64 = -0.6              # borrowing limit (debt is a < 0; deep enough for an interior default region)
    a_max :: Float64 = 3.0               # top (large enough to hold cash-on-hand x = a + y in-grid)
end

Base.Broadcast.broadcastable(p::ArellanoParams) = Ref(p)

const arellano_params = ArellanoParams()


# Endowment process — Rouwenhorst discretization of the log-AR(1) #
#----------------------------------------------------------------#

"""
Rouwenhorst `(y_grid, P)` for a log-AR(1) endowment with persistence `ρ` and unconditional std `σ`
on `N` states. Returns the LEVEL grid `y = exp(state)` normalized to a stationary mean of 1, and the
row-stochastic transition `P[from, to]`. Rouwenhorst (vs Tauchen) matches the unconditional moments
exactly and is the right tool for persistent processes.
"""
function rouwenhorst(N::Int, ρ::Real, σ::Real)
    p = (1 + ρ) / 2
    Θ = [p (1 - p); (1 - p) p]
    for n in 3:N
        Z = zeros(n - 1)
        Θ = p     * [Θ Z; Z' 0] +
            (1-p) * [Z Θ; 0 Z'] +
            (1-p) * [Z' 0; Θ Z] +
            p     * [0 Z'; Z Θ]
        Θ[2:end-1, :] ./= 2                               # interior rows double-count; halve them
    end
    spread = σ * sqrt(N - 1)                              # the Rouwenhorst grid half-width in logs
    states = range(-spread, spread; length = N)
    y      = exp.(collect(states))
    π      = stationary_distribution(Θ)
    y    ./= dot(π, y)                                    # normalize to a stationary mean of 1
    return y, Θ
end

"""
The stationary distribution of a row-stochastic matrix `P` by power iteration on `Pᵀ`.
"""
function stationary_distribution(P::AbstractMatrix; tol = 1e-12, maxiter = 10_000)
    n = size(P, 1)
    π = fill(1 / n, n)
    for _ in 1:maxiter
        π_new = P' * π
        maximum(abs.(π_new .- π)) < tol && return π_new
        π = π_new
    end
    return π
end


# Asset grid — dense in the debt/default region, sparse above #
#------------------------------------------------------------#

"""
A sorted asset grid concentrated in the debt region `[a_min, 0]` (where the borrower's mass and the
whole default decision live) and coarse over the saving region `(0, a_max]`. `0` is ON the grid (its
index is returned), which the excluded-autarky `a' = 0` branch needs. Returns `(grid, idx0)`.
"""
function asset_grid(p::ArellanoParams)
    n_neg = max(2, round(Int, 0.6 * p.N_a))
    neg   = collect(range(p.a_min, 0.0; length = n_neg))          # ends exactly at 0
    pos   = collect(range(0.0, p.a_max; length = p.N_a - n_neg + 1))[2:end]
    grid  = vcat(neg, pos)
    return grid, n_neg                                            # idx0 = n_neg (grid[n_neg] == 0)
end


# Savings reward — the destination-AND-income-priced budget #
#----------------------------------------------------------#

const INFEASIBLE = -1e10                  # finite gate: brute argmax asserts all-finite V_start

"""
The `(after, before)` savings reward face for one `(income, status)` stratum. `before` indexes
cash-on-hand `x = g[before]`, `after` indexes next assets `a' = g[after]`. Good standing prices the
bond at `q`: `c = x − q(a',income)·a'`, with `a' ≥ 0` priced risk-free (`1/(1+rf)`) and `a' < 0`
priced off the schedule `qcol`. An EXCLUDED agent is in autarky — only `a' = 0` (index `idx0`) is
feasible and `c = x`. Infeasible `c ≤ 0` cells get a large finite penalty (not `−Inf`, so the brute
argmax's all-finite assertion holds for the never-visited negative-`x` corner).
"""
function savings_reward_face(g::Vector{Float64}, qcol, σ::Float64, rf::Float64,
                             excluded::Bool, idx0::Int)
    n = length(g)
    M = fill(INFEASIBLE, n, n)
    if excluded
        @inbounds for before in 1:n
            x = g[before]
            x > 0 && (M[idx0, before] = u_crra(x, Val(σ)))
        end
    else
        @inbounds for after in 1:n
            a_next = g[after]
            q      = a_next ≥ 0 ? 1 / (1 + rf) : qcol[after]      # risk-free on savings, schedule on debt
            net    = q * a_next
            for before in 1:n
                c = g[before] - net
                c > 0 && (M[after, before] = u_crra(c, Val(σ)))
            end
        end
    end
    return M
end


# Household block assembly #
#--------------------------#

"""
Build the Arellano household block
`IncomeShock ∘ DefaultChoice ∘ DebtReset ∘ Receipt ∘ Savings ∘ Readmission`, returning the
`define_moments!`-wrapped chain plus the asset grid and the index of `a' = 0`. Library stages only:
the ONE non-`examples/default` piece is the savings step, a raw `ArgmaxStage(:wealth)` whose reward
closure builds the destination-and-income-priced `M[after, before]` (a plain closure, no reward
struct) composed with `TimeDiscountingStage(β)` for the discount. The price schedule arrives via
`env.q`. Moments: `mean_assets = ∫ a dΛ` and `excluded_rate = ∫ 1{status = excluded} dΛ`.
"""
function arellano_household(p = arellano_params)
    y_grid, P_y = rouwenhorst(p.N_y, p.ρ_y, p.σ_y)
    g, idx0     = asset_grid(p)

    layout = GriddedLayout(
        :wealth => GriddedContinuous(g),
        :income => Discrete(y_grid),
        :status => Discrete([1, 2]),               # 1 = good standing, 2 = excluded / in default
    )

    shock = MarkovStage(layout; axis = :income, transition_matrix = P_y)

    # DefaultChoice: good standing chooses repay (stay 1, score 0) or default (→2, score −χ); an
    # excluded agent is gated to stay excluded. The heavy punishment (output haircut, discharged
    # debt, multi-period spell) is carried by the following branches and the readmission stage.
    excl_gate(before, after) = before == 1 ? true : after == 2
    default = DefaultStage(layout; default_penalty = p.χ, avail = excl_gate)

    # DebtReset: an excluded / defaulting agent carries zero assets (debt discharged); a repayer
    # keeps `a`.
    reset = WealthChangeStage(layout; axis = :wealth,
        wealth_post = (; status, wealth) -> status == 2 ? 0.0 : wealth)

    # Receipt: cash-on-hand. One-period DISCOUNT bonds mature at face value, so good standing has
    # `x = a + y` (no `(1+r)`; the return lives in `q < 1`); an excluded agent gets the Arellano
    # asymmetric default cost `min(y, ŷ)` and no debt — output is capped at `ŷ` (a big loss in good
    # states, none in bad), the device that makes default cheap when poor ⇒ countercyclical default.
    receipt = WealthChangeStage(layout; axis = :wealth,
        wealth_post = (; status, income, wealth, env) ->
            status == 2 ? min(income, env.ŷ)
                        : wealth + income)

    # Savings: the destination-AND-income-priced argmax. The reward closure declares its deps
    # (`income`, `status`) and `env` as kwargs and returns the `(after, before)` face for each
    # stratum; `env.q` is the `(N_a, N_y)` price schedule over `(a', current income)`.
    savings = ArgmaxStage(layout; axis = :wealth, search = :brute,
        reward = (; income, status, env) -> begin
            iy   = findfirst(==(income), y_grid)
            qcol = view(env.q, :, iy)
            savings_reward_face(g, qcol, p.σ, env.rf, status == 2, idx0)
        end) ∘ TimeDiscountingStage(layout; β = p.β)

    # Readmission: an excluded agent (status 2) regains good standing w.p. ψ; a good-standing agent
    # stays good. The geometric exclusion spell is the persistent cost of default.
    readmit = MarkovStage(layout; axis = :status,
        transition_matrix = [1.0 0.0; p.ψ (1 - p.ψ)])

    hh = shock ∘ default ∘ reset ∘ receipt ∘ savings ∘ readmit
    hh = define_moments!(hh;
        mean_assets   = at_end(integrand = :wealth, reduce = sum),
        excluded_rate = at_end(integrand = (; status) -> status == 2 ? 1.0 : 0.0, reduce = sum),
    )
    return (; hh, grid = g, idx0, y_grid, P_y)
end


# Environment — the haircut, the risk-free rate, and the price schedule #
#----------------------------------------------------------------------#

"""
The Arellano environment at a given price schedule `q` (an `(N_a, N_y)` matrix over `(a', current
income)`): the asymmetric default-cost cap `ŷ = ŷ_frac·E[y]` (with `E[y] = 1` by the Rouwenhorst
normalization), the risk-free rate `rf`, and `q` itself (read by the savings reward closure).
"""
arellano_env(q::AbstractMatrix, p = arellano_params) = (; ŷ = p.ŷ_frac, p.rf, q)
