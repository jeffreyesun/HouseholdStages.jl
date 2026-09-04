###########################################################
# Krueger–Mitman–Perri (2016) — Aiyagari GE with UI        #
###########################################################

# A general-equilibrium incomplete-markets economy in the Krueger–Mitman–Perri
# (Handbook 2016) tradition: an Aiyagari production economy whose idiosyncratic
# risk is EMPLOYMENT risk, with an unemployment-insurance scheme financed by a
# flat payroll tax. The household block is the canonical three-stage spine with
# the UI policy carried in the budget closure (exactly the
# `examples/unemployment_insurance` device), and the model is CLOSED in general
# equilibrium by tatonnement on aggregate capital K (exactly the
# `examples/aiyagari` outer loop).
#
#     EmploymentShock ∘ IncomeReceipt(UI) ∘ ConsumptionSavingsStage
#
# `EmploymentShock` (MarkovStage, axis = :employment) — the 2-state
#     employed/unemployed Markov draw `P_e`.
# `IncomeReceipt(UI)` (IncomeStage) — receipt
#     `a ↦ (1+r) a + [e·w(1−τ) + (1−e)·ρw]`: the employed (e = 1) earn the
#     after-tax wage `w(1−τ)`, the unemployed (e = 0) collect the UI benefit
#     `b = ρw` (replacement rate ρ of the wage). The interest rate r and wage w
#     come from the Cobb–Douglas production sector and move with K along the
#     tatonnement; the tax τ and benefit b ride in `env`.
# `ConsumptionSavingsStage` — choose next-period assets on the wealth grid;
#     implicit budget `c = a_in − a_end`, CRRA utility.
#
# UI budget balance. The employment Markov is exogenous and independent of K, so
# its stationary employment share `π_e` is a constant. Balancing the UI budget
#
#     τ · w · L  =  b · π_u  =  ρ · w · π_u ,        L = π_e   (effective labor)
#
# pins the flat tax `τ = ρ · π_u / π_e`, INDEPENDENT of the wage w (it cancels)
# and hence constant along the tatonnement. `π_e, π_u` are computed once from the
# stationary distribution of `P_e`.
#
# General equilibrium closes on K: Cobb–Douglas factor prices `r(K), w(K)` with
# effective labor `L = π_e`, and the fixed point `K = ∫ a dΛ(K)` is found by the
# damped tatonnement in `steady_state.jl` (the Aiyagari outer loop).

using HouseholdStages


# Parameters #
#------------#

@kwdef struct KMPParams
    β :: Float64       = 0.96
    σ :: Float64       = 1.5
    α :: Float64       = 0.36     # capital share
    δ :: Float64       = 0.08     # depreciation
    ρ :: Float64       = 0.40     # UI replacement rate (unemployed get ρ·w)
    # Employment axis as a 0/1 indicator: unemployed (0) / employed (1).
    e_grid :: Vector{Float64} = [0.0, 1.0]
    # Two-state employment Markov (rows = current state, order [unemployed,
    # employed]). Stationary share π_u ≈ 0.0909 (≈ 9% unemployment).
    P_e    :: Matrix{Float64} = [0.50 0.50;
                                 0.05 0.95]
    N_a   :: Int       = 250
    a_min :: Float64   = 0.0      # zero-borrowing constraint (Aiyagari φ)
    a_max :: Float64   = 100.0
end

Base.Broadcast.broadcastable(p::KMPParams) = Ref(p)

const kmp_params = KMPParams()


# Employment shares and balanced-budget tax #
#-------------------------------------------#

"""
Stationary distribution of a finite Markov matrix `P` (rows sum to 1), by
iterating `π ← π P` from uniform to convergence. Used to fix the employment
shares that pin effective labor and the balanced-budget UI tax.
"""
function stationary_distribution(P::Matrix{Float64}; tol = 1e-14, maxiter = 100_000)
    n = size(P, 1)
    π = fill(1.0 / n, n)
    for _ in 1:maxiter
        π_new = vec(π' * P)
        maximum(abs, π_new .- π) < tol && return π_new
        π = π_new
    end
    return π
end

"""
Employment shares `(π_u, π_e)` and the balanced-budget flat payroll tax
`τ = ρ · π_u / π_e` (wage-independent: the wage cancels in `τ w π_e = ρ w π_u`).
Effective labor `L = π_e`. All three are constants of the calibration, computed
once from `P_e`.
"""
function kmp_labor_market(p = kmp_params)
    π = stationary_distribution(p.P_e)         # [π_u, π_e] in e_grid order
    π_u, π_e = π[1], π[2]
    τ = p.ρ * π_u / π_e
    return (; π_u, π_e, L = π_e, τ)
end


# Household chain assembly #
#--------------------------#

"""
Build the moment-attached KMP household block
`EmploymentShock ∘ IncomeReceipt(UI) ∘ ConsumptionSavingsStage`. The receipt
closure reads the 0/1 employment indicator and maps it to NET income via the UI
policy in `env`: employed get `w(1−τ)`, unemployed get the benefit `ρ·w`.
Attaches `K_supplied = ∫ a dΛ` (aggregate capital supply, driven to the
production capital demand by the tatonnement).
"""
function kmp_household(p = kmp_params)
    layout = GriddedLayout(
        :wealth     => GriddedContinuous(p.a_min, p.a_max, p.N_a; spacing = :log),
        :employment => Discrete(p.e_grid),
    )

    shock   = MarkovStage(layout; axis = :employment, transition_matrix = p.P_e)
    receipt = IncomeStage(layout;
        wealth_post = (; wealth, employment, env) -> (1 + env.r) * wealth +
            employment * env.w * (1 - env.τ) + (1 - employment) * env.ρ * env.w,
        axis        = :wealth,
    )
    savings = ConsumptionSavingsStage(layout;
        β       = p.β,
        utility = (cell, c) -> u_crra(c, Val(p.σ)),
        axis    = :wealth,
    )

    hh = shock ∘ receipt ∘ savings
    return define_moments!(hh;
        K_supplied   = at_end(integrand = :wealth, reduce = sum),
        unemployed_a = at_end(integrand = (; wealth, employment) -> employment == 0 ? wealth : 0.0,
                              reduce = sum),
    )
end


# Production prices and env (plain functions) #
#---------------------------------------------#

"""
Cobb–Douglas factor prices at aggregate capital `K` with effective labor `L`
(the employed share). `r = α (K/L)^{α−1} − δ`, `w = (1−α)(K/L)^α`.
"""
function kmp_prices(K::Real, p = kmp_params; L = kmp_labor_market(p).L)
    (; α, δ) = p
    r = α * (K / L)^(α - 1) - δ
    w = (1 - α) * (K / L)^α
    return (; r, w)
end

"""
Env at aggregate capital `K`: the Cobb–Douglas prices `(r, w)`, the UI
replacement rate `ρ`, and the balanced-budget tax `τ`. The receipt closure reads
all four; `K` and `L` ride along for bookkeeping.
"""
function kmp_env(K::Real, p = kmp_params)
    lm = kmp_labor_market(p)
    pr = kmp_prices(K, p; L = lm.L)
    return (; K, pr..., ρ = p.ρ, τ = lm.τ, L = lm.L)
end
