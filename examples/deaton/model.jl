#####################################################
# Deaton (1991) — Buffer-Stock Saving Under Liquidity Constraints #
#####################################################

# Deaton's (1991) buffer-stock saving model. An IMPATIENT household
# (`β(1+r) < 1`) faces a persistent AR(1) labour-income process and a hard
# borrowing constraint (`a ≥ 0`). Impatience means it would like to borrow
# against future income, but the constraint forbids it; the precautionary
# motive then makes it hold a small "buffer stock" of assets that it draws
# down in bad income spells and rebuilds in good ones. Assets stay low and
# the constraint binds frequently — the signature buffer-stock behaviour.
#
# The within-period problem is the canonical three-stage spine:
#
#     IncomeShock ∘ IncomeReceipt ∘ ConsumptionSavingsStage
#
# `IncomeShock` (MarkovStage, axis = :income) resolves the AR(1) income
# draw, discretized OFFLINE by Rouwenhorst (the grid + transition matrix
# are ordinary parameter pre-computation — NOT a stage). `IncomeReceipt`
# (IncomeStage) is the receipt `a ↦ (1+r) a + w·y`. `ConsumptionSavings-
# Stage` chooses next-period assets on the grid; the grid floor `a_min = 0`
# IS the hard borrowing constraint (the savings choice cannot go below the
# lowest grid node), so no separate constraint stage is needed.
#
# Single fixed-`r` solve: partial equilibrium, no market clearing.

using HouseholdStages


# Rouwenhorst discretization of an AR(1) (offline parameter pre-computation) #
#---------------------------------------------------------------------------#

"""
Rouwenhorst (1995) discretization of the AR(1) `z' = ρ z + ε`,
`ε ~ N(0, σ_ε²)`, onto `N` states. Returns `(z, P)`: the symmetric state
grid `z` (spanning `±√(N−1)·σ_z`, `σ_z² = σ_ε²/(1−ρ²)`) and the row-
stochastic transition matrix `P`. Built recursively from the 2-state kernel
`[[p 1−p];[1−p p]]` with `p = (1+ρ)/2`. Pure offline pre-computation of a
grid and a matrix — not a stage.
"""
function rouwenhorst(N::Int, ρ::Float64, σ_ε::Float64)
    p = (1 + ρ) / 2
    P = [p (1 - p); (1 - p) p]
    for n in 3:N
        Q = zeros(n, n)
        Q[1:n-1, 1:n-1] .+= p .* P
        Q[1:n-1, 2:n]   .+= (1 - p) .* P
        Q[2:n,   1:n-1] .+= (1 - p) .* P
        Q[2:n,   2:n]   .+= p .* P
        Q[2:n-1, :]    ./= 2          # interior rows double-counted; renormalize
        P = Q
    end
    σ_z = σ_ε / sqrt(1 - ρ^2)
    ψ   = sqrt(N - 1) * σ_z
    z   = collect(range(-ψ, ψ; length = N))
    return z, P
end


# Parameters #
#------------#

@kwdef struct DeatonParams
    β   :: Float64 = 0.95
    σ   :: Float64 = 2.0
    r   :: Float64 = 0.03         # β(1+r) = 0.9785 < 1  ⇒  impatient (buffer-stock)
    w   :: Float64 = 1.0
    # AR(1) log-income process, discretized by Rouwenhorst below.
    ρ_y :: Float64 = 0.90
    σ_ε :: Float64 = 0.10
    N_y :: Int     = 7
    N_a   :: Int     = 250
    a_min :: Float64 = 0.0        # hard borrowing constraint = grid floor
    a_max :: Float64 = 40.0
end

Base.Broadcast.broadcastable(p::DeatonParams) = Ref(p)

const deaton_params = DeatonParams()

"""
Income grid and transition for `DeatonParams`: Rouwenhorst the AR(1) in
logs, exponentiate, and recentre so `exp(z)` has mean ≈ 1 (subtract the
log-normal correction `σ_z²/2`). Returns `(y_grid, P_y)`.
"""
function deaton_income(p = deaton_params)
    z, P = rouwenhorst(p.N_y, p.ρ_y, p.σ_ε)
    σ_z2 = p.σ_ε^2 / (1 - p.ρ_y^2)
    y    = exp.(z .- σ_z2 / 2)          # mean-one log-normal income
    return y, P
end


# Household chain assembly #
#--------------------------#

"""
Build the moment-attached Deaton buffer-stock block
`IncomeShock ∘ IncomeReceipt ∘ ConsumptionSavingsStage` with the AR(1)
income discretized offline by Rouwenhorst. Attaches `A_mean` (the buffer
stock) and `frac_constrained` (mass at the binding `a = 0` constraint).
Identical spine to Aiyagari/Bewley; the Deaton content is impatience
(`β(1+r) < 1`) plus the AR(1) income process.
"""
function deaton_household(p = deaton_params)
    y_grid, P_y = deaton_income(p)

    layout = GriddedLayout(
        :wealth => GriddedContinuous(p.a_min, p.a_max, p.N_a; spacing = :log),
        :income => Discrete(y_grid),
    )

    shock   = MarkovStage(layout; axis = :income, transition_matrix = P_y)
    receipt = IncomeStage(layout;
        wealth_post = (; wealth, income, env) -> (1 + env.r) * wealth + env.w * income,
        axis        = :wealth,
    )
    savings = ConsumptionSavingsStage(layout;
        β       = p.β,
        utility = (cell, c) -> u_crra(c, Val(p.σ)),
        axis    = :wealth,
    )

    a_floor = p.a_min
    hh = shock ∘ receipt ∘ savings
    return define_moments!(hh;
        A_mean           = at_end(integrand = :wealth, reduce = sum),
        frac_constrained = at_end(integrand = (; wealth) -> wealth <= a_floor + 1e-9 ? 1.0 : 0.0,
                                  reduce = sum),
    )
end


# Prices (plain function, no AbstractBlock) #
#-------------------------------------------#

"Env for the Deaton buffer-stock experiment at the fixed exogenous return `r`."
deaton_env(p = deaton_params) = (; r = p.r, w = p.w)
