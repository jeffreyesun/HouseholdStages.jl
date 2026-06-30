###############################################################
# Cocco–Gomes–Maenhout (2005) — life-cycle portfolio choice     #
###############################################################

# The CGM life-cycle portfolio problem: a finite-horizon household that
# saves AND chooses a risky portfolio share, with non-tradable, hump-shaped
# labor income. The economics: labor income is a bond-like implicit asset
# (human wealth). When young, human wealth dwarfs financial wealth, so the
# optimal share of TOTAL wealth in equities translates into a financial
# portfolio tilted heavily toward stocks (θ near the cap). As financial
# wealth accumulates and human wealth runs down with age, the financial
# share descends toward the Merton interior level. The age-declining risky
# share is the CGM signature.
#
# The household block is the life-cycle Aiyagari chain with a portfolio
# stage APPENDED — and nothing else. It is `replicate_age` of a `∘`-chain
# of FOUR existing library stages:
#
#     replicate_age(IncomeShock ∘ Receipt ∘ ConsumptionSavings ∘ Portfolio,
#                   N; axis = :age)
#
#   IncomeShock — `MarkovStage` on the income axis (persistent earnings risk).
#   Receipt     — `WealthChangeStage` `b ↦ b + y(age)·ε` (cash-on-hand). NOTE
#                 there is NO `(1+r)` here (contrast the plain life_cycle
#                 example): the gross return is delivered entirely by the
#                 portfolio stage, so baking it into Receipt too would
#                 double-count it.
#   ConsumptionSavings — `ConsumptionSavingsStage` picks invested wealth `b'`;
#                 `c = x − b'`.
#   Portfolio   — `MeanVarianceStage` picks the risky share `θ`, so start-of-
#                 next-period wealth is `b'·(R_f + θ·(R_k − R_f))`. The
#                 risk-free leg `R_f` plays the role of the life_cycle `(1+r)`.
#
# **No bespoke household stage is rolled here** — this is exactly the
# `examples/portfolio` chain, wrapped in the `examples/life_cycle`
# `replicate_age` finite-horizon driver.
#
# The finite-horizon solve (backward sweep threading the continuation value
# across ages, then a forward cohort simulation) is example-side outer-loop
# logic in `steady_state.jl`, identical in shape to the life_cycle driver.
# Returns and the age-earnings profile are exogenous (partial equilibrium).

using HouseholdStages


# Parameters #
#------------#

@kwdef struct CGMParams
    β :: Float64       = 0.96
    σ :: Float64       = 4.0                        # CRRA; with the premium below, the Merton
                                                    # interior share is ≈ 0.28, well under the cap,
                                                    # so the share has room to descend in old age
                                                    # as financial wealth overtakes human wealth.
    N :: Int           = 25                         # life-cycle periods (ages)
    # Persistent income state (CGM-style 3-state Markov on earnings). A tight
    # spread keeps labor income bond-like (the CGM human-wealth hedge).
    ε_grid :: Vector{Float64} = [0.85, 1.0, 1.15]
    P_ε    :: Matrix{Float64} = [0.80 0.15 0.05;
                                 0.10 0.80 0.10;
                                 0.05 0.15 0.80]
    # Hump-shaped deterministic age earnings (Gourinchas–Parker): a downward
    # quadratic in age peaking at `peak_age`, with a flat retirement
    # replacement after `retire_age`.
    peak_age   :: Int     = 15
    retire_age :: Int     = 20
    y_peak     :: Float64 = 1.0
    y_curv     :: Float64 = 0.5
    repl       :: Float64 = 0.4                     # retirement replacement of peak earnings
    # Portfolio block (mean premium ≈ 5%, excess-return variance ≈ 0.044).
    R_f     :: Float64 = 1.02                        # gross risk-free return (the implicit "1+r")
    R_risky :: Vector{Float64} = [0.86, 1.28]        # gross risky returns (mean 1.07)
    p_risky :: Vector{Float64} = [0.5, 0.5]
    shares  :: Vector{Float64} = collect(0.0:0.05:1.0)
    N_w   :: Int       = 100
    w_min :: Float64   = 0.0
    w_max :: Float64   = 60.0
    # Newborns enter working life with a financial endowment (≈ 1.5× peak
    # earnings) rather than literally zero wealth. This is both realistic and
    # necessary for the CGM share profile: at (or near) zero financial wealth
    # the CRRA continuation value is so concave that the cash-poor young
    # de-risk, whereas the CGM story is that the young hold ~100% equity because
    # their large (bond-like) human wealth substitutes for the riskless asset.
    # The package's `MeanVarianceStage` allocates over FINANCIAL wealth and does
    # not augment human capital into the portfolio-relevant wealth, so the
    # endowment keeps newborns above the concavity floor through the early ages
    # and recovers the high-young, declining-with-age share. Fully reproducing
    # CGM's flat-high young share without any endowment would need a bespoke
    # human-wealth-augmented portfolio stage — which the dogfooding rule forbids.
    w0_init :: Float64 = 1.5
end

Base.Broadcast.broadcastable(p::CGMParams) = Ref(p)

const cgm_params = CGMParams()


# Earnings profile and the ergodic newborn income distribution #
#-------------------------------------------------------------#

"""
Deterministic age-earnings `y(age)` (Gourinchas–Parker hump): a downward
quadratic in age peaking at `p.peak_age`, normalised to `p.y_peak` at the
peak and dropping toward `p.y_peak·(1−p.y_curv)` at the life endpoints; after
`p.retire_age` earnings are the flat retirement replacement `p.repl·p.y_peak`.
"""
function age_earnings(age::Integer, p = cgm_params)
    age > p.retire_age && return p.repl * p.y_peak
    span = max(p.peak_age - 1, p.N - p.peak_age)        # half-width to the farther endpoint
    drop = p.y_curv * ((age - p.peak_age) / span)^2
    return p.y_peak * (1 - drop)
end

"""
Stationary distribution of the income Markov chain `p.P_ε` — the newborn draw
over the persistent income state. Power-iterates the row-stochastic transpose.
"""
function income_stationary(p = cgm_params)
    n = length(p.ε_grid)
    π = fill(1 / n, n)
    for _ in 1:10_000
        π_next = p.P_ε' * π
        maximum(abs, π_next - π) < 1e-14 && (π = π_next; break)
        π = π_next
    end
    return π ./ sum(π)
end


# Household chain assembly #
#--------------------------#

"""
The CGM life-cycle portfolio block: `replicate_age(IncomeShock ∘ Receipt ∘
ConsumptionSavings ∘ Portfolio, N; axis = :age)` with a life-cycle
`mean_wealth` moment attached. The within-period chain is the
`examples/portfolio` chain; `replicate_age` stacks `N` age-specific copies
along an `:age` axis (which enters the layout as a size-1 singleton the
product grows to `N`). Returns the moment-attached `ProductStage`; the
backward+forward sweep that wires the age-slices together lives in
`steady_state.jl`.
"""
function cgm_household(p = cgm_params)
    layout = GriddedLayout(
        :wealth => GriddedContinuous(p.w_min, p.w_max, p.N_w; spacing = :log),
        :income => Discrete(p.ε_grid),
        :age => Discrete([1]),
    )

    shock   = MarkovStage(layout; axis = :income, transition_matrix = p.P_ε)
    receipt = WealthChangeStage(layout;                                       # cash-on-hand x = b + y(age)·ε
        wealth_post = (; wealth, income, env) -> wealth + env.y * income)        # NO (1+r): the portfolio stage carries the return
    savings = ConsumptionSavingsStage(layout;
        β       = p.β,
        utility = (cell, c; env) -> u_crra(c, Val(p.σ)),
    )
    portfolio = MeanVarianceStage(layout;                                     # next wealth b'·(R_f + θ·(R_k − R_f))
        shares = p.shares, risk_free = p.R_f, risky_returns = p.R_risky, probs = p.p_risky)

    age_chain = shock ∘ receipt ∘ savings ∘ portfolio
    hh = replicate_age(age_chain, p.N; axis = :age)
    return define_moments!(hh; mean_wealth = at_end(integrand = :wealth, reduce = sum))
end
