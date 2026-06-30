######################################################################
# Manuelli–Seshadri (2014) — multi-stage human-capital production     #
######################################################################

# Human capital is produced over TWO life phases — a full-time SCHOOLING phase
# followed by an ON-THE-JOB phase — using the SAME Ben-Porath technology in both,
# only with phase-specific ability and earnings availability. The household block
# is therefore ONE existing library stage:
#
#     Household block = CapitalInvestmentStage(:h)
#
# exactly as in `examples/human_capital`. The Manuelli–Seshadri content is NOT a
# new within-period primitive; it is DRIVER logic — a phase switch threaded through
# `env` by the finite-horizon backward/forward sweep:
#
#   Schooling phase (ages 1 … T_school):  no labour earnings (`earn = 0`), HIGH
#       learning ability `a_school`. With no opportunity cost of working and a long
#       horizon, the agent devotes essentially all resources to producing human
#       capital — the full-time-schooling CORNER. The corner is produced by the
#       phase env (earnings off, ability high), NOT by any new stage.
#
#   Work phase (ages T_school+1 … N_age):  labour earnings `earn·h`, ability that
#       tapers with age (`a_work_young → a_work_old`). Investment is now interior
#       and declining — the agent trades current earnings against further human
#       capital — so the on-the-job profile flattens into the usual hump.
#
# The Ben-Porath reward is `production(h; env) − effort_cost(i; env)` with
# `i = h' − (1−δ)h` the gross investment. Earnings flow `production = earn·h`
# (`earn = 0` in school). The convex cost `effort_cost(i) = price·(i/a)^{1/γ}` is
# the inverted production function (`Q = a·input^γ`), cheaper at high ability `a` —
# the engine of the schooling burst. With `γ ∈ (0,1)` the exponent `1/γ > 1` is
# convex, so `production − effort_cost` is supermodular in `(h', h)` and
# `CapitalInvestmentStage`'s `:divide_conquer` monotone solve is valid.
#
# Reuse note: Caucutt–Lochner, Lee–Seshadri, and Daruich are the SAME household
# block — `CapitalInvestmentStage(:h)` — with MORE phases (a longer `env_at(t)` schedule)
# and, for Daruich, an extra public-transfer term in the budget. None needs a new
# stage; they are longer driver schedules over this exact block. See `README.md`.

using HouseholdStages


# Parameters #
#------------#

@kwdef struct ManuelliSeshadriParams
    β            :: Float64 = 0.96               # patience
    γ            :: Float64 = 0.55               # Ben-Porath production curvature ∈ (0,1)
    δ            :: Float64 = 0.04               # human-capital depreciation
    wage         :: Float64 = 1.0                # rental rate of human capital while WORKING
    # Schooling phase
    T_school     :: Int     = 8                  # number of full-time-schooling ages
    a_school     :: Float64 = 0.45               # learning ability in school (high)
    price_school :: Float64 = 0.55               # marginal cost of producing HC in school (cheap)
    # Work phase
    a_work_young :: Float64 = 0.32               # ability at the first working age
    a_work_old   :: Float64 = 0.04               # ability at the last working age (tapers ⇒ hump)
    price_work   :: Float64 = 1.0                # marginal cost of HC production on the job
    # Horizon & grid
    N_age        :: Int     = 40                 # total life length (schooling + work)
    h0           :: Float64 = 2.0                # human capital at birth
    N_h          :: Int     = 240                # human-capital grid points
    h_min        :: Float64 = 1.0
    h_max        :: Float64 = 45.0
end

Base.Broadcast.broadcastable(p::ManuelliSeshadriParams) = Ref(p)

const manuelli_seshadri_params = ManuelliSeshadriParams()

"""
Phase-specific environment at age `t` — the Manuelli–Seshadri schooling→work
switch threaded through `env`. Schooling ages (`t ≤ T_school`) carry no earnings
(`earn = 0`) and high learning ability `a_school` at a cheap production price;
working ages carry earnings and a log-linearly tapering ability. Returns
`(; earn, a, price)`.
"""
function env_at_age(t::Integer, p = manuelli_seshadri_params)
    if t <= p.T_school
        return (; earn = 0.0, a = p.a_school, price = p.price_school)
    else
        s = (t - p.T_school - 1) / max(p.N_age - p.T_school - 1, 1)
        a = exp((1 - s) * log(p.a_work_young) + s * log(p.a_work_old))
        return (; earn = p.wage, a = a, price = p.price_work)
    end
end


# Household chain assembly — ONE library stage #
#----------------------------------------------#

"""
Build the Manuelli–Seshadri human-capital household block — a single library
[`CapitalInvestmentStage`](@ref) on the `:h` axis. Earnings flow `production = earn·h`
(zero in school), convex effort cost `price·(i/a)^{1/γ}` on gross investment. No
bespoke household stage; the schooling→work phase is threaded through `env` by the
finite-horizon driver. `mean_h` and `mean_earnings` moments attached.
"""
function manuelli_seshadri_household(p = manuelli_seshadri_params)
    layout = GriddedLayout(
        :h => GriddedContinuous(p.h_min, p.h_max, p.N_h),
    )

    invest = CapitalInvestmentStage(layout;
        β            = p.β,
        depreciation = p.δ,
        production   = (h; env) -> env.earn * h,
        effort_cost  = (i; env) -> i <= 0 ? 0.0 : env.price * (i / env.a)^(1 / p.γ))
        # defaults: (; axis = :h, monotone_search = :divide_conquer, assume_monotone = false)

    return define_moments!(invest;
        mean_h        = at_end(integrand = :h, reduce = sum),
        mean_earnings = at_end(integrand = (; h, env) -> env.earn * h, reduce = sum))
end
