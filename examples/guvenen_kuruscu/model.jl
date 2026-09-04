######################################################################
# Guvenen–Kuruşçu (2010) — human capital with permanent ability types  #
######################################################################

# The Guvenen–Kuruşçu (2010) "Understanding the Evolution of the U.S. Wage
# Distribution" model: a Ben-Porath human-capital life cycle in which households
# differ in a PERMANENT ability type. The household block PER TYPE is exactly
# `examples/human_capital`'s single library stage — `CapitalInvestmentStage` on `:h` — with
# NO bespoke household stage. The whole point of this Part-3 example:
#
#     Household block (per type) = CapitalInvestmentStage(:h)
#     Population                 = ⊕_type [ CapitalInvestmentStage(:h) ]
#
# Catalog chain: a direct sum over ability types of the single Ben-Porath stage.
# Each type runs the SAME stage; what differs across types is the ability profile
# `a` threaded through `env` (and, with it, the skill-price interaction). Higher-
# ability types face a lower effort cost of producing human capital, so they invest
# more and reach higher human capital — the paper's headline mechanism for WIDENING
# lifetime-earnings inequality across the distribution.
#
# Why the `⊕`-over-types is realized in the DRIVER, not as a built composite stage.
# `CapitalInvestmentStage`'s `production`/`effort_cost` are `(value; env)` closures that
# build an `(h', h)` reward reading ONLY the operative axis value + `env` — they
# cannot read a second `:type` axis. So a literal `⊕` of stages stacked on a `:type`
# group axis would need a DIFFERENT `env.a` per group slab simultaneously, which the
# single shared `env` of one `backward!`/`forward!` call cannot provide. The
# economically faithful realization is therefore driver-level: run the finite-horizon
# Ben-Porath life cycle once per ability type, each with that type's `env.a` profile,
# then STACK the per-type cross-sections weighted by type population shares. This is
# the `⊕`-over-types as cross-section aggregation — exactly the "type heterogeneity
# is realized in the driver" pattern, since each type needs its own `env.a`. The
# per-type backward/forward is the IDENTICAL pattern to `examples/human_capital`.
#
# Effort cost (same as `examples/human_capital`). New human capital `Q = a·(s·h)^γ`
# from time share `s` at ability `a` ⇒ to deliver gross investment
# `i = h' − (1−δ)h` the agent forgoes earnings on `s·h = (i/a)^{1/γ}`, i.e.
#   effort_cost(i; env) = env.R · (i / env.a)^{1/γ},  γ ∈ (0,1) ⇒ convex.
# The reward `R·h − effort_cost(h' − (1−δ)h)` is supermodular in `(h', h)`, so
# `CapitalInvestmentStage`'s divide-and-conquer monotone solve is valid. Higher ability `a`
# lowers the cost — the engine of the cross-type investment gap.

using HouseholdStages


# Parameters #
#------------#

@kwdef struct GKParams
    β       :: Float64 = 0.97                 # patience
    γ       :: Float64 = 0.7                  # Ben-Porath production curvature ∈ (0,1)
    δ       :: Float64 = 0.05                 # human-capital depreciation
    R       :: Float64 = 1.0                  # skill price (rental rate of human capital; earnings = R·h)
    N_age   :: Int     = 40                   # working-life length
    h0      :: Float64 = 2.0                  # human capital at labor-market entry
    N_h     :: Int     = 200                  # human-capital grid points
    h_min   :: Float64 = 1.0
    h_max   :: Float64 = 45.0
    # Permanent ability types. Each type has its own (a_young, a_old) ability profile
    # and a population share. Higher ability ⇒ cheaper human-capital production.
    type_names  :: NTuple{3, Symbol}  = (:low, :mid, :high)
    a_young     :: NTuple{3, Float64} = (0.22, 0.32, 0.46)   # ability to produce h, age 1, by type
    a_old       :: NTuple{3, Float64} = (0.03, 0.04, 0.06)   # ability at last working age, by type
    type_share  :: NTuple{3, Float64} = (0.30, 0.40, 0.30)   # population shares (sum to 1)
end

Base.Broadcast.broadcastable(p::GKParams) = Ref(p)

const gk_params = GKParams()

"Age-`t` ability for type `k` (`t = 1…N_age`, `k = 1…n_types`): a log-linear taper
from `a_young[k]` to `a_old[k]` — each ability type's declining ability-to-produce-
human-capital profile (Huggett–Ventura–Yaron form, indexed by Guvenen–Kuruşçu type)."
function ability_at_age(t::Integer, k::Integer, p = gk_params)
    s = (t - 1) / max(p.N_age - 1, 1)
    return exp((1 - s) * log(p.a_young[k]) + s * log(p.a_old[k]))
end


# Household chain assembly — ONE library stage (per type) #
#--------------------------------------------------------#

"""
Build the per-type Ben-Porath human-capital household block — a single library
[`CapitalInvestmentStage`](@ref) on the `:h` axis, with `mean_h = ∫ h dΛ` and
`mean_earnings = ∫ R·h dΛ` attached. No bespoke household stage; the type's ability
profile is threaded through `env` by the finite-horizon driver, which calls this
builder once per ability type. The `⊕`-over-types is driver-level aggregation (the
per-type cross-sections, stacked by population share).
"""
function gk_household(p = gk_params)
    layout = GriddedLayout(
        :h => GriddedContinuous(p.h_min, p.h_max, p.N_h),
    )

    invest = CapitalInvestmentStage(layout;
        β               = p.β,
        depreciation    = p.δ,
        production      = (h; env) -> env.R * h,
        effort_cost     = (i; env) -> i <= 0 ? 0.0 : env.R * (i / env.a)^(1 / p.γ))
        # defaults: (; axis = :h)

    return define_moments!(invest;
        mean_h        = at_end(integrand = :h, reduce = sum),
        mean_earnings = at_end(integrand = (; h, env) -> env.R * h, reduce = sum))
end
