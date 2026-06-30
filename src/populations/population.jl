# Populations — representations of a distribution over the state space.
#
# THE LOAD-BEARING ABSTRACTION (PHASE2_PLAN §6a): a stage's kernel is a LINEAR
# OPERATOR on a distribution. A population is a *representation* of a distribution,
# not "a different kind of thing." Applying a kernel to a population is the SAME
# linear operator acting on a distribution — only the `forward!` method realising it
# differs by the representation. The histogram `GriddedPopulation` is the on-grid
# representation (one mass per layout cell); the swarm `DynamicGridPopulation` (a
# cloud of off-grid agents) is a second representation of the same kind of object.
# Do NOT collapse the swarm into a bespoke agent loop — it is a distribution under
# the kernel's linear action.
#
# MASS INVARIANT: Λ need NOT sum to 1. The mass-conserving stages (Markov, argmax,
# logit choice, forgetful-sum, deterministic-continuous) conserve `Σ(Λ)`; the
# mass-CHANGING stages move it on purpose — `EntryStage` adds `Σg`, `ExogenousExit`/
# `EndogenousExit`/`LogitEndogenousExit` remove the exiters' mass, `ReproductionStage`
# scales by `s`. So a stationary Λ carries
# whatever total mass the entry/exit (etc.) balance dictates — it is NOT renormalized
# to 1. Consequently the population inner products below (`expectation`, the moments
# in `moments.jl`) are AGGREGATES `Σ f·λ`, not normalized means: a mean would have to
# divide by the actual `Σ(Λ)` (never assume it is 1). `uniform_distribution` sums to 1
# only as an iteration SEED; nothing downstream requires the post-stage Λ to.

"Supertype of distribution representations. A kernel acts on these as a linear operator."
abstract type AbstractPopulation end

"""
A distribution as a histogram of `masses` on a layout's grid cells — the on-grid
representation. `(GriddedLayout, GriddedPopulation)` `forward!` IS the kernel application
(the transition kernel acting on the histogram).
"""
struct GriddedPopulation{A<:AbstractArray} <: AbstractPopulation
    masses :: A
end

"The raw mass array backing a `GriddedPopulation`."
masses(p::GriddedPopulation) = p.masses
Base.Array(p::GriddedPopulation) = p.masses
Base.sum(p::GriddedPopulation)   = sum(p.masses)

"Wrap a raw mass array as a `GriddedPopulation`; a population passes through (auto-wrap for the forward seam)."
as_population(Λ::AbstractArray)    = GriddedPopulation(Λ)
as_population(Λ::AbstractPopulation) = Λ

"""
The uniform distribution over a gridded layout — equal mass in every cell, summing
to one. The natural `Λ`-iteration SEED (`Λ` converges fast from uniform); summing to 1
is a property of the seed only, not an invariant the iteration preserves (mass-changing
stages move `Σ(Λ)` away from 1 — see the file header).
"""
uniform_distribution(layout::GriddedLayout) =
    GriddedPopulation(fill(inv(prod(layout_size(layout))), layout_size(layout)))

"""
`⟨f, Λ⟩ = Σ f·λ` — the AGGREGATE mass-weighted inner product of a layout-shaped state
function `f` against the population `Λ`. The inner product behind moments (a moment =
⟨state-function, population⟩) and the V/Λ duality pairing. This is an aggregate, NOT a
normalized mean: it does not divide by `Σ(Λ)`, which need not be 1 (so `⟨1, Λ⟩ = Σ(Λ)`
recovers total mass). For a per-capita mean under non-unit mass, divide by `sum(Λ)`.
"""
expectation(f::AbstractArray, Λ::GriddedPopulation) = sum(f .* masses(Λ))
expectation(f::AbstractArray, Λ::AbstractArray)     = sum(f .* Λ)

# The stage → population forward seam (`forward!(stage::AbstractStage, Λ::GriddedPopulation)`) lives
# with the stage protocol in stages/abstract.jl — it needs `AbstractStage`, which is defined there,
# so keeping populations a foundational layer (loaded before stages) requires the seam to live above.
