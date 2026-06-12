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
to one. The natural `Λ`-iteration seed (`Λ` converges fast from uniform).
"""
uniform_distribution(layout::GriddedLayout) =
    GriddedPopulation(fill(inv(prod(layout_size(layout))), layout_size(layout)))

"""
`⟨f, Λ⟩` — the mass-weighted expectation of a layout-shaped state function `f` under
the population `Λ`. The inner product behind moments (a moment = ⟨state-function,
population⟩) and the V/Λ duality pairing.
"""
expectation(f::AbstractArray, Λ::GriddedPopulation) = sum(f .* masses(Λ))
expectation(f::AbstractArray, Λ::AbstractArray)     = sum(f .* Λ)

# Forward seam — apply the kernel (the linear operator) to the population. Unwrapping
# to the raw masses, applying the stage's existing `forward!`, and rewrapping keeps
# the "kernel acts on a distribution representation" framing literal: a
# `GriddedPopulation` flows through unchanged in kind, only its masses move.
forward!(stage::AbstractStage, Λ::GriddedPopulation) =
    GriddedPopulation(forward!(stage, masses(Λ)))
