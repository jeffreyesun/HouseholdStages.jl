# Populations — representations of a distribution over the state space. `GriddedPopulation` holds one
# mass per layout cell. `Λ` need not sum to 1 and is never renormalized.
#TODO An off-grid representation (a swarm of agents) has no type yet; it belongs here, under the
#     kernel's linear action, not in a bespoke agent loop.

"Supertype of distribution representations. A kernel acts on these as a linear operator."
abstract type AbstractPopulation end

"A distribution as a histogram of `masses` over a layout's grid cells."
struct GriddedPopulation{A<:AbstractArray} <: AbstractPopulation
    masses :: A
end

"The raw mass array backing a `GriddedPopulation`."
masses(p::GriddedPopulation) = p.masses
Base.Array(p::GriddedPopulation) = p.masses
Base.sum(p::GriddedPopulation)   = sum(p.masses)

"Wrap a raw mass array as a `GriddedPopulation`; a population passes through unchanged."
as_population(Λ::AbstractArray)    = GriddedPopulation(Λ)
as_population(Λ::AbstractPopulation) = Λ

"Equal mass in every cell of the layout, summing to one."
uniform_distribution(layout::GriddedLayout) =
    GriddedPopulation(fill(inv(prod(layout_size(layout))), layout_size(layout)))

"""
The mass-weighted sum `⟨f, Λ⟩ = Σ f·λ` of a layout-shaped state function `f` against the population
`Λ`. It does not divide by `Σ(Λ)`, so `⟨1, Λ⟩` gives total mass rather than one.
"""
expectation(f::AbstractArray, Λ::GriddedPopulation) = sum(f .* masses(Λ))
expectation(f::AbstractArray, Λ::AbstractArray)     = sum(f .* Λ)
