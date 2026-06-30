#################################################
# Preference building blocks — CRRA felicity    #
#################################################
# Factored out of the example folders, which each redefined a byte-identical `u_crra`
# (see examples/PLAN.md). The `Val(σ)` form keeps the curvature in the type so the inner
# power specialises; `c ≤ 0 ↦ -Inf` enforces the consumption floor.

"""
CRRA flow utility `c^(1-σ)/(1-σ)` (`log c` at `σ = 1`), `-Inf` for `c ≤ 0`. Pass `σ` as
`Val(σ)` for a type-stable specialisation on the curvature; the bare-`σ` method wraps it.
"""
@inline _u_crra(c, ::Union{Val{1}, Val{1.0}}) = log(c)
@inline _u_crra(c, ::Val{σ}) where {σ}        = (c^(1 - σ)) / (1 - σ)
@inline u_crra(c, valσ::Val) = c <= 0 ? -Inf : _u_crra(c, valσ)
@inline u_crra(c, σ::Real)   = u_crra(c, Val(σ))
