#################################################
# Preference building blocks — CRRA felicity    #
#################################################

"CRRA flow utility `c^(1-σ)/(1-σ)`, or `log c` at `σ = 1`; `-Inf` for `c ≤ 0`."
@inline _u_crra(c, ::Union{Val{1}, Val{1.0}}) = log(c)
@inline _u_crra(c, ::Val{σ}) where {σ}        = (c^(1 - σ)) / (1 - σ)
@inline u_crra(c, valσ::Val) = c <= 0 ? -Inf : _u_crra(c, valσ)
@inline u_crra(c, σ::Real)   = u_crra(c, Val(σ))
