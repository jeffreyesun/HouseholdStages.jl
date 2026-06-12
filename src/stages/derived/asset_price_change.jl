"""
Read field `F` of `x` with `F` in the type domain (`Val{F}`), so the access is
type-stable — the captured-`Symbol` form `getfield(x, sym)` infers as `Any`.
"""
@inline _field(x, ::Val{F}) where {F} = getfield(x, F)

"""
Convenience constructor for the asset-revaluation pattern: existing
holders of an asset on `holdings_axis` gain `(env.q - env.q_last) *
cell.holdings_axis` on their wealth. Returns a plain
[`WealthChangeStage`](@ref).
"""
function AssetPriceChangeStage(layout::GriddedLayout;
                               holdings_axis::Symbol,
                               wealth_axis::Symbol=:wealth,
                               q_field::Symbol=:q,
                               q_last_field::Symbol=:q_last)
    # Capture the axis/field names as `Val`s so the closure's field reads are
    # type-stable, letting the `backward!` broadcast specialise (no per-cell box).
    wealth_post = let wa=Val(wealth_axis), ha=Val(holdings_axis),
                      qf=Val(q_field), qlf=Val(q_last_field)
        (cell; env) -> _field(cell, wa) +
                       (_field(env, qf) - _field(env, qlf)) * _field(cell, ha)
    end
    return WealthChangeStage(layout; wealth_post, wealth_axis)
end
