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
    # The axis/field names are runtime symbols, so `wealth_post` rides a `DepClosure` carrying its
    # declared axes for the Sources path. Dedupe for the single-asset case (`holdings_axis ==
    # wealth_axis`) — `NamedTuple{(:wealth, :wealth)}` would throw a duplicate-field error.
    decl_axes = wealth_axis == holdings_axis ? (wealth_axis,) : (wealth_axis, holdings_axis)
    wealth_post = DepClosure(decl_axes, true) do nt
        nt[wealth_axis] + (getfield(nt.env, q_field) - getfield(nt.env, q_last_field)) * nt[holdings_axis]
    end
    return WealthChangeStage(layout; wealth_post, axis=wealth_axis)
end
