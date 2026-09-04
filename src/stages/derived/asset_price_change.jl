"""
Asset revaluation — a [`WealthChangeStage`](@ref) adding `(env.q − env.q_last) · holdings` to each
cell's wealth, `holdings` being its coordinate on `holdings_axis`.
"""
function AssetPriceChangeStage(start_layout::GriddedLayout,
                               end_layout::GriddedLayout = start_layout;
                               holdings_axis::Symbol,
                               wealth_axis::Symbol=:wealth,
                               q_field::Symbol=:q,
                               q_last_field::Symbol=:q_last)
    wealth_post = DepClosure((wealth_axis, holdings_axis), true) do nt
        nt[wealth_axis] + (getfield(nt.env, q_field) - getfield(nt.env, q_last_field)) * nt[holdings_axis]
    end
    return WealthChangeStage(start_layout, end_layout; wealth_post, axis=wealth_axis)
end
