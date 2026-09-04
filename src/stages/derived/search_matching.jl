"""
Job search on a two-level `axis` (1 = unemployed, 2 = employed): separation at rate `separation`,
then the unemployed choose a job-finding probability `p` at cost `κ·((1−p)log(1−p) + p)`, for
`κ = χ/(A·θ)` at `χ = effort_cost_scale`, `A = matching_efficiency` and `θ = tightness` — a
[`MixingStage`](@ref) lottery toward employment with closed form
`p*(y) = clamp(1 − exp(−y·A·θ/χ), 0, 1)` in the value gain `y` from employment. `separation` and
`tightness` are numbers or `(; env)` closures, `tightness` defaulting to `env.θ`; `policy` on the
returned chain gives `p*` per cell.
"""
function SearchMatchingStage(layout::GriddedLayout; axis::Symbol=:emp,
                             separation, effort_cost_scale, matching_efficiency,
                             tightness=(; env) -> env.θ)
    @assert _axis_size(layout, axis) == 2
    χ  = Float64(effort_cost_scale)
    A  = Float64(matching_efficiency)
    θt = tightness isa Real ? (let v = Float64(tightness); (; env) -> v end) : tightness

    cost   = (p; env) -> (κ = χ / (A * θt(; env)); p >= 1 ? κ : κ * ((1 - p) * log1p(-p) + p))
    policy = (y; env) -> clamp(1 - exp(-y * A * θt(; env) / χ), 0.0, 1.0)
    matching = MixingStage(layout; axis, K_A = [0.0 1.0; 0.0 1.0],   # search succeeds: jump to employed
                           K_B = [1.0 0.0; 0.0 1.0], cost, policy)   # fails: stay (employed rows equal ⇒ degenerate there)

    sep = separation isa Real ? (δ = Float64(separation); [1.0 0.0; δ 1.0-δ]) :
          (; env) -> (δ = separation(; env); [1.0 0.0; δ 1.0-δ])
    return MarkovStage(layout; axis, transition_matrix = sep) ∘ matching
end
