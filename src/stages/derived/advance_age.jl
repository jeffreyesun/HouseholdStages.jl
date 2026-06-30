# Advance age — deterministic OLG age roll, sugar over a MarkovStage with a
# shift-by-one matrix. Newborn entry is the consumer's outer-loop job, not this stage.

"""
Shift-by-one transition on `n` age levels (`T[a, a+1] = 1`). `absorb_top` sets
`T[n,n] = 1` (oldest cohort absorbing, mass conserved); otherwise terminal mass
rolls off — the OLG death event.
"""
function _age_shift_matrix(n::Integer; absorb_top::Bool=true)
    T = zeros(Float64, n, n)
    for a in 1:(n - 1)
        T[a, a + 1] = 1.0
    end
    absorb_top && (T[n, n] = 1.0)
    return T
end

"""
Deterministic age-advance stage — a [`MarkovStage`](@ref) with a shift-by-one
transition on the age `axis`, so each household ages one period. `absorb_top`
(default `true`) makes the oldest cohort absorbing and conserves mass; `false`
lets terminal mass roll off, the OLG death event.
"""
AdvanceAgeStage(layout::GriddedLayout; axis::Symbol=:age, absorb_top::Bool=true) =
    MarkovStage(layout; axis=axis,
                transition_matrix=_age_shift_matrix(axissize(layout.axes[axis_position(layout, axis)]);
                                                    absorb_top=absorb_top))
