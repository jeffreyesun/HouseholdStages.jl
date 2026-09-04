"""shift-by-one transition on `n` age levels (`T[a, a+1] = 1`); `absorb_top` also sets `T[n, n] = 1`."""
function _age_shift_matrix(n::Integer; absorb_top::Bool=true)
    T = zeros(Float64, n, n)
    for a in 1:(n - 1)
        T[a, a + 1] = 1.0
    end
    absorb_top && (T[n, n] = 1.0)
    return T
end

"""
Age every household by one period — a [`MarkovStage`](@ref) shifting the age `axis` up one level,
the oldest cohort held in place under `absorb_top` and dropped from the population otherwise.
Newborns are not added here.
"""
AdvanceAgeStage(layout::GriddedLayout; axis::Symbol=:age, absorb_top::Bool=true) =
    MarkovStage(layout; axis=axis,
                transition_matrix=_age_shift_matrix(axissize(layout.axes[axis_position(layout, axis)]);
                                                    absorb_top=absorb_top))
