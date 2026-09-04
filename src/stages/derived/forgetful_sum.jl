"""
Marginalise `axis` away — a [`MarkovStage`](@ref) onto a single level of `axis`: forward sums
`Λ_start` along it, backward broadcasts `V_end` over it, and the end layout has `axis` resized to 1.
"""
ForgetfulSumStage(start_layout::GriddedLayout; axis::Symbol) =
    MarkovStage(start_layout, resize_axis(start_layout, axis, 1); axis = axis,
                transition_matrix = ones(Float64, _axis_size(start_layout, axis), 1))
