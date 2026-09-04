"""each cell's wealth moves to `wealth_post`, a closure of the layout axes it names as keyword arguments, plus `env`."""
WealthChangeStage(start_layout::GriddedLayout, end_layout::GriddedLayout = start_layout;
                  wealth_post, axis::Symbol=:wealth) =
    DeterministicContinuousStage(start_layout, end_layout; destination=wealth_post, axis=axis)
