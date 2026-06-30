# AR(1) income via Rouwenhorst — offline discretization

Standard incomplete-markets self-insurance with a **continuous log-AR(1)**
income process

    log y_t = ρ · log y_{t-1} + ε_t,   ε_t ~ N(0, σ_ε²)

discretized into an N-state Markov chain.

## The chain

    MarkovStage(:income; transition_matrix = T) ∘ IncomeStage ∘ ConsumptionSavingsStage

Byte-identical to the Aiyagari/Bewley spine. Built purely by composing
existing exported stages.

## The point: offline discretization vs. in-package Markov

The AR(1)-to-Markov discretization is a **plain offline routine**
(`rouwenhorst` / `tauchen` in `model.jl`) — it runs once, in user space,
and returns a grid `y_grid` and a constant row-stochastic matrix `T`. It
is **not a stage**: it never touches a layout, kernel, or value function.

The package only ever sees the finished constant matrix, handed to a plain
`MarkovStage(:income; transition_matrix = T)`. From the package's point of
view this is indistinguishable from Aiyagari's hand-written 3×3 `P_y`. "AR(1)
income" is a calibration choice resolved entirely *before* the household
block is built — `MarkovStage` is agnostic about where its matrix came from.
`rouwenhorst` and `tauchen` are interchangeable; the package never sees the
choice.

## Run

    julia --startup-file=no --project=. examples/income_ar1/steady_state.jl

Fixed-r single solve (`r = 0.03 < 1/β − 1 ≈ 0.0417`).

## Result

7-state Rouwenhorst (ρ = 0.95, σ_ε = 0.20):

    income grid : [0.17, 0.287, 0.484, 0.816, 1.377, 2.324, 3.92]
    ΣΛ          = 1.000000
    A_mean      = 15.48
