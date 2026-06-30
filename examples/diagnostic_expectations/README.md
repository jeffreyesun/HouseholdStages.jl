# Diagnostic expectations (Bordalo-Gennaioli-Shleifer 2018)

Households form **diagnostic** (representativeness-distorted) expectations of
income, over-weighting states that have become more likely relative to a
reference (BGS 2018). The distortion is applied to the income transition:
from a true row-stochastic `T` with stationary distribution `π`,

    T_diag[y, y'] ∝ T[y, y'] · ( T[y, y'] / π[y'] )^θ,

renormalized per row. θ > 0 is the diagnosticity; θ = 0 recovers rational
expectations.

## The chain

    MarkovStage(:income; transition_matrix = T_diag) ∘ IncomeStage ∘ ConsumptionSavingsStage

The canonical Aiyagari/Bewley spine. Built purely by composing existing
exported stages.

## The point: offline distortion, plain in-package Markov

The diagnostic tilt is a **plain offline matrix computation**
(`diagnostic_tilt` in `model.jl`) — stationary distribution of `T`, then the
likelihood-ratio reweighting, then row-renormalization. It is **not a
stage**. The household solves and simulates under the resulting **constant**
distorted matrix via an ordinary `MarkovStage`; the package never sees the
tilt, only the finished matrix — exactly as in `income_ar1`, where an offline
routine produces the matrix the package consumes.

## Run

    julia --startup-file=no --project=. examples/diagnostic_expectations/steady_state.jl

Fixed-r; two solves (θ = 0 rational, θ = 1 diagnostic).

## Result

    θ = 0.00 : ΣΛ = 1.000000, A_mean = 5.73   (rational)
    θ = 1.00 : ΣΛ = 1.000000, A_mean = 9.65   (diagnostic)
    ΔA (diagnostic − rational) = +3.92  (+68% vs rational)

The diagnostic tilt over-weights the persistent tail states, so perceived
income looks more volatile and persistent than it is — a stronger
precautionary motive and more self-insurance wealth than under rational
expectations.
