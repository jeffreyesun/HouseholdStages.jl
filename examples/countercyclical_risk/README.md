# Countercyclical income risk (Storesletten-Telmer-Yaron 2004)

Idiosyncratic earnings risk is **higher in recessions**: the variance of
the income shock rises when the aggregate state is bad (StY 2004). Here the
income transition matrix is a function of the **aggregate state** `env.z`.

## The chain

    MarkovStage(:income; transition_matrix = (; env) -> T(env.z))
        ∘ IncomeStage ∘ ConsumptionSavingsStage

The income `MarkovStage` is handed an **env-closure** rather than a constant
matrix. The kernel re-seats its transition whenever `env` changes (the §5.3
static-refill contract: an env-dependent field is refilled on the first
`backward!` after any env change). This is a first-class `MarkovStage`
capability — no new machinery.

## The point: the kernel re-seats T when env changes

In a steady state `z` is fixed, so we solve **twice on the same `hh`
object** — once at `z = :boom`, once at `z = :recession` — and compare the
stationary wealth distributions. The income grid and persistence ρ are held
fixed across the two solves; only the innovation std σ(z) of the offline
Tauchen fill changes (σ_recession = 2·σ_boom). That the package returns two
different stationary distributions is the demonstration that the env-closure
transition genuinely re-seats.

## Run

    julia --startup-file=no --project=. examples/countercyclical_risk/steady_state.jl

Fixed-r; two solves.

## Result

    z = boom       : ΣΛ = 1.000000, A_mean = 3.49
    z = recession  : ΣΛ = 1.000000, A_mean = 4.82
    ΔA (recession − boom) = +1.33  (38% more self-insurance wealth in recession)

Higher idiosyncratic risk in recession ⇒ stronger precautionary motive ⇒
more self-insurance wealth, at unchanged persistence — the StY mechanism.
