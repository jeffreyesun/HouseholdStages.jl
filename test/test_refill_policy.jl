using Test
using HouseholdStages
using HouseholdStages: reads_env

# Static refill policy + the `env_changed` flag (end-goal §5.3). There is NO per-field cache record:
# the decision is purely STATIC (`reads_env`, stored once per stage as a `Bool`) + the one runtime
# `env_changed` flag. An env-independent field is filled once at construction; an env-dependent field
# is NaN-filled at allocation and refilled each `backward!` UNLESS the caller passes
# `env_changed = false`. The hot path is the fixed-env VFI loop: first `backward!` of an episode runs
# `env_changed = true` (seats the env-dependent fields), the rest `env_changed = false` (skip the
# now-redundant refills). The CORRECTNESS risk is a stale field — a wrong ANSWER, not a crash — so
# the equivalence + stale-detection tests below are the load-bearing coverage.

@testset "refill policy — env-dependence classification" begin
    @test reads_env([1.0 2.0; 3.0 4.0]) == false              # constant matrix: fill once, ever
    @test reads_env((; env) -> [1.0 0.0; 0.0 1.0]) == true     # env-declaring closure
    @test reads_env((; income) -> [1.0 0.0; 0.0 1.0]) == false # dep-only closure, no env
    @test reads_env(FromEnv(:k)) == true                       # FromEnv
end

@testset "refill policy — stage cache classification" begin
    # Argmax: env-dependent reward ⇒ cache marked env-dependent; constant ⇒ not.
    klay   = GriddedLayout(:k => GriddedContinuous([0.0, 1.0, 2.0, 3.0]))
    a_env  = ArgmaxStage(klay; reward = (; env) -> [i == j ? Float64(env.b) : -Inf for i in 1:4, j in 1:4],
                         axis = :k)
    a_cons = ArgmaxStage(klay; reward = [i == j ? 0.0 : -Inf for i in 1:4, j in 1:4], axis = :k)
    @test a_env.cache.reward_env_dep  == true
    @test a_cons.cache.reward_env_dep == false

    # Logit: the `ε`-baked field is env-dependent iff the cost reads env OR `ε` is FromEnv.
    alay = GriddedLayout(:a => Discrete([1, 2]))
    C    = [0.0 0.5; 0.5 0.0]
    @test LogitChoiceStage(alay; axis = :a, cost_matrix = C, ε = FromEnv(:eps)).cache.cost_env_dep == true
    @test LogitChoiceStage(alay; axis = :a, cost_matrix = C, ε = 1.0).cache.cost_env_dep == false

    # Markov: `reads_env` sees THROUGH the `MappedField` to the transition source (Phase 0), so a
    # CONSTANT transition matrix is env-INDEPENDENT (its `K = Tᵀ` is seated once at construction); an
    # env-reading transition closure stays env-dependent (refilled each `backward!`).
    zlay = GriddedLayout(:z => Discrete([1, 2]))
    @test MarkovStage(zlay; axis = :z, transition_matrix = [0.9 0.1; 0.2 0.8]).cache.transition_env_dep == false
    @test MarkovStage(zlay; axis = :z,
                      transition_matrix = (; env) -> [env.a (1 - env.a); 0.2 0.8]).cache.transition_env_dep == true
end

# Run a fixed-env `backward!` loop two ways and collect the full V trajectory. `skip_after_first`
# mimics the VFI inner loop (first `backward!` true, the rest false); `false` keeps every iteration
# at `env_changed = true`. Each `backward!` returns the stage's reused buffer, so we `copy`.
function _refill_trajectory(stage, V0, env, n; skip_after_first::Bool)
    traj = Vector{typeof(V0)}()
    V = copy(V0)
    for i in 1:n
        ec = skip_after_first ? (i == 1) : true
        V  = copy(backward!(stage, V, env; env_changed = ec))
        push!(traj, V)
    end
    return traj
end

@testset "refill policy — fixed-env loop is bit-for-bit identical (Argmax)" begin
    klay   = GriddedLayout(:k => GriddedContinuous([0.0, 1.0, 2.0, 3.0]))
    reward = (; env) -> [i == j ? Float64(env.b) : -Inf for i in 1:4, j in 1:4]   # env-dependent field
    mk()   = ArgmaxStage(klay; reward = reward, axis = :k)

    env = (b = 2.0,)
    V0  = Float64[0.1, 0.2, 0.3, 0.4]
    all_true = _refill_trajectory(mk(), V0, env, 6; skip_after_first = false)
    skipped  = _refill_trajectory(mk(), V0, env, 6; skip_after_first = true)
    @test all_true == skipped               # `==`, not `≈`: skipping a redundant refill must not perturb a bit
end

@testset "refill policy — fixed-env loop is bit-for-bit identical (Logit, gains a skip)" begin
    # The logit cost field takes the `env_changed` skip like any other env-dependent field; this
    # pins that skipping it is exact for the `ε`-baked cost.
    alay = GriddedLayout(:a => Discrete([1, 2]))
    C    = [0.0 0.5; 0.5 0.0]
    mk() = LogitChoiceStage(alay; axis = :a, cost_matrix = C, ε = FromEnv(:eps))

    env = (eps = 0.7,)
    V0  = Float64[0.0, 1.0]
    all_true = _refill_trajectory(mk(), V0, env, 6; skip_after_first = false)
    skipped  = _refill_trajectory(mk(), V0, env, 6; skip_after_first = true)
    @test all_true == skipped
end

@testset "refill policy — stale-field detection (the flag actually controls refresh)" begin
    # Negative check: priming at env A then running env B with env_changed=false WRONGLY asserts env
    # unchanged, so the field stays at A's value — a different (stale) answer than the correct refresh.
    # Proves the flag drives the refill and that a mis-primed stale field would be caught.
    klay   = GriddedLayout(:k => GriddedContinuous([0.0, 1.0, 2.0, 3.0]))
    reward = (; env) -> [i == j ? Float64(env.b) : -Inf for i in 1:4, j in 1:4]   # V_start = b .+ V_end
    mk()   = ArgmaxStage(klay; reward = reward, axis = :k)

    A, B  = (b = 1.0,), (b = 5.0,)
    V_end = zeros(4)

    s_correct = mk()
    backward!(s_correct, V_end, A; env_changed = true)                  # prime the field at A
    V_correct = copy(backward!(s_correct, V_end, B; env_changed = true)) # env changed ⇒ refill at B

    s_stale = mk()
    backward!(s_stale, V_end, A; env_changed = true)                    # prime the field at A
    V_stale = copy(backward!(s_stale, V_end, B; env_changed = false))   # WRONG: skip ⇒ A's field reused

    @test V_correct != V_stale          # the flag genuinely controls the refresh
    @test V_correct == fill(5.0, 4)     # B's reward refilled (b = 5)
    @test V_stale   == fill(1.0, 4)     # A's reward (b = 1) silently reused — the bug a stale flag causes

    # The same stale check on the logit cost field.
    alay = GriddedLayout(:a => Discrete([1, 2]))
    C    = [0.0 0.5; 0.5 0.0]
    lo() = LogitChoiceStage(alay; axis = :a, cost_matrix = C, ε = FromEnv(:eps))
    Vl   = Float64[0.0, 1.0]

    l_correct = lo(); backward!(l_correct, Vl, (eps = 0.7,); env_changed = true)
    Vl_correct = copy(backward!(l_correct, Vl, (eps = 1.8,); env_changed = true))
    l_stale = lo(); backward!(l_stale, Vl, (eps = 0.7,); env_changed = true)
    Vl_stale = copy(backward!(l_stale, Vl, (eps = 1.8,); env_changed = false))
    @test Vl_correct != Vl_stale
end

# A fill crosses a function barrier carrying the dep names and the source's env-dependence in the type
# domain, so it costs its fibers and nothing per cell. The four bounds below are that property, not a
# timing, and each guards one named mechanism of it — its threshold sits between the delivered bytes
# and those of a mutant that removes exactly that mechanism and nothing else:
#
#   scalar fill      2 912 <   8 000  — `_fill_scalar_buffer!`'s `source::S` (18 912) and the scalar
#                                       fill's two `Val`s (339 520)
#   argmax fill     61 088 < 100 000  — `_fill_field!`'s `source::S` (157 088)
#   CS sweep        29 272 <  57 600  — `_cs_cell`'s type-domain axis names (202 072)
#   exit composite  80 304 < 240 000  — the `Val`-forwarding `evaluate` of `MappedField` (707 504) and
#                                       of `ExitHazardSource` (713 904)

@testset "refill — a fill allocates its fibers and nothing per cell" begin
    n      = 100
    layout = GriddedLayout(:wealth => GriddedContinuous(range(0.0, 4.0; length = n)), :income => Discrete([0.5, 1.5]))
    dest   = (; wealth, income, env) -> (1 + env.r) * wealth + env.w * income
    sf     = HouseholdStages.ScalarField(dest, layout, Float64)
    env    = (r = 0.04, w = 1.2)
    seat() = HouseholdStages.fill_scalar_field!(sf, dest, layout, env)
    seat()                                              # compile before measuring
    @test (@allocated seat()) < 8_000                   # 200 dep-combinations, compact buffer preallocated

    # A bare dep closure reaches `_fill_field!` as a `Function`-typed argument, which is where the
    # barrier's `source::S` earns its keep; `ArgmaxStage` hands `reward` to the fill verbatim.
    nz     = 200
    zlay   = GriddedLayout(:z => Discrete([1, 2]), :wealth => GriddedContinuous(range(0.0, 4.0; length = nz)),
                           :income => Discrete([0.5, 1.5]))
    am     = ArgmaxStage(zlay; axis = :z, reward = (; wealth, income, env) -> [0.0 (-wealth * env.ρ); (income - 1.0) 0.0])
    Vz     = zeros(2, nz, 2)
    faces() = backward!(am, Vz, (ρ = 1.0,))
    faces()
    @test (@allocated faces()) < 100_000                # 400 dep-combinations, one 2×2 fiber each

    m      = 60
    wlay   = GriddedLayout(:wealth => GriddedContinuous(range(0.1, 6.0; length = m)))
    cs     = ConsumptionSavingsStage(wlay; β = 0.96, utility = (cell, c; env) -> log(c) + env.r, axis = :wealth)
    V      = zeros(m)
    sweep() = backward!(cs, V, (r = 0.04,))
    sweep()
    @test (@allocated sweep()) < 2 * sizeof(Float64) * m^2   # one face-sized reward matrix, no per-cell boxing

    # The exit composite's hazard source nests `ExitHazardSource` inside `MappedField`, so a wrapper
    # that fails to forward the env-dependence `Val` to what it wraps shows up here.
    xlay   = GriddedLayout(:wealth => GriddedContinuous(range(0.0, 4.0; length = nz)),
                           :income => Discrete([0.5, 1.5]), :exiting => Discrete([1]))
    xchain = ExogenousExit(xlay; bequest = 0.0,
                           survival = (; wealth, income, env) -> 0.99 - 0.001 * wealth * income * env.δ)
    Vx     = zeros(nz, 2, 1)
    leave() = backward!(xchain, Vx, (δ = 1.0,))
    leave()
    @test (@allocated leave()) < 240_000                # 400 dep-combinations through two source wrappers
end
