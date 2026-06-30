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
                         axis = :k, search = :brute)
    a_cons = ArgmaxStage(klay; reward = [i == j ? 0.0 : -Inf for i in 1:4, j in 1:4],
                         axis = :k, search = :brute)
    @test a_env.cache.reward_env_dep  == true
    @test a_cons.cache.reward_env_dep == false

    # Logit: the `ε`-baked field is env-dependent iff the cost reads env OR `ε` is FromEnv.
    alay = GriddedLayout(:a => Discrete([1, 2]))
    C    = [0.0 0.5; 0.5 0.0]
    @test LogitChoiceStage(alay; axis = :a, cost_matrix = C, ε = FromEnv(:eps)).cache.cost_env_dep == true
    @test LogitChoiceStage(alay; axis = :a, cost_matrix = C, ε = 1.0).cache.cost_env_dep == false

    # Markov: the stored `K = Tᵀ` rides a `MappedField`, opaque to `reads_env` ⇒ conservatively
    # env-dependent even for a constant matrix (refills only ever cost a redundant fill; the
    # `env_changed = false` skip then elides them — MappedField introspection is a P3 concern).
    zlay = GriddedLayout(:z => Discrete([1, 2]))
    @test MarkovStage(zlay; axis = :z, transition_matrix = [0.9 0.1; 0.2 0.8]).cache.transition_env_dep == true
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
    mk()   = ArgmaxStage(klay; reward = reward, axis = :k, search = :brute)

    env = (b = 2.0,)
    V0  = Float64[0.1, 0.2, 0.3, 0.4]
    all_true = _refill_trajectory(mk(), V0, env, 6; skip_after_first = false)
    skipped  = _refill_trajectory(mk(), V0, env, 6; skip_after_first = true)
    @test all_true == skipped               # `==`, not `≈`: skipping a redundant refill must not perturb a bit
end

@testset "refill policy — fixed-env loop is bit-for-bit identical (Logit, gains a skip)" begin
    # Logit filled UNCONDITIONALLY before this chunk (no cache); it now GAINS the env_changed skip.
    # This is the specific regression check that the skip is exact for the logit cost field.
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
    mk()   = ArgmaxStage(klay; reward = reward, axis = :k, search = :brute)

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

    # And a logit stale check, since logit newly participates in the skip.
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
