using Test
using HouseholdStages

# ScalarField — the scalar-per-base-point (broadcast) sibling of the matrix heterogeneity field. A
# source (scalar / array / FromEnv / dep-closure) materialises to a broadcastable that the stage
# applies with any op. The Source itself lives in the stage's Spec; the field is the pure buffer.
# `materialize_scalar!(sf, source, layout, env; env_changed)` applies the static refill policy (§5.3):
# env-independent ⇒ filled once at construction; env-dependent ⇒ refilled unless `env_changed = false`.

@testset "ScalarField — scalar broadcasts as itself" begin
    layout = GriddedLayout(:wealth => GriddedContinuous([0.0, 1.0, 2.0]), :income => Discrete([1.0, 2.0]))
    sf = ScalarField(2.5, layout)
    @test scalar_broadcastable(sf) == 2.5
    V = zeros(3, 2)
    @test V .+ materialize_scalar!(sf, 2.5, layout, nothing) == fill(2.5, 3, 2)
end

@testset "ScalarField — precomputed full array used as-is" begin
    layout = GriddedLayout(:wealth => GriddedContinuous([0.0, 1.0]), :income => Discrete([1.0, 2.0]))
    arr = [10.0 20.0; 30.0 40.0]
    sf  = ScalarField(arr, layout)
    @test materialize_scalar!(sf, arr, layout, nothing) === arr
end

@testset "ScalarField — FromEnv resolves from env" begin
    layout = GriddedLayout(:wealth => GriddedContinuous([0.0, 1.0]))
    sf = ScalarField(FromEnv(:b), layout)
    @test materialize_scalar!(sf, FromEnv(:b), layout, (; b = 7.0)) == 7.0
    @test materialize_scalar!(sf, FromEnv(:b), layout, (; b = -3.0)) == -3.0   # env change re-resolves
end

@testset "ScalarField — declared-dep closure broadcasts over the non-dep axes" begin
    layout = GriddedLayout(:wealth => GriddedContinuous([0.0, 1.0, 2.0]),
                           :income => Discrete([10.0, 20.0]))
    # varies along income only ⇒ compact (income,) buffer, broadcast over wealth
    src = (; income, env) -> income * env.r
    sf  = ScalarField(src, layout)
    m = materialize_scalar!(sf, src, layout, (; r = 0.5))
    @test size(m) == (1, 2)                                  # reshaped to broadcast: (1, n_income)
    V = zeros(3, 2)
    @test (V .+ m) == [5.0 10.0; 5.0 10.0; 5.0 10.0]         # 10·0.5 over wealth, 20·0.5 over wealth
end

@testset "ScalarField — zero-axis (env-only) closure broadcasts as a scalar" begin
    layout = GriddedLayout(:wealth => GriddedContinuous([0.0, 1.0, 2.0]), :income => Discrete([1.0, 2.0]))
    src = (; env) -> 3.0 * env.k                              # declares only :env ⇒ no axis deps
    sf  = ScalarField(src, layout)
    m = materialize_scalar!(sf, src, layout, (; k = 2.0))
    @test all(m .== 6.0)
    @test (zeros(3, 2) .+ m) == fill(6.0, 3, 2)              # broadcasts across the whole grid
end

@testset "ScalarField — static refill policy: env_changed flag drives the skip" begin
    # Env-DEPENDENT field: refills every materialize by default (env_changed=true); skips only when
    # the caller asserts env_changed=false (no env-value comparison — no per-field cache record).
    layout = GriddedLayout(:wealth => GriddedContinuous([0.0, 1.0]))
    calls = Ref(0)
    src = (; wealth, env) -> (calls[] += 1; wealth + env.k)
    sf  = ScalarField(src, layout)
    @test calls[] == 0                                       # env-dependent ⇒ NaN at construction, not filled
    materialize_scalar!(sf, src, layout, (; k = 1.0)); c1 = calls[]
    @test c1 > 0
    materialize_scalar!(sf, src, layout, (; k = 1.0))                       # default env_changed=true ⇒ refill
    @test calls[] > c1
    c2 = calls[]
    materialize_scalar!(sf, src, layout, (; k = 1.0); env_changed = false) # caller asserts env unchanged ⇒ skip
    @test calls[] == c2
    materialize_scalar!(sf, src, layout, (; k = 2.0); env_changed = true)  # env changed ⇒ refill
    @test calls[] > c2
end

@testset "ScalarField — env-independent field fills once at construction, never refills" begin
    layout = GriddedLayout(:wealth => GriddedContinuous([0.0, 1.0]))
    calls = Ref(0)
    src = (; wealth) -> (calls[] += 1; 2.0 * wealth)         # no :env ⇒ env-independent
    sf  = ScalarField(src, layout)
    c1 = calls[]
    @test c1 > 0                                             # filled eagerly at construction
    materialize_scalar!(sf, src, layout, nothing; env_changed = true)      # env-independent ⇒ never refills
    @test calls[] == c1
    materialize_scalar!(sf, src, layout, nothing; env_changed = false)
    @test calls[] == c1
end
