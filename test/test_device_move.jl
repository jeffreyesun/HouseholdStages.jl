# Relocation coverage for `to_device`/`to_host` (src/lifts/gpu.jl), run on the CPU with a *copying*
# adaptor so the whole battery is exercised without a device. The copy makes relocation observable
# by object identity: a type with no rebuild rule would hand the relocated stage the host's own
# array, and the completeness check below catches exactly that. The same copy is what makes the
# write-through check meaningful — a relocated stage must own its buffers, so running it leaves the
# host stage's arrays untouched.

module DeviceMoveTest

using Test
using HouseholdStages
using Adapt
using ForwardDiff

const HS = HouseholdStages

include("device_walk.jl")

# A relocation that copies rather than crossing a device boundary.
struct CopyTo end
Adapt.adapt_storage(::CopyTo, x::AbstractArray) = copy(x)

# --- battery: one case per declared relocation rule -----------------------

# MarkovStage — DenseKernel over a MatrixField.
function build_markov()
    P      = [0.7 0.3; 0.3 0.7]
    layout = GriddedLayout(:z => Discrete([0.5, 1.5]))
    (MarkovStage(layout; axis = :z, transition_matrix = P), [1.0, 2.0], [0.4, 0.6], NamedTuple())
end

# MarkovStage between layouts that DIFFER — the transition marginalizes `:z` away, so the stage's
# start layout carries the axis at size 2 and its end layout at size 1. Everything else in the roster
# is built through the one-layout sugar, which makes a stage's two layouts the same object; a
# relocation that carried them across in the wrong order would then be indistinguishable from a
# correct one. This is the case that tells them apart.
function build_markov_marginalize()
    start_layout = GriddedLayout(:z => Discrete([0.5, 1.5]))
    end_layout   = GriddedLayout(:z => Discrete([0.0]))
    stage = MarkovStage(start_layout, end_layout; axis = :z, transition_matrix = ones(2, 1))
    (stage, [3.0], [0.4, 0.6], NamedTuple())
end

# LogitChoiceStage — LogitChoiceKernel (a contained DenseKernel + the two softmax buffers).
function build_logit()
    layout = GriddedLayout(:a => Discrete([1, 2]))
    stage  = LogitChoiceStage(layout; axis = :a, cost_matrix = [0.0 0.5; 0.5 0.0], ε = 0.5)
    (stage, [1.0, 2.0], [0.4, 0.6], NamedTuple())
end

# ArgmaxStage — ScatterKernel over a DestinationField, plus the reward face in `scratch.U`.
function build_argmax()
    layout = GriddedLayout(:k => Discrete([1, 2, 3, 4]))
    reward = [i == j ? 0.0 : -1.0 for i in 1:4, j in 1:4]
    stage  = ArgmaxStage(layout; reward = reward, axis = :k)
    (stage, [0.4, 0.1, 0.9, 0.2], [0.1, 0.2, 0.3, 0.4], NamedTuple())
end

# ContinuousArgmaxStage — InterpKernel, a lowered closure source in scratch, the cache grid, and
# the extension's lazily-seated `Ref{Any}` device slot.
function build_continuous_argmax()
    grid   = collect(range(0.0, 10.0; length = 21))
    layout = GriddedLayout(:a => GriddedContinuous(grid))
    stage  = ContinuousArgmaxStage(layout;
        reward = (a, a_next) -> -0.5 * (a_next - (0.5 * a + 1.0))^2, axis = :a)
    Λ = fill(1 / 21, 21)
    (stage, [-0.05 * (g - 5.0)^2 for g in grid], Λ, NamedTuple())
end

# DeterministicContinuousStage — InterpKernel + an env-dependent destination ScalarField.
function build_deterministic_continuous()
    layout = GriddedLayout(
        :wealth => GriddedContinuous([0.0, 1.0, 2.0, 3.0, 4.0]),
        :income => Discrete([0.6, 1.0, 1.4]),
    )
    stage = DeterministicContinuousStage(layout;
        destination = (; wealth, income, env) -> (1 + env.r) * wealth + env.w * income,
        axis = :wealth)
    Λ = fill(1 / 15, 5, 3)
    (stage, reshape(Float64.(1:15), 5, 3), Λ, (r = 0.03, w = 1.0))
end

# UtilityStage — a dep-closure ScalarField (compact buffer, broadcast shape).
function build_utility()
    layout = GriddedLayout(
        :wealth => GriddedContinuous([0.0, 1.0, 2.0]),
        :income => Discrete([0.5, 1.0]),
    )
    stage = UtilityStage(layout; utility = (; wealth, env) -> wealth + env.bonus)
    (stage, zeros(3, 2), fill(1 / 6, 3, 2), (bonus = 10.0,))
end

# UtilityStage with an env-resolved array payoff — the `ScalarField{Any}` re-inference path.
function build_utility_fromenv()
    layout = GriddedLayout(:wealth => GriddedContinuous([0.0, 1.0, 2.0]))
    stage  = UtilityStage(layout; utility = FromEnv(:u))
    (stage, zeros(3), fill(1 / 3, 3), (u = [0.5, 1.5, 2.5],))
end

# EntryStage — EntryKernel (a mutable `Any` slot) + the entry ScalarField.
function build_entry()
    layout = GriddedLayout(:x => Discrete([1, 2, 3]))
    (EntryStage(layout; entry = [0.1, 0.2, 0.3]), zeros(3), [0.2, 0.5, 0.3], NamedTuple())
end

# EntryStage with an env-resolved array inflow — the second `ScalarField{Any}` path.
function build_entry_fromenv()
    layout = GriddedLayout(:x => Discrete([1, 2, 3]))
    (EntryStage(layout; entry = FromEnv(:g)), zeros(3), [0.2, 0.5, 0.3], (g = [0.05, 0.05, 0.9],))
end

# TimeDiscountingStage — PointwiseScale over two `Ref{Any}` scales.
function build_time_discounting()
    layout = GriddedLayout(:z => Discrete([0.5, 1.5]))
    (TimeDiscountingStage(layout; β = 0.96), [1.0, 2.0], [0.4, 0.6], NamedTuple())
end

# MixingStage — MixingKernel (two seated DenseKernels + the frozen θ* policy).
function build_mixing()
    K_A    = [0.9 0.1 0.0; 0.1 0.8 0.1; 0.0 0.1 0.9]
    K_B    = [0.4 0.4 0.2; 0.3 0.4 0.3; 0.2 0.4 0.4]
    layout = GriddedLayout(:x => Discrete([1.0, 2.0, 3.0]))
    stage  = MixingStage(layout; axis = :x, K_A = K_A, K_B = K_B, cost_curvature = 2.0)
    (stage, [4.0, 1.0, 2.0], [0.2, 0.5, 0.3], NamedTuple())
end

# MeanPreservingSpreadStage — MeanPreservingSpreadKernel + the plan's axis grid.
function build_mean_preserving_spread()
    xs     = collect(0.0:0.5:10.0)
    layout = GriddedLayout(:x => GriddedContinuous(xs))
    stage  = MeanPreservingSpreadStage(layout; axis = :x, θ_max = 2.0,
                                       cost = (θ; env) -> env.λ * θ^2)
    V_end  = @. 2.0 * exp(-(xs - 5.0)^2 / 4.0)
    (stage, V_end, fill(1 / length(xs), length(xs)), (λ = 0.05,))
end

# GaussianLoadingStage — GaussianLoadingKernel + the plan's axis grid.
function build_gaussian_loading()
    ws     = collect(0.5:0.25:8.0)
    layout = GriddedLayout(:wealth => GriddedContinuous(ws))
    stage  = GaussianLoadingStage(layout; axis = :wealth, anchor = 1.02,
                                  increment_mean = 0.03, increment_sd = 0.2,
                                  cost = (θ; env) -> 0.01 * θ^2)
    (stage, sqrt.(ws), fill(1 / length(ws), length(ws)), NamedTuple())
end

# ChainStage — a ChainStageBuffer of two leaves; the buffer owns no tensors of its own.
function build_chain()
    P      = [0.7 0.3; 0.3 0.7]
    layout = GriddedLayout(:z => Discrete([0.5, 1.5]))
    chain  = MarkovStage(layout; axis = :z, transition_matrix = P) ∘
             TimeDiscountingStage(layout; β = 0.96)
    (chain, [1.0, 2.0], [0.4, 0.6], NamedTuple())
end

# ProductStage — a ProductStageBuffer, the one combinator buffer owning fused V/Λ tensors.
function build_product()
    P      = [0.7 0.3; 0.3 0.7]
    layout = GriddedLayout(:z => Discrete([0.5, 1.5]), :group => Discrete([1]))
    s      = MarkovStage(layout; axis = :z, transition_matrix = P)
    ps     = product(s, s; axis = :group)
    (ps, reshape(collect(range(-1.0, 1.0; length = 4)), 2, 2), fill(0.25, 2, 2), NamedTuple())
end

const CASES = [
    ("MarkovStage",                  build_markov),
    ("MarkovStage (marginalizing)",   build_markov_marginalize),
    ("LogitChoiceStage",             build_logit),
    ("ArgmaxStage",                  build_argmax),
    ("ContinuousArgmaxStage",        build_continuous_argmax),
    ("DeterministicContinuousStage", build_deterministic_continuous),
    ("UtilityStage",                 build_utility),
    ("UtilityStage (FromEnv array)", build_utility_fromenv),
    ("EntryStage",                   build_entry),
    ("EntryStage (FromEnv array)",   build_entry_fromenv),
    ("TimeDiscountingStage",         build_time_discounting),
    ("MixingStage",                  build_mixing),
    ("MeanPreservingSpreadStage",    build_mean_preserving_spread),
    ("GaussianLoadingStage",         build_gaussian_loading),
    ("ChainStage",                   build_chain),
    ("ProductStage",                 build_product),
]

# --- checks ---------------------------------------------------------------

@testset "to_device — relocation is complete, contained, and side-effect free" begin
    for (name, build) in CASES
        @testset "$name" begin
            stage, V_end, Λ_start, env = build()
            backward!(stage, copy(V_end), env)                  # seat the lazily-filled slots
            forward!(stage, copy(Λ_start))
            V_ref = copy(backward!(stage, copy(V_end), env))
            Λ_ref = copy(forward!(stage, copy(Λ_start)))

            host     = reachable_arrays(stage)
            host_ids = Base.IdSet{Any}(host)
            moved    = to_device(stage, CopyTo())

            # Completeness: no array object survives the relocation by reference.
            @test !any(a -> a in host_ids, reachable_arrays(moved))

            # Containment: the spec and the two layouts are carried, not rebuilt — and each lands in
            # its own slot, which only a stage whose ends differ can tell (see the testset below).
            @test moved.spec === stage.spec
            @test start_layout(moved) === start_layout(stage)
            @test end_layout(moved)   === end_layout(stage)

            # The relocated stage owns its buffers: running it leaves the host stage alone and
            # reproduces the host result exactly.
            snapshot = map(copy, host)
            V_moved  = copy(backward!(moved, copy(V_end), env))
            Λ_moved  = copy(forward!(moved, copy(Λ_start)))
            @test all(isequal(a, b) for (a, b) in zip(host, snapshot))
            @test isequal(V_moved, V_ref)
            @test isequal(Λ_moved, Λ_ref)

            # Round trip: the `Array` adaptor is the inverse spelling.
            home = to_host(moved)
            @test isequal(copy(backward!(home, copy(V_end), env)), V_ref)
            @test isequal(copy(forward!(home, copy(Λ_start))), Λ_ref)
        end
    end
end

@testset "to_device — the two layouts land in their own slots" begin
    # `lifts/gpu.jl` rebuilds a primitive positionally, so the start and end layouts are two
    # arguments in a fixed order. On a stage built through the one-layout sugar they are one object,
    # and every check above it passes whichever order the rebuild uses. A stage that regrids is what
    # gives the order content: swap the two slots and its relocated copy reports its ends the wrong
    # way round, and A4.7 — layouts matching the buffers they describe — fails on the relocated stage.
    stage, = build_markov_marginalize()
    moved  = to_device(stage, CopyTo())
    @test layout_size(start_layout(stage)) != layout_size(end_layout(stage))
    @test start_layout(moved) == start_layout(stage)
    @test end_layout(moved)   == end_layout(stage)
    @test size(V_start_buffer(moved)) == layout_size(start_layout(moved))
    @test size(Λ_end_buffer(moved))   == layout_size(end_layout(moved))
end

@testset "to_device — closure captures and mutable slots" begin
    # A lowered reward source is a closure: it is carried unchanged, so its captured axis grid
    # stays host-resident and `fill_field!` keeps staging fibers across.
    stage, V_end, Λ_start, env = build_continuous_argmax()
    backward!(stage, copy(V_end), env)
    moved = to_device(stage, CopyTo())
    @test moved.scratch.src === stage.scratch.src
    @test moved.scratch.bestk !== stage.scratch.bestk          # walk buffer relocates like any array

    # The two-sided scale lives in `Ref{Any}`s; the relocated stage gets its own, at the same
    # declared element type, so an env rewrite on one is invisible to the other.
    td, V, Λ, e = build_time_discounting()
    backward!(td, copy(V), e)
    mtd = to_device(td, CopyTo())
    @test mtd.kernel !== td.kernel
    @test mtd.kernel.a !== td.kernel.a
    @test mtd.kernel.a isa Base.RefValue{Any}

    # `EntryKernel` is a mutable `Any` slot — the relocated stage owns its own.
    es, Ve, Λe, ee = build_entry()
    backward!(es, copy(Ve), ee)
    mes = to_device(es, CopyTo())
    @test mes.kernel !== es.kernel
end

@testset "to_device — an env-resolved field refills where the stage lives" begin
    # Relocation BEFORE the first `backward!`, the ordering `build → to_device → solve` produces and
    # the ordering `to_device(lift_jacobian(stage), …)` produces unconditionally. Nothing has seated
    # the field yet, so the refill itself has to put the env-resolved payload on the far side of the
    # relocation: with a copying adaptor that shows up as a copy of the env array, not the array.
    wealth = GriddedLayout(:wealth => GriddedContinuous([0.0, 1.0, 2.0]))
    v      = [0.5, 1.5, 1.9]
    cases  = (
        ("UtilityStage",  UtilityStage(wealth; utility = FromEnv(:v)),                  (v = v,)),
        ("EntryStage",    EntryStage(wealth; entry = FromEnv(:v)),                      (v = v,)),
        ("DetContStage",  DeterministicContinuousStage(wealth;
                              destination = FromEnv(:v), axis = :wealth),               (v = v,)),
    )
    for (name, stage, env) in cases
        @testset "$name" begin
            moved  = to_device(stage, CopyTo())
            backward!(moved, zeros(3), env)
            fields = [f for f in values(moved.cache) if f isa HS.ScalarField]
            @test !isempty(fields)
            @test all(f -> f.data !== env.v, fields)     # relocated, not aliased to the env array
            @test all(f -> f.data == env.v, fields)
        end
    end
end

@testset "to_device — a jacobian-lifted FromEnv field stays open for a Dual" begin
    # `lift_jacobian` hands back a freshly allocated stage, so this relocation is always the
    # unseated one. The field holds the `NaN` sentinel, not an array: narrowing its declared slot to
    # that sentinel's `Float64` would turn the `Dual` refill into a conversion error.
    layout = GriddedLayout(:x => Discrete([1, 2, 3]))
    stage  = lift_jacobian(UtilityStage(layout; utility = FromEnv(:c)); n_dual = 1)
    moved  = to_device(stage, CopyTo())
    D      = eltype(V_start_buffer(moved))
    c      = D(2.0, ForwardDiff.Partials((1.0,)))
    V      = backward!(moved, fill(zero(D), 3), (c = c,))
    @test all(v -> ForwardDiff.value(v) == 2.0 && ForwardDiff.partials(v)[1] == 1.0, V)
end

@testset "to_device — the isbits-eltype gate" begin
    payload = (syms = [:a, :b], nums = [1.0, 2.0])
    moved   = adapt(HS.MoveTo(CopyTo()), payload)
    @test moved.syms === payload.syms       # a non-isbits eltype stays host-resident
    @test moved.nums !== payload.nums
    @test moved.nums == payload.nums
end

@testset "to_device — a chain carries its moment menu" begin
    # The menu survives the relocation and evaluates against a host-resident `Λ`; `compute_moments`
    # against a genuinely device-resident `Λ` is out of reach (`OPEN_QUESTIONS.md`, 2026-07-30).
    P      = [0.5 0.5; 0.5 0.5]
    layout = GriddedLayout(
        :wealth => GriddedContinuous([1.0, 2.0, 3.0, 4.0]),
        :income => Discrete([0.5, 1.5]),
    )
    chain = define_moments!(ChainStage((MarkovStage(layout; axis = :income, transition_matrix = P),));
        avg_wealth = at_end(integrand = :wealth, reduce = sum),
    )
    backward!(chain, zeros(4, 2), NamedTuple())
    Λ_end = copy(forward!(chain, fill(1 / 8, 4, 2)))

    moved = to_device(chain, CopyTo())
    @test haskey(moved.spec.moments, :avg_wealth)
    backward!(moved, zeros(4, 2), NamedTuple())
    Λ_moved = copy(forward!(moved, fill(1 / 8, 4, 2)))
    @test compute_moments(moved, Λ_moved, NamedTuple()).avg_wealth ≈ 2.5 atol = 1e-12
    @test isequal(Λ_moved, Λ_end)
end

@testset "to_device — a mover function is rejected at the door" begin
    layout = GriddedLayout(:z => Discrete([0.5, 1.5]))
    stage  = MarkovStage(layout; axis = :z, transition_matrix = [0.7 0.3; 0.3 0.7])
    @test_throws ErrorException to_device(stage, x -> x)
    @test_throws ErrorException lift_gpu(stage)
end

end # module DeviceMoveTest
