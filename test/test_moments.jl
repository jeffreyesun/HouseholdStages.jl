using Test
using HouseholdStages

@testset "compute_moments — mean wealth on a uniform distribution" begin
    P = [0.5 0.5; 0.5 0.5]
    layout = GriddedLayout(
        :wealth => GriddedContinuous([1.0, 2.0, 3.0, 4.0]),
        :income => Discrete([0.5, 1.5]),
    )
    chain = MarkovStage(layout; axis = :income, transition_matrix = P)
    mc = define_moments!(ChainStage((chain,));
        avg_wealth = at_end(integrand = :wealth, reduce = sum),
    )

    # Use a uniform distribution; one forward step keeps it uniform under P.
    backward!(mc, zeros(4, 2), NamedTuple())        # seat the kernel
    Λ_start = fill(1 / 8, 4, 2)
    Λ_end = forward!(mc, Λ_start)
    moments = compute_moments(mc, Λ_end, NamedTuple())
    # Σ_w wealth * Σ_z (1/8) = (1+2+3+4) / 4 = 2.5
    @test moments.avg_wealth ≈ 2.5 atol = 1e-12
end

@testset "compute_moment — pure path is stage/kernel-free (§7)" begin
    # GUARD (end-goal §7): the PURE computation half — `compute_moment(layout, spec, Λ, env)` —
    # must reference NO stage or kernel. Exercise it standalone: a MomentSpec + Layout + Λ, with
    # no ChainStage/MarkovStage/IdentityStage and no kernel seating anywhere. If a stage
    # dependency ever leaks back into the computation half, this testset stops compiling/running.
    layout = GriddedLayout(
        :wealth => GriddedContinuous([1.0, 2.0, 3.0, 4.0]),
        :income => Discrete([0.5, 1.5]),
    )
    Λ    = fill(1 / 8, 4, 2)
    spec = at_end(integrand = :wealth, reduce = sum)
    # Σ_w wealth * Σ_z (1/8) = (1+2+3+4) * 2 / 8 = 2.5 — no stage in sight.
    @test compute_moment(layout, spec, Λ, NamedTuple()) ≈ 2.5 atol = 1e-12

    # An env-dependent integrand Source flows through the pure path standalone too.
    spec_env = at_end(integrand = (; wealth, env) -> wealth * env.scale, reduce = sum)
    @test compute_moment(layout, spec_env, Λ, (scale = 2.0,)) ≈ 5.0 atol = 1e-12
end

@testset "compute_moments — multiple specs in one call" begin
    P = [0.5 0.5; 0.5 0.5]
    layout = GriddedLayout(
        :wealth => GriddedContinuous([1.0, 2.0, 3.0]),
        :income => Discrete([0.5, 1.5]),
    )
    chain = MarkovStage(layout; axis = :income, transition_matrix = P)
    mc = define_moments!(ChainStage((chain,));
        K = at_end(integrand = :wealth, reduce = sum),
        N = at_end(integrand = :income, reduce = sum),
    )
    backward!(mc, zeros(3, 2), NamedTuple())        # seat the kernel
    Λ_start = fill(1 / 6, 3, 2)
    Λ_end = forward!(mc, Λ_start)
    moments = compute_moments(mc, Λ_end, NamedTuple())
    @test moments.K ≈ 2.0 atol = 1e-12  # (1+2+3)/3
    @test moments.N ≈ 1.0 atol = 1e-12  # (0.5 + 1.5)/2
end

@testset "compute_moments — env-dependent integrand evaluated at call time" begin
    layout = GriddedLayout(:wealth => GriddedContinuous([1.0, 2.0]))
    s = IdentityStage(layout)
    mc = define_moments!(ChainStage((s,));
        scaled_wealth = at_end(
            integrand = (; wealth, env) -> wealth * env.scale,
            reduce    = sum,
        ),
    )
    Λ_start = [0.5, 0.5]
    Λ_end = forward!(mc, Λ_start)
    moments = compute_moments(mc, Λ_end, (scale = 3.0,))
    # 1.0 * 0.5 * 3 + 2.0 * 0.5 * 3 = 4.5
    @test moments.scaled_wealth ≈ 4.5 atol = 1e-12
end

@testset "define_moments! — singleton stage gets wrapped into a ChainStage" begin
    layout = GriddedLayout(:w => GriddedContinuous([0.0, 1.0]))
    s = IdentityStage(layout)
    mc = define_moments!(s; total = at_end(integrand = :w, reduce = sum))

    @test mc isa ChainStage
    @test !isempty(mc.spec.moments)
    @test length(mc.spec.stages) == 1

    Λ_start = [0.4, 0.6]
    Λ_end = forward!(mc, Λ_start)
    @test Λ_end == Λ_start
    @test compute_moments(mc, Λ_end, NamedTuple()).total ≈ 0.6 atol = 1e-12
end

@testset "define_moments! — re-defining a moment errors by default" begin
    layout = GriddedLayout(:w => GriddedContinuous([0.0, 1.0]))
    s   = IdentityStage(layout)
    mc1 = define_moments!(s; total = at_end(integrand = :w, reduce = sum))
    @test_throws ErrorException define_moments!(mc1; total = at_end(integrand = :w, reduce = sum))
end
