using Test
using HouseholdStages

# The `:sequential` vs `:divide_conquer` monotone-argmax paths through
# `ConsumptionSavingsStage.backward!`. (The standalone `k1_argmax_*` helpers were removed — they
# duplicated the production `_ca_table_walk!`; the walk modes are exercised here through the real
# stage — `:seq` and the non-pow2 `:rec_dc` below — and `:iter_dc` additionally on the GPU gate.)

@testset "ConsumptionSavingsStage — :divide_conquer matches :sequential on the Aiyagari calibration" begin
    # Small Aiyagari-shape problem; exponential wealth grid; CRRA log utility.
    n_w = 64
    layout = GriddedLayout(
        :wealth => GriddedContinuous(
            [exp(t) - 1.0 for t in range(0.0, log(101.0); length = n_w)]),
        :income => Discrete([0.6, 1.0, 1.4]),
    )
    u = (cell, c; env) -> log(c)
    seq = ConsumptionSavingsStage(layout; β = 0.96, utility = u,
                                       axis = :wealth,
                                       monotone_search = :sequential)
    dc  = ConsumptionSavingsStage(layout; β = 0.96, utility = u,
                                       axis = :wealth,
                                       monotone_search = :divide_conquer)

    V_end = [0.1 * w_i + 0.05 * y_j for w_i in 1:n_w, y_j in 1:3]
    env   = NamedTuple()

    V_seq = copy(backward!(seq, V_end, env))
    V_dc  = copy(backward!(dc,  V_end, env))

    @test policy(seq) == policy(dc)
    @test V_seq ≈ V_dc atol = 1e-12
end

@testset "ConsumptionSavingsStage — :divide_conquer rejects unknown search mode" begin
    layout = GriddedLayout(
        :wealth => GriddedContinuous([0.0, 1.0, 2.0]),
        :y => Discrete([1.0]),
    )
    @test_throws AssertionError ConsumptionSavingsStage(layout;
        β = 0.96,
        utility = (cell, c; env) -> log(c),
        monotone_search = :something_else,
    )
end
