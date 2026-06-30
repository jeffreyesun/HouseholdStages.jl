using Test
using HouseholdStages

# Direct tests of the DeterministicContinuousStage primitive (the generalisation behind
# WealthChangeStage). WealthChangeStage is now a domain alias of this primitive,
# so the existing WealthChange/AssetPrice tests already cover the wrapper; these
# exercise the primitive name + the shared forward-redistribution seam.

@testset "DeterministicContinuousStage — equals the WealthChangeStage wrapper" begin
    layout = GriddedLayout(
        :wealth => GriddedContinuous([0.0, 1.0, 2.0, 3.0, 4.0]),
        :income => Discrete([0.6, 1.0, 1.4]),
    )
    dest = (; wealth, income, env) -> (1 + env.r) * wealth + env.w * income

    cm = DeterministicContinuousStage(layout; destination = dest, axis = :wealth)
    wc = WealthChangeStage(layout; wealth_post = dest, axis = :wealth)

    @test wc isa DeterministicContinuousStage    # the wrapper returns the primitive
    @test cm.spec.axis === :wealth
    @test wc.spec.axis === :wealth        # axis-neutral spec field

    env = (; r = 0.03, w = 1.0)
    V_end = reshape(Float64.(1:15), (5, 3))
    @test backward!(cm, V_end, env) == backward!(wc, V_end, env)

    Λ_start = rand(5, 3); Λ_start ./= sum(Λ_start)
    @test forward!(cm, Λ_start) == forward!(wc, Λ_start)
end

@testset "DiscreteMoveStage — mass lands on the nearest grid index; K/Kᵀ duality" begin
    grid   = [0.0, 1.0, 2.0, 3.0, 4.0]
    inc    = [0.7, 1.2]
    layout = GriddedLayout(:wealth => GriddedContinuous(grid),
                           :income => Discrete(inc))
    # An off-grid float destination, snapped to the nearest grid INDEX (the discrete sibling of the
    # continuous move's Young split).
    dest  = (; wealth, income, env) -> 0.5 * wealth + income
    stage = DiscreteMoveStage(layout; destination = dest, axis = :wealth)

    V_end   = reshape(Float64.(1:10), (5, 2))
    V_start = copy(backward!(stage, V_end, NamedTuple()))

    for j in 1:2, i in 1:5
        target = 0.5 * grid[i] + inc[j]
        nk     = argmin(abs.(grid .- target))      # nearest grid index
        @test policy(stage)[i, j] == nk
        @test V_start[i, j] == V_end[nk, j]        # backward gathers from the snapped index
    end

    Λ0 = rand(5, 2); Λ0 ./= sum(Λ0)
    Λ1 = copy(forward!(stage, Λ0))
    @test sum(Λ1) ≈ 1.0 atol = 1e-12               # 0/1 scatter conserves mass

    # K/Kᵀ duality (a 0/1 selection, no flow payoff): ⟨V_end, K·Λ⟩ = ⟨Kᵀ·V_end, Λ⟩.
    @test sum(Λ1 .* V_end) ≈ sum(Λ0 .* V_start) atol = 1e-12
end

@testset "DiscreteMoveStage vs DeterministicContinuousStage — agree on an on-grid destination" begin
    # When the destination lands exactly on grid points, the nearest-index scatter and the
    # off-grid Young split coincide (the split puts all mass on the single bracketing node).
    grid   = [0.0, 1.0, 2.0, 3.0, 4.0]
    layout = GriddedLayout(:wealth => GriddedContinuous(grid))
    dest   = (; wealth, env) -> clamp(wealth + 1.0, 0.0, 4.0)   # on-grid shift

    dm = DiscreteMoveStage(layout; destination = dest, axis = :wealth)
    cm = DeterministicContinuousStage(layout; destination = dest, axis = :wealth)

    V_end = Float64.(1:5)
    @test backward!(dm, V_end, NamedTuple()) ≈ backward!(cm, V_end, NamedTuple()) atol = 1e-12

    Λ0 = rand(5); Λ0 ./= sum(Λ0)
    @test forward!(dm, Λ0) ≈ forward!(cm, Λ0) atol = 1e-12
end

@testset "redistribute_along! — on-grid :nearest equals a single-point Young split" begin
    # An exactly-on-grid destination Young-splits to all-mass-on-one-point. For a
    # *monotone* policy (the supermodular case CS guarantees, where destinations
    # are sorted along the axis so the :share cursor walk is well-defined), the
    # :nearest integer-scatter and the :share split of `grid[policy]` must agree
    # to machine precision. (An unsorted policy is exactly why CS keeps the
    # integer-scatter path rather than routing through :share — see the primitive
    # forward!.)
    grid  = [0.0, 1.0, 2.0, 3.0, 4.0]
    n     = length(grid)
    Λ0    = reshape(collect(1.0:n), n)
    Λ0  ./= sum(Λ0)

    policy = [1, 2, 2, 4, 5]                      # monotone on-grid destinations
    Λ_near = zeros(n)
    HouseholdStages.redistribute_along!(Λ_near, Λ0, policy, nothing, Val(1), Val(:nearest))

    dest_vals = grid[policy]                      # exact grid values ⇒ on-grid
    Λ_share = zeros(n)
    HouseholdStages.redistribute_along!(Λ_share, Λ0, dest_vals, grid, Val(1), Val(:share))

    @test Λ_near ≈ Λ_share atol = 1e-15
    @test sum(Λ_near) ≈ sum(Λ0) atol = 1e-15      # mass preserved
end
