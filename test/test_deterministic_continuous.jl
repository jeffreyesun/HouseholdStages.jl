using Test
using HouseholdStages

# Direct tests of the DeterministicContinuousStage primitive (the generalisation behind
# WealthChangeStage). WealthChangeStage is now a domain alias of this primitive,
# so the existing WealthChange/AssetPrice tests already cover the wrapper; these
# exercise the primitive name + the shared forward-redistribution seam.

@testset "DeterministicContinuousStage — equals the WealthChangeStage wrapper" begin
    layout = GriddedLayout(
        StateAxis(:wealth, continuous_grid([0.0, 1.0, 2.0, 3.0, 4.0])),
        StateAxis(:income, discrete_finite([0.6, 1.0, 1.4])),
    )
    dest = (cell; env) -> (1 + env.r) * cell.wealth + env.w * cell.income

    cm = DeterministicContinuousStage(layout; destination = dest, axis = :wealth)
    wc = WealthChangeStage(layout; wealth_post = dest, wealth_axis = :wealth)

    @test wc isa DeterministicContinuousStage    # the wrapper returns the primitive
    @test cm.spec.axis === :wealth
    @test wc.spec.axis === :wealth        # axis-neutral spec field

    env = (; r = 0.03, w = 1.0)
    V_end = reshape(Float64.(1:15), (5, 3))
    @test backward!(cm, V_end, env) == backward!(wc, V_end, env)

    Λ_start = rand(5, 3); Λ_start ./= sum(Λ_start)
    @test forward!(cm, Λ_start) == forward!(wc, Λ_start)
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
