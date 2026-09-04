using Test
using HouseholdStages

# Direct tests of the DeterministicContinuousStage primitive. `WealthChangeStage` is a domain alias
# of it, so the WealthChange/AssetPrice files cover the wrapper; this one exercises the primitive
# under its own name and the shared forward-redistribution seam.

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

@testset "integer scatter — on-grid landing equals a single-point Young split" begin
    # An exactly-on-grid destination splits to all-mass-on-one-point, so the `NearestScatterOp`
    # integer scatter and the `LotteryScatterOp` split of `grid[policy]` agree to machine precision.
    # The index-policy stages take the integer scatter: an index lands exactly, so the selection is
    # 0/1 and needs no grid plan (`ScatterKernel`'s forward!).
    grid  = [0.0, 1.0, 2.0, 3.0, 4.0]
    n     = length(grid)
    Λ0    = reshape(collect(1.0:n), n)
    Λ0  ./= sum(Λ0)

    policy = [1, 2, 2, 4, 5]                      # on-grid destinations
    Λ_near = zeros(n)
    HouseholdStages.stratified!(HouseholdStages.NearestScatterOp(), Λ_near, Λ0, policy; dims=1)

    dest_vals = grid[policy]                      # exact grid values ⇒ on-grid
    lo, hi    = ones(Int32, n), ones(Int32, n)
    HouseholdStages.stratified!(HouseholdStages.SeatInterpOp(), lo, hi, dest_vals, grid; dims=1)
    Λ_share = zeros(n)
    HouseholdStages.stratified!(HouseholdStages.LotteryScatterOp(), Λ_share, Λ0, lo, hi, dest_vals, grid; dims=1)

    @test Λ_near ≈ Λ_share atol = 1e-15
    @test sum(Λ_near) ≈ sum(Λ0) atol = 1e-15      # mass preserved
end

@testset "a non-monotone destination map still seats weights in [0, 1]" begin
    # Nothing forbids a destination that runs down the operative axis and back up — a means test,
    # a notched schedule, an argmax that jumps. Both verbs read ONE seated pair, so they are exact
    # transposes of each other whatever that pair is; what has to be checked directly is that the
    # pair CONTAINS the position. Where it does not, the derived weight leaves `[0, 1]` and `K`
    # stops being an averager: signed weights, negative mass, no Bellman contraction. A collapsed
    # pair is the endpoint clamp and takes the whole mass.
    function seated_well(x, g, lo, hi)
        lo == hi && return x <= g[1] || x >= g[end]
        return g[lo] <= x <= g[hi] && 0 <= HouseholdStages._interp_share(x, g, lo, hi) <= 1
    end

    g      = collect(0.0:5.0)
    x      = [3.4, 1.2, 4.6, 0.7, 2.9, 5.0]        # deliberately out of order along the fiber
    lo, hi = ones(Int32, 6), ones(Int32, 6)
    HouseholdStages.stratified!(HouseholdStages.SeatInterpOp(), lo, hi, x, g; dims=1)
    @test all(seated_well.(x, Ref(g), lo, hi))
    @test all(lo .!= hi)                           # every cell takes the genuine two-node path

    # And through a stage: a means-tested transfer withdrawn at a notch sends post-wealth DOWN as
    # wealth crosses the threshold, so the map is non-monotone on any grid spanning it.
    grid   = collect(range(0.0, 20.0; length = 41))
    n      = length(grid)
    layout = GriddedLayout(:w => GriddedContinuous(grid))
    stage  = DeterministicContinuousStage(layout; axis = :w,
                                          destination = (; w) -> 1.03w + (w < 8 ? 3.0 : 0.0))
    backward!(stage, collect(1.0:n), NamedTuple())
    k, xs = stage.kernel, HouseholdStages.destinations(stage.kernel)
    @test !issorted(xs)                            # the fixture really is non-monotone
    @test all(seated_well.(xs, Ref(grid), k.lo, k.hi))

    # `K` is an averager: nonnegative entries, unit column sums.
    K = reduce(hcat, [copy(forward!(stage, [i == j for i in 1:n] .* 1.0)) for j in 1:n])
    @test minimum(K) >= 0
    @test all(isapprox.(vec(sum(K; dims = 1)), 1.0; atol = 1e-14))
end

@testset "a position sitting on a node takes that node's value, sentinel included" begin
    # The `-Inf` fallback picks the better of the pair because the slope through a sentinel is
    # degenerate. On a node that is the wrong question: the whole weight sits on that one node, so
    # the gather owes exactly its value — and a borrowing constraint puts the sentinel on a PREFIX
    # of the grid, where taking the better of the pair would report an infeasible destination as
    # feasible. Reading the node also makes the gather agree with the linear transpose there.
    g  = collect(0.0:5.0)
    V  = [-Inf, -Inf, -Inf, 30.0, 40.0, 50.0]      # nodes 1:3 unreachable
    xs = [2.0, 2.5, 3.0]                           # the boundary node, inside, the first feasible node
    lo, hi = ones(Int32, 3), ones(Int32, 3)
    HouseholdStages.stratified!(HouseholdStages.SeatInterpOp(), lo, hi, xs, g; dims=1)
    vin = zeros(3)
    HouseholdStages.stratified!(HouseholdStages.LotteryGatherOp(), vin, V, lo, hi, xs, g; dims=1)
    @test vin[1] == -Inf                           # ON the last infeasible node
    @test vin[2] == 30.0                           # strictly inside — the better of the pair
    @test vin[3] == 30.0                           # ON the first feasible node
end
