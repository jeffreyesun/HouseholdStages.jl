using Test
using HouseholdStages

# Same housing layout convention as test_buy_home.jl: :h index 1 is the renter
# level, indices ≥ 2 are owned sizes.
function _sell_layout(; n_w = 3, h_levels = [0.0, 1.0, 2.0])
    return GriddedLayout(
        :wealth => GriddedContinuous(collect(range(0.0, 1.0; length = n_w))),
        :h => Discrete(h_levels),
    )
end

@testset "SellHomeStage — owners choose keep vs sell, renters pass through" begin
    layout = _sell_layout()
    stage = SellHomeStage(layout; axis = :h)
    n_w, n_h = axissize.(layout.axes)

    # Renting (index 1) strictly dominates ⇒ every owner sells.
    V_end = zeros(n_w, n_h)
    V_end[:, 1] .= 1.0
    V_start = copy(backward!(stage, V_end, nothing))
    @test all(V_start .== 1.0)             # owners take the renter value; renters keep it

    pol = reshape(policy(stage), n_w, n_h)
    @test all(pol .== 1)                   # everyone lands on the renter index

    # Keeping (own size) strictly dominates ⇒ no owner sells.
    V_end2 = zeros(n_w, n_h)
    V_end2[:, 2] .= 1.0
    V_end2[:, 3] .= 1.0
    backward!(stage, V_end2, nothing)
    pol2 = reshape(policy(stage), n_w, n_h)
    @test all(pol2[:, 1] .== 1)            # renters stay renters
    @test all(pol2[:, 2] .== 2)            # owners keep size 2
    @test all(pol2[:, 3] .== 3)            # owners keep size 3
end

@testset "SellHomeStage — gate: an owner can only keep or become a renter" begin
    layout = _sell_layout()
    stage = SellHomeStage(layout; axis = :h)
    n_w, n_h = axissize.(layout.axes)

    # Size 3 (index 3) looks best, but an owner of size 2 may NOT switch to it —
    # only {keep size 2, sell → renter} are available.
    V_end = zeros(n_w, n_h)
    V_end[:, 3] .= 10.0
    V_end[:, 2] .= 0.2
    V_end[:, 1] .= 0.5
    backward!(stage, V_end, nothing)

    pol = reshape(policy(stage), n_w, n_h)
    # Owner of size 2: sell (→1, value 0.5) beats keep (0.2) ⇒ index 1, never 3.
    @test all(pol[:, 2] .== 1)
    # The owner's value is the better of keep/sell, never the unreachable 10.0.
    V_start = reshape(stage.scratch.V_start, n_w, n_h)
    @test all(V_start[:, 2] .== 0.5)
end

@testset "SellHomeStage — forward: sellers collapse to renter slice, mass conserved" begin
    layout = _sell_layout()
    stage = SellHomeStage(layout; axis = :h)
    n_w, n_h = axissize.(layout.axes)

    V_end = zeros(n_w, n_h)
    V_end[:, 1] .= 1.0          # renting dominates ⇒ all owners sell
    backward!(stage, V_end, nothing)

    Λ_start = zeros(n_w, n_h)
    Λ_start[1, 2] = 0.3         # owner of size 2
    Λ_start[2, 3] = 0.5         # owner of size 3
    Λ_start[1, 1] = 0.2         # an existing renter
    Λ_end = copy(forward!(stage, Λ_start))

    @test sum(Λ_end) ≈ 1.0 atol = 1e-12
    @test sum(Λ_end[:, 2:end]) ≈ 0.0 atol = 1e-12   # nobody owns after a full sell-off
    @test Λ_end[1, 1] ≈ 0.3 + 0.2 atol = 1e-12      # size-2 seller + existing renter, same wealth row
    @test Λ_end[2, 1] ≈ 0.5 atol = 1e-12            # size-3 seller, its wealth row
end

@testset "SellHomeStage — forward mass conservation on a full distribution" begin
    layout = _sell_layout(; n_w = 4, h_levels = [0.0, 1.0, 2.0, 3.0])
    stage = SellHomeStage(layout; axis = :h)
    n_w, n_h = axissize.(layout.axes)

    V_end = randn(n_w, n_h)
    backward!(stage, V_end, nothing)

    Λ_start = abs.(randn(n_w, n_h)); Λ_start ./= sum(Λ_start)
    Λ_end = copy(forward!(stage, Λ_start))
    @test sum(Λ_end) ≈ sum(Λ_start) atol = 1e-12
end

@testset "SellHomeStage ∘ WealthChangeStage — sale-proceeds composition smoke test" begin
    # Sell chooses {keep, sell→renter}; a following WealthChangeStage credits
    # sale proceeds (net of a realtor fee) to sellers, identified post-sell by
    # having landed at the renter level. Here we model the fee as a flat wealth
    # bump on renters as a stand-in (the real model carries pre-sell size on a
    # separate axis); the test only checks the chain runs and conserves mass.
    layout = _sell_layout(; n_w = 5, h_levels = [0.0, 1.0, 2.0])
    sell  = SellHomeStage(layout; axis = :h)
    fee   = WealthChangeStage(layout;
        wealth_post = (; h, wealth, env) -> h == 0.0 ? wealth + env.proceeds : wealth,
        axis = :wealth,
    )
    chain = sell ∘ fee
    n_w, n_h = axissize.(layout.axes)

    V_end = randn(n_w, n_h)
    env   = (proceeds = 0.05,)
    V_start = backward!(chain, V_end, env)
    @test size(V_start) == (n_w, n_h)
    @test all(isfinite, V_start)

    Λ_start = ones(n_w, n_h) ./ (n_w * n_h)
    Λ_end = forward!(chain, Λ_start)
    @test sum(Λ_end) ≈ 1.0 atol = 1e-12
end
