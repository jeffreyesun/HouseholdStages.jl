using Test
using HouseholdStages

# Housing layout: index 1 of :h is the renter level, indices ≥ 2 are owned
# house sizes. Wealth is a passive axis carried through the choice.
function _house_layout(; n_w = 3, h_levels = [0.0, 1.0, 2.0])
    return GriddedLayout(
        StateAxis(:wealth, continuous_grid(collect(range(0.0, 1.0; length = n_w)))),
        StateAxis(:h,      discrete_finite(h_levels)),
    )
end

@testset "BuyHomeStage — renters choose best size, owners pass through" begin
    layout = _house_layout()
    stage = BuyHomeStage(layout; housing_axis = :h)
    n_w, n_h = axissize.(layout.axes)

    # Make house size 3 (index 3) the strict best for renters.
    V_end = zeros(n_w, n_h)
    V_end[:, 3] .= 1.0
    V_end[:, 2] .= 0.5

    V_start = copy(backward!(stage, V_end, nothing))
    # Renters (h index 1) take max over sizes ⇒ 1.0.
    @test all(V_start[:, 1] .== 1.0)
    # Owners keep their own value (pass-through): no buying.
    @test all(V_start[:, 2] .== V_end[:, 2])
    @test all(V_start[:, 3] .== V_end[:, 3])

    # Policy: renters point at the chosen size (index 3); owners at their own h.
    pol = reshape(policy(stage), n_w, n_h)
    @test all(pol[:, 1] .== 3)
    @test all(pol[:, 2] .== 2)
    @test all(pol[:, 3] .== 3)
end

@testset "BuyHomeStage — forward scatters renters, owners unchanged" begin
    layout = _house_layout()
    stage = BuyHomeStage(layout; housing_axis = :h)
    n_w, n_h = axissize.(layout.axes)

    V_end = zeros(n_w, n_h)
    V_end[:, 2] .= 1.0     # size 2 (index 2) is the best buy
    backward!(stage, V_end, nothing)

    # Unit mass: one renter cell and one owner cell.
    Λ_start = zeros(n_w, n_h)
    Λ_start[1, 1] = 0.4    # a renter
    Λ_start[2, 3] = 0.6    # an owner of size 3
    Λ_end = copy(forward!(stage, Λ_start))

    @test sum(Λ_end) ≈ 1.0 atol = 1e-12
    # The renter bought size 2 at the same wealth row.
    @test Λ_end[1, 2] ≈ 0.4 atol = 1e-12
    @test Λ_end[1, 1] ≈ 0.0 atol = 1e-12
    # The owner is untouched.
    @test Λ_end[2, 3] ≈ 0.6 atol = 1e-12
end

@testset "BuyHomeStage — gate: an owner never buys a different size" begin
    layout = _house_layout()
    stage = BuyHomeStage(layout; housing_axis = :h)
    n_w, n_h = axissize.(layout.axes)

    # Even if another size looks far better, owners are gated to their own h.
    V_end = zeros(n_w, n_h)
    V_end[:, 1] .= 10.0      # renting suddenly dominates — but owners can't switch
    backward!(stage, V_end, nothing)

    pol = reshape(policy(stage), n_w, n_h)
    @test all(pol[:, 2] .== 2)     # owner of size 2 stays at 2
    @test all(pol[:, 3] .== 3)     # owner of size 3 stays at 3

    # Forward: an owner's mass cannot move off its size.
    Λ_start = zeros(n_w, n_h); Λ_start[1, 2] = 1.0
    Λ_end = copy(forward!(stage, Λ_start))
    @test Λ_end[1, 2] ≈ 1.0 atol = 1e-12
    @test sum(Λ_end) ≈ 1.0 atol = 1e-12
end

@testset "BuyHomeStage — forward mass conservation on a full distribution" begin
    layout = _house_layout(; n_w = 4, h_levels = [0.0, 1.0, 2.0, 3.0])
    stage = BuyHomeStage(layout; housing_axis = :h)
    n_w, n_h = axissize.(layout.axes)

    V_end = randn(n_w, n_h)
    backward!(stage, V_end, nothing)

    Λ_start = abs.(randn(n_w, n_h)); Λ_start ./= sum(Λ_start)
    Λ_end = copy(forward!(stage, Λ_start))
    @test sum(Λ_end) ≈ sum(Λ_start) atol = 1e-12
    # Owners never move off their size, so total owner mass can only rise as
    # renters who chose to own join it (renters who pick index 1 stay renters).
    owner_in  = sum(Λ_start[:, 2:end])
    owner_out = sum(Λ_end[:, 2:end])
    @test owner_out ≥ owner_in - 1e-12          # owners stay; some renters may buy
end

@testset "BuyHomeStage ∘ WealthChangeStage — composition smoke test" begin
    # The buy choice picks the size; a following WealthChangeStage deducts the
    # purchase price q·h from wealth, reading the post-buy housing.
    layout = _house_layout(; n_w = 5, h_levels = [0.0, 1.0, 2.0])
    buy   = BuyHomeStage(layout; housing_axis = :h)
    pay   = WealthChangeStage(layout;
        wealth_post = (cell; env) -> cell.wealth - env.q * cell.h,
        wealth_axis = :wealth,
    )
    chain = buy ∘ pay
    n_w, n_h = axissize.(layout.axes)

    V_end = randn(n_w, n_h)
    env   = (q = 0.1,)
    V_start = backward!(chain, V_end, env)
    @test size(V_start) == (n_w, n_h)
    @test all(isfinite, V_start)

    Λ_start = ones(n_w, n_h) ./ (n_w * n_h)
    Λ_end = forward!(chain, Λ_start)
    @test sum(Λ_end) ≈ 1.0 atol = 1e-12
end
