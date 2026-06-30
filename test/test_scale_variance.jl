using Test
using HouseholdStages

# ScaleVarianceStage: the STREAMING mean-preserving-spread primitive (O(nx), no θ-axis). It must
# reproduce the gridded ScaleChoiceStage value (same MPS, same hard argmax), conserve mass with
# frozen-policy duality, and be forward-mode-Dual compatible with the policy frozen — derivative
# parity with consumption-savings.

xs  = collect(0.0:0.5:10.0); nx = length(xs)
shk, wts = [-1.0, 1.0], [0.5, 0.5]
disp = [0.0, 0.5, 1.0, 1.5, 2.0]
Vbump = @. 2.0 * exp(-(xs - 5.0)^2 / 4.0)       # convex flanks ⇒ interior θ*, concave peak ⇒ θ*=0

@testset "scale-variance — value matches gridded ScaleChoiceStage" begin
    cost = [0.0, 0.05, 0.2, 0.45, 0.8]
    prim = ScaleVarianceStage(GriddedLayout(:x => GriddedContinuous(xs));
                              axis = :x, dispersions = disp, shocks = shk, weights = wts, cost = cost)
    Vp = backward!(prim, Vbump, NamedTuple())

    gblock = GriddedLayout(:x => GriddedContinuous(xs), :scale => Discrete([1]))
    grid   = ScaleChoiceStage(gblock; x_axis = :x, scale_axis = :scale,
                              dispersions = disp, shocks = shk, weights = wts, cost = cost)
    Vg = backward!(grid, reshape(Vbump, nx, 1), NamedTuple())
    @test Vp ≈ vec(Vg)
end

@testset "scale-variance — concave insures (θ*=0), convex gambles (θ*=max)" begin
    # Central band only: away from the boundaries (where clamping spreads even a concave V toward
    # the higher-value interior), the choice is driven purely by V's curvature.
    mid  = findall(x -> 4.0 ≤ x ≤ 6.0, xs)
    prim = ScaleVarianceStage(GriddedLayout(:x => GriddedContinuous(xs));
                              axis = :x, dispersions = disp, shocks = shk, weights = wts)  # zero cost
    backward!(prim, (@. -(xs - 5.0)^2), NamedTuple())            # globally concave ⇒ insure
    @test all(policy(prim)[mid] .== 0.0)
    backward!(prim, (@. (xs - 5.0)^2), NamedTuple())             # globally convex ⇒ gamble
    @test all(policy(prim)[mid] .== maximum(disp))
end

@testset "scale-variance — duality and mass conservation (zero cost)" begin
    prim = ScaleVarianceStage(GriddedLayout(:x => GriddedContinuous(xs));
                              axis = :x, dispersions = disp, shocks = shk, weights = wts)
    V       = Vbump
    V_start = backward!(prim, V, NamedTuple())
    Λ_start = abs.(randn(nx)); Λ_start ./= sum(Λ_start)
    Λ_end   = forward!(prim, Λ_start)
    @test sum(Λ_end) ≈ sum(Λ_start)                              # MPS is mass-conserving
    @test sum(V_start .* Λ_start) ≈ sum(V .* Λ_end)              # zero cost ⇒ ⟨V_start,Λ⟩ = ⟨V_end,Λ_end⟩
end

@testset "scale-variance — reverse adjoints (envelope-frozen, exact transposes)" begin
    prim = ScaleVarianceStage(GriddedLayout(:x => GriddedContinuous(xs));
                              axis = :x, dispersions = disp, shocks = shk, weights = wts)
    backward!(prim, Vbump, NamedTuple())
    dV = randn(nx); dΛ = randn(nx)
    # ⟨forward_adjoint!(dΛ), dV⟩ = ⟨dΛ, backward_adjoint!(dV)⟩  (K and Kᵀ are genuine transposes)
    @test sum(forward_adjoint!(prim, dΛ) .* dV) ≈ sum(dΛ .* backward_adjoint!(prim, dV))
end

@testset "scale-variance — forward-mode Dual: policy frozen, ∂V/∂(uniform V_end shift)=1" begin
    prim   = ScaleVarianceStage(GriddedLayout(:x => GriddedContinuous(xs));
                                axis = :x, dispersions = disp, shocks = shk, weights = wts)
    dstage = lift_jacobian(prim; mode = :forward)
    Vd     = HouseholdStages.ForwardDiff.Dual{Nothing}.(Vbump, 1.0)   # seed the uniform-shift tangent
    Vs     = backward!(dstage, Vd, NamedTuple())
    @test eltype(Vs) <: HouseholdStages.ForwardDiff.Dual
    @test eltype(policy(dstage)) == Float64                      # policy stays FROZEN (not Dual)
    # K_θ is stochastic ⇒ shifting V_end by c shifts V_start by c ⇒ the seeded partial is 1.
    @test all(HouseholdStages.ForwardDiff.partials.(Vs, 1) .≈ 1.0)
end
