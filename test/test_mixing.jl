using Test
using HouseholdStages

# MixingStage (chosen lottery over two kernels, closed form): a = K_A·W, b = K_B·W, y = a − b,
# θ* = policy(y) ∈ [0,1], V = b + θ*·y − cost(θ*) — the conjugate value computed FROM the
# (cost, policy) pair, so the identity holds by construction (no separately-supplied conjugate).
# Checked against the hand-built closed form and, independently, a brute max over a fine θ grid.
# Wrapped in a module so its fixture globals don't leak.

module MixingTest
using Test, HouseholdStages
using HouseholdStages.ForwardDiff: Dual, value, partials, tagtype
include("envelope_oracle.jl")

const mx_n  = 3
const mx_xs = [1.0, 2.0, 3.0]
const mx_block = GriddedLayout(:x => Discrete(mx_xs))
const mx_KA = [0.9 0.1 0.0; 0.1 0.8 0.1; 0.0 0.1 0.9]
const mx_KB = [0.4 0.4 0.2; 0.3 0.4 0.3; 0.2 0.4 0.4]
const mx_κ  = 2.0
const mx_W  = [4.0, 1.0, 2.0]

# Closed-form reference for the default quadratic pair c(θ) = θ²/(2κ), θ*(y) = clamp(κy, 0, 1).
ref_value(W) = begin
    a = mx_KA * W; b = mx_KB * W; y = a .- b
    θ = clamp.(mx_κ .* y, 0.0, 1.0)
    (b .+ (θ .* y .- θ .^ 2 ./ (2mx_κ)), θ)
end

@testset "mixing — V = b + θ*·y − θ*²/(2κ) with θ* = clamp(κy, 0, 1)" begin
    stage   = MixingStage(mx_block; axis = :x, K_A = mx_KA, K_B = mx_KB, cost_curvature = mx_κ)
    V_start = backward!(stage, mx_W, NamedTuple())
    Vref, θref = ref_value(mx_W)
    @test V_start ≈ Vref
    @test policy(stage) ≈ θref

    # Brute max over a fine θ grid agrees (the conjugate identity IS the max, by construction).
    θgrid = range(0, 1; length = 2001)
    brute = [maximum(θ * (mx_KA * mx_W)[x] + (1 - θ) * (mx_KB * mx_W)[x] - θ^2 / (2mx_κ)
                     for θ in θgrid) for x in 1:mx_n]
    @test V_start ≈ brute atol = 1e-3
end

@testset "retention — K_A = I, pay to stay; θ* > 0 ⟺ staying dominates" begin
    W      = [4.0, 1.0, 2.5]                       # y = W − K_B·W clear of 0 on every cell
    stage  = RetentionStage(mx_block; axis = :x, exit_kernel = mx_KB, cost_curvature = mx_κ)
    V      = backward!(stage, W, NamedTuple())
    exitV  = mx_KB * W
    y      = W .- exitV
    θ      = clamp.(mx_κ .* y, 0.0, 1.0)
    @test V ≈ exitV .+ (θ .* y .- θ .^ 2 ./ (2mx_κ))
    @test policy(stage) ≈ θ
    @test all((policy(stage) .> 0) .== (y .> 0))   # pay to stay exactly where staying dominates
end

@testset "mixing — seated-policy duality ⟨V,Λ⟩ = ⟨−cost(θ*),Λ⟩ + ⟨W,Λ_end⟩" begin
    stage   = MixingStage(mx_block; axis = :x, K_A = mx_KA, K_B = mx_KB, cost_curvature = mx_κ)
    V_start = backward!(stage, mx_W, NamedTuple())
    Λ_start = [0.2, 0.5, 0.3]
    Λ_end   = forward!(stage, Λ_start)
    @test sum(Λ_end) ≈ sum(Λ_start)                # both corners row-stochastic ⇒ the mixture is
    # V = Kᵀ_{θ*}W − c(θ*) and forward is K_{θ*} — the cost read straight off the spec closure.
    cst = [stage.spec.cost(t; env = NamedTuple()) for t in policy(stage)]
    @test sum(V_start .* Λ_start) ≈ sum(-cst .* Λ_start) + sum(mx_W .* Λ_end)
end

@testset "mixing — forward-mode Dual pass-through; θ* rides the buffer eltype" begin
    stage  = MixingStage(mx_block; axis = :x, K_A = mx_KA, K_B = mx_KB, cost_curvature = mx_κ)
    dstage = lift_jacobian(stage)
    Dm     = Dual{tagtype(eltype(V_start_buffer(dstage)))}       # the lift owns the tag
    Vd = backward!(dstage, Dm.(mx_W, 1.0), NamedTuple())
    @test eltype(Vd) <: Dual
    @test eltype(policy(dstage)) == eltype(Vd)     # the eltype contract: θ* rides the buffer
    # A uniform V_end shift leaves y = a − b unchanged (row-stochastic corners) ⇒ dV = 1 exactly.
    @test all(partials.(Vd, 1) .≈ 1.0)
end

@testset "mixing — Dual value tangent in a NON-uniform direction matches FD" begin
    # The uniform-shift test above has ẏ ≡ 0; this direction moves y, so it kills a mutation that
    # freezes y in the value line (V's tangent must carry θ*·ẏ through the conjugate identity).
    dW = [1.0, -2.0, 0.5]
    mk() = MixingStage(mx_block; axis = :x, K_A = mx_KA, K_B = mx_KB, cost_curvature = mx_κ)
    dstage = lift_jacobian(mk())
    Dm = Dual{tagtype(eltype(V_start_buffer(dstage)))}
    Vd = backward!(dstage, Dm.(mx_W, dW), NamedTuple())
    h  = 1e-6
    fd = (backward!(mk(), mx_W .+ h .* dW, NamedTuple()) .-
          backward!(mk(), mx_W .- h .* dW, NamedTuple())) ./ (2h)
    @test maximum(abs, partials.(Vd, 1) .- fd) < 1e-6
end

@testset "mixing — the seated θ* carries the Fenchel tangent: κ·ẏ interior, EXACT 0 at either clamp" begin
    # This `W` puts one cell at each clamp and one interior (θ* = [1, 0, 0.2]), so a single fixture
    # exercises all three branches of the quadratic default's `θ*(y) = clamp(κy, 0, 1)`, whose
    # tangent is analytic: `κ·ẏ` where the clamp is slack and exactly zero where it binds.
    W, dW  = [3.0, 1.0, 2.0], [1.0, -2.0, 0.5]
    mk()   = MixingStage(mx_block; axis = :x, K_A = mx_KA, K_B = mx_KB, cost_curvature = mx_κ)
    dstage = lift_jacobian(mk())
    Dm     = Dual{tagtype(eltype(V_start_buffer(dstage)))}
    backward!(dstage, Dm.(W, dW), NamedTuple())
    θd     = policy(dstage)
    slack  = 0.0 .< value.(θd) .< 1.0
    @test count(slack) == 1 && count(value.(θd) .== 0.0) == 1 && count(value.(θd) .== 1.0) == 1
    @test partials.(θd, 1)[slack] ≈ (mx_κ .* ((mx_KA - mx_KB) * dW))[slack]
    @test all(iszero, partials.(θd, 1)[.!slack])   # a binding bound's tangent is EXACTLY zero

    envelope_vs_reoptimize(() -> (; stage = mk(), V_end = W, env = NamedTuple());
                           mode = :V_end, direction = dW, h = 1e-6, label = "Mixing")
end

@testset "mixing — construction-time argmax spot-probe on (cost, policy)" begin
    # A deliberately-inconsistent pair (right cost, wrong slope in the policy) must throw…
    @test_throws ErrorException MixingStage(mx_block; axis = :x, K_A = mx_KA, K_B = mx_KB,
                                            cost   = (θ; env) -> θ^2 / (2mx_κ),
                                            policy = (y; env) -> clamp(0.5 * mx_κ * y, 0.0, 1.0))
    # …as must a policy escaping [0, 1].
    @test_throws ErrorException MixingStage(mx_block; axis = :x, K_A = mx_KA, K_B = mx_KB,
                                            cost   = (θ; env) -> θ^2 / (2mx_κ),
                                            policy = (y; env) -> mx_κ * y)
    # An env-DEPENDENT pair can't be probed at construction (documented skip) — it must build
    # fine and still solve to the closed form once env arrives.
    stage = MixingStage(mx_block; axis = :x, K_A = mx_KA, K_B = mx_KB,
                        cost   = (θ; env) -> θ^2 / (2 * env.κ),
                        policy = (y; env) -> clamp(env.κ * y, 0.0, 1.0))
    V = backward!(stage, mx_W, (; κ = mx_κ))
    Vref, θref = ref_value(mx_W)
    @test V ≈ Vref
    @test policy(stage) ≈ θref
end

end # module
