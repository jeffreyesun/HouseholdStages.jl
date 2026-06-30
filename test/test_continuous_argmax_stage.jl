using Test
using HouseholdStages
using ForwardDiff

# Tests of the NEW `ContinuousArgmaxStage` primitive (end-goal §10) — distinct from
# `test_continuous_argmax.jl`, which exercises the discrete `ArgmaxStage`'s reward API.
#
# Correctness is shown two ways: (1) the off-grid value/policy match a DENSE-grid discrete
# reference of the SAME continuous objective `reward(x, a') + interp(V_end, a')` (as the reference
# grid refines, its discrete optimum → the continuous optimum the stage computes); (2) ForwardDiff
# flows through `backward!` wrt an `env` parameter, giving a finite derivative that matches a finite
# difference (the float policy/value responds smoothly to env — §13).

# Clamp-linear interpolation matching the stage's `_interp1d` and the InterpKernel `:clip` backward.
function interp_clamp(grid, v, a)
    n = length(grid)
    a <= grid[1] && return v[1]
    a >= grid[n] && return v[n]
    j = min(searchsortedlast(grid, a), n - 1)
    w = (a - grid[j]) / (grid[j+1] - grid[j])
    return (1 - w) * v[j] + w * v[j+1]
end

@testset "ContinuousArgmaxStage — value & policy match a dense-grid reference (1-D)" begin
    N      = 21
    grid   = collect(range(0.0, 10.0; length = N))
    layout = GriddedLayout(:a => GriddedContinuous(grid))
    # Concave reward whose target depends on the origin x; smooth concave continuation.
    reward = (a, a_next) -> -0.5 * (a_next - (0.5 * a + 1.0))^2
    stage  = ContinuousArgmaxStage(layout; reward = reward, axis = :a)

    V_end  = [-0.05 * (g - 5.0)^2 for g in grid]
    V_cont = copy(backward!(stage, V_end, NamedTuple()))
    pol    = copy(policy(stage))

    # Dense-grid reference: max over a fine a' of reward(x, a') + clamp-interp(V_end, a').
    fine    = collect(range(0.0, 10.0; length = 20_001))
    ref_val = similar(V_cont)
    ref_pol = similar(V_cont)
    for i in 1:N
        x = grid[i]
        best = -Inf; bk = 1
        for (k, ap) in enumerate(fine)
            val = -0.5 * (ap - (0.5 * x + 1.0))^2 + interp_clamp(grid, V_end, ap)
            val > best && (best = val; bk = k)
        end
        ref_val[i] = best
        ref_pol[i] = fine[bk]
    end

    @test V_cont ≈ ref_val atol = 1e-4          # value matches the dense reference
    @test pol    ≈ ref_pol atol = 1e-2          # recovered float policy ≈ dense-grid argmax
    @test eltype(pol) === Float64               # policy is a frozen float position
end

@testset "ContinuousArgmaxStage — reward dep axis threaded; matches dense reference (2-D)" begin
    Na     = 16
    grid   = collect(range(0.0, 8.0; length = Na))
    zlev   = [1.0, 2.0]
    layout = GriddedLayout(:a => GriddedContinuous(grid), :z => Discrete(zlev))
    # Reward varies along the discrete dep axis z (a declared kwarg → a field dep).
    reward = (a, a_next; z) -> -0.5 * (a_next - (0.4 * a + z))^2
    stage  = ContinuousArgmaxStage(layout; reward = reward, axis = :a)

    V_end  = [-0.05 * (g - 4.0)^2 + 0.1 * zz for g in grid, zz in zlev]
    V_cont = copy(backward!(stage, V_end, NamedTuple()))
    pol    = copy(policy(stage))

    fine = collect(range(0.0, 8.0; length = 16_001))
    for j in 1:2, i in 1:Na
        x = grid[i]
        best = -Inf; bk = 1
        for (k, ap) in enumerate(fine)
            val = -0.5 * (ap - (0.4 * x + zlev[j]))^2 + interp_clamp(grid, view(V_end, :, j), ap)
            val > best && (best = val; bk = k)
        end
        @test V_cont[i, j] ≈ best     atol = 1e-4
        @test pol[i, j]    ≈ fine[bk] atol = 1e-2
    end
end

@testset "ContinuousArgmaxStage — AD through backward! matches finite differences (§13)" begin
    N      = 41
    grid   = collect(range(0.0, 10.0; length = N))
    layout = GriddedLayout(:a => GriddedContinuous(grid))
    θ      = 2.0
    slope  = 0.5
    # objective(a') = -0.5θ(a' - t)² + slope·a' (linear continuation), so the interior optimum is
    # a'* = t + slope/θ and V* = slope·t + 0.5·slope²/θ (constant across origins). dV*/dt = slope.
    reward = (a, a_next; env) -> -0.5 * θ * (a_next - env.t)^2
    stage  = ContinuousArgmaxStage(layout; reward = reward, axis = :a)
    V_end  = slope .* grid
    t0     = 5.0
    Vstar_cf = slope * t0 + 0.5 * slope^2 / θ

    # Primal: closed-form value and policy.
    Vp = copy(backward!(stage, V_end, (t = t0,)))
    @test all(isapprox.(Vp, Vstar_cf; atol = 1e-6))
    @test all(isapprox.(policy(stage), t0 + slope / θ; atol = 1e-2))

    # Forward-mode AD: a finite, sensible derivative of the value wrt env.t.
    sd     = lift_jacobian(stage; mode = :forward, n_dual = 1)
    D      = eltype(sd.scratch.V_start)
    t_dual = D(t0, ForwardDiff.Partials((1.0,)))
    Vd     = backward!(sd, V_end, (t = t_dual,))
    dV_ad  = ForwardDiff.partials(sum(Vd))[1]

    # Central finite difference on the Float64 stage.
    h       = 1e-6
    Vp_plus  = sum(copy(backward!(stage, V_end, (t = t0 + h,))))
    Vp_minus = sum(copy(backward!(stage, V_end, (t = t0 - h,))))
    dV_fd    = (Vp_plus - Vp_minus) / (2h)

    @test isfinite(dV_ad)
    @test dV_ad ≈ dV_fd     rtol = 1e-5
    @test dV_ad ≈ N * slope rtol = 1e-5         # envelope theorem: dV*/dt = slope per cell
end

@testset "ContinuousArgmaxStage — forward conserves mass; K/Kᵀ duality" begin
    N      = 21
    grid   = collect(range(0.0, 10.0; length = N))
    layout = GriddedLayout(:a => GriddedContinuous(grid))
    reward = (a, a_next) -> -0.5 * (a_next - (0.5 * a + 1.0))^2
    stage  = ContinuousArgmaxStage(layout; reward = reward, axis = :a)

    V_end   = [-0.05 * (g - 5.0)^2 for g in grid]
    V_start = copy(backward!(stage, V_end, NamedTuple()))

    Λ0 = rand(N); Λ0 ./= sum(Λ0)
    Λ1 = copy(forward!(stage, Λ0))
    @test sum(Λ1) ≈ 1.0 atol = 1e-12            # Young-split conserves mass

    # Linear-part duality of the seated InterpKernel: ⟨KᵀV_end·part, ·⟩ via the forward adjoint.
    dΛ  = randn(N)
    dΛ0 = forward_adjoint!(stage, dΛ)
    @test sum(Λ1 .* dΛ) ≈ sum(Λ0 .* dΛ0) atol = 1e-12
end
