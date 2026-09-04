using Test
using HouseholdStages
using ForwardDiff

include("envelope_oracle.jl")

# Tests of the `ContinuousArgmaxStage` primitive (end-goal §10) — distinct from
# `test_argmax.jl`, which exercises the discrete `ArgmaxStage`'s reward API.
#
# Correctness is read against two references. The POLICY is held to a DENSE-grid discrete argmax of
# the SAME continuous objective `reward(x, a') + interp(V_end, a')`: as the reference grid refines,
# its discrete optimum → the continuous optimum the vertex fits. The VALUE is held to the exact
# on-grid NODE reference `maxₖ reward(x, gₖ) + V_end[k]` — the stage's node-value contract. Where the
# parabola model is EXACT — a quadratic reward against a linear continuation — the seated policy and
# its `env`-tangent are analytic and pinned as such.

# Clamp-linear interpolation matching the InterpKernel clip backward.
function interp_clamp(grid, v, a)
    n = length(grid)
    a <= grid[1] && return v[1]
    a >= grid[n] && return v[n]
    j = min(searchsortedlast(grid, a), n - 1)
    w = (a - grid[j]) / (grid[j+1] - grid[j])
    return (1 - w) * v[j] + w * v[j+1]
end

# The exact node-value reference: maxₖ reward(x, gₖ) + V_end[k] per origin.
node_ref(reward, grid, V_end) =
    [maximum(reward(x, grid[k]) + V_end[k] for k in eachindex(grid)) for x in grid]

@testset "ContinuousArgmaxStage — node value exact; policy matches a dense-grid reference (1-D)" begin
    N      = 21
    grid   = collect(range(0.0, 10.0; length = N))
    layout = GriddedLayout(:a => GriddedContinuous(grid))
    # Concave reward whose target depends on the origin x; smooth concave continuation.
    reward = (a, a_next) -> -0.5 * (a_next - (0.5 * a + 1.0))^2
    stage  = ContinuousArgmaxStage(layout; reward = reward, axis = :a)

    V_end = [-0.05 * (g - 5.0)^2 for g in grid]
    V     = copy(backward!(stage, V_end, NamedTuple()))
    pol   = copy(policy(stage))

    # The VALUE is the on-grid node value — exact against the brute node reference.
    @test V ≈ node_ref(reward, grid, V_end) atol = 1e-12

    # Dense-grid reference for the POLICY: argmax over a fine a' of reward(x, a') + clamp-interp.
    # The parabolic vertex differs from the true continuous optimum by the scheme's approximation
    # error (the interpolant has a kink the parabola cannot see), so agreement is O(spacing) —
    # ≈0.02 here. The §13 testset, whose continuation is linear (objective a true parabola), pins
    # the vertex EXACTLY at tight tolerance.
    fine    = collect(range(0.0, 10.0; length = 20_001))
    ref_pol = similar(V)
    for i in 1:N
        x = grid[i]
        best = -Inf; bk = 1
        for (k, ap) in enumerate(fine)
            val = -0.5 * (ap - (0.5 * x + 1.0))^2 + interp_clamp(grid, V_end, ap)
            val > best && (best = val; bk = k)
        end
        ref_pol[i] = fine[bk]
    end
    @test pol ≈ ref_pol atol = 8e-2             # recovered float policy ≈ dense-grid argmax

    # The policy's eltype is the buffer's: a Float64 chain seats a float, a Dual rebuild seats the
    # vertex carrying its tangent, with the primal lane untouched to the bit.
    @test eltype(pol) === Float64
    lift = lift_jacobian(stage; n_dual = 1)
    D    = eltype(V_start_buffer(lift))
    backward!(lift, D.(V_end, ForwardDiff.Partials.(tuple.(ones(N)))), NamedTuple())
    @test eltype(policy(lift)) === D
    @test ForwardDiff.value.(policy(lift)) == pol
end

@testset "ContinuousArgmaxStage — reward dep axis threaded; node value exact (2-D)" begin
    Na     = 16
    grid   = collect(range(0.0, 8.0; length = Na))
    zlev   = [1.0, 2.0]
    layout = GriddedLayout(:a => GriddedContinuous(grid), :z => Discrete(zlev))
    # Reward varies along the discrete dep axis z (a declared kwarg → a field dep).
    reward = (a, a_next; z) -> -0.5 * (a_next - (0.4 * a + z))^2
    stage  = ContinuousArgmaxStage(layout; reward = reward, axis = :a)

    V_end = [-0.05 * (g - 4.0)^2 + 0.1 * zz for g in grid, zz in zlev]
    V     = copy(backward!(stage, V_end, NamedTuple()))
    pol   = copy(policy(stage))

    fine = collect(range(0.0, 8.0; length = 16_001))
    for j in 1:2, i in 1:Na
        x = grid[i]
        # Exact node value per (origin, z) cell.
        vnode = maximum(-0.5 * (grid[k] - (0.4 * x + zlev[j]))^2 + V_end[k, j] for k in 1:Na)
        @test V[i, j] ≈ vnode atol = 1e-12
        best = -Inf; bk = 1
        for (k, ap) in enumerate(fine)
            val = -0.5 * (ap - (0.4 * x + zlev[j]))^2 + interp_clamp(grid, view(V_end, :, j), ap)
            val > best && (best = val; bk = k)
        end
        @test pol[i, j] ≈ fine[bk] atol = 3e-2
    end
end

@testset "ContinuousArgmaxStage — pow2+1 grid (N = 17): the general walk is exact" begin
    # Safety net for the shared divide-and-conquer walk at `N = 2^k + 1`: a target-chasing reward
    # whose optimal a' EXCEEDS a at low a, which any walk capping the choice index at the origin
    # index mis-solves. The divide-and-conquer walk makes no such triangular assumption.
    N      = 17
    grid   = collect(range(0.0, 10.0; length = N))
    layout = GriddedLayout(:a => GriddedContinuous(grid))
    reward = (a, a_next) -> -0.5 * (a_next - (0.5 * a + 1.0))^2
    stage  = ContinuousArgmaxStage(layout; reward = reward, axis = :a)

    V_end = [-0.05 * (g - 5.0)^2 for g in grid]
    V     = copy(backward!(stage, V_end, NamedTuple()))
    @test V ≈ node_ref(reward, grid, V_end) atol = 1e-12
end

@testset "ContinuousArgmaxStage — supermodularity guard throws at fill time" begin
    grid   = collect(range(0.0, 10.0; length = 9))
    layout = GriddedLayout(:a => GriddedContinuous(grid))
    sub    = (a, a_next) -> -a * a_next                  # strictly SUBmodular (∂²r/∂a∂a' = −1)

    # Env-INDEPENDENT reward: the face fills at construction, so the guard throws there …
    @test_throws ErrorException ContinuousArgmaxStage(layout; reward = sub, axis = :a)
    # … unless `skip_monotonicity_check = true` skips it (at the user's own risk).
    @test ContinuousArgmaxStage(layout; reward = sub, axis = :a,
                                skip_monotonicity_check = true) isa ContinuousArgmaxStage

    # Env-DEPENDENT reward: the guard tracks each refill — a supermodular env passes, then the
    # same stage throws when the refilled face turns submodular.
    st = ContinuousArgmaxStage(layout; reward = (a, a_next; env) -> env.s * a * a_next, axis = :a)
    V_end = zeros(length(grid))
    @test isfinite(sum(backward!(st, V_end, (s = 1.0,))))                # supermodular fill: solves
    @test_throws ErrorException backward!(st, V_end, (s = -1.0,))        # submodular refill: refused

    # `ConsumptionSavingsStage` inherits the guard, so an S-shaped utility — concave nowhere near
    # the inflexion, and nothing in the package checks the user's closure — is refused rather than
    # silently mis-solved by the monotone walk.
    L    = GriddedLayout(:wealth => GriddedContinuous(collect(range(0.5, 6.0; length = 25))))
    ushp = (cell, c) -> -1 / c + 0.6 * atan(4 * (c - 2))
    @test_throws ErrorException ConsumptionSavingsStage(L; β = 0.95, utility = ushp)
    @test ConsumptionSavingsStage(L; β = 0.95, utility = ushp,
                                  skip_monotonicity_check = true) isa HouseholdStages.ChainStage
end

@testset "ContinuousArgmaxStage — AD through backward! matches finite differences (§13)" begin
    N      = 41
    grid   = collect(range(0.0, 10.0; length = N))
    layout = GriddedLayout(:a => GriddedContinuous(grid))
    θ      = 2.0
    slope  = 0.5
    # objective(a') = -0.5θ(a' - t)² + slope·a' (linear continuation), so the interior optimum is
    # a'* = t + slope/θ. At t0 = 5.0 that is 5.25 — exactly a grid node, so the node argmax `bestk`
    # is FD-stable and the node objective is a true parabola the vertex fits EXACTLY.
    reward = (a, a_next; env) -> -0.5 * θ * (a_next - env.t)^2
    stage  = ContinuousArgmaxStage(layout; reward = reward, axis = :a)
    V_end  = slope .* grid
    t0     = 5.0

    # Primal: node value against the exact node reference; policy pinned to the closed form.
    Vp = copy(backward!(stage, V_end, (t = t0,)))
    vnode = maximum(-0.5 * θ * (g - t0)^2 + slope * g for g in grid)
    @test all(isapprox.(Vp, vnode; atol = 1e-12))
    @test all(isapprox.(policy(stage), t0 + slope / θ; atol = 1e-8))   # vertex exact (linear continuation)

    # Forward-mode AD: the node-value tangent wrt env.t matches a central finite difference (both
    # sides node-valued; bestk is locally stable at t0).
    sd     = lift_jacobian(stage; n_dual = 1)
    D      = eltype(sd.scratch.V_start)
    t_dual = D(t0, ForwardDiff.Partials((1.0,)))
    Vd     = backward!(sd, V_end, (t = t_dual,))
    dV_ad  = ForwardDiff.partials(sum(Vd))[1]

    h        = 1e-6
    Vp_plus  = sum(copy(backward!(stage, V_end, (t = t0 + h,))))
    Vp_minus = sum(copy(backward!(stage, V_end, (t = t0 - h,))))
    dV_fd    = (Vp_plus - Vp_minus) / (2h)

    @test isfinite(dV_ad)
    @test dV_ad ≈ dV_fd rtol = 1e-5
end

@testset "ContinuousArgmaxStage — the vertex policy is exact where the parabola model is" begin
    # A quadratic reward against a LINEAR continuation makes the objective a true parabola on the
    # whole interval, so the three-node fit IS that parabola and its vertex is the analytic
    # continuous argmax `a'* = t + slope/θ` — exactly, not to O(spacing). `t0` is chosen so
    # `a'* = 5.35` falls strictly BETWEEN nodes (spacing 0.25), which is where an on-grid answer and
    # the continuous one part company: the seated node value is 1.00e-2 short of the continuous
    # maximum here and its `t`-tangent is `θ(g_k − t0) = 0.3` against the continuous `slope = 0.5`,
    # the `O(spacing)` value bias measured on the fixture that makes everything else exact.
    N      = 41
    grid   = collect(range(0.0, 10.0; length = N))
    layout = GriddedLayout(:a => GriddedContinuous(grid))
    θ, slope, t0 = 2.0, 0.5, 5.1
    stage  = ContinuousArgmaxStage(layout;
                                   reward = (a, a_next; env) -> -0.5 * θ * (a_next - env.t)^2,
                                   axis = :a)
    V_end  = slope .* grid

    V      = copy(backward!(stage, V_end, (t = t0,)))
    astar  = t0 + slope / θ
    vstar  = -0.5 * θ * (astar - t0)^2 + slope * astar     # the analytic continuous maximum
    @test all(isapprox.(policy(stage), astar; atol = 1e-12))
    vnode  = maximum(-0.5 * θ * (g - t0)^2 + slope * g for g in grid)
    @test all(isapprox.(V, vnode; atol = 1e-12))           # the seated value is the node max …
    @test isapprox(vstar - vnode, 1.0e-2; atol = 1e-12)    # … short of the continuous max by O(Δ)

    # The vertex tangent, exactly `∂a'*/∂t = 1` per cell; the value tangent is the node's `0.3`.
    lift   = lift_jacobian(stage; n_dual = 1)
    D      = eltype(V_start_buffer(lift))
    Vd     = copy(backward!(lift, V_end, (t = D(t0, ForwardDiff.Partials((1.0,))),)))
    @test all(isapprox.(ForwardDiff.partials.(policy(lift), 1), 1.0; atol = 1e-12))
    @test all(isapprox.(ForwardDiff.partials.(Vd, 1), 0.3; atol = 1e-12))
end

@testset "ContinuousArgmaxStage — envelope vs reoptimize, both directions" begin
    # One CA stage with an env-declaring reward and a concave continuation: the P4 direction seeds
    # `env`, the P3 direction seeds `V_end`. Both compare the seated tangents against central
    # differences of the RE-SOLVED primal. `h = 1e-5` sits in the flat of the h-scan — truncation
    # dominates above it, cancellation below.
    N    = 21
    grid = collect(range(0.0, 10.0; length = N))
    θ    = 1.5
    build() = (; stage = ContinuousArgmaxStage(GriddedLayout(:a => GriddedContinuous(grid));
                                               reward = (a, ap; env) -> -0.5 * θ * (ap - (0.5 * a + env.t))^2,
                                               axis = :a),
                 V_end = [-0.05 * (g - 5.0)^2 for g in grid],
                 env   = (t = 2.0,))

    p4 = envelope_vs_reoptimize(build; mode = :env, direction = (t = 1.0,), h = 1e-5,
                                label = "ContinuousArgmax")
    p3 = envelope_vs_reoptimize(build; mode = :V_end, direction = cos.(grid ./ 3), h = 1e-5,
                                label = "ContinuousArgmax")

    # Family-local: the vertex tangent is a genuine channel, not a zero — a policy seated through
    # primals would report exactly zero on both directions and pass the comparison anyway.
    @test maximum(abs, p4.policy_ad) > 0.1
    @test maximum(abs, p3.policy_ad) > 1e-3
    @test maximum(abs, p4.value_ad)  > 0.1
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

# The two-mode lottery #
#----------------------#
# A bin whose image under `a*(·)` is disconnected splits its mass across the two modes. The fixtures
# below make the reward LINEAR in the origin, so the mode value gap `Φ(x) = f(x, k2) − f(x, k1)` is
# exactly linear and its root is the true crossing of the two modes — every quantity checked here is
# therefore an analytic number, not a re-statement of the code under test.

# Two wells at a' = 0.2 and 0.8; the `−b·a'` tilt puts the argmax crossing at x = b/θ.
two_well_reward(x, ap; env) = env.θ * x * ap - env.b * ap - 40 * (ap - 0.2)^2 * (ap - 0.8)^2

# Three wells at a' = 0.2, 0.5, 0.8, at heights `0, −0.162, −0.330`, so the argmax crosses twice —
# at x = 0.54 and x = 0.56, which the `spacing = 0.05` origin grid puts inside ONE bin.
_well3_tilt(ap) = -(1 / 30) * ap^2 - (31 / 60) * ap + 157 / 1500
three_well_reward(x, ap; env) =
    env.θ * x * ap - 300 * (ap - 0.2)^2 * (ap - 0.5)^2 * (ap - 0.8)^2 + _well3_tilt(ap)

# The bin `[l, r]` around origin node `c`: midpoints, truncated at the grid's ends.
function origin_bin(g, c)
    n = length(g)
    return (c == 1 ? g[1] : (g[c-1] + g[c]) / 2, c == n ? g[n] : (g[c] + g[c+1]) / 2)
end

bimodal_stage(reward, g) =
    ContinuousArgmaxStage(GriddedLayout(:a => GriddedContinuous(g)); reward, axis = :a)

@testset "ContinuousArgmaxStage — a straddling bin splits across the two modes" begin
    g     = collect(range(0.0, 1.0; length = 21))
    env   = (; θ = 1.0, b = 0.54)
    stage = bimodal_stage(two_well_reward, g)
    backward!(stage, zeros(21), env)

    k1, k2, c = 5, 17, 12                                   # a' = 0.2 and 0.8; the bin holding x̄
    @test stage.scratch.bestk[c-1] == k1                    # the walk switches across this pair
    @test stage.scratch.bestk[c]   == k2
    @test (stage.kernel.lo[c], stage.kernel.hi[c]) == (Int32(k1), Int32(k2))

    # The crossing is analytic: Φ(x) = (θ·x − b)·(g[k2] − g[k1]), so x̄ = b/θ.
    x̄    = env.b / env.θ
    l, r = origin_bin(g, c)
    w    = (x̄ - l) / (r - l)
    @test 0 < w < 1
    @test w ≈ 0.3 atol = 1e-12

    # x̄ IS the flip: the two modes swap rank across it, which is what the split is measuring.
    f(x, k) = two_well_reward(x, g[k]; env)
    @test f(x̄ - 1e-6, k1) > f(x̄ - 1e-6, k2)
    @test f(x̄ + 1e-6, k1) < f(x̄ + 1e-6, k2)

    # Λ: the straddling bin's mass divides by the fraction either side of the crossing.
    Λ0 = zeros(21); Λ0[c] = 1.0
    Λ1 = copy(forward!(stage, Λ0))
    @test Λ1[k1] ≈ w     atol = 1e-13
    @test Λ1[k2] ≈ 1 - w atol = 1e-13
    @test sum(Λ1) ≈ 1.0  atol = 1e-14
    @test count(!iszero, Λ1) == 2

    # The stored position is the pair's barycentre — the cell's mean destination.
    @test policy(stage)[c] ≈ w * g[k1] + (1 - w) * g[k2] atol = 1e-13

    # A steep but single-peaked policy jumps several nodes per origin step and must NOT split: the
    # jump test alone fires on it, the interior-valley gate is what declines it.
    steep = bimodal_stage((a, a_next) -> -0.5 * (a_next - (5 * a - 1.0))^2, g)
    backward!(steep, zeros(21), NamedTuple())
    @test maximum(diff(steep.scratch.bestk)) > 1                    # the jump test would fire
    @test all(steep.kernel.hi .- steep.kernel.lo .<= Int32(1))      # every pair stays adjacent
end

@testset "ContinuousArgmaxStage — the split's env-tangent, analytic and against re-solved FD" begin
    g      = collect(range(0.0, 1.0; length = 21))
    k1, k2, c = 5, 17, 12
    l, r   = origin_bin(g, c)
    θ0, b0 = 1.0, 0.54
    Λ0     = zeros(21); Λ0[c] = 1.0

    push_Λ(env) = (st = bimodal_stage(two_well_reward, g);
                   backward!(st, zeros(eltype(Λ0), 21), env); copy(forward!(st, Λ0)))

    # x̄ = b/θ, so ∂w/∂θ = −b/(θ²·(r − l)) — the whole tangent lives in the split weight.
    dw_dθ = -b0 / (θ0^2 * (r - l))
    @test dw_dθ ≈ -10.8 atol = 1e-12

    stage_d = bimodal_stage(two_well_reward, g)
    chain_d = lift_jacobian(stage_d; n_dual = 1)
    D       = eltype(V_start_buffer(chain_d))
    backward!(chain_d, zeros(D, 21), (; θ = D(θ0, ForwardDiff.Partials((1.0,))), b = b0))
    Λ1 = copy(forward!(chain_d, D.(Λ0)))

    @test ForwardDiff.value.(Λ1) ≈ push_Λ((; θ = θ0, b = b0)) atol = 1e-13
    @test ForwardDiff.partials(Λ1[k1], 1) ≈  dw_dθ atol = 1e-10
    @test ForwardDiff.partials(Λ1[k2], 1) ≈ -dw_dθ atol = 1e-10
    @test sum(ForwardDiff.partials.(Λ1, 1)) ≈ 0.0 atol = 1e-10   # the flux moves mass, never makes it

    # Central differences of the RE-SOLVED primal: the detector's firing set is invariant over the
    # step, so the difference does not straddle a change of seating.
    h  = 1e-5
    fd = (push_Λ((; θ = θ0 + h, b = b0)) .- push_Λ((; θ = θ0 - h, b = b0))) ./ (2h)
    @test fd[k1] ≈ ForwardDiff.partials(Λ1[k1], 1) rtol = 1e-6
    @test fd[k2] ≈ ForwardDiff.partials(Λ1[k2], 1) rtol = 1e-6
end

@testset "ContinuousArgmaxStage — two switches in one bin: the right-hand one seats" begin
    g     = collect(range(0.0, 1.0; length = 21))
    stage = bimodal_stage(three_well_reward, g)
    backward!(stage, zeros(21), (; θ = 1.0))

    c = 12
    @test stage.scratch.bestk[c-1:c+1] == [5, 11, 17]        # two jumps, both wider than one node
    @test (stage.kernel.lo[c], stage.kernel.hi[c]) == (Int32(11), Int32(17))   # right-hand survives

    l, r = origin_bin(g, c)
    Λ0   = zeros(21); Λ0[c] = 1.0
    Λ1   = copy(forward!(stage, Λ0))
    @test Λ1[11] ≈ (0.56 - l) / (r - l) atol = 1e-11
    @test sum(Λ1) ≈ 1.0 atol = 1e-14
end

@testset "ContinuousArgmaxStage — a feasibility cliff is declined, not split" begin
    # The scope limit, in the direction borrowing constraints and default boundaries take: the far
    # mode is infeasible at the left origin (`f(k2; i) = −Inf`), so there is no valley across the
    # cliff and the secant would be degenerate anyway (`Φ₀ = −Inf`, `Φ₀ − Φ₁ = −Inf`, `x̄ = NaN`).
    # The bracket keeps its single-mode seating.
    g     = collect(range(0.0, 1.0; length = 21))
    rew   = (x, ap; env) -> ap > x + env.slack ? -Inf :
                            (x * ap - 0.2 * ap - 40 * (ap - 0.2)^2 * (ap - 0.8)^2)
    stage = bimodal_stage(rew, g)
    backward!(stage, zeros(21), (; slack = 0.35))

    bk = stage.scratch.bestk
    i  = only(findall(j -> bk[j+1] - bk[j] > 1, 1:20))          # one bracket, eleven nodes wide
    @test bk[i+1] - bk[i] == 11
    @test !isfinite(stage.scratch.U.array[bk[i+1], i])           # the far mode is infeasible here
    @test !HouseholdStages._caC_has_valley(stage.scratch.U.array, zeros(21), bk[i], bk[i+1], i)
    @test all(stage.kernel.hi .- stage.kernel.lo .<= Int32(1))   # nothing seated as a two-mode pair
    @test !any(isnan, policy(stage))                             # and no NaN reaches the kernel
end
