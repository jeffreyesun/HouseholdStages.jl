# GPU kernel verification for the two stages with hand-written CUDA kernels:
# ConsumptionSavingsStage and WealthChangeStage (HouseholdStagesCUDAExt).
#
# For each case we build a CPU stage, run backward!/forward! to get a Float64
# reference, move the stage to the GPU via the production `to_device`, re-run on
# CuArrays, and assert a tight match (~1e-10). We cover:
#   * WealthChange with the wealth axis leading and non-leading;
#   * ConsumptionSavings with n_w-1 a power of two (exercises the iterative
#     reference k1_argmax kernel) and not (exercises the sequential fallback),
#     and with the wealth axis non-leading.

using HouseholdStages
using CUDA
using Test
using Printf
using Random

const HS = HouseholdStages

CUDA.allowscalar(false)

# Float64-preserving device mover (CUDA.cu would degrade to Float32).
to_dev(x::AbstractArray) = CuArray(collect(x))
to_dev(x)               = x
gpu_stage(stage::AbstractStage) = to_device(stage, to_dev)

# Tight match that lets matching ±Inf entries (borrowing-constraint cells) pass.
function match_tight(a, b; atol = 1e-10, rtol = 1e-9)
    size(a) == size(b) || return (false, Inf)
    m = 0.0
    for (x, y) in zip(a, b)
        if isinf(x) || isinf(y)
            x == y || return (false, Inf)
        else
            d = abs(x - y)
            m = max(m, d)
            isapprox(x, y; atol = atol, rtol = rtol) || return (false, m)
        end
    end
    return (true, m)
end

# Run one stage CPU vs GPU, returning (bw_ok, bw_Δ, fw_ok, fw_Δ).
function cpu_gpu_compare(stage, V_end, Λ_start, env)
    V_cpu = copy(backward!(stage, V_end, env))
    Λ_cpu = copy(forward!(stage, Λ_start))

    gstage = gpu_stage(stage)
    V_gpu  = Array(backward!(gstage, to_dev(V_end), env))
    Λ_gpu  = Array(forward!(gstage, to_dev(Λ_start)))

    bw_ok, bw_d = match_tight(V_gpu, V_cpu)
    fw_ok, fw_d = match_tight(Λ_gpu, Λ_cpu)
    return (bw_ok, bw_d, fw_ok, fw_d)
end

# --- builders -------------------------------------------------------------

# WealthChange, wealth axis leading.
function build_wc_leading()
    layout = GriddedLayout(
        :wealth => GriddedContinuous([0.0, 1.0, 2.0, 3.0, 4.0]),
        :income => Discrete([0.5, 1.0, 1.5]),
    )
    stage = WealthChangeStage(layout;
        wealth_post = (; wealth, env) -> (1 + env.r) * wealth - 0.1,
        axis = :wealth)
    V_end = [0.3 * w + 0.1 * y for w in 1:5, y in 1:3]
    Λ = rand(5, 3); Λ ./= sum(Λ)
    (stage, V_end, Λ, (r = 0.04,))
end

# WealthChange, wealth axis NOT leading (income first).
function build_wc_nonleading()
    layout = GriddedLayout(
        :income => Discrete([0.5, 1.0, 1.5]),
        :wealth => GriddedContinuous([0.0, 1.0, 2.0, 3.0, 4.0]),
    )
    stage = WealthChangeStage(layout;
        wealth_post = (; wealth, env) -> (1 + env.r) * wealth - 0.1,
        axis = :wealth)
    V_end = [0.1 * y + 0.3 * w for y in 1:3, w in 1:5]
    Λ = rand(3, 5); Λ ./= sum(Λ)
    (stage, V_end, Λ, (r = 0.04,))
end

# ConsumptionSavings with n_w-1 NOT a power of two (sequential fallback kernel).
function build_cs_seq(n_w = 16)
    layout = GriddedLayout(
        :wealth => GriddedContinuous(
            [exp(t) - 1.0 for t in range(0.0, log(21.0); length = n_w)]),
        :income => Discrete([0.6, 1.0, 1.4]),
    )
    stage = ConsumptionSavingsStage(layout; β = 0.96,
        utility = (cell, c; env) -> log(c), axis = :wealth)
    V_end = [0.1 * w + 0.05 * y for w in 1:n_w, y in 1:3]
    Λ = rand(n_w, 3); Λ ./= sum(Λ)
    (stage, V_end, Λ, NamedTuple())
end

# ConsumptionSavings with n_w-1 a power of two (iterative reference kernel).
function build_cs_pow2(n_w = 17)
    @assert ispow2(n_w - 1)
    layout = GriddedLayout(
        :wealth => GriddedContinuous(
            [exp(t) - 1.0 for t in range(0.0, log(21.0); length = n_w)]),
        :income => Discrete([0.6, 1.0, 1.4]),
    )
    stage = ConsumptionSavingsStage(layout; β = 0.96,
        utility = (cell, c; env) -> log(c), axis = :wealth)
    V_end = [0.1 * w + 0.05 * y for w in 1:n_w, y in 1:3]
    Λ = rand(n_w, 3); Λ ./= sum(Λ)
    (stage, V_end, Λ, NamedTuple())
end

# ConsumptionSavings, wealth axis NOT leading.
function build_cs_nonleading(n_w = 17)
    @assert ispow2(n_w - 1)
    layout = GriddedLayout(
        :income => Discrete([0.6, 1.0, 1.4]),
        :wealth => GriddedContinuous(
            [exp(t) - 1.0 for t in range(0.0, log(21.0); length = n_w)]),
    )
    stage = ConsumptionSavingsStage(layout; β = 0.96,
        utility = (cell, c; env) -> log(c), axis = :wealth)
    V_end = [0.05 * y + 0.1 * w for y in 1:3, w in 1:n_w]
    Λ = rand(3, n_w); Λ ./= sum(Λ)
    (stage, V_end, Λ, NamedTuple())
end

# --- :brute argmax builders (buy/sell-home) -------------------------------
# The discrete `(max, +)` argmax over an UNORDERED axis (housing). These exercise the
# GPU `:brute` kernel in the ext (the only JMP-fresh backward gap on device). Each uses a
# TIE-HEAVY integer-valued V_end (exact ties across the housing axis) so the first-index
# tie-break is stressed: the policy index must match CPU bit-identically, not just the value.

# Housing axis at the front (WD == 1) vs non-leading (WD == 3, pre > 1 — the real JMP layout
# where housing sits behind wealth × income). `_to_front`/`_from_front!` are exercised by the
# non-leading case; the leading case takes the no-permute branch.
function _house_layout(hpos::Int)
    h = Discrete([0.0, 1.0, 2.0, 3.0])
    w = GriddedContinuous(collect(range(0.0, 4.0; length = 6)))
    y = Discrete([0.6, 1.0, 1.4])
    return hpos == 1 ?
        GriddedLayout(:h => h, :wealth => w, :income => y) :
        GriddedLayout(:wealth => w, :income => y, :h => h)
end

# Tie-heavy continuation: small-integer values ⇒ many EXACT ties across housing, so a renter's
# argmax frequently lands on a tied set — the policy must resolve to the first (lowest) index.
function _tie_heavy_Vend(layout)
    Random.seed!(20260629)
    return float.(rand(0:2, axissize.(layout.axes)...))
end

function build_buyhome(; hpos::Int)
    layout = _house_layout(hpos)
    stage  = BuyHomeStage(layout; axis = :h, renter_index = 1)
    V_end  = _tie_heavy_Vend(layout)
    Λ = rand(axissize.(layout.axes)...); Λ ./= sum(Λ)
    (stage, V_end, Λ, nothing)
end

function build_sellhome(; hpos::Int)
    layout = _house_layout(hpos)
    stage  = SellHomeStage(layout; axis = :h, renter_index = 1)
    V_end  = _tie_heavy_Vend(layout)
    Λ = rand(axissize.(layout.axes)...); Λ ./= sum(Λ)
    (stage, V_end, Λ, nothing)
end

const BRUTE_CASES = [
    ("BuyHome  (housing leading)",     () -> build_buyhome(hpos = 1)),
    ("BuyHome  (housing non-leading)", () -> build_buyhome(hpos = 3)),
    ("SellHome (housing leading)",     () -> build_sellhome(hpos = 1)),
    ("SellHome (housing non-leading)", () -> build_sellhome(hpos = 3)),
]

# Run one :brute stage CPU vs GPU. Returns (vΔ exact, policy-mismatch count, fw_ok, fw_Δ).
# Value AND policy must be bit-identical (exact); the forward scatter uses the tight match
# (float += collision order differs between the CPU loop and the per-column GPU kernel).
function brute_cpu_gpu_compare(stage, V_end, Λ_start, env)
    V_cpu   = copy(backward!(stage, V_end, env))
    pol_cpu = copy(policy(stage))
    Λ_cpu   = copy(forward!(stage, Λ_start))

    gstage  = gpu_stage(stage)
    V_gpu   = Array(backward!(gstage, to_dev(V_end), env))
    pol_gpu = Array(policy(gstage))
    Λ_gpu   = Array(forward!(gstage, to_dev(Λ_start)))

    vΔ    = maximum(abs.(V_gpu .- V_cpu))          # gate guarantees a finite value everywhere
    polmis = count(pol_gpu .!= pol_cpu)
    fw_ok, fw_d = match_tight(Λ_gpu, Λ_cpu)
    return (vΔ, polmis, fw_ok, fw_d)
end

# --- choice-collapse forward builders (grown scatter axis) ----------------
# The forward scatter axis GROWS from source to destination: these stages collapse a
# size-1 choice axis whose ArgmaxStage forward grew it (source 1 → destination 2). They
# exercise the `_cs_forward_scatter!` path where `size(Λ_start, WD) ≠ size(Λ_end, WD)`
# (the bug the ext fix targets). Source has one row per column ⇒ each destination cell is
# written once, so the forward is bit-identical to the CPU scatter (no collision order).

# EndogenousExit: collapse on :exiting (the survivor/exit choice).
function build_exit_collapse()
    layout = GriddedLayout(:x => Discrete([1, 2, 3, 4]), :exiting => Discrete([0]))
    stage  = EndogenousExit(layout; bequest = 2.0)
    V_end  = reshape([3.0, 2.5, 1.0, 2.0], 4, 1)
    Λ      = reshape([0.1, 0.2, 0.3, 0.4], 4, 1)
    (stage, V_end, Λ, NamedTuple())
end

# KernelChoiceStage: collapse on :θ (the kernel-family choice).
function build_kernelchoice_collapse()
    xs = [1.0, 2.0, 3.0]
    K  = [[0.8 0.2 0.0; 0.1 0.8 0.1; 0.0 0.2 0.8],
          [0.5 0.5 0.0; 0.3 0.4 0.3; 0.0 0.5 0.5]]
    c  = [0.0, 0.3]
    block = GriddedLayout(:x => Discrete(xs), :θ => Discrete([1]))
    stage = KernelChoiceStage(block; choice_axis = :θ, x_axis = :x, kernels = K, cost = c)
    W = reshape([4.0, 1.0, 2.0], 3, 1)
    Λ = reshape([0.2, 0.5, 0.3], 3, 1)
    (stage, W, Λ, NamedTuple())
end

# PortfolioStage: collapse on :share (sugar over KernelChoice).
function build_portfolio_collapse()
    xs = [1.0, 2.0, 3.0]
    K  = [[0.8 0.2 0.0; 0.1 0.8 0.1; 0.0 0.2 0.8],
          [0.5 0.5 0.0; 0.3 0.4 0.3; 0.0 0.5 0.5]]
    c  = [0.0, 0.3]
    pblock = GriddedLayout(:wealth => Discrete(xs), :share => Discrete([1]))
    stage  = PortfolioStage(pblock; kernels = K, cost = c)
    W = reshape([4.0, 1.0, 2.0], 3, 1)
    Λ = reshape([0.2, 0.5, 0.3], 3, 1)
    (stage, W, Λ, NamedTuple())
end

const COLLAPSE_CASES = [
    ("EndogenousExit",    build_exit_collapse),
    ("KernelChoiceStage", build_kernelchoice_collapse),
    ("PortfolioStage",    build_portfolio_collapse),
]

# --- ProductStage (⊕ direct sum) builders ---------------------------------
# The `⊕` lift `to_device(::ProductStage, …)` recurses over each bundled factor and moves the
# product's OWN fused V/Λ tensors — the only combinator buffer that owns arrays (a ChainStage owns
# none). These exercise that recursion with GPU-able factors: MarkovStage factors (the modern dense
# path) and ConsumptionSavings factors (each a ChainStage → the hand-written CS CUDA kernels). Kept
# small (shared card). The product axis position is toggled so the fused-slice writes hit pdim ≠ 1.

# `n` uniform MarkovStage factors joined along :group (axis position toggled by `group_first`).
function build_prod_markov(; n::Int, group_first::Bool)
    P = [0.7 0.3; 0.3 0.7]
    layout = group_first ?
        GriddedLayout(:group => Discrete([1]), :z => Discrete([0.5, 1.5])) :
        GriddedLayout(:z => Discrete([0.5, 1.5]), :group => Discrete([1]))
    s  = MarkovStage(layout; axis = :z, transition_matrix = P)
    ps = product(ntuple(_ -> s, n)...; axis = :group)
    osz = layout_size(output_layout(ps)); isz = layout_size(input_layout(ps))
    V_end = reshape(collect(range(-1.0, 1.0; length = prod(osz))), osz)
    Λ = rand(isz...); Λ ./= sum(Λ)
    (ps, V_end, Λ, NamedTuple())
end

# Two ConsumptionSavings factors joined along :group — the recursion reaches the CS CUDA kernels.
function build_prod_cs(; n_w::Int = 9)
    layout = GriddedLayout(
        :wealth => GriddedContinuous([exp(t) - 1.0 for t in range(0.0, log(21.0); length = n_w)]),
        :income => Discrete([0.6, 1.0, 1.4]),
        :group  => Discrete([1]))
    cs = ConsumptionSavingsStage(layout; β = 0.96, utility = (cell, c; env) -> log(c), axis = :wealth)
    ps = product(cs, cs; axis = :group)
    isz = layout_size(input_layout(ps))
    V_end = [0.1 * w + 0.05 * y for w in 1:n_w, y in 1:3, g in 1:2]
    Λ = rand(isz...); Λ ./= sum(Λ)
    (ps, V_end, Λ, NamedTuple())
end

const PRODUCT_CASES = [
    ("⊕ MarkovStage ×3 (group last)",  () -> build_prod_markov(n = 3, group_first = false)),
    ("⊕ MarkovStage ×2 (group first)", () -> build_prod_markov(n = 2, group_first = true)),
    ("⊕ ConsumptionSavings ×2",        () -> build_prod_cs(n_w = 9)),
]

# --- MixingStage / RetentionStage builders --------------------------------
# The closed-form rung-(a) blend `V = b + c*(a−b)` is TWO MarkovStage backward applies (dense `mul!`,
# the modern device path) plus a pointwise Fenchel conjugate `c*` and policy `θ*`, both broadcast.
# `to_device(::MixingStage,…)` recurses into the bundled `markA`/`markB` and rides the policy/V/Λ
# scratch along; the spec (K_A/K_B matrices, closures) stays host-side. RetentionStage is the
# `K_A = I` special case. The mixing axis position is toggled (leading vs non-leading) to drive the
# Markov sub-applies' permute branch. Value AND seated policy θ* are checked (tight: the dense `mul!`
# may reassociate ~1e-13, and θ* is a continuous clamp of the matmul output).

# A small row-stochastic transition on a size-`n` axis (rows sum to 1; deterministic-free so the
# blend is non-degenerate).
function _row_stochastic(n, seed)
    Random.seed!(seed)
    M = rand(n, n) .+ 0.1
    return M ./ sum(M; dims = 2)
end

# MixingStage: blend K_A (θ=1) and K_B (θ=0) on `:x`, mixing axis leading or non-leading.
function build_mixing(; axis_first::Bool, n = 4)
    K_A = _row_stochastic(n, 20260629)
    K_B = _row_stochastic(n, 20260630)
    layout = axis_first ?
        GriddedLayout(:x => Discrete(collect(1:n)), :z => Discrete([0.5, 1.0, 1.5])) :
        GriddedLayout(:z => Discrete([0.5, 1.0, 1.5]), :x => Discrete(collect(1:n)))
    stage = MixingStage(layout; axis = :x, K_A = K_A, K_B = K_B, cost_curvature = 0.8)
    V_end = axis_first ? [0.3 * x + 0.1 * z for x in 1:n, z in 1:3] :
                         [0.1 * z + 0.3 * x for z in 1:3, x in 1:n]
    Λ = axis_first ? rand(n, 3) : rand(3, n); Λ ./= sum(Λ)
    (stage, V_end, Λ, NamedTuple())
end

# RetentionStage: K_A = I (pay to stay) vs exit_kernel (K_B), axis leading or non-leading.
function build_retention(; axis_first::Bool, n = 4)
    exit_kernel = _row_stochastic(n, 20260701)
    layout = axis_first ?
        GriddedLayout(:x => Discrete(collect(1:n)), :z => Discrete([0.5, 1.0, 1.5])) :
        GriddedLayout(:z => Discrete([0.5, 1.0, 1.5]), :x => Discrete(collect(1:n)))
    stage = RetentionStage(layout; axis = :x, exit_kernel = exit_kernel, cost_curvature = 1.2)
    V_end = axis_first ? [0.3 * x + 0.1 * z for x in 1:n, z in 1:3] :
                         [0.1 * z + 0.3 * x for z in 1:3, x in 1:n]
    Λ = axis_first ? rand(n, 3) : rand(3, n); Λ ./= sum(Λ)
    (stage, V_end, Λ, NamedTuple())
end

const MIXING_CASES = [
    ("Mixing    (axis leading)",     () -> build_mixing(axis_first = true)),
    ("Mixing    (axis non-leading)", () -> build_mixing(axis_first = false)),
    ("Retention (axis leading)",     () -> build_retention(axis_first = true)),
    ("Retention (axis non-leading)", () -> build_retention(axis_first = false)),
]

# Run one mixing/retention stage CPU vs GPU. Returns
# (on_dev, bw_ok, bw_Δ, pol_ok, pol_Δ, fw_ok, fw_Δ): value, seated θ*, and forward all tight.
function mixing_cpu_gpu_compare(stage, V_end, Λ_start, env)
    V_cpu   = copy(backward!(stage, V_end, env))
    pol_cpu = copy(policy(stage))
    Λ_cpu   = copy(forward!(stage, Λ_start))

    gstage  = gpu_stage(stage)
    V_gpu   = Array(backward!(gstage, to_dev(V_end), env))
    on_dev  = gstage.buffer.policy isa CuArray && gstage.buffer.V_start isa CuArray
    pol_gpu = Array(policy(gstage))
    Λ_gpu   = Array(forward!(gstage, to_dev(Λ_start)))

    bw_ok,  bw_d  = match_tight(V_gpu, V_cpu)
    pol_ok, pol_d = match_tight(pol_gpu, pol_cpu)
    fw_ok,  fw_d  = match_tight(Λ_gpu, Λ_cpu)
    return (on_dev, bw_ok, bw_d, pol_ok, pol_d, fw_ok, fw_d)
end

# --- DiscreteMoveStage builders (integer-destination gather) --------------
# The nearest-INDEX deterministic move: backward is the adjoint of the integer scatter — a gather
# `V_start[cell] = V_end[ν(cell)]` where `ν` is the DETERMINISTIC snapped destination (NOT an argmax
# over the axis — that gated case is the `:brute` kernel). The gather is the device port of the CPU
# `_gather_along!`; a pure relocation copy, so value AND policy must be bit-identical (Δ == 0). The
# forward (integer scatter) reuses `_cs_forward_scatter!`. We exercise the operative axis leading
# (WD == 1, no-permute branch) and non-leading (WD == 3, `_to_front`/`_from_front!` branch).

# Off-grid destination `0.5·wealth + income`, snapped to the nearest wealth INDEX.
function build_discrete_move(; wfirst::Bool)
    grid = collect(range(0.0, 4.0; length = 5))
    inc  = Discrete([0.7, 1.2])
    z    = Discrete([0.0, 1.0, 2.0])
    layout = wfirst ?
        GriddedLayout(:wealth => GriddedContinuous(grid), :income => inc, :z => z) :
        GriddedLayout(:income => inc, :z => z, :wealth => GriddedContinuous(grid))
    stage = DiscreteMoveStage(layout; destination = (; wealth, income, z, env) -> 0.5 * wealth + income,
                              axis = :wealth)
    sz = axissize.(layout.axes)
    V_end = reshape(collect(range(-1.0, 1.0; length = prod(sz))), sz)
    Λ = rand(sz...); Λ ./= sum(Λ)
    (stage, V_end, Λ, NamedTuple())
end

const DISCRETE_MOVE_CASES = [
    ("DiscreteMove (wealth leading)",     () -> build_discrete_move(wfirst = true)),
    ("DiscreteMove (wealth non-leading)", () -> build_discrete_move(wfirst = false)),
]

# Run one DiscreteMove stage CPU vs GPU. Returns (vΔ, policy-mismatch count, fwΔ). The gather is a
# pure copy and the integer scatter writes each destination once per source (deterministic landing),
# so value, policy, AND forward are all bit-identical (Δ == 0) — no collision-order drift.
function discrete_move_cpu_gpu_compare(stage, V_end, Λ_start, env)
    V_cpu   = copy(backward!(stage, V_end, env))
    pol_cpu = copy(policy(stage))
    Λ_cpu   = copy(forward!(stage, Λ_start))

    gstage  = gpu_stage(stage)
    V_gpu   = Array(backward!(gstage, to_dev(V_end), env))
    pol_gpu = Array(policy(gstage))
    Λ_gpu   = Array(forward!(gstage, to_dev(Λ_start)))

    vΔ     = maximum(abs.(V_gpu .- V_cpu))
    polmis = count(pol_gpu .!= pol_cpu)
    fwΔ    = maximum(abs.(Λ_gpu .- Λ_cpu))
    return (vΔ, polmis, fwΔ)
end

# --- ContinuousArgmaxStage builders (off-grid 1-D maximiser) --------------
# The off-grid sibling of the discrete `:brute` argmax: per origin cell a CONTINUOUS max of
# `reward(x, a') + interp(V_end, a')` via a coarse grid scan (to bracket) then a 45-iteration
# golden-section refine, writing a FLOAT policy position into the InterpKernel. The device kernel
# evaluates the user payoff CLOSURE on-device and reuses the SAME `_interp1d`/`_golden_max` as the CPU
# (only `env_dep` is lifted to a compile-time `Val` to elide the dead payoff branch). It is therefore
# NOT bit-identical: the value matches to sub-ulp (observed ≤1.7e-16, one case exactly 0), but the float
# policy position drifts ~1e-8. That drift is the golden-section's own resolution floor near the FLAT
# optimum — at convergence `fc ≈ fd`, so float-reassociation noise (GPU FMA contraction) flips a late
# golden step, landing on a position ~the-then-bracket-width away. Same bracketing grid interval, same
# answer, only sub-position drift — the inherent float-nondeterminism of a derivative-free optimiser,
# not a wrong optimum. Tolerances below are set to those observed magnitudes, explicitly NOT 0.

const CA_VAL_ATOL = 1e-12     # value: observed ≤1.7e-16 (sub-ulp); 1e-12 a tight, justified ceiling
const CA_POL_ATOL = 1e-6      # float policy position: observed ≤1.1e-8 (golden-section resolution floor)
const CA_FWD_ATOL = 1e-7      # Young-split forward through the drifted position: observed ≤1.3e-9

# 1-D, no env, operative axis leading.
function build_ca_1d_leading()
    N      = 21
    grid   = collect(range(0.0, 10.0; length = N))
    layout = GriddedLayout(:a => GriddedContinuous(grid))
    reward = (a, a_next) -> -0.5 * (a_next - (0.5 * a + 1.0))^2
    stage  = ContinuousArgmaxStage(layout; reward = reward, axis = :a)
    V_end  = [-0.05 * (g - 5.0)^2 for g in grid]
    Random.seed!(20260629); Λ = rand(N); Λ ./= sum(Λ)
    (stage, V_end, Λ, NamedTuple())
end

# 2-D, reward dep axis :z threaded as a kwarg, operative axis leading.
function build_ca_dep_leading()
    Na     = 16
    grid   = collect(range(0.0, 8.0; length = Na))
    zlev   = [1.0, 2.0]
    layout = GriddedLayout(:a => GriddedContinuous(grid), :z => Discrete(zlev))
    reward = (a, a_next; z) -> -0.5 * (a_next - (0.4 * a + z))^2
    stage  = ContinuousArgmaxStage(layout; reward = reward, axis = :a)
    V_end  = [-0.05 * (g - 4.0)^2 + 0.1 * zz for g in grid, zz in zlev]
    Random.seed!(20260630); Λ = rand(Na, 2); Λ ./= sum(Λ)
    (stage, V_end, Λ, NamedTuple())
end

# env-dependent reward (the payoff takes an `env` kwarg), operative axis leading.
function build_ca_env_leading()
    N      = 41
    grid   = collect(range(0.0, 10.0; length = N))
    layout = GriddedLayout(:a => GriddedContinuous(grid))
    reward = (a, a_next; env) -> -0.5 * 2.0 * (a_next - env.t)^2
    stage  = ContinuousArgmaxStage(layout; reward = reward, axis = :a)
    V_end  = 0.5 .* grid
    Random.seed!(20260631); Λ = rand(N); Λ ./= sum(Λ)
    (stage, V_end, Λ, (t = 5.0,))
end

# 2-D dep axis :z, operative axis NON-leading (z first → exercises permute-to-front / dep-column map).
function build_ca_dep_nonleading()
    Na     = 16
    grid   = collect(range(0.0, 8.0; length = Na))
    zlev   = [1.0, 2.0, 3.0]
    layout = GriddedLayout(:z => Discrete(zlev), :a => GriddedContinuous(grid))
    reward = (a, a_next; z) -> -0.5 * (a_next - (0.4 * a + z))^2
    stage  = ContinuousArgmaxStage(layout; reward = reward, axis = :a)
    V_end  = [-0.05 * (g - 4.0)^2 + 0.1 * zz for zz in zlev, g in grid]
    Random.seed!(20260632); Λ = rand(3, Na); Λ ./= sum(Λ)
    (stage, V_end, Λ, NamedTuple())
end

# env-dependent, operative axis NON-leading (y first).
function build_ca_env_nonleading()
    N      = 21
    grid   = collect(range(0.0, 10.0; length = N))
    layout = GriddedLayout(:y => Discrete([0.5, 1.5]), :a => GriddedContinuous(grid))
    reward = (a, a_next; env) -> -0.5 * (a_next - (0.5 * a + env.t))^2
    stage  = ContinuousArgmaxStage(layout; reward = reward, axis = :a)
    V_end  = [-0.05 * (g - 5.0)^2 + 0.1 * yy for yy in [0.5, 1.5], g in grid]
    Random.seed!(20260633); Λ = rand(2, N); Λ ./= sum(Λ)
    (stage, V_end, Λ, (t = 1.0,))
end

const CA_CASES = [
    ("ContinuousArgmax (no-env, leading)",  build_ca_1d_leading),
    ("ContinuousArgmax (dep-z, leading)",   build_ca_dep_leading),
    ("ContinuousArgmax (env, leading)",     build_ca_env_leading),
    ("ContinuousArgmax (dep-z, non-lead)",  build_ca_dep_nonleading),
    ("ContinuousArgmax (env, non-lead)",    build_ca_env_nonleading),
]

# Run one ContinuousArgmax stage CPU vs GPU. Returns (vΔ, pΔ, fΔ, all-finite). Value is sub-ulp, the
# float policy position drifts ~1e-8 (the optimiser's resolution floor), forward ~1e-9 — all matched
# against the explicit, justified tolerances above, NOT bit-identical.
function ca_cpu_gpu_compare(stage, V_end, Λ_start, env)
    V_cpu   = copy(backward!(stage, V_end, env))
    pol_cpu = copy(policy(stage))
    Λ_cpu   = copy(forward!(stage, Λ_start))

    gstage  = gpu_stage(stage)
    V_gpu   = Array(backward!(gstage, to_dev(V_end), env))
    pol_gpu = Array(policy(gstage))
    Λ_gpu   = Array(forward!(gstage, to_dev(Λ_start)))

    vΔ     = maximum(abs.(V_gpu .- V_cpu))
    pΔ     = maximum(abs.(pol_gpu .- pol_cpu))
    fΔ     = maximum(abs.(Λ_gpu .- Λ_cpu))
    finite = all(isfinite, V_gpu) && (eltype(pol_gpu) === Float64)
    return (vΔ, pΔ, fΔ, finite)
end

# --- SearchMatchingStage builders (internal-effort discrete argmax) -------
# The unemployed value is a discrete max over an INTERNAL effort grid (never a state axis):
# `Vu_new = maxₖ −cost[k] + pe[k]·Ve + (1−pe[k])·Vu`, with the maximiser index → the effort policy
# and `pe[k*]` cached for the forward replay; the employed value is the separation mix
# `(1−δ)·Ve + δ·Vu`. The device FUSES the per-effort scan into one thread per non-labor cell (no `Q`
# tensor), mirroring the CPU `findmax` argmax (same left-associated objective, same first-index
# tie-break). It is a discrete max over a FIXED grid, so the discrete CHOICE is bit-identical — the
# effort policy index `polmis == 0` and the cached `p = pe[k*]` Δ == 0 (exact gather through the
# identical policy). The VALUE is NOT bit-identical, unlike `:brute` (whose inner op is a pure add):
# the per-effort objective and the separation mix are multiply-adds, which the device contracts to
# FMA — a one-ulp drift (observed ≤2.2e-16). So value is value-exact-to-sub-ulp, policy + p exact.
# Both labor-axis positions (leading / non-leading) drive the permute-to-front branch.

const SAM_VAL_ATOL = 1e-12     # value: observed ≤2.2e-16 (one ulp); FMA contraction of the hazard/Q multiply-adds
const SAM_FWD_ATOL = 1e-12     # forward replay through the (bit-identical) policy/p: observed ≤1.4e-17

# Two-level labor axis × wealth; θ drives job-finding via a host closure (run host-side, never on
# device — only the pre-evaluated numeric effort vectors reach the kernel). `emp_first` toggles the
# labor-axis position (leading → no-op permute; non-leading → `_to_front`/`_from_front!`). The
# continuation is separated (employed > unemployed) so the effort argmax has a well-defined optimum.
function build_sam(; emp_first::Bool, n_w = 6)
    efforts = collect(range(0.0, 2.0; length = 6))
    cost    = e -> 0.5 * e^2
    jf      = (e, θ) -> 1 - exp(-e * θ)
    grid    = collect(range(0.0, 3.0; length = n_w))
    layout  = emp_first ?
        GriddedLayout(:emp => Discrete([:u, :e]), :wealth => GriddedContinuous(grid)) :
        GriddedLayout(:wealth => GriddedContinuous(grid), :emp => Discrete([:u, :e]))
    stage = SearchMatchingStage(layout; axis = :emp, efforts = efforts,
                                cost = cost, job_finding = jf,
                                separation = 0.10, tightness = FromEnv(:θ))
    V_end = emp_first ?
        [l == 1 ? 0.2 * w : 1.0 + 0.2 * w for l in 1:2, w in 1:n_w] :
        [l == 1 ? 0.2 * w : 1.0 + 0.2 * w for w in 1:n_w, l in 1:2]
    Random.seed!(20260629)
    Λ = emp_first ? rand(2, n_w) : rand(n_w, 2); Λ ./= sum(Λ)
    (stage, V_end, Λ, (θ = 1.0,))
end

const SAM_CASES = [
    ("SearchMatching (emp leading)",     () -> build_sam(emp_first = true)),
    ("SearchMatching (emp non-leading)", () -> build_sam(emp_first = false)),
]

# Run one SearchMatchingStage CPU vs GPU. Returns (vΔ, polmis, pΔ, fΔ, on_dev). The discrete effort
# policy and the cached `p` are bit-identical (polmis == 0, pΔ == 0); the value is sub-ulp (FMA);
# the forward replay is sub-ulp.
function sam_cpu_gpu_compare(stage, V_end, Λ_start, env)
    V_cpu   = copy(backward!(stage, V_end, env))
    pol_cpu = copy(stage.kernel.policy)
    p_cpu   = copy(stage.kernel.p)
    Λ_cpu   = copy(forward!(stage, Λ_start))

    gstage  = gpu_stage(stage)
    V_gpu   = Array(backward!(gstage, to_dev(V_end), env))
    on_dev  = gstage.kernel.policy isa CuArray && gstage.kernel.p isa CuArray
    pol_gpu = Array(gstage.kernel.policy)
    p_gpu   = Array(gstage.kernel.p)
    Λ_gpu   = Array(forward!(gstage, to_dev(Λ_start)))

    vΔ     = maximum(abs.(V_gpu .- V_cpu))
    polmis = count(pol_gpu .!= pol_cpu)
    pΔ     = maximum(abs.(p_gpu .- p_cpu))
    fΔ     = maximum(abs.(Λ_gpu .- Λ_cpu))
    return (vΔ, polmis, pΔ, fΔ, on_dev)
end

# --- streaming kernel-choice builders (MeanVariance / ScaleVariance) ------
# The streaming kernel-choice family: the household picks a per-cell scalar θ from a fixed grid along
# the choice axis. backward (the θ choice) FUSES the per-θ clamped-interp gather + per-θ cost + running
# argmax into one thread per non-axis column (mirroring the SearchMatching fuse); forward (the replay)
# is the per-cell Young-split scatter (one thread owns each column ⇒ no atomics). Because the θ grid is
# FIXED, the seated θ* policy is bit-identical to the CPU (same gather, same `acc − cost` objective,
# same strict-`>` first-θ tie-break); the VALUE is value-exact-to-sub-ulp — the gather's
# `weights·((1−w)V + wV)` multiply-adds and the forward's `m·(1−w)` split contract to FMA on device, a
# one-ulp drift (like SearchMatching, NOT bit-identical). MeanVariance (multiplicative portfolio
# return) and ScaleVariance (additive mean-preserving spread) share the SAME device kernels, split only
# by a compile-time landing-family tag. Both axis positions (leading / non-leading) drive the permute.

const KC_VAL_ATOL = 1e-12     # value: FMA contraction of the gather/scatter multiply-adds (observed sub-ulp)
const KC_FWD_ATOL = 1e-12     # forward replay: FMA on the Young-split `m·(1−w)` (observed sub-ulp)

# MeanVariance (portfolio share θ on wealth): concave √-wealth V ⇒ interior share. `wfirst` toggles
# the choice-axis position. Grid chosen so the up-returns stay interior (exercise off-clamp interp).
function build_meanvar(; wfirst::Bool)
    ws     = collect(0.5:0.25:8.0); nw = length(ws)
    shares = [0.0, 0.25, 0.5, 0.75, 1.0]
    Rf     = 1.02
    Rrisky = [0.7, 1.4]
    probs  = [0.5, 0.5]
    layout = wfirst ?
        GriddedLayout(:wealth => GriddedContinuous(ws), :z => Discrete([0.5, 1.5])) :
        GriddedLayout(:z => Discrete([0.5, 1.5]), :wealth => GriddedContinuous(ws))
    stage  = MeanVarianceStage(layout; axis = :wealth, shares = shares,
                               risk_free = Rf, risky_returns = Rrisky, probs = probs,
                               cost = (θ; env) -> 0.01 * θ^2)
    V_end  = wfirst ? [sqrt(w) + 0.1 * z for w in ws, z in [0.5, 1.5]] :
                      [0.1 * z + sqrt(w) for z in [0.5, 1.5], w in ws]
    Random.seed!(20260629)
    Λ = wfirst ? rand(nw, 2) : rand(2, nw); Λ ./= sum(Λ)
    (stage, V_end, Λ, NamedTuple())
end

# ScaleVariance (mean-preserving-spread dispersion θ): convex-flanked V ⇒ interior θ. `xfirst` toggles
# the choice-axis position; a per-θ vector cost exercises the `_cost_at(::AbstractVector)` path.
function build_scalevar(; xfirst::Bool)
    xs   = collect(0.0:0.5:10.0); nx = length(xs)
    shk  = [-1.0, 1.0]; wts = [0.5, 0.5]
    disp = [0.0, 0.5, 1.0, 1.5, 2.0]
    cost = [0.0, 0.05, 0.2, 0.45, 0.8]
    layout = xfirst ?
        GriddedLayout(:x => GriddedContinuous(xs), :z => Discrete([0.5, 1.5])) :
        GriddedLayout(:z => Discrete([0.5, 1.5]), :x => GriddedContinuous(xs))
    stage  = ScaleVarianceStage(layout; axis = :x, dispersions = disp,
                                shocks = shk, weights = wts, cost = cost)
    Vb     = @. 2.0 * exp(-(xs - 5.0)^2 / 4.0)
    V_end  = xfirst ? [Vb[i] + 0.1 * z for i in 1:nx, z in [0.5, 1.5]] :
                      [0.1 * z + Vb[i] for z in [0.5, 1.5], i in 1:nx]
    Random.seed!(20260630)
    Λ = xfirst ? rand(nx, 2) : rand(2, nx); Λ ./= sum(Λ)
    (stage, V_end, Λ, NamedTuple())
end

const KERNEL_CHOICE_CASES = [
    ("MeanVariance  (wealth leading)",      () -> build_meanvar(wfirst = true)),
    ("MeanVariance  (wealth non-leading)",  () -> build_meanvar(wfirst = false)),
    ("ScaleVariance (x leading)",           () -> build_scalevar(xfirst = true)),
    ("ScaleVariance (x non-leading)",       () -> build_scalevar(xfirst = false)),
]

# Run one streaming kernel-choice stage CPU vs GPU. Returns (vΔ, polmis, fwΔ, on_dev): value sub-ulp,
# the seated θ* policy bit-identical (fixed grid, first-θ tie-break), forward replay sub-ulp.
function kernel_choice_cpu_gpu_compare(stage, V_end, Λ_start, env)
    V_cpu   = copy(backward!(stage, V_end, env))
    pol_cpu = copy(policy(stage))
    Λ_cpu   = copy(forward!(stage, Λ_start))

    gstage  = gpu_stage(stage)
    V_gpu   = Array(backward!(gstage, to_dev(V_end), env))
    on_dev  = gstage.kernel.θstar isa CuArray
    pol_gpu = Array(policy(gstage))
    Λ_gpu   = Array(forward!(gstage, to_dev(Λ_start)))

    vΔ     = maximum(abs.(V_gpu .- V_cpu))
    polmis = count(pol_gpu .!= pol_cpu)
    fwΔ    = maximum(abs.(Λ_gpu .- Λ_cpu))
    return (vΔ, polmis, fwΔ, on_dev)
end

# --- run ------------------------------------------------------------------

const CASES = [
    ("WealthChange (wealth leading)",     build_wc_leading),
    ("WealthChange (wealth non-leading)", build_wc_nonleading),
    ("ConsumptionSavings (seq, n_w=16)",  () -> build_cs_seq(16)),
    ("ConsumptionSavings (pow2, n_w=17)", () -> build_cs_pow2(17)),
    ("ConsumptionSavings (non-leading)",  () -> build_cs_nonleading(17)),
]

function run_kernel_tests()
    @assert CUDA.functional() "CUDA not functional"
    println("CUDA functional on: ", CUDA.name(CUDA.device()))
    all_ok = true
    @testset "GPU kernels (CS + WealthChange)" begin
        for (name, build) in CASES
            stage, V_end, Λ_start, env = build()
            bw_ok, bw_d, fw_ok, fw_d = cpu_gpu_compare(stage, V_end, Λ_start, env)
            @printf("%-38s backward Δ=%.2e %s   forward Δ=%.2e %s\n",
                    name, bw_d, bw_ok ? "OK" : "FAIL", fw_d, fw_ok ? "OK" : "FAIL")
            all_ok &= bw_ok & fw_ok
            @testset "$name" begin
                @test bw_ok
                @test fw_ok
            end
        end
    end
    @testset "GPU :brute argmax (buy/sell-home, tie-heavy)" begin
        for (name, build) in BRUTE_CASES
            stage, V_end, Λ_start, env = build()
            vΔ, polmis, fw_ok, fw_d = brute_cpu_gpu_compare(stage, V_end, Λ_start, env)
            @printf("%-34s value Δ=%.2e %s   policy mism=%d %s   forward Δ=%.2e %s\n",
                    name, vΔ, vΔ == 0 ? "OK" : "FAIL",
                    polmis, polmis == 0 ? "OK" : "FAIL", fw_d, fw_ok ? "OK" : "FAIL")
            all_ok &= (vΔ == 0) & (polmis == 0) & fw_ok
            @testset "$name" begin
                @test vΔ == 0           # value bit-identical
                @test polmis == 0       # policy index bit-identical, including ties
                @test fw_ok             # forward scatter matches (tight)
            end
        end
    end
    @testset "GPU DiscreteMove gather (integer-destination, deterministic)" begin
        for (name, build) in DISCRETE_MOVE_CASES
            stage, V_end, Λ_start, env = build()
            vΔ, polmis, fwΔ = discrete_move_cpu_gpu_compare(stage, V_end, Λ_start, env)
            @printf("%-34s value Δ=%.2e %s   policy mism=%d %s   forward Δ=%.2e %s\n",
                    name, vΔ, vΔ == 0 ? "OK" : "FAIL",
                    polmis, polmis == 0 ? "OK" : "FAIL", fwΔ, fwΔ == 0 ? "OK" : "FAIL")
            all_ok &= (vΔ == 0) & (polmis == 0) & (fwΔ == 0)
            @testset "$name" begin
                @test vΔ == 0           # backward gather bit-identical (pure copy)
                @test polmis == 0       # snapped destination index bit-identical
                @test fwΔ == 0          # forward integer scatter bit-identical
            end
        end
    end
    @testset "GPU ContinuousArgmaxStage (off-grid maximiser)" begin
        for (name, build) in CA_CASES
            stage, V_end, Λ_start, env = build()
            vΔ, pΔ, fΔ, finite = ca_cpu_gpu_compare(stage, V_end, Λ_start, env)
            @printf("%-34s value Δ=%.2e %s   policy Δ=%.2e %s   forward Δ=%.2e %s\n",
                    name, vΔ, vΔ <= CA_VAL_ATOL ? "OK" : "FAIL",
                    pΔ, pΔ <= CA_POL_ATOL ? "OK" : "FAIL", fΔ, fΔ <= CA_FWD_ATOL ? "OK" : "FAIL")
            all_ok &= (vΔ <= CA_VAL_ATOL) & (pΔ <= CA_POL_ATOL) & (fΔ <= CA_FWD_ATOL) & finite
            @testset "$name" begin
                @test finite                # value finite everywhere; policy a Float64 position
                @test vΔ <= CA_VAL_ATOL     # value sub-ulp (same optimum)
                @test pΔ <= CA_POL_ATOL     # float policy position at the golden-section floor
                @test fΔ <= CA_FWD_ATOL     # Young-split forward through the seated position
            end
        end
    end
    @testset "GPU choice-collapse forward (grown scatter axis)" begin
        for (name, build) in COLLAPSE_CASES
            stage, V_end, Λ_start, env = build()
            # `cpu_gpu_compare` seats each side's policy (backward!) before forward!. The
            # forward scatter is the ext fix's target: the source choice axis is size 1, so
            # each destination cell is written exactly once ⇒ bit-identical (Δ == 0), no
            # collision-order drift. Backward is the brute kernel (already FULL; tight match).
            bw_ok, bw_d, fw_ok, fw_d = cpu_gpu_compare(stage, V_end, Λ_start, env)
            @printf("%-34s backward Δ=%.2e %s   forward Δ=%.2e %s\n",
                    name, bw_d, bw_ok ? "OK" : "FAIL", fw_d, fw_d == 0 ? "OK" : "FAIL")
            all_ok &= bw_ok & (fw_d == 0)
            @testset "$name" begin
                @test fw_d == 0         # forward scatter bit-identical (the fix's target)
                @test bw_ok             # backward matches (tight; brute kernel already FULL)
            end
        end
    end
    @testset "GPU ProductStage (⊕ direct sum)" begin
        for (name, build) in PRODUCT_CASES
            stage, V_end, Λ_start, env = build()
            # The fix moves the product's fused tensors to the device too — assert they actually
            # landed on-device (a silent host round-trip would still pass the value match but defeat
            # the lift). Then compare CPU vs GPU backward/forward via the shared tight match.
            gstage = gpu_stage(stage)
            on_dev = gstage.buffer.V_fused isa CuArray && gstage.buffer.Λ_fused isa CuArray
            bw_ok, bw_d, fw_ok, fw_d = cpu_gpu_compare(stage, V_end, Λ_start, env)
            @printf("%-34s fused-on-dev %s   backward Δ=%.2e %s   forward Δ=%.2e %s\n",
                    name, on_dev ? "OK" : "FAIL", bw_d, bw_ok ? "OK" : "FAIL", fw_d, fw_ok ? "OK" : "FAIL")
            all_ok &= on_dev & bw_ok & fw_ok
            @testset "$name" begin
                @test on_dev            # fused V/Λ live on the device (the lift's core step)
                @test bw_ok             # per-factor backward matches the CPU reference (tight)
                @test fw_ok             # per-factor forward matches the CPU reference (tight)
            end
        end
    end
    @testset "GPU MixingStage / RetentionStage (closed-form blend)" begin
        for (name, build) in MIXING_CASES
            stage, V_end, Λ_start, env = build()
            # The lift moves the bundled markA/markB sub-stages and the policy/V/Λ scratch to the
            # device — assert the scratch actually landed (a silent host round-trip would still
            # pass the value match but defeat the lift). Value, seated θ*, and forward all tight
            # (the two Markov applies are dense `mul!`, which may reassociate ~1e-13 on device).
            on_dev, bw_ok, bw_d, pol_ok, pol_d, fw_ok, fw_d =
                mixing_cpu_gpu_compare(stage, V_end, Λ_start, env)
            @printf("%-30s on-dev %s   backward Δ=%.2e %s   policy Δ=%.2e %s   forward Δ=%.2e %s\n",
                    name, on_dev ? "OK" : "FAIL", bw_d, bw_ok ? "OK" : "FAIL",
                    pol_d, pol_ok ? "OK" : "FAIL", fw_d, fw_ok ? "OK" : "FAIL")
            all_ok &= on_dev & bw_ok & pol_ok & fw_ok
            @testset "$name" begin
                @test on_dev            # policy/V scratch live on the device (the lift's core step)
                @test bw_ok             # backward value matches the CPU reference (tight)
                @test pol_ok            # seated mixing policy θ* matches (tight)
                @test fw_ok             # forward (blended push through markA/markB) matches (tight)
            end
        end
    end
    @testset "GPU SearchMatchingStage (internal-effort discrete argmax)" begin
        for (name, build) in SAM_CASES
            stage, V_end, Λ_start, env = build()
            vΔ, polmis, pΔ, fΔ, on_dev = sam_cpu_gpu_compare(stage, V_end, Λ_start, env)
            @printf("%-34s value Δ=%.2e %s   policy mism=%d %s   pΔ=%.2e %s   forward Δ=%.2e %s\n",
                    name, vΔ, vΔ <= SAM_VAL_ATOL ? "OK" : "FAIL",
                    polmis, polmis == 0 ? "OK" : "FAIL",
                    pΔ, pΔ == 0 ? "OK" : "FAIL", fΔ, fΔ <= SAM_FWD_ATOL ? "OK" : "FAIL")
            all_ok &= (vΔ <= SAM_VAL_ATOL) & (polmis == 0) & (pΔ == 0) & (fΔ <= SAM_FWD_ATOL) & on_dev
            @testset "$name" begin
                @test on_dev               # policy/p live on the device (the fused kernel's outputs)
                @test polmis == 0          # effort policy index bit-identical (the discrete choice)
                @test pΔ == 0              # cached p = pe[k*] exact (gather through the identical policy)
                @test vΔ <= SAM_VAL_ATOL    # value sub-ulp (FMA contraction of the hazard/Q multiply-adds)
                @test fΔ <= SAM_FWD_ATOL    # forward replay sub-ulp
            end
        end
    end
    @testset "GPU streaming kernel-choice (MeanVariance / ScaleVariance)" begin
        for (name, build) in KERNEL_CHOICE_CASES
            stage, V_end, Λ_start, env = build()
            vΔ, polmis, fwΔ, on_dev = kernel_choice_cpu_gpu_compare(stage, V_end, Λ_start, env)
            @printf("%-34s value Δ=%.2e %s   policy mism=%d %s   forward Δ=%.2e %s\n",
                    name, vΔ, vΔ <= KC_VAL_ATOL ? "OK" : "FAIL",
                    polmis, polmis == 0 ? "OK" : "FAIL", fwΔ, fwΔ <= KC_FWD_ATOL ? "OK" : "FAIL")
            all_ok &= (vΔ <= KC_VAL_ATOL) & (polmis == 0) & (fwΔ <= KC_FWD_ATOL) & on_dev
            @testset "$name" begin
                @test on_dev               # the per-cell θ* policy lives on the device
                @test polmis == 0          # seated θ* bit-identical (fixed grid, first-θ tie-break)
                @test vΔ <= KC_VAL_ATOL    # value sub-ulp (FMA on the gather multiply-adds)
                @test fwΔ <= KC_FWD_ATOL   # forward replay sub-ulp (FMA on the Young-split)
            end
        end
    end
    return all_ok
end

if abspath(PROGRAM_FILE) == @__FILE__
    ok = run_kernel_tests()
    exit(ok ? 0 : 1)
end
