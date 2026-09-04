# GPU kernel verification for the stages with hand-written CUDA kernels in
# HouseholdStagesCUDAExt: ConsumptionSavings/ContinuousArgmax (the off-grid savings
# choice — the shared divide-and-conquer walk + parabolic vertex on the device-resident reward
# face, the same inner source as the CPU column loop), WealthChange, the discrete
# brute argmax (buy/sell-home), DiscreteMove, the choice-collapse forwards,
# ⊕ products, Mixing/Retention (a closed-form blend of dense applies + broadcasts, so no bespoke
# ext code), and the two continuous-θ Gaussian siblings, MeanPreservingSpread and
# GaussianLoading (on-device scan+Newton + Gaussian-row verbs).
#
# For each case we build a CPU stage, run backward!/forward! to get a Float64
# reference, relocate the stage to the GPU with `to_device(stage, CuArray)`, re-run on
# CuArrays, and assert a tight match (bit-identical where the arithmetic order is
# shared; explicit, justified tolerances where device FMA contraction can drift a
# few ulp). Each stage family covers its operative axis leading and non-leading.

using HouseholdStages
using CUDA
using Test
using Printf
using Random

const HS = HouseholdStages

CUDA.allowscalar(false)

include("../device_walk.jl")

# Device inputs (`V_end` / `Λ_start`) for the comparisons below.
to_dev(x::AbstractArray) = CuArray(collect(x))
to_dev(x)               = x
gpu_stage(stage::AbstractStage) = to_device(stage, CuArray)

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

# ConsumptionSavings at n_w = 16 — the continuous-argmax device kernel at a small grid size.
function build_cs_n16(n_w = 16)
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

# ConsumptionSavings at n_w = 17 — the same device kernel at a second grid size.
function build_cs_n17(n_w = 17)
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

# --- brute argmax builders (buy/sell-home) -------------------------------
# The discrete `(max, +)` argmax over an UNORDERED axis (housing). These exercise the
# GPU brute kernel in the ext (the only JMP-fresh backward gap on device). Each uses a
# TIE-HEAVY integer-valued V_end (exact ties across the housing axis) so the first-index
# tie-break is stressed: the policy index must match CPU bit-identically, not just the value.

# Housing axis at the front (WD == 1) vs non-leading (WD == 3, pre > 1 — the real JMP layout
# where housing sits behind wealth × income). The non-leading case is the one whose fibers are
# strided in memory; the leading case has them contiguous.
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

# A DEP-CARRYING brute reward: the reward table varies along :z, so the walk reads a different
# (after, before) face per stratum. The device path reaches it through the generic stratified seam —
# the reward field rides in as a payload and `_slice` projects it onto its dep axes — so value and
# policy must match the CPU bit-for-bit, exactly as the dep-free cases do.
function build_buyhome_dep()
    h = Discrete([0.0, 1.0, 2.0, 3.0])
    w = GriddedContinuous(collect(range(0.0, 4.0; length = 6)))
    z = Discrete([0.5, 1.5, 2.5])
    layout = GriddedLayout(:wealth => w, :h => h, :z => z)
    base   = [b - a >= 0 ? 0.5 * log1p(b - a) : -Inf for a in 1:4, b in 1:4]
    stage  = ArgmaxStage(layout; reward = (; z) -> base .+ 0.1z, axis = :h)
    Random.seed!(20260731)
    V_end = float.(rand(0:2, axissize.(layout.axes)...))
    Λ = rand(axissize.(layout.axes)...); Λ ./= sum(Λ)
    (stage, V_end, Λ, nothing)
end

const BRUTE_CASES = [
    ("BuyHome  (housing leading)",     () -> build_buyhome(hpos = 1)),
    ("BuyHome  (housing non-leading)", () -> build_buyhome(hpos = 3)),
    ("SellHome (housing leading)",     () -> build_sellhome(hpos = 1)),
    ("SellHome (housing non-leading)", () -> build_sellhome(hpos = 3)),
    ("BuyHome  (dep-carrying reward)",  build_buyhome_dep),
]

# Run one brute stage CPU vs GPU. Returns (vΔ exact, policy-mismatch count, fw_ok, fw_Δ).
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
# The forward scatter axis GROWS from source to destination: the stage collapses a
# size-1 choice axis whose ArgmaxStage forward grew it (source 1 → destination 2). It
# exercises the `NearestScatterOp` grown-axis path where `size(Λ_start, WD) ≠ size(Λ_end, WD)`.
# Source has one row per column ⇒ each destination cell is written once, so the forward is
# bit-identical to the CPU scatter (no collision order).

# EndogenousExit: collapse on :exiting (the survivor/exit choice).
function build_exit_collapse()
    layout = GriddedLayout(:x => Discrete([1, 2, 3, 4]), :exiting => Discrete([0]))
    stage  = EndogenousExit(layout; bequest = 2.0)
    V_end  = reshape([3.0, 2.5, 1.0, 2.0], 4, 1)
    Λ      = reshape([0.1, 0.2, 0.3, 0.4], 4, 1)
    (stage, V_end, Λ, NamedTuple())
end

const COLLAPSE_CASES = [
    ("EndogenousExit", build_exit_collapse),
]

# --- ProductStage (⊕ direct sum) builders ---------------------------------
# Relocating a `ProductStage` recurses over each bundled factor and carries the product's OWN fused
# V/Λ tensors across — the only combinator buffer that owns arrays (a ChainStage owns
# none). These exercise that recursion with GPU-able factors: MarkovStage factors (the dense
# path) and ConsumptionSavings factors (each a ChainStage → the continuous-argmax CUDA kernel). Kept
# small (shared card). The product axis position is toggled so the fused-slice writes hit pdim ≠ 1.

# `n` uniform MarkovStage factors joined along :group (axis position toggled by `group_first`).
function build_prod_markov(; n::Int, group_first::Bool)
    P = [0.7 0.3; 0.3 0.7]
    layout = group_first ?
        GriddedLayout(:group => Discrete([1]), :z => Discrete([0.5, 1.5])) :
        GriddedLayout(:z => Discrete([0.5, 1.5]), :group => Discrete([1]))
    s  = MarkovStage(layout; axis = :z, transition_matrix = P)
    ps = product(ntuple(_ -> s, n)...; axis = :group)
    osz = layout_size(end_layout(ps)); isz = layout_size(start_layout(ps))
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
    isz = layout_size(start_layout(ps))
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
# The closed-form blend `V = b + θ*·(a−b) − cost(θ*)` is TWO DenseKernel backward applies
# (batched `mul!`, the dense device path) plus pointwise broadcasts for the frozen policy θ* and
# the conjugate value — no bespoke ext code. The primitive-stage relocation rule carries the
# `MixingKernel` (both seated DenseKernels + the θstar field) across, along with the io scratch and
# `kernel_scratch` (the two dense plans plus the `mixA`/`mixB` blend workspaces);
# the spec (K_A/K_B matrices, closures) stays host-side. RetentionStage is the `K_A = I` special case.
# The mixing axis position is toggled (leading vs non-leading) to drive the dense sub-applies'
# permute branch. Value AND seated policy θ* are checked (tight: the dense `mul!` may reassociate
# ~1e-13, and θ* is a continuous clamp of the matmul output).

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
    on_dev  = gstage.kernel.θstar isa CuArray && gstage.scratch.V_start isa CuArray
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
# over the axis — that gated case is the brute kernel). Both verbs are the shared fiber ops
# (`NearestGatherOp`, `NearestScatterOp`) ported to the device. The operative axis is exercised
# leading (WD == 1, contiguous fibers) and non-leading (WD == 3, strided fibers).

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
# Both backends run THE SAME shared source per column — `_ca_divide_conquer_walk!` for the best
# grid node over the materialised reward face, then the closed-form PARABOLIC vertex
# `_caC_refine_vertex` for the FLOAT policy position — so CPU and GPU differ only in the outer loop
# (column sweep vs one thread per front-permuted column). The payoff never runs on the device; only
# the face (filled host-side) is consumed. The result is NOT bit-identical: the node value matches
# to sub-ulp (the same adds in the same order), but the vertex is a ratio of node-value differences,
# so GPU FMA contraction can perturb the float policy's last few bits near a flat optimum. Same
# bracketing interval, same answer, only sub-position drift — not a wrong optimum. Tolerances below
# are set to the observed magnitudes, explicitly NOT 0.

const CA_VAL_ATOL = 1e-12     # node value: observed sub-ulp; 1e-12 a tight, justified ceiling
const CA_POL_ATOL = 1e-6      # float policy position: parabolic-vertex FMA-reassociation drift (few ulp)
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

# Bimodal reward: two wells at a' = 0.2 and 0.8 with the argmax crossing at x = env.b, which the
# origin grid straddles — so the device runs the switch detector and the two-mode seating, not only
# the vertex. Without such a case the device gate never reaches the branch.
bimodal_reward(x, ap; env) = x * ap - env.b * ap - 40 * (ap - 0.2)^2 * (ap - 0.8)^2

function build_ca_bimodal_leading()
    N      = 21
    grid   = collect(range(0.0, 1.0; length = N))
    layout = GriddedLayout(:a => GriddedContinuous(grid))
    stage  = ContinuousArgmaxStage(layout; reward = bimodal_reward, axis = :a)
    Random.seed!(20260805); Λ = rand(N); Λ ./= sum(Λ)
    (stage, zeros(N), Λ, (b = 0.54,))
end

function build_ca_bimodal_nonleading()
    N      = 21
    grid   = collect(range(0.0, 1.0; length = N))
    layout = GriddedLayout(:y => Discrete([0.5, 1.5]), :a => GriddedContinuous(grid))
    stage  = ContinuousArgmaxStage(layout; reward = bimodal_reward, axis = :a)
    Random.seed!(20260806); Λ = rand(2, N); Λ ./= sum(Λ)
    (stage, zeros(2, N), Λ, (b = 0.54,))
end

const CA_CASES = [
    ("ContinuousArgmax (no-env, leading)",  build_ca_1d_leading),
    ("ContinuousArgmax (dep-z, leading)",   build_ca_dep_leading),
    ("ContinuousArgmax (env, leading)",     build_ca_env_leading),
    ("ContinuousArgmax (dep-z, non-lead)",  build_ca_dep_nonleading),
    ("ContinuousArgmax (env, non-lead)",    build_ca_env_nonleading),
    ("ContinuousArgmax (bimodal, leading)", build_ca_bimodal_leading),
    ("ContinuousArgmax (bimodal, non-lead)",build_ca_bimodal_nonleading),
]

# How many cells each backend seated as a two-mode lottery, and whether they are the same cells. A
# vertex that drifts across a node moves a bracket by one index but can never widen it past one, so
# the straddled set is the branch's own signature and is compared exactly.
straddled(k) = findall(>(Int32(1)), Array(k.hi) .- Array(k.lo))

# Run one ContinuousArgmax stage CPU vs GPU, held to the tolerances above rather than bit-for-bit.
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
    sw     = straddled(stage.kernel)
    return (vΔ, pΔ, fΔ, finite, sw, straddled(gstage.kernel) == sw)
end

# --- continuous-θ Gaussian builders (MeanPreservingSpread / GaussianLoading) ------
# The two streaming kernel-choice siblings share the banded Gaussian-row primitives and the
# scan+Newton solver, and the SAME device story: CPU and GPU differ not only by FMA contraction
# but by erf IMPLEMENTATION (openlibm vs libdevice, ulp-level), and the per-cell Newton solve
# AMPLIFIES that ulp drift through the FOC — so the float θ* policy gets a loose *_POL_ATOL and
# the value/forward budgets sit between the bit-exact discrete families and the CA float-vertex
# family. MeanPreservingSpread moves the WEIGHTS (mean fixed at x, sd θ); GaussianLoading moves the
# LANDING (mean w·(anchor + θμ), sd |w|θσ) — each gets its own cases at both axis positions.

const MPS_VAL_ATOL = 1e-9     # envelope value: erf ulp drift, flat objective at θ* ⇒ second-order in pΔ
const MPS_POL_ATOL = 1e-6     # float θ* policy: Newton amplifies the libdevice-vs-openlibm erf divergence
const MPS_FWD_ATOL = 1e-7     # forward replay through the seated Gaussian row at (possibly drifted) θ*

# MeanPreservingSpread (continuous dispersion θ, Gaussian-Young row): convex-flanked V ⇒ interior θ
# solved by the on-device scan+Newton; the env-reading closure cost exercises the device cost-closure
# seam (isbits gate). `xfirst` toggles the choice-axis position.
function build_mps(; xfirst::Bool)
    xs = collect(0.0:0.5:10.0); nx = length(xs)
    layout = xfirst ?
        GriddedLayout(:x => GriddedContinuous(xs), :z => Discrete([0.5, 1.5])) :
        GriddedLayout(:z => Discrete([0.5, 1.5]), :x => GriddedContinuous(xs))
    stage  = MeanPreservingSpreadStage(layout; axis = :x, θ_max = 2.0,
                                       cost = (θ; env) -> env.λ * θ^2)
    Vb     = @. 2.0 * exp(-(xs - 5.0)^2 / 4.0)
    V_end  = xfirst ? [Vb[i] + 0.1 * z for i in 1:nx, z in [0.5, 1.5]] :
                      [0.1 * z + Vb[i] for z in [0.5, 1.5], i in 1:nx]
    Random.seed!(20260630)
    Λ = xfirst ? rand(nx, 2) : rand(2, nx); Λ ./= sum(Λ)
    (stage, V_end, Λ, (; λ = 0.05))
end

const GL_VAL_ATOL = 1e-12     # envelope value at the frozen θ*: flat objective ⇒ second-order in pΔ
const GL_POL_ATOL = 1e-6      # float θ* policy: Newton amplifies the libdevice-vs-openlibm erf divergence
const GL_FWD_ATOL = 1e-7      # forward replay through the seated Gaussian row at (possibly drifted) θ*

# GaussianLoading (continuous loading θ, here the risky share on a truncated-Gaussian gross
# return): concave √-wealth V ⇒ interior θ solved by the on-device scan+Newton; the nonzero cost captures a SCALAR so the
# closure stays isbits (the device cost-closure gate). Positive grid keeps s = |w|θσ > 0 off the
# deterministic branch at interior θ. `wfirst` toggles the choice-axis position.
function build_gl_gauss(; wfirst::Bool)
    ws = collect(0.5:0.25:8.0); nw = length(ws)
    κc = 0.01                                     # captured scalar ⇒ isbits closure
    layout = wfirst ?
        GriddedLayout(:wealth => GriddedContinuous(ws), :z => Discrete([0.5, 1.5])) :
        GriddedLayout(:z => Discrete([0.5, 1.5]), :wealth => GriddedContinuous(ws))
    stage  = GaussianLoadingStage(layout; axis = :wealth, anchor = 1.02,
                               increment_mean = 0.03, increment_sd = 0.2,
                               cost = (θ; env) -> κc * θ^2)
    V_end  = wfirst ? [sqrt(w) + 0.1 * z for w in ws, z in [0.5, 1.5]] :
                      [0.1 * z + sqrt(w) for z in [0.5, 1.5], w in ws]
    Random.seed!(20260729)
    Λ = wfirst ? rand(nw, 2) : rand(2, nw); Λ ./= sum(Λ)
    (stage, V_end, Λ, NamedTuple())
end

const MPS_CASES = [
    ("MeanPreservingSpread (x leading)",     () -> build_mps(xfirst = true)),
    ("MeanPreservingSpread (x non-leading)", () -> build_mps(xfirst = false)),
]

const GL_CASES = [
    ("GaussianLoading (wealth leading)",     () -> build_gl_gauss(wfirst = true)),
    ("GaussianLoading (wealth non-leading)", () -> build_gl_gauss(wfirst = false)),
]

# Run one continuous-θ Gaussian stage (MeanPreservingSpread OR GaussianLoading — both seat a
# `kernel.θstar` field) CPU vs GPU. `on_dev` covers the plan's two griddings as well as θ*: a
# host-resident grid still gives the right answers (the extension re-uploads it per call) and would
# slip through the value comparison, so it needs its own assertion.
function mps_cpu_gpu_compare(stage, V_end, Λ_start, env)
    V_cpu   = copy(backward!(stage, V_end, env))
    pol_cpu = copy(policy(stage))
    Λ_cpu   = copy(forward!(stage, Λ_start))

    gstage  = gpu_stage(stage)
    V_gpu   = Array(backward!(gstage, to_dev(V_end), env))
    on_dev  = gstage.kernel.θstar isa CuArray &&
              gstage.scratch.kernel_scratch.origin_grid isa CuArray &&
              gstage.scratch.kernel_scratch.dest_grid   isa CuArray
    pol_gpu = Array(policy(gstage))
    Λ_gpu   = Array(forward!(gstage, to_dev(Λ_start)))

    vΔ  = maximum(abs.(V_gpu .- V_cpu))
    pΔ  = maximum(abs.(pol_gpu .- pol_cpu))
    fwΔ = maximum(abs.(Λ_gpu .- Λ_cpu))
    return (vΔ, pΔ, fwΔ, on_dev)
end

# --- env-resolved array payoff builders -----------------------------------
# A `FromEnv` payoff whose env value is a layout-shaped ARRAY. The stage seats it into a
# `ScalarField`, and every refill of a relocated field routes the host-built values through the
# relocation target ON the way in: the next broadcast (`V_start .= payoff .+ V_end`,
# `Λ_end .= Λ_start .+ kernel.g`, `destinations(kernel) .= …`) mixes the payoff with device tensors
# and will not compile against a host array. The three stages below are the three shapes that reach
# a `ScalarField` — an additive value payoff, an additive measure source, and a per-cell
# destination — each run in both relocation orders (see `fromenv_cpu_gpu_compare`).

const FROMENV_LAYOUT = GriddedLayout(:x => GriddedContinuous([0.0, 1.0, 2.0]),
                                     :z => Discrete([0.5, 1.5]))

# A deterministic layout-shaped mass, normalised — no RNG, so the comparison is reproducible.
_fromenv_mass(m, n) = (Λ = [0.1 * i + 0.05 * j for i in 1:m, j in 1:n]; Λ ./ sum(Λ))

# UtilityStage with `utility = FromEnv(:u)`: the value payoff is an env array.
function build_utility_fromenv()
    stage = UtilityStage(FROMENV_LAYOUT; utility = FromEnv(:u))
    u     = [0.5 * i + 0.25 * j for i in 1:3, j in 1:2]
    V_end = [0.1 * i + 0.2 * j for i in 1:3, j in 1:2]
    (stage, V_end, _fromenv_mass(3, 2), (u = u,))
end

# EntryStage with `entry = FromEnv(:g)`: the measure source is an env array.
function build_entry_fromenv()
    stage = EntryStage(FROMENV_LAYOUT; entry = FromEnv(:g))
    g     = [0.01 * (i + j) for i in 1:3, j in 1:2]
    (stage, zeros(3, 2), _fromenv_mass(3, 2), (g = g,))
end

# DeterministicContinuousStage with `destination = FromEnv(:d)`: the per-cell landing is an env array.
function build_detcont_fromenv()
    grid   = collect(range(0.0, 4.0; length = 5))
    layout = GriddedLayout(:wealth => GriddedContinuous(grid), :income => Discrete([0.6, 1.4]))
    stage  = DeterministicContinuousStage(layout; destination = FromEnv(:d), axis = :wealth)
    d      = [0.5 * w + 0.5 for w in grid, y in 1:2]
    V_end  = [0.3 * w + 0.1 * y for w in 1:5, y in 1:2]
    (stage, V_end, _fromenv_mass(5, 2), (d = d,))
end

const FROMENV_CASES = [
    ("Utility (FromEnv array)",  build_utility_fromenv),
    ("Entry   (FromEnv array)",  build_entry_fromenv),
    ("DetCont (FromEnv array)",  build_detcont_fromenv),
]

# Run one env-resolved-array stage CPU vs GPU, in the relocation order `seat_on_host` picks:
# `true` relocates the stage the host reference just solved, `false` relocates a freshly built one,
# so the device refill is the first thing ever to seat the payoff — the order
# `build → to_device → solve` gives, and the only order `to_device(lift_jacobian(stage), …)` can
# give. Returns (on_dev, bw_ok, bw_Δ, fw_ok, fw_Δ): the device `backward!` must both agree with the
# CPU reference and leave the refilled payoff buffer device-resident.
function fromenv_cpu_gpu_compare(build; seat_on_host::Bool)
    stage, V_end, Λ_start, env = build()
    V_cpu = copy(backward!(stage, V_end, env))
    Λ_cpu = copy(forward!(stage, Λ_start))

    gstage = gpu_stage(seat_on_host ? stage : first(build()))
    V_gpu  = Array(backward!(gstage, to_dev(V_end), env))
    on_dev = all(f -> !(f isa ScalarField) || f.data isa CuArray, values(gstage.cache))
    Λ_gpu  = Array(forward!(gstage, to_dev(Λ_start)))

    bw_ok, bw_d = match_tight(V_gpu, V_cpu)
    fw_ok, fw_d = match_tight(Λ_gpu, Λ_cpu)
    return (on_dev, bw_ok, bw_d, fw_ok, fw_d)
end

# --- run ------------------------------------------------------------------

const CASES = [
    ("WealthChange (wealth leading)",     build_wc_leading),
    ("WealthChange (wealth non-leading)", build_wc_nonleading),
    ("ConsumptionSavings (n_w=16)",       () -> build_cs_n16(16)),
    ("ConsumptionSavings (n_w=17)",       () -> build_cs_n17(17)),
    ("ConsumptionSavings (non-leading)",  () -> build_cs_nonleading(17)),
]

const RESIDENCY_CASES = vcat(CASES, BRUTE_CASES, DISCRETE_MOVE_CASES, CA_CASES, COLLAPSE_CASES,
                             PRODUCT_CASES, MIXING_CASES, MPS_CASES, GL_CASES, FROMENV_CASES)

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
    @testset "GPU brute argmax (buy/sell-home, tie-heavy)" begin
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
            vΔ, pΔ, fΔ, finite, sw, swok = ca_cpu_gpu_compare(stage, V_end, Λ_start, env)
            @printf("%-34s value Δ=%.2e %s   policy Δ=%.2e %s   forward Δ=%.2e %s   straddled=%d %s\n",
                    name, vΔ, vΔ <= CA_VAL_ATOL ? "OK" : "FAIL",
                    pΔ, pΔ <= CA_POL_ATOL ? "OK" : "FAIL", fΔ, fΔ <= CA_FWD_ATOL ? "OK" : "FAIL",
                    length(sw), swok ? "OK" : "FAIL")
            all_ok &= (vΔ <= CA_VAL_ATOL) & (pΔ <= CA_POL_ATOL) & (fΔ <= CA_FWD_ATOL) & finite & swok
            @testset "$name" begin
                @test finite                # value finite everywhere; policy a Float64 position
                @test vΔ <= CA_VAL_ATOL     # value sub-ulp (same optimum)
                @test pΔ <= CA_POL_ATOL     # float policy position: parabolic-vertex FMA drift
                @test fΔ <= CA_FWD_ATOL     # the seated split replayed forward
                @test swok                  # the same bins straddle a switch on both backends
                occursin("bimodal", name) && @test !isempty(sw)   # the branch is actually reached
            end
        end
    end
    @testset "GPU choice-collapse forward (grown scatter axis)" begin
        for (name, build) in COLLAPSE_CASES
            stage, V_end, Λ_start, env = build()
            # `cpu_gpu_compare` seats each side's policy (backward!) before forward!.
            bw_ok, bw_d, fw_ok, fw_d = cpu_gpu_compare(stage, V_end, Λ_start, env)
            @printf("%-34s backward Δ=%.2e %s   forward Δ=%.2e %s\n",
                    name, bw_d, bw_ok ? "OK" : "FAIL", fw_d, fw_d == 0 ? "OK" : "FAIL")
            all_ok &= bw_ok & (fw_d == 0)
            @testset "$name" begin
                @test fw_d == 0         # forward scatter bit-identical: one write per destination
                @test bw_ok             # backward (the brute kernel) matches, tight
            end
        end
    end
    @testset "GPU ProductStage (⊕ direct sum)" begin
        for (name, build) in PRODUCT_CASES
            stage, V_end, Λ_start, env = build()
            # Relocation carries the product's fused tensors to the device too — assert they actually
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
            # The generic lift moves the MixingKernel (both seated DenseKernels + θstar)
            # and the V/Λ/mix scratch to the device — assert they actually landed (a silent host
            # round-trip would still pass the value match but defeat the lift). Value, seated θ*,
            # and forward all tight (the dense `mul!` may reassociate ~1e-13 on device).
            on_dev, bw_ok, bw_d, pol_ok, pol_d, fw_ok, fw_d =
                mixing_cpu_gpu_compare(stage, V_end, Λ_start, env)
            @printf("%-30s on-dev %s   backward Δ=%.2e %s   policy Δ=%.2e %s   forward Δ=%.2e %s\n",
                    name, on_dev ? "OK" : "FAIL", bw_d, bw_ok ? "OK" : "FAIL",
                    pol_d, pol_ok ? "OK" : "FAIL", fw_d, fw_ok ? "OK" : "FAIL")
            all_ok &= on_dev & bw_ok & pol_ok & fw_ok
            @testset "$name" begin
                @test on_dev            # kernel θ* and V scratch live on the device (the lift's core step)
                @test bw_ok             # backward value matches the CPU reference (tight)
                @test pol_ok            # seated mixing policy θ* matches (tight)
                @test fw_ok             # forward (blended push through kA/kB) matches (tight)
            end
        end
    end
    @testset "GPU MeanPreservingSpread (continuous-θ Gaussian-Young)" begin
        for (name, build) in MPS_CASES
            stage, V_end, Λ_start, env = build()
            vΔ, pΔ, fwΔ, on_dev = mps_cpu_gpu_compare(stage, V_end, Λ_start, env)
            @printf("%-38s value Δ=%.2e %s   policy Δ=%.2e %s   forward Δ=%.2e %s\n",
                    name, vΔ, vΔ <= MPS_VAL_ATOL ? "OK" : "FAIL",
                    pΔ, pΔ <= MPS_POL_ATOL ? "OK" : "FAIL", fwΔ, fwΔ <= MPS_FWD_ATOL ? "OK" : "FAIL")
            all_ok &= (vΔ <= MPS_VAL_ATOL) & (pΔ <= MPS_POL_ATOL) & (fwΔ <= MPS_FWD_ATOL) & on_dev
            @testset "$name" begin
                @test on_dev               # per-cell θ* policy AND plan grid live on the device
                @test pΔ <= MPS_POL_ATOL   # float θ*: Newton-amplified erf-implementation drift
                @test vΔ <= MPS_VAL_ATOL   # envelope value (flat in θ at θ* ⇒ second-order in pΔ)
                @test fwΔ <= MPS_FWD_ATOL  # forward replay through the seated Gaussian row
            end
        end
    end
    @testset "GPU GaussianLoading (continuous-θ)" begin
        for (name, build) in GL_CASES
            stage, V_end, Λ_start, env = build()
            vΔ, pΔ, fwΔ, on_dev = mps_cpu_gpu_compare(stage, V_end, Λ_start, env)
            @printf("%-38s value Δ=%.2e %s   policy Δ=%.2e %s   forward Δ=%.2e %s\n",
                    name, vΔ, vΔ <= GL_VAL_ATOL ? "OK" : "FAIL",
                    pΔ, pΔ <= GL_POL_ATOL ? "OK" : "FAIL", fwΔ, fwΔ <= GL_FWD_ATOL ? "OK" : "FAIL")
            all_ok &= (vΔ <= GL_VAL_ATOL) & (pΔ <= GL_POL_ATOL) & (fwΔ <= GL_FWD_ATOL) & on_dev
            @testset "$name" begin
                @test on_dev               # per-cell θ* policy AND plan grid live on the device
                @test pΔ <= GL_POL_ATOL    # float θ*: Newton-amplified erf-implementation drift
                @test vΔ <= GL_VAL_ATOL    # envelope value (flat in θ at θ* ⇒ second-order in pΔ)
                @test fwΔ <= GL_FWD_ATOL   # forward replay through the seated Gaussian row
            end
        end
    end
    @testset "GPU env-resolved array payoffs (FromEnv fields)" begin
        for (name, build) in FROMENV_CASES, seat_on_host in (true, false)
            order = seat_on_host ? "solve→relocate" : "relocate→solve"
            on_dev, bw_ok, bw_d, fw_ok, fw_d = fromenv_cpu_gpu_compare(build; seat_on_host)
            @printf("%-24s %-15s payoff-on-dev %s   backward Δ=%.2e %s   forward Δ=%.2e %s\n",
                    name, order, on_dev ? "OK" : "FAIL", bw_d, bw_ok ? "OK" : "FAIL",
                    fw_d, fw_ok ? "OK" : "FAIL")
            all_ok &= on_dev & bw_ok & fw_ok
            @testset "$name ($order)" begin
                @test on_dev            # the refilled payoff buffer lands on the device
                @test bw_ok             # backward matches the CPU reference (tight)
                @test fw_ok             # forward matches the CPU reference (tight)
            end
        end
    end
    @testset "GPU device residency (every relocated array is a CuArray)" begin
        # The walker in `test/device_walk.jl` visits exactly what the relocation visits, so a type
        # with no rebuild rule shows up here as a host-resident array inside a relocated stage.
        for (name, build) in RESIDENCY_CASES
            stage   = first(build())
            arrays  = reachable_arrays(gpu_stage(stage))
            on_host = count(a -> !(a isa CuArray), arrays)
            @printf("%-38s arrays=%3d   host-resident=%d %s\n",
                    name, length(arrays), on_host, on_host == 0 ? "OK" : "FAIL")
            all_ok &= (on_host == 0) & !isempty(arrays)
            @testset "$name" begin
                @test !isempty(arrays)  # the walk reaches the stage's buffers at all
                @test on_host == 0      # every one of them is device-resident
            end
        end
    end
    return all_ok
end

if abspath(PROGRAM_FILE) == @__FILE__
    ok = run_kernel_tests()
    exit(ok ? 0 : 1)
end
