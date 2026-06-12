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
        StateAxis(:wealth, continuous_grid([0.0, 1.0, 2.0, 3.0, 4.0])),
        StateAxis(:income, discrete_finite([0.5, 1.0, 1.5])),
    )
    stage = WealthChangeStage(layout;
        wealth_post = (cell; env) -> (1 + env.r) * cell.wealth - 0.1,
        wealth_axis = :wealth)
    V_end = [0.3 * w + 0.1 * y for w in 1:5, y in 1:3]
    Λ = rand(5, 3); Λ ./= sum(Λ)
    (stage, V_end, Λ, (r = 0.04,))
end

# WealthChange, wealth axis NOT leading (income first).
function build_wc_nonleading()
    layout = GriddedLayout(
        StateAxis(:income, discrete_finite([0.5, 1.0, 1.5])),
        StateAxis(:wealth, continuous_grid([0.0, 1.0, 2.0, 3.0, 4.0])),
    )
    stage = WealthChangeStage(layout;
        wealth_post = (cell; env) -> (1 + env.r) * cell.wealth - 0.1,
        wealth_axis = :wealth)
    V_end = [0.1 * y + 0.3 * w for y in 1:3, w in 1:5]
    Λ = rand(3, 5); Λ ./= sum(Λ)
    (stage, V_end, Λ, (r = 0.04,))
end

# ConsumptionSavings with n_w-1 NOT a power of two (sequential fallback kernel).
function build_cs_seq(n_w = 16)
    layout = GriddedLayout(
        StateAxis(:wealth, continuous_grid(
            [exp(t) - 1.0 for t in range(0.0, log(21.0); length = n_w)])),
        StateAxis(:income, discrete_finite([0.6, 1.0, 1.4])),
    )
    stage = ConsumptionSavingsStage(layout; β = 0.96,
        utility = (cell, c; env) -> log(c), wealth_axis = :wealth)
    V_end = [0.1 * w + 0.05 * y for w in 1:n_w, y in 1:3]
    Λ = rand(n_w, 3); Λ ./= sum(Λ)
    (stage, V_end, Λ, NamedTuple())
end

# ConsumptionSavings with n_w-1 a power of two (iterative reference kernel).
function build_cs_pow2(n_w = 17)
    @assert ispow2(n_w - 1)
    layout = GriddedLayout(
        StateAxis(:wealth, continuous_grid(
            [exp(t) - 1.0 for t in range(0.0, log(21.0); length = n_w)])),
        StateAxis(:income, discrete_finite([0.6, 1.0, 1.4])),
    )
    stage = ConsumptionSavingsStage(layout; β = 0.96,
        utility = (cell, c; env) -> log(c), wealth_axis = :wealth)
    V_end = [0.1 * w + 0.05 * y for w in 1:n_w, y in 1:3]
    Λ = rand(n_w, 3); Λ ./= sum(Λ)
    (stage, V_end, Λ, NamedTuple())
end

# ConsumptionSavings, wealth axis NOT leading.
function build_cs_nonleading(n_w = 17)
    @assert ispow2(n_w - 1)
    layout = GriddedLayout(
        StateAxis(:income, discrete_finite([0.6, 1.0, 1.4])),
        StateAxis(:wealth, continuous_grid(
            [exp(t) - 1.0 for t in range(0.0, log(21.0); length = n_w)])),
    )
    stage = ConsumptionSavingsStage(layout; β = 0.96,
        utility = (cell, c; env) -> log(c), wealth_axis = :wealth)
    V_end = [0.05 * y + 0.1 * w for y in 1:3, w in 1:n_w]
    Λ = rand(3, n_w); Λ ./= sum(Λ)
    (stage, V_end, Λ, NamedTuple())
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
    return all_ok
end

if abspath(PROGRAM_FILE) == @__FILE__
    ok = run_kernel_tests()
    exit(ok ? 0 : 1)
end
