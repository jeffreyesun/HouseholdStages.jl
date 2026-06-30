# GPU survey for HouseholdStages stages.
#
# For each stage we can construct simply, build a small CPU instance,
# run backward!/forward! on the CPU to get a reference, then move the
# stage's buffer to the GPU and re-run on CuArray inputs. PASS means the
# GPU result matches the CPU reference; FAIL records the error class;
# NEEDS-KERNEL flags stages known to require a hand-written GPU kernel.
#
# CPU→GPU path. We move a *constructed* stage onto the GPU through the
# production `to_device(stage, move_fn)` (src/lifts/gpu.jl), passing a
# Float64-preserving mover so the ≈ comparison against the CPU reference is
# clean rather than degraded to Float32. `to_device` moves the buffer's
# isbits-eltype arrays (kernel/scratch/V_start/Λ_end) and the Spec's array
# fields (transition, cost_matrix, …) to the device, reconstructing the
# kernel/scratch structs and re-inferring the Spec's type params. Non-isbits
# arrays (a Symbol-celled cell_array) stay host-side. Inputs (V_end / Λ_start)
# are passed as device arrays.

using HouseholdStages
using CUDA
using Printf

const HS = HouseholdStages

# Fail loudly on accidental scalar indexing rather than silently crawling.
CUDA.allowscalar(false)

# ---------------------------------------------------------------------
# CPU→GPU move — the production `to_device(stage, move_fn)` from
# `src/lifts/gpu.jl`. CUDA is not a dependency of the package, so the
# package takes the device mover as an argument; the GPU test env
# supplies it here. We pass a *Float64-preserving* mover (rather than
# `CUDA.cu`, which defaults to Float32) so the GPU result compares
# cleanly against the Float64 CPU reference instead of degrading to
# single precision. `to_device` itself:
#   - moves isbits-eltype arrays to the device, leaving non-isbits
#     arrays (e.g. a Symbol-celled `cell_array`) host-side;
#   - reconstructs kernel/scratch structs (so their array fields must
#     be declared on an `<:AbstractArray` type parameter);
#   - moves the Spec's array fields and re-infers its type params;
#   - installs a GPU-safe `_seat_cache!` fingerprint (no host `hash`
#     shim is needed any more — `abstract.jl` dispatches the
#     fingerprint on the array type).
# ---------------------------------------------------------------------

# Float64-preserving device mover for isbits arrays; pass everything else
# through (`to_device` already guards on isbits-ness, but keep the eltype
# explicit here so a stray non-`Array` AbstractArray still lands as Float64).
to_dev(x::AbstractArray) = CuArray(collect(x))
to_dev(x)               = x

gpu_stage(stage::AbstractStage) = to_device(stage, to_dev)

# ---------------------------------------------------------------------
# Result recording
# ---------------------------------------------------------------------

mutable struct Outcome
    name   :: String
    status :: Symbol         # :PASS, :FAIL, :NEEDS_KERNEL
    detail :: String
end

const RESULTS = Outcome[]

record!(name, status, detail) = push!(RESULTS, Outcome(name, status, detail))

# Classify a thrown error into a short class label.
function error_class(err)
    s = sprint(showerror, err)
    occursin("scalar indexing", lowercase(s)) && return "scalar indexing"
    occursin("ScalarIndexing", s)             && return "scalar indexing"
    occursin("Illegal conversion", s) && occursin("Ptr", s) &&
        return "host×device matmul (Spec array not on GPU)"
    occursin("GPU compilation", s) &&
        return "host cell_array closure → host operand in device broadcast"
    occursin("InvalidIRError", s)              && return "GPU-invalid IR (kernel)"
    isa(err, MethodError)                      && return "MethodError (unsupported op)"
    occursin("not implemented", s)             && return "not implemented"
    first(split(s, '\n'))
end

# Run one stage's survey. `build` returns (stage, V_end, Λ_start, env).
# We compare GPU backward!/forward! against the CPU reference.
function survey_stage(name; expect_needs_kernel=false)
    return function (build)
        local stage, V_end, Λ_start, env
        try
            stage, V_end, Λ_start, env = build()
        catch err
            record!(name, :FAIL, "CPU construction failed: " * error_class(err))
            return
        end

        # CPU reference.
        local V_cpu, Λ_cpu
        try
            V_cpu = copy(backward!(stage, V_end, env))
            Λ_cpu = copy(forward!(stage, Λ_start))
        catch err
            record!(name, :FAIL, "CPU run failed: " * error_class(err))
            return
        end

        # Move the stage's buffer onto the GPU (backward! seats the kernel on
        # the GPU buffer, so forward! reads GPU kernel data).
        local gstage
        try
            gstage = gpu_stage(stage)
        catch err
            tag = expect_needs_kernel ? :NEEDS_KERNEL : :FAIL
            record!(name, tag, "buffer→GPU move: " * error_class(err))
            return
        end
        V_g     = to_dev(V_end)
        Λ_g     = to_dev(Λ_start)

        local V_gpu, Λ_gpu
        try
            V_gpu = Array(backward!(gstage, V_g, env))
        catch err
            tag = expect_needs_kernel ? :NEEDS_KERNEL : :FAIL
            record!(name, tag, "backward! on GPU: " * error_class(err))
            return
        end
        try
            Λ_gpu = Array(forward!(gstage, Λ_g))
        catch err
            tag = expect_needs_kernel ? :NEEDS_KERNEL : :FAIL
            record!(name, tag, "forward! on GPU: " * error_class(err))
            return
        end

        # Compare. Allow -Inf cells (borrowing constraint) to match.
        bw_ok = _approx_with_inf(V_gpu, V_cpu)
        fw_ok = _approx_with_inf(Λ_gpu, Λ_cpu)
        if bw_ok && fw_ok
            record!(name, :PASS, "backward! & forward! match CPU (≈)")
        else
            parts = String[]
            bw_ok || push!(parts, @sprintf("backward Δmax=%.2e", _maxdiff(V_gpu, V_cpu)))
            fw_ok || push!(parts, @sprintf("forward Δmax=%.2e",  _maxdiff(Λ_gpu, Λ_cpu)))
            record!(name, :FAIL, "numeric mismatch: " * join(parts, ", "))
        end
        return
    end
end

# ≈ that treats matching ±Inf entries as equal.
function _approx_with_inf(a, b; atol=1e-8, rtol=1e-6)
    size(a) == size(b) || return false
    for (x, y) in zip(a, b)
        if isinf(x) || isinf(y)
            (x == y) || return false
        else
            isapprox(x, y; atol=atol, rtol=rtol) || return false
        end
    end
    return true
end

function _maxdiff(a, b)
    size(a) == size(b) || return Inf
    m = 0.0
    for (x, y) in zip(a, b)
        (isinf(x) || isinf(y)) && continue
        m = max(m, abs(x - y))
    end
    return m
end

# ---------------------------------------------------------------------
# Per-stage builders
# ---------------------------------------------------------------------

function run_survey()
    empty!(RESULTS)

    # --- MarkovStage ---
    survey_stage("MarkovStage")() do
        P = [0.7 0.3; 0.3 0.7]
        layout = GriddedLayout(
            :wealth => GriddedContinuous([0.0, 1.0, 2.0, 3.0]),
            :income => Discrete([0.5, 1.5]),
        )
        stage = MarkovStage(layout; axis = :income, transition_matrix = P)
        V_end = reshape(Float64.(1:8), 4, 2)
        Λ = rand(4, 2); Λ ./= sum(Λ)
        (stage, V_end, Λ, nothing)
    end

    # --- LogitChoiceStage ---
    survey_stage("LogitChoiceStage")() do
        layout = GriddedLayout(
            :wealth => GriddedContinuous([0.0, 1.0, 2.0]),
            :a => Discrete([1, 2]),
        )
        stage = LogitChoiceStage(layout;
            axis = :a,
            cost_matrix = [0.0 0.5; 0.5 0.0],
            ε           = 0.7)
        V_end = [0.1 * w + 0.2 * j for w in 1:3, j in 1:2]
        Λ = rand(3, 2); Λ ./= sum(Λ)
        (stage, V_end, Λ, NamedTuple())
    end

    # --- MigrationStage (sugar over LogitChoice) ---
    survey_stage("MigrationStage")() do
        layout = GriddedLayout(
            :wealth => GriddedContinuous([0.0, 0.5, 1.0]),
            :location => Discrete([:home, :abroad]),
        )
        stage = MigrationStage(layout;
            axis = :location,
            migration_cost = [0.0 0.5; 0.5 0.0],
            ε = 1.0)
        V_end = [0.1 * w + 0.3 * (l - 1) for w in 1:3, l in 1:2]
        Λ = rand(3, 2); Λ ./= sum(Λ)
        (stage, V_end, Λ, NamedTuple())
    end

    # --- SectorSwitchingStage (sugar over LogitChoice) ---
    survey_stage("SectorSwitchingStage")() do
        layout = GriddedLayout(
            :wealth => GriddedContinuous([0.0, 0.5, 1.0]),
            :sector => Discrete([:ag, :mfg, :svc]),
        )
        C = [0.0 0.4 0.6; 0.4 0.0 0.4; 0.6 0.4 0.0]
        stage = SectorSwitchingStage(layout; axis = :sector,
                                     switching_cost = C, ε = 0.7)
        V_end = [0.1 * w + 0.2 * s for w in 1:3, s in 1:3]
        Λ = rand(3, 3); Λ ./= sum(Λ)
        (stage, V_end, Λ, NamedTuple())
    end

    # --- ArgmaxStage ---
    # GPU-complete (both legs). backward! is the discrete `(max, +)` `:brute` argmax over an
    # unordered axis — now a device kernel (`_brute_argmax_kernel!` in the ext): one thread per
    # stratum column, scalar `>` first-index tie-break, bit-identical to the CPU brute walk.
    # forward! is the integer-policy scatter, which reuses the CS `_cs_forward_scatter!` device
    # kernel (distinct origins → one destination, one thread owns each column ⇒ no atomics).
    survey_stage("ArgmaxStage")() do
        layout = GriddedLayout(:s => Discrete([:A, :B]))
        # Reward matrix on :s: choosing :B (after index 2) scores +1, :A scores 0.
        stage = ArgmaxStage(layout; axis = :s, reward = [0.0 0.0; 1.0 1.0])
        V_end = Float64[0.0, 0.0]
        Λ = Float64[0.6, 0.4]
        (stage, V_end, Λ, nothing)
    end

    # --- UtilityStage ---
    survey_stage("UtilityStage")() do
        layout = GriddedLayout(
            :wealth => GriddedContinuous([0.0, 1.0, 2.0]),
            :income => Discrete([0.5, 1.0]),
        )
        stage = UtilityStage(layout; utility = (; wealth, env) -> wealth + env.bonus)
        V_end = zeros(3, 2)
        Λ = rand(3, 2); Λ ./= sum(Λ)
        (stage, V_end, Λ, (bonus = 10.0,))
    end

    # --- IdentityStage ---
    survey_stage("IdentityStage")() do
        layout = GriddedLayout(
            :w => GriddedContinuous([0.0, 1.0, 2.0]),
            :z => Discrete([0.5, 1.5]),
        )
        stage = IdentityStage(layout)
        V_end = randn(3, 2)
        Λ = rand(3, 2); Λ ./= sum(Λ)
        (stage, V_end, Λ, nothing)
    end

    # --- BorrowingConstraintStage ---
    survey_stage("BorrowingConstraintStage")() do
        layout = GriddedLayout(
            :wealth => GriddedContinuous([-1.0, 0.0, 1.0, 2.0]),
            :y => Discrete([0.5, 1.0]),
        )
        mask = falses(4, 2); mask[1, :] .= true
        stage = BorrowingConstraintStage(layout; infeasible = mask)
        V_end = reshape(Float64.(1:8), 4, 2)
        Λ = rand(4, 2); Λ[1, :] .= 0.0; Λ ./= sum(Λ)
        (stage, V_end, Λ, NamedTuple())
    end

    # --- ForgetfulSumStage ---
    survey_stage("ForgetfulSumStage")() do
        layout = GriddedLayout(
            :wealth => GriddedContinuous([0.0, 1.0, 2.0, 3.0]),
            :income => Discrete([0.5, 1.0, 1.5]),
            :taste => Discrete([:a, :b, :c, :d, :e]),
        )
        stage = ForgetfulSumStage(layout; axis = :taste)
        V_end = randn(4, 3, 1)         # output layout: :taste resized to size 1
        Λ = rand(4, 3, 5); Λ ./= sum(Λ)
        (stage, V_end, Λ, nothing)
    end

    # --- TimeDiscountingStage ---
    survey_stage("TimeDiscountingStage")() do
        layout = GriddedLayout(
            :wealth => GriddedContinuous([0.0, 1.0, 2.0]),
            :income => Discrete([0.5, 1.0]),
        )
        stage = TimeDiscountingStage(layout; β = 0.96)
        V_end = reshape(Float64.(1:6), 3, 2)
        Λ = rand(3, 2); Λ ./= sum(Λ)
        (stage, V_end, Λ, nothing)
    end

    # --- AdvanceAgeStage (sugar over Markov) ---
    survey_stage("AdvanceAgeStage")() do
        layout = GriddedLayout(
            :age => Discrete([1, 2, 3, 4]),
            :wealth => GriddedContinuous([0.0, 1.0, 2.0]),
        )
        stage = AdvanceAgeStage(layout; axis = :age)
        V_end = reshape(Float64.(1:12), 4, 3)
        Λ = rand(4, 3); Λ ./= sum(Λ)
        (stage, V_end, Λ, nothing)
    end

    # --- WealthChangeStage (hand-written CUDA kernels via HouseholdStagesCUDAExt) ---
    survey_stage("WealthChangeStage")() do
        layout = GriddedLayout(
            :wealth => GriddedContinuous([0.0, 1.0, 2.0, 3.0]),
            :income => Discrete([0.5, 1.0]),
        )
        stage = WealthChangeStage(layout;
            wealth_post = (; wealth, env) -> (1 + env.r) * wealth,
            axis = :wealth)
        V_end = reshape(Float64.(1:8), 4, 2)
        Λ = rand(4, 2); Λ ./= sum(Λ)
        (stage, V_end, Λ, (r = 0.04,))
    end

    # --- ConsumptionSavingsStage (hand-written CUDA kernels via HouseholdStagesCUDAExt) ---
    survey_stage("ConsumptionSavingsStage")() do
        n_w = 16
        layout = GriddedLayout(
            :wealth => GriddedContinuous(
                [exp(t) - 1.0 for t in range(0.0, log(21.0); length = n_w)]),
            :income => Discrete([0.6, 1.0, 1.4]),
        )
        stage = ConsumptionSavingsStage(layout; β = 0.96,
            utility = (cell, c; env) -> log(c),
            axis = :wealth)
        V_end = [0.1 * w + 0.05 * y for w in 1:n_w, y in 1:3]
        Λ = rand(n_w, 3); Λ ./= sum(Λ)
        (stage, V_end, Λ, NamedTuple())
    end

    return RESULTS
end

# ---------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------

function print_matrix(io = stdout)
    println(io, "GPU survey — ", length(RESULTS), " stages")
    println(io)
    badge(s) = s === :PASS ? "PASS        " :
               s === :NEEDS_KERNEL ? "NEEDS-KERNEL" : "FAIL        "
    for o in RESULTS
        @printf(io, "%-26s %s  %s\n", o.name, badge(o.status), o.detail)
    end
    np = count(o -> o.status === :PASS, RESULTS)
    nk = count(o -> o.status === :NEEDS_KERNEL, RESULTS)
    nf = count(o -> o.status === :FAIL, RESULTS)
    println(io)
    @printf(io, "PASS=%d  NEEDS-KERNEL=%d  FAIL=%d\n", np, nk, nf)
end

# Emit the pass/fail matrix as a markdown table to `path`.
function write_markdown(path, gpu_name)
    badge(s) = s === :PASS ? "PASS" :
               s === :NEEDS_KERNEL ? "NEEDS-KERNEL" : "FAIL"
    np = count(o -> o.status === :PASS, RESULTS)
    nk = count(o -> o.status === :NEEDS_KERNEL, RESULTS)
    nf = count(o -> o.status === :FAIL, RESULTS)
    open(path, "w") do io
        println(io, "# GPU survey — HouseholdStages stages")
        println(io)
        println(io, "Generated by `test/gpu/gpu_survey.jl` (run with ",
                    "`julia --project=test/gpu test/gpu/gpu_survey.jl`).")
        println(io)
        println(io, "- **GPU:** ", gpu_name)
        println(io, "- **`CUDA.functional()`:** `true`")
        println(io, "- **Tally:** PASS=", np, "  NEEDS-KERNEL=", nk, "  FAIL=", nf,
                    " (of ", length(RESULTS), ")")
        println(io)
        println(io, "PASS = GPU backward!/forward! match the CPU reference (≈, ",
                    "−Inf cells matched exactly). FAIL = ran on the GPU path but ",
                    "errored or mismatched. NEEDS-KERNEL = known to need a ",
                    "hand-written GPU kernel (scalar-index loops).")
        println(io)
        println(io, "| Stage | Status | Detail |")
        println(io, "|---|---|---|")
        for o in RESULTS
            println(io, "| `", o.name, "` | ", badge(o.status), " | ", o.detail, " |")
        end
        println(io)
        println(io, "**Hand-written CUDA kernels (`HouseholdStagesCUDAExt`).** ",
                    "`ConsumptionSavingsStage` and `WealthChangeStage` — the two ",
                    "stages whose hot path is a scalar-index loop the device ",
                    "cannot broadcast — now run hand-written CUDA kernels via a ",
                    "package extension (`ext/HouseholdStagesCUDAExt.jl`, gated on ",
                    "a `[weakdeps]` CUDA). CUDA stays out of the main ",
                    "`Project.toml`; the extension only loads when CUDA is ",
                    "present, and defines `CuArray`-typed methods at the ",
                    "`src` dispatch seams (`_cs_backward_columns!`, ",
                    "`_cs_forward_scatter!`, `_along_axis`, ",
                    "`_fill_utility_table!`), leaving every CPU path unchanged. ",
                    "The kernels are faithful ports of the GPU-tested reference ",
                    "kernels in `reference_materials/example_stages` ",
                    "(`k1_argmax_kernel!`, `reinterpolate_GPU_kernel!`, ",
                    "`convert_distribution_kernel!`, `get_λ_postc_kernel!`), ",
                    "adapted from the reference's wealth-axis-leading layout to ",
                    "this package's layout-generic arrays by permuting the ",
                    "wealth axis to the front before the kernel and back after. ",
                    "CS backward uses the iterative divide-and-conquer reference ",
                    "kernel when `ispow2(n_w-1)` and a sequential monotone-walk ",
                    "kernel otherwise; both match the CPU result to machine ",
                    "precision (verified in `test/gpu/test_kernels.jl`, including ",
                    "non-leading wealth-axis layouts). The one remaining ",
                    "open item is the CS GPU backward's restriction to ",
                    "consumption-only utilities (the `U` table singleton outside ",
                    "choice×wealth); a state-dependent flow utility falls back to ",
                    "the CPU backward.")
        println(io)
        println(io, "**Note on `ArgmaxStage`.** GPU-complete on both legs. The ",
                    "reward face is assembled host-side (cells may be ",
                    "`Symbol`-valued, which no device broadcast can evaluate) and ",
                    "copied into a device-resident `U` buffer. `backward!` — the ",
                    "discrete `(max, +)` `:brute` argmax over an unordered axis — ",
                    "runs the ext's `_brute_argmax_kernel!` (one thread per stratum ",
                    "column, scalar `>` first-index tie-break, bit-identical to the ",
                    "CPU brute walk including ties; verified value AND policy index ",
                    "in `test/gpu/test_kernels.jl`). `forward!` — the integer-policy ",
                    "scatter (distinct origins → one destination) — reuses the CS ",
                    "`_cs_forward_scatter!` device kernel (one thread owns each ",
                    "column, so the colliding `+=` needs no atomics).")
        println(io)
        println(io, "## CPU→GPU path used by the survey")
        println(io)
        println(io, """
        The survey moves a *constructed* stage onto the GPU through the
        production `to_device(stage, move_fn)` (`src/lifts/gpu.jl`). CUDA is
        not a dependency of the package, so the device mover is supplied by
        this GPU test env; the survey passes a Float64-preserving mover (not
        `CUDA.cu`, which would default to Float32) so the device result
        compares cleanly against the Float64 CPU reference. `to_device`:

        1. **Buffer.** Rebuilds the `StageBuffer`, moving every isbits-eltype
           array (`V_start`, `Λ_end`, and each array in the `kernel`/`scratch`
           structs) to the device. **Non-isbits arrays are left host-side** —
           a `Symbol`-celled `cell_array` cannot be broadcast over on-device,
           so the owning stage (ArgmaxStage) assembles its payoff host-side
           and copies only the numeric result to the device.
        2. **Spec.** Reconstructs the Spec with its array fields (`transition`,
           `cost_matrix`, `infeasible` mask, …) moved to the device
           — Markov/Logit `mul!` their Spec arrays against device data.
        3. **Inputs.** `V_end` / `Λ_start` are passed as device arrays.

        The two former harness shims are now closed in `src`:

        - **`_seat_cache!` fingerprint** — `abstract.jl` no longer calls the
          element-iterating `hash` on a device `V_end`. `_v_fingerprint`
          dispatches on the array type: a host `Array` keeps the full content
          `hash` (what the cache tests assert and what catches in-place
          mutation); any other array fingerprints cheap on-device reductions
          (`size`, `length`, `sum`, `sum-of-squares`). No `hash(::CuArray)`
          shim is needed.
        - **Concrete-typed kernel structs** — `LogitChoiceKernel` now declares
          its array fields on `<:AbstractMatrix`/`<:AbstractVector` type
          parameters (mirroring `WealthChangeKernel`/`ArgmaxKernel`), so the
          buffer move accepts device arrays.
        """)
    end
    return path
end

if abspath(PROGRAM_FILE) == @__FILE__
    if !CUDA.functional()
        println("CUDA NOT FUNCTIONAL — aborting survey.")
        exit(1)
    end
    gpu_name = CUDA.name(CUDA.device())
    println("CUDA functional on: ", gpu_name)
    run_survey()
    print_matrix()
    md = get(ENV, "GPU_SURVEY_MD",
             joinpath(@__DIR__, "..", "..", "..", "GPU_SURVEY.md"))
    write_markdown(md, gpu_name)
    println("\nWrote ", abspath(md))
end
