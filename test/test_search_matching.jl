using Test
using HouseholdStages
using HouseholdStages: fix

# SearchMatchingStage (SAM_RI_STAGE_PROPOSAL.md §3, Route B). A dedicated stage:
# effort is an INTERNAL parameter grid (never a state axis); backward maxes over
# effort to form the unemployed value and stores the effort policy; forward
# replays the policy-selected θ-dependent Bernoulli matching row plus the fixed
# separation row. The labor axis `:emp` has level 1 = unemployed, 2 = employed.
#
# Faces tested:
#   (1) max-collapse correctness — V_unemp matches the closed-form max over effort,
#       V_emp matches the separation mix; the stored effort policy is the argmax;
#   (2) p(θ) comparative static — a tighter market (higher θ) raises job-finding,
#       so the unemployed value rises and forward pushes more mass into employment;
#   (3) duality on the linear matching/separation operator (the cost flow reward is
#       additive on the V side, so the clean identity is on the K part — cf. the
#       MigrationStage duality test);
#   (4) mass conservation (each labor column is stochastic);
#   (5) env-dependent matching via FromEnv(:θ) — θ is declared in the env slice and
#       re-resolved each backward.

# A two-level labor axis × a wealth axis. θ drives job-finding via the closure.
function _sam_layout(; n_w = 4)
    return GriddedLayout(
        StateAxis(:wealth, continuous_grid(collect(range(0.0, 3.0; length = n_w)))),
        StateAxis(:emp,    categorical([:unemp, :emp])),   # level 1 = unemployed, 2 = employed
    )
end

# Job-finding probability increasing in effort and tightness; bounded in (0,1).
_p(e, θ) = 1 - exp(-e * θ)
_cost(e) = 0.5 * e^2

function _sam_stage(layout; efforts = collect(range(0.0, 2.0; length = 6)),
                    δ = 0.10, tightness = FromEnv(:θ))
    return SearchMatchingStage(layout; labor_axis = :emp, efforts = efforts,
                               cost = _cost, job_finding = _p,
                               separation = δ, tightness = tightness)
end

@testset "SearchMatchingStage — labor axis must be 2 levels" begin
    bad = GriddedLayout(StateAxis(:wealth, continuous_grid([0.0, 1.0])),
                      StateAxis(:emp, categorical([:a, :b, :c])))
    @test_throws AssertionError _sam_stage(bad)
end

@testset "SearchMatchingStage — env slice declares θ (FromEnv)" begin
    stage = _sam_stage(_sam_layout())
    @test :θ in effective_env_slice(stage)
end

@testset "SearchMatchingStage — max-collapse + separation correctness" begin
    layout  = _sam_layout()
    efforts = collect(range(0.0, 2.0; length = 6))
    δ       = 0.10
    θ       = 1.0
    stage   = _sam_stage(layout; efforts = efforts, δ = δ)
    n_w     = axissize(layout.axes[1])

    # An asymmetric continuation: employment worth more, value rising in wealth.
    V_out = zeros(n_w, 2)
    V_out[:, 1] .= 0.1 .* (1:n_w)              # unemployed continuation
    V_out[:, 2] .= 0.1 .* (1:n_w) .+ 1.0       # employed continuation (better)

    V_start = copy(backward!(stage, V_out, (θ = θ,)))

    Vu = V_out[:, 1]; Ve = V_out[:, 2]
    # Unemployed value: hard max over effort of −cost(e) + p·Ve + (1−p)·Vu.
    for w in 1:n_w
        expected = maximum(-_cost(e) + _p(e, θ) * Ve[w] + (1 - _p(e, θ)) * Vu[w]
                           for e in efforts)
        @test V_start[w, 1] ≈ expected atol = 1e-12
    end
    # Employed value: separation mix (no choice).
    for w in 1:n_w
        @test V_start[w, 2] ≈ (1 - δ) * Ve[w] + δ * Vu[w] atol = 1e-12
    end

    # The stored effort policy is the argmax effort index per wealth cell.
    policy = stage.kernel.policy
    for w in 1:n_w
        Qs = [-_cost(e) + _p(e, θ) * Ve[w] + (1 - _p(e, θ)) * Vu[w] for e in efforts]
        @test policy[w] == argmax(Qs)
    end
end

@testset "SearchMatchingStage — p(θ) comparative static (value + flows)" begin
    layout = _sam_layout()
    stage  = _sam_stage(layout)
    n_w    = axissize(layout.axes[1])
    V_out  = zeros(n_w, 2); V_out[:, 2] .= 1.0      # employment strictly better

    Vu(θ) = copy(backward!(stage, V_out, (θ = θ,)))[:, 1]
    # Tighter market ⇒ cheaper to find a job ⇒ unemployed value (weakly) higher.
    Vu_loose = Vu(0.3)
    Vu_tight = Vu(3.0)
    @test all(Vu_tight .>= Vu_loose .- 1e-12)
    @test any(Vu_tight .> Vu_loose .+ 1e-9)          # strictly higher somewhere

    # Forward: start with all mass unemployed; a tighter market sends more to work.
    Λ0 = zeros(n_w, 2); Λ0[:, 1] .= 1.0 / n_w
    emp_mass(θ) = begin
        backward!(stage, V_out, (θ = θ,))
        sum(forward!(stage, copy(Λ0))[:, 2])
    end
    @test emp_mass(3.0) > emp_mass(0.3)
end

@testset "SearchMatchingStage — mass conservation" begin
    layout = _sam_layout()
    stage  = _sam_stage(layout)
    n_w    = axissize(layout.axes[1])
    V_out  = randn(n_w, 2)
    backward!(stage, V_out, (θ = 1.5,))

    Λ_start = abs.(randn(n_w, 2)); Λ_start ./= sum(Λ_start)
    Λ_end   = copy(forward!(stage, Λ_start))
    @test sum(Λ_end) ≈ sum(Λ_start) atol = 1e-12
    @test all(Λ_end .>= -1e-15)                       # a probability distribution
end

@testset "SearchMatchingStage — duality on the linear matching/separation operator" begin
    # The unemployed flow reward −cost(e*) is additive on the V side, so the clean
    # duality identity is on the LINEAR operator K (matching + separation, no cost)
    # and its transpose (the forward scatter). Build K from the solved policy/p and
    # the separation rate, and check ⟨K V_out, Λ_in⟩ = ⟨V_out, K Λ_in⟩ where the
    # second is exactly the stage's forward.
    layout = _sam_layout()
    δ      = 0.10
    θ      = 2.0
    stage  = _sam_stage(layout; δ = δ)
    n_w    = axissize(layout.axes[1])

    V_out   = randn(n_w, 2)
    Λ_start = abs.(randn(n_w, 2)); Λ_start ./= sum(Λ_start)
    backward!(stage, V_out, (θ = θ,))
    p = stage.kernel.p                          # p(e*(w), θ), per wealth

    # Linear K applied to V_out (value side), per wealth column:
    #   (K V)_unemp = (1−p)·Vu + p·Ve         (job-finding lottery at chosen effort)
    #   (K V)_emp   = δ·Vu + (1−δ)·Ve         (separation)
    Vu = V_out[:, 1]; Ve = V_out[:, 2]
    KV = similar(V_out)
    @. KV[:, 1] = (1 - p) * Vu + p * Ve
    @. KV[:, 2] = δ * Vu + (1 - δ) * Ve

    Λ_end = copy(forward!(stage, Λ_start))            # = Kᵀ Λ_start (mass pushforward)
    @test sum(KV .* Λ_start) ≈ sum(V_out .* Λ_end) atol = 1e-12
end

@testset "SearchMatchingStage — separation can be FromEnv" begin
    layout = _sam_layout()
    stage  = SearchMatchingStage(layout; labor_axis = :emp,
                                 efforts = collect(range(0.0, 2.0; length = 5)),
                                 cost = _cost, job_finding = _p,
                                 separation = FromEnv(:δ), tightness = FromEnv(:θ))
    @test :δ in effective_env_slice(stage)
    @test :θ in effective_env_slice(stage)
    n_w   = axissize(layout.axes[1])
    V_out = randn(n_w, 2)

    # Higher separation ⇒ employed value pulled toward the (lower) unemployed value.
    Ve_lo = copy(backward!(stage, V_out, (θ = 1.0, δ = 0.05)))[:, 2]
    Ve_hi = copy(backward!(stage, V_out, (θ = 1.0, δ = 0.40)))[:, 2]
    Vu = V_out[:, 1]; Ve = V_out[:, 2]
    for w in 1:n_w
        @test Ve_lo[w] ≈ 0.95 * Ve[w] + 0.05 * Vu[w] atol = 1e-12
        @test Ve_hi[w] ≈ 0.60 * Ve[w] + 0.40 * Vu[w] atol = 1e-12
    end

    # Forward replays the env's δ that backward last saw; mass still conserved.
    Λ_start = abs.(randn(n_w, 2)); Λ_start ./= sum(Λ_start)
    @test sum(forward!(stage, copy(Λ_start))) ≈ 1.0 atol = 1e-12
end
