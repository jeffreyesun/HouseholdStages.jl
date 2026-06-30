using Test
using HouseholdStages
using LinearAlgebra: I

# Phase-6 flexibility sweep (acceptance criterion #3): a transition matrix can
# vary along an arbitrary layout axis and on env, with backing storage sized ONLY
# over its dependency axes (never replicated over the full layout). Three faces:
#
#   (1) a StratifiedKernel varying along an axis (:age) — duality through a stage,
#       and the dep-only-storage assertion (the crux of criterion #3);
#   (2) an env-dependent transition (FromEnv migration cost) — re-materialises
#       when env changes, doesn't when it doesn't (the cache + the refresh-vs-static
#       switch on the transition closure's `env` kwarg);
#   (3) an effort/θ-indexed matching-style transition — a Bernoulli row whose
#       job-finding probability `p(θ)` rides an env scalar (previews SearchMatching
#       as a *pure transition* test, no choice-collapse here).
#
# Internals (the dense-kernel builder `dense_kernel` from test_kernel.jl — same module
# scope — and the Markov env/static predicates) are not exported; reach them through the
# module.
const HS = HouseholdStages
using .HouseholdStages: forward!, backward!

_pairing(a, b) = sum(a .* b)

# ---------------------------------------------------------------------------- #
# (1) Axis-varying transition: storage only over the dep axis, never the layout #
# ---------------------------------------------------------------------------- #

@testset "Axis-varying dense kernel — dep-only storage + duality" begin
    # A 5-wealth × 3-age layout. The transition acts on :wealth and VARIES along
    # :age; it does NOT depend on a third axis :z. The backing (the kernel's compact
    # parent) must be sized (n_wealth, n_wealth, n_age) — :z stays a SINGLETON, never
    # replicated.
    layout = GriddedLayout(
        :wealth => GriddedContinuous(collect(range(0.0, 1.0; length = 5))), # transitioned (5)
        :age => Discrete([1.0, 2.0, 3.0]),                      # dep axis (3)
        :z => Discrete([0.5, 1.5]),                           # NOT a dep (2)
    )
    n_w, n_age, n_z = 5, 3, 2

    # One column-stochastic 5×5 forward matrix per age (a genuine K).
    rand_col_stoch() = (M = rand(n_w, n_w); M ./ sum(M; dims = 1))
    Ks = stack([rand_col_stoch() for _ in 1:n_age])               # (5, 5, 3)
    d  = dense_kernel(Ks, layout, :wealth; dep = (:age,))

    # ---- the dep-only-storage assertion (criterion #3) ----
    P = parent(d.kernel)                                          # compact (n_out, n_in, age, z-singleton)
    @test size(P)[1:3] == (n_w, n_w, n_age)                       # over deps only…
    @test size(P, 4) == 1                                         # …:z is singleton, NOT replicated
    @test length(P) == n_w * n_w * n_age                          # 75, not ×n_z = 150

    # ---- duality through the transition over the FULL (5,3,2) layout ----
    V_out = randn(n_w, n_age, n_z)
    Λ_in  = rand(n_w, n_age, n_z); Λ_in ./= sum(Λ_in)
    V_in  = dbwd(similar(V_out), d, V_out)
    Λ_out = dfwd(similar(Λ_in), d, Λ_in)
    @test isapprox(_pairing(V_in, Λ_in), _pairing(V_out, Λ_out); atol = 1e-12)

    # Each (age, z) slice contracts with its own age-fiber (constant along z).
    for a in 1:n_age, zi in 1:n_z
        @test isapprox(Λ_out[:, a, zi], Ks[:, :, a] * Λ_in[:, a, zi]; atol = 1e-12)
    end
end

@testset "Axis-varying transition — dep-only storage on a 2-axis layout" begin
    # The same dep-only-storage transition on a (wealth, age) layout: the parent is
    # exactly (3, 3, 2), and the duality identity holds.
    layout = GriddedLayout(
        :wealth => GriddedContinuous([0.0, 1.0, 2.0]),
        :age => Discrete([1.0, 2.0]),
    )
    Ks = stack([[0.6 0.2 0.1; 0.3 0.5 0.3; 0.1 0.3 0.6],
                [0.8 0.1 0.0; 0.1 0.7 0.2; 0.1 0.2 0.8]])         # (3,3,2)
    d  = dense_kernel(Ks, layout, :wealth; dep = (:age,))

    @test size(parent(d.kernel)) == (3, 3, 2)                     # dep-only

    V_out = randn(3, 2); Λ_in = rand(3, 2); Λ_in ./= sum(Λ_in)
    V_in  = dbwd(similar(V_out), d, V_out)
    Λ_out = dfwd(similar(Λ_in), d, Λ_in)
    @test isapprox(_pairing(V_in, Λ_in), _pairing(V_out, Λ_out); atol = 1e-12)
end

# ---------------------------------------------------------------------------- #
# (2) Env-dependent transition: re-materialises on env change, caches otherwise #
# ---------------------------------------------------------------------------- #

@testset "Env-dependent transition (FromEnv migration cost) — re-materialisation" begin
    # A migration logit whose cost field C[origin,destination] is supplied via env
    # (FromEnv(:C)). Changing env re-runs backward! and re-materialises eψC =
    # exp(−C/ε) (hence the policy); re-running at the SAME (V_end, env) is
    # deterministic. The modern stage has no freshness cache, so we observe the
    # re-materialisation directly through the kernel's eψC.
    layout = GriddedLayout(
        :wealth => GriddedContinuous([0.0, 1.0, 2.0]),
        :location => Discrete([:home, :abroad]),
    )
    stage = MigrationStage(layout; axis = :location,
                           migration_cost = FromEnv(:C), ε = 1.0)
    @test :C in effective_env_slice(stage)            # env-dep is declared, not a side channel

    n_w = 3
    V_end = zeros(n_w, 2); V_end[:, 2] .= 0.4         # :abroad a touch more valuable

    C_cheap = [0.0 0.1; 0.1 0.0]                      # easy to move
    C_dear  = [0.0 3.0; 3.0 0.0]                      # costly to move

    V_cheap   = copy(backward!(stage, V_end, (C = C_cheap,)))
    eψC_cheap = copy(parent(stage.kernel.eψC))      # exp(−C_cheap/ε), seated by backward!

    # Forward at the same (V_end, env) pushes mass conservatively.
    Λ0 = fill(1.0 / (n_w * 2), n_w, 2)
    Λ_out = copy(forward!(stage, copy(Λ0)))
    @test isapprox(sum(Λ_out), 1.0; atol = 1e-12)

    # Different env (the cost field changed) ⇒ eψC re-materialises (env-dep cost).
    V_dear   = copy(backward!(stage, V_end, (C = C_dear,)))
    eψC_dear = copy(parent(stage.kernel.eψC))
    @test !(eψC_cheap ≈ eψC_dear)                     # env-dependent transition re-materialised
    @test !(V_cheap ≈ V_dear)                         # and the value genuinely changes

    # Re-running at the SAME dear env reproduces value AND eψC (deterministic refresh).
    V_dear2 = copy(backward!(stage, V_end, (C = C_dear,)))
    @test V_dear ≈ V_dear2
    @test parent(stage.kernel.eψC) ≈ eψC_dear
end

@testset "Env-dependent vs env-independent transition fibers" begin
    # An env-reading matrix closure yields different fibers on different env (the kernel
    # re-materialises each backward); a constant matrix gives the same fiber regardless of env.
    layout = GriddedLayout(:s => Discrete([1.0, 2.0]))

    # Env-dependent: a job-finding Bernoulli row whose probability rides env tightness.
    # The closure returns the ROW-stochastic T[from, to]; unemp(1)→emp(2) at rate p(θ).
    pfind(θ) = clamp(0.3 * θ, 0.0, 1.0)
    Tmat(; env) = (p = pfind(env.θ); [1 - p p; 0.1 0.9])                # T[from, to]
    env_stage = MarkovStage(layout; axis = :s, transition_matrix = Tmat)

    # Re-solving at different env genuinely changes the fiber (re-materialised).
    backward!(env_stage, zeros(2), (; θ = 1.0))
    K1 = copy(parent(env_stage.kernel))
    backward!(env_stage, zeros(2), (; θ = 2.0))
    K2 = copy(parent(env_stage.kernel))
    @test K1 != K2
    @test K2[2, 1] ≈ 2 * K1[2, 1]                                       # find-prob doubled

    # Env-independent: a constant matrix gives the same fiber (K = Tᵀ) on any env.
    static_stage = MarkovStage(layout; axis = :s, transition_matrix = [0.5 0.5; 0.5 0.5])
    backward!(static_stage, zeros(2), (; θ = 1.0)); Ks1 = copy(parent(static_stage.kernel))
    backward!(static_stage, zeros(2), (; θ = 9.0)); Ks2 = copy(parent(static_stage.kernel))
    @test Ks1 == Ks2 == [0.5 0.5; 0.5 0.5]
end

# ---------------------------------------------------------------------------- #
# (3) Effort/θ-indexed matching transition (pure transition; SaM preview)      #
# ---------------------------------------------------------------------------- #

@testset "θ-indexed Bernoulli matching transition — duality + θ moves it" begin
    # Labor-state axis :emp = {unemployed=1, employed=2}. The unemployed→employed
    # row is the job-finding Bernoulli [1−p, p]; the employed row is the (fixed)
    # separation Bernoulli [δ, 1−δ]. `p = p(θ)` rides the env scalar θ (tightness).
    # We assemble the 2×2 forward matrix K(θ), contract it as a StratifiedKernel,
    # and check (a) duality, (b) different θ ⇒ different forward mass — the SaM
    # "env-dependent stochastic transition" piece, as a PURE transition (no
    # effort choice-collapse here; that is the separate SearchMatching dispatch).
    layout = GriddedLayout(
        :wealth => GriddedContinuous([0.0, 1.0, 2.0, 3.0]),
        :emp => Discrete([:unemp, :emp]),
    )
    δ = 0.10                                          # separation rate (fixed)

    # The env-dep job-finding probability p(θ) ∈ (0,1), increasing in tightness θ.
    pof(θ) = 1 - exp(-θ)

    # Build the column-stochastic forward operator K(θ) over :emp (n_out,n_in)=(2,2).
    # forward!: Λ_out[j] = Σ_i K[j,i] Λ_in[i]; columns sum to 1.
    #   from unemp (i=1): stays unemp w.p. 1−p, becomes emp w.p. p
    #   from emp   (i=2): separates to unemp w.p. δ, stays emp w.p. 1−δ
    function K_of_θ(θ)
        p = pof(θ)
        return [1 - p   δ;            # row :unemp (j=1)
                p        1 - δ]        # row :emp   (j=2)
    end

    n_w = 4
    Λ_in  = rand(n_w, 2); Λ_in ./= sum(Λ_in)
    V_out = randn(n_w, 2)

    forward_mass(θ) = begin
        d = dense_kernel(K_of_θ(θ), layout, :emp)
        # duality at this θ
        V_in  = dbwd(similar(V_out), d, V_out)
        Λ_out = dfwd(similar(Λ_in), d, Λ_in)
        @test isapprox(_pairing(V_in, Λ_in), _pairing(V_out, Λ_out); atol = 1e-12)
        # mass conserved (each column stochastic)
        @test isapprox(sum(Λ_out), sum(Λ_in); atol = 1e-12)
        return Λ_out
    end

    Λ_loose = forward_mass(0.2)                       # slack market: low p
    Λ_tight = forward_mass(3.0)                       # tight market: high p
    # Higher θ ⇒ more job-finding ⇒ more mass in :emp (column 2), less in :unemp.
    @test sum(Λ_tight[:, 2]) > sum(Λ_loose[:, 2])
    @test sum(Λ_tight[:, 1]) < sum(Λ_loose[:, 1])
end
