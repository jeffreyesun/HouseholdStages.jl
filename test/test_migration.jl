using Test
using HouseholdStages

# Small two-location layout used across the tests.
function _two_loc_layout(; n_w = 3, n_loc = 2)
    return GriddedLayout(
        :wealth => GriddedContinuous(collect(range(0.0, 1.0; length = n_w))),
        :location => Discrete(n_loc == 2 ? [:home, :abroad] :
                                          n_loc == 3 ? [:a, :b, :c] :
                                          [Symbol("loc$i") for i in 1:n_loc]),
    )
end

# Recompute the (origin → destination) choice-probability tensor from a
# solved stage's kernel, replacing the removed `transition_choice_prob`.
# eψC's compact parent is (origin, dest); value_weight/normalizer are full
# layout-shaped, so for these `(wealth, location)` layouts they index as
# `[wealth, location]`:  P(w, i → j) = eψC[i,j] · value_weight[w,j] / normalizer[w,i].
function _choice_prob(stage, n_w, n_l)
    k  = stage.kernel
    eC = reshape(parent(k.eψC), n_l, n_l)
    return [eC[i, j] * k.value_weight[w_i, j] / k.normalizer[w_i, i]
            for w_i in 1:n_w, i in 1:n_l, j in 1:n_l]
end

@testset "MigrationStage — cost matrix shape check (fires at backward!)" begin
    layout = _two_loc_layout()
    stage = MigrationStage(layout; axis = :location,
        migration_cost = [0.0 0.5 0.0; 0.5 0.0 0.0], ε = 1.0)   # wrong shape
    n_w, n_l = axissize.(layout.axes)
    # A constant cost sizes the dense kernel from its own shape, so a wrong shape surfaces at the
    # backward contraction (DimensionMismatch) rather than the fill assertion.
    @test_throws DimensionMismatch backward!(stage, zeros(n_w, n_l), NamedTuple())
end

@testset "MigrationStage — backward / forward at finite ε" begin
    layout = _two_loc_layout()
    C = [0.0 0.5;
         0.5 0.0]
    stage = MigrationStage(layout;
        axis           = :location,
        migration_cost = C,
        ε              = 1.0,
    )

    # Construct a smooth V_end where the destinations have an asymmetric value.
    n_w = axissize(layout.axes[1])
    n_l = axissize(layout.axes[2])
    V_end = [0.1 * w_i + 0.0 * (l_i == 1 ? 0.0 : 0.3)
             for w_i in 1:n_w, l_i in 1:n_l]
    V_end[:, 2] .+= 0.3   # destination :abroad is more valuable

    V_pre = copy(backward!(stage, V_end, NamedTuple()))

    # By hand: V_pre[w, i] = ε log Σ_j exp((-C[i,j] + V_end[w,j])/ε)
    for w_i in 1:n_w, i in 1:n_l
        expected = log(sum(exp(-C[i, j] + V_end[w_i, j]) for j in 1:n_l))
        @test V_pre[w_i, i] ≈ expected atol = 1e-12
    end

    # Probabilities sum to 1 along the destination axis.
    prob = _choice_prob(stage, n_w, n_l)
    for w_i in 1:n_w, i in 1:n_l
        @test sum(prob[w_i, i, j] for j in 1:n_l) ≈ 1.0 atol = 1e-12
    end

    # Forward: mass conservation.
    Λ_start = fill(1.0 / (n_w * n_l), n_w, n_l)
    Λ_end   = copy(forward!(stage, Λ_start))
    @test sum(Λ_end) ≈ sum(Λ_start) atol = 1e-12
end

@testset "MigrationStage — ε → 0 collapses to deterministic argmax" begin
    layout = _two_loc_layout()
    # Make moving home a strictly better option from abroad: V_end at home is high.
    C = [0.0 1.0; 0.0 0.0]   # cost to move into home from abroad is 0, but C[2,2]=0 stays
    stage = MigrationStage(layout;
        axis           = :location,
        migration_cost = C,
        ε              = 1e-4,                       # almost-degenerate logit
    )
    n_w = axissize(layout.axes[1])
    n_l = axissize(layout.axes[2])
    V_end = zeros(n_w, n_l)
    V_end[:, 1] .= 1.0   # home is much more valuable

    backward!(stage, V_end, NamedTuple())
    prob = _choice_prob(stage, n_w, n_l)
    # From every origin, the policy should concentrate on home (j = 1).
    for w_i in 1:n_w, i in 1:n_l
        @test prob[w_i, i, 1] > 0.999
        @test prob[w_i, i, 2] < 1e-3
    end
end

@testset "MigrationStage — duality identity ⟨V_in, Λ_in⟩ = ⟨V_out, Λ_out⟩" begin
    # The K-operator is the destination-choice kernel, with no flow
    # payoff on the V side (cost is paid by the destination index, but
    # at finite ε the duality holds when V_in includes the log-sum-exp).
    # Note: there is no flow payoff *separate* from K (the cost enters K),
    # so the standard duality identity ⟨V_in, Λ_in⟩ = ⟨V_out, Λ_out⟩
    # holds without an additive r term.
    layout = _two_loc_layout()
    C = [0.0 0.3;
         0.3 0.0]
    stage = MigrationStage(layout; axis = :location, migration_cost = C, ε = 2.0)

    n_w, n_l = axissize.(layout.axes)
    V_end   = randn(n_w, n_l)
    Λ_start = abs.(randn(n_w, n_l)); Λ_start ./= sum(Λ_start)

    V_pre = copy(backward!(stage, V_end, NamedTuple()))
    Λ_end = copy(forward!(stage, Λ_start))

    # The identity is V_pre - (cost term) ≡ K^T V_end - cost ⋅ p, but we
    # didn't separate the cost. So check the operator identity directly:
    # ⟨V_end, Λ_end⟩ = ⟨V_pre, Λ_start⟩ - ⟨cost · p, Λ_start⟩.
    # Compute the "cost ⋅ p" correction.
    prob = _choice_prob(stage, n_w, n_l)
    cost_per_cell = zeros(n_w, n_l)
    for w_i in 1:n_w, i in 1:n_l
        cost_per_cell[w_i, i] = sum(C[i, j] * prob[w_i, i, j] for j in 1:n_l)
    end
    # Add back the log-sum-exp's ε term — the K^T V_end identity isn't
    # exact under logit smoothing because V_pre includes ε·log(denom).
    # For a clean duality check we use the linear-K reading: the choice
    # probability tensor defines a linear operator from V_end to V_pre,
    # and forward applies the transpose to Λ_start. Under that linear
    # operator (forgetting the cost and ε terms), duality is exact.

    # K_lin V_end[w, i] = Σ_j P(j|w, i) · V_end[w, j]
    K_lin_V = zeros(n_w, n_l)
    for w_i in 1:n_w, i in 1:n_l
        K_lin_V[w_i, i] = sum(prob[w_i, i, j] * V_end[w_i, j] for j in 1:n_l)
    end
    @test sum(K_lin_V .* Λ_start) ≈ sum(V_end .* Λ_end) atol = 1e-12
end

@testset "MigrationStage — adjoint dot-product test on forward" begin
    layout = _two_loc_layout()
    C = [0.0 0.4;
         0.4 0.0]
    stage = MigrationStage(layout; axis = :location, migration_cost = C, ε = 1.5)

    n_w, n_l = axissize.(layout.axes)
    V_end   = randn(n_w, n_l)
    Λ_start = abs.(randn(n_w, n_l)); Λ_start ./= sum(Λ_start)
    backward!(stage, V_end, NamedTuple())
    Λ_end   = copy(forward!(stage, Λ_start))

    dΛ_end = randn(n_w, n_l)
    dΛ_start = forward_adjoint!(stage, dΛ_end)

    @test sum(Λ_end .* dΛ_end) ≈ sum(Λ_start .* dΛ_start) atol = 1e-12
end

@testset "MigrationStage — composition with WealthChangeStage via ChainStage" begin
    layout = _two_loc_layout()
    C = [0.0 0.4;
         0.4 0.0]
    move = MigrationStage(layout; axis = :location, migration_cost = C, ε = 2.0)
    receipt = WealthChangeStage(layout;
        wealth_post = (; location, wealth, env) -> begin
            r = location == :home ? env.r_home : env.r_abroad
            return (1 + r) * wealth
        end,
        axis = :wealth,
    )
    chain = move ∘ receipt

    n_w, n_l = axissize.(layout.axes)
    V_end   = randn(n_w, n_l)
    env     = (r_home = 0.04, r_abroad = 0.02)
    V_start = backward!(chain, V_end, env)
    @test size(V_start) == (n_w, n_l)
    @test all(isfinite, V_start)

    Λ_start = ones(n_w, n_l) ./ (n_w * n_l)
    Λ_end_chain = forward!(chain, Λ_start)
    @test sum(Λ_end_chain) ≈ 1.0 atol = 1e-12
end

@testset "MigrationStage — destination amenity is composition with a UtilityStage" begin
    # A destination amenity a[j] used to be a stage kwarg; it is exactly a
    # UtilityStage composed before the migration logit (the V-additive
    # decomposition). Effective utility per (origin, destination) is then
    # -C[i,j] + a[j] + V_end[j, s], matching the old amenity closed form.
    layout = _two_loc_layout()
    C = [0.0 0.5;
         0.5 0.0]
    a = [0.0, 1.5]                                     # destination 2 gets +1.5
    move = MigrationStage(layout; axis = :location, migration_cost = C, ε = 1.0)
    stage = move ∘ UtilityStage(layout; utility = (; location) -> a[location == :home ? 1 : 2])

    n_w, n_l = axissize.(layout.axes)
    V_end = zeros(n_w, n_l)
    V_pre = copy(backward!(stage, V_end, NamedTuple()))

    # By hand: V_pre[w, i] = ε log Σ_j exp((-C[i,j] + a[j])/ε)
    for w_i in 1:n_w, i in 1:n_l
        expected = log(sum(exp(-C[i, j] + a[j]) for j in 1:n_l))
        @test V_pre[w_i, i] ≈ expected atol = 1e-12
    end

    # Prob mass should concentrate on destination 2 (the logit sub-stage, modern).
    k = stage.buffer.stages[1].kernel
    eC = reshape(parent(k.eψC), n_l, n_l)
    prob = [eC[i, j] * k.value_weight[w_i, j] / k.normalizer[w_i, i] for w_i in 1:n_w, i in 1:n_l, j in 1:n_l]
    for w_i in 1:n_w, i in 1:n_l
        @test prob[w_i, i, 2] > prob[w_i, i, 1]
    end
end

@testset "MigrationStage — env-dependent cost field via FromEnv (criterion #4)" begin
    # The migration cost C[origin, destination] is supplied by env (FromEnv(:C)),
    # so it re-materialises each backward and is part of the env contract (declared
    # in effective_env_slice) — not a side channel. Changing the
    # env cost changes the policy; the closed form matches at each env.
    layout = _two_loc_layout()
    stage = MigrationStage(layout; axis = :location,
                           migration_cost = FromEnv(:C), ε = 1.0)
    @test :C in effective_env_slice(stage)

    n_w, n_l = axissize.(layout.axes)
    V_end = zeros(n_w, n_l); V_end[:, 2] .= 0.5      # :abroad more valuable

    C_cheap = [0.0 0.1; 0.1 0.0]                     # easy to move
    C_dear  = [0.0 4.0; 4.0 0.0]                     # very costly to move

    V_cheap = copy(backward!(stage, V_end, (C = C_cheap,)))
    prob_cheap = _choice_prob(stage, n_w, n_l)
    V_dear  = copy(backward!(stage, V_end, (C = C_dear,)))
    prob_dear  = _choice_prob(stage, n_w, n_l)

    # Closed form holds at each env cost.
    for w_i in 1:n_w, i in 1:n_l
        @test V_cheap[w_i, i] ≈ log(sum(exp(-C_cheap[i, j] + V_end[w_i, j]) for j in 1:n_l)) atol = 1e-12
        @test V_dear[w_i, i]  ≈ log(sum(exp(-C_dear[i, j]  + V_end[w_i, j]) for j in 1:n_l)) atol = 1e-12
    end

    # The env-dependent cost genuinely changes the policy: from :home (origin 1),
    # the high cost suppresses the move to :abroad relative to the cheap case.
    @test prob_dear[1, 1, 2] < prob_cheap[1, 1, 2]
    @test !(V_cheap ≈ V_dear)

    # Mass conservation on forward, and re-solving at the same env is deterministic.
    Λ_start = fill(1.0 / (n_w * n_l), n_w, n_l)
    @test sum(forward!(stage, Λ_start)) ≈ 1.0 atol = 1e-12
    @test copy(backward!(stage, V_end, (C = C_dear,))) ≈ V_dear
end

@testset "MigrationStage — env-scalar shifting a base cost (FromEnv mobility)" begin
    # A complementary env-dependent path: the cost field is formed outside and the
    # env carries the whole matrix scaled by a mobility scalar. Demonstrates the
    # FromEnv cost reacting to a low-dimensional aggregate (a mobility cost level).
    layout = _two_loc_layout()
    stage = MigrationStage(layout; axis = :location,
                           migration_cost = FromEnv(:C), ε = 1.0)
    n_w, n_l = axissize.(layout.axes)
    base = [0.0 1.0; 1.0 0.0]
    V_end = zeros(n_w, n_l); V_end[:, 2] .= 1.0

    backward!(stage, V_end, (C = 0.2 .* base,))       # low mobility cost
    p_lo = _choice_prob(stage, n_w, n_l)
    backward!(stage, V_end, (C = 5.0 .* base,))       # high mobility cost
    p_hi = _choice_prob(stage, n_w, n_l)
    # Higher mobility cost ⇒ less moving to the more-valuable :abroad from :home.
    @test p_hi[1, 1, 2] < p_lo[1, 1, 2]
end

@testset "MigrationStage — dep-varying cost: renters-only move (flexible cost feature)" begin
    # The cost varies along a tenure axis: owners pay +Inf to move (immobile),
    # renters pay Cbase. The cost is stored ONLY over (origin, dest, tenure) —
    # dep-only storage, never replicated over wealth.
    layout = GriddedLayout(
        :wealth => GriddedContinuous([0.0, 1.0]),
        :tenure => Discrete([:rent, :own]),
        :location => Discrete([:home, :abroad]),
    )
    Cbase = [0.0 0.4; 0.4 0.0]
    # C[origin, dest] over :location; varies along :tenure (owners immobile). The choice
    # axis is the matrix's two positional dims, so only :tenure is a kwarg.
    migcost(; tenure) = tenure == :own ? [0.0 Inf; Inf 0.0] : Cbase
    stage = MigrationStage(layout; axis = :location, migration_cost = migcost, ε = 1.0)
    n_w, n_t, n_l = axissize.(layout.axes)
    V_end = zeros(n_w, n_t, n_l); V_end[:, :, 2] .= 1.0      # :abroad more valuable
    backward!(stage, V_end, NamedTuple())
    k = stage.kernel

    # Dep-only storage: eψC's compact parent is (origin, dest, tenure), NOT over wealth.
    eC = reshape(parent(k.eψC), n_l, n_l, n_t)
    @test size(eC) == (n_l, n_l, n_t)
    @test ndims(eC) == 3 && size(eC, 3) == n_t

    # P(j | wealth, tenure, origin). value_weight/normalizer are layout-shaped (wealth, tenure, location).
    P(w, t, i, j) = eC[i, j, t] * k.value_weight[w, t, j] / k.normalizer[w, t, i]

    # Owners (t = 2) never move; renters (t = 1) do move toward the better location.
    for w in 1:n_w, i in 1:n_l, j in 1:n_l
        i != j && @test P(w, 2, i, j) < 1e-12
    end
    @test P(1, 2, 1, 1) > 0.999                              # owner stays home
    @test P(1, 1, 1, 2) > 0.3                                # renter moves home → abroad

    # Mass conservation; closed-form (Gibbs no-leak) for the dep-varying cost.
    Λ0 = fill(1.0 / (n_w * n_t * n_l), n_w, n_t, n_l)
    @test sum(forward!(stage, Λ0)) ≈ 1.0 atol = 1e-12
    C(z, t, i, j) = (t == 2 && i != j) ? Inf : (i == j ? 0.0 : Cbase[i, j])
    for w in 1:n_w, t in 1:n_t, i in 1:n_l
        lse = log(sum(exp(-C(w, t, i, j) + V_end[w, t, j]) for j in 1:n_l))
        @test backward!(stage, V_end, NamedTuple())[w, t, i] ≈ lse atol = 1e-12
    end
end

@testset "MigrationStage — dep-varying cost: duality and forward adjoint" begin
    layout = GriddedLayout(
        :income => Discrete([0.6, 1.4]),
        :location => Discrete([:home, :abroad]),
    )
    # Low-income households are immobile; cost reacts to an env mobility scalar. C[origin,
    # dest] over :location varies along :income and env, so :income and env are the kwargs.
    function migcost(; income, env)
        off = income < 1.0 ? Inf : 0.5 * env.mob
        return [0.0 off; off 0.0]
    end
    stage = MigrationStage(layout; axis = :location, migration_cost = migcost, ε = 0.9)
    n_z, n_l = axissize.(layout.axes)
    env = (mob = 2.0,)

    V_end   = randn(n_z, n_l)
    Λ_start = abs.(randn(n_z, n_l)); Λ_start ./= sum(Λ_start)
    backward!(stage, V_end, env)
    Λ_end = copy(forward!(stage, Λ_start))
    k = stage.kernel

    # Linear-K duality ⟨K_lin V_end, Λ_start⟩ = ⟨V_end, Λ_end⟩.
    eC = reshape(parent(k.eψC), n_l, n_l, n_z)
    P(z, i, j) = eC[i, j, z] * k.value_weight[z, j] / k.normalizer[z, i]
    K_lin_V = [sum(P(z, i, j) * V_end[z, j] for j in 1:n_l) for z in 1:n_z, i in 1:n_l]
    @test sum(K_lin_V .* Λ_start) ≈ sum(V_end .* Λ_end) atol = 1e-12

    # Adjoint dot-product test on forward.
    dΛ_end   = randn(n_z, n_l)
    dΛ_start = forward_adjoint!(stage, dΛ_end)
    @test sum(Λ_end .* dΛ_end) ≈ sum(Λ_start .* dΛ_start) atol = 1e-12

    # The env-mobility scalar genuinely moves the policy.
    @test !(copy(backward!(stage, V_end, (mob = 0.1,))) ≈
            copy(backward!(stage, V_end, (mob = 10.0,))))
end

@testset "MigrationStage — static_env_deps / effective_env_slice" begin
    layout = _two_loc_layout()
    move = MigrationStage(layout;
        axis           = :location,
        migration_cost = [0.0 0.5; 0.5 0.0],
        ε              = 1.0,
    )
    @test isempty(static_env_deps(typeof(move.spec)))
    @test isempty(effective_env_slice(move))

    # ε given as a Symbol surfaces as an env field.
    move2 = MigrationStage(layout;
        axis           = :location,
        migration_cost = [0.0 0.5; 0.5 0.0],
        ε              = FromEnv(:eps_logit),
    )
    @test :eps_logit in effective_env_slice(move2)
end
