using Test
using HouseholdStages
using ForwardDiff
using ForwardDiff: Dual, Tag

@testset "lift_jacobian(ProductStage) — the lift threads the factors' interiors" begin
    # The `⊕` lift goes through `with_eltype(::ProductStage)`. A CHAIN factor with a regridding
    # interior is what distinguishes threading its interiors from re-bundling it from the spec: the
    # spec-only route would force every interior boundary to the product's start layout and the
    # rectangular factor would not fit.
    l2 = GriddedLayout(:z => Discrete([0.5, 1.5]),           :group => Discrete([1]))
    l3 = GriddedLayout(:z => Discrete([0.5, 1.0, 1.5]),      :group => Discrete([1]))
    up   = MarkovStage(l2, l3; axis = :z, transition_matrix = [0.5 0.3 0.2; 0.1 0.6 0.3])
    down = MarkovStage(l3, l2; axis = :z, transition_matrix = [0.7 0.3; 0.4 0.6; 0.2 0.8])
    flat = MarkovStage(l2; axis = :z, transition_matrix = [0.7 0.3; 0.4 0.6])
    ps   = product(up ∘ down, flat; axis = :group)

    ps_d = lift_jacobian(ps; n_dual = 1)
    D    = eltype(V_start_buffer(ps_d))
    @test D <: Dual
    @test eltype(V_start_buffer(ps_d.buffer.components[1])) === D
    @test layout_size(boundaries(ps_d.buffer.components[1])[2]) == layout_size(l3)

    V_end   = fill(D(1.0, ForwardDiff.Partials((0.25,))), 2, 2)
    V_start = backward!(ps_d, V_end, NamedTuple())
    @test ForwardDiff.value.(V_start) ≈ backward!(ps, ones(2, 2), NamedTuple())
    @test all(ForwardDiff.partials.(V_start, 1) .≈ 0.25)     # row-stochastic ⇒ a uniform shift rides through
end

@testset "adjoints(ProductStage) — ⊕ is block-diagonal, duality holds, shapes follow the far layout" begin
    layout = GriddedLayout(:z => Discrete([0.5, 1.5]), :group => Discrete([1]))
    P1 = [0.7 0.3; 0.4 0.6]
    P2 = [0.5 0.5; 0.2 0.8]
    f1 = MarkovStage(layout; axis = :z, transition_matrix = P1)
    f2 = MarkovStage(layout; axis = :z, transition_matrix = P2) ∘
         MarkovStage(layout; axis = :z, transition_matrix = P1)     # a chain factor: the adjoint recurses
    ps = product(f1, f2; axis = :group)

    V_end = randn(2, 2)
    backward!(ps, V_end, NamedTuple())                      # seat every component kernel (§8: no cold adjoint)
    backward!(f1, V_end[:, 1:1], NamedTuple())
    backward!(f2, V_end[:, 2:2], NamedTuple())

    x = randn(2, 2)                                         # start-shaped
    y = randn(2, 2)                                         # end-shaped
    dV_end   = backward_adjoint!(ps, x)
    dΛ_start = forward_adjoint!(ps, y)
    @test sum(dV_end .* y) ≈ sum(x .* dΛ_start) atol = 1e-12

    # Block-diagonal: each slice is the factor's own adjoint, with no cross-block term.
    @test dV_end[:, 1:1]   ≈ backward_adjoint!(f1, x[:, 1:1]) atol = 1e-12
    @test dV_end[:, 2:2]   ≈ backward_adjoint!(f2, x[:, 2:2]) atol = 1e-12
    @test dΛ_start[:, 1:1] ≈ forward_adjoint!(f1, y[:, 1:1])  atol = 1e-12
    @test dΛ_start[:, 2:2] ≈ forward_adjoint!(f2, y[:, 2:2])  atol = 1e-12

    # The adjoints return a cotangent; they do not disturb the fused tensors holding the seated
    # primal state. Re-seat first, so the assertions read state no earlier adjoint call could have
    # written.
    V_start = copy(backward!(ps, V_end, NamedTuple()))
    Λ_end   = copy(forward!(ps, fill(0.25, 2, 2)))
    backward_adjoint!(ps, x); forward_adjoint!(ps, y)
    @test V_start == V_start_buffer(ps)
    @test Λ_end   == Λ_end_buffer(ps)

    # The driver's Step 2 on a ⊕ block, against the dense propagator: `ℰ_t = (Kᵀ)^{t−1} ℰ₀` with
    # `Kᵀ` block-diagonal — `P1·` on f1's slice, `P2·P1·` on the chain factor's (its components'
    # `Kᵀ` in reverse order). A blend of the blocks, or a K/Kᵀ swap, moves these numbers.
    Kᵀ(e) = hcat(P1 * e[:, 1], P2 * (P1 * e[:, 2]))
    ℰ     = expectation_vectors(ps, cell -> cell.z, 3)
    @test length(ℰ) == 3
    @test ℰ[1] == [0.5 0.5; 1.5 1.5]
    @test ℰ[2] ≈ Kᵀ(ℰ[1])     atol = 1e-12
    @test ℰ[3] ≈ Kᵀ(Kᵀ(ℰ[1])) atol = 1e-12

    # A RECTANGULAR product. Both adjoints allocate their output at the FAR layout, so a 2 → 3
    # factor is sized right in each direction — the property a square fixture cannot see.
    l2 = GriddedLayout(:z => Discrete([0.5, 1.5]),      :group => Discrete([1]))
    l3 = GriddedLayout(:z => Discrete([0.5, 1.0, 1.5]), :group => Discrete([1]))
    R1 = [0.5 0.3 0.2; 0.1 0.6 0.3]
    R2 = [0.2 0.5 0.3; 0.4 0.4 0.2]
    rp = product(MarkovStage(l2, l3; axis = :z, transition_matrix = R1),
                 MarkovStage(l2, l3; axis = :z, transition_matrix = R2); axis = :group)
    backward!(rp, randn(3, 2), NamedTuple())            # seat every component kernel (§8)

    xr = randn(2, 2)                                    # start-shaped
    yr = randn(3, 2)                                    # end-shaped
    dV_end_r   = backward_adjoint!(rp, xr)              # K·  : start → end
    dΛ_start_r = forward_adjoint!(rp, yr)               # Kᵀ· : end → start
    @test size(dV_end_r)   == layout_size(end_layout(rp))
    @test size(dΛ_start_r) == layout_size(start_layout(rp))
    @test sum(dV_end_r .* yr) ≈ sum(xr .* dΛ_start_r)        atol = 1e-12
    @test dV_end_r   ≈ hcat(R1' * xr[:, 1], R2' * xr[:, 2])  atol = 1e-12
    @test dΛ_start_r ≈ hcat(R1  * yr[:, 1], R2  * yr[:, 2])  atol = 1e-12
end

@testset "expectation_vectors(chain over ⊕) — the shape a driver presents" begin
    # Moments attach at chain end, so the drivers are `ChainStage`-keyed and a `⊕` block reaches
    # Step 2 only from inside a chain. `replicate_age` is the OLG member of that class — `N`
    # independent rebuilds of one factor, joined by `⊕`. The moment reads the product axis, so the
    # blocks carry distinct data and a blend across them would move the numbers.
    layout = GriddedLayout(:z => Discrete([0.5, 1.5]), :group => Discrete([1]))
    P      = [0.7 0.3; 0.4 0.6]
    ages   = replicate_age(MarkovStage(layout; axis = :z, transition_matrix = P), 3; axis = :group)
    chain  = IdentityStage(start_layout(ages)) ∘ ages

    backward!(chain, randn(2, 3), NamedTuple())         # seat every component kernel (§8)
    x = randn(2, 3); y = randn(2, 3)
    @test sum(backward_adjoint!(chain, x) .* y) ≈ sum(x .* forward_adjoint!(chain, y)) atol = 1e-12

    ℰ = expectation_vectors(chain, cell -> cell.z * cell.group, 3)
    @test length(ℰ) == 3
    @test ℰ[1] == [0.5 1.0 1.5; 1.5 3.0 4.5]
    @test ℰ[2] ≈ P * ℰ[1]       atol = 1e-12
    @test ℰ[3] ≈ P * (P * ℰ[1]) atol = 1e-12
end

@testset "adjoints(MixingKernel) — the mixture is a K/Kᵀ pair" begin
    block = GriddedLayout(:x => Discrete([1.0, 2.0, 3.0]))
    KA = [0.9 0.1 0.0; 0.1 0.8 0.1; 0.0 0.1 0.9]
    KB = [0.4 0.4 0.2; 0.3 0.4 0.3; 0.2 0.4 0.4]
    mk() = MixingStage(block; axis = :x, K_A = KA, K_B = KB, cost_curvature = 2.0)
    mix  = mk()
    # `V_end` is chosen so θ* lands INTERIOR at two of the three cells (0.44, 0, 0.16): a corner
    # policy routes every cell through one kernel outright and would leave the blend untested.
    backward!(mix, [1.0, 0.4, 0.8], NamedTuple())           # seat θ* (§8: no cold adjoint)
    θ = copy(policy(mix))
    @test count(t -> 0 < t < 1, θ) == 2

    x = randn(3); y = randn(3)
    dV_end   = backward_adjoint!(mix, x)
    dΛ_start = forward_adjoint!(mix, y)
    @test sum(dV_end .* y) ≈ sum(x .* dΛ_start) atol = 1e-12

    # Against the closed form: K_θ = KAᵀ·D_θ + KBᵀ·D_{1−θ} (the stored kernels are the user's Tᵀ).
    @test dV_end   ≈ KA' * (θ .* x) + KB' * ((1 .- θ) .* x) atol = 1e-12
    @test dΛ_start ≈ θ .* (KA * y) + (1 .- θ) .* (KB * y)   atol = 1e-12
    @test dV_end   ≈ forward!(mix, x) atol = 1e-12          # backward_adjoint! IS the primal push

    # Step 2 through the mixture: `ℰ_t = (K_θᵀ)^{t−1} ℰ₀` against the closed form, so a wrong `Kᵀ`
    # moves the numbers rather than merely staying finite.
    Kᵀ_mix(e) = θ .* (KA * e) + (1 .- θ) .* (KB * e)
    ℰ_mix    = expectation_vectors(mix, cell -> cell.x, 3)
    @test ℰ_mix[1] == [1.0, 2.0, 3.0]
    @test ℰ_mix[2] ≈ Kᵀ_mix(ℰ_mix[1])            atol = 1e-12
    @test ℰ_mix[3] ≈ Kᵀ_mix(Kᵀ_mix(ℰ_mix[1]))    atol = 1e-12

    # SearchMatchingStage is MarkovStage ∘ MixingStage, so it rides the chain adjoint.
    lay = GriddedLayout(:x => Discrete([0.5, 1.0, 2.0]), :emp => Discrete([:unemp, :emp]))
    sm  = SearchMatchingStage(lay; separation = 0.1, effort_cost_scale = 0.5,
                              matching_efficiency = 0.5, tightness = 1.5)
    backward!(sm, [1.0 4.0; 0.5 3.0; 2.0 6.0], NamedTuple())
    xs = randn(3, 2); ys = randn(3, 2)
    @test sum(backward_adjoint!(sm, xs) .* ys) ≈ sum(xs .* forward_adjoint!(sm, ys)) atol = 1e-12
    ℰ = expectation_vectors(sm, cell -> cell.x, 3)
    @test length(ℰ) == 3
    @test all(e -> all(isfinite, e), ℰ)
end

@testset "with_eltype — buffer eltype changes; static fields shared" begin
    layout = GriddedLayout(:z => Discrete([0.5, 1.5]))
    P = [0.7 0.3; 0.3 0.7]
    s = MarkovStage(layout; axis = :z, transition_matrix = P)
    D = ForwardDiff.Dual{Nothing, Float64, 1}
    s_d = with_eltype(s, D)
    @test eltype(s_d.scratch.V_start) === D
    @test eltype(s_d.scratch.Λ_end)   === D
    @test s_d.spec.transition_matrix === P     # static field shared
    @test start_layout(s_d)   === start_layout(s)
    @test s_d.spec.axis       === s.spec.axis
end

@testset "lift_jacobian(MarkovStage) — Dual flows through backward correctly" begin
    # MarkovStage is V_θ-independent, so the Jacobian of V_in wrt V_end is exactly P, transposed in
    # the Markov-rows-are-conditioning convention.
    layout = GriddedLayout(:z => Discrete([0.5, 1.5]))
    P = [0.7 0.3; 0.3 0.7]
    s = MarkovStage(layout; axis = :z, transition_matrix = P)
    s_d = lift_jacobian(s; n_dual = 1)

    # V_end carries a tangent ξ = e_1 = (1, 0) along the z axis.
    D = eltype(s_d.scratch.V_start)
    V_end = D[
        D(0.0, ForwardDiff.Partials((1.0,))),
        D(0.0, ForwardDiff.Partials((0.0,))),
    ]
    V_start = backward!(s_d, V_end, NamedTuple())
    # Rows are the conditioning state, so V_start[i] = Σ_j P[i,j]·V_end[j] and the seeded partial is
    # dV_start/dV_end[1] = column 1 of P = [P[1,1], P[2,1]] = [0.7, 0.3].
    @test ForwardDiff.partials(V_start[1])[1] ≈ P[1, 1] atol = 1e-12
    @test ForwardDiff.partials(V_start[2])[1] ≈ P[2, 1] atol = 1e-12
    @test ForwardDiff.value(V_start[1]) ≈ 0.0 atol = 1e-12
    @test ForwardDiff.value(V_start[2]) ≈ 0.0 atol = 1e-12
end

@testset "lift_jacobian(3-stage chain) — ∂K_supplied/∂r matches finite diffs" begin
    # Forward-mode AD through the canonical L03/L04 decomposition `IncomeShock ∘ IncomeReceipt ∘
    # ConsumptionSavingsStage`, on `K_supplied(r, w) = Σ Λ_end · wealth` at a fixed `V_terminal` and
    # `Λ_init`.

    layout = GriddedLayout(
        :wealth => GriddedContinuous([0.0, 1.0, 2.0, 3.0]),
        :income => Discrete([0.5, 1.5]),
    )
    P = [0.7 0.3; 0.3 0.7]
    shock   = MarkovStage(layout; axis = :income, transition_matrix = P)
    receipt = WealthChangeStage(layout;
        wealth_post  = (; wealth, income, env) -> (1 + env.r) * wealth + env.w * income,
        axis         = :wealth,
    )
    saves = ConsumptionSavingsStage(layout;
        β            = 0.96,
        utility      = (cell, c; env) -> log(c),
        axis         = :wealth,
    )
    chain = shock ∘ receipt ∘ saves

    function K_supplied(chain_use, r::T, w::T) where {T<:Real}
        # Smooth, non-degenerate V_terminal to avoid argmax kinks near
        # the tangent direction.
        dims = layout_size(layout)
        V_term = T.([0.1 * w_i + 0.05 * y_j for w_i in 1:dims[1], y_j in 1:dims[2]])
        Λ_init = ones(T, dims...) ./ prod(dims)
        backward!(chain_use, V_term, (r = r, w = w))
        Λ_end = forward!(chain_use, Λ_init)
        wealth_grid = T.([0.0, 1.0, 2.0, 3.0])
        s = zero(T)
        for w_i in 1:dims[1], y_j in 1:dims[2]
            s += Λ_end[w_i, y_j] * wealth_grid[w_i]
        end
        return s
    end

    r0, w0 = 0.04, 1.2
    # AD path.
    chain_dual = lift_jacobian(chain; n_dual = 1)
    D = eltype(V_start_buffer(chain_dual.buffer.stages[1]))
    r_dual = D(r0, ForwardDiff.Partials((1.0,)))
    w_dual = D(w0, ForwardDiff.Partials((0.0,)))
    K_dual = K_supplied(chain_dual, r_dual, w_dual)
    dKdr_ad = ForwardDiff.partials(K_dual)[1]
    K_primal_ad = ForwardDiff.value(K_dual)

    # Finite-difference path on the Float64 chain.
    h = 1e-6
    K_plus  = K_supplied(chain, r0 + h, w0)
    K_minus = K_supplied(chain, r0 - h, w0)
    K_primal_fd = K_supplied(chain, r0, w0)
    dKdr_fd = (K_plus - K_minus) / (2h)

    @test K_primal_ad ≈ K_primal_fd atol = 1e-10
    @test dKdr_ad ≈ dKdr_fd rtol = 1e-4
end

@testset "backward_adjoint!(MarkovStage) — dot-product test" begin
    # Verify the operator adjointness identity:
    #     ⟨backward!(V_end), dV_start⟩ = ⟨V_end, backward_adjoint!(dV_start)⟩
    # (without the flow payoff r, since MarkovStage has r = 0).
    layout = GriddedLayout(:z => Discrete([0.5, 1.5]))
    P = [0.6 0.4; 0.25 0.75]
    s = MarkovStage(layout; axis = :z, transition_matrix = P)

    V_end    = randn(2)
    dV_start = randn(2)
    V_start  = copy(backward!(s, V_end, NamedTuple()))
    dV_end   = backward_adjoint!(s, dV_start)

    @test sum(V_start .* dV_start) ≈ sum(V_end .* dV_end) atol = 1e-12
end

@testset "forward_adjoint!(MarkovStage) — dot-product test" begin
    layout = GriddedLayout(:z => Discrete([0.5, 1.5]))
    P = [0.6 0.4; 0.25 0.75]
    s = MarkovStage(layout; axis = :z, transition_matrix = P)
    backward!(s, randn(2), NamedTuple())   # seat K = Tᵀ before applying the forward operator (§9 contract)

    Λ_start = abs.(randn(2)); Λ_start ./= sum(Λ_start)
    dΛ_end  = randn(2)
    Λ_end   = copy(forward!(s, Λ_start))
    dΛ_start = forward_adjoint!(s, dΛ_end)

    @test sum(Λ_end .* dΛ_end) ≈ sum(Λ_start .* dΛ_start) atol = 1e-12
end

@testset "adjoints(IdentityStage) — pass-through" begin
    layout = GriddedLayout(:z => Discrete([0.5, 1.5]))
    s = IdentityStage(layout)

    dV = randn(2)
    dΛ = randn(2)
    @test backward_adjoint!(s, dV) == dV
    @test forward_adjoint!(s, dΛ) == dΛ
end

@testset "adjoints(ForgetfulSumStage) — dot-product tests" begin
    layout = GriddedLayout(
        :w => GriddedContinuous([1.0, 2.0, 3.0]),
        :t => Discrete([:a, :b]),
    )
    s = ForgetfulSumStage(layout; axis = :t)
    backward!(s, randn(3, 1), NamedTuple())   # seat the transition before applying forward! (§9 contract)

    # Forward: Λ_start (3,2) → Λ_end (3,1) (:t resized to size 1, not dropped).
    Λ_start = abs.(randn(3, 2)); Λ_start ./= sum(Λ_start)
    dΛ_end  = randn(3, 1)
    Λ_end   = copy(forward!(s, Λ_start))
    dΛ_start = forward_adjoint!(s, dΛ_end)
    @test sum(Λ_end .* dΛ_end) ≈ sum(Λ_start .* dΛ_start) atol = 1e-12

    # Backward: V_end (3,1) → V_start (3,2).
    V_end    = randn(3, 1)
    dV_start = randn(3, 2)
    V_start  = copy(backward!(s, V_end, NamedTuple()))
    dV_end   = backward_adjoint!(s, dV_start)
    @test sum(V_start .* dV_start) ≈ sum(V_end .* dV_end) atol = 1e-12
end

@testset "adjoints(ArgmaxStage) — dot-product test" begin
    layout = GriddedLayout(:s => Discrete([:A, :B]))
    stage = ArgmaxStage(layout; axis = :s, reward = [0.0 0.0; 1.0 1.0])
    V_end = randn(2)
    Λ_start = abs.(randn(2)); Λ_start ./= sum(Λ_start)

    V_start = copy(backward!(stage, V_end, NamedTuple()))
    Λ_end   = copy(forward!(stage, Λ_start))

    # Sensitivities at the input and output of each pass.
    dV_in    = randn(2)
    dΛ_end   = randn(2)
    dV_out   = backward_adjoint!(stage, dV_in)
    dΛ_start = forward_adjoint!(stage, dΛ_end)

    # ArgmaxStage flow_payoff at the chosen action is non-zero. The duality
    # check for the BACKWARD pass needs to account for that; we check
    # the *linear* part of the adjoint via the FORWARD pass identity
    # ⟨Λ_end, dΛ_end⟩ = ⟨Λ_start, dΛ_start⟩ (no payoff term).
    @test sum(Λ_end .* dΛ_end) ≈ sum(Λ_start .* dΛ_start) atol = 1e-12

    # For backward, ⟨V_in - r, dV_in⟩ = ⟨V_end, dV_out⟩ where r is the
    # flow-payoff contribution. Compute V_in - r explicitly:
    # at every cell, V_in = max(r + V_out[ν]); the chosen action's
    # contribution is r(action*) + V_out[ν(s,action*)]. Subtracting r:
    actions = [:A, :B]
    V_in_minus_r = similar(V_start)
    for (idx, cell) in cells(layout)
        ci  = CartesianIndex(Tuple(idx))
        a_i = policy(stage)[ci]
        # V_start[ci] = r(action*) + V_end[ν(s,action*)]
        r_val = (actions[a_i] == :B ? 1.0 : 0.0)
        V_in_minus_r[ci] = V_start[ci] - r_val
    end
    @test sum(V_in_minus_r .* dV_in) ≈ sum(V_end .* dV_out) atol = 1e-12
end

@testset "adjoints(LogitChoiceStage) — dot-product test on forward" begin
    layout = GriddedLayout(:a => Discrete([1, 2]))
    stage = LogitChoiceStage(layout;
        axis        = :a,
        cost_matrix = [0.0 0.5; 0.5 0.0],
        ε           = 0.5,
    )
    V_end = randn(2)
    Λ_start = abs.(randn(2)); Λ_start ./= sum(Λ_start)

    V_start = copy(backward!(stage, V_end, NamedTuple()))
    Λ_end   = copy(forward!(stage, Λ_start))

    dΛ_end = randn(2)
    dΛ_start = forward_adjoint!(stage, dΛ_end)
    @test sum(Λ_end .* dΛ_end) ≈ sum(Λ_start .* dΛ_start) atol = 1e-12

    # Verify backward_adjoint! against the analytic envelope-theorem
    # derivative: ∂V_in[i]/∂V_end[j] = P(j | origin i). For dV_in = e_1
    # (only origin i=1 contributes), backward_adjoint! returns dV_end[j] =
    # P(j | i=1). This layout is the choice axis only, so the seated policy is
    # the `(origin, dest)` matrix and origin 1 is its first row.
    P_from1 = choice_probabilities(stage)[1, :]
    @test sum(P_from1) ≈ 1.0 atol = 1e-12

    e1 = Float64[1.0, 0.0]
    dV_out_via_adj = backward_adjoint!(stage, e1)
    @test dV_out_via_adj[1] ≈ P_from1[1] atol = 1e-12
    @test dV_out_via_adj[2] ≈ P_from1[2] atol = 1e-12
end

@testset "adjoints(LogitChoiceStage, FromEnv cost) — match static-matrix VJP" begin
    # The lift adjoints size `n` from the layout (`_logit_n`), not from `spec.cost_matrix`: an
    # env-supplied cost is a `FromEnv` marker, which has no `size`. The same stage built both ways —
    # static matrix and `FromEnv(:C)` — must give the identical VJP at the same evaluated point.
    layout = GriddedLayout(:loc => Discrete([1, 2, 3]))
    C      = [0.0 0.4 0.7; 0.4 0.0 0.3; 0.7 0.3 0.0]
    ε      = 0.5

    static = LogitChoiceStage(layout; axis = :loc, cost_matrix = C, ε = ε)
    envcost = MigrationStage(layout; axis = :loc,
                             migration_cost = FromEnv(:C), ε = ε)
    env = (C = C,)

    V_end   = randn(3)
    Λ_start = abs.(randn(3)); Λ_start ./= sum(Λ_start)

    # Evaluate both primals at the same point (env populates the FromEnv kernel).
    backward!(static,  V_end, NamedTuple())
    backward!(envcost, V_end, env)
    forward!(static,  Λ_start)
    forward!(envcost, Λ_start)

    dV_in  = randn(3)
    dΛ_end = randn(3)

    # (i) The FromEnv adjoints run at all — nothing asks the marker for a `size`.
    dV_out_env   = backward_adjoint!(envcost, dV_in)
    dΛ_start_env = forward_adjoint!(envcost, dΛ_end)

    # (ii) They must equal the static-matrix adjoints exactly (same kernel point).
    dV_out_static   = backward_adjoint!(static, dV_in)
    dΛ_start_static = forward_adjoint!(static, dΛ_end)
    @test dV_out_env   ≈ dV_out_static   atol = 1e-12
    @test dΛ_start_env ≈ dΛ_start_static atol = 1e-12

    # And the forward adjoint satisfies the duality identity on the env stage.
    Λ_end_env = copy(forward!(envcost, Λ_start))
    @test sum(Λ_end_env .* dΛ_end) ≈ sum(Λ_start .* dΛ_start_env) atol = 1e-12
end

@testset "adjoints(ConsumptionSavingsStage) — dot-product test on forward" begin
    layout = GriddedLayout(
        :wealth => GriddedContinuous([1.0, 2.0]),
        :y => Discrete([1.0]),
    )
    stage = ConsumptionSavingsStage(layout;
        β            = 0.95,
        utility      = (cell, c; env) -> log(c),
        axis         = :wealth,
    )
    backward!(stage, zeros(2, 1), NamedTuple())  # populate policy

    Λ_start = reshape([0.4, 0.6], (2, 1))
    Λ_end = copy(forward!(stage, Λ_start))

    dΛ_end = reshape([1.0, 0.5], (2, 1))
    dΛ_start = forward_adjoint!(stage, dΛ_end)
    @test sum(Λ_end .* dΛ_end) ≈ sum(Λ_start .* dΛ_start) atol = 1e-12
end

@testset "adjoints(ChainStage of linear-K stages) — dot-product test" begin
    layout = GriddedLayout(:z => Discrete([0.5, 1.5]))
    P1 = [0.7 0.3; 0.4 0.6]
    P2 = [0.5 0.5; 0.2 0.8]
    s1 = MarkovStage(layout; axis = :z, transition_matrix = P1)
    s2 = MarkovStage(layout; axis = :z, transition_matrix = P2)
    chain = s1 ∘ s2
    backward!(chain, randn(2), NamedTuple())   # seat both component kernels before forward! (§9 contract)

    Λ_start = abs.(randn(2)); Λ_start ./= sum(Λ_start)
    dΛ_end  = randn(2)
    Λ_end   = copy(forward!(chain, Λ_start))
    dΛ_start = forward_adjoint!(chain, dΛ_end)
    @test sum(Λ_end .* dΛ_end) ≈ sum(Λ_start .* dΛ_start) atol = 1e-12

    V_end    = randn(2)
    dV_start = randn(2)
    V_start  = copy(backward!(chain, V_end, NamedTuple()))
    dV_end   = backward_adjoint!(chain, dV_start)
    @test sum(V_start .* dV_start) ≈ sum(V_end .* dV_end) atol = 1e-12
end

@testset "lift_jacobian(ChainStage with moments) — works through define_moments!" begin
    # The lift propagates through `define_moments!`, whose `ChainStage` carries a non-empty
    # `moments` field.
    layout = GriddedLayout(
        :wealth => GriddedContinuous([1.0, 2.0, 3.0]),
        :income => Discrete([0.5, 1.5]),
    )
    P = [0.5 0.5; 0.5 0.5]
    shock = MarkovStage(layout; axis = :income, transition_matrix = P)
    mc = define_moments!(shock; K = at_end(integrand = :wealth, reduce = sum))
    mc_d = lift_jacobian(mc; n_dual = 1)

    @test mc_d isa ChainStage
    @test !isempty(mc_d.spec.moments)                          # moments preserved through with_eltype
    inner_stage = mc_d.spec.stages[1]
    @test eltype(inner_stage.transition_matrix) === Float64    # transition stays Float64
    inner_stage_d = mc_d.buffer.stages[1]
    @test eltype(V_start_buffer(inner_stage_d))    !== Float64          # buffer is Dual
end

@testset "Dual-eltype sweeps run and match the primal on a -Inf-masked chain" begin
    # WP1's sentinel/comparison contract, on the stock `-Inf`-masked consumption-savings chain: the
    # wealth-0 row is infeasible at every choice, so the sentinel reaches the node walk and the
    # `LotteryGatherOp` guard on every sweep. Feasibility is read at the primal, values stay live.
    #
    # A FIXED number of sweeps, not a solve. The fixed-point solvers refuse a Dual iterate, because
    # a loop whose trip count is chosen by a primal comparison is not a thing forward-mode AD can
    # differentiate. A fixed sweep count is a finite recursion and is exactly what AD handles.
    nw     = 40
    wgrid  = collect(range(0.0, 10.0; length = nw))
    layout = GriddedLayout(
        :wealth => GriddedContinuous(wgrid),
        :income => Discrete([0.5, 1.5]),
    )
    P = [0.7 0.3; 0.3 0.7]
    build() =
        MarkovStage(layout; axis = :income, transition_matrix = P) ∘
        WealthChangeStage(layout;
            wealth_post = (; wealth, income, env) -> (1 + env.r) * wealth + env.w * income,
            axis        = :wealth) ∘
        ConsumptionSavingsStage(layout; β = 0.96, utility = (cell, c; env) -> log(c), axis = :wealth)

    NSWEEP = 400                       # β = 0.96 ⇒ 0.96^400 ≈ 1e-7 of the initial gap
    sweep(ch, env, V0) = (V = V0; for _ in 1:NSWEEP; V = copy(backward!(ch, V, env)); end; V)

    r0, w0  = 0.04, 1.2
    chain_p = build()
    V_p     = sweep(chain_p, (r = r0, w = w0), zero(V_start_buffer(chain_p)))

    chain_d = lift_jacobian(build(); n_dual = 1)
    D       = eltype(V_start_buffer(chain_d.buffer.stages[1]))
    V_d     = sweep(chain_d, (r = D(r0, ForwardDiff.Partials((1.0,))),
                              w = D(w0, ForwardDiff.Partials((0.0,)))),
                    zero(V_start_buffer(chain_d)))

    @test ForwardDiff.value.(V_d) == V_p        # bitwise: the primal lane is untouched by the lift
    # The seated policy is the forward map. It is the node walk's output, and the walk's comparison
    # is where a live tangent on the infeasible row would otherwise select a different node — with
    # the value lane still agreeing to the bit, since an infeasible cell's value is `-Inf` either way.
    @test ForwardDiff.value.(policy(chain_d.buffer.stages[3])) == policy(chain_p.buffer.stages[3])

    # ∂V/∂r after the same fixed number of sweeps, against a central difference of the SAME finite
    # recursion — an object AD reproduces exactly, where the fixed point's derivative would not be.
    dVdr = [ForwardDiff.partials(v, 1) for v in V_d]
    @test all(isfinite, dVdr)
    h  = 1e-6
    fp = build(); fm = build()
    fd = (sweep(fp, (r = r0 + h, w = w0), zero(V_start_buffer(fp))) .-
          sweep(fm, (r = r0 - h, w = w0), zero(V_start_buffer(fm)))) ./ (2h)
    k  = argmax(abs.(dVdr))
    @test dVdr[k] ≈ fd[k] rtol = 1e-6
end

@testset "tangent_grade — a hard-argmax rebuild is REFUSED at the Dual eltype, not silently zeroed" begin
    g   = [0.0, 1.0, 2.0, 3.0]
    lay = GriddedLayout(:k => GriddedContinuous(g))
    rew = (; env) -> [ g[b] * (1 + env.r) - g[a] > 0 ? log(g[b] * (1 + env.r) - g[a]) : -Inf
                       for a in 1:4, b in 1:4 ]
    argm  = ArgmaxStage(lay; reward = rew, axis = :k)
    smooth = TimeDiscountingStage(lay; β = 0.96)
    move  = DiscreteMoveStage(lay; axis = :k, destination = (; k) -> 0.5k + 1)

    # The grade is the declaration, and it aggregates worst-over-components through `∘` and `⊕`.
    @test tangent_grade(argm) === :wrong_object
    @test tangent_grade(move) === :wrong_object
    @test tangent_grade(smooth) === :exact
    @test tangent_grade(ContinuousArgmaxStage(lay; reward = rew, axis = :k)) === :exact_ae
    @test tangent_grade(smooth ∘ argm) === :wrong_object
    @test tangent_grade(smooth ∘ ContinuousArgmaxStage(lay; reward = rew, axis = :k)) === :exact_ae
    @test tangent_grade(smooth ∘ IdentityStage(lay)) === :exact

    lay_g = GriddedLayout(:k => GriddedContinuous(g), :group => Discrete([1]))
    prod  = product(ArgmaxStage(lay_g; reward = rew, axis = :k),
                    TimeDiscountingStage(lay_g; β = 0.96); axis = :group)
    @test tangent_grade(prod) === :wrong_object

    # One guard at the allocating constructor covers every route to a Dual-eltype primitive: a
    # chain rebuild, a bare stage rebuild, a product component. Float64 rebuilds are untouched.
    @test_throws ErrorException lift_jacobian(smooth ∘ argm; n_dual = 1)
    @test_throws ErrorException with_eltype(move, ForwardDiff.Dual{Nothing, Float64, 1})
    @test_throws ErrorException lift_jacobian(prod; n_dual = 1)
    @test with_eltype(move, Float64) isa DiscreteMoveStage
    @test lift_jacobian(smooth ∘ IdentityStage(lay); n_dual = 1) isa ChainStage
end

@testset "Dual lottery gather — the -Inf mask is a primal fact" begin
    # `WealthChangeStage` gathers the constraint stage's `V_start`, whose masked cells are `-Inf` in
    # the value lane and carry a LIVE tangent: the mask is a Float64 `-Inf` added by a `UtilityStage`
    # to a live `V_end`, so it arrives as `Dual(-Inf, ṗ)`. Testing that against `-Inf` on the live
    # value misses it, and the `(V[hi] - V[lo])` slope then makes `Inf - Inf`.
    nw    = 12
    wgrid = collect(range(0.0, 5.0; length = nw))
    lay   = GriddedLayout(:wealth => GriddedContinuous(wgrid))
    build() =
        WealthChangeStage(lay;
            wealth_post = (; wealth, env) -> (1 + env.r) * wealth + 0.3,
            axis        = :wealth) ∘
        BorrowingConstraintStage(lay; infeasible = (; wealth) -> wealth < 1.0)

    V_end_p = [0.5 + 0.1 * i for i in 1:nw]
    V_p     = copy(backward!(build(), V_end_p, (r = 0.04,)))

    chain_d = lift_jacobian(build(); n_dual = 1)
    D       = eltype(V_start_buffer(chain_d.buffer.stages[1]))
    V_end_d = [D(v, ForwardDiff.Partials((1.0 + 0.2 * v,))) for v in V_end_p]
    V_d     = copy(backward!(chain_d, V_end_d, (r = D(0.04, ForwardDiff.Partials((1.0,))),)))

    @test any(==(-Inf), V_p)                                   # the mask does reach the interpolation
    @test ForwardDiff.value.(V_d) == V_p                       # bitwise on the primal lane
    @test !any(isnan, ForwardDiff.partials.(V_d, 1))
end

@testset "Dual _brute_smallaxis! — the -Inf continuation is a primal fact" begin
    # The asymmetry that makes this reachable only one way: the fused scan prunes `-Inf` REWARDS
    # before the comparison (`isfinite(uas) || continue`), so the best-so-far test sees a sentinel
    # only through `V_end`. Origin row `i = 1` is infeasible at every choice with a distinct live
    # tangent on each, so a comparison on the live value would seat the largest tangent's choice.
    # The call is direct: a Dual-eltype `ArgmaxStage` is refused at construction (`tangent_grade`),
    # so this is the one route by which a `Dual` reaches the walk's sentinel comparisons.
    pre, n_start, n_end = 2, 2, 3
    u  = [0.1 * a + 0.01 * s for a in 1:n_end, s in 1:n_start]  # all finite: nothing is pruned
    Ve = [-Inf -Inf -Inf; 1.5 2.5 0.5]
    ps = [ 3.0  9.0  5.0; 0.0 0.0 0.0]

    V_f, P_f = fill(0.0, pre, n_start), fill(0, pre, n_start)
    HouseholdStages._brute_smallaxis!(vec(V_f), vec(P_f), vec(Ve), u, pre, n_start, n_end)

    D        = Dual{Nothing, Float64, 1}                        # a direct call: no lift, no default tag
    Ve_d     = [D(Ve[i, a], ForwardDiff.Partials((ps[i, a],))) for i in 1:pre, a in 1:n_end]
    V_d, P_d = fill(zero(D), pre, n_start), fill(0, pre, n_start)
    HouseholdStages._brute_smallaxis!(vec(V_d), vec(P_d), vec(Ve_d), u, pre, n_start, n_end)

    @test P_d == P_f
    @test ForwardDiff.value.(V_d) == V_f                       # bitwise, `-Inf` rows included
    @test all(iszero, ForwardDiff.partials.(V_d, 1))
end
