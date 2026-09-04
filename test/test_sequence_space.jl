using Test
using HouseholdStages
using HouseholdStages: _perturb, _fake_news_YD, _moment_seed, _seed_env, FakeNewsTag
using ForwardDiff: Dual, value, partials, tagtype
using LinearAlgebra: norm
using Logging: with_logger

# Driver fixtures and the direct-method oracle #
#----------------------------------------------#
# The oracle the driver is measured against is the zero-AD baseline of `SSJ_DERIVATIVES.md` §3.2:
# perturb one date of an `env` path, re-solve the transition (which re-optimizes, so no channel is
# frozen), difference the moment paths. Both it and the driver are `O(h²)` objects, so the fixtures
# solve their steady states to `1e-11`. At the package defaults (`vfi_tol = 1e-7`,
# `lambda_tol = 1e-6`) the two agree only to 2.2e-5 relative, and that residual is not the driver:
# it is `lambda_tol` alone — the distance from `Λ_ss` to the true stationary distribution, which
# the two methods propagate differently. Tightening `vfi_tol` to 1e-11 at the default
# `lambda_tol` leaves it at 2.2e-5; tightening `lambda_tol` to 1e-11 collapses it to 2.0e-10.

_SSJ_H   = 1e-5      # FD step: flat over 1e-4…1e-6, where the `O(h²)`/roundoff trade sits
_SSJ_TOL = 1e-11     # steady-state solve tolerance for both inner fixed points

"The 3-stage Aiyagari chain of `test_lift_jacobian.jl`, on a wealth grid fine enough to hold an interior stationary distribution."
function _ssj_aiyagari(; nw::Int = 30, wmax::Real = 12.0,
                         moments = (; K_supplied = at_end(integrand = :wealth, reduce = sum)))
    layout = GriddedLayout(
        :wealth => GriddedContinuous(0.0, wmax, nw),
        :income => Discrete([0.5, 1.5]),
    )
    shock   = MarkovStage(layout; axis = :income, transition_matrix = [0.7 0.3; 0.3 0.7])
    receipt = WealthChangeStage(layout;
        wealth_post = (; wealth, income, env) -> (1 + env.r) * wealth + env.w * income,
        axis        = :wealth,
    )
    saves = ConsumptionSavingsStage(layout; β = 0.96, utility = (cell, c) -> log(c), axis = :wealth)
    return define_moments!(shock ∘ receipt ∘ saves; moments...)
end

"An OLG-class `⊕` chain: two Aiyagari factors differing in their income persistence, joined on `:group`."
function _ssj_product()
    layout = GriddedLayout(
        :wealth => GriddedContinuous(0.0, 6.0, 12),
        :income => Discrete([0.5, 1.5]),
        :group  => Discrete([1]),
    )
    factor(P) = MarkovStage(layout; axis = :income, transition_matrix = P) ∘
                WealthChangeStage(layout;
                    wealth_post = (; wealth, income, env) -> (1 + env.r) * wealth + env.w * income,
                    axis        = :wealth) ∘
                ConsumptionSavingsStage(layout; β = 0.96, utility = (cell, c) -> log(c), axis = :wealth)
    ps = product(factor([0.7 0.3; 0.3 0.7]), factor([0.9 0.1; 0.1 0.9]); axis = :group)
    return define_moments!(ps; wealth = at_end(integrand = :wealth, reduce = sum))
end

"""
The direct method by finite differences on transition paths: column `s` perturbs `env_path[s]` by
`±h` and differences the moment paths, from `Λ_0 = Λ_ss` to `V_T = V_ss`.
"""
function _ssj_direct_J(hh, env_ss, V_ss, Λ_ss, T, input, output; h = _SSJ_H)
    lane(δ, s) = solve_transition_given_env_path!(hh,
                     [t == s ? _perturb(env_ss, input, δ) : env_ss for t in 1:T];
                     Λ_0 = Λ_ss, V_T = V_ss).moments_path
    return hcat(map(1:T) do s
        m_plus, m_minus = lane(+h, s), lane(-h, s)
        [(m_plus[t][output] - m_minus[t][output]) / (2h) for t in 1:T]
    end...)
end

"""
The driver's own `(𝒟, 𝒴)` for one input, read straight out of the lane machinery. Given the same
`(V_ss, Λ_ss)` the driver reads off the chain's buffers, this reproduces its lanes exactly.
"""
_ssj_YD(hh, env, V_ss, Λ_ss, T, input, output; h = _SSJ_H, mode::Symbol = :fd) =
    only(_fake_news_YD(hh, env, V_ss, Λ_ss, T, (input,), Val(mode); h, out_names = (output,)))

"The quadratic mixing pair at an env-carried curvature — `c(θ) = θ²/2κ`, `θ*(y) = clamp(κy, 0, 1)`."
_ssj_mix_cost(θ; env)   = θ^2 / (2 * env.κ)
_ssj_mix_policy(y; env) = clamp(env.κ * y, 0.0, 1.0)

"""
An Aiyagari chain whose income transition is a CHOSEN mixture: the household pays `θ²/2κ` to raise
the probability that the persistent kernel operates rather than the mean-reverting one. The seated
`θ*` carries a tangent in both channels a driver drives — `:r` through `V_end`, `:κ` through the
cost's own env argument — and the forward routes mass by that same `θ*`.
"""
function _ssj_mixing(; nw::Int = 16)
    layout = GriddedLayout(
        :wealth => GriddedContinuous(0.0, 8.0, nw),
        :income => Discrete([0.5, 1.5]),
    )
    mix = MixingStage(layout; axis = :income, K_A = [0.95 0.05; 0.05 0.95], K_B = [0.4 0.6; 0.6 0.4],
                      cost = _ssj_mix_cost, policy = _ssj_mix_policy)
    receipt = WealthChangeStage(layout;
        wealth_post = (; wealth, income, env) -> (1 + env.r) * wealth + env.w * income,
        axis        = :wealth,
    )
    saves = ConsumptionSavingsStage(layout; β = 0.96, utility = (cell, c) -> log(c), axis = :wealth)
    return define_moments!(mix ∘ receipt ∘ saves;
                           K_supplied = at_end(integrand = :wealth, reduce = sum))
end

"The seated mixing weight of a chain whose leading component is the `MixingStage`."
_ssj_θ(hh) = policy(first(hh.buffer.stages))

@testset "J_from_F — recursive cumulation by hand" begin
    # F[s, t]: row 1 = direct effect at shock time t; row > 1 = effect s
    # periods after the shock. J cumulates anti-diagonals.
    F = Float64[
        1  2  3 ;       # direct effects at t = 1, 2, 3
        0  0  0 ;       # 1-period-ahead distribution-mediated effects
        0  0  0 ;       # 2-period-ahead
    ]
    # With no distribution-mediated effects the cumulation only carries the row-1 direct effects
    # down the anti-diagonals: J[s+1, t] += J[s, t-1].
    J = J_from_F(F)
    @test J[1, :] == [1, 2, 3]    # direct effects unchanged
    @test J[2, 1] == 0
    @test J[2, 2] == 1            # added from J[1, 1] = 1
    @test J[2, 3] == 2            # added from J[1, 2] = 2
    @test J[3, 1] == 0
    @test J[3, 2] == 0
    @test J[3, 3] == 1            # added from J[2, 2] = 1
end

@testset "J_from_F — non-trivial distribution-mediated effects" begin
    F = Float64[
        1  0  0 ;
        2  0  0 ;
        3  0  0 ;
    ]
    J = J_from_F(F)
    @test J[1, :] == [1, 0, 0]
    @test J[2, :] == [2, 1, 0]    # row 2: F[2] + shifted-J[1]
    @test J[3, :] == [3, 2, 1]    # row 3: F[3] + shifted-J[2]
end

@testset "build_F — direct effects in row 1; outer products in rows ≥ 2" begin
    curlyY = Float64[1.0, 2.0]
    curlyD = [Float64[0.5, 0.5], Float64[1.0, 0.0]]  # one per shock time
    curlyE = [Float64[2.0, 0.0]]                       # one period of E
    # F has shape (T_lookahead, T) = (2, 2).
    F = build_F(curlyY, curlyD, curlyE)
    @test F[1, :] == [1.0, 2.0]
    # F[2, 1] = sum(curlyE[1] .* curlyD[1]) = 2.0 * 0.5 + 0 * 0.5 = 1.0
    # F[2, 2] = sum(curlyE[1] .* curlyD[2]) = 2.0 * 1.0 + 0 * 0.0 = 2.0
    @test F[2, 1] ≈ 1.0
    @test F[2, 2] ≈ 2.0
end

@testset "expectation_vectors — pure Markov chain at SS" begin
    # Two-state Markov with transition matrix P (rows = today, cols =
    # tomorrow). For pure Markov, K = P^T (forward); K^T = P (backward).
    # E_t[integrand](s) = P^t · integrand applied as a vector.
    P = [0.7 0.3; 0.3 0.7]
    layout = GriddedLayout(:z => Discrete([0.0, 1.0]))
    chain = MarkovStage(layout; axis = :z, transition_matrix = P)
    # Seed kernels at the steady state via a backward call.
    backward!(chain, zeros(2), NamedTuple())
    # Integrand: identity on z (so E_t[z | s_0 = s] is the t-step
    # forward expected value of z).
    integrand = cell -> cell.z
    Es = expectation_vectors(chain, integrand, 3)
    @test length(Es) == 3
    # E_0 = [0.0, 1.0] (identity).
    @test Es[1] ≈ [0.0, 1.0]
    # E_1[s] = sum_{s'} P[s, s'] z[s'] = P[s, 1]·0 + P[s, 2]·1 = P[s, 2].
    @test Es[2] ≈ [P[1, 2], P[2, 2]]
    # E_2 = P · E_1 (matrix product).
    @test Es[3] ≈ P * Es[2] atol = 1e-12
end

@testset "expectation_vectors — chain (Markov ∘ Identity)" begin
    P = [0.8 0.2; 0.2 0.8]
    layout = GriddedLayout(:z => Discrete([0.0, 1.0]))
    s1 = MarkovStage(layout; axis = :z, transition_matrix = P)
    s2 = IdentityStage(layout)
    chain = s1 ∘ s2
    # Seed kernels.
    backward!(chain, zeros(2), NamedTuple())
    integrand = cell -> cell.z
    Es = expectation_vectors(chain, integrand, 2)
    @test length(Es) == 2
    @test Es[1] ≈ [0.0, 1.0]
    # The chain's K^T = (IdentityStage K^T) · (MarkovStage K^T) = I · P.
    @test Es[2] ≈ P * Es[1] atol = 1e-12
end

@testset "expectation_vectors — a block whose two ends differ is refused by name" begin
    l2 = GriddedLayout(:z => Discrete([0.5, 1.5]))
    l3 = GriddedLayout(:z => Discrete([0.5, 1.0, 1.5]))
    rect = MarkovStage(l2, l3; axis = :z, transition_matrix = [0.5 0.3 0.2; 0.1 0.6 0.3])
    backward!(rect, zeros(3), NamedTuple())
    @test_throws "expectation_vectors" expectation_vectors(rect, cell -> cell.z, 3)
end

@testset "compute_fake_news_ssj — the fake-news identity against the direct method" begin
    # The integration oracle. `J` from the driver against the direct method (FD on transition
    # paths) on the Aiyagari chain, plus the fake-news property `F[t,s] = J[t,s] − J[t−1,s−1]`.
    T   = 8
    hh  = _ssj_aiyagari()
    env = (; r = 0.03, w = 1.0)
    ss  = solve_steady_state_given_env!(hh, env; vfi_tol = _SSJ_TOL, lambda_tol = _SSJ_TOL)
    @test sum(ss.Λ) ≈ 1.0 atol = 1e-10
    @test ss.moments.K_supplied > 1.0                     # an interior stationary distribution

    J = compute_fake_news_ssj(hh, env, T;
                              inputs = (:r,), outputs = (:K_supplied,), h = _SSJ_H)
    @test J isa Matrix{Float64}
    @test size(J) == (T, T)

    # The base point the driver solved for and left in the buffers. Every oracle below is built at
    # it, so a gap is the method's, not a difference of linearization points.
    V_ss, Λ_ss = copy(V_start_buffer(hh)), copy(Λ_end_buffer(hh))

    J_direct = _ssj_direct_J(hh, env, V_ss, Λ_ss, T, :r, :K_supplied)
    # Worst relative gap over all 64 cells: 2.0e-10 as delivered, against a 1e-8 bar.
    @test maximum(abs, J .- J_direct) / maximum(abs, J_direct) < 1e-8
    @test J ≈ J_direct rtol = 1e-8

    # THE ACCEPTANCE GATE. `J[:, s]` for a shock date `s > 0` must have a nonzero ANTICIPATION
    # block — the entries at output dates `t < s`, which respond before the shock lands. A
    # frozen-policy contract zeroes `𝒟_u` for `u ≥ 1` and this whole triangle with it.
    # The weakest of the 28 is 7.0% of the whole matrix's largest entry as delivered.
    anticipation = [J[t, s] for s in 2:T for t in 1:(s - 1)]
    @test minimum(abs, anticipation) > 0.01 * maximum(abs, J)   # every entry, not merely their maximum
    @test all(s -> abs(J[1, s]) > 0.01 * maximum(abs, J), 2:T)  # date-0 output moves for every future shock
    @test maximum(abs, [J[t, s] - J_direct[t, s] for s in 2:T for t in 1:(s - 1)]) < 1e-8

    # The fake-news property proper, measured against the DIRECT method's own anti-diagonal
    # differences. `J` is what `J_from_F` cumulated out of this same `F`, so reading `F` back
    # against `J` is arithmetic of the assembly: it pins that the driver's `_moment_seed` seeding
    # and its `ℰ[2:end]` slice reproduce this `F`, but a wrong `ℰ` sits on both sides of it and
    # cancels. Only the direct-method form measures the object.
    YD = _ssj_YD(hh, env, V_ss, Λ_ss, T, :r, :K_supplied)
    backward!(hh, V_ss, env)                              # `expectation_vectors`' seating precondition
    ℰ = expectation_vectors(hh, cell -> cell.wealth, T)
    F = build_F(YD.curlyY[:K_supplied], YD.curlyD, ℰ[2:end])
    # Worst over the 49 cells `t,s ≥ 2` as delivered: 2.1e-10 against the direct method, on a
    # scale of `maximum(abs, J_direct)` = 4.68; 3.6e-16 against `J`.
    @test maximum(abs, [F[t, s] - (J_direct[t, s] - J_direct[t - 1, s - 1]) for t in 2:T, s in 2:T]) < 1e-8
    @test maximum(abs, F[1, :] .- J_direct[1, :]) < 1e-8
    @test maximum(abs, [F[t, s] - (J[t, s] - J[t - 1, s - 1]) for t in 2:T, s in 2:T]) < 1e-12
    @test F[1, :] == J[1, :]

    # End-of-period timing, independently of `build_F`: `𝒴_u = ⟨ℰ₀, 𝒟_u⟩` at every `u`, the
    # integrand's `∂f/∂env` term vanishing because `:wealth` does not read env. Worst over the
    # eight `u` as delivered: 2.6e-11.
    @test maximum(u -> abs(YD.curlyY[:K_supplied][u] - sum(ℰ[1] .* YD.curlyD[u])), 1:T) < 1e-9
end

@testset "compute_fake_news_ssj — an env-reading integrand and the ∂f/∂env term" begin
    # The sweep reads its moments at the period's OWN env, so at `u = 0` — the shocked one — `𝒴_0`
    # carries the integrand's `⟨∂f/∂env, Λ_ss⟩` on top of `⟨ℰ₀, 𝒟_0⟩`, and at `u ≥ 1` it does not.
    # An env-free integrand zeroes that term and makes the `u = 0` env unobservable; this fixture's
    # integrand reads `env.r`, where the term is 97% of `𝒴_0`.
    T   = 6
    env = (; r = 0.03, w = 1.0)
    hh  = _ssj_aiyagari(moments = (; inc = at_end(integrand = (; wealth, env) -> env.r * wealth,
                                                  reduce = sum)))
    ss  = solve_steady_state_given_env!(hh, env; vfi_tol = _SSJ_TOL, lambda_tol = _SSJ_TOL)

    J        = compute_fake_news_ssj(hh, env, T; inputs = (:r,), outputs = (:inc,), h = _SSJ_H)
    J_direct = _ssj_direct_J(hh, env, ss.V, ss.Λ, T, :r, :inc)
    # Worst relative gap over the 36 cells: 1.5e-10 as delivered. Dropping the `∂f/∂env` term
    # takes the diagonal from 2.3381 to 0.0625 and this gap to 0.95.
    @test maximum(abs, J .- J_direct) / maximum(abs, J_direct) < 1e-8

    YD   = _ssj_YD(hh, env, ss.V, ss.Λ, T, :r, :inc)
    ℰ₀   = _moment_seed(hh, :inc, env)
    ∂f∂r = (_moment_seed(hh, :inc, _perturb(env, :r, +_SSJ_H)) .-
            _moment_seed(hh, :inc, _perturb(env, :r, -_SSJ_H))) ./ (2 * _SSJ_H)
    𝒴, 𝒟 = YD.curlyY[:inc], YD.curlyD
    # Worst over the five `u ≥ 1` as delivered: 1.1e-12. The `u = 0` gap is the term itself, and
    # the residual after subtracting it is 1.5e-10 — the `O(h²)` of the two-sided difference.
    @test maximum(u -> abs(𝒴[u] - sum(ℰ₀ .* 𝒟[u])), 2:T) < 1e-9
    @test abs(𝒴[1] - sum(ℰ₀ .* 𝒟[1]) - sum(∂f∂r .* ss.Λ)) < 1e-8
    @test abs(sum(∂f∂r .* ss.Λ)) > 0.5 * abs(𝒴[1])        # the term carries 𝒴_0, not a rounding effect
end

@testset "compute_fake_news_ssj — mass structure of 𝒟" begin
    # Columns of 𝐊 sum to 1 on a mass-conserving chain, so columns of 𝐊̇ sum to 0 and Σ𝒟ᵤ = 0
    # for every u. An aggregate identity rather than a discriminating one: it reads column sums
    # of 𝐊̇, so anything that leaves the sweep mass-conserving passes it. Where it bites is the
    # entry fixture below, whose Σ𝒟₀ is pinned to an independently computed ġ. In `:fd` it is
    # h-noise-limited, hence the relative bar.
    T   = 6
    hh  = _ssj_aiyagari()
    env = (; r = 0.03, w = 1.0)
    ss     = solve_steady_state_given_env!(hh, env; vfi_tol = _SSJ_TOL, lambda_tol = _SSJ_TOL)
    curlyD = _ssj_YD(hh, env, ss.V, ss.Λ, T, :r, :K_supplied).curlyD
    # Worst |Σ𝒟ᵤ|/‖𝒟ᵤ‖ over the six u as delivered: 8.3e-12.
    @test maximum(u -> abs(sum(curlyD[u])) / norm(curlyD[u]), 1:T) < 1e-9
    @test minimum(u -> norm(curlyD[u]), 1:T) > 1e-3       # the ratio is not vacuous

    # On an entry chain mass is not conserved: Σ𝒟₀ is the entry source's own tangent mass. Entry
    # sits last, so no downstream kernel touches ġ and the identity is exact.
    layout = GriddedLayout(:z => Discrete([0.0, 1.0]))
    entry  = MarkovStage(layout; axis = :z, transition_matrix = 0.9 .* [0.7 0.3; 0.3 0.7]) ∘
             EntryStage(layout; entry = (; z, env) -> env.inflow * (1 + z))
    hh_e   = define_moments!(entry; pop = at_end(integrand = (; z) -> 1.0, reduce = sum))
    env_e  = (; inflow = 0.2)
    ss_e = solve_steady_state_given_env!(hh_e, env_e; vfi_tol = _SSJ_TOL, lambda_tol = _SSJ_TOL)
    D_e  = _ssj_YD(hh_e, env_e, ss_e.V, ss_e.Λ, 4, :inflow, :pop).curlyD
    ġ_mass = sum(1 .+ [0.0, 1.0])                          # ∂g/∂inflow = (1 + z), summed over cells
    @test sum(D_e[1]) ≈ ġ_mass atol = 1e-9
    @test maximum(u -> abs(sum(D_e[u])), 2:4) < 1e-12     # nothing else in this chain reads env
end

@testset "compute_fake_news_ssj — 𝒟 is time-invariant across the horizon" begin
    # The date-0 innovations depend only on the anticipation distance u, so the first T₁ of a
    # T₂-long sweep are the whole of a T₁-long one.
    hh  = _ssj_aiyagari(nw = 16, wmax = 6.0)
    env = (; r = 0.03, w = 1.0)
    ss    = solve_steady_state_given_env!(hh, env; vfi_tol = _SSJ_TOL, lambda_tol = _SSJ_TOL)
    short = _ssj_YD(hh, env, ss.V, ss.Λ, 4, :r, :K_supplied)
    long  = _ssj_YD(hh, env, ss.V, ss.Λ, 9, :r, :K_supplied)
    @test maximum(u -> maximum(abs, short.curlyD[u] .- long.curlyD[u]), 1:4) < 1e-12
    @test maximum(u -> abs(short.curlyY[:K_supplied][u] - long.curlyY[:K_supplied][u]), 1:4) < 1e-12
    @test maximum(u -> maximum(abs, short.curlyD[u]), 1:4) > 1e-3
end

@testset "compute_fake_news_ssj — trivial structure and the refusals" begin
    T      = 4
    layout = GriddedLayout(:z => Discrete([0.0, 1.0]))
    markov = MarkovStage(layout; axis = :z, transition_matrix = [0.7 0.3; 0.3 0.7])
    hh     = define_moments!(markov; m = at_end(integrand = :z, reduce = sum))
    env    = (; dummy = 1.0)
    solve_steady_state_given_env!(hh, env)

    # A constant transition matrix is a constant, not an env dependence: every channel is zero.
    J = compute_fake_news_ssj(hh, env, T; inputs = (:dummy,), outputs = (:m,))
    @test size(J) == (T, T)
    @test all(iszero, J)

    # Declared dependence is the only way env acts, and an input that is not in env is refused.
    @test_throws AssertionError compute_fake_news_ssj(hh, env, T; inputs = (:nope,), outputs = (:m,))
    # This chain declares no env fields at all, so the default input tuple is empty and says so.
    @test_throws "declares no env fields" compute_fake_news_ssj(hh, env, T; outputs = (:m,))
    @test_throws "the implemented modes are :fd and :dual" compute_fake_news_ssj(hh, env, T; inputs = (:dummy,), mode = :sideways)

    # A nonlinear reduce breaks the bilinear assembly.
    hh_mean = define_moments!(MarkovStage(layout; axis = :z, transition_matrix = [0.7 0.3; 0.3 0.7]);
                              m = at_end(integrand = :z, reduce = maximum))
    solve_steady_state_given_env!(hh_mean, env)
    @test_throws "reduce = sum" compute_fake_news_ssj(hh_mean, env, T; inputs = (:dummy,))

    # THE STEADY-STATE BASE POINT is the driver's own, not the caller's. `J` is taken AT a steady
    # state of `env_ss`, and the solver's defaults are four orders looser than that needs, so the
    # driver solves for it and the answer cannot depend on what the caller left in the buffers.
    # Each state below used to be a different wrong number, then a refusal; all four now agree.
    aiy_env = (; r = 0.03, w = 1.0)
    J_of(hh) = compute_fake_news_ssj(hh, aiy_env, T; inputs = (:r,), outputs = (:K_supplied,))
    fresh() = _ssj_aiyagari()   # the default grid: on `nw = 10` this chain's `J` is 1e-12, all noise

    reference = J_of(let hh = fresh()                       # solved tightly, at the right env
        solve_steady_state_given_env!(hh, aiy_env; vfi_tol = _SSJ_TOL, lambda_tol = _SSJ_TOL); hh
    end)
    @test reference isa Matrix{Float64}

    @test J_of(fresh()) ≈ reference rtol = 1e-8             # (a) nobody solved it
    @test J_of(let hh = fresh()                             # (b) solved at the package defaults
        solve_steady_state_given_env!(hh, aiy_env); hh
    end) ≈ reference rtol = 1e-8
    @test J_of(let hh = fresh()                             # (c) `V` tight, `Λ` at the default
        solve_steady_state_given_env!(hh, aiy_env; vfi_tol = _SSJ_TOL, lambda_tol = 1e-6); hh
    end) ≈ reference rtol = 1e-8
    @test J_of(let hh = fresh()                             # (d) converged at a DIFFERENT env
        solve_steady_state_given_env!(hh, _perturb(aiy_env, :r, 1e-4);
                                      vfi_tol = _SSJ_TOL, lambda_tol = _SSJ_TOL); hh
    end) ≈ reference rtol = 1e-7

    # And the seating is left where the sweeps need it: both buffers stationary at `env_ss` after
    # the call, whatever the chain was handed in.
    seated = fresh(); J_of(seated)
    V_ss, Λ_ss = copy(V_start_buffer(seated)), copy(Λ_end_buffer(seated))
    @test maximum(abs.(copy(backward!(seated, V_ss, aiy_env)) .- V_ss)) ≤ 1e-6 * (1 + maximum(abs, V_ss))
    @test maximum(abs.(copy(forward!(seated, Λ_ss)) .- Λ_ss)) ≤ 1e-10

    # Moments attach at the end of a chain, so a bare product block has no moment surface.
    prod_layout = GriddedLayout(:z => Discrete([0.0, 1.0]), :g => Discrete([1]))
    leg() = MarkovStage(prod_layout; axis = :z, transition_matrix = [0.7 0.3; 0.3 0.7])
    @test_throws "keyed on `ChainStage`" compute_fake_news_ssj(product(leg(), leg(); axis = :g), env, T)
end

@testset "compute_fake_news_ssj — a ⊕-bearing chain" begin
    # The OLG class: a `product` component inside the chain. Step 2's `ℰ` recursion reaches through
    # the block-diagonal adjoint, and the driver's sweeps run on the fused tensors.
    T   = 4
    hh  = _ssj_product()
    env = (; r = 0.03, w = 1.0)
    ss  = solve_steady_state_given_env!(hh, env; vfi_tol = _SSJ_TOL, lambda_tol = _SSJ_TOL)
    @test ss.moments.wealth > 0.5

    J        = compute_fake_news_ssj(hh, env, T; inputs = (:r,), outputs = (:wealth,), h = _SSJ_H)
    J_direct = _ssj_direct_J(hh, env, ss.V, ss.Λ, T, :r, :wealth)
    @test size(J) == (T, T)
    # Worst relative gap over the 16 cells: 8.5e-11 as delivered.
    @test maximum(abs, J .- J_direct) / maximum(abs, J_direct) < 1e-8
    @test minimum(abs, [J[t, s] for s in 2:T for t in 1:(s - 1)]) > 0.01 * maximum(abs, J)

    curlyD = _ssj_YD(hh, env, ss.V, ss.Λ, T, :r, :wealth).curlyD
    @test maximum(u -> abs(sum(curlyD[u])) / norm(curlyD[u]), 1:T) < 1e-9
end

@testset "compute_fake_news_ssj — :dual against :fd, and the block it is exact on" begin
    # `:dual` sweeps one tangent-seated Dual chain and reads `(𝒟, 𝒴)` off the partials, so `h` never
    # enters. The oracles are the same two `:fd` faces: the driver's own `:fd` lane, and the direct
    # method (FD on transition paths).
    T   = 8
    hh  = _ssj_aiyagari()
    env = (; r = 0.03, w = 1.0)
    ss  = solve_steady_state_given_env!(hh, env; vfi_tol = _SSJ_TOL, lambda_tol = _SSJ_TOL)

    J_dual = compute_fake_news_ssj(hh, env, T; inputs = (:r,), outputs = (:K_supplied,), mode = :dual)
    J_fd   = compute_fake_news_ssj(hh, env, T; inputs = (:r,), outputs = (:K_supplied,), h = _SSJ_H)
    @test J_dual isa Matrix{Float64}
    @test size(J_dual) == (T, T)
    # 1.11e-11 relative as delivered. Both modes sweep from the SAME `(V_ss, Λ_ss)`, so the solve's
    # own convergence error is common-mode and drops out: the gap is flat in the steady-state
    # tolerance (1.11e-11 … 1.65e-11 across `vfi_tol = lambda_tol` from 1e-6 to 1e-13) and is `:fd`'s
    # `O(h²)`. Against the RE-SOLVED direct method the convergence floor does bite, and the fixture
    # solves to 1e-11 for it: at the package defaults that comparison caps at 2.2e-5.
    @test maximum(abs, J_dual .- J_fd) / maximum(abs, J_dual) < 1e-8

    J_direct = _ssj_direct_J(hh, env, ss.V, ss.Λ, T, :r, :K_supplied)
    # Worst relative gap over all 64 cells: 1.95e-10 as delivered.
    @test maximum(abs, J_dual .- J_direct) / maximum(abs, J_direct) < 1e-8

    # THE EXACTNESS STATEMENT, on the anticipation block `t < s`. At the default `h = 1e-5` both
    # modes sit on the direct method's own convergence floor — 1.89e-10 for `:dual`, 1.72e-10 for
    # `:fd` over the 28 entries — and comparing them there reads noise. Two coarser `h` separate
    # them, by different mechanisms, so both are asserted.
    antic(J) = maximum(abs, [J[t, s] - J_direct[t, s] for s in 2:T for t in 1:(s - 1)])
    fd_at(h) = antic(compute_fake_news_ssj(hh, env, T; inputs = (:r,), outputs = (:K_supplied,), h))

    # (a) `h = 1e-3` is inside the argmax NODE-SWITCH regime: `:fd` re-solves across a switch and
    # carries the step (2.35e-2), where the seated tangent cannot. Not truncation — the gap is flat
    # in `h` through here, and O(h²) truncation at this `h` would be ~2.4e-7.
    @test fd_at(1e-3)   > 1e-3
    @test antic(J_dual) < 1e-6 * fd_at(1e-3)

    # (b) `h = 1.25e-4` is where FD is back in its own O(h²) regime (ratio 3.90 per halving), and
    # this is the `h` at which the exactness claim means what it says: 3.70e-9 for `:fd` against
    # 1.89e-10 for `:dual`, a genuine 20×.
    @test fd_at(1.25e-4) > 10 * antic(J_dual)
    @test antic(J_dual)  < 1e-8

    # The WP5 oracle re-run in `:dual`: end-of-period timing (`𝒴_u = ⟨ℰ₀, 𝒟_u⟩` at every u, the
    # `∂f/∂env` term vanishing because `:wealth` does not read env), the fake-news identity against
    # the direct method's own anti-diagonal differences, and the mass structure of `𝒟`.
    YD = _ssj_YD(hh, env, ss.V, ss.Λ, T, :r, :K_supplied; mode = :dual)
    backward!(hh, ss.V, env)                              # `expectation_vectors`' seating precondition
    ℰ = expectation_vectors(hh, cell -> cell.wealth, T)
    F = build_F(YD.curlyY[:K_supplied], YD.curlyD, ℰ[2:end])
    @test maximum(u -> abs(YD.curlyY[:K_supplied][u] - sum(ℰ[1] .* YD.curlyD[u])), 1:T) < 1e-12   # 2.2e-16
    @test maximum(abs, [F[t, s] - (J_direct[t, s] - J_direct[t - 1, s - 1]) for t in 2:T, s in 2:T]) < 1e-8
    @test maximum(abs, F[1, :] .- J_direct[1, :]) < 1e-8
    # Σ𝒟ᵤ = 0 on a mass-conserving chain, and in `:dual` there is no h-noise to relax the bar to.
    @test maximum(u -> abs(sum(YD.curlyD[u])) / norm(YD.curlyD[u]), 1:T) < 1e-14   # 1.3e-16
end

@testset "compute_fake_news_ssj — :dual through a seated Mixing policy" begin
    # The reason a Dual `θ*` is seated at all: the FORWARD routes mass by it, so `Λ̇` carries
    # `K̇Λ_ss` through `θ̇`. `:fd` gets that channel by re-optimizing in each lane; `:dual` gets it
    # from the Fenchel tangent. Agreement is the gate on the tangent, in both channels that move
    # `θ*` — `:r` through `V_end`, `:κ` through the cost's own env argument.
    T   = 5
    hh  = _ssj_mixing()
    env = (; r = 0.03, w = 1.0, κ = 1.0)
    ss  = solve_steady_state_given_env!(hh, env; vfi_tol = _SSJ_TOL, lambda_tol = _SSJ_TOL)
    @test count(θ -> 0 < θ < 1, _ssj_θ(hh)) == 16         # half the cells seat interior: θ̇ ≠ 0 there
    @test maximum(_ssj_θ(hh)) > 0.5

    for input in (:r, :κ)
        J_dual   = compute_fake_news_ssj(hh, env, T; inputs = (input,), outputs = (:K_supplied,), mode = :dual)
        J_fd     = compute_fake_news_ssj(hh, env, T; inputs = (input,), outputs = (:K_supplied,), h = _SSJ_H)
        J_direct = _ssj_direct_J(hh, env, ss.V, ss.Λ, T, input, :K_supplied)
        # Worst relative gap as delivered: `:r` 3.2e-11 / 9.7e-11, `:κ` 1.8e-10 / 2.8e-10.
        @test maximum(abs, J_dual .- J_fd)     / maximum(abs, J_dual)   < 1e-8
        @test maximum(abs, J_dual .- J_direct) / maximum(abs, J_direct) < 1e-8
        @test maximum(abs, J_dual) > 1e-3                 # the channel is not vacuously flat
    end

    # Chunking is a partition of the input list, not a second lane: `n_dual` sweeps one Dual chain
    # of that many partials, and the Jacobians are identical whichever way the inputs fall.
    wide   = compute_fake_news_ssj(hh, env, T; inputs = (:r, :κ), outputs = (:K_supplied,), mode = :dual)
    narrow = compute_fake_news_ssj(hh, env, T; inputs = (:r, :κ), outputs = (:K_supplied,), mode = :dual, n_dual = 1)
    @test Set(keys(wide)) == Set([(:r, :K_supplied), (:κ, :K_supplied)])
    # 8.8e-16 relative as delivered — a few ulps of reassociation, nothing else.
    @test maximum(k -> maximum(abs, wide[k] .- narrow[k]) / maximum(abs, wide[k]), keys(wide)) < 1e-12
end

@testset "compute_fake_news_ssj — tangent-level duality on the three operator classes" begin
    # `⟨Kᵀ V_end, Λ_start⟩ = ⟨V_end, K Λ_start⟩` for the SEATED Dual kernel, in the value lane and
    # in the partials, with the affine terms of the primal verbs written down rather than tolerated.
    # `MixingStage`'s backward writes `V_start = K_θᵀ V_end − c(θ*)`, so the flow payoff enters as
    # `⟨c(θ*), Λ_start⟩` — evaluated at the LIVE Dual `θ*`, which makes the same term correct in
    # both lanes, since `c′(θ*) = y` is the stage's own Fenchel condition. `EntryStage`'s forward is
    # `Λ_end = K Λ_start + g`, adding `⟨V_end, g⟩`. Mixing leads each chain so its `Λ_in` IS
    # `Λ_start`; the correction is the whole gap, 3.6e-3 … 0.92 uncorrected against 2.3e-16 with it.
    lay = GriddedLayout(:z => Discrete([0.5, 1.0, 1.5, 2.0]))
    P_A = [0.7 0.2 0.1 0.0; 0.1 0.6 0.2 0.1; 0.1 0.2 0.5 0.2; 0.0 0.1 0.3 0.6]
    P_B = [0.4 0.3 0.2 0.1; 0.5 0.3 0.1 0.1; 0.6 0.2 0.1 0.1; 0.7 0.2 0.1 0.0]
    env = (; κ = 2.5, inflow = 0.2)
    mixer(A, B) = MixingStage(lay; axis = :z, K_A = A, K_B = B,
                              cost = _ssj_mix_cost, policy = _ssj_mix_policy)
    V_val, V_tan, Λ_start = [0.9, 0.2, 0.7, 0.35], [0.2, -0.5, 0.9, 0.4], [0.4, 0.3, 0.2, 0.1]
    g_entry = [env.inflow * (1 + z) for z in [0.5, 1.0, 1.5, 2.0]]

    fixtures = (
        ("mass-conserving", mixer(P_A, P_B) ∘ MarkovStage(lay; axis = :z, transition_matrix = P_A), 0.0),
        ("sub-stochastic exit", RetentionStage(lay; axis = :z, exit_kernel = 0.6 .* P_B,
                                               cost = _ssj_mix_cost, policy = _ssj_mix_policy) ∘
                                MarkovStage(lay; axis = :z, transition_matrix = P_A), 0.0),
        ("affine entry", mixer(P_A, P_B) ∘ EntryStage(lay; entry = (; z, env) -> env.inflow * (1 + z)), g_entry),
    )
    masses = map(fixtures) do (name, chain, g)
        chain_d = lift_jacobian(chain; n_dual = 1)
        D       = eltype(V_start_buffer(chain_d))
        V_end   = Dual{tagtype(D)}.(V_val, V_tan)
        Λ_s     = D.(Λ_start)
        V_st    = copy(backward!(chain_d, V_end, env))     # seats every kernel, θ* among them
        Λ_e     = copy(forward!(chain_d, Λ_s))
        θ       = policy(first(chain_d.buffer.stages))
        raw     = sum(V_end .* Λ_e) - sum(V_st .* Λ_s)
        gap     = raw - sum(_ssj_mix_cost.(θ; env) .* Λ_s) - sum(V_end .* g)
        @testset "$name" begin
            @test abs(value(gap))              < 1e-10
            @test abs(partials(gap, 1))        < 1e-10
            @test maximum(abs, partials.(θ, 1)) > 1e-3     # the seated θ* really does carry a tangent
            @test abs(value(raw))              > 1e-3      # the correction is load-bearing, not a rounding term
        end
        sum(value.(Λ_e)) / sum(Λ_start)
    end
    # The three classes are distinct in what the forward does to mass: conserved, leaking, grown.
    @test masses[1] ≈ 1.0 atol = 1e-12
    @test masses[2] < 0.95
    @test masses[3] > 1.5
end

@testset "compute_fake_news_ssj — the grade matrix: :fd warns by name, :dual refuses" begin
    # A `:wrong_object` factor buried inside a `⊕` component — the OLG shape. The service names the
    # offending STAGE, which is what a caller can act on, rather than the product that holds it.
    lay = GriddedLayout(:z => Discrete([0.0, 1.0]), :g => Discrete([1]))
    leg() = MarkovStage(lay; axis = :z, transition_matrix = [0.7 0.3; 0.3 0.7]) ∘
            DiscreteMoveStage(lay; axis = :z, destination = (; z) -> 0.0)
    hh  = define_moments!(product(leg(), leg(); axis = :g); m = at_end(integrand = :z, reduce = sum))
    env = (; dummy = 1.0)
    solve_steady_state_given_env!(hh, env)
    @test tangent_grade(hh) === :wrong_object

    J = @test_logs (:warn, r"DiscreteMoveStageSpec") match_mode = :any compute_fake_news_ssj(hh, env, 3; inputs = (:dummy,))
    @test J isa Matrix{Float64}
    @test_throws "DiscreteMoveStageSpec" compute_fake_news_ssj(hh, env, 3; inputs = (:dummy,), mode = :dual)
    @test_throws "mode = :fd" compute_fake_news_ssj(hh, env, 3; inputs = (:dummy,), mode = :dual)

    # The warning is a property of the number each call returns, and `:fd` is the remedy the `:dual`
    # refusal above points at — so every call raises it, over any number of chains. A per-call-site
    # cap would announce the first hard-argmax chain of the session and none of the rest.
    other = define_moments!(MarkovStage(lay; axis = :z, transition_matrix = [0.6 0.4; 0.4 0.6]) ∘
                            DiscreteMoveStage(lay; axis = :z, destination = (; z) -> 1.0);
                            m = at_end(integrand = :z, reduce = sum))
    solve_steady_state_given_env!(other, env)
    logs = TestLogger()
    with_logger(logs) do
        compute_fake_news_ssj(hh, env, 3; inputs = (:dummy,))
        compute_fake_news_ssj(other, env, 3; inputs = (:dummy,))
        compute_fake_news_ssj(hh, env, 3; inputs = (:dummy,))
    end
    @test count(r -> occursin("wrong_object", r.message), logs.logs) == 3
end


# The steady-state gradient and the path tangents #
#-------------------------------------------------#

@testset "compute_steady_state_gradient — ∂K_supplied/∂r against an independent re-solve" begin
    # Layer 1's success criterion (plan §11): the permanent-shock comparative static that the
    # frozen-policy contract got 61.3% too small (study `02b` §E2). The steady state is a fixed
    # point, so it is not differentiated — the driver re-solves at `env_ss ± h`, and the oracle is a
    # central difference written here over FRESH chains, sharing none of the driver's plumbing.
    hh  = _ssj_aiyagari(; nw = 60,
                        moments = (; K_supplied = at_end(integrand = :wealth, reduce = sum),
                                     mass       = at_end(integrand = (; wealth) -> 1.0, reduce = sum)))
    env = (; r = 0.03, w = 1.0)
    ss  = solve_steady_state_given_env!(hh, env; vfi_tol = _SSJ_TOL, lambda_tol = _SSJ_TOL)
    @test ss.moments.mass ≈ 1.0 atol = 1e-10

    g(h) = compute_steady_state_gradient(hh, env; inputs = (:r, :w), h,
                                         vfi_tol = _SSJ_TOL, lambda_tol = _SSJ_TOL)
    @test g(_SSJ_H) isa Dict{Tuple{Symbol, Symbol}, Float64}
    @test Set(keys(g(_SSJ_H))) == Set([(:r, :K_supplied), (:r, :mass), (:w, :K_supplied), (:w, :mass)])

    "Central difference over fresh chains — no driver code on this path."
    function oracle(input, h)
        lane(δ) = solve_steady_state_given_env!(_ssj_aiyagari(; nw = 60,
                      moments = (; K_supplied = at_end(integrand = :wealth, reduce = sum))),
                      _perturb(env, input, δ); vfi_tol = _SSJ_TOL, lambda_tol = _SSJ_TOL).moments.K_supplied
        return (lane(h) - lane(-h)) / (2h)
    end

    # The best `h` is INPUT-SCALED: the error is `O(h²) + O(ε/h)` with `ε` the solve floor in the
    # MOMENT, and a step at `w = 1.0` moves `K_supplied` ~55× less than the same step at `r = 0.03`,
    # so the floor bites 55× sooner. `:r` bottoms at 3e-5…1e-5, `:w` at 3e-4; at a shared `h = 1e-5`
    # the `:w` row is already 1.5e-4 off. A single `h` cannot serve both.
    # 2.7e-6 and 5.1e-6 as delivered: the driver re-solves from the chain's warm buffers and the
    # oracle from cold ones, so the two land at different points inside `_SSJ_TOL`. That spread is
    # the solve floor, not the method.
    @test g(1e-5)[(:r, :K_supplied)] ≈ oracle(:r, 1e-5) rtol = 1e-4
    @test g(3e-4)[(:w, :K_supplied)] ≈ oracle(:w, 3e-4) rtol = 1e-4
    @test g(1e-5)[(:r, :K_supplied)] > 100        # the channel is large, not a rounding term

    # Total mass is conserved at every `env`, so its gradient is structurally zero — free of any
    # oracle. A re-solve gets it to the difference of two solve residuals.
    @test abs(g(1e-5)[(:r, :mass)]) < 1e-6
end

@testset "a Dual-eltype iterate is refused at the fixed point" begin
    # THE SCOPE GATE. Forward-mode AD differentiates the program that ran, and a fixed-point loop's
    # trip count is chosen by a comparison AD does not see. Warm-started at a converged primal the
    # value lane is stationary, the stopping rule reads it, and the solve returns after one pass with
    # a tangent that never iterated — silently. Differentiating a fixed point is a separate problem
    # with a separate method; this package refuses rather than answers it.
    hh  = _ssj_aiyagari()
    env = (; r = 0.03, w = 1.0)
    ss  = solve_steady_state_given_env!(hh, env; vfi_tol = _SSJ_TOL, lambda_tol = _SSJ_TOL)
    TD    = Dual{FakeNewsTag, Float64, 1}
    chain = ChainStage(hh.spec, boundaries(hh), interiors(hh), TD)
    env_d = _seed_env(env, (:r,), TD)

    @test_throws "Dual-eltype iterate is refused" solve_vfi_steady_state_given_env!(
        hh.spec, env_d, chain.buffer; V_init = TD.(ss.V), tol = 1e-10)
    @test_throws "Dual-eltype iterate is refused" solve_lambda_steady_state_given_env!(
        hh.spec, chain.buffer; Λ_init = TD.(ss.Λ), tol = 1e-10)
    @test_throws "Dual-eltype iterate is refused" solve_steady_state_given_env!(
        lift_jacobian(_ssj_aiyagari()), env_d)
    # The Float64 lane is untouched: same call, no tangents, still solves.
    @test solve_vfi_steady_state_given_env!(hh.spec, env, hh.buffer;
                                            V_init = ss.V, tol = 1e-10).converged

    # CF-25: tangent-level duality on the stages WP6/WP6b/WP7 rewrote — `InterpKernel`,
    # `ContinuousArgmaxStage`, `PointwiseScaleStage` — which no fixture had reached at Dual eltype.
    # These are SINGLE applications, not fixed points, so they carry tangents and are in scope.
    # `𝐊ᵀ` is β-free on both sides, so the identity closes with no correction term.
    Λ_s, V_e = TD.(ss.Λ), TD.(ss.V)
    backward!(chain, V_e, env_d)          # ONE application seats the Dual kernels; not a fixed point
    E, Λ_e   = copy(forward_adjoint!(chain, V_e)), copy(forward!(chain, Λ_s))
    gap      = sum(E .* Λ_s) - sum(V_e .* Λ_e)
    @test abs(value(gap))       < 1e-12       # 1.1e-16, at a scale of 0.747
    @test abs(partials(gap, 1)) < 1e-11       # 2.7e-15
    @test maximum(abs, partials.(E, 1)) > 1   # 15.7 — neither side is vacuously untangent
end

@testset "compute_direct_ssj — a column against FD, and the whole matrix against the fake-news route" begin
    # The transition chains are built at `eltype(V_T)`, so a Dual `V_T` and a tangent seeded on one
    # date's env field replay the whole path in the tangent lane. The oracle is the same FD-on-paths
    # direct method the fake-news driver is measured against: both replay the same transition, so
    # the steady-state solve error is common-mode and only `h` separates them.
    T, s = 6, 3
    hh  = _ssj_aiyagari()
    env = (; r = 0.03, w = 1.0)
    ss  = solve_steady_state_given_env!(hh, env; vfi_tol = _SSJ_TOL, lambda_tol = _SSJ_TOL)

    col = compute_direct_ssj(hh, fill(env, T); input = :r, s, Λ_0 = ss.Λ, V_T = ss.V)
    @test col isa NamedTuple{(:K_supplied,)}
    J_direct = _ssj_direct_J(hh, env, ss.V, ss.Λ, T, :r, :K_supplied)
    # Worst relative gap over the six output dates: 1.6e-11 as delivered.
    @test maximum(abs, col.K_supplied .- J_direct[:, s]) / maximum(abs, J_direct[:, s]) < 1e-8
    @test maximum(abs, col.K_supplied) > 1.0              # the column is not vacuously flat

    # The package's two exported routes to `J`, column by column, at the base point the fake-news
    # driver seated. Neither side carries an `h`, and they agree to 1.7e-10 — the same floor the FD
    # oracles hit, which places that floor in the base point's residual and not in the FD step.
    J_fake     = compute_fake_news_ssj(hh, env, T; inputs = (:r,), outputs = (:K_supplied,), mode = :dual)
    V_ss, Λ_ss = copy(V_start_buffer(hh)), copy(Λ_end_buffer(hh))
    J_path     = reduce(hcat, [compute_direct_ssj(hh, fill(env, T); input = :r, s = σ,
                                                  Λ_0 = Λ_ss, V_T = V_ss).K_supplied for σ in 1:T])
    @test maximum(abs, J_fake .- J_path) / maximum(abs, J_path) < 1e-8
    @test maximum(abs, J_fake) > 1.0                      # not a comparison of two zero matrices

    @test_throws "outside the 1:$T path" compute_direct_ssj(hh, fill(env, T); input = :r, s = T + 1,
                                                            Λ_0 = ss.Λ, V_T = ss.V)
end

@testset "compute_steady_state_gradient — the grade matrix: :dual refuses BY DEFAULT" begin
    # The service's default is `:dual`, so a chain carrying a hard argmax is refused without the
    # caller having asked for anything: opting into the h-fragile primal-difference estimate is an
    # explicit `mode = :fd`, and the refusal says so.
    lay = GriddedLayout(:z => Discrete([0.0, 1.0]), :g => Discrete([1]))
    leg() = MarkovStage(lay; axis = :z, transition_matrix = [0.7 0.3; 0.3 0.7]) ∘
            DiscreteMoveStage(lay; axis = :z, destination = (; z) -> 0.0)
    hh  = define_moments!(product(leg(), leg(); axis = :g); m = at_end(integrand = :z, reduce = sum))
    env = (; dummy = 1.0)
    solve_steady_state_given_env!(hh, env)

    # `:fd` re-solves the primal, which does carry a hard argmax's threshold flux — h-fragilely,
    # within `h` of a switch — so a `:wrong_object` chain WARNS here rather than being refused.
    g = @test_logs (:warn, r"DiscreteMoveStageSpec") match_mode = :any compute_steady_state_gradient(
            hh, env; inputs = (:dummy,))
    @test g isa Dict{Tuple{Symbol, Symbol}, Float64}
    # Differentiating the fixed point is out of scope, and asking for it says so by name.
    @test_throws "out of scope" compute_steady_state_gradient(hh, env; inputs = (:dummy,), mode = :dual)
end
