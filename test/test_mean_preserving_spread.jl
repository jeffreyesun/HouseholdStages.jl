using Test
using HouseholdStages

# MeanPreservingSpreadStage: the CONTINUOUS mean-preserving-spread primitive over the banded
# Gaussian-Young row (MeanPreservingSpreadKernel). Value and policy are held to an independent
# brute-quadrature oracle, which reaches the same numbers by a different code path from the
# package's closed-form Φ/φ segment sweep. The two tangent channels are held to central differences
# of the RE-SOLVED primal: the envelope makes the value tangent exact and the IFT the policy's, so
# a re-optimizing difference is what distinguishes a live seat from a frozen one. Wrapped in a
# module so its oracle globals don't leak.

module MeanPreservingSpreadTest
using Test, HouseholdStages, SpecialFunctions
using HouseholdStages.ForwardDiff: Dual, value, partials, tagtype
const HS = HouseholdStages
include("envelope_oracle.jl")

# Bumpy V on a nonuniform grid — the prototype configuration (convex flanks force interior θ*).
const mp_n  = 60
const mp_xs = [10.0 * ((i - 1) / (mp_n - 1))^1.6 for i in 1:mp_n]
const mp_V  = @. 2.0 * exp(-(mp_xs - 5.0)^2 / 4.0) + 0.15 * mp_xs
const mp_layout = GriddedLayout(:x => GriddedContinuous(mp_xs))
const mp_θmax   = 3.0
const mp_ic     = argmin(abs.(mp_xs .- 5.0))     # central: the ±8θ window clears both edges at small θ

mp_stage(; cost=(θ; env) -> env.λ * θ^2) =
    MeanPreservingSpreadStage(mp_layout; axis=:x, θ_max=mp_θmax, cost)

# Independent brute oracle: fine-trapezoid ∫Ṽ(t)φ_θ(t−x)dt over the grid + exact clamp tails —
# a different code path from the package's closed-form Φ/φ segment sweep.
Φor(z) = erfc(-z / sqrt(2)) / 2
φor(z) = exp(-z^2 / 2) / sqrt(2π)
function lin_or(t)
    j = clamp(searchsortedlast(mp_xs, t), 1, mp_n - 1)
    return mp_V[j] + (mp_V[j+1] - mp_V[j]) * (t - mp_xs[j]) / (mp_xs[j+1] - mp_xs[j])
end
function A_brute(x, θ; N=100_000)
    θ == 0 && return lin_or(x)
    ts = range(mp_xs[1], mp_xs[end]; length=N); dt = step(ts)
    acc = sum(lin_or(t) * φor((t - x) / θ) / θ for t in ts) * dt
    acc -= 0.5 * dt * (lin_or(mp_xs[1]) * φor((mp_xs[1] - x) / θ) / θ +
                       lin_or(mp_xs[end]) * φor((mp_xs[end] - x) / θ) / θ)
    return acc + mp_V[1] * Φor((mp_xs[1] - x) / θ) + mp_V[end] * (1 - Φor((mp_xs[end] - x) / θ))
end

"""
Brute `θ*`: a coarse global scan (the objective is multimodal), golden-section refine on the fine
quadrature, then the EXACT `θ = 0` value. The quadrature degenerates for σ below the trapezoid step,
so the refine bracket is floored at 0.005 and the boundary taken analytically.
"""
function brute_argmax(x, λ)
    gθ(θ, N) = A_brute(x, θ; N) - λ * θ^2
    θs   = 0.0:0.01:mp_θmax
    best = argmax([gθ(θ, 8_000) for θ in θs])
    lo, hi = max(θs[best] - 0.01, 0.005), min(θs[best] + 0.01, mp_θmax)
    gr = (sqrt(5) - 1) / 2
    a, b = lo, hi
    c = b - gr * (b - a); d = a + gr * (b - a)
    fc = gθ(c, 100_000); fd = gθ(d, 100_000)
    for _ in 1:50
        if fc > fd
            b, d, fd = d, c, fc
            c = b - gr * (b - a); fc = gθ(c, 100_000)
        else
            a, c, fc = c, d, fd
            d = a + gr * (b - a); fd = gθ(d, 100_000)
        end
    end
    θb = (a + b) / 2
    vb = gθ(θb, 100_000)
    v0 = A_brute(x, 0.0)                           # exact boundary value (no quadrature)
    return v0 >= vb ? (0.0, v0) : (θb, vb)
end

@testset "mean-preserving spread — value/policy match the brute-quadrature oracle" begin
    prim = mp_stage()
    Vs   = backward!(prim, mp_V, (; λ = 0.02))
    θ    = policy(prim)
    for i in (mp_ic, 17, 50)                       # central, near-edge (multimodal), upper-flank
        θb, vb = brute_argmax(mp_xs[i], 0.02)
        @test abs(θ[i] - θb) ≤ 1e-6
        @test abs(Vs[i] - vb) ≤ 1e-9
    end
end

@testset "mean-preserving spread — interior mean preservation and linear-V identity" begin
    # The interior row is exactly mean-preserving: gathering the grid itself gives back x.
    for θ in (0.1, 0.4)
        @test HS._gs_gather_cell(mp_xs, mp_xs[mp_ic], θ, mp_xs) ≈ mp_xs[mp_ic] atol=1e-12
    end
    # Linear V ⇒ interior A(θ) ≡ V(x) ⇒ θ* = 0.0 exactly under any increasing cost, value exact.
    Vlin = @. 0.3 + 0.15 * mp_xs
    @test HS._gs_gather_cell(Vlin, mp_xs[mp_ic], 0.4, mp_xs) ≈ Vlin[mp_ic] atol=1e-12
    # θ = 0 is the deterministic landing at `x`: exactly the node when `x` is one, the two-node Young
    # split between the bracketing nodes when it is not, and the lone node on a one-node grid.
    xoff = 0.5 * (mp_xs[mp_ic] + mp_xs[mp_ic + 1])
    @test HS._gs_gather_cell(Vlin, mp_xs[mp_ic], 0.0, mp_xs) == Vlin[mp_ic]
    @test HS._gs_gather_cell(Vlin, xoff, 0.0, mp_xs) ≈ 0.3 + 0.15 * xoff atol=1e-12
    @test HS._gs_gather_cell([7.0], 2.0, 0.0, [5.0]) == 7.0
    out1 = zeros(1); HS._gs_point_scatter!(out1, 0.4, 2.0, [5.0])
    @test out1 == [0.4]
    prim = mp_stage()
    Vs   = backward!(prim, Vlin, (; λ = 0.01))
    @test policy(prim)[mp_ic] === 0.0
    @test Vs[mp_ic] == Vlin[mp_ic]                 # cost(0) = 0 ⇒ V_start = V(x) exactly
end

@testset "mean-preserving spread — FOC residual at interior optima" begin
    prim = mp_stage()
    backward!(prim, mp_V, (; λ = 0.02))
    θ = policy(prim)
    for i in (mp_ic, 17)
        0.0 < θ[i] < mp_θmax || continue
        d1, _ = HS._mps_derivs(mp_V, mp_xs[i], θ[i], mp_xs)
        @test abs(d1 - 2 * 0.02 * θ[i]) ≤ 1e-10 * max(1.0, abs(d1))
    end
end

@testset "mean-preserving spread — θ*(λ) smooth, IFT slope matches FD" begin
    prim = mp_stage()
    θat(λ) = (backward!(prim, mp_V, (; λ)); policy(prim)[17])
    λs = collect(0.016:0.001:0.029)                # interior range at cell 17 (prototype ladder)
    θs = θat.(λs)
    @test all(0.0 .< θs .< mp_θmax)
    @test maximum(abs.(diff(θs))) < 0.1            # no jump: |dθ*/dλ| ≈ 38 ⇒ steps ≈ 0.04
    λ0 = 0.02
    θ0 = θat(λ0)
    _, d2 = HS._mps_derivs(mp_V, mp_xs[17], θ0, mp_xs)
    ift = 2θ0 / (d2 - 2λ0)                         # dθ*/dλ = −g_θλ/g_θθ
    h  = 1e-5
    fd = (θat(λ0 + h) - θat(λ0 - h)) / (2h)
    @test abs(ift - fd) ≤ 1e-4 * abs(fd)
end

@testset "mean-preserving spread — envelope-exact Dual value tangent, policy at the buffer eltype" begin
    prim   = mp_stage()
    backward!(prim, mp_V, (; λ = 0.02))
    θp     = copy(policy(prim))                    # the primal solve's policy
    dstage = lift_jacobian(prim)
    Dm  = Dual{tagtype(eltype(V_start_buffer(dstage)))}          # the lift owns the tag
    Vd  = Dm.(mp_V, 0.0)
    Vsd = backward!(dstage, Vd, (; λ = Dm(0.02, 1.0)))
    θ   = policy(dstage)
    @test eltype(Vsd) <: Dual
    @test eltype(θ) == eltype(Vsd)                 # the policy takes the BUFFER eltype
    @test all(value.(θ) .=== θp)                   # …and its values are the primal policy, bitwise
    @test all(partials.(Vsd, 1) .≈ -θp .^ 2)       # ∂V/∂λ = −θ*² at the primal θ* (exact)
    # Envelope theorem: the value tangent equals the FD of the RE-SOLVED value (θ* re-optimised) —
    # exact to first order at interior optima and trivially at bounds (policy locally constant).
    h = 1e-5
    p1 = mp_stage(); p2 = mp_stage()
    fdV = (backward!(p1, mp_V, (; λ = 0.02 + h)) .- backward!(p2, mp_V, (; λ = 0.02 - h))) ./ (2h)
    for i in (mp_ic, 17, 50)
        @test abs(fdV[i] - partials(Vsd[i], 1)) ≤ 1e-4
    end
end

@testset "mean-preserving spread — envelope vs reoptimize, both directions" begin
    # The seated θ* carries the IFT tangent −g_θλ/g_θθ; both channels are held to a central
    # difference of the RE-SOLVED primal. The policy bar is Newton-tolerance-limited and the value
    # bar is not, so they are stated separately; both gaps fall as h² down to this `h`, which is
    # what says the residual is FD truncation rather than a discrepancy.
    build() = (; stage = mp_stage(), V_end = mp_V, env = (; λ = 0.02))
    p4 = envelope_vs_reoptimize(build; mode = :env, direction = (; λ = 1.0), h = 1e-6, rtol = 1e-5,
                                label = "MeanPreservingSpread")
    p3 = envelope_vs_reoptimize(build; mode = :V_end, direction = cos.(0.7 .* (1:mp_n)), h = 1e-6,
                                rtol = 1e-5, label = "MeanPreservingSpread")
    for p in (p4, p3)
        # The value bar is ABSOLUTE: FD truncation scales with `h²·|V‴|`, not with the size of the
        # derivative, so normalising by `max|fd|` would bind hardest where the tangent is smallest.
        @test maximum(abs, p.value_ad .- p.value_fd) ≤ 1e-7
        @test maximum(abs, p.policy_ad) > 1e-3     # a genuine channel: a primal-seated θ* reports 0
    end
    # Bound cells are locally constant, so their tangent is EXACTLY zero and the FD confirms it.
    prim  = mp_stage(); backward!(prim, mp_V, (; λ = 0.02))
    bound = findall(t -> t == 0.0 || t == mp_θmax, policy(prim))
    @test !isempty(bound)
    @test all(p4.policy_ad[bound] .== 0.0)
    @test all(abs.(p4.policy_fd[bound]) .≤ 1e-9)
end

@testset "mean-preserving spread — distribution channel: ∂Λ_end/∂λ through the seated row" begin
    dstage = lift_jacobian(mp_stage())
    Dm = Dual{tagtype(eltype(V_start_buffer(dstage)))}
    backward!(dstage, Dm.(mp_V, 0.0), (; λ = Dm(0.02, 1.0)))
    Λ0   = fill(1 / mp_n, mp_n)
    Λ_ad = partials.(copy(forward!(dstage, Dm.(Λ0, 0.0))), 1)
    leg(h) = (s = mp_stage(); backward!(s, mp_V, (; λ = 0.02 + h)); copy(forward!(s, Λ0)))
    h    = 1e-5
    Λ_fd = (leg(h) .- leg(-h)) ./ (2h)
    @test maximum(abs, Λ_ad .- Λ_fd) ≤ 1e-5 * maximum(abs, Λ_fd)
    @test maximum(abs, Λ_ad) > 1e-4                # the spread genuinely moves mass
    @test abs(sum(Λ_ad)) ≤ 1e-12                   # rows stay stochastic ⇒ the tangent has zero mass
end

@testset "mean-preserving spread — Integer and Bool env entries freeze, and the rest still seeds" begin
    # An `Integer` and a `Bool` env entry are ordinary model inputs (a table index, a regime switch):
    # the probe reads them at their primal, since a `Dual` there would land in an index or a branch.
    tbl = [0.02, 0.05, 0.10]
    mkix() = mp_stage(cost = (θ; env) -> env.on ? tbl[env.k] * env.λ * θ^2 : 0.0)
    dstage = lift_jacobian(mkix())
    Dm  = Dual{tagtype(eltype(V_start_buffer(dstage)))}
    backward!(dstage, Dm.(mp_V, 0.0), (; k = 2, on = true, λ = Dm(1.0, 1.0)))
    θ_ad = partials.(copy(policy(dstage)), 1)
    h = 1e-6
    leg(s) = (st = mkix(); backward!(st, mp_V, (; k = 2, on = true, λ = 1.0 + s)); copy(policy(st)))
    θ_fd = (leg(h) .- leg(-h)) ./ 2h
    @test maximum(abs, θ_ad) > 1e-3                # the frozen index still leaves a live λ channel
    @test maximum(abs, θ_ad .- θ_fd) ≤ 1e-5 * maximum(abs, θ_fd)
end

@testset "mean-preserving spread — a tangent hidden in a non-scalar env entry is refused" begin
    # A `Dual` inside a nested `NamedTuple` or an array reaches no probe lane, so its direction would
    # come back zero while the value channel stayed envelope-exact and hid the loss: it is refused by
    # name. A non-scalar entry with no tangent in it is not a hidden tangent, and both lanes run.
    nst = lift_jacobian(mp_stage(cost = (θ; env) -> env.p.λ * θ^2))
    Dn  = Dual{tagtype(eltype(V_start_buffer(nst)))}
    @test_throws ErrorException backward!(nst, Dn.(mp_V, 0.0), (; p = (; λ = Dn(0.02, 1.0))))
    ast = lift_jacobian(mp_stage(cost = (θ; env) -> env.v[1] * θ^2))
    Da  = Dual{tagtype(eltype(V_start_buffer(ast)))}
    @test_throws ErrorException backward!(ast, Da.(mp_V, 0.0), (; v = Da.([0.02], 1.0)))
    @test all(isfinite, backward!(mp_stage(cost = (θ; env) -> env.p.λ * θ^2), mp_V, (; p = (; λ = 0.02))))
    fst = lift_jacobian(mp_stage(cost = (θ; env) -> env.v[1] * env.λ * θ^2))
    Df  = Dual{tagtype(eltype(V_start_buffer(fst)))}
    @test all(isfinite, value.(backward!(fst, Df.(mp_V, 0.0), (; v = [1.0], λ = Df(0.02, 1.0)))))
end

@testset "mean-preserving spread — the seat's gates: interiority, concavity, finiteness" begin
    # Only a strictly concave, finite curvature at a strictly interior optimum licenses the IFT
    # tangent: at `g_θθ ≥ 0` the stationary point is not a maximum of the frozen objective, and
    # −n/g_θθ would point away from it.
    D = Dual{HS.HhsLiftTag, Float64, 1}
    seat(n, gθθ; θf = 0.5) = HS._ift_seat(D, θf, 0.0, 1.0, k -> n, () -> gθθ)
    @test partials(seat(1.0, -2.0), 1) === 0.5     # concave interior: the IFT tangent −n/g_θθ
    @test partials(seat(1.0, 2.0), 1) === 0.0      # convex: no attach, exactly zero
    @test partials(seat(1.0, 0.0), 1) === 0.0      # flat: likewise
    @test partials(seat(1.0, NaN), 1) === 0.0
    @test partials(seat(NaN, -2.0), 1) === 0.0
    @test partials(seat(1.0, -2.0; θf = 0.0), 1) === 0.0        # at a bound: locally constant
end

@testset "mean-preserving spread — private probe tags; nested lifts are refused" begin
    # Every tag the package manufactures is a private singleton, so the four are pairwise distinct
    # TYPES. Built from a Symbol they would all be `Tag{Symbol, Float64}` — the symbol erased.
    @test length(unique((HS._MPS_TAG_IN, HS._MPS_TAG_OUT, HS.HhsLiftTag, HS.FakeNewsTag))) == 4
    nested = Dual{HS.HhsLiftTag, Dual{HS.HhsLiftTag, Float64, 1}, 1}
    @test_throws ErrorException with_eltype(mp_stage(), nested)
end

@testset "mean-preserving spread — concave insures/convex gambles exactly; clamp option value" begin
    Vcc = @. -(mp_xs - 5.0)^2
    Vcv = @. (mp_xs - 5.0)^2
    prim = mp_stage()
    backward!(prim, Vcc, (; λ = 0.005))
    @test policy(prim)[mp_ic] === 0.0              # concave central cell: exact bound return
    @test policy(prim)[17] > 0.0                   # near-edge concave cell SPREADS: clamp option value
    prim0 = MeanPreservingSpreadStage(mp_layout; axis=:x, θ_max=mp_θmax)   # zero cost
    backward!(prim0, Vcv, NamedTuple())
    @test policy(prim0)[mp_ic] === mp_θmax         # convex central cell: exact bound return
end

@testset "mean-preserving spread — duality, mass conservation, adjoints, Dual shift" begin
    prim0 = MeanPreservingSpreadStage(mp_layout; axis=:x, θ_max=mp_θmax)   # zero cost
    V_start = copy(backward!(prim0, mp_V, NamedTuple()))
    Λ_start = abs.(sin.(1.0:mp_n)) .+ 0.1; Λ_start ./= sum(Λ_start)
    Λ_end   = forward!(prim0, Λ_start)
    @test sum(Λ_end) ≈ sum(Λ_start)                              # MPS is mass-conserving
    # Continuous-max value = seated-operator value ⇒ the duality identity is exact (zero cost).
    @test isapprox(sum(V_start .* Λ_start), sum(mp_V .* Λ_end); rtol=1e-12)
    dV = sin.(2.0:(mp_n + 1)); dΛ = cos.(1.0:mp_n)
    @test sum(forward_adjoint!(prim0, dΛ) .* dV) ≈ sum(dΛ .* backward_adjoint!(prim0, dV))
    # Rows are stochastic ⇒ a uniform V_end shift moves V_start one-for-one.
    dstage = lift_jacobian(prim0)
    Dm = Dual{tagtype(eltype(V_start_buffer(dstage)))}
    Vs = backward!(dstage, Dm.(mp_V, 1.0), NamedTuple())
    @test all(partials.(Vs, 1) .≈ 1.0)
end

@testset "mean-preserving spread — value continuous across basin ties (all-local-max refine)" begin
    # Manufactured interior basins via a wiggly cost on a FLAT V (rows stochastic ⇒ A(θ) ≡ 1, so
    # g(θ) = 1 − cost(θ) is fully controlled): narrow wells OFF the scan nodes create competing
    # local maxima whose refined values tie as the well depth `d` sweeps. Refining only the sampled
    # winner jumps ~4e-2 at the tie (the MW VFI limit-cycle mechanism); refining every scan-local
    # maximum keeps the value continuous — |dv/dd| ≈ 1, so steps of 5e-4 must move v by ≈ 5e-4.
    nb = 101; xb = collect(range(0.0, 10.0; length=nb)); ib = 51
    Vflat = fill(1.0, nb)
    well(θ, t) = exp(-(θ - t)^2 / 0.02)
    mk2(d) = (θ; env) -> 0.001 * θ^2 - 0.30 * well(θ, 1.11) - d * well(θ, 2.13)
    mk3(d) = (θ; env) -> 0.001 * θ^2 - 0.298 * well(θ, 0.36) - 0.30 * well(θ, 1.11) - d * well(θ, 2.13)
    for mk in (mk2, mk3)                           # two-basin tie, then the three-basin stress
        prev = NaN; prevθ = NaN; worst = 0.0; flips = 0; disc = 0.0
        for d in 0.28:5e-4:0.32
            v, θ = HS._mps_solve_cell(Vflat, xb[ib], xb, mk(d), (;), (;), 3.0, 12, Float64)
            vr, _ = HS._mps_solve_cell(Vflat, xb[ib], xb, mk(d), (;), (;), 3.0, 400, Float64)
            disc = max(disc, abs(v - vr))
            isnan(prev) || (worst = max(worst, abs(v - prev)))
            isnan(prevθ) || (abs(θ - prevθ) > 0.3 && (flips += 1))
            prev = v; prevθ = θ
        end
        @test flips ≥ 1                            # the sweep genuinely crosses a global-argmax switch
        @test worst ≤ 2e-3                         # no value jump at the tie (winner-only: ~4e-2)
        @test disc ≤ 1e-7                          # agrees with a dense-scan reference throughout
    end
end

@testset "mean-preserving spread — -Inf robustness (hardened solver)" begin
    Vinf = copy(mp_V); Vinf[3] = -Inf
    prim = mp_stage()
    for λ in (1e-4, 5e-3, 2e-2)                    # 1e-4 forces Newton brackets straddling band entry
        Vs = backward!(prim, Vinf, (; λ))
        θ  = policy(prim)
        @test !any(isnan, Vs)
        @test all(isfinite, Vs[[1:2; 4:mp_n]])     # finite cells stay finite (θ keeps -Inf out of band)
        @test Vs[3] == -Inf && θ[3] === 0.0        # own-node -Inf: infeasible cell convention
        @test all(0.0 .≤ θ .≤ mp_θmax)
    end
end

end # module
