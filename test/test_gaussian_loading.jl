using Test
using HouseholdStages

# GaussianLoadingStage: the CONTINUOUS loading-θ primitive over the truncated-Gaussian increment
# family (GaussianLoadingKernel) — next coordinate t = w·(anchor + θ·(μ + σZ)), Z ~ ±8σ-truncated
# N(0,1), solved per cell by the shared scan+Newton recipe. Exercised here in its canonical
# portfolio reading (anchor = risk-free rate, increment = excess return, θ = risky share), which is
# what supplies the closed forms the primal is held to: exact endpoint returns wherever the premium's
# sign settles the choice, and the Merton/myopic interior share, wealth-independent under CRRA on an
# exactly-geometric grid. The tangent channels are held to central differences of the RE-SOLVED
# primal, in both the ways env reaches the objective — a cost parameter and a FromEnv increment
# primitive. Wrapped in a module so its configuration globals don't leak into the other tests.

module GaussianLoadingTest
using Test, HouseholdStages
using HouseholdStages.ForwardDiff: Dual, value, partials, tagtype
const HS = HouseholdStages
include("envelope_oracle.jl")

# Exactly-geometric wealth grid: with CRRA V and multiplicative returns the per-cell objective is
# A_i(θ) = w_i^(1-γ)·f(θ) with f IDENTICAL across interior cells (band nodes sit at the same
# multiplicative offsets), so θ* is wealth-independent up to solver tolerance — the Merton anchor.
const gl_n  = 80
const gl_ws = exp.(range(log(0.1), log(40.0); length = gl_n))
const gl_layout = GriddedLayout(:wealth => GriddedContinuous(collect(gl_ws)))
const gl_rf = 1.0
const gl_μ  = 0.03
const gl_σ  = 0.2
const gl_γ  = 3.0
const gl_Vcrra = @. gl_ws^(1 - gl_γ) / (1 - gl_γ)      # CRRA continuation, γ = 3

gl_stage(; rf = gl_rf, μ = gl_μ, σ = gl_σ, cost = (θ; env) -> 0.0) =
    GaussianLoadingStage(gl_layout; axis = :wealth, anchor = rf,
                      increment_mean = μ, increment_sd = σ, cost)

# Interior mid cells: at the myopic share (≈ 0.25) the full ±8σ band [m − 8s, m + 8s] clears both
# grid edges, and at θ_hi = 1 the up-band w·(rf + μ + 8σ) ≈ 2.63w stays under the top node — so
# no clamp mass contaminates the endpoint or interior tests on these cells.
const gl_mid = findall(w -> 0.5 <= w <= 12.0, gl_ws)

# The TOP cell is the one the derivative oracles leave out: `w·rf` is the top node exactly, so the
# primal itself kinks in `rf` there and both channels report the average of two one-sided slopes.
const gl_kinkfree = 1:(gl_n - 1)

@testset "gaussian loading — exact endpoints: risk-neutral chases the premium, μ<0 shuns it" begin
    Vlin = collect(gl_ws)                          # linear (risk-neutral) continuation
    up = gl_stage()                                # μ > 0: A′(θ_hi) ≥ 0 ⇒ ε-probe returns EXACT θ_hi
    backward!(up, Vlin, NamedTuple())
    @test all(policy(up)[gl_mid] .== 1.0)
    dn = gl_stage(μ = -gl_μ)                       # μ < 0: A′(θ_lo⁺) ≤ 0 ⇒ ε-probe returns EXACT θ_lo
    Vs = backward!(dn, Vlin, NamedTuple())
    @test all(policy(dn)[gl_mid] .== 0.0)
    # θ* = 0 with rf = 1 is the deterministic on-node landing: zero cost ⇒ V_start = V_end exactly.
    @test Vs[gl_mid] == Vlin[gl_mid]
end

@testset "gaussian loading — Merton interior share, wealth-independent under CRRA" begin
    prim = gl_stage()
    backward!(prim, gl_Vcrra, NamedTuple())
    θm = policy(prim)[gl_mid]
    @test all(0.0 .< θm .< 1.0)                    # strictly interior (risk–return tradeoff binds)
    @test all(abs.(θm ./ θm[1] .- 1.0) .< 0.02)    # wealth-independence (loose: interp phase noise)
    @test 0.15 < θm[1] < 0.40                      # myopic band around μ/(γσ²) = 0.25
end

@testset "gaussian loading — FOC residual at interior optima" begin
    ic = argmin(abs.(gl_ws .- 2.0))
    grid = collect(gl_ws); h = 1e-5
    # Zero cost: central FD of the frozen objective A(θ) at the solved θ*.
    prim = gl_stage()
    backward!(prim, gl_Vcrra, NamedTuple())
    θ0 = policy(prim)[ic]
    A(θ) = HS._gl_A(gl_Vcrra, gl_ws[ic], θ, gl_rf, gl_μ, gl_σ, grid, HS._frz)
    @test 0.0 < θ0 < 1.0
    @test abs((A(θ0 + h) - A(θ0 - h)) / (2h)) <= 1e-6
    # Quadratic cost: FD of A(θ) − λθ² at the (shifted, still interior) θ*.
    λ0 = 5e-4
    prc = gl_stage(cost = (θ; env) -> λ0 * θ^2)
    backward!(prc, gl_Vcrra, NamedTuple())
    ic2 = argmin(abs.(gl_ws .- 0.5))               # cost breaks scale invariance: pick a low-w cell
    θc  = policy(prc)[ic2]
    g(θ) = HS._gl_A(gl_Vcrra, gl_ws[ic2], θ, gl_rf, gl_μ, gl_σ, grid, HS._frz) - λ0 * θ^2
    @test 0.0 < θc < 1.0
    @test abs((g(θc + h) - g(θc - h)) / (2h)) <= 1e-6
end

@testset "gaussian loading — duality, mass conservation, adjoints, edge piling" begin
    prim = gl_stage()                              # zero cost
    V_start = copy(backward!(prim, gl_Vcrra, NamedTuple()))
    Λ_start = abs.(sin.(1.0:gl_n)) .+ 0.1; Λ_start ./= sum(Λ_start)
    Λ_end   = forward!(prim, Λ_start)
    @test sum(Λ_end) ≈ sum(Λ_start) rtol = 1e-12   # Gaussian rows are stochastic ⇒ mass conserved
    # Zero cost ⇒ V_start IS the seated gather Kᵀ_{θ*}V_end, and forward is its exact transpose.
    @test isapprox(sum(V_start .* Λ_start), sum(gl_Vcrra .* Λ_end); rtol = 1e-12)
    dV = sin.(2.0:(gl_n + 1)); dΛ = cos.(1.0:gl_n)
    @test sum(forward_adjoint!(prim, dΛ) .* dV) ≈ sum(dΛ .* backward_adjoint!(prim, dV))
    # Boundary piling: all mass on the top cell, whose landing straddles the top node — the
    # off-grid remainder piles at the edge (the package-wide clamp convention), none is lost.
    up = gl_stage()
    backward!(up, collect(gl_ws), NamedTuple())    # risk-neutral: θ* = 1 drives real spread
    Λtop = zeros(gl_n); Λtop[end] = 1.0
    Λe   = forward!(up, Λtop)
    @test sum(Λe) ≈ 1.0 atol = 1e-12
    @test Λe[end] > 0.5                            # mean 40·(1+θ*μ) ≥ top node ⇒ pile at the edge
end

@testset "gaussian loading — envelope-exact Dual tangents, policy at the buffer eltype" begin
    λ0 = 5e-4; h = 1e-6
    mk() = gl_stage(cost = (θ; env) -> env.λ * θ^2)
    prim = mk()
    backward!(prim, gl_Vcrra, (; λ = λ0))
    θ = copy(policy(prim))
    dstage = lift_jacobian(prim)
    Dg  = Dual{tagtype(eltype(V_start_buffer(dstage)))}          # the lift owns the tag
    Vd  = Dg.(gl_Vcrra, 0.0)
    Vsd = backward!(dstage, Vd, (; λ = Dg(λ0, 1.0)))
    @test eltype(Vsd) <: Dual
    @test eltype(policy(dstage)) == eltype(Vsd)    # θ* takes the BUFFER eltype
    @test all(value.(policy(dstage)) .=== θ)       # …and its values are the primal policy, bitwise
    @test all(partials.(Vsd, 1) .≈ -θ .^ 2)        # ∂V/∂λ = −θ*² at the primal θ* (exact algebra)
    # Envelope theorem: the frozen tangent equals the FD of the RE-SOLVED (θ* re-optimised) value.
    p1 = mk(); p2 = mk()
    fdV = (backward!(p1, gl_Vcrra, (; λ = λ0 + h)) .-
           backward!(p2, gl_Vcrra, (; λ = λ0 - h))) ./ (2h)
    for i in (gl_mid[1], argmin(abs.(gl_ws .- 0.5)), argmin(abs.(gl_ws .- 2.0)))
        @test abs(fdV[i] - partials(Vsd[i], 1)) <= 1e-5
    end

    # FromEnv anchor (the risk-free rate): the increment primitive itself carries the tangent, and
    # the kernel is reseated with the LIVE resolved primitive each backward (GE asset-pricing entry
    # point), so the forward row carries it too.
    mkr() = GaussianLoadingStage(gl_layout; axis = :wealth, anchor = FromEnv(:rf),
                              increment_mean = gl_μ, increment_sd = gl_σ)
    hr = 1e-5
    rstage = mkr()
    backward!(rstage, gl_Vcrra, (; rf = gl_rf))
    drstage = lift_jacobian(rstage)
    Dr  = Dual{tagtype(eltype(V_start_buffer(drstage)))}
    Vrd = backward!(drstage, Dr.(gl_Vcrra, 0.0), (; rf = Dr(gl_rf, 1.0)))
    @test eltype(Vrd) <: Dual
    @test eltype(policy(drstage)) == eltype(Vrd)
    @test value(drstage.kernel.anchor) === gl_rf      # seated primitive at the resolved value…
    @test partials(drstage.kernel.anchor, 1) === 1.0  # …carrying env's tangent
    r1 = mkr(); r2 = mkr()
    fdR = (backward!(r1, gl_Vcrra, (; rf = gl_rf + hr)) .-
           backward!(r2, gl_Vcrra, (; rf = gl_rf - hr))) ./ (2hr)
    for i in (gl_mid[1], argmin(abs.(gl_ws .- 2.0)), argmin(abs.(gl_ws .- 8.0)))
        @test abs(fdR[i] - partials(Vrd[i], 1)) <= 1e-5 * max(1.0, abs(fdR[i]))
    end
end

@testset "gaussian loading — envelope vs reoptimize, both directions and both env channels" begin
    # `rf` is a FromEnv increment primitive and `λ` a cost parameter, so the two `:env` calls seed
    # the two ways env reaches the objective; `:V_end` seeds the continuation. `h = 1e-7` is the
    # joint optimum of a 1e-4…1e-8 scan (PLAN1_BUILD_LOG, WP8 §CF-15). Cells pinned at a bound are
    # compared too: their tangent is the bound's exact zero and the difference agrees.
    rows = gl_kinkfree
    build() = (; stage = GaussianLoadingStage(gl_layout; axis = :wealth, anchor = FromEnv(:rf),
                                              increment_mean = gl_μ, increment_sd = gl_σ,
                                              cost = (θ; env) -> env.λ * θ^2),
                 V_end = gl_Vcrra, env = (; rf = gl_rf, λ = 5e-4))
    oracle(mode, direction, label) =
        envelope_vs_reoptimize(build; mode, direction, h = 1e-7, rtol = 1e-5,
                               policy_of = st -> policy(st)[rows], readout = V -> V[rows], label)
    prf = oracle(:env, (; rf = 1.0), "GaussianLoading anchor")
    pλ  = oracle(:env, (; λ = 1.0), "GaussianLoading cost")
    pV  = oracle(:V_end, cos.(0.7 .* (1:gl_n)), "GaussianLoading V_end")
    for p in (prf, pλ, pV)
        # The value bar is ABSOLUTE: at this h the gap is cancellation, ~ε·|V|/h, scaling with
        # neither the tangent's size nor h².
        @test maximum(abs, p.value_ad .- p.value_fd) ≤ 3e-7
        @test maximum(abs, p.policy_ad) > 1e-3     # a genuine channel: a primal-seated θ* reports 0
    end
end

@testset "gaussian loading — bound cells seat EXACTLY zero partials (S11)" begin
    # A bound cell's θ* is locally constant, so its tangent is exactly zero — and that exactness is
    # what keeps `_gl_A`'s lexicographic `s = |x|·θ·σ > 0` test safe: a seated θ of value 0 with a
    # live partial passes it and divides the banded row by zero.
    dstage = lift_jacobian(gl_stage(cost = (θ; env) -> env.λ * θ^2))
    Dg  = Dual{tagtype(eltype(V_start_buffer(dstage)))}
    backward!(dstage, Dg.(gl_Vcrra, 0.0), (; λ = Dg(5e-4, 1.0)))
    θd  = policy(dstage)
    bnd = findall(t -> value(t) == 0.0 || value(t) == 1.0, θd)
    int = findall(t -> 0.0 < value(t) < 1.0, θd)
    @test !isempty(bnd) && !isempty(int)           # the fixture exercises both
    @test all(partials.(θd[bnd], 1) .=== 0.0)      # exactly zero, not merely small
    @test all(partials.(θd[int], 1) .!= 0.0)
    Λ = forward!(dstage, Dg.(fill(1 / gl_n, gl_n), 0.0))
    @test all(isfinite, value.(Λ)) && all(isfinite, partials.(Λ, 1))
end

@testset "gaussian loading — a tangent hidden in a non-scalar env entry is refused" begin
    # The guard is shared, and `test_mean_preserving_spread.jl` covers its four shapes; what this
    # asserts is that GL's own `backward!` is wired to it.
    dstage = lift_jacobian(gl_stage(cost = (θ; env) -> env.p.λ * θ^2))
    Dg = Dual{tagtype(eltype(V_start_buffer(dstage)))}
    @test_throws ErrorException backward!(dstage, Dg.(gl_Vcrra, 0.0), (; p = (; λ = Dg(5e-4, 1.0))))
end

@testset "gaussian loading — the seated ops stay isbits at a Dual eltype (device seam)" begin
    # The CUDA extension passes the op into the kernel by value, so a Dual-parametrized op that was
    # not isbits would fail at the seam. Asserted host-side so the suite catches it without a GPU.
    Dg = Dual{HS.HhsLiftTag, Float64, 2}
    @test isbits(HS.GlGatherOp(Dg(gl_rf), Dg(gl_μ), Dg(gl_σ)))
    @test isbits(HS.GlScatterOp(Dg(gl_rf), Dg(gl_μ), Dg(gl_σ)))
    @test_throws ErrorException with_eltype(gl_stage(), Dual{HS.HhsLiftTag, Dg, 1})
end

@testset "gaussian loading — distribution channel: ∂Λ_end/∂rf through the seated row" begin
    mkr() = GaussianLoadingStage(gl_layout; axis = :wealth, anchor = FromEnv(:rf),
                                 increment_mean = gl_μ, increment_sd = gl_σ)
    dstage = lift_jacobian(mkr())
    Dr = Dual{tagtype(eltype(V_start_buffer(dstage)))}
    backward!(dstage, Dr.(gl_Vcrra, 0.0), (; rf = Dr(gl_rf, 1.0)))
    # Mass rides on `gl_mid`: the top cell's θ* sits at the bound on a one-sided kink in `rf`, so a
    # central difference there would report mass teleporting rather than a derivative.
    Λ0 = zeros(gl_n); Λ0[gl_mid] .= 1 / length(gl_mid)
    Λ_ad = partials.(copy(forward!(dstage, Dr.(Λ0, 0.0))), 1)
    leg(h) = (s = mkr(); backward!(s, gl_Vcrra, (; rf = gl_rf + h)); copy(forward!(s, Λ0)))
    h    = 1e-6
    Λ_fd = (leg(h) .- leg(-h)) ./ (2h)
    @test maximum(abs, Λ_ad .- Λ_fd) ≤ 1e-5 * maximum(abs, Λ_fd)
    @test maximum(abs, Λ_ad) > 1e-4                # the loading genuinely moves mass
    @test abs(sum(Λ_ad)) ≤ 1e-12                   # rows stay stochastic ⇒ the tangent has zero mass
end

@testset "gaussian loading — participation cost: exact θ* = 0 nonparticipation region" begin
    # √-wealth continuation: the participation gain A(θ*) − A(0) scales like √w, so a fixed entry
    # cost F between the low- and high-wealth gains splits the grid into an exact-zero
    # nonparticipation region and a participating one.
    Vsq  = @. sqrt(gl_ws)
    grid = collect(gl_ws)
    iLo  = argmin(abs.(gl_ws .- 0.5)); iHi = argmin(abs.(gl_ws .- 10.0))
    gain(i) = HS._gl_A(Vsq, gl_ws[i], 1.0, gl_rf, gl_μ, gl_σ, grid) -
              HS._gl_A(Vsq, gl_ws[i], 0.0, gl_rf, gl_μ, gl_σ, grid)
    @test 0.0 < gain(iLo) < gain(iHi)              # √w continuation: θ* = θ_hi, gain increasing in w
    F = sqrt(gain(iLo) * gain(iHi))                # strictly between the two gains
    prim = gl_stage(cost = (θ; env) -> θ > 0 ? F : 0.0)
    Vp = backward!(prim, Vsq, NamedTuple())
    θ  = policy(prim)
    @test θ[iLo] === 0.0                           # priced out: EXACT zero, not merely small
    @test Vp[iLo] == HS._gl_A(Vsq, gl_ws[iLo], 0.0, gl_rf, gl_μ, gl_σ, grid)
    @test θ[iHi] > 0.0                             # rich enough to pay the fixed cost
    @test Vp[iHi] ≈ HS._gl_A(Vsq, gl_ws[iHi], θ[iHi], gl_rf, gl_μ, gl_σ, grid) - F
end

@testset "gaussian loading — loading bounds must be nonnegative (shorting is not a flag)" begin
    # θ < 0 would make the written sd |w|·θ·σ negative and silently route shorts to the
    # deterministic (risk-free) branch — the constructor refuses the domain outright.
    @test_throws AssertionError GaussianLoadingStage(gl_layout; anchor = gl_rf,
                                                  increment_mean = gl_μ, increment_sd = gl_σ,
                                                  loading_bounds = (-1.0, 1.0))
end

@testset "gaussian loading — deterministic (θ* = 0) forward Young-splits at w·rf" begin
    # Exercises the s == 0 scatter branch WITH mass (the suite's other forwards run at interior
    # or cap shares): seat θ* ≡ 0 directly and compare against the manual two-node split.
    stage = GaussianLoadingStage(gl_layout; anchor = gl_rf, increment_mean = gl_μ, increment_sd = gl_σ)
    fill!(stage.kernel.θstar, 0.0)                 # frozen policy: every landing is w·rf exactly
    grid = collect(gl_ws)
    n  = length(grid)
    Λ0 = fill(1.0 / n, n)
    Λ1 = copy(forward!(stage, Λ0))
    ref = zeros(n)
    for i in 1:n
        t = clamp(grid[i] * gl_rf, grid[1], grid[n])
        j = min(searchsortedlast(grid, t), n - 1)
        ω = (t - grid[j]) / (grid[j+1] - grid[j])
        ref[j] += (1 - ω) * Λ0[i]; ref[j+1] += ω * Λ0[i]
    end
    @test Λ1 ≈ ref atol = 1e-14
    @test sum(Λ1) ≈ 1.0 atol = 1e-12
end

end # module
