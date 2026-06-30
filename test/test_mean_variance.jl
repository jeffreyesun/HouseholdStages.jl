using Test
using HouseholdStages

# MeanVarianceStage: streaming portfolio choice (rung d) on the same core as ScaleVarianceStage —
# next wealth w·(R_f + θ·(R_k−R_f)). Risk-neutral V ⇒ max share (chase the premium); concave V ⇒
# interior share (risk–return tradeoff). Value matches a brute max over shares; duality holds.

ws      = collect(0.5:0.25:8.0); nw = length(ws)
shares  = [0.0, 0.25, 0.5, 0.75, 1.0]
Rf      = 1.02
Rrisky  = [0.7, 1.4]                              # mean-1.05 risky asset (premium 0.03 over R_f), real variance
probs   = [0.5, 0.5]
layout  = GriddedLayout(:wealth => GriddedContinuous(ws))

# Brute reference using the stage's own clamp+interp landing.
function brute(V, cost)
    interp(t) = (tc = clamp(t, ws[1], ws[nw]); j = min(searchsortedlast(ws, tc), nw - 1);
                 w = (tc - ws[j]) / (ws[j+1] - ws[j]); (1 - w) * V[j] + w * V[j+1])
    [maximum(sum(probs[k] * interp(ws[i] * (Rf + θ * (Rrisky[k] - Rf))) for k in eachindex(probs)) - cost(θ)
             for θ in shares) for i in 1:nw]
end

@testset "mean-variance — value matches brute max over shares" begin
    stage = MeanVarianceStage(layout; axis = :wealth, shares = shares,
                              risk_free = Rf, risky_returns = Rrisky, probs = probs)
    V = @. sqrt(ws)
    @test backward!(stage, V, NamedTuple()) ≈ brute(V, _ -> 0.0)
end

# In-grid central band: even the max-share up-return stays on the grid (no clamping artifacts).
mid = findall(i -> ws[i] * (Rf + maximum(Rrisky .- Rf)) ≤ ws[nw] && i > 2, 1:nw)

@testset "mean-variance — risk-neutral chases the premium (θ*=max)" begin
    stage = MeanVarianceStage(layout; axis = :wealth, shares = shares,
                              risk_free = Rf, risky_returns = Rrisky, probs = probs)
    backward!(stage, collect(ws), NamedTuple())              # linear (risk-neutral) V
    @test all(policy(stage)[mid] .== maximum(shares))        # positive premium ⇒ go all-risky
end

@testset "mean-variance — risk aversion pulls to an interior share" begin
    stage = MeanVarianceStage(layout; axis = :wealth, shares = shares,
                              risk_free = Rf, risky_returns = Rrisky, probs = probs)
    backward!(stage, (@. sqrt(ws)), NamedTuple())            # concave (risk-averse) V
    θ = policy(stage)[mid]
    @test all(0.0 .< θ .< maximum(shares))                   # strictly interior
    @test all(θ .== θ[1])                                    # CRRA + multiplicative returns ⇒ share independent of wealth
end

@testset "mean-variance — duality and mass conservation" begin
    stage   = MeanVarianceStage(layout; axis = :wealth, shares = shares,
                                risk_free = Rf, risky_returns = Rrisky, probs = probs)
    V       = @. sqrt(ws)
    V_start = backward!(stage, V, NamedTuple())
    Λ_start = abs.(randn(nw)); Λ_start ./= sum(Λ_start)
    Λ_end   = forward!(stage, Λ_start)
    @test sum(Λ_end) ≈ sum(Λ_start)                          # portfolio move conserves mass
    @test sum(V_start .* Λ_start) ≈ sum(V .* Λ_end)          # zero cost ⇒ ⟨V_start,Λ⟩ = ⟨V_end,Λ_end⟩
end
