using Test
using HouseholdStages

# SearchMatchingStage (derived sugar, re-introduced 2026-07-29): one call returning the
# separation `MarkovStage ∘` job-search `MixingStage` pair on a 2-level `:emp` axis, with the
# entropy-family probability cost c(p) = κ·((1−p)log(1−p) + p), κ = χ/(A·θ), and its Fenchel
# argmax p*(y) = clamp(1 − exp(−y·Aθ/χ), 0, 1) single-homed in the sugar. Checked: leaf-level
# equivalence with the hand-written pair to 1e-14, policy readback through the chain (interior
# on unemployed cells, exactly 0.0 on employed), the tightness comparative static via env and
# via a scalar `tightness`, the closed-form p* against a hand-computed gap y = a − b, forward
# mass conservation, and the 2-level-axis assert. Wrapped in a module so fixture globals don't leak.

module SearchMatchingSugarTest
using Test, HouseholdStages

const sm_χ   = 0.5
const sm_A   = 0.5
const sm_δ   = 0.1
const sm_env = (; θ = 1.5)
const sm_lay = GriddedLayout(:x   => Discrete([0.5, 1.0, 2.0]),          # spectator axis
                             :emp => Discrete([:unemp, :emp]))           # 1 = unemployed, 2 = employed
const sm_W   = [1.0 4.0; 0.5 3.0; 2.0 6.0]     # V_end (x, emp): employment worth more everywhere
const sm_Λ   = [0.10 0.20; 0.15 0.25; 0.05 0.25]

sugar(; kw...) = SearchMatchingStage(sm_lay; separation = sm_δ, effort_cost_scale = sm_χ,
                                     matching_efficiency = sm_A, kw...)

"The hand-written pair the sugar replaces, at a fixed tightness θ (constant κ_s)."
function manual_chain(θ)
    κ_s = sm_χ / (sm_A * θ)
    matching = MixingStage(sm_lay; axis = :emp,
        K_A    = [0.0 1.0; 0.0 1.0],                              # search succeeds → employed
        K_B    = [1.0 0.0; 0.0 1.0],                              # search fails → stay
        cost   = (p; env) -> p >= 1 ? κ_s : κ_s * ((1 - p) * log1p(-p) + p),
        policy = (y; env) -> clamp(1 - exp(-y / κ_s), 0.0, 1.0))
    sep = MarkovStage(sm_lay; axis = :emp, transition_matrix = [1.0 0.0; sm_δ 1.0 - sm_δ])
    return sep ∘ matching
end

@testset "search-matching sugar ≡ manual MarkovStage ∘ MixingStage (1e-14)" begin
    chain = sugar()                                # default tightness reads env.θ
    ref   = manual_chain(sm_env.θ)
    @test length(chain.buffer.stages) == 2         # flat: the same two leaves as the hand-written pair
    V, Vr = backward!(chain, sm_W, sm_env), backward!(ref, sm_W, sm_env)
    @test maximum(abs, V .- Vr) < 1e-14
    Λ, Λr = forward!(chain, sm_Λ), forward!(ref, sm_Λ)
    @test maximum(abs, Λ .- Λr) < 1e-14
end

@testset "policy readback through the chain — interior p* unemployed, exactly 0.0 employed" begin
    chain = sugar()
    backward!(chain, sm_W, sm_env)
    p = policy(chain)                              # reaches the unique policy-bearing (mixing) leaf
    @test size(p) == (3, 2)
    @test all(0 .< p[:, 1] .< 1)                   # unemployed: interior search
    @test all(p[:, 2] .== 0.0)                     # employed rows coincide ⇒ degenerate, exactly 0
end

@testset "closed form p* = clamp(1 − exp(−y·Aθ/χ), 0, 1) at hand-computed y = a − b" begin
    chain = sugar()
    backward!(chain, sm_W, sm_env)
    # On the unemployed cells: a = (K_A·W)ᵤ = W[:, emp], b = (K_B·W)ᵤ = W[:, unemp].
    y = sm_W[:, 2] .- sm_W[:, 1]
    @test policy(chain)[:, 1] ≈ clamp.(1 .- exp.(-y .* sm_A .* sm_env.θ ./ sm_χ), 0.0, 1.0)
end

@testset "tightness comparative static — p* rises with env.θ; scalar tightness matches" begin
    p_loose = (c = sugar(); backward!(c, sm_W, (; θ = 0.5)); copy(policy(c)[:, 1]))
    p_tight = (c = sugar(); backward!(c, sm_W, (; θ = 2.0)); copy(policy(c)[:, 1]))
    @test all(p_tight .> p_loose)                  # tighter market ⇒ cheaper search ⇒ more of it
    scalar = sugar(tightness = 2.0)                # Real tightness: no env read at all
    backward!(scalar, sm_W, NamedTuple())
    @test policy(scalar)[:, 1] ≈ p_tight
end

@testset "mass conservation through both legs" begin
    chain = sugar()
    backward!(chain, sm_W, sm_env)                 # seat the mixing policy first
    Λ_end = forward!(chain, sm_Λ)
    @test sum(Λ_end) ≈ sum(sm_Λ) atol = 1e-14
    @test all(Λ_end .>= 0)
end

@testset "2-level axis assert — a 3-level axis throws at construction" begin
    lay3 = GriddedLayout(:emp => Discrete([:u, :e, :retired]))
    @test_throws AssertionError SearchMatchingStage(lay3; separation = sm_δ,
        effort_cost_scale = sm_χ, matching_efficiency = sm_A, tightness = 1.0)
end

end # module
