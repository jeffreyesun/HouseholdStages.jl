using Test
using HouseholdStages
using ForwardDiff

# EntryStage (forward source Λ_end = Λ + g, V identity) and the measure-half derived stage
# ReproductionStage (Λ_end = s·Λ). Both grow/shrink Λ — mass is NOT conserved by design.

@testset "EntryStage — scalar uniform inflow grows Λ; V passes through" begin
    layout = GriddedLayout(:x => Discrete([1, 2, 3]))
    stage = EntryStage(layout; entry = 0.1)

    V_end = [4.0, 1.0, 2.0]
    @test backward!(stage, V_end, nothing) == V_end          # backward identity (incumbents' V unchanged)

    Λ_start = [0.2, 0.5, 0.3]
    Λ_end = forward!(stage, Λ_start)
    @test Λ_end ≈ Λ_start .+ 0.1
    @test sum(Λ_end) ≈ sum(Λ_start) + 3 * 0.1                # mass GREW by Σg
end

@testset "EntryStage — targeted inflow via array / FromEnv / closure fields" begin
    layout = GriddedLayout(:x => Discrete([1, 2, 3]))
    Λ = [0.2, 0.5, 0.3]

    arr = EntryStage(layout; entry = [0.1, 0.2, 0.3])        # layout-shaped targeted inflow
    backward!(arr, zeros(3), nothing)
    @test forward!(arr, Λ) ≈ Λ .+ [0.1, 0.2, 0.3]

    fe = EntryStage(layout; entry = FromEnv(:g))             # entry field lives in env
    @test effective_env_slice(fe) == (:g,)
    backward!(fe, zeros(3), (g = [0.05, 0.05, 0.9],))
    @test forward!(fe, Λ) ≈ Λ .+ [0.05, 0.05, 0.9]

    cl = EntryStage(layout; entry = (; x) -> x == 1 ? 1.0 : 0.0)  # newborns enter cell 1
    backward!(cl, zeros(3), nothing)
    @test forward!(cl, Λ) ≈ Λ .+ [1.0, 0.0, 0.0]
end

@testset "EntryStage — adjoints are the identity (source constant in Λ; backward = I)" begin
    layout = GriddedLayout(:x => Discrete([1, 2, 3]))
    stage = EntryStage(layout; entry = 0.3)
    backward!(stage, zeros(3), nothing)
    dV, dΛ = randn(3), randn(3)
    @test backward_adjoint!(stage, dV) == dV
    @test forward_adjoint!(stage, dΛ) == dΛ
    # ⟨Kᵀ dΛ, dV⟩ = ⟨dΛ, K dV⟩ with both K = I.
    @test sum(forward_adjoint!(stage, dΛ) .* dV) ≈ sum(dΛ .* backward_adjoint!(stage, dV))
end

@testset "EntryStage — AD through a FromEnv scalar source (Dual mass)" begin
    layout = GriddedLayout(:x => Discrete([1, 2, 3]))
    stage = lift_jacobian(EntryStage(layout; entry = FromEnv(:m)); n_dual = 1)
    D = eltype(Λ_end_buffer(stage))
    md = D(0.2, ForwardDiff.Partials((1.0,)))
    backward!(stage, zeros(3), (m = md,))
    out = forward!(stage, [0.1, 0.1, 0.1])
    @test sum(ForwardDiff.partials(o)[1] for o in out) ≈ 3.0   # d/dm Σ(Λ + m) = n_cells = 3
end

@testset "EntryStage — in a chain, the source sustains a stationary cross-section" begin
    # Mortality (reproduction s<1) + entry: the source keeps Λ from collapsing to 0.
    ys = [1, 2]; P = [0.7 0.3; 0.3 0.7]
    layout = GriddedLayout(:income => Discrete(ys))
    s = 0.96
    shock = MarkovStage(layout; axis = :income, transition_matrix = P)
    death = ReproductionStage(layout; s = s)                 # attrition: Λ *= s
    birth = EntryStage(layout; entry = (1 - s) / 2)          # replacement inflow ⇒ mass ≈ 1
    hh = shock ∘ death ∘ birth
    hh = define_moments!(hh; pop = at_end(integrand = 1.0, reduce = sum))

    res = solve_steady_state_given_env!(hh, NamedTuple())
    @test all(res.Λ .> 0)                                    # nonzero everywhere (no collapse)
    @test isapprox(sum(res.Λ), 1.0; atol = 1e-4)             # replacement keeps total mass ≈ 1
    @test all(isfinite, res.V)
end

@testset "ReproductionStage — Λ_end = s·Λ (growth / attrition), V identity" begin
    layout = GriddedLayout(:x => Discrete([1, 2, 3]))
    Λ = [0.2, 0.5, 0.3]

    grow = ReproductionStage(layout; s = 1.1)
    @test backward!(grow, [4.0, 1.0, 2.0], nothing) == [4.0, 1.0, 2.0]   # V untouched
    @test forward!(grow, Λ) ≈ 1.1 .* Λ
    @test sum(forward!(grow, Λ)) ≈ 1.1 * sum(Λ)              # mass grew

    attr = ReproductionStage(layout; s = 0.9)
    backward!(attr, zeros(3), nothing)
    @test forward!(attr, Λ) ≈ 0.9 .* Λ                       # s < 1 ⇒ attrition

    fe = ReproductionStage(layout; s = FromEnv(:s))
    @test effective_env_slice(fe) == (:s,)
    backward!(fe, zeros(3), (s = 1.2,))
    @test forward!(fe, Λ) ≈ 1.2 .* Λ
end

@testset "Detrending is just ReproductionStage with a reciprocal factor" begin
    layout = GriddedLayout(:x => Discrete([1, 2, 3]))
    Λ = [0.2, 0.5, 0.3]

    # Renorm by a given balanced-growth factor g is `ReproductionStage(s = 1/g)` — no separate stage.
    renorm = ReproductionStage(layout; s = 1 / 2.0)
    @test backward!(renorm, [4.0, 1.0, 2.0], nothing) == [4.0, 1.0, 2.0]   # V untouched
    @test forward!(renorm, Λ) ≈ Λ ./ 2.0

    # State-dependent detrending rides the same FromEnv contract (the reciprocal supplied in env).
    fe = ReproductionStage(layout; s = FromEnv(:inv_g))
    @test effective_env_slice(fe) == (:inv_g,)
    backward!(fe, zeros(3), (inv_g = 1 / 4.0,))
    @test forward!(fe, Λ) ≈ Λ ./ 4.0

    # Detrending by the BG factor exactly undoes a reproduction by the same factor.
    grow = ReproductionStage(layout; s = 1.05)
    chain = grow ∘ ReproductionStage(layout; s = 1 / 1.05)
    backward!(chain, zeros(3), nothing)
    @test forward!(chain, copy(Λ)) ≈ Λ                       # (1.05)·(1/1.05)·Λ = Λ
end
