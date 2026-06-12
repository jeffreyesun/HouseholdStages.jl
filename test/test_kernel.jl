using Test
using HouseholdStages
using LinearAlgebra: I

# Phase-2 transition protocol: the duality identity ⟨V_in, Λ_in⟩ = ⟨V_out, Λ_out⟩
# asserted directly on each transition (no stage wrapper). `backward!` = Kᵀ pulls
# values back; `forward!` = K pushes population forward; the bare pairing balances
# whenever the two are an adjoint pair (every transition here except discount).
const HS = HouseholdStages
using .HouseholdStages: BackwardScale, _present_perm, _gather_perm,
                        axis_position, layout_size, forward!, backward!

pairing(a, b) = sum(a .* b)

# Build a dense kernel + its gather scratch from a compact fiber `Kc` (n_out, n_in,
# dep_sizes…), exactly as a stage's allocate_kernel/allocate_scratch would: a
# self-describing `PermutedDimsArray` over the compact parent, the cached src-gather perm,
# and two work-buffers. Shared by test_transition_flexibility.jl (same module scope).
function dense_kernel(Kc, layout, axis; dep = ())
    n_out, n_in = size(Kc, 1), size(Kc, 2)
    N         = length(layout)
    adim      = axis_position(layout, axis)
    depdims   = Tuple(axis_position(layout, a) for a in dep)
    dep_sizes = size(Kc)[3:end]
    parent    = zeros(eltype(Kc), n_out, n_in, dep_sizes..., ntuple(_ -> 1, N - 1 - length(dep))...)
    reshape(parent, n_out, n_in, dep_sizes...) .= Kc
    perm, _ = _gather_perm(adim, depdims, Val(N))
    gsz     = max(n_out, n_in) * (prod(layout_size(layout)) ÷ n_in)
    return (kernel = PermutedDimsArray(parent, _present_perm(adim, depdims, Val(N))),
            perm = perm, gin = zeros(eltype(Kc), gsz), gout = zeros(eltype(Kc), gsz))
end

dbwd(dest, d, src) = backward!(dest, d.kernel, src, d.perm, d.gin, d.gout)
dfwd(dest, d, src) = forward!(dest, d.kernel, src, d.perm, d.gin, d.gout)

"Assert ⟨backward!(V_out), Λ_in⟩ ≈ ⟨V_out, forward!(Λ_in)⟩ for a dense transition."
function assert_duality(d, V_out, Λ_in, V_in_buf, Λ_out_buf)
    V_in  = dbwd(V_in_buf, d, V_out)
    Λ_out = dfwd(Λ_out_buf, d, Λ_in)
    @test isapprox(pairing(V_in, Λ_in), pairing(V_out, Λ_out); atol = 1e-12)
    return (; V_in, Λ_out)
end

@testset "Identity transition (UniformScaling) — copy both ways, duality holds" begin
    V_out = randn(3, 4); Λ_in = rand(3, 4); Λ_in ./= sum(Λ_in)
    V_in  = backward!(similar(V_out), I, V_out)
    Λ_out = forward!(similar(Λ_in), I, Λ_in)
    @test V_in == V_out
    @test Λ_out == Λ_in
    @test isapprox(pairing(V_in, Λ_in), pairing(V_out, Λ_out); atol = 1e-12)
end

@testset "Constant dense matrix-field along one axis — duality holds" begin
    layout = GriddedLayout(
        StateAxis(:wealth, continuous_grid([0.0, 0.5, 1.0])),
        StateAxis(:z,      discrete_finite([0.5, 1.5])),
    )
    # Stored fiber = K (forward operator): forward! = K·Λ, backward! = Kᵀ·V.
    K  = [0.6 0.25; 0.4 0.75]                      # 2×2 along :z
    d  = dense_kernel(K, layout, :z)

    V_out = randn(3, 2); Λ_in = rand(3, 2); Λ_in ./= sum(Λ_in)
    (; V_in, Λ_out) = assert_duality(d, V_out, Λ_in, similar(V_out), similar(Λ_in))

    # Cross-check the forward action against an explicit Kᵀ contraction over :z.
    @test isapprox(Λ_out, (K * Λ_in')'; atol = 1e-12)
end

@testset "Axis-already-first matrix-field — no-permute fast path" begin
    layout = GriddedLayout(
        StateAxis(:z,      discrete_finite([0.5, 1.5])),
        StateAxis(:wealth, continuous_grid([0.0, 1.0, 2.0])),
    )
    K  = [0.9 0.2; 0.1 0.8]
    d  = dense_kernel(K, layout, :z)
    V_out = randn(2, 3); Λ_in = rand(2, 3); Λ_in ./= sum(Λ_in)
    assert_duality(d, V_out, Λ_in, similar(V_out), similar(Λ_in))
    # forward! = K along the first axis.
    @test isapprox(dfwd(similar(Λ_in), d, Λ_in), K * Λ_in; atol = 1e-12)
end

@testset "Axis-varying matrix-field (varies along a dep axis) — duality holds" begin
    layout = GriddedLayout(
        StateAxis(:wealth, continuous_grid([0.0, 1.0, 2.0])),   # transitioned axis (3)
        StateAxis(:age,    discrete_finite([1.0, 2.0])),        # dep axis (2)
    )
    # One 3×3 forward matrix per age; each column-stochastic so it's a genuine K.
    Ks = stack([[0.5 0.1 0.0; 0.3 0.6 0.2; 0.2 0.3 0.8],
                [0.7 0.2 0.1; 0.2 0.5 0.3; 0.1 0.3 0.6]])      # (3,3,2)
    d  = dense_kernel(Ks, layout, :wealth; dep = (:age,))

    V_out = randn(3, 2); Λ_in = rand(3, 2); Λ_in ./= sum(Λ_in)
    assert_duality(d, V_out, Λ_in, similar(V_out), similar(Λ_in))

    # Each age slice contracts with its own fiber on forward.
    Λ_out = dfwd(similar(Λ_in), d, Λ_in)
    for a in 1:2
        @test isapprox(Λ_out[:, a], Ks[:, :, a] * Λ_in[:, a]; atol = 1e-12)
    end
end

@testset "Matrix-field varying along two dep axes — batched path, duality holds" begin
    layout = GriddedLayout(
        StateAxis(:wealth, continuous_grid([0.0, 1.0, 2.0])),   # transitioned axis (3)
        StateAxis(:age,    discrete_finite([1.0, 2.0])),        # dep axis (2)
        StateAxis(:health, discrete_finite([1.0, 2.0, 3.0])),   # dep axis (3) ⇒ batch = 6
    )
    # One 3×3 forward matrix per (age, health) cell; each column-stochastic.
    rand_K() = (M = rand(3, 3); M ./ sum(M; dims = 1))
    Ks = stack([rand_K() for _ in 1:6])                         # (3,3,6)
    Ks = reshape(Ks, 3, 3, 2, 3)                                # (n_out,n_in,age,health)
    d  = dense_kernel(Ks, layout, :wealth; dep = (:age, :health))

    V_out = randn(3, 2, 3); Λ_in = rand(3, 2, 3); Λ_in ./= sum(Λ_in)
    assert_duality(d, V_out, Λ_in, similar(V_out), similar(Λ_in))

    # Each (age, health) cell contracts with its own fiber on forward.
    Λ_out = dfwd(similar(Λ_in), d, Λ_in)
    for a in 1:2, h in 1:3
        @test isapprox(Λ_out[:, a, h], Ks[:, :, a, h] * Λ_in[:, a, h]; atol = 1e-12)
    end
end

@testset "Rectangular marginalize (drop an axis) — duality holds" begin
    layout = GriddedLayout(
        StateAxis(:wealth, continuous_grid([0.0, 1.0, 2.0, 3.0])),
        StateAxis(:m,      discrete_finite([1.0, 2.0, 3.0])),   # marginalised (3 → 1)
    )
    K  = ones(1, 3)                                  # forward sums :m; Kᵀ broadcasts
    d  = dense_kernel(K, layout, :m)

    V_out = randn(4, 1); Λ_in = rand(4, 3); Λ_in ./= sum(Λ_in)
    V_in  = dbwd(zeros(4, 3), d, V_out)              # Kᵀ broadcasts V_out across :m
    Λ_out = dfwd(zeros(4, 1), d, Λ_in)               # K sums Λ_in over :m
    @test isapprox(pairing(V_in, Λ_in), pairing(V_out, Λ_out); atol = 1e-12)
    @test all(V_in[:, j] == V_out[:, 1] for j in 1:3)
    @test isapprox(vec(Λ_out), vec(sum(Λ_in; dims = 2)); atol = 1e-12)
end

@testset "BackwardScale (discount) — asymmetric, not an adjoint pair" begin
    β  = 0.96
    tr = BackwardScale(β)
    V_out = randn(3, 2); Λ_in = rand(3, 2)
    # forward! = copy (the *undiscounted* population is pushed forward).
    @test forward!(similar(Λ_in), tr, Λ_in) == Λ_in
    # backward! = β · V (values are discounted).
    @test isapprox(backward!(similar(V_out), tr, V_out), β .* V_out; atol = 1e-14)
end

@testset "Dense kernel op is type-stable (@inferred)" begin
    layout = GriddedLayout(
        StateAxis(:wealth, continuous_grid([0.0, 1.0, 2.0])),
        StateAxis(:age,    discrete_finite([1.0, 2.0])),
    )
    Ks = stack([[0.5 0.1 0.0; 0.3 0.6 0.2; 0.2 0.3 0.8],
                [0.7 0.2 0.1; 0.2 0.5 0.3; 0.1 0.3 0.6]])
    d  = dense_kernel(Ks, layout, :wealth; dep = (:age,))
    V_out = randn(3, 2)
    @test (@inferred dbwd(similar(V_out), d, V_out)) isa AbstractArray{Float64, 2}
    @test (@inferred dfwd(similar(V_out), d, V_out)) isa AbstractArray{Float64, 2}
end
