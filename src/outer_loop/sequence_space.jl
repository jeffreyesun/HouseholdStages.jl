###################################
# Aggregate-Jacobian Utilities    #
# (fake-news Steps 2, 3, 4)       #
###################################
#
# Outer-loop accounting steps in SSJ's fake-news algorithm
# (stages_paper/notes/proofs.md Cell 1.7). Convert per-stage tangents
# into sequence-space Jacobians at the steady state. Step 1 (per-stage
# tangent propagation via F_J_fwd) lives in `src/lifts/jacobian.jl`
# via `lift_jacobian(stage; mode=:forward)`.

"""
Time-`t` expectations of `integrand` viewed from the start of the period, for
`t = 0,…,T-1`, at the steady state — Step 2 of SSJ's fake-news algorithm. The
`t=0` array evaluates `integrand` at every terminal-layout cell; later `t`
iterate the chain's K-transpose (`forward_adjoint!`). Requires the chain's
kernels to have been seated by a prior `backward!(chain, V_terminal, env_ss)`.
"""
function expectation_vectors(chain::AbstractStage, integrand::Function, T::Int)
    @assert T ≥ 1 "expectation_vectors: T must be at least 1"
    out_layout = _terminal_out_layout(chain)
    dims = layout_size(out_layout)
    Tnum = eltype_from_chain(chain)
    cells_arr = cell_array(out_layout)          # hoist out of the broadcast (conv. 2b)
    E0 = Tnum.(integrand.(cells_arr))
    results = Vector{Array{Tnum, length(dims)}}()
    push!(results, copy(E0))
    E_prev = E0
    for _ in 2:T
        # K-transpose step: forward_adjoint! reads the seated kernels from chain.buffer.
        E_next = forward_adjoint!(chain, E_prev)
        push!(results, copy(E_next))
        E_prev = E_next
    end
    return results
end

# Terminal output layout — used to size the integrand-broadcast array.
_terminal_out_layout(s::AbstractStage) = output_layout(s)

# Eltype helper: fish out the V_start buffer eltype (works for legacy + modern + chain).
eltype_from_chain(s::AbstractStage) = eltype(V_start_buffer(s))

"""
Cumulate the fake-news matrix `F` into a sequence-space Jacobian `J` along
anti-diagonals (`J[s+1, t] += J[s, t-1]`) — Step 4 of SSJ's fake-news algorithm.
"""
function J_from_F(F::AbstractMatrix)
    T_lookahead, T = size(F)
    J = copy(F)
    for t in 2:T
        for s in 1:(T_lookahead - 1)
            J[s + 1, t] += J[s, t - 1]
        end
    end
    return J
end

"""
Assemble the fake-news matrix from the direct output impact `curlyY[t]`, the
end-of-period distribution perturbation `curlyD[t]`, and the `s`-ahead
expectation vectors `curlyE[s]` — Step 3 of SSJ's fake-news algorithm. Row 1 is
the direct effect; rows `s+1` are the distribution-mediated `sum(curlyE[s] .* curlyD[t])`.
"""
function build_F(curlyY::AbstractVector, curlyD::AbstractVector,
                 curlyE::AbstractVector)
    T = length(curlyY)
    T_lookahead = length(curlyE) + 1
    @assert length(curlyD) == T "curlyD must have the same length as curlyY (one per shock time)"
    F = zeros(eltype(curlyY), T_lookahead, T)
    for t in 1:T
        F[1, t] = curlyY[t]
        for s in eachindex(curlyE)
            F[s + 1, t] = sum(curlyE[s] .* curlyD[t])
        end
    end
    return F
end
