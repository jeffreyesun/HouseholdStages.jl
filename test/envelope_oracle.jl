# The envelope-vs-reoptimize oracle, shared by every optimizing-stage family.
#
# One Dual `backward!` of a freshly built chain, seeded in the P4 direction (`:env`) or the P3
# direction (`:V_end`), against a central difference of the RE-SOLVED primal at `±h`. Re-solving is
# what makes the difference an independent oracle: each `±h` leg runs its own build and its own
# solve, so it re-optimizes rather than reproducing a frozen chain's answer.
#
# The harness checks the two channels every family shares — `partials.(policy)` and
# `partials.(V_start)`. Whatever is specific to a family (bound-cell zeros, an increment seeding, an
# analytic tangent) stays an assertion in the calling test file, so the harness cannot mask a
# family's own failure mode.

if !@isdefined(ENVELOPE_ORACLE_LOADED)
const ENVELOPE_ORACLE_LOADED = true

using Test
using HouseholdStages
using ForwardDiff

"""
`env` with a tangent on each field named in `direction`, at Dual type `D`.
"""
_eo_seed_env(::Type{D}, env, direction) where {D} =
    NamedTuple{keys(env)}(map(k -> D(env[k], ForwardDiff.Partials((Float64(get(direction, k, 0.0)),))),
                              keys(env)))

"""
`env` stepped `h` along `direction`.
"""
_eo_step_env(env, direction, h) =
    NamedTuple{keys(env)}(map(k -> env[k] + h * get(direction, k, 0.0), keys(env)))

"""
The largest gap between a tangent and its finite difference, and the bar it is held to.
"""
_eo_gap(ad, fd; rtol, atol) = (maximum(abs, ad .- fd), atol + rtol * maximum(abs, fd))

"""
Envelope-vs-reoptimize on one optimizing stage, `build()` returning a fresh `(; stage, V_end, env)`
per leg. `mode` picks the direction seeded: `:env` (P4, `direction` a NamedTuple of env tangents) or
`:V_end` (P3, `direction` an array shaped like `V_end`).
"""
function envelope_vs_reoptimize(build; mode::Symbol, direction, h=1e-5, policy_of=policy,
                                readout=identity, rtol=1e-7, atol=1e-9, label="")
    @assert mode in (:env, :V_end) "mode is :env (P4) or :V_end (P3)"
    dual  = build()
    lift  = lift_jacobian(dual.stage; n_dual=1)
    D     = eltype(V_start_buffer(lift))
    V_end = mode === :V_end ? D.(dual.V_end, ForwardDiff.Partials.(tuple.(direction))) : dual.V_end
    env   = mode === :env ? _eo_seed_env(D, dual.env, direction) : dual.env
    V_ad  = ForwardDiff.partials.(copy(readout(backward!(lift, V_end, env))), 1)
    p_ad  = ForwardDiff.partials.(copy(policy_of(lift)), 1)

    function leg(step)
        fx = build()
        V  = mode === :V_end ? fx.V_end .+ step .* direction : fx.V_end
        e  = mode === :env ? _eo_step_env(fx.env, direction, step) : fx.env
        V_start = copy(readout(backward!(fx.stage, V, e)))
        return (V_start, copy(policy_of(fx.stage)))
    end
    (V_plus, p_plus), (V_minus, p_minus) = leg(h), leg(-h)
    V_fd, p_fd = (V_plus .- V_minus) ./ 2h, (p_plus .- p_minus) ./ 2h

    @testset "envelope vs reoptimize — $label ($mode)" begin
        pgap, pbar = _eo_gap(p_ad, p_fd; rtol, atol)
        vgap, vbar = _eo_gap(V_ad, V_fd; rtol, atol)
        @test pgap <= pbar
        @test vgap <= vbar
    end
    return (; policy_ad=p_ad, policy_fd=p_fd, value_ad=V_ad, value_fd=V_fd)
end

end # ENVELOPE_ORACLE_LOADED
