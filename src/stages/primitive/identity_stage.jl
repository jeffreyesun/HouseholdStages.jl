"No-op stage: the identity transition on both V and Λ."
struct IdentityStageSpec <: AbstractStageSpec end

@definestage IdentityStage IdentityStageSpec


##########################
# Gridded implementation #
##########################

operative_axis(::IdentityStageSpec) = nothing

allocate_kernel(::IdentityStageSpec, ::Type, ::GriddedLayout, ::GriddedLayout) = I

function backward!(V_start, ::IdentityStageSpec, ::GriddedLayout, ::GriddedLayout, V_end;
                   env, kernel, scratch, cache, env_changed::Bool = true)
    backward!(V_start, kernel, V_end)
    return (V_start, kernel)
end

# forward! is the generic default.
