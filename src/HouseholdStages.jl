module HouseholdStages

# Public surface for the household-layer package. The categorical content
# (stage composition, product, K-operator framing, per-stage functorial lifts,
# V/Λ duality, sequence-space utilities) lives here.
#
# Every stage and outer-loop helper splits into a Spec/Buffer-keyed primitive
# (in `*_internal.jl` / the stage file) and a Stage-keyed public wrapper, sharing
# one function name with multiple dispatch routing the call.

using ForwardDiff: ForwardDiff, Dual

export
    # Layout
    AxisKind, ContinuousGrid, DiscreteFinite,
    StateAxis, AbstractLayout, GriddedLayout,
    continuous_grid, discrete_finite, categorical,
    axis_position, axisname, axisnames, axissize, axisvalues,
    layout_size, cells, cell_array, drop_axis, resize_axis_to_one,
    # Stage-parameter env resolution
    resolve, FromEnv,
    # Interpolation helpers
    reinterpolate!, reinterpolate_arr!,
    convert_distribution!, convert_distribution_arr!,
    k1_argmax_monotone!, k1_argmax_dc!,
    # Stage interface
    AbstractStage, AbstractStageSpec, AbstractStageBuffer,
    AbstractLegacyStage, AbstractModernStage,
    allocate, allocate_kernel, allocate_scratch, allocate_cache, allocate_buffer, io_scratch,
    backward!, forward!, policy,
    V_start_buffer, Λ_end_buffer, input_layout, output_layout,
    bundle, invalidate!, @definestage,
    default_eltype,
    # Stage dependency machinery
    static_env_deps, effective_env_slice, validate_env, chain_env_names,
    env_schema, make_env,
    # Concrete stages (bundled types only — Spec/Buffer are internal)
    MarkovStage,
    ArgmaxStage, LogitChoiceStage,
    MigrationStage, SectorSwitchingStage,
    BuyHomeStage, SellHomeStage,
    DeterministicContinuousStage,
    WealthChangeStage,
    AssetPriceChangeStage,
    ContinuousArgmaxStage,
    ConsumptionSavingsStage,
    ForgetfulSumStage,
    IdentityStage,
    UtilityStage,
    TimeDiscountingStage,
    AdvanceAgeStage,
    BorrowingConstraintStage,
    LogitUtilityStage,
    SearchMatchingStage,
    # Composition — `∘` is the canonical (time-ordered) operator.
    ChainStage,
    # Product — `×` is the canonical operator.
    ProductStage, product, ×, replicate_age,
    # Populations — representations of a distribution (a kernel acts on these linearly)
    AbstractPopulation, GriddedPopulation, masses, as_population,
    uniform_distribution, expectation,
    # Moments
    MomentSpec, at_end,
    define_moment!, define_moments!, compute_moments,
    # Aggregate-Jacobian utilities (sequence-space)
    expectation_vectors, build_F, J_from_F,
    # Lifts
    lift_jacobian, with_eltype,
    backward_adjoint!, forward_adjoint!,
    lift_gpu, to_device,
    # Outer-loop computation surface — Spec/Buffer-keyed primitives
    # (in outer_loop_internal.jl) and Stage-keyed public wrappers
    # (in outer_loop.jl) share these names; dispatch routes them.
    solve_vfi_steady_state_given_env!,
    solve_lambda_steady_state_given_env!,
    solve_steady_state_given_env!,
    solve_transition_given_env_path!,
    compute_direct_jacobian!

include("layouts/abstract.jl")    # axis kinds, StateAxis, AbstractLayout + stubs
include("layouts/gridded_layout.jl") # GriddedLayout geometry + the layout-keyed allocators
include("helper/axis_ops.jl")
include("helper/closures.jl")     # declared-dependency discovery (Base.kwarg_decl) — shared backbone
include("helper/interpolations.jl")
include("stages/kernel.jl")       # transition protocol (forward!/backward!)
include("stages/abstract.jl")
include("helper/field.jl")        # matrix-valued heterogeneity field — field_deps/matrix_for/dep_combos
# Primitive stages — define their own backward!/forward!/kernel.
include("stages/primitive/markov_along.jl")
include("stages/primitive/argmax.jl")
include("stages/primitive/logit_choice.jl")        # LogitChoiceStage (transition-cost logit primitive)
include("stages/primitive/deterministic_continuous.jl")   # DeterministicContinuousStage (+ WealthChangeStage alias)
include("stages/primitive/continuous_argmax.jl")   # ContinuousArgmaxStage (generalised monotone grid argmax)
include("stages/primitive/forgetful_sum.jl")
include("stages/primitive/identity_stage.jl")
include("stages/primitive/utility.jl")
include("stages/primitive/time_discounting.jl")    # TimeDiscountingStage — V_start = β·V_end
# Combinators — the stage algebra (∘ and ×) that closes over stages.
include("stages/composition.jl")
include("stages/product.jl")
include("population.jl")           # distribution representations (GriddedPopulation, swarm seam)
# Derived stages — wrappers / compositions over primitives (need the
# combinators above for the ∘/× they build on).
include("stages/derived/migration.jl")          # MigrationStage — sugar over LogitChoiceStage
include("stages/derived/sector_switching.jl")   # SectorSwitchingStage — sugar over LogitChoiceStage
include("stages/derived/buy_home.jl")           # BuyHomeStage — gated ArgmaxStage on a housing axis
include("stages/derived/sell_home.jl")          # SellHomeStage — gated ArgmaxStage on a housing axis
include("stages/derived/consumption_savings.jl") # ConsumptionSavingsStage — sugar over ContinuousArgmaxStage
include("stages/derived/wealth_change.jl")      # WealthChangeStage — domain wrapper over DeterministicContinuousStage
include("stages/derived/asset_price_change.jl") # AssetPriceChangeStage — sugar over WealthChangeStage
include("stages/derived/advance_age.jl")        # AdvanceAgeStage — sugar over MarkovStage (age shift)
include("stages/derived/borrowing_constraint.jl") # BorrowingConstraintStage — UtilityStage with -Inf
include("stages/derived/logit_utility.jl")      # LogitUtilityStage — LogitChoiceStage ∘ UtilityStage
include("stages/derived/search_matching.jl")    # SearchMatchingStage — dedicated SaM (effort-max + θ-dependent matching)
include("moments.jl")
include("lifts/jacobian.jl")
include("lifts/gpu.jl")
include("sequence_space.jl")
include("outer_loop_internal.jl")
include("outer_loop.jl")

end # module HouseholdStages
