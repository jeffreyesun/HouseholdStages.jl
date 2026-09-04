module HouseholdStages

# The household layer of heterogeneous-agent macro models, built by composing stages. A stage is a
# transition operator with two sweeps: `backward!` on a value function `V`, `forward!` on a
# distribution `Λ`. `∘` composes stages in time, `⊕` runs them in parallel.

using Adapt: Adapt, adapt
using ForwardDiff: ForwardDiff, Dual

export
    # Layout
    AxisRep, GriddedContinuous, Discrete,
    AbstractLayout, GriddedLayout,
    axis_position, axisnames, axissize, axisvalues, axis_grid,
    layout_size, cells, cell_array, drop_axis, resize_axis, grow_axis,
    # Stage-parameter env resolution
    resolve, FromEnv,
    # Preference building blocks
    u_crra,
    # Heterogeneity fields
    ScalarField, materialize_scalar!, scalar_broadcastable,
    # Refill instrumentation (opt-in diagnostic)
    reset_refill_count!, refill_count,
    # Stage interface
    AbstractStage, AbstractStageSpec, AbstractStageBuffer,
    AbstractPrimitiveStage, AbstractCompositeStage,
    allocate, allocate_kernel, allocate_scratch, allocate_cache, allocate_buffer, io_scratch,
    backward!, forward!, policy, choice_probabilities,
    V_start_buffer, Λ_end_buffer,
    start_layout, end_layout, layout, boundaries, interiors, operative_axis, tangent_grade,
    bundle, spec_type, @definestage,
    default_eltype,
    # Stage dependency machinery
    static_env_deps, effective_env_slice, validate_env, chain_env_names,
    env_schema, make_env,
    # Concrete stages (bundled types only — Spec/Buffer are internal)
    MarkovStage,
    ArgmaxStage, ContinuousArgmaxStage, LogitChoiceStage,
    MigrationStage, SectorSwitchingStage,
    BuyHomeStage, SellHomeStage,
    DeterministicContinuousStage, DiscreteMoveStage,
    WealthChangeStage,
    IncomeStage,
    AssetPriceChangeStage,
    ConsumptionSavingsStage,
    ForgetfulSumStage,
    IdentityStage,
    UtilityStage,
    PointwiseScaleStage,
    TimeDiscountingStage,
    ReproductionStage,
    EntryStage,
    ExogenousExit, EndogenousExit, LogitEndogenousExit,
    AdvanceAgeStage,
    BorrowingConstraintStage,
    LogitUtilityStage,
    GaussianLoadingStage,
    MeanPreservingSpreadStage,
    MixingStage, RetentionStage, SearchMatchingStage,
    CapitalInvestmentStage, DurableAdjustmentStage,
    DefaultStage, DirectedSearchStage,
    # Composition in time — `∘`
    ChainStage,
    # Direct sum — `⊕`
    ProductStage, product, ⊕, replicate_age,
    # Populations — representations of a distribution
    AbstractPopulation, GriddedPopulation, masses, as_population,
    uniform_distribution, expectation,
    # Moments
    MomentSpec, at_end,
    define_moment!, define_moments!, compute_moment, compute_moments,
    # Sequence-space Jacobians (the fake-news algorithm)
    expectation_vectors, build_F, J_from_F, compute_fake_news_ssj,
    # Lifts
    lift_jacobian, with_eltype,
    backward_adjoint!, forward_adjoint!,
    lift_gpu, to_device, to_host,
    # Outer-loop computation surface
    solve_vfi_steady_state_given_env!,
    solve_lambda_steady_state_given_env!,
    solve_steady_state_given_env!,
    solve_transition_given_env_path!,
    compute_steady_state_gradient, compute_direct_ssj

include("layouts/abstract.jl")        # axis representations and the AbstractLayout interface
include("layouts/gridded_layout.jl")  # GriddedLayout geometry and its allocators
include("sources/sources.jl")         # the Sources API and its dependency discovery
include("sources/to_matrix_source.jl")# rewards and costs that depend on both ends of a transition
include("fields/scalar_field.jl")     # ScalarField — one scalar per state
include("fields/matrix_field.jl")     # MatrixField, MappedField, and the field-filling drivers
include("fields/stratified.jl")       # stratified! — apply one operation independently per stratum
include("fields/interpolations.jl")   # the two-node lottery over a grid
include("populations/population.jl")  # AbstractPopulation, GriddedPopulation
include("moments/moments.jl")         # MomentSpec, compute_moment
include("kernels/kernel.jl")          # the forward!/backward! protocol, Identity, DestinationField
include("kernels/pointwise_scale.jl") # PointwiseScale — two-sided diagonal scaling
include("kernels/dense_kernel.jl")    # DenseKernel — owns a MatrixField
include("kernels/logit_kernel.jl")    # LogitChoiceKernel — wraps a DenseKernel
include("kernels/scatter_kernel.jl")  # ScatterKernel — discrete destination
include("kernels/interp_kernel.jl")   # InterpKernel — continuous destination, via the two-node lottery
include("kernels/mean_preserving_spread_kernel.jl") # banded Gaussian mean-preserving spread
include("kernels/gaussian_loading_kernel.jl")       # loading on a Gaussian increment
include("stages/abstract.jl")                      # the stage protocol: spec, layouts, kernel, scratch, cache
include("stages/primitive/markov_along.jl")        # MarkovStage — a given transition matrix along one axis
include("stages/primitive/argmax.jl")              # ArgmaxStage — brute-force (max,+) over an axis
include("stages/primitive/continuous_argmax.jl")   # ContinuousArgmaxStage — (max,+) with an off-grid policy
include("stages/primitive/mean_preserving_spread.jl") # MeanPreservingSpreadStage — continuous dispersion choice
include("stages/primitive/gaussian_loading.jl")   # GaussianLoadingStage — continuous loading on a Gaussian increment
include("stages/primitive/logit_choice.jl")        # LogitChoiceStage — logit over transition costs
include("stages/primitive/deterministic_continuous.jl")   # DeterministicContinuousStage
include("stages/primitive/discrete_move.jl")       # DiscreteMoveStage — the on-grid counterpart of the continuous move
include("stages/primitive/identity_stage.jl")
include("stages/primitive/utility.jl")
include("stages/primitive/pointwise_scale.jl")     # PointwiseScaleStage — V_start = a·V_end, Λ_end = f·Λ_start
include("stages/primitive/time_discounting.jl")    # TimeDiscountingStage — PointwiseScale(backward=β, forward=1)
include("stages/primitive/entry.jl")               # EntryStage — forward source Λ_end = Λ + g
include("combinators/composition.jl")   # ∘ → ChainStage
include("combinators/direct_sum.jl")    # ⊕ → ProductStage
include("combinators/moment_menu.jl")   # define_moment!, compute_moments
include("stages/derived/forgetful_sum.jl")      # ForgetfulSumStage — a MarkovStage with a rectangular ones-row
include("stages/derived/migration.jl")          # MigrationStage — a LogitChoiceStage over locations
include("stages/derived/sector_switching.jl")   # SectorSwitchingStage — a LogitChoiceStage over sectors
include("stages/derived/buy_home.jl")           # BuyHomeStage — gated ArgmaxStage on a housing axis
include("stages/derived/sell_home.jl")          # SellHomeStage — gated ArgmaxStage on a housing axis
include("stages/derived/consumption_savings.jl") # ConsumptionSavingsStage — ContinuousArgmaxStage ∘ TimeDiscountingStage
include("stages/derived/wealth_change.jl")      # WealthChangeStage — a DeterministicContinuousStage on wealth
include("stages/derived/income.jl")             # IncomeStage — a WealthChangeStage adding income to cash on hand
include("stages/derived/asset_price_change.jl") # AssetPriceChangeStage — a WealthChangeStage revaluing assets
include("stages/derived/advance_age.jl")        # AdvanceAgeStage — a MarkovStage shifting the age axis
include("stages/derived/reproduction.jl")       # ReproductionStage — a PointwiseScale acting only on Λ
include("stages/derived/borrowing_constraint.jl") # BorrowingConstraintStage — a UtilityStage giving -Inf outside the limit
include("stages/derived/logit_utility.jl")      # LogitUtilityStage — LogitChoiceStage ∘ UtilityStage
include("stages/derived/lottery_mixing.jl")     # MixingStage/RetentionStage — a chosen lottery over two kernels
include("stages/derived/search_matching.jl")    # SearchMatchingStage — separation ∘ a job-search MixingStage
include("stages/derived/investment.jl")         # CapitalInvestmentStage/DurableAdjustmentStage — ArgmaxStages
include("stages/derived/default_choice.jl")     # DefaultStage — gated ArgmaxStage over repay/default
include("stages/derived/directed_search.jl")    # DirectedSearchStage — a LogitChoiceStage over submarkets
include("stages/derived/exit.jl")               # ExogenousExit/EndogenousExit/LogitEndogenousExit — mass leaves via a transient :exiting axis
include("lifts/jacobian.jl")
include("lifts/gpu.jl")
include("outer_loop/sequence_space.jl")
include("outer_loop/outer_loop_internal.jl")
include("outer_loop/outer_loop.jl")
include("helper/preferences.jl")      # CRRA felicity (u_crra)

end # module HouseholdStages
