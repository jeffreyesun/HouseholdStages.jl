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
    AxisRep, GriddedContinuous, Discrete,
    AbstractLayout, GriddedLayout,
    axis_position, axisnames, axissize, axisvalues, axis_grid,
    layout_size, cells, cell_array, drop_axis, resize_axis, grow_axis,
    # Stage-parameter env resolution
    resolve, FromEnv,
    # Interpolation helpers
    reinterpolate!, convert_distribution!,
    # Preference building blocks
    u_crra,
    # Heterogeneity fields
    ScalarField, materialize_scalar!, scalar_broadcastable,
    # Stage interface
    AbstractStage, AbstractStageSpec, AbstractStageBuffer,
    AbstractLegacyStage, AbstractModernStage,
    allocate, allocate_kernel, allocate_scratch, allocate_cache, allocate_buffer, io_scratch,
    backward!, forward!, policy,
    V_start_buffer, Λ_end_buffer, input_layout, output_layout,
    bundle, @definestage,
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
    SearchMatchingStage,
    KernelChoiceStage, PortfolioStage, ScaleChoiceStage,
    StreamingChoiceStage, ScaleVarianceStage, MeanVarianceStage,
    MixingStage, RetentionStage,
    CapitalInvestmentStage, DurableAdjustmentStage,
    DefaultStage, DirectedSearchStage,
    # Composition — `∘` is the canonical (time-ordered) operator.
    ChainStage,
    # Direct sum — `⊕` is the canonical operator.
    ProductStage, product, ⊕, replicate_age,
    # Populations — representations of a distribution (a kernel acts on these linearly)
    AbstractPopulation, GriddedPopulation, masses, as_population,
    uniform_distribution, expectation,
    # Moments
    MomentSpec, at_end,
    define_moment!, define_moments!, compute_moment, compute_moments,
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

# Layer 1 — layouts: state-space representations
include("layouts/abstract.jl")        # axis representations, AbstractLayout + stubs
include("layouts/gridded_layout.jl")  # GriddedLayout geometry (incl. name-keyed `fix`) + the layout-keyed allocators
# Layer 2 — sources: the closure/Sources backbone (evaluate/declared_deps/reads_env/dep_combos)
include("sources/sources.jl")         # declared-dependency discovery (Base.kwarg_decl) + the Sources API
include("sources/to_matrix_source.jl")# start-and-end (positional pair) reward/cost lowering (§4.1)
# Layer 3 — fields: stratified data + generic apply
include("fields/scalar_field.jl")     # ScalarField — scalar-per-base-point (broadcast) field
include("fields/matrix_field.jl")     # MatrixField + MappedField + stratified_apply!/fill_field!
include("fields/interpolations.jl")   # reinterpolate / Young-split / monotone closures
# Layer 4 — populations: distribution representations (a kernel acts on these linearly)
include("populations/population.jl")  # AbstractPopulation, GriddedPopulation
# Layer 5 — moments: pure ⟨Source, Population⟩ computation (no ChainStage)
include("moments/moments.jl")         # MomentSpec + compute_moment(layout, spec, Λ, env)
# Layer 6 — kernels: the transition operators (depend on fields)
include("kernels/kernel.jl")          # forward!/backward! protocol + Identity + DestinationField
include("kernels/pointwise_scale.jl") # PointwiseScale (the two-sided diagonal scale)
include("kernels/dense_kernel.jl")    # DenseKernel + dense_kernel front door (owns a MatrixField)
include("kernels/logit_kernel.jl")    # LogitChoiceKernel (contains a DenseKernel)
include("kernels/scatter_kernel.jl")  # ScatterKernel (discrete destination)
include("kernels/interp_kernel.jl")   # InterpKernel (continuous destination)
include("kernels/mps_kernel.jl")      # MPSKernel + the shared kernel-choice scatter/gather core
include("kernels/mean_variance_kernel.jl") # MeanVarianceKernel + the kernel-choice union verbs
# Layer 7 — stages: the stage protocol (Spec/Layout/Kernel/Scratch/Cache, FromEnv/resolve)
include("stages/abstract.jl")
# Primitive stages — define their own backward!/forward!/kernel.
include("stages/primitive/markov_along.jl")
include("stages/primitive/argmax.jl")              # ArgmaxStage — (max,+) over an axis: :brute or monotone search
include("stages/primitive/continuous_argmax.jl")   # ContinuousArgmaxStage — off-grid (max,+) → InterpKernel float policy (§10)
include("stages/primitive/logit_choice.jl")        # LogitChoiceStage (transition-cost logit primitive)
include("stages/primitive/deterministic_continuous.jl")   # DeterministicContinuousStage (+ WealthChangeStage alias)
include("stages/primitive/discrete_move.jl")       # DiscreteMoveStage — nearest-index ScatterKernel sibling of the continuous move (§10)
include("stages/primitive/identity_stage.jl")
include("stages/primitive/utility.jl")
include("stages/primitive/pointwise_scale.jl")     # PointwiseScaleStage — V_start = a·V_end, Λ_end = f·Λ_start
include("stages/primitive/time_discounting.jl")    # TimeDiscountingStage — PointwiseScale(backward=β, forward=1)
include("stages/primitive/entry.jl")               # EntryStage — forward source Λ_end = Λ + g (gap G2, entry half)
# Layer 8 — combinators: the stage algebra (∘ and ⊕) + the moment menu (chain attachment).
include("combinators/composition.jl")  # ∘ (time-ordered)        ← ChainStage
include("combinators/direct_sum.jl")    # ⊕ (direct sum)          ← ProductStage
include("combinators/moment_menu.jl")   # define_moment! / compute_moments(::ChainStage…)
# Layer 9 — derived stages — wrappers / compositions over primitives (need the
# combinators above for the ∘/⊕ they build on).
include("stages/derived/forgetful_sum.jl")      # ForgetfulSumStage — sugar over MarkovStage (rectangular ones-row)
include("stages/derived/migration.jl")          # MigrationStage — sugar over LogitChoiceStage
include("stages/derived/sector_switching.jl")   # SectorSwitchingStage — sugar over LogitChoiceStage
include("stages/derived/buy_home.jl")           # BuyHomeStage — gated ArgmaxStage on a housing axis
include("stages/derived/sell_home.jl")          # SellHomeStage — gated ArgmaxStage on a housing axis
include("stages/derived/consumption_savings.jl") # ConsumptionSavingsStage — sugar over ArgmaxStage
include("stages/derived/wealth_change.jl")      # WealthChangeStage — domain wrapper over DeterministicContinuousStage
include("stages/derived/income.jl")             # IncomeStage — WealthChangeStage with the standard cash-on-hand receipt
include("stages/derived/asset_price_change.jl") # AssetPriceChangeStage — sugar over WealthChangeStage
include("stages/derived/advance_age.jl")        # AdvanceAgeStage — sugar over MarkovStage (age shift)
include("stages/derived/reproduction.jl")       # ReproductionStage — PointwiseScale measure half
include("stages/derived/borrowing_constraint.jl") # BorrowingConstraintStage — UtilityStage with -Inf
include("stages/derived/logit_utility.jl")      # LogitUtilityStage — LogitChoiceStage ∘ UtilityStage
include("stages/derived/search_matching.jl")    # SearchMatchingStage — dedicated SaM (effort-max + θ-dependent matching)
include("stages/derived/kernel_choice.jl")      # KernelChoiceStage/PortfolioStage/ScaleChoiceStage — gridded Collapse∘Markov[θ-dep]∘ForgetfulSum (equivalence-test references only, §12)
include("stages/derived/kernel_choice_stages.jl") # StreamingChoice/ScaleVariance/MeanVariance — kernel-choice primitives over MPSKernel/MeanVarianceKernel (§10)
include("stages/derived/mixing.jl")             # MixingStage/RetentionStage — closed-form conjugate (two Markov + c*)
include("stages/derived/investment.jl")         # CapitalInvestmentStage/DurableAdjustmentStage — ArgmaxStage sugar
include("stages/derived/default_choice.jl")     # DefaultStage — gated ArgmaxStage (repay/default)
include("stages/derived/directed_search.jl")    # DirectedSearchStage — LogitChoiceStage over submarkets
include("stages/derived/exit.jl")               # ExogenousExit/EndogenousExit/LogitEndogenousExit — Choice ∘ Utility ∘ Markov over a transient :exiting axis (mass leaves)
# RI-discrete is deliberately NOT a stage — it is the composition LogitChoice(ε=λ)∘UtilityStage(λ·log q)
# (= LogitUtilityStage); see test/test_rational_inattention.jl for why the old RationalInattentionStage was cut.
# Layer 10 — lifts: functorial re-typings
include("lifts/jacobian.jl")
include("lifts/gpu.jl")
# Layer 11 — outer loop: the top consumer
include("outer_loop/sequence_space.jl")
include("outer_loop/outer_loop_internal.jl")
include("outer_loop/outer_loop.jl")
# helper — cross-cutting pure-math leaf (depends on nothing in the stack, used by examples + derived)
include("helper/preferences.jl")      # CRRA felicity (u_crra)

end # module HouseholdStages
