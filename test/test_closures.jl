using Test
using HouseholdStages

# Declared-dependency discovery (`helper/closures.jl`) — the shared backbone that
# replaced the retired `AxisField` dep machinery. A producer closure declares which
# axes it varies along by NAMING them as kwargs; `_closure_deps` recovers them (in
# layout order, regardless of kwarg order), `_closure_env_dep` reads the `env` switch,
# and both reject multi-method / slurping closures. These assertions are the parts of
# the old `test_axis_field.jl` worth keeping — folded onto the live API.
const HS = HouseholdStages
using .HouseholdStages: _closure_deps, _closure_env_dep, _closure_deps_from_kwargs,
                        _closure_kwargs_raw

layout = GriddedLayout(
    :income => Discrete([0.5, 1.0, 1.5]),
    :wealth => GriddedContinuous([0.0, 1.0, 2.0, 3.0]),
    :location => Discrete([:A, :B]),
)

@testset "closure deps — recovered in layout order, regardless of kwarg order" begin
    @test _closure_deps((; income, wealth) -> income + wealth, layout) === (:income, :wealth)
    # Declared out of order — still sorted by layout position.
    @test _closure_deps((; wealth, income) -> 10income + wealth, layout) === (:income, :wealth)
    @test _closure_deps((; location) -> location === :B, layout) === (:location,)
    @test _closure_deps((;) -> 3.5, layout) === ()                 # constant: no deps
end

@testset "closure env-dependence — the `env` kwarg is the switch" begin
    @test _closure_env_dep((; wealth, env) -> env.r * wealth) === true
    @test _closure_env_dep((; wealth) -> 2wealth) === false
    # `env` is not counted as a dep axis (it is the refresh switch, not a layout axis).
    @test _closure_deps((; wealth, env) -> env.r * wealth, layout) === (:wealth,)
end

@testset "closure deps — reserved roles are dropped (per-entry convention)" begin
    # `:from`/`:to` are reserved endpoint roles, not stored deps.
    f = (; from, to, income) -> from + to + income
    @test _closure_deps_from_kwargs(_closure_kwargs_raw(f), axisnames(layout);
                                    reserved = (:from, :to)) === (:income,)
end

multimethod(x::Int; income) = income
multimethod(x::Float64; wealth) = wealth

@testset "closure deps — errors on unknown kwarg, slurp, and multi-method" begin
    @test_throws ErrorException _closure_deps((; incmoe) -> incmoe, layout)         # typo
    @test_throws ErrorException _closure_deps((; wealth, kwargs...) -> wealth, layout)  # slurp
    @test_throws ErrorException _closure_deps(multimethod, layout)   # two methods ⇒ deps unknowable
end
