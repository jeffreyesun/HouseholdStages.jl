using Test
using HouseholdStages

@testset "layout — axis representations" begin
    g = GriddedContinuous(0.0, 10.0, 5)
    @test g isa GriddedContinuous
    @test length(g.grid) == 5
    @test g.grid[1] ≈ 0.0
    @test g.grid[end] ≈ 10.0
    @test axissize(g) == 5
    @test axisvalues(g) == g.grid

    g2 = GriddedContinuous([0.0, 0.5, 1.0])
    @test g2 isa GriddedContinuous
    @test g2.grid == [0.0, 0.5, 1.0]

    glog = GriddedContinuous(0.0, 10.0, 5; spacing = :log)   # first/last pinned, dense near lo
    @test glog.grid[1] ≈ 0.0
    @test glog.grid[end] ≈ 10.0

    d_int = Discrete([1, 2, 3])
    @test d_int isa Discrete{Int}
    @test d_int.levels == [1, 2, 3]
    @test axissize(d_int) == 3

    d_float = Discrete([0.0, 1.0, 2.0])
    @test d_float isa Discrete{Float64}

    d_sym = Discrete([:NYC, :LA, :Chicago])                  # Symbol levels (the former `categorical`)
    @test d_sym isa Discrete{Symbol}
    @test axisvalues(d_sym) == [:NYC, :LA, :Chicago]
end

@testset "layout — names live in the layout, reps are nameless" begin
    layout = GriddedLayout(
        :wealth => GriddedContinuous(0.0, 1.0, 4),
        :loc    => Discrete([:NYC, :LA]),
    )
    @test axisnames(layout) === (:wealth, :loc)
    @test layout.axes[2] isa Discrete            # the stored rep carries no name
    @test axissize(layout.axes[1]) == 4
    @test axis_grid(layout, :wealth) |> length == 4
    @test axis_grid(layout, :loc) == [:NYC, :LA]
end

@testset "layout — GriddedLayout construction and axis_position" begin
    layout = GriddedLayout(
        :wealth => GriddedContinuous(0.0, 100.0, 8),
        :income => Discrete([0.5, 1.0, 1.5]),
        :loc    => Discrete([:NYC, :LA]),
    )
    @test length(layout) == 3
    @test axisnames(layout) === (:wealth, :income, :loc)
    @test layout_size(layout) == (8, 3, 2)
    @test axis_grid(layout, :income) == [0.5, 1.0, 1.5]

    @test axis_position(layout, :wealth) == 1
    @test axis_position(layout, :income) == 2
    @test axis_position(layout, :loc) == 3
    @test_throws ErrorException axis_position(layout, :nope)
end

@testset "layout — duplicate axis names error" begin
    @test_throws ErrorException GriddedLayout(
        :x => GriddedContinuous(0.0, 1.0, 2),
        :x => Discrete([1, 2]),
    )
end

@testset "layout — cells iteration" begin
    layout = GriddedLayout(
        :wealth => GriddedContinuous([0.0, 1.0, 2.0]),
        :loc    => Discrete([:NYC, :LA]),
    )
    pairs = collect(cells(layout))
    @test length(pairs) == 6

    expected = [
        ((wealth=1, loc=1), (wealth=0.0, loc=:NYC)),
        ((wealth=2, loc=1), (wealth=1.0, loc=:NYC)),
        ((wealth=3, loc=1), (wealth=2.0, loc=:NYC)),
        ((wealth=1, loc=2), (wealth=0.0, loc=:LA)),
        ((wealth=2, loc=2), (wealth=1.0, loc=:LA)),
        ((wealth=3, loc=2), (wealth=2.0, loc=:LA)),
    ]
    for ((iexp, cexp), (igot, cgot)) in zip(expected, pairs)
        @test igot === iexp
        @test cgot === cexp
    end
end

@testset "layout — cells iteration eltype is concrete NamedTuple per leaf type" begin
    # All-Float64 layout: cell values should be NamedTuple of Floats.
    layout = GriddedLayout(
        :a => GriddedContinuous([0.0, 1.0]),
        :b => Discrete([10.0, 20.0]),
    )
    for (idx, cell) in cells(layout)
        @test idx isa NamedTuple{(:a, :b), <:NTuple{2, Int}}
        @test cell isa NamedTuple{(:a, :b), <:NTuple{2, Float64}}
    end
end

@testset "layout — cells iteration is type-stable" begin
    layout = GriddedLayout(
        :x => GriddedContinuous([0.0, 0.5, 1.0]),
        :y => Discrete([:A, :B]),
    )
    it = cells(layout)
    # Iterate once and ensure inference yields a concrete return type.
    f(it) = begin
        s = 0.0
        for (idx, cell) in it
            s += cell.x
        end
        s
    end
    @inferred f(it)
    @test f(it) ≈ 3.0  # (0.0 + 0.5 + 1.0) * 2
end
