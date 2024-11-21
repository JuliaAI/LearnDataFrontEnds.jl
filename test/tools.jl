using Test
using LearnDataFrontEnds
using LearnAPI
using Tables
import DataFrames
using LinearAlgebra
import LearnDataFrontEnds: DoView, DontView, Multitarget, Unitarget
using CategoricalArrays
using Random
using StableRNGs
import LearnDataFrontEnds: classes,  decoder

# include("_some_learners.jl")

@testset "decompose" begin
    # tables without transposable elements:
    a = ones(5)
    b = 2ones(5)
    c = fill("a", 5)
    df = DataFrames.DataFrame((; c, b, a))
    tables = [
        df,
        Tables.rowtable(df),
        Tables.columntable(df),
        Tables.dictrowtable(df),
        Tables.dictcolumntable(df),
        DataFrames.DataFrame(df),
    ]
    for table in tables
        A, colnames, B = LearnDataFrontEnds.decompose(table, DontView(), :b)
        @test A isa Matrix
        @test size(B)[1] == 1
        @test A == hcat(c, a) |> permutedims
        @test vec(B) == b
        @test colnames == [:c, :a]
        A, colnames, B =
            LearnDataFrontEnds.decompose(table, DontView(), (:a, :b))
        @test A == hcat(c) |> permutedims
        @test colnames == [:c,]
        @test B == hcat(a, b) |> permutedims
        @test_throws(
            LearnDataFrontEnds.ERR_BAD_TARGETS((:B,)),
            LearnDataFrontEnds.decompose(table, DontView(), :B)
        )
    end

    # tables with transposable elements:
    c = 5ones(5)
    A = hcat(a, b, c)
    df = DataFrames.DataFrame((; a, b, c))
    tables = [
        df,
        Tables.rowtable(df),
        Tables.columntable(df),
        Tables.dictrowtable(df),
        Tables.dictcolumntable(df),
        DataFrames.DataFrame(df),
    ]
    for table in tables
        A, colnames, B = LearnDataFrontEnds.decompose(table, DoView(), "b")
        @test A isa LinearAlgebra.Transpose
        @test size(B)[1] == 1
        @test A == hcat(a, c) |> permutedims
        @test vec(B) == b
        @test colnames == [:a, :c]
        A, colnames, B =
            LearnDataFrontEnds.decompose(table, DoView(), ("b", "a"))
        @test A == hcat(c) |> permutedims
        @test B isa LinearAlgebra.Transpose
        @test B == hcat(b, a) |> permutedims
        @test colnames == [:c,]
        @test_throws(
            LearnDataFrontEnds.ERR_BAD_TARGETS((:B,)),
            LearnDataFrontEnds.decompose(table, DoView(), "B")
        )
    end
end


rng = StableRNGs.StableRNG(123)

@testset "classes" begin
    v = categorical(collect("asqfasqffqsaaaa"), ordered=true)
    @test classes(v[1]) == levels(v)
    @test classes(v) == levels(v)
    levels!(v, reverse(levels(v)))
    @test classes(v[1]) == levels(v)
    @test classes(v) == levels(v)
    vsub = view(v, 1:2)
    @test classes(vsub) == classes(v)
end

const int = CategoricalArrays.refcode

@testset "decoder" begin
    N = 10
    mix = shuffle(rng, 0:N - 1)

    Xraw = broadcast(x->mod(x,N), rand(rng, Int, 2N, 3N))
    Yraw = string.(Xraw)

    # to turn a categ matrix into a ordinary array with categorical
    # elements. Needed because broadcasting the identity gives a
    # categorical array in CategoricalArrays > 0.5.2
    function matrix_(X)
        ret = Array{Any}(undef, size(X))
        for i in eachindex(X)
            ret[i] = X[i]
        end
        return ret
    end

    X = categorical(Xraw)
    x = X[1]
    Y = categorical(Yraw)
    y = Y[1]
    V = matrix_(X)
    W = matrix_(Y)

    # encoding is right-inverse to decoding:
    d = decoder(x)
    @test d.(int.(V)) == V # ie have the same elements
    e = decoder(y)
    @test e.(int.(W)) == W

    @test int.(classes(y)) == 1:length(classes(x))

    v = categorical(['a', 'b', 'c'], ordered=true)
end

@testset "features_names" begin
    p, n = 3, 5
    A = ones(p, n)
    X = Tables.table(A')
    model = fit(LearnerNotReportingNames(), X)
    true_names = collect(keys(X))
    bad_names = [:Column42, :Column1]
    @test isempty(LearnDataFrontEnds.feature_names(model, true_names))
    @test isempty(LearnDataFrontEnds.feature_names(model, 3))

    model = fit(LearnerReportingNames(), X)
    @test LearnDataFrontEnds.feature_names(model, true_names) == true_names
    @test_throws(
        LearnDataFrontEnds.ERR_INCONSISTENT_FEATURES,
        LearnDataFrontEnds.feature_names(model, bad_names),
    )
    @test_throws(
        LearnDataFrontEnds.ERR_INCONSISTENT_FEATURE_COUNT(2),
        LearnDataFrontEnds.feature_names(model, 2),
    )
end

true
