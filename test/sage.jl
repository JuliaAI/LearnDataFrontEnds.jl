using Test
using LearnDataFrontEnds
using LearnAPI
import MLCore
using StableRNGs
import DataFrames
using Tables
using LinearAlgebra
import StatsModels.@formula
import CategoricalArrays
import CategoricalDistributions
import CategoricalDistributions.OrderedCollections.OrderedDict
import CategoricalDistributions.Distributions.StatsBase.proportionmap

CA = CategoricalDistributions

# include("_some_learners.jl")

n = 2
rng = StableRNG(345)
# has a "hidden" level, `C`:
t = CA.categorical(repeat("ABA", 3n)*"CC" |> collect)[1:3n]
t2 = CA.categorical(repeat("BAB", 3n)*"CC" |> collect)[1:3n]
c, a = randn(rng, 3n), rand(rng, 3n)
y = t
Y = (; t)
ymulti = hcat(t, t2) |> permutedims
Ymulti = (; t, t2)

# feature matrix:
x = hcat(c, a) |> permutedims

# feature tables:
X = (; c, a)
Xs = [
    X,
    Tables.rowtable(X),
    Tables.dictrowtable(X),
    Tables.dictcolumntable(X),
    DataFrames.DataFrame(X),
]

# full tables:
T = (; c, t, a)
Ts = [
    T,
    Tables.rowtable(T),
    Tables.dictrowtable(T),
    Tables.dictcolumntable(T),
    DataFrames.DataFrame(T),
]

# StatsModels.jl @formula:
f = @formula(t ~ c + a)

# little letters, `x`, `y`, are for arrays
# big letters, `X`, `Y`, are for tables

@testset "`fitobs(learner, (x, y))`" begin
    o = fitobs("learner", (x, y), Sage())
    @test o.features == x
    @test o.names == [:x1, :x2]
    @test o.target == repeat([1, 2, 1], n)
    @test eltype(o.target) == Int
    @test o.classes_seen == CA.levels(y)[1:2]
    @test o.classes_seen isa CA.CategoricalArray
    yy = o.decoder.(o.target)
    @test  yy == y
    @test yy isa CA.CategoricalVector
    @test CA.levels(yy) == CA.levels(y)

    # multi-target:
    o = fitobs("learner", (x, ymulti), Sage(multitarget=true))
    @test o.features == x
    @test o.names == [:x1, :x2]
    @test o.target == hcat(repeat([1, 2, 1], n), repeat([2, 1, 2], n)) |> permutedims
    @test eltype(o.target) == Int
    yy = o.decoder.(o.target)
    @test  yy == ymulti
    @test yy isa CA.CategoricalMatrix
    @test CA.levels(yy) == CA.levels(ymulti)
end

@testset "`fitobs(learner, (X, y))`" begin
    o = fitobs("learner", (X, y), Sage())
    @test o.features == x
    @test o.names == [:c, :a]
    @test o.target == repeat([1, 2, 1], n)
    @test eltype(o.target) == Int
    @test o.classes_seen == CA.levels(y)[1:2]
    @test o.classes_seen isa CA.CategoricalArray
    yy = o.decoder.(o.target)
    @test  yy == y
    @test yy isa CA.CategoricalVector
    @test CA.levels(yy) == CA.levels(y)


    # multi-target:
    o = fitobs("learner", (X, ymulti), Sage(multitarget=true))
    @test o.features == x
    @test o.names == [:c, :a]
    @test o.target == hcat(repeat([1, 2, 1], n), repeat([2, 1, 2], n)) |> permutedims
    @test eltype(o.target) == Int
    yy = o.decoder.(o.target)
    @test  yy == ymulti
    @test yy isa CA.CategoricalMatrix
    @test CA.levels(yy) == CA.levels(ymulti)
end

@testset "`fitobs(learner, (x, Y))`" begin
    o = fitobs("learner", (x, Y), Sage())
    @test o.features == x
    @test o.target == repeat([1, 2, 1], n)
    @test eltype(o.target) == Int
    @test o.classes_seen == CA.levels(y)[1:2]
    @test o.classes_seen isa CA.CategoricalArray
    yy = o.decoder.(o.target)
    @test  yy == y
    @test yy isa CA.CategoricalVector
    @test CA.levels(yy) == CA.levels(y)
    @test o.names == [:x1, :x2]

    # multi-target:
    o = fitobs("learner", (x, ymulti), Sage(multitarget=true))
    @test o.features == x
    @test o.names == [:x1, :x2]
    @test o.target == hcat(repeat([1, 2, 1], n), repeat([2, 1, 2], n)) |> permutedims
    @test eltype(o.target) == Int
    yy = o.decoder.(o.target)
    @test  yy == ymulti
    @test yy isa CA.CategoricalMatrix
    @test CA.levels(yy) == CA.levels(ymulti)
end

@testset "`fitobs(learner, (X, Y))`" begin
    o = fitobs("learner", (X, Y), Sage())
    @test o.features == x
    @test o.names == [:c, :a]
    @test o.target == repeat([1, 2, 1], n)
    @test eltype(o.target) == Int
    @test o.classes_seen == CA.levels(y)[1:2]
    @test o.classes_seen isa CA.CategoricalArray
    yy = o.decoder.(o.target)
    @test  yy == y
    @test yy isa CA.CategoricalVector
    @test CA.levels(yy) == CA.levels(y)

    # multi-target:
    o = fitobs("learner", (X, ymulti), Sage(multitarget=true))
    @test o.features == x
    @test o.names == [:c, :a]
    @test o.target == hcat(repeat([1, 2, 1], n), repeat([2, 1, 2], n)) |> permutedims
    @test eltype(o.target) == Int
    yy = o.decoder.(o.target)
    @test  yy == ymulti
    @test yy isa CA.CategoricalMatrix
    @test CA.levels(yy) == CA.levels(ymulti)
end

@testset "fitobs(learner, (T, formula))`" begin
    o = fitobs("learner", (T, f), Sage())
    @test o.features == x
    @test o.names == [:c, :a]
    @test o.target == repeat([1, 2, 1], n)
    @test eltype(o.target) == Int
    @test o.classes_seen == CA.levels(y)[1:2]
    @test o.classes_seen isa CA.CategoricalArray
    yy = o.decoder.(o.target)
    @test  yy == y
    @test yy isa CA.CategoricalVector
    @test CA.levels(yy) == CA.levels(y)
end

# from test/_some_learners.jl:
learner = LearnerReportingNames()
model = fit(learner, X)

@testset "`obs(model, x)`" begin
    o = obs(model, x, Sage())
    @test o.features == x
    @test o.names == [:c, :a]
end

@testset "`obs(model, X)`" begin
    o = obs(model, X, Sage())
    @test o.features == x
    @test o.names == [:c, :a]
end

@testset "`obs(model, (T, target))`" begin
    o = obs(model, (T, :t), Sage())
    @test o.features == x
    @test o.names == [:c, :a]
end

@testset "`obs(model, (T, formula))`" begin
    o = obs(model, (T, f), Sage())
    @test o.features == x
    @test o.names == [:c, :a]
end


# # A CONSANT CLASSIFIER USING SAFFRON FRONT END

# We need this for integration tests to follow.

"""
    ConstantClassifier()

Instantiate a constant (dummy) classifier. Can predict `Point` or `Distribution` targets.

"""
struct ConstantClassifier end

struct ConstantClassifierFitted
    learner::ConstantClassifier
    probabilities
    names::Vector{Symbol}
    classes_seen
    codes_seen
    decoder
end

Base.isapprox(
    model1::ConstantClassifierFitted,
    model2::ConstantClassifierFitted;
    kwargs...,
) = model1.learner == model2.learner &&
    isapprox(model1.probabilities, model2.probabilities; kwargs...) &&
    model1.names == model2.names

LearnAPI.learner(model::ConstantClassifierFitted) = model.learner

# following both return objects with abstract type `Obs`:
frontend = Sage(code_type=:small)
LearnAPI.obs(learner::ConstantClassifier, data) = fitobs(learner, data, frontend)
LearnAPI.obs(model::ConstantClassifierFitted, X) =  obs(model, X, frontend)

# data deconstructors:
LearnAPI.features(learner::ConstantClassifier, data) =
    LearnAPI.features(learner, data, frontend)
LearnAPI.target(learner::ConstantClassifier, data) =
    LearnAPI.target(learner, data, frontend)

function LearnAPI.fit(learner::ConstantClassifier, observations::Obs; verbosity=1)

    # Note we don't choose the most efficient solution here, but one that demonstrates
    # patterns that apply more generally to classifiers.

    y = observations.target # integer "codes"
    names = observations.names
    classes_seen = observations.classes_seen
    codes_seen = sort(unique(y))
    decoder = observations.decoder

    d = proportionmap(y)
    # proportions, ordered by key, i.e., by codes seen:
    probabilities = values(sort!(OrderedDict(d))) |> collect

    return ConstantClassifierFitted(
        learner,
        probabilities,
        names,
        classes_seen,
        codes_seen,
        decoder,
    )
end
LearnAPI.fit(learner::ConstantClassifier, data; kwargs...) =
    fit(learner, obs(learner, data); kwargs...)

function LearnAPI.predict(model::ConstantClassifierFitted, ::Point, observations::Obs)
    n = MLCore.numobs(observations)
    idx = argmax(model.probabilities)
    code_of_mode = model.codes_seen[idx]
    return model.decoder.(fill(code_of_mode, n))
end
LearnAPI.predict(model::ConstantClassifierFitted, ::Point, Xnew) =
    predict(model, Point(), obs(model, Xnew))

function LearnAPI.predict(model::ConstantClassifierFitted, ::Distribution, observations::Obs)
    n = MLCore.numobs(observations)
    probs = model.probabilities
    # repeat vertically to get rows of a matrix:
    probs_matrix = reshape(repeat(probs, n), (length(probs), n))'
    return CategoricalDistributions.UnivariateFinite(model.classes_seen, probs_matrix)
end
LearnAPI.predict(model::ConstantClassifierFitted, ::Distribution, Xnew) =
        predict(model, Distribution(), obs(model, Xnew))

# accessor function:
LearnAPI.feature_names(model::ConstantClassifierFitted) = model.names

@trait(
    ConstantClassifier,
    constructor = ConstantClassifier,
    kinds_of_proxy = (Point(),Distribution()),
    tags = ("classification",),
    functions = (
        :(LearnAPI.fit),
        :(LearnAPI.learner),
        :(LearnAPI.clone),
        :(LearnAPI.strip),
        :(LearnAPI.obs),
        :(LearnAPI.features),
        :(LearnAPI.target),
        :(LearnAPI.predict),
        :(LearnAPI.feature_names),
   )
)


# # INTEGRATION TESTS

learner = ConstantClassifier()

@testset "tabular features for fit" begin
    models1 = map(Xs) do X
        observations = obs(learner, (X, y))
        @test observations.target isa Vector{<:Unsigned}
        @test LearnAPI.target(learner, observations) == y
        feats = LearnAPI.features(learner, observations)
        @test feats isa Obs
        @test feats.features == x
        @test feats.names == [:c, :a]
        model = fit(learner, (X, y))
        @test LearnAPI.feature_names(model) == [:c, :a]
        model
    end
    models2 = map(Ts) do T
        observations = obs(learner, (T, :t))
        @test LearnAPI.target(learner, observations) == y
        feats = LearnAPI.features(learner, observations)
        @test feats isa Obs
        @test feats.features == x
        @test feats.names == [:c, :a]
        model = fit(learner, (T, :t))
        @test LearnAPI.feature_names(model) == [:c, :a]
        model
    end
    m = first(models1)
    models = vcat(models1[2:end], models2)
    @test all(≈(m), models)
end

model0 = fit(learner, (X, y))

@testset "matrix features for fit" begin
    observations = obs(learner, (x, y))
    @test LearnAPI.target(learner, observations) == y
    feats = LearnAPI.features(learner, observations)
    @test feats isa Obs
    @test feats.features == x
    @test feats.names == [:x1, :x2]
    model = fit(learner, (x, y))
    @test LearnAPI.feature_names(model) == [:x1, :x2]
end

@testset "tabular input to point predict" begin
    @test_throws(
        LearnDataFrontEnds.ERR_INCONSISTENT_FEATURES,
        predict(model0, (; a, c)), # feature names reversed
    )
    ys1 = map(Xs) do X
        predict(model0, X)
    end
    ys2 = map(Ts) do T
        predict(model0, (T, :t))
    end
    y0 = first(ys1)
    @test y0 isa CategoricalArrays.CategoricalVector
    @test length(y0) == 3n
    ys = vcat(ys1[2:end], ys2)
    @test all(==(y0), ys)
end

@testset "matrix input to predict" begin
    @test_throws(
        LearnDataFrontEnds.ERR_INCONSISTENT_FEATURE_COUNT(1),
        predict(model0, rand(1, 3)) # only one feature
    )
    @test predict(model0, x) == predict(model0, X)
end

@testset "probabilistic prediction" begin
    yhat = predict(model0, Distribution(), X)
    @test first(yhat) ≈ CategoricalDistributions.UnivariateFinite(proportionmap(y))
end

@testset "RandomAccess() interface" begin
    Xshort = Tables.subset(X, 1:3n-2)
    yshort = y[1:3n-2]

    # subsampling by hand:
    yhat = predict(fit(learner, (Xshort, yshort)), Xshort)

    # subsampling using `getobs`:
    fit_obs = obs(learner, (X, y))
    model = fit(
        learner,
        MLCore.getobs(fit_obs, 1:MLCore.numobs(fit_obs) - 2),
    )
    _obs = obs(model, X)
    yhat2 = predict(model, MLCore.getobs(_obs, 1:MLCore.numobs(_obs) - 2))

    # compare:
    @test yhat == yhat2
end

@testset "collection of a feature observation" begin
    observations = obs(learner, (x, y))
    features = LearnAPI.features(learner, observations)
    @test collect(MLCore.getobs(features, 1)) == x[:,1]
end

true
