using Test
using LearnDataFrontEnds
using LearnAPI
import MLCore
using StableRNGs
import DataFrames
using Tables
using LinearAlgebra
import StatsModels.@formula

# include("_some_learners.jl")

n = 20
rng = StableRNG(345)
c, t, a, t2 = randn(rng, n), rand(rng, n), rand(rng, n), rand(rng, n)
y = t
Y = (; t)
ymulti = hcat(t, t2)'
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
    o = fitobs("learner", (x, y), Saffron())
    @test o.features == x
    @test o.names == [:x1, :x2]
    @test o.target == t

    # multi-target:
    o = fitobs("learner", (x, ymulti), Saffron(multitarget=true))
    @test o.features == x
    @test o.names == [:x1, :x2]
    @test o.target == ymulti
end

@testset "`fitobs(learner, (X, y))`" begin
    o = fitobs("learner", (X, y), Saffron())
    @test o.features == x
    @test o.names == [:c, :a]
    @test o.target == t

    # multi-target:
    o = fitobs("learner", (X, ymulti), Saffron(multitarget=true))
    @test o.features == x
    @test o.names == [:c, :a]
    @test o.target == ymulti
end

@testset "`fitobs(learner, (x, Y))`" begin
    o = fitobs("learner", (x, Y), Saffron())
    @test o.features == x
    @test o.names == [:x1, :x2]
    @test o.target == t

    # multi-target:
    o = fitobs("learner", (x, ymulti), Saffron(multitarget=true))
    @test o.features == x
    @test o.names == [:x1, :x2]
    @test o.target == ymulti
end

@testset "`fitobs(learner, (X, Y))`" begin
    o = fitobs("learner", (X, Y), Saffron())
    @test o.features == x
    @test o.names == [:c, :a]
    @test o.target == t

    # multi-target:
    o = fitobs("learner", (X, ymulti), Saffron(multitarget=true))
    @test o.features == x
    @test o.names == [:c, :a]
    @test o.target == ymulti
end

@testset "fitobs(learner, (T, formula))`" begin
    o = fitobs("learner", (T, f), Saffron())
    @test o.features == x
    @test o.names == [:c, :a]
    @test o.target == t
end

# from test/_some_learners.jl:
learner = LearnerReportingNames()
model = fit(learner, X)

@testset "`obs(model, x)`" begin
    o = obs(model, x, Saffron())
    @test o.features == x
    @test o.names == [:c, :a]
end

@testset "`obs(model, X)`" begin
    o = obs(model, X, Saffron())
    @test o.features == x
    @test o.names == [:c, :a]
end

@testset "`obs(model, (T, target))`" begin
    o = obs(model, (T, :t), Saffron())
    @test o.features == x
    @test o.names == [:c, :a]
end

@testset "`obs(model, (T, formula))`" begin
    o = obs(model, (T, f), Saffron())
    @test o.features == x
    @test o.names == [:c, :a]
end


# # A RIDGE IMPLEMENTATION USING SAFFRON FRONT END

# We need this for integration tests to follow.

struct Ridge
    lambda::Float64
end

"""
    Ridge(; lambda=0.1)

Instantiate a ridge regression learner, with regularization of `lambda`.

"""
Ridge(; lambda=0.1) = Ridge(lambda) # LearnAPI.constructor defined later

struct RidgeFitted{T}
    learner::Ridge
    coefficients::Vector{T}
    names::Vector{Symbol}
end

import Base.(==)
==(model1::RidgeFitted, model2::RidgeFitted) =
    model1.learner == model2.learner &&
    model1.coefficients == model2.coefficients &&
    model1.names == model2.names

LearnAPI.learner(model::RidgeFitted) = model.learner

# following both return objects with abstract type `Obs`:
LearnAPI.obs(learner::Ridge, data) = fitobs(learner, data, Saffron())
LearnAPI.obs(model::RidgeFitted, X) =  obs(model, X, Saffron())

# data deconstructors:
LearnAPI.features(learner::Ridge, data) = LearnAPI.features(learner, data, Saffron())
LearnAPI.target(learner::Ridge, data) = LearnAPI.target(learner, data, Saffron())

function LearnAPI.fit(learner::Ridge, observations::Obs; verbosity=1)

    # unpack hyperparameters and data:
    lambda = learner.lambda
    A = observations.features
    y = observations.target
    names = observations.names

    # apply core learner:
    coefficients = (A*A' + learner.lambda*I)\(A*y) # 1 x p matrix

    return RidgeFitted(learner, coefficients, names)

end
LearnAPI.fit(learner::Ridge, data; kwargs...) = fit(learner, obs(learner, data); kwargs...)

LearnAPI.predict(model::RidgeFitted, ::Point, observations::Obs) =
        (observations.features)'*model.coefficients
LearnAPI.predict(model::RidgeFitted, ::Point, Xnew) =
        predict(model, Point(), obs(model, Xnew))

# accessor function:
LearnAPI.feature_names(model::RidgeFitted) = model.names
LearnAPI.coefficients(model::RidgeFitted) = Pair.(model.names, model.coefficients)

@trait(
    Ridge,
    constructor = Ridge,
    human_name = "ridge regression",
    kinds_of_proxy = (Point(),),
    tags = ("regression",),
    functions = (
        :(LearnAPI.fit),
        :(LearnAPI.learner),
        :(LearnAPI.strip),
        :(LearnAPI.obs),
        :(LearnAPI.clone),
        :(LearnAPI.features),
        :(LearnAPI.target),
        :(LearnAPI.predict),
        :(LearnAPI.coefficients),
        :(LearnAPI.feature_names),
   )
)


# # INTEGRATION TESTS

learner = Ridge()

@testset "tabular features for fit" begin
    models1 = map(Xs) do X
        observations = obs(learner, (X, y))
        @test LearnAPI.target(learner, observations) == y
        feats = LearnAPI.features(learner, observations)
        @test feats isa LearnDataFrontEnds.BasicObs
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
        @test feats isa LearnDataFrontEnds.BasicObs
        @test feats.features == x
        @test feats.names == [:c, :a]
        model = fit(learner, (T, :t))
        @test LearnAPI.feature_names(model) == [:c, :a]
        model
    end
    m = first(models1)
    models = vcat(models1[2:end], models2)
    @test all(==(m), models)
end

model0 = fit(learner, (X, y))

@testset "matrix features for fit" begin
    observations = obs(learner, (x, y))
    @test LearnAPI.target(learner, observations) == y
    feats = LearnAPI.features(learner, observations)
    @test feats isa LearnDataFrontEnds.BasicObs
    @test feats.features == x
    @test feats.names == [:x1, :x2]
    model = fit(learner, (x, y))
    @test last.(LearnAPI.coefficients(model)) == last.(LearnAPI.coefficients(model0))
    @test LearnAPI.feature_names(model) == [:x1, :x2]
end

@testset "tabular input to predict" begin
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
    @test y0 isa Vector{Float64}
    @test length(y0) == n
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

@testset "RandomAccess() interface" begin
    Xshort = Tables.subset(X, 1:n-2)
    yshort = y[1:n-2]

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
