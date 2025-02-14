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
c, a = randn(rng, n), rand(rng, n)

# feature matrix:
x = hcat(c, a) |> permutedims

# feature tables:
X = (; c, a)

# formula
formula = @formula(0 ~ c + a)

# little letters, `x`, are for arrays
# big letters, `X`, are for tables

@testset "`fitobs(learner, x, ...)`" begin
    o = fitobs("learner", x, Tarragon())
    @test o.features == x
    @test o.names == [:x1, :x2]
end

@testset "`fitobs(learner, X, ...)`" begin
    o = fitobs("learner", X, Tarragon())
    @test o.features == x
    @test o.names == [:c, :a]
end

@testset "`fitobs(learner, (X, formula), ...)`" begin
    o = fitobs("learner", (X, formula), Tarragon())
    @test o.features == x
    @test o.names == [:c, :a]
end

# from test/_some_learners.jl:
learner = LearnerReportingNames()
model = fit(learner, X)

@testset "`obs(model, x, ...)`" begin
    o = obs(model, x, Tarragon())
    @test o.features == x
    @test o.names == [:c, :a]
end

@testset "`obs(model, X, ...)`" begin
    o = obs(model, X, Tarragon())
    @test o.features == x
    @test o.names == [:c, :a]
end

@testset "`obs(model, (X, formula), ...)`" begin
    o = obs(model, (X, formula), Tarragon())
    @test o.features == x
    @test o.names == [:c, :a]
end


# # A DIMENSION REDUCTION TRANSFORMER IMPLEMENTING THE TUMERIC FRONT END

# Needed for integration tests to follow.

struct TruncatedSVD
    codim::Int
end

"""
    TruncatedSVD(; codim=1)

Instantiate a truncated singular value decomposition algorithm for reducing the dimension
of observations by `codim`.

```julia
learner = Truncated()
X = rand(3, 100)  # 100 observations in 3-space
model = fit(learner, X)
W = transform(model, X)
```
"""
TruncatedSVD(; codim=1) = TruncatedSVD(codim)

struct TruncatedSVDFitted
    learner::TruncatedSVD
    U  # of size `(p - codim, p)` for input observations in `p`-space
    Ut # of size `(p, p - codim)`
    singular_values
    feature_names
end

import Base.(==)
==(model1::TruncatedSVDFitted, model2::TruncatedSVDFitted) =
    model1.learner == model2.learner &&
    model1.U == model2.U &&
    model1.singular_values == model2.singular_values &&
    model1.feature_names == model2.feature_names

LearnAPI.learner(model::TruncatedSVDFitted) = model.learner

# implementation of the `Tarragon` front end, both return object of type `Obs`:
LearnAPI.obs(learner::TruncatedSVD, data) = fitobs(learner, data, Tarragon())
LearnAPI.obs(model::TruncatedSVDFitted, data) = obs(model, data, Tarragon())

# training data deconstructor:
LearnAPI.features(learner::TruncatedSVD, data) =
    LearnAPI.features(learner, data, Tarragon())

function LearnAPI.fit(learner::TruncatedSVD, observations::Obs; verbosity=1)

    x = observations.features # p x n matrix
    names = observations.names

    # unpack hyperparameters:
    codim = learner.codim
    p, n = size(x)
    n ≥ p || error("Insufficient number observations. ")
    outdim = p - codim

    # apply core algorithm:
    result = svd(x)
    Ut = adjoint(@view result.U[:,1:outdim])
    U = adjoint(Matrix(Ut))
    singular_values = result.S
    dropped = singular_values[end - codim:end]

    return TruncatedSVDFitted(learner, U, Ut, singular_values, names)

end
LearnAPI.fit(learner::TruncatedSVD, data; kwargs...) =
    fit(learner, obs(learner, data); kwargs...)

LearnAPI.feature_names(model::TruncatedSVDFitted) = model.feature_names

LearnAPI.transform(model::TruncatedSVDFitted, observations::Obs) =
    model.Ut*(observations.features)
LearnAPI.transform(model::TruncatedSVDFitted, data) =
    transform(model, obs(model, data))

@trait(
    TruncatedSVD,
    constructor = TruncatedSVD,
    tags = ("dimension reduction", "transformers"),
    functions = (
        :(LearnAPI.fit),
        :(LearnAPI.learner),
        :(LearnAPI.strip),
        :(LearnAPI.obs),
        :(LearnAPI.clone),
        :(LearnAPI.features),
        :(LearnAPI.transform),
        :(LearnAPI.feature_names),
   )
)


# # INTEGRATION TESTS

# synthesize test data:
n = 100
rng = StableRNG(123)
r = svd(rand(rng, 5, n))
U, Vt = r.U, r.Vt
x = U*diagm([1, 2, 3, 0.01, 0.01])*Vt
X = Tables.table(x')
Xs = [
    X,
    Tables.rowtable(X),
    Tables.dictrowtable(X),
    Tables.dictcolumntable(X),
    DataFrames.DataFrame(X),
];

learner = TruncatedSVD(codim=2)

@testset "tabular features for fit" begin
    models = map(Xs) do X
        observations = obs(learner, X)
        feats = LearnAPI.features(learner, observations)
        @test feats isa Obs
        @test feats.features == x
        @test feats.names == [:Column1, :Column2, :Column3, :Column4, :Column5]
        model = fit(learner, X; verbosity=0)
        @test LearnAPI.feature_names(model) == feats.names
        model
    end
    m = first(models)
    @test all(==(m), models[2:end])
end

model0 = fit(learner, X)

@testset "matrix features for fit" begin
    observations = obs(learner, x)
    feats = LearnAPI.features(learner, observations)
    @test LearnAPI.features(learner, x).features == feats.features
    @test feats isa Obs
    @test feats.features == x
    @test feats.names == [:x1, :x2, :x3, :x4, :x5]
    model = fit(learner, x)
    @test model.U == model0.U
    @test LearnAPI.feature_names(model) == feats.names
end

@testset "tabular input to transform" begin
    @test_throws(
        LearnDataFrontEnds.ERR_INCONSISTENT_FEATURES,
        transform(model0, reverse(Tables.columntable(X))),
    )
    Ws = map(Xs) do X
        transform(model0, X)
    end
    W0 = first(Ws)
    @test W0 isa Matrix{Float64}
    @test size(W0)[2] == n
    @test all(==(W0), Ws[2:end])
        end

@testset "matrix input to transform" begin
    @test_throws(
        LearnDataFrontEnds.ERR_INCONSISTENT_FEATURE_COUNT(1),
        transform(model0, rand(1, 3)) # only one feature
    )
    @test transform(model0, x) == transform(model0, X)
end

@testset "RandomAccess() interface" begin
    Xshort = Tables.subset(X, 1:n-2)

    # subsampling by hand:
    W = transform(fit(learner, Xshort), Xshort)

    # subsampling using `getobs`:
    fit_obs = obs(learner, X)
    model = fit(
        learner,
        MLCore.getobs(fit_obs, 1:MLCore.numobs(fit_obs) - 2),
    )
    _obs = obs(model, X)
    W2 = transform(model, MLCore.getobs(_obs, 1:MLCore.numobs(_obs) - 2))

    # compare:
    @test W == W2
end

@testset "collection of a feature observation" begin
    observations = obs(learner, x)
    features = LearnAPI.features(learner, observations)
    @test collect(MLCore.getobs(features, 1)) == x[:,1]
end

true
