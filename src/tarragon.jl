"""
    Tarragon(; view=false)

A LearnAPI.jl data front end, implemented for some transformers.  If `learner` implements
this front end, then `data` in the call [`LearnAPI.fit`](@extref)`(learner, data)` or
[`LearnAPI.transform`](@extref)`(model, data)`, where [`LearnAPI.learner`](@extref)`(model) == learner`, can take any of the following
forms:

- matrix

- table

- tuple `(T, formula)`, where `T` is a table and `formula` an R-style formula, as provided
  by [StatsModels.jl](https://juliastats.org/StatsModels.jl/latest/) (with `0` for the
  "left-hand side").

$DOC_FORMULAS

# Back end API

When a learner implements the `Tarragon` front end, as described under "Extended help"
below, the objects returned by [`LearnAPI.obs`](@extref)`(learner, data)` and
[`LearnAPI.obs`](@extref)`(model, data)` expose array representations of the features and
feature names, as described under [`Obs`](@ref).

If `fit` output records feature names (`LearnAPI.feature_names` has been implemented) then
the front end checks that data presented to [`LearnAPI.transform`](@extref) has feature
names (or feature count, in the case of matrices) consistent with what has been recorded
in training.

# Extended help

# Options

When tables are converted to matrices (and so the roles of rows and columns are reversed)
`transpose` is used if `view=true` and `permutedims` if `view=false`. The first option is
only available for tables with transposable element types (e.g., floats).

# Implementation

For learners of type `MyLearner`, with `LearnAPI.fit(::MyLearner, data)` returning
objects of type `MyModel`, implement the `Tarragon` data front
by making these declarations:

```julia
using LearnDataFrontEnds
const frontend = Tarragon() # optionally specify `view=true`

# both `obs` below return objects with abstract type `Obs`:
LearnAPI.obs(model::MyModel, data) = obs(model, data, frontend)
LearnAPI.obs(learner::MyLearner, data) = fitobs(learner, data, frontend)
LearnAPI.features(learner::MyLearner, data) = LearnAPI.features(learner, data, frontend)
```

Include the last two lines if your learner generalizes to new data, i.e.,
[`LearnAPI.fit`](@extref) has `data` in its signature). Assuming this is the case, your
[`LearnAPI.fit`](@extref) implementation will look like this:

```julia
function LearnAPI.fit(
    learner::MyLearner,
    observations::Obs;
    verbosity=1,
    )
    X = observations.features # p x n matrix
    feature_names = observations.names

    # do stuff with `X` and `feature_names`:
    ...

end
LearnAPI.fit(learner::MyLearner, data; kwargs...) =
    LearnAPI.fit(learner, LearnAPI.obs(learner, data); kwargs...)
```

Your [`LearnAPI.transform`](@extref) implementation will look like this:

```julia
function LearnAPI.transform(model::MyModel, observations::Obs)
    X = observations.features # p x n matrix
    feature_names = observations.names # if really needed

    # do stuff with `X`:
    ...
end
LearnAPI.transform(model::MyModel, X) = LearnAPI.transform(model, obs(model, X))
```

Remember to include `:(LearnAPI.features)` in the return value of
[`LearnAPI.functions`](@extref) if your learner generalizes to new data.

"""
struct Tarragon{V} <: FrontEnd
    v::V # DoView() or DontView()
end
function Tarragon(; view=false)
    v = view ? DoView() : DontView()
    return Tarragon(v)
end


# # METHODS

# ## learner

# for input `x::AbstractMatrix`:
function fitobs(learner, x::AbstractMatrix, frontend::Tarragon)
    names = [Symbol("x$i") for i in 1:first(size(x))]
    BasicObs(x, names)
end

# for input `X`, a table:
function fitobs(learner, X, frontend::Tarragon)
    x, names, _ = LearnDataFrontEnds.decompose(X, frontend.v)
    BasicObs(x, names)
end

# for input `(T, formula)`, `T` a table, `formula` a StatsModels.jl formula:
function fitobs(
    learner,
    data::Tuple{Any,StatsModels.FormulaTerm},
    frontend::Tarragon,
    )

    _T, formula = data

    # we do the following conversion, because StatsModels does it anyway and we need the
    # column names (different from final "coefficient" names in StatsModels):
    T = Tables.columntable(_T)
    names = Tables.columnnames(T) |> collect

    schema = StatsModels.schema(formula, T)
    formula = StatsModels.apply_schema(formula, schema)
    _, xtrans = StatsModels.modelcols(formula, T);
    x = LearnDataFrontEnds.swapdims(xtrans, frontend.v)
    BasicObs(x, names)
end

# involutivity:
fitobs(model, observations::BasicObs, ::Tarragon) = observations

# ## model

# for input `x::AbstractMatrix`:
function obs(model, x::AbstractMatrix, ::Tarragon)
    p = first(size(x))
    names = LearnDataFrontEnds.feature_names(model, p)
    BasicObs(x, names)
end

# for input `X`, a table:
function obs(model, X, frontend::Tarragon)
    x, names, _ = LearnDataFrontEnds.decompose(X, frontend.v)

    # checks column names are consistent with `model`,
    # if `model` records feature names:
    LearnDataFrontEnds.feature_names(model, names)
    BasicObs(x, names)
end

# for input `(T, formula)`, `T` a table, `formula` a StatsModels formula:
function obs(
    model,
    data::Tuple{Any,StatsModels.FormulaTerm},
    frontend::Tarragon,
    )

    _T, formula = data

    # we do the following conversion, because StatsModels does it anyway and we need the
    # column names (different from final "coefficient" names in StatsModels):
    T = Tables.columntable(_T)
    names = Tables.columnnames(T) |> collect

    schema = StatsModels.schema(formula, T)
    formula = StatsModels.apply_schema(formula, schema)
    _, xtrans = StatsModels.modelcols(formula, T);
    x = LearnDataFrontEnds.swapdims(xtrans, frontend.v)

    # checks column names are consistent with `model`,
    # if `model` records feature names:
    LearnDataFrontEnds.feature_names(model, names)

    BasicObs(x, names)
end

# involutivity:
obs(model, observations::BasicObs, ::Tarragon) = observations
