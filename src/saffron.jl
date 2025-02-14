"""
    Saffron(; multitarget=false, view=false)

A LearnAPI.jl data front end implemented for some supervised learners, typically
regressors, consuming structured data. If `learner` implements this front end, then `data`
in the call [`LearnAPI.fit`](@extref)`(learner, data)` can take any of the following
forms:

- `(X, y)`, where `X` is a feature matrix or table and `y` is a target vector, matrix or
  table

- `(T, target)`, where `T` is a table and `target` is a column name or a tuple (not
  vector!)  of column names

- `(T, formula)`, where `formula` is an R-style formula, as provided by
  [StatsModels.jl](https://juliastats.org/StatsModels.jl/latest/)

$DOC_FORMULAS

Similarly, if [`LearnAPI.learner`](@extref)`(model) == learner`, then `data` in the call
[`LearnAPI.predict`](@extref)`(model, data)` or [`LearnAPI.transform`](@extref)`(model,
data)` can take any of these forms:

- `X`, a feature matrix or table

- `(T, target)`, where `T` is a table and `target` is a column name or tuple of column
  names (for exclusion from `T`)

- `(T, formula)`, where `formula` is an R-style formula (left-hand side ignored)


Check `learner` documentation to see if it implements this front end.

# Back end API

When a learner implements the `Saffron` front end, as described under "Extended help"
below, the objects returned by [`LearnAPI.obs`](@extref)`(learner, data)` and
[`LearnAPI.obs`](@extref)`(model, data)` expose array representations of the features,
feature names, and target, as described under [`Obs`](@ref).

If `model` records feature names ([`LearnAPI.feature_names`](@extref) has
been implemented) then the front end checks that data presented to
[`LearnAPI.predict`](@extref)/[`LearnAPI.transform`](@extref) has feature names (or
feature count, in the case of matrices) consistent with what has been recorded in
training.

# Extended help

# Options

When `multitarget=true`, the internal representation of the the target is always a matrix,
even if only a single target (e.g., vector) is presented. When `multitarget=false`, the
internal representation of the target is always a vector.

When tables are converted to matrices (and so the roles of rows and columns are reversed)
`transpose` is used if `view=true` and `permutedims` is used if `view=false`. The first
option is only available for tables with transposable element types (e.g., floats).

# Implementation

For learners of type `MyLearner`, with `LearnAPI.fit(::MyLearner, data)` returning
objects of type `MyModel`, implement the `Saffron` data front
by making these declarations:

```julia
using LearnDataFrontEnds
const frontend = Saffron() # optionally specify `view=true` and/or `multitarget=true`

# both `obs` methods return objects of abstract type `Obs`:
LearnAPI.obs(learner::MyLearner, data) = fitobs(learner, data, frontend)
LearnAPI.obs(model::MyModel, X) = obs(model, data, frontend)

# training data deconstructors:
LearnAPI.features(learner::MyLearner, data) = LearnAPI.features(learner, data, frontend)
LearnAPI.target(learner::MyLearner, data) = LearnAPI.target(learner, data, frontend)
```

Your [`LearnAPI.fit`](@extref) implementation will then look like this:

```julia
function LearnAPI.fit(
    learner::MyLearner,
    observations::Obs;
    verbosity=1,
    )
    X = observations.features # p x n matrix
    y = observations.target   # n-vector or q x n matrix
    feature_names = observations.names

    # do stuff with `X`, `y` and `feature_names`:
    ...

end
LearnAPI.fit(learner::MyLearner, data; kwargs...) =
    LearnAPI.fit(learner, LearnAPI.obs(learner, data); kwargs...)
```

For each [`LearnAPI.KindOfProxy`](@extref) subtype `K` to be supported (e.g., `Point`),
your [`LearnAPI.predict`](@extref) implementation(s) will look like this:

```julia
function LearnAPI.predict(model::MyModel, :K, observations::Obs)
    X = observations.features # p x n matrix
    names = observations.names # if really needed

    # do stuff with `X` (and `names`):
    ...
end
```
with the final declaration
```
LearnAPI.predict(model::MyModel, kind_of_proxy, X) =
    LearnAPI.predict(model, kind_of_proxy, obs(model, X))
```

Don't forget to include `:(LearnAPI.target)` and `:(LearnAPI.features)` (unless `learner`
is static) in the return value of [`LearnAPI.functions`](@extref).

"""
struct Saffron{M,V,L} <: FrontEnd
    m::M # Multitarget() or SingleTarget()
    v::V # DoView() or DontView()
    l::L # RawCode(), IntCode() or SmallIntCode()
end

# here `code_type` is not public, but provides a switch to the `Sage` front end, when not
# equal to `:raw`; see `sage.jl`
function Saffron(; view=false, multitarget=false, code_type=:raw)
    m = multitarget ? Multitarget() : Unitarget()
    v = view ? DoView() : DontView()
    l = code_type == :int ? IntCode() :
        code_type == :small ? SmallIntCode() :
        code_type == :raw ? RawCode() :
        throw(ERR_BAD_LEVELS)
    return Saffron(m, v, l)
end


# # METHODS

# ## learner

finalize(x, names, y, ::Saffron{<:Any,<:Any,RawCode}) = SaffronObs(x, names, y)
finalize(x, names, y, ::Saffron{<:Any,<:Any,IntCode}) =
    finalize(x, names, y, CategoricalArrays.levelcode)
finalize(x, names, y, ::Saffron{<:Any,<:Any,SmallIntCode}) =
    finalize(x, names, y, CategoricalArrays.refcode)
function finalize(x, names, y, int)  # here `int` is `levelcode` or `refcode` function
    y isa Union{
        CategoricalArrays.CategoricalArray,
        SubArray{<:Any, <:Any, <:CategoricalArrays.CategoricalArray},
    } || throw(ERR_EXPECTED_CATEGORICAL)
    l = LearnDataFrontEnds.classes(y)
    u = unique(y)
    mask = map(in(u), l)
    _classes_seen = l[mask]
    _decoder = LearnDataFrontEnds.decoder(l)

    return SageObs(x, names, int.(y), _classes_seen, _decoder)
end

# for input `(x::AbstractMatrix, y::MatrixOrVector)`:
function fitobs(learner, data::Tuple{AbstractMatrix,MatrixOrVector}, frontend::Saffron)
    x, y = data
    names = [Symbol("x$i") for i in 1:first(size(x))]
    finalize(x, names, LearnDataFrontEnds.canonify(y, frontend.m), frontend)
end

# for input `(X, y::MatrixOrVector)`, `X` a table:
function fitobs(learner, data::Tuple{Any,MatrixOrVector}, frontend::Saffron)
    X, y = data
    x, names, _ = LearnDataFrontEnds.decompose(X, frontend.v)
    finalize(x, names, LearnDataFrontEnds.canonify(y, frontend.m), frontend)
end

# for input `(x::Matrix, Y)`, `Y` a table:
function fitobs(learner, data::Tuple{AbstractMatrix,Any}, frontend::Saffron)
    x, Y = data
    names = [Symbol("x$i") for i in 1:first(size(x))]
    y, _, _ = LearnDataFrontEnds.decompose(Y, frontend.v)
    finalize(x, names, LearnDataFrontEnds.canonify(y, frontend.m), frontend)
end

# for input `(X, Y)`, `X` and `Y` both tables:
function fitobs(learner, data::Tuple{Any,Any}, frontend::Saffron)
    X, Y = data
    x, names, _ = LearnDataFrontEnds.decompose(X, frontend.v)
    y, _, _ = LearnDataFrontEnds.decompose(Y, frontend.v)
    finalize(x, names, LearnDataFrontEnds.canonify(y, frontend.m), frontend)
end

# for input `(T, targets)`, `T` a table, `targets` a symbol or tuple thereof:
function fitobs(
    learner,
    data::Tuple{Any,Union{StringOrSymbol,NTuple{<:Any,<:StringOrSymbol}}},
    frontend::Saffron,
    )

    T, targets = data
    x, names, y = LearnDataFrontEnds.decompose(T, frontend.v, targets)
    finalize(x, names, LearnDataFrontEnds.canonify(y, frontend.m), frontend)
end

# for input `(T, formula)`, `T` a table, `formula` a StatsModels.jl formula:
function fitobs(
    learner,
    data::Tuple{Any,StatsModels.AbstractTerm},
    frontend::Saffron,
    )

    _T, formula = data

    # we do the following conversion, because StatsModels does it anyway and we need the
    # column names (different from final "coefficient" names in StatsModels):
    T = Tables.columntable(_T)
    allnames = Tables.columnnames(T)

    schema = StatsModels.schema(formula, T)
    formula = StatsModels.apply_schema(formula, schema)
    _, xtrans = StatsModels.modelcols(formula, T);
    x = LearnDataFrontEnds.swapdims(xtrans, frontend.v)
    targetnames = Symbol(first(StatsModels.termnames(formula))) |> tuple
    names = setdiff(allnames, targetnames)
    y = Tables.getcolumn(T, first(targetnames))

    finalize(x, names, LearnDataFrontEnds.canonify(y, frontend.m), frontend)
end

# involutivity:
fitobs(learner, observations::Union{SaffronObs,SageObs}, ::Saffron) = observations

# data deconstructors:
LearnAPI.features(learner, observations::Union{SaffronObs,SageObs}, ::Saffron) =
    BasicObs(observations.features, observations.names)
LearnAPI.features(learner, data, frontend::Saffron) =
    LearnAPI.features(learner, obs(learner, data), frontend)
LearnAPI.target(learner, observations::SaffronObs, ::Saffron) =
    observations.target
LearnAPI.target(learner, observations::SageObs, ::Saffron) =
    observations.decoder.(observations.target)
LearnAPI.target(learner, data, frontend::Saffron) =
    LearnAPI.target(learner, obs(learner, data), frontend)


# ## model


# for input `x::AbstractMatrix`:
function obs(model, x::AbstractMatrix, ::Saffron)
    p = first(size(x))
    names = LearnDataFrontEnds.feature_names(model, p)
    BasicObs(x, names)
end

# for input `X`, a table:
function obs(model, X, frontend::Saffron)
    x, names, _ = LearnDataFrontEnds.decompose(X, frontend.v)

    # checks column names are consistent with `model`,
    # if `model` records feature names:
    LearnDataFrontEnds.feature_names(model, names)
    BasicObs(x, names)
end

# for input `(T, targets)`, `T` a table, `targets` a symbol or tuple thereof
function obs(
    model,
    data::Tuple{Any,Union{StringOrSymbol,NTuple{<:Any,<:StringOrSymbol}}},
    frontend::Saffron,
    )

    T, targets = data
    x, names, _ = LearnDataFrontEnds.decompose(T, frontend.v, targets)

    # checks column names are consistent with `model`,
    # if `model` records feature names:
    LearnDataFrontEnds.feature_names(model, names)

    BasicObs(x, names)
end

# for input `(T, formula)`, `T` a table, `formula` a StatsModels formula:
function obs(
    model,
    data::Tuple{Any,StatsModels.AbstractTerm},
    frontend::Saffron,
    )

    _T, formula = data

    # we do the following conversion, because StatsModels does it anyway and we need the
    # column names (different from final "coefficient" names in StatsModels):
    T = Tables.columntable(_T)
    allnames = Tables.columnnames(T)

    schema = StatsModels.schema(formula, T)
    formula = StatsModels.apply_schema(formula, schema)
    _, xtrans = StatsModels.modelcols(formula, T);
    x = LearnDataFrontEnds.swapdims(xtrans, frontend.v)
    targetnames = Symbol(first(StatsModels.termnames(formula))) |> tuple
    names = setdiff(allnames, targetnames)

    # checks column names are consistent with `model`,
    # if `model` records feature names:
    LearnDataFrontEnds.feature_names(model, names)

    BasicObs(x, names)
end

# involutivity:
obs(model, observations::BasicObs, ::Saffron) = observations
