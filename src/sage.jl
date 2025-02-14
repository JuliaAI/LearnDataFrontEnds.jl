"""
    Sage(; multitarget=false, view=false, code_type=:int)

A LearnAPI.jl data front end implemented for some supervised classifiers consuming
structured data. If `learner` implements this front end, then `data` in the call
[`LearnAPI.fit`](@extref)`(learner, data)` can take any of the following forms:

- `(X, y)`, where `X` is a feature matrix or table and `y` is a `CategoricalVector` or
  `CategoricalMatrix` or table with categorical columns.

- `(T, target)`, where `T` is a table and `target` is a column name or a tuple (not
  vector!)  of column names

- `(T, formula)`, where `formula` is an R-style formula, as provided by
  [StatsModels.jl](https://juliastats.org/StatsModels.jl/latest/)

$DOC_FORMULAS

Unlike StatsModels.jl, the left-hand side of a formula (the target) is not one-hot
encoded.

Similarly, if [`LearnAPI.learner`](@extref)`(model) == learner`, then `data` in the call
[`LearnAPI.predict`](@extref)`(model, data)` or [`LearnAPI.transform`](@extref)`(model,
data)` can take any of these forms:

- `X`, a feature matrix or table

- `(T, target)`, where `T` is a table and `target` is a column name or tuple of column
  names (for exclusion from `T`)

- `(T, formula)`, where `formula` is an R-style formula (left-hand side ignored)


Check `learner` documentation to see if it implements this front end.

# Back end API

When a learner implements the `Sage` front end, as described under "Extended help"
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

- `multiclass=false`: When `multitarget=true`, the internal representation of the the
  target is always a matrix, even if only a single target (e.g., vector) is
  presented. When `multitarget=false`, the internal representation of the target is always
  a vector.

- `view=false`: When tables are converted to matrices (and the roles of rows and
  columns are reversed) `transpose` is used if `view=true` and `permutedims` is used if
  `view=false`. The first option is only available for tables with transposable element
  types (e.g., floats).

- `code_type` determines the internal representation `y` of the target. Possible values are:

  - `:small`: the element type of `y` is the reference (code) type `R <: Unsigned` for the
    categorical array supplied by user (internal eltype for the array). Choose this to
    minimize memory requirements.

  - `:int`: `y` has an `Integer` element type `widen(R) <: Integer`. Choose this to
    safeguard against arithmetic overflows if these are likely; run `@doc widen` for
    details.


# Implementation

If a core algorithm is happy to work with a `CategoricalArray` target, without
integer-encoding it, consider using the [`Saffron`](@ref) frontend instead.

For learners of type `MyLearner`, with `LearnAPI.fit(::MyLearner, data)` returning
objects of type `MyModel`, implement the `Sage` data front
by making these declarations:

```julia
using LearnDataFrontEnds
const frontend = Sage() # see above for options

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
    decoder = observations.decoder
    classes_seen = observations.classes_seen
    feature_names = observations.names

    # do stuff with `X`, `y` and `feature_names`:
    # return a `model` object which also stores the `decoder` and/or
    # `classes_seen` to make them available to `predict`.
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

    # Do stuff with `X` and `model` to obtain raw `predictions` (a vector of integer
    # codes for `K = Point`, or an `n x c` matrix of probabilities for `K = Distribution`).
    # Extract `decoder` or `classes_seen` from `model`.
    # For `K = Point`, return `decoder.(predictions)`.
    # For `K = Distribution`, return, say,
    # `CategoricalDistributions.Univariate(classes_seen, predictions)`.
    ...
end
LearnAPI.predict(model::MyModel, kind_of_proxy, X) = LearnAPI.predict(model,
    kind_of_proxy, obs(model, X))
```

Don't forget to include `:(LearnAPI.target)` and `:(LearnAPI.features)` (unless `learner`
is static) in the return value of [`LearnAPI.functions`](@extref).

"""
Sage(; view=false, multitarget=false, code_type=:int) =
    Saffron(; view, multitarget, code_type)
