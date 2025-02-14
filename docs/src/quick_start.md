# Quick start

- [Supervised regressors](@ref)
- [Supervised classifiers](@ref)
- [Transformers](@ref)

	Refer to the front end [docstrings](@ref front_ends) for options ignored below.

## Supervised regressors

Supervised regressors handling structured data can typically add the `Saffron`
front end to their LearnAPI.jl implementations.

For a supervised learner of type `MyLearner`, with `LearnAPI.fit(::MyLearner, data)`
returning objects of type `MyModel`, make these declarations:

```julia
using LearnDataFrontEnds
const frontend = Saffron()

# both methods below return objects with abstract type `Obs`:
LearnAPI.obs(learner::MyLearner, data) = fitobs(learner, data, frontend)
LearnAPI.obs(model::MyModel, data) = obs(model, data, frontend)

# training data deconstructors:
LearnAPI.features(learner::MyLearner, data) = LearnAPI.features(learner, data, frontend)
LearnAPI.target(learner::MyLearner, data) = LearnAPI.target(learner, data, frontend)
```

Your [`LearnAPI.fit`](@ref) implementation will then look like this:

```julia
function LearnAPI.fit(
	learner::MyLearner,
	observations::Obs;
	verbosity=1,
	)
	X = observations.features # p x n matrix
	y = observations.target   # n-vector (use `Saffron(multitarget=true)` for matrix)
	feature_names = observations.names

	# do stuff with `X`, `y` and `feature_names`:
	...

end
LearnAPI.fit(learner::MyLearner, data; kwargs...) =
	LearnAPI.fit(learner, LearnAPI.obs(learner, data); kwargs...)
```

For each [`KindOfProxy`](@ref) subtype `K` to be supported (e.g., `Point`), your
[`LearnAPI.predict`](@ref) implementation(s) will look like this:

```julia
function LearnAPI.predict(model::MyModel, :K, observations::Obs)
	X = observations.features # p x n matrix
	names = observations.names # if really needed

	# do stuff with `X`:
	...
end
LearnAPI.predict(model::MyModel, kind_of_proxy, X) =
	LearnAPI.predict(model, kind_of_proxy, obs(model, X))
```

## Supervised classifiers

Supervised classifiers handling structured data can typically add the `Sage`
front end to their LearnAPI.jl implementations.

For a supervised learner of type `MyLearner`, with `LearnAPI.fit(::MyLearner, data)`
returning objects of type `MyModel`, make these declarations:

```julia
using LearnDataFrontEnds
const frontend = Sage()

# both methods below return objects with abstract type `Obs`:
LearnAPI.obs(learner::MyLearner, data) = fitobs(learner, data, frontend)
LearnAPI.obs(model::MyModel, data) = obs(model, data, frontend)

# training data deconstructors:
LearnAPI.features(learner::MyLearner, data) = LearnAPI.features(learner, data, frontend)
LearnAPI.target(learner::MyLearner, data) = LearnAPI.target(learner, data, frontend)
```

Your [`LearnAPI.fit`](@ref) implementation will then look like this:

```julia
function LearnAPI.fit(
    learner::MyLearner,
    observations::Obs;
    verbosity=1,
    )
    X = observations.features # p x n matrix
    y = observations.target   # n-vector
    decoder = observations.decoder
    classes_seen = observatioins.classes_seen
    feature_names = observations.names

    # do stuff with `X`, `y` and `feature_names`:
    # return a `model` object which also stores the `decoder` and/or `classes_seen` 
	# to make them available to `predict`.
	...
end
LearnAPI.fit(learner::MyLearner, data; kwargs...) =
    LearnAPI.fit(learner, LearnAPI.obs(learner, data); kwargs...)
```

For each [`KindOfProxy`](@ref) subtype `K` to be supported (e.g., `Point`), your
[`LearnAPI.predict`](@ref) implementation(s) will look like this:

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


## Transformers

Transformers can typically add the `Tarragon` front end to their LearnAPI.jl
implementations. For simplicity, we assume below that the transformer is not static
(i.e., it generalizes to new data).

For your learners of type `MyLearner`, with `LearnAPI.fit(::MyLearner, data)` returning
objects of type `MyModel`, make these declarations:

```julia
using LearnDataFrontEnds
const frontend = Tarragon()

# both the following return objects with abstract type `Obs`:
LearnAPI.obs(model::MyModel, X) = obs(model, data, frontend)
LearnAPI.obs(learner::MyLearner, data) = fitobs(learner, data, frontend)

# training data deconstructors:
LearnAPI.features(learner::MyLearner, data) = LearnAPI.features(learner, data, frontend)
```

Your [`LearnAPI.fit`](@ref) implementation will then look like this:

```julia
function LearnAPI.fit(
	learner::MyLearner,
	observations::Obs;
	verbosity=1,
	)
	x = observations.features # p x n matrix
	feature_names = observations.names

	# do stuff with `x` and `feature_names`:
	...
end
LearnAPI.fit(learner::MyLearner, data; kwargs...) =
	LearnAPI.fit(learner, LearnAPI.obs(learner, data); kwargs...)
```

Your [`LearnAPI.transform`](@ref) implementation will look like this:

```julia
function LearnAPI.transform(model::MyModel, observations::Obs)
	x = observations.features # p x n matrix
	feature_names = observations.names # if really needed

	# do stuff with `x`:
	...
end
LearnAPI.transform(model::MyModel, X) = LearnAPI.transform(model, obs(model, X))
```

There is no need to overload [`LearnAPI.features`](@ref) for the output of `obs` but you
still need to include `:(LearnAPI.features)` in the return value of
[`LearnAPI.functions`](@ref).
