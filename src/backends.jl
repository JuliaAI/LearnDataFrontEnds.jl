"""
    Obs

Abstract type for all "observations" returned by learners implementing a front end from
LearnDataFrontEnds.jl - that is, for any object returned by `LearnAPI.obs(learner, data)`
or `LearnAPI.obs(model, data)`, where `learner` implements such a front end and `model` is
an object returned by `fit(learner, ...)`.

Any instance, `observations`, supports the following property access:

- `observations.features`: size `(p, n)` feature matrix (`n` the number of observations)

- `observations.names`: length `p` vector of feature names (as symbols)

Any instance `observations` also implements the [`LearnAPI.RandomAccess`](@extref)
interface for accessing individual observations, for purposes of resampling, for example.

# Specific to `Saffron` and `Sage`

Additionally, when `observations = fit(learner, data)` and the
[`Saffron`](@ref)`(multitarget=...)` or [`Sage`](@ref)`(multitarget=...)` front end has
been implemented, one has:

- `observations.target`: length `n` target vector (`multitarget=false`) or size `(q, n)`
  target matrix (`multivariate=true`); this array has the same element type as the
  user-provided one in the `Saffron` case

# Specific to `Sage`

If [`Sage`](@ref)`(multitarget=..., code_type=...)` has been implemented, then
`observations.target` has an integer element type controlled by `code_type`, and we
additionally have:

- `observations.classes`: A categorical vector of the ordered target classes, as actually
  seen in the user-supplied target, with the full pool of classes available by applying
  `Categorical.levels` to the result. The corresponding integer codes will be
  `sort(unique(observations.target))`.

- `observations.decoder`: A callable function that converts an integer code back to the
  original `CategoricalValue` it represents.

Pass the first onto `predict` for making probabilistic predictions, and the second for
point predictions; see [`Sage`](@ref) for details.

"""
abstract type Obs end


# # BASIC OBS

# for features and their names

struct BasicObs{F} <: Obs
    features::F  # p x n
    names::Vector{Symbol}
end

function Base.show(io::IO, ::MIME"text/plain", observations::BasicObs)
    A = observations.features
    println(io, "BasicObs")
    println(io, "  features :: $(typeof(A))($(size(A)))")
    println(io, "  names: $(observations.names)")
end


# `getobs/numobs` interface:
Base.getindex(observations::BasicObs, idx) =
    BasicObs(
        MLCore.getobs(observations.features, idx),
        observations.names,
    )
Base.length(observations::BasicObs) = size(observations.features) |> last
Base.collect(observations::BasicObs) = observations.features


# # SAFFRON OBS

# for features, names and raw target

struct SaffronObs{F,T} <: Obs
    features::F  # p x n
    names::Vector{Symbol}
    target::T
end

function Base.show(io::IO, ::MIME"text/plain", observations::SaffronObs)
    A = observations.features
    y = observations.target
    println(io, "SaffronObs")
    println(io, "  features :: $(typeof(A))($(size(A)))")
    println(io, "  names: $(observations.names)")
    print(io, "  target :: $(typeof(y))($(size(y)))")
end

# `getobs/numobs` interface:
Base.getindex(observations::SaffronObs, idx) =
    SaffronObs(
        MLCore.getobs(observations.features, idx),
        observations.names,
        MLCore.getobs(observations.target, idx),
    )

Base.length(observations::SaffronObs) = size(observations.features) |> last


# # SAGE OBS

# for features, names, and an integer-encoded categorical target

struct SageObs{F,T,E,D} <: Obs
    features::F  # p x n
    names::Vector{Symbol}
    target::T
    classes_seen::CategoricalArrays.CategoricalVector{E}
    decoder::D
end

function Base.show(io::IO, ::MIME"text/plain", observations::SageObs)
    A = observations.features
    y = observations.target
    println(io, "SageObs")
    println(io, "  features :: $(typeof(A))($(size(A)))")
    println(io, "  names: $(observations.names)")
    println(io, "  target :: $(typeof(y))($(size(y)))")
    println(io, "  classes_seen: "*
        "$(CategoricalArrays.unwrap.(observations.classes_seen)) "*
        "(categorical vector with complete pool)")
    print(io, "  decoder: <callable>")
end

Base.getindex(observations::SageObs, idx) =
    SageObs(
        MLCore.getobs(observations.features, idx),
        observations.names,
        MLCore.getobs(observations.target, idx),
        observations.classes_seen,
        observations.decoder,
    )

Base.length(observations::SageObs) = size(observations.features) |> last
