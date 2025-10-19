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

- `observations.levels`: A categorical vector of the ordered target levels, as actually
  seen in the user-supplied target. The corresponding integer codes will be
  `sort(unique(observations.target))`. To get the full pool of levels, apply
  `CategoricalArrays.levels` to `observations.levels_seen`; see the example below.

- `observations.decoder`: A callable function that converts an integer code back to the
  original `CategoricalValue` it represents.

Pass the first onto `predict` for making probabilistic predictions, and the second for
point predictions; see [`Sage`](@ref) for details.

# Extended help

In the example below, `observations` implements the full `Obs` interface described above,
for a learner implementing the `Sage` front end:

```julia-repl
using LearnAPI, LearnDataFrontEnds, LearnTestAPI
using CategoricalDistributions, CategoricalArrays, DataFrames
X = DataFrame(rand(10, 3), :auto)
y = categorical(collect("ababababac"))
learner = LearnTestAPI.ConstantClassifier()
observations = obs(learner, (X[1:9,:], y[1:9]))

julia> observations.features
3×9 Matrix{Float64}:
 0.234043  0.526468  0.227417  0.956471    …  0.00587146  0.169291  0.353518  0.402631
 0.631083  0.151317  0.781049  0.00320728     0.756519    0.15317   0.452169  0.127005
 0.285315  0.347433  0.69174   0.516915       0.900343    0.404006  0.448986  0.962649

julia> yint = observations.target
9-element Vector{UInt32}:
 0x00000001
 0x00000002
 0x00000001
 0x00000002
 0x00000001
 0x00000002
 0x00000001
 0x00000002
 0x00000001

julia> observations.levels_seen
2-element CategoricalArray{Char,1,UInt32}:
 'a'
 'b'

julia> sort(unique(observations.target))
2-element Vector{UInt32}:
 0x00000001
 0x00000002

julia> observations.levels_seen |> levels
3-element CategoricalArray{Char,1,UInt32}:
 'a'
 'b'
 'c'

julia> observations.decoder.(yint)
9-element CategoricalArray{Char,1,UInt32}:
 'a'
 'b'
 'a'
 'b'
 'a'
 'b'
 'a'
 'b'
 'a'

julia> d = UnivariateFinite(observations.levels_seen, [0.4, 0.6])
UnivariateFinite{Multiclass{3}}(a=>0.4, b=>0.6)

julia> levels(d)
3-element CategoricalArray{Char,1,UInt32}:
 'a'
 'b'
 'c'
```

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
    levels_seen::CategoricalArrays.CategoricalVector{E}
    decoder::D
end

function Base.show(io::IO, ::MIME"text/plain", observations::SageObs)
    A = observations.features
    y = observations.target
    println(io, "SageObs")
    println(io, "  features :: $(typeof(A))($(size(A)))")
    println(io, "  names: $(observations.names)")
    println(io, "  target :: $(typeof(y))($(size(y)))")
    println(io, "  levels_seen: "*
        "$(CategoricalArrays.unwrap.(observations.levels_seen)) "*
        "(categorical vector with complete pool)")
    print(io, "  decoder: <callable>")
end

Base.getindex(observations::SageObs, idx) =
    SageObs(
        MLCore.getobs(observations.features, idx),
        observations.names,
        MLCore.getobs(observations.target, idx),
        observations.levels_seen,
        observations.decoder,
    )

Base.length(observations::SageObs) = size(observations.features) |> last
