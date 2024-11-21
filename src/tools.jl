const ERR_INCONSISTENT_FEATURES = ArgumentError(
    "Feature names encountered are different from those presented in "*
        "fit or update. "
)

ERR_INCONSISTENT_FEATURE_COUNT(p) = ArgumentError(
    "The number of features presented, $p, is inconsistent with that seen in training. "*
        "Perhaps you have presented a matrix with observations-as-rows instead of "*
        "observations-as-columns? "
)


ERR_BAD_TARGETS(omit) = ArgumentError(
    "One or more feature names in `$omit` do not appear to exist in the "*
        "table provided. "
)

"""
    feature_names(model, names_apparent)

*Private method.*

Return the feature names recorded in model where available, and check these agree with
`names_apparent` (a list of names or an integer count).

In more detail:

If the names are available, meaning `:(LearnAPI.feature_names) in
LearnAPI.functions(learner)`, for `learner = LearnAPI.learner(model)`, then:

- If `names_apparent` is an integer, throw an exception if `LearnAPI.feature_names(model)`
  does not have this integer as length.

- Otherwise, throw an exception if `LearnAPI.feature_names(model)` is different from
  `names_apparent`.

If feature names are not recorded in training, then return `Symbol[]`.

"""
function feature_names(model, names_apparent)
    :(LearnAPI.feature_names) in LearnAPI.functions(LearnAPI.learner(model)) ||
        return Symbol[]
    names = LearnAPI.feature_names(model)
    names_apparent == names || throw(ERR_INCONSISTENT_FEATURES)
    return names
end
function feature_names(model, p::Int)
    :(LearnAPI.feature_names) in LearnAPI.functions(LearnAPI.learner(model)) ||
        return Symbol[]
    names = LearnAPI.feature_names(model)
    p == length(names) || throw(ERR_INCONSISTENT_FEATURE_COUNT(p))
    return names
end

"""
    canonify(y, m)

*Private method.*

Return `y` as a matrix, if `m == Multitarget()`, or as a vector otherwise.

"""
canonify(y::AbstractMatrix, ::Multitarget) = y
canonify(y::AbstractVector, ::Multitarget) = reshape(y, 1, length(y))
canonify(y, ::Unitarget) = vec(y)

"""
    swapdims(A, v)

*Private method.*

Return `transose(A)` if `v == DoView()`, and `permutedims(A)` if `v == DontView()`.

"""
swapdims(A, ::DoView) = transpose(A)
swapdims(A, ::DontView) = permutedims(A)

"""
    decompose(X, v, targets=())

*Private method.*

Return `(A, names, B)` where:

- `A` is the matrix form of those columns of table `X` with names not in `targets` (a
  single symbol or vector thereof)

- `names` is those column names not in `targets`

- `B` is the matrix form of those columns with names in `targets`

The columns of `A` and `B` always correspond to rows of `X`. However, if `v == DoView()`
then `A` and `B` are `Transpose`s; otherwise they are regular `Matrix`s.

An informative exception is thrown if `target` contains names that are not the names of
columns of `X`.

"""
decompose(X, v) = decompose(X, v, ())
decompose(X, v, target) = decompose(X, v, (Symbol(target),))
function decompose(X, v, _targets::NTuple)
    targets = Symbol.(_targets)
    table = Tables.columns(X)
    targets ⊆ Tables.columnnames(table) || throw(ERR_BAD_TARGETS(targets))
    zipped = zip(values(table), keys(table))
    filtered = filter(collect(zipped)) do (col, name)
        !(name in targets)
    end
    cols, names = zip(filtered...) |> collect
    A = hcat(cols...)
    B = hcat(Tables.getcolumn.(Ref(table), targets)...)
    return swapdims(A, v), collect(names), swapdims(B, v)
end

"""
    classes(x)

*Private method.*

Return, as a `CategoricalVector`, all the categorical elements with
the same pool as `CategoricalValue` `x` (including `x`), with an
ordering consistent with the pool. Note that `x in classes(x)` is
always true.

Not to be confused with `levels(x.pool)`. See the example below.

Also, overloaded for `x` a `CategoricalArray`, `CategoricalPool`, and for views of
`CategoricalArray`.

    julia>  v = categorical(['c', 'b', 'c', 'a'])
    4-element CategoricalArrays.CategoricalArray{Char,1,UInt32}:
     'c'
     'b'
     'c'
     'a'

    julia> levels(v)
    3-element Array{Char,1}:
     'a'
     'b'
     'c'

    julia> x = v[4]
    CategoricalArrays.CategoricalValue{Char,UInt32} 'a'

    julia> classes(x)
    3-element CategoricalArrays.CategoricalArray{Char,1,UInt32}:
     'a'
     'b'
     'c'

    julia> levels(x.pool)
    3-element Array{Char,1}:
     'a'
     'b'
     'c'

"""
classes(p::CategoricalArrays.CategoricalPool) = [p[i] for i in 1:length(p)]
classes(x::CategoricalArrays.CategoricalValue) = classes(CategoricalArrays.pool(x))
classes(v::CategoricalArrays.CategoricalArray) = classes(CategoricalArrays.pool(v))
classes(v::SubArray{<:Any, <:Any, <:CategoricalArrays.CategoricalArray}) = classes(parent(v))


struct CategoricalDecoder{V,R}
    classes::CategoricalArrays.CategoricalVector{
        V,
        R,
        V,
        CategoricalArrays.CategoricalValue{V,R}, Union{},
    }
end

"""
    d = decoder(x)

A callable object for decoding the integer representation of a
`CategoricalValue` sharing the same pool as the `CategoricalValue`
`x`. Specifically, one has `d(int(y)) == y` for all `y` in the same
pool as `x`.

    julia> v = categorical(['c', 'b', 'c', 'a'])
    julia> levelcode(v)
    4-element Array{Int64,1}:
     3
     2
     3
     1
    julia> d = decoder(v[3])
    julia> d.(levelcode.(v)) == v
    true

*Warning:* There is no guarantee that `levelcode.(d.(u)) == u` will always holds.

"""
decoder(x) = CategoricalDecoder(classes(x))

(d::CategoricalDecoder{V,R})(i::Integer) where {V,R} =
    CategoricalArrays.CategoricalValue{V,R}(d.classes[i])
