const DOC_FORMULAS = """
   In matrices, each column is an individual observation.

   See [`LearnAPI.RandomAccess`](@extref) for what constitutes a valid table.  When
   providing a formula, integer data is recast as `Float64` and, by default, non-numeric
   data is dummy-encoded as `Float64`. Refer to StatsModels.jl
   [documentation](https://juliastats.org/StatsModels.jl/latest/) for details.

"""

const ERR_BAD_LEVELS = ArgumentError(
    "`levels` must be one of these: `:raw`, `:int`, `:small`. "
)
