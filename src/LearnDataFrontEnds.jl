"""
    LearnDataFrontEnds

Module providing the following commonly applicable data front ends for implementations of
the [LearnAPI.jl](https://juliaai.github.io/LearnAPI.jl/dev/) interface:

- [`Saffron`](@ref): good for most supervised regressors operating on structured data

- [`Sage`](@ref): good for most supervised classifiers operating on structured data

- [`Tarragon`](@ref): good for most transformers

See [`Obs`](@ref) for the corresponding back end API (the interface for the output of
[`obs`](@extref))

# Why add a front end from this package?

- *Users* get to specify data in flexible ways: ordinary arrays or most tabular formats
  supported by [Tables.jl](https://tables.juliadata.org/stable/). Targets or multitargets
  can be specified separately, or by column name(s). Standard data preprocessing, such as
  one-hot encoding and adding higher order feature interactions, can be specified by an
  R-style
  ["formula"](https://juliastats.org/StatsModels.jl/stable/formula/#The-@formula-language),
  as provided by [StatsModels.jl](https://juliastats.org/StatsModels.jl/stable/).

- *Developers* can focus on core algorithm development, in which data conforms to a
  standard interface; see [`Obs`](@ref).

"""
module LearnDataFrontEnds


import LearnAPI
import LearnAPI.obs
import Tables
import MLCore
import StatsModels
import CategoricalArrays

# aliases:
const MatrixOrVector = Union{AbstractMatrix,AbstractVector}
const StringOrSymbol = Union{AbstractString,Symbol}

# switches:
struct DoView end
struct DontView end
struct Unitarget end
struct Multitarget end
struct RawCode end
struct IntCode end
struct SmallIntCode end


abstract type FrontEnd end

include("backends.jl")
include("constants.jl")
include("tools.jl")
include("saffron.jl")
include("sage.jl")
include("tarragon.jl")

export obs, fitobs
export Saffron, Sage, Tarragon
export Obs

end # module
