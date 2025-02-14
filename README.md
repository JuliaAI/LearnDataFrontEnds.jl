# LearnDataFrontEnds.jl

LearnDataFrontEnds.jl is a package for developers writing
[LearnAPI.jl](https://juliaai.github.io/LearnAPI.jl/dev/) interfaces for algorithms in
machine learning or statistics.


[![Lifecycle:Maturing](https://img.shields.io/badge/Lifecycle-Maturing-007EC6)]()
[![Build Status](https://github.com/JuliaAI/LearnDataFrontEnds.jl/workflows/CI/badge.svg)](https://github.com/JuliaAI/LearnDataFrontEnds.jl/actions)
[![codecov](https://codecov.io/gh/JuliaAI/LearnDataFrontEnds.jl/graph/badge.svg?token=tIJyIuCipO)](https://codecov.io/gh/JuliaAI/LearnDataFrontEnds.jl)
[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://juliaai.github.io/LearnDataFrontEnds.jl/dev/)

A LearnAPI.jl implementation including a data front end from this package gives users the
flexibility to supply data in a variety of ways (matrices, tables, R-style formulas, etc)
without the need to write the relevant boiler plate to coerce that data into array
representations ultimately needed by the core algorithm. Moreover, the design of
LearnAPI.jl means these conversions are not unnecessarily repeated when applying
meta-algorithms, such as cross-validation.

Documentation is [here](https://juliaai.github.io/LearnDataFrontEnds.jl/dev/).
