using Test
using LearnDataFrontEnds
import CategoricalArrays

@testset "show methods" begin
    x = fill(0,2,3)
    y = [3, 2, 1]
    names = [:x1, :x2]
    ycat = CategoricalArrays.categorical(y)
    c = LearnDataFrontEnds.classes(ycat)
    d = LearnDataFrontEnds.decoder(ycat)
    mime =  MIME"text/plain"()

    @test sprint(show, mime, LearnDataFrontEnds.BasicObs(x, names)) ==
        "BasicObs\n  features :: Matrix{Int64}((2, 3))\n  names: [:x1, :x2]\n"
    @test sprint(show, mime, LearnDataFrontEnds.SaffronObs(x, names, y)) ==
        "SaffronObs\n  features :: Matrix{Int64}((2, 3))\n  names: "*
        "[:x1, :x2]\n  target :: Vector{Int64}((3,))"
    @test sprint(show, mime, LearnDataFrontEnds.SageObs(x, names, y, c, d)) ==
        "SageObs\n  features :: Matrix{Int64}((2, 3))\n  names: "*
        "[:x1, :x2]\n  target :: Vector{Int64}((3,))\n  classes_seen: "*
        "[1, 2, 3] (categorical vector with complete pool)\n  decoder: <callable>"
end

true
