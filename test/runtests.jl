using Test

test_files = [
    "tools.jl",
    "saffron.jl",
    "sage.jl",
    "tarragon.jl",
]

files = isempty(ARGS) ? test_files : ARGS

include("_some_learners.jl")

for file in files
    quote
        @testset $file begin
            include($file)
        end
    end |> eval
end
