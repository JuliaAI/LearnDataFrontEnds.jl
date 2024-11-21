using Documenter
using DocumenterInterLinks
using LearnDataFrontEnds

const  REPO = Remotes.GitHub("JuliaAI", "LearnDataFrontEnds.jl")

const links = InterLinks(
    "LearnAPI" => (
        "https://juliaai.github.io/LearnAPI.jl/dev/",
        "https://juliaai.github.io/LearnAPI.jl/dev/objects.inv",
        # todo: after 0.2 is released, use next two lines instead:
        # "https://juliaai.github.io/LearnAPI.jl/dev/",
        # "https://juliaai.github.io/LearnAPI.jl/dev/objects.inv",
    ),
);

makedocs(
    modules=[LearnDataFrontEnds,],
    format=Documenter.HTML(
        prettyurls = true,#get(ENV, "CI", nothing) == "true",
        collapselevel = 1,
    ),
    plugins=[links,],
    pages=[
        "Home" => "index.md",
        "Quick start" => "quick_start.md",
        "Reference" => "reference.md",
    ],
    sitename="LearnDataFrontEnds.jl",
    warnonly = [:cross_references, :missing_docs],
    repo = Remotes.GitHub("JuliaAI", "LearnDataFrontEnds.jl"),
)

deploydocs(
    devbranch="dev",
    push_preview=false,
    repo="github.com/JuliaAI/LearnDataFrontEnds.jl.git",
)
