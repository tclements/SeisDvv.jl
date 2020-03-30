using Documenter, SeisDvv

makedocs(;
    modules=[SeisDvv],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/tclements/SeisDvv.jl/blob/{commit}{path}#L{line}",
    sitename="SeisDvv.jl",
    authors="Tim Clements <thclements@g.harvard.edu>",
    assets=String[],
)

deploydocs(;
    repo="github.com/tclements/SeisDvv.jl",
)
