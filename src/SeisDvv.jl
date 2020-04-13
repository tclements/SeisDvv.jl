module SeisDvv

using GLM, Statistics, FFTW, Interpolations, SeisNoise

include("WCC.jl")
include("Stretching.jl")
include("DTW.jl")
include("MWCS.jl")
include("Wavelets.jl")

end
