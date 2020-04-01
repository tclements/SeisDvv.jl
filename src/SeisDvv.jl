module SeisDvv

using GLM, FFTW, Interpolations, SeisNoise

include("VelocityChange/MWCS.jl")
include("VelocityChange/Stretching.jl")
include("VelocityChange/Wavelets.jl")

end 
