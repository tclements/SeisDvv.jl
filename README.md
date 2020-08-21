# SeisDvv

Seismic dv/v analysis in Julia!

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://tclements.github.io/SeisDvv.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://tclements.github.io/SeisDvv.jl/dev)
[![Build Status](https://travis-ci.com/tclements/SeisDvv.jl.svg?branch=master)](https://travis-ci.com/tclements/SeisDvv.jl)

**SeisDvv.jl** is a Julia package for computing the relative change in seismic velocity, or dv/v. SeisDvv.jl supplies time-, frequency-, and wavelet-domain methods for calculating dv/v. Currently implemented algorithms include
* Time-Domain
  * Windowed Cross-Correlation (WCC)
  * Trace Stretching (TS)
  * Dynamic Time Warping (DTW)
* Frequency-Domain
  * Moving-Window Cross Spectrum
* Wavelet-Domain
  * Wavelet Cross-Spectrum (WXS)
  * Wavelet-Transform Stretching (WTS)
  * Wavelet-Transform Dynamic Time Warping (WTDTW)

## Installation

```julia
julia> using Pkg; Pkg.add(PackageSpec(url="https://github.com/tclements/SeisDvv.jl", rev="master"))
```

## Quickstart

Follow along with the [tutorial](https://nextjournal.com/a/MQdVdjuUHy6TGTRcdAAbH?token=SH2k5KRceh9VFTZmLFG5yJ "dv/v tutorial")!