export stretching

"""

  stretching(ref,cur,t,window,fmin,fmax;dvmin,dvmax,ntrials)

This function compares the Reference waveform to stretched/compressed current
waveforms to get the relative seismic velocity variation (and associated error).
It also computes the correlation coefficient between the Reference waveform and
the current waveform.

# Arguments
- `ref::AbstractArray`: Reference correlation.
- `cur::AbstractArray`: Current correlation.
- `t::AbstractArray`: time vector, common to both `ref` and `cur`.
- `window::AbstractArray`: vector of the indices of the `cur` and `ref` windows
                          on which you want to do the measurements
- `fmin::Float64`: minimum frequency in the correlation [Hz]
- `fmax::Float64`: maximum frequency in the correlation [Hz]
- `dvmin::Float64`: minimum bound for the velocity variation; e.g. dvmin=-0.03
                   for -3% of relative velocity change
- `dvmax::Float64`: maximum bound for the velocity variation; e.g. dvmin=0.03
                  for 3% of relative velocity change
- `ntrial::Int`:  number of stretching coefficient between dvmin and dvmax, no need to be higher than 100

# Returns
- `dvv::Float64`: Relative Velocity Change dv/v (in %)
- `cc::Float64`: Correlation coefficient between the reference waveform and the
                      best stretched/compressed current waveform
- `cdp::Float64`: Correlation coefficient between the reference waveform and the
                 initial current waveform
- `ϵ::Array{Float64,1}`: Vector of Epsilon values (ϵ =-dt/t = dv/v)
- `err::Float64`: Errors in the dv/v measurements based on [Weaver et al., 2011](https://onlinelibrary.wiley.com/doi/full/10.1111/j.1365-246X.2011.05015.x)
- `allC::Array{Float64,1}`: Vector of the correlation coefficient between the
                        reference waveform and every stretched/compressed
                        current waveforms

This code is a Julia translation of the Python code from [Viens et al., 2018](https://github.com/lviens/2018_JGR).
"""
function stretching(ref::AbstractArray, cur::AbstractArray, t::AbstractArray,
                    window::AbstractArray, fmin::Float64, fmax::Float64;
                    dvmin::Float64=-0.1, dvmax::Float64=0.1, ntrial::Int=100)
    ϵ = range(dvmin, stop=dvmax, length=ntrial)
    L = 1. .+ ϵ
    tau = t * L'
    allC = zeros(ntrial)

    # set of stretched/compressed current waveforms
    waveform_ref = ref[window]
    for ii = 1:ntrial
        s = LinearInterpolation(tau[:,ii],cur,extrapolation_bc=Flat())(t)
        waveform_cur = s[window]
        allC[ii] = cor(waveform_ref,waveform_cur)
    end

    cdp = cor(cur[window],ref[window])
    # find the maximum correlation coefficient
    imax = argmax(allC)
    if imax >= ntrial -1
        imax = ntrial - 2
    end
    if imax <= 3
        imax += 3
    end
    # Proceed to the second step to get a more precise dv/v measurement
    dtfiner = Array(range(ϵ[imax-2],stop=ϵ[imax+1],length=500))
    # to get same result as scipy, use line below; this requires gcc
    # using Dierckx; etp = Spline1D(ϵ[range(imax-3,stop=imax+1)],allC[range(imax-3,stop=imax+1)])
    etp = CubicSplineInterpolation(ϵ[range(imax-3,stop=imax+1)],allC[range(imax-3,stop=imax+1)])
    CCfiner = etp(dtfiner)
    dvv = 100. * dtfiner[argmax(CCfiner)]
    cc = maximum(CCfiner) # Maximum correlation coefficient of the refined analysis

    # Error computation based on Weaver, R., C. Hadziioannou, E. Larose, and M.
    # Campillo (2011), On the precision of noise-correlation interferometry,
    # Geophys. J. Int., 185(3), 1384?1392
    T = 1 / (fmax - fmin)
    X = cc
    # extremely similar signals can return cc>1.0 (not possible), so we limit cc to 1.0 to prevent sqrt(neg)
    if X > 1.0
        X=1.0
    end

    wc = π * (fmin + fmax)
    tmin = t[window][1]
    tmax = t[window][end]
    t1 = minimum([tmin,tmax])
    t2 = maximum([tmin,tmax])
    err = 100 * (sqrt(1-X^2)/(2*X)*sqrt((6*sqrt(π/2)*T)/(wc^2*(t2^3-t1^3))))

    return dvv,cc,cdp,Array(ϵ),err,allC
end

"""

    stretching(C,t,window,fmin,fmax;dvmin,dvmax,ntrials)

This function compares the Reference waveform to stretched/compressed current
waveforms to get the relative seismic velocity variation (and associated error)
for all correlations in CorrData `C`. It also computes the correlation
coefficient between the Reference waveform and the current waveform.

# Arguments
- `C::CorrData`: Correlation for dv/v.
- `cur::AbstractArray`: Current correlation.
- `t::AbstractArray`: time vector, common to both `ref` and `cur`.
- `window::AbstractArray`: vector of the indices of the `cur` and `ref` windows
                          on which you want to do the measurements
- `fmin::Float64`: minimum frequency in the correlation [Hz]
- `fmax::Float64`: maximum frequency in the correlation [Hz]
- `dvmin::Float64`: minimum bound for the velocity variation; e.g. dvmin=-0.03
                   for -3% of relative velocity change
- `dvmax::Float64`: maximum bound for the velocity variation; e.g. dvmin=0.03
                  for 3% of relative velocity change
- `ntrial::Int`:  number of stretching coefficient between dvmin and dvmax, no need to be higher than 100

# Returns
- `dvv::Array{Float64,1}`: Relative Velocity Change dv/v (in %)
- `cc::Array{Float64,1}`: Correlation coefficient between the reference waveform and the
                      best stretched/compressed current waveform
- `cdp::Array{Float64,1}`: Correlation coefficient between the reference waveform and the
                 initial current waveform
- `ϵ::Array{Float64,1}`: Vector of Epsilon values (ϵ =-dt/t = dv/v)
- `err::Array{Float64,1}`: Errors in the dv/v measurements based on [Weaver et al., 2011](https://onlinelibrary.wiley.com/doi/full/10.1111/j.1365-246X.2011.05015.x)
- `allC::Array{Float64,2}`: Matrix of the correlation coefficient between the
                        reference waveform and every stretched/compressed
                        current waveforms
"""
function stretching(C::CorrData, t::AbstractArray, window::AbstractArray,
                    fmin::Float64, fmax::Float64; dvmin::Float64=-0.1,
                    dvmax::Float64=0.1, ntrial::Int=100)

    N = length(C.t)
    ref = SeisNoise.stack(C,allstack=true)
    dvv = zeros(length(C.t))
    cc = similar(dv)
    cdp = similar(dv)
    err = similar(dv)
    allC = zeros(ntrial,N)
    lags = -ref.maxlag:1/ref.fs:ref.maxlag

    for ii = 1:N
        dvv[ii], cc[ii], cdp[ii], ϵ, err[ii], allC[:,ii] = stretching(ref.corr[:],
                                                            C.corr[:,ii],
                                                            lags,
                                                            window,
                                                            fmin,
                                                            fmax,
                                                            dvmin=dvmin,
                                                            dvmax=dvmax,
                                                            ntrial=ntrial)
    end
    ϵ = collect(range(dvmin, stop=dvmax, length=ntrial))

    return dvv, cc, cdp, ϵ, err, allC
end

"""

stretching(ref,cur,t,window,freqbands;dvmin,dvmax,ntrials,norm)

This function compares the Reference waveform to stretched/compressed current
waveforms at a range of frequencies to get the frequency-dependent relative seismic
velocity variation (and associated error). It also computes the correlation
coefficient between the Reference waveform and the current waveform.

# Arguments
- `ref::AbstractArray`: Reference correlation.
- `cur::AbstractArray`: Current correlation.
- `t::AbstractArray`: time vector, common to both `ref` and `cur`.
- `window::AbstractArray`: vector of the indices of the `cur` and `ref` windows
                          on which you want to do the measurements
- `freqbands::AbstractArray`: Frequency bands over which to compute dv/v
- `dvmin::Float64`: minimum bound for the velocity variation; e.g. dvmin=-0.03
                   for -3% of relative velocity change
- `dvmax::Float64`: maximum bound for the velocity variation; e.g. dvmin=0.03
                  for 3% of relative velocity change
- `ntrial::Int`:  number of stretching coefficient between dvmin and dvmax, no need to be higher than 100
- `norm::Bool`: Whether or not to normalize signals before dv/v

# Returns
- `freqbands::AbstractArray`: Array of frequencies where dv/v was measured
- `dvv::Array{Float64,1}`: Relative Velocity Change dv/v (in %)
- `cc::Array{Float64,1}`: Correlation coefficient between the reference waveform and the
                      best stretched/compressed current waveform
- `cdp::Array{Float64,1}`: Correlation coefficient between the reference waveform and the
                 initial current waveform
- `ϵ::Array{Float64,1}`: Vector of Epsilon values (ϵ =-dt/t = dv/v)
- `err::Array{Float64,1}`: Errors in the dv/v measurements based on [Weaver et al., 2011](https://onlinelibrary.wiley.com/doi/full/10.1111/j.1365-246X.2011.05015.x)
"""
function stretching(ref::AbstractArray, cur::AbstractArray, t::AbstractArray,
                    window::AbstractArray, freqbands::AbstractArray;
                    dvmin::Float64=-0.1, dvmax::Float64=0.1, ntrial::Int=100, norm::Bool=true)
     # define sampling interval
     dt = mean(diff(t))
     fs = 1/dt

     # calculate the CWT of the time series, using identical parameters for both calculations
     cwt1, sj, freqs, coi = cwt(ref, dt, minimum(freqbands), maximum(freqbands))
     cwt2, sj, freqs, coi = cwt(cur, dt, minimum(freqbands), maximum(freqbands))

     # if a frequency window is given (instead of a set of frequency bands), we assume
     # dv/v should be calculated for each frequency. We construct a 2D array of the
     # form [f1 f1; f2 f2; ...], which can be treated the same as a 2D array of frequency bands
     if ndims(freqbands)==1
         freqbands = hcat(freqs, freqs)
     end
     # number of frequency bands
     (nbands,_) = size(freqbands)

     # initialize dvv and err arrays
     dvv = zeros(nbands)
     cc = similar(dvv)
     cdp = similar(dvv)
     err = similar(dvv)
     allC = zeros(ntrial, nbands)

     # loop over frequency bands
     for iband=1:nbands
         (fmin, fmax) = freqbands[iband, :]

         # get current frequencies over which we apply icwt
         # frequency checks
         if fmax < fmin
             println("Error: please ensure columns 1 and 2 are right frequency limits in freqbands!")
         else
             freq_ind = findall(f->(f>=fmin && f<=fmax), freqs)
         end

         # perform icwt
         icwt1 = icwt(cwt1[:,freq_ind], sj[freq_ind], dt)
         icwt2 = icwt(cwt2[:,freq_ind], sj[freq_ind], dt)
         wcwt1 = real.(icwt1)
         wcwt2 = real.(icwt2)

         # normalize both signals, if appropriate
         if norm
             ncwt1 = ((wcwt1 .- mean(wcwt1)) ./ std(wcwt1))[:]
             ncwt2 = ((wcwt2 .- mean(wcwt2)) ./ std(wcwt2))[:]
         else
             ncwt1 = wcwt1[:]
             ncwt2 = wcwt2[:]
         end

         dvv[iband], cc[iband], cdp[iband], ϵ, err[iband], allC[:,iband] = stretching(ncwt1, ncwt2, t, window, fmin, fmax, dvmin=dvmin, dvmax=dvmax, ntrial=ntrial)
     end

     return freqbands, dvv, cc, cdp, ϵ, err, allC
end
