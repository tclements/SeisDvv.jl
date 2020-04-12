using GLM
export applyOverFrequencies, cwt, icwt, wts_dvv

"""
    waveletMethodDvv(cur, ref, t, twindow, freqbands, dj, s0, J)

Perform time-domain dv/v algorithms after cwt, frequency selection, and icwt.

# Arguments
`cur::AbstractArray`: Input signal
`ref::AbstractArray`: Reference signal
`t::AbstractArray`: Time vector
`twindow::AbstractArray`: Times over which to compute dv/v
`freqbands::AbstractArray`: Frequency bands over which to compute dv/v
`dj::AbstractFloat`: Spacing between discrete scales. Default value is 1/12
`method::String`: 'stretching' or 'dtw'
`normalize::Bool`: Whether or not to normalize signals before dv/v
``

# Returns
`freqbands::AbstractArray`: fmin and fmax for each iteration of the dv/v algorithm
`dvv::AbstractArray`: dv/v values for each frequency band
`err::AbstractArray`: errors in dv/v measurements
"""
function wavelet_dvv(cur::AbstractArray, ref::AbstractArray, t::AbstractArray, twindow::AbstractArray, freqbands::AbstractArray, dj::AbstractFloat; method::String="stretching", normalize::Bool=true)
    # define sample frequency
    dt = t[2] - t[1]
    fs = 1/dt

    cwt1, sj, freqs, coi = cwt(cur, dt, minimum(freqbands), maximum(freqbands))
    cwt2, sj, freqs, coi = cwt(ref, dt, minimum(freqbands), maximum(freqbands))

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
    err = zeros(nbands)

    # loop over frequency bands
    for iband=1:nbands
        println(iband)
        (fmin, fmax) = freqbands[iband, :]

        # get current frequencies over which we apply icwt
        # frequency checks
        if (fmax > maximum(freqs) || fmax < fmin)
            println("Error: please ensure columns 1 and 2 are right frequency limits in freqbands!")
        else
            freq_ind = findall(f->(f>=fmin && f<=fmax), freqs)
        end

        # perform icwt
        icwt1 = icwt(cwt1[:,freq_ind], sj, dt)
        icwt2 = icwt(cwt2[:,freq_ind], sj, dt)

        # get times over which we apply dv/v algorithm
        tmin = twindow[1]
        tmax = twindow[2]
        # time checks
        if tmin < minimum(t) || tmax > maximum(t) || tmax <= tmin
            println("Error: please input right time limits in the time window!")
        else
            # trim time vector
            t_ind = findall(x->(x≤tmax && x≥tmin), t)
        end

        # trim vectors to correspond to time
        wt = t[t_ind]
        window = collect(1:length(wt))
        rcwt1 = real.(icwt1)
        rcwt2 = real.(icwt2)
        wcwt1 = rcwt1[t_ind]
        wcwt2 = rcwt2[t_ind]

        # normalize both signals, if appropriate
        if normalize
            ncwt1 = ((wcwt1 .- mean(wcwt1)) ./ std(wcwt1))[:]
            ncwt2 = ((wcwt2 .- mean(wcwt2)) ./ std(wcwt2))[:]
        else
            ncwt1 = wcwt1[:]
            ncwt2 = wcwt2[:]
        end

        # perform dv/v
        if method=="stretching"
            (dvv[iband], cc, cdp, eps, err[iband], allC) = stretching(ncwt1, ncwt2, wt, window, fmin, fmax, dvmin=-0.03, dvmax=0.03, ntrial=10000)
        elseif method=="dtw"
            (stbarTime, stbar, dist, error) = dtwdt(ncwt1, ncwt2, dt, maxLag=maxLag, b=b, direction=direction)
            # perform linear regression
            model = glm(@formula(Y ~0 + X),DataFrame(X=wt,Y=stbarTime),Normal(),
                        IdentityLink(),wts=ones(length(wt)))

            dvv[iband] = coef(model)[1]*100
            err[iband] = stderror(model)[1]*100
        else
            println("Please choose a valid method")
        end
    end
    return freqbands, -dvv, err
end


"""
  cwt(signal,dt,freqmin,freqmax)

Continuous wavelet transform of the signal at specified scales.
Note: uses the Morlet wavelet ONLY.

# Arguments
- `signal::AbstractArray`: Input signal array.
- `dt::AbstractFloat`: Sampling interval [s].
- `freqmin::AbstractFloat`: Minimum frequency for cwt [Hz].
- `freqmax::AbstractFloat`: Maximum frequency for cwt [Hz].
- `f0::Real`: Nondimensional frequency from Torrence & Campo, 1998 eq. 1 [Hz].
- `dj::AbstractFloat`: Spacing between discrete scales. Default value is 1/12.

# Returns
- `W::AbstractArray`: Wavelet transform from Morlet wavelet.
- `freqs::AbstractArray`: Fourier frequencies for wavelet scales [Hz].
- `coi::AbstractArray`: Cone of influence - maximum period (in s) of useful
    information at that particular time. Periods greater than
    those are subject to edge effects.

This is a Julia translation of the cwt in pycwt https://github.com/regeirk/pycwt
"""
function cwt(signal::AbstractArray{T,1},dt::AbstractFloat,freqmin::AbstractFloat,
             freqmax::AbstractFloat;f0=6.,dj=1/12) where T <: AbstractFloat
    n0 = length(signal)
    flambda = T.((4 .* π) ./ (f0 .+ sqrt(2 .+ f0^2)))
    s0 = 2 * dt / flambda
    J = convert(Int,round(log2(n0 .* dt ./ s0) ./ dj))

    # The scales as of Mallat 1999
    sj = T.(s0 .* 2 .^ ((0:J) .* dj))
    freqs = 1 ./ (flambda .* sj)

    # subset by freqmin & freqmax
    ind = findall((freqs .> freqmin) .& (freqs .< freqmax))
    sj = sj[ind]
    freqs = freqs[ind]

    # signal fft
    signal_ft = fft(signal,1)
    N = length(signal_ft)
    # Fourier angular frequencies
    ftfreqs = T.(2 .* π * FFTW.fftfreq(n0,1/dt))

    # Creates wavelet transform matrix as outer product of scaled transformed
    # wavelets and transformed signal according to the convolution theorem.
    # (i)   Transform scales to column vector for outer product;
    # (ii)  Calculate 2D matrix [s, f] for each scale s and Fourier angular
    #       frequency f;
    # (iii) Calculate wavelet transform;
    psi_ft_bar = ((sj' .* ftfreqs[2] .* T(N)) .^ T(0.5)) .* conj.(psi_ft(ftfreqs * sj',f0))
    W = ifft(signal_ft .* psi_ft_bar,1)

    # Checks for NaN in transform results and removes them from the scales if
    # needed, frequencies and wavelet transform. Trims wavelet transform at
    # length `n0`.
    sel = findall(.!all(isnan.(W),dims=1)[:])
    if length(sel) > 0
        sj = sj[sel]
        freqs = freqs[sel]
        W = W[:,sel]
    end

    # Determines the cone-of-influence. Note that it is returned as a function
    # of time in Fourier periods. Uses triangualr Bartlett window with
    # non-zero end-points.
    coi = n0 ./ 2 .- abs.((0:n0-1) .- (n0 - 1.) ./2.)
    coi = T.(flambda ./ sqrt(2.) .* dt .* coi)
    return W, sj, freqs, coi
end

"""

  icwt(W,sj,dt)

Inverse continuous wavelet transform at specified scales.
Note: uses the Morlet wavelet ONLY.

# Arguments
- `W::AbstractArray`: Wavelet transform, the result of the `cwt` function.
- `sj::AbstractArray`: Scale indices as returned by the `cwt` function.
- `dt::AbstractFloat`: Sampling interval [s].
- `dj::AbstractFloat`: Spacing between discrete scales. Default value is 1/12.

# Returns
- `iW::AbstractArray`: Inverse wavelet transform from Morlet wavelet.

This is a Julia translation of the icwt in pycwt https://github.com/regeirk/pycwt
Note that the pycwt version has incorrect scaling.
"""
function icwt(W::AbstractArray, sj::AbstractArray, dt::AbstractFloat;dj=1/12)
    T = real(eltype(W))
    # As of Torrence and Compo (1998), eq. (11)
    iW = T(dj .* sqrt(dt) ./ 0.776 .* (pi ^ 0.25)) .* sum(real.(W) ./ (sj .^ T(0.5))',dims=2)
    return iW
end

"""

  icwt(W,sj,dt)

Inverse continuous wavelet transform at specified scale.
Note: uses the Morlet wavelet ONLY.

# Arguments
- `W::AbstractArray`: Wavelet transform, the result of the `cwt` function.
- `sj::AbstractFloat`: Scale index as returned by the `cwt` function.
- `dt::AbstractFloat`: Sampling interval [s].
- `dj::AbstractFloat`: Spacing between discrete scales. Default value is 1/12.

# Returns
- `iW::AbstractArray`: Inverse wavelet transform from Morlet wavelet.

This is a Julia translation of the icwt in pycwt https://github.com/regeirk/pycwt
Note that the pycwt version has incorrect scaling.
"""
function icwt(W::AbstractArray, sj::AbstractFloat, dt::AbstractFloat;dj=1/12)
    T = real(eltype(W))
    # As of Torrence and Compo (1998), eq. (11)
    iW = T(dj .* sqrt(dt) ./ 0.776 .* (pi ^ 0.25)) .* real.(W) ./ sj ^ T(0.5)
    return iW
end

function psi_ft(A::AbstractArray{T},f0::Real) where T <: AbstractFloat
    return exp.(T(-0.5) .* (A .- T(f0)) .^2) .* T(π ^ -0.25)
end
