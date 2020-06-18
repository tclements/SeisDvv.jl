export WCC, WCC_dt

"""

    WCC(ref, cur, t, window, fs, window_length, window_step)

Measure dv/v using the windowed cross-correlation following Snieder et al., 2012

# Arguments
- `ref::AbstractArray`: Reference time series
- `cur::AbstractArray`: Current time series
- `t::AbstractArray`: Time axis common to ref and cur
- `window::AbstractArray`: Indices for a subset of t over which to measure phase lags
- `fs::Float64`: Sampling frequency
- `window_length::Float64`: Length of time window within which to find time shifts
- `window_step::Float64`: Time step to advance window

# Returns
- `dvv::Float64`: dv/v for current correlation for a range of frequencies
- `dvv_err::Float64`: Error for calculation of dv/v for a range of frequencies
- `int::Float64`: Intercept for regression calculation
- `int_err::Float64`: Error for intercept
- `dvv0::Float64`: dv/v for current correlation forced through the origin
- `dvv0_err::Float64`: Error for calculation of dvv0 for a range of frequencies
"""
function WCC(ref::AbstractArray, cur::AbstractArray, t::AbstractArray, window::AbstractArray, fs::Float64,
             window_length::Float64, window_step::Float64)
    # measure phase shifts
    time_axis, delta_t, cc_max = WCC_dt(ref, cur, t, window, fs, window_length, window_step)
    # perform linear regression of dt/t=-dv/v
    dvv, dvv_err, int, int_err, dvv0, dvv0_err = dvv_lstsq(time_axis, delta_t)

    return dvv, dvv_err, int, int_err, dvv0, dvv0_err
end

"""

    WCC(ref, cur, t, window, fs, window_length, window_step, freqbands; norm=true)

Measure dv/v over a range of frequencies using the windowed cross-correlation following Snieder et al., 2012

# Arguments
- `ref::AbstractArray`: Reference time series
- `cur::AbstractArray`: Current time series
- `t::AbstractArray`: Time axis common to ref and cur
- `window::AbstractArray`: Indices for a subset of t over which to measure phase lags
- `fs::Float64`: Sampling frequency
- `window_length::Float64`: Length of time window within which to find time shifts
- `window_step::Float64`: Time step to advance window
- `freqbands::AbstractArray`: Frequency bands over which to compute dv/v
- `norm::Bool`: Whether or not to normalize signals before dv/v

# Returns
- `freqbands::AbstractArray`: Array of frequencies where dv/v was measured
- `dvv::Float64`: dv/v for current correlation for a range of frequencies
- `dvv_err::Float64`: Error for calculation of dv/v for a range of frequencies
- `int::Float64`: Intercept for regression calculation
- `int_err::Float64`: Error for intercept
- `dvv0::Float64`: dv/v for current correlation forced through the origin
- `dvv0_err::Float64`: Error for calculation of dvv0 for a range of frequencies
"""
function WCC(ref::AbstractArray, cur::AbstractArray, t::AbstractArray, window::AbstractArray, fs::Float64,
             window_length::Float64, window_step::Float64, freqbands::AbstractArray; norm::Bool=true)
     # define sampling interval
     dt = 1/fs

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
     dvv_err = similar(dvv)
     int = similar(dvv)
     int_err = similar(dvv)
     dvv0 = similar(dvv)
     dvv0_err = similar(dvv)

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
             ncwt1 = ((wcwt1 .- mean(wcwt1)) ./ std(wcwt1))
             ncwt2 = ((wcwt2 .- mean(wcwt2)) ./ std(wcwt2))
         else
             ncwt1 = wcwt1
             ncwt2 = wcwt2
         end

         time_axis, delta_t, cc_max = WCC_dt(ncwt1, ncwt2, t, window, fs, window_length, window_step)
         dvv[iband], dvv_err[iband], int[iband], int_err[iband], dvv0[iband], dvv0_err[iband] = dvv_lstsq(time_axis, delta_t)
     end

     return freqbands, dvv, dvv_err, int, int_err, dvv0, dvv0_err
end

"""

    WCC_dt(ref, cur, fs, tmin, window_length, window_step)

Measure phase shifts using the windowed cross-correlation following Snieder et al., 2012

# Arguments
- `ref::AbstractArray`: Reference time series
- `cur::AbstractArray`: Current time series
- `t::AbstractArray`: Time axis common to ref and cur
- `window::AbstractArray`: Indices for a subset of t over which to measure phase lags
- `fs::Float64`: Sampling frequency
- `window_length::Float64`: Length of time window within which to find time shifts
- `window_step::Float64`: Time step to advance window

# Returns
- `time_axis::AbstractArray`: Center times of windows
- `dt::AbstractArray`: Time shifts
- `cc_max::AbstractArray`: Maximum correlation coefficient for each window
"""
function WCC_dt(ref::AbstractArray, cur::AbstractArray, t::AbstractArray, window::AbstractArray, fs::Float64,
             window_length::Float64, window_step::Float64)
     # create time axis for mwcs
     time_axis = Array(t[window[1]] + window_length / 2. : window_step :
                       t[window[end]] - window_length / 2.)

     window_length_samples = convert(Int,window_length * fs)
     window_step_samples = convert(Int,window_step * fs)
     minind = window[1]:window_step_samples:window[end] - window_length_samples
     padd = convert(Int,2 ^ (ceil(log2(abs(window_length_samples))) + 2))

     N = length(minind)
     dt = zeros(N)
     err = zeros(N)

     cci = zeros(window_length_samples,N)
     cri = zeros(window_length_samples,N)

     # fill matrices
     for ii = 1:N
         cci[:,ii] = cur[minind[ii]:minind[ii]+window_length_samples-1]
         cri[:,ii] = ref[minind[ii]:minind[ii]+window_length_samples-1]
     end

     # preprocess
     SeisNoise.demean!(cci)
     SeisNoise.detrend!(cci)
     SeisNoise.taper!(cci,fs,max_percentage=0.85)
     SeisNoise.demean!(cri)
     SeisNoise.detrend!(cri)
     SeisNoise.taper!(cri,fs,max_percentage=0.85)

     xcorr = zeros(2*window_length_samples-1, N)
     for ii=1:N
         xcorr[:,ii] = DSP.xcorr(cci[:,ii], cri[:,ii])
     end

     # get maximum correlation coefficient and its index
     cc_max, cc_max_ind = findmax(xcorr, dims=1)
     cc_max_ind = [cc_max_ind[i][1] for i=1:length(cc_max_ind)]
     # get time shift
     dt = (cc_max_ind .- window_length_samples)/fs

     return time_axis, dt, cc_max
end
