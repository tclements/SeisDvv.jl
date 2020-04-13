export WCC, WCC_dvv

"""
    WCC(ref, cur, fs, tmin, window_length, window_step, maxlag)

Windowed cross-correlation following Snieder et al., 2012

# Arguments
`ref::AbstractArray`: Input signal
`cur::AbstractArray`: Reference signal
`fs::Float64`: Sampling frequency
`tmin::Float64`: Minimum time
`window_length::Float64`: Length of time window within which to find time shifts
`window_step::Float64`: Time step to advance window
`maxlag::Float64`: Maximum time to consider

# Returns
`time_axis::AbstractArray`: Center times of windows
`dt::AbstractArray`: Time shifts
`err::AbstractArray`: Errors in time shift measurements
"""
function WCC(ref::AbstractArray, cur::AbstractArray, fs::Float64, tmin::Float64,
             window_length::Float64, window_step::Float64, maxlag::Float64)

     # create time axis for mwcs
     time_axis = Array(tmin + window_length / 2. : window_step : tmin +
                       length(ref) / fs - window_length / 2.)

     window_length_samples = convert(Int,window_length * fs)
     window_step_samples = convert(Int,window_step * fs)
     minind = 1:window_step_samples:length(ref) - window_length_samples
     padd = convert(Int,2 ^ (ceil(log2(abs(window_length_samples))) + 2))

     N = length(minind)
     dt = zeros(N)
     err = zeros(N)
     time_axis = time_axis[1:N]

     # Find values in frequency range of interest
     freq_vec = FFTW.rfftfreq(padd,fs)
     index_range = findall(x -> x >= fmin && x <= fmax,freq_vec)
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

     # take fourier transform
     fcur = rfft(cci,1)
     fref = rfft(cri,1)

     # calculate cross-correlation
     xcorr = irfft(conj.(fref) .* fcur, window_length_samples, 1)
     xcorr ./= sqrt.(sum(cci.^2, dims=1) .* sum(cri.^2, dims=1))[1]

     # return corr[-maxlag:maxlag]
     t = vcat(0:Int(window_length_samples/2)-1, -Int(window_length_samples/2):-1)
     ind = findall(abs.(t) .<= maxlag*fs)
     newind = FFTW.fftshift(ind,1)
     xcorr = xcorr[newind,:]

     # get maximum correlation coefficient and its index
     cc_max, cc_max_ind = findmax(xcorr, dims=1)
     cc_max_ind = [cc_max_ind[i][1] for i=1:length(cc_max_ind)]

     # get time shift
     dt = (cc_max_ind .- fld(window_length_samples, 2))/fs

     return time_axis, dt, cc_max
end

"""
    WCC_dvv(time_axis, dt)

Regreses dv/v from dt/t measurements.

# Arguments
`time_axis::AbstractArray`: Center times of windows.
`dt::AbstractArray`: Time shifts

# Returns
`dvv::AbstractArray`: Velocity chnages corresponding to change in time lags
`err::AbstractArray`: Errors in velocity change measurements
"""
function WCC_dvv(time_axis::AbstractArray, dt::AbstractArray)
    model = glm(@formula(Y ~0 + X),DataFrame(X=time_axis,Y=dt),Normal(),
                    IdentityLink(),wts=ones(length(dt)))
    dvv = -coef(model)[1]*100
    err = stderror(model)[1]*100

    return dvv, err
end
