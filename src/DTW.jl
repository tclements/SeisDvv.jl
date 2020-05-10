export dtwdt, dtw_dvv, computeErrorFunction, accumulateErrorFunction, backtrackDistanceFunction, computeDTWerror

"""

    dtwdt(u0, u1, dt; dtwnorm='L2', maxlag=80, b=1, direction=1)

returns minimum distance time lag and index in dist array, and dtw error between traces.

# Arguments
- `u0::AbstractArray`: time series #1
- `u1::AbstractArray`: time series #2
- `dt::Float64`: time step
- `dtwnorm::String`: norm used to calculate distance; effect on the unit of dtw error. (L2 or L1)
-  Note: L2 is not squard, thus distance is calculated as (u1[i]-u0[j])^2
- `maxlag::Int64`: number of maxLag id to search the distance.
- `b::Int64t`: b value to controll in distance calculation algorithm (see Mikesell et al. 2015).
- `direction::Int64`: length of noise data window, in seconds, to cross-correlate.

# Returns
- `stbarTime::Array{Float64,1}`: series of time shift at t.
- `stbar::Array{Int64,1}`: series of minimum distance index in distance array.
- `dist::Array{Int64,2}`: distance array.
- `dtwerror::Float64`: dtw error (distance) between two time series.
"""
function dtwdt(u0::AbstractArray, u1::AbstractArray, dt::Float64;
    dtwnorm::String="L2",     #norm to calculate distance; effect on the unit of dtw error
    maxLag::Int64=80,         #number of maxLag id to search the distance
    b::Int64=1,               #b value to controll in distance calculation algorithm (see Mikesell et al. 2015)
    direction::Int64=1,       #direction to accumulate errors (1=forward, -1=backward, 0=double to smooth)
    )

    #prepare datasize and time vector
    lvec   = (-maxLag:maxLag).*dt; # lag array for plotting below
    npts   = length(u0);            # number of samples
    if length(u0) != length(u1) error("u0 and u1 must be same length.") end
    tvec   = ( 0 : npts-1 ) .* dt; # make the time axis

    #compute distance between traces
    err = computeErrorFunction(u1, u0, npts, maxLag, norm=dtwnorm);

    #compute distance array and backtrack index
    if direction == 1 || direction == -1
        dist  = accumulateErrorFunction(direction, err, npts, maxLag, b); # forward accumulation to make distance function
        stbar = backtrackDistanceFunction( -1*direction, dist, err, -maxLag, b ); # find shifts
    elseif direction == 0
        #calculate double time to smooth distance array
        dist1 = accumulateErrorFunction( -1, err, npts, maxLag, b ); # forward accumulation to make distance function
        dist2 = accumulateErrorFunction( 1, err, npts, maxLag, b ); # backwward accumulation to make distance function
        dist  = dist1 .+ dist2 .- err; # add them and remove 'err' to not count twice (see Hale's paper)
        stbar = backtrackDistanceFunction( -1, dist, err, -maxLag, b );
    else
        error("direction must be +1, -1 or 0(smoothing).")
    end

    stbarTime = stbar .* dt;      # convert from samples to time
    tvec2     = tvec + stbarTime; # make the warped time axis

    #accumulate distance in distance array to calculate dtw error
    error = computeDTWerror( err, stbar, maxLag );

    return stbarTime, stbar, dist, error
end

"""

    dtwdt(u0, u1, dt, freqbands; dtwnorm='L2', maxlag=80, b=1, direction=1, norm=true)

returns minimum distance time lag and index in dist array, and dtw error between traces.

# Arguments
- `u0::AbstractArray`: time series #1
- `u1::AbstractArray`: time series #2
- `dt::Float64`: time step
- `freqbands::AbstractArray`: Frequency bands over which to compute dv/v
- `dtwnorm::String`: norm used to calculate distance; effect on the unit of dtw error. (L2 or L1)
-  Note: L2 is not squard, thus distance is calculated as (u1[i]-u0[j])^2
- `maxlag::Int64`: number of maxLag id to search the distance.
- `b::Int64t`: b value to controll in distance calculation algorithm (see Mikesell et al. 2015).
- `direction::Int64`: length of noise data window, in seconds, to cross-correlate.
- `norm::Bool`: Whether or not to normalize signals before dv/v

# Returns
- `-dtt::AbstractArray`: dv/v for current correlation for a range of frequencies
- `err::Float64`: Error for calculation of dv/v for a range of frequencies
- `a::Float64`: Intercept for regression calculation
- `ea::Float64`: Error on intercept
- `m0::Float64`: dt/t for current correlation with no intercept
- `em0::Float64`: Error for calculation of `m0`
"""
function dtwdt(u0::AbstractArray, u1::AbstractArray, dt::Float64, freqbands::AbstractArray;
    dtwnorm::String="L2",     #norm to calculate distance; effect on the unit of dtw error
    maxLag::Int64=80,         #number of maxLag id to search the distance
    b::Int64=1,               #b value to controll in distance calculation algorithm (see Mikesell et al. 2015)
    direction::Int64=1,       #direction to accumulate errors (1=forward, -1=backward, 0=double to smooth)
    norm::Bool=true)
    # define sample frequency
    fs = 1/dt

    # calculate the CWT of the time series, using identical parameters for both calculations
    cwt1, sj, freqs, coi = cwt(u0, dt, minimum(freqbands), maximum(freqbands))
    cwt2, sj, freqs, coi = cwt(u1, dt, minimum(freqbands), maximum(freqbands))

    # if a frequency window is given (instead of a set of frequency bands), we assume
    # dv/v should be calculated for each frequency. We construct a 2D array of the
    # form [f1 f1; f2 f2; ...], which can be treated the same as a 2D array of frequency bands
    if ndims(freqbands)==1
        freqbands = hcat(freqs, freqs)
    end
    # number of frequency bands
    (nbands,_) = size(freqbands)

    # initialize arrays
    dtt = zeros(nbands)
    err = similar(dtt)
    a = similar(dtt)
    ea = similar(dtt)
    m0 = similar(dtt)
    em0 = similar(dtt)

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

        # perform dv/v
        stbarTime, stbar, dist, error = dtwdt(ncwt1, ncwt2, dt, maxLag=maxLag, b=b, direction=direction)
        # perform linear regression
        dtt[iband], err[iband], a[iband], ea[iband], m0[iband], em0[iband] = dtw_dvv(collect(1:length(u0))*dt, stbarTime)
    end

    return freqbands, -dtt, err, a, ea, m0, em0
end


"""
    dtw_dvv(X, Y)

Linear regression of phase shifts to time to give dt/t=-dv/v

# Arguments
- `time_axis::AbstractArray`: Array of lag times
- `dt::AbstractArray`: Time shifts for each lag time

# Returns
- `m::Float64`: dt/t for current correlation
- `em::Float64`: Error for calculation of `m`
- `a::Float64`: Intercept for regression calculation
- `ea::Float64`: Error on intercept
- `m0::Float64`: dt/t for current correlation with no intercept
- `em0::Float64`: Error for calculation of `m0`
"""
function dtw_dvv(time_axis::AbstractArray, dt::AbstractArray)
    # regress data using least squares
    model0 = glm(@formula(Y ~0 + X),DataFrame(X=time_axis,Y=dt),Normal(),
                IdentityLink(),wts=ones(length(time_axis)))
    model = glm(@formula(Y ~ X),DataFrame(X=time_axis,Y=dt),Normal(),
                IdentityLink(),wts=ones(length(time_axis)))

    a,m = coef(model).*100
    ea, em = stderror(model).*100

    m0 = coef(model0)[1]*100
    em0 = stderror(model0)[1]*100

    return m, em, a, ea, m0, em0
end

"""

    computerErrorFunction(u1, u0, nSample, lag, norm='L2')

Compute error function for each sample and lag, see Hale, 2013.
Dave Hale, (2013), "Dynamic warping of seismic images," GEOPHYSICS 78: S105-S115.
https://doi.org/10.1190/geo2012-0327.1

# Arguments
- `u1::AbstractArray`: Trace we intend to warp
- `u0::AbstractArray`: Reference trace to compare with
- `nSample::Int64`: Number of points to compare in the traces
- `lag::Int64`: Maximum lag in sample number to search
- `norm::String`: 'L2' or 'L1'

# Returns
- `err::AbstractArray`: 2D error function in (samples, lags)
"""
function computeErrorFunction(u1::AbstractArray, u0::AbstractArray, nSample::Int64, lag::Int64; norm::String="L2")
    if lag >= nSample
        error("computeErrorFunction:lagProblem ","lag must be smaller than nSample");
    end

    # Allocate error function variable
    err = zeros(Float64, nSample, 2 * lag + 1 );

    # initial error calculation
    for ll = -lag:lag # loop over lags
        thisLag = ll + lag + 1;
        for ii = 1:nSample # loop over samples
            if ( ii + ll >= 1 && ii + ll <= nSample ) # skip corners for now, we will come back to these
                diff = u1[ii] - u0[ii + ll]; # sample difference
                if norm == "L2"
                        err[ii, thisLag] = diff^2; # difference squared error
                elseif norm == "L1"
                        err[ii, thisLag] = abs(diff); # absolute value errors
                else
                    error("norm type is not defined.")
                end
            end
        end
    end

    # Now fix corners with constant extrapolation
    for ll = -lag:lag # loop over lags
        thisLag = ll + lag + 1;
        for ii = 1:nSample # loop over samples
            if ( ii + ll < 1 ) # lower left corner (negative lag, early time)
                err[ii, thisLag] = err[-ll + 1, thisLag];
            elseif ( ii + ll > nSample ) # upper right corner (positive lag, late time)
                err[ii, thisLag] = err[nSample - ll, thisLag];
            end
        end
    end

    return err
end


"""

    accumulateErrorFunction(dir, err, nSample, lag, b)

# Arguments
- `dir::Int64`: accumulation direction (dir > 0 ⟹ forward in time, dir ≦ 0 ⟹ backward in time)
- `err::Array{Float64,2}`: 2D error function
- `nSample::Int64`: number of points to compare in the traces
- `lag::Int64`: maximum lag in sample number to search
- `b::Int64`: strain limit (b∈Ints, b≥1)

# Returns
- `d::AbstractArray`: 2D distance function

The function is equation 6 in Hale, 2013.
Original by Di Yang Last modified by Dylan Mikesell (25 Feb. 2015)
"""
function accumulateErrorFunction(dir::Int64, err::Array{Float64,2}, nSample::Int64, lag::Int64, b::Int64)
    nLag = (2 * lag ) + 1; # number of lags from [ -lag : +lag ]

    # allocate distance matrix
    d = zeros(Float64, nSample, nLag);

    #--------------------------------------------------------------------------
    # Setup indices based on forward or backward accumulation direction
    #--------------------------------------------------------------------------
    if dir > 0            # FORWARD
        iBegin = 1;       # start index
        iEnd   = nSample; # end index
        iInc   = 1;       # increment
    else                  # BACKWARD
        iBegin = nSample; # start index
        iEnd   = 1;       # stop index
        iInc   = -1;      # increment
    end

    #--------------------------------------------------------------------------
    # Loop through all times ii in forward or backward direction
    for ii = iBegin:iInc:iEnd
        # min/max to account for the edges/boundaries
        ji = max(1, min(nSample, ii - iInc ));     # i-1 index
        jb = max(1, min(nSample, ii - iInc * b )); # i-b index

        # loop through all lags l
        for ll = 1:nLag
            # -----------------------------------------------------------------
            # check limits on lag indices
            lMinus1 = ll - 1; # lag at l-1

            if lMinus1 < 1  # check lag index is greater than 1
                lMinus1 = 1; # make lag = first lag
            end

            lPlus1 = ll + 1; # lag at l+1

            if lPlus1 > nLag # check lag index less than max lag
                lPlus1 = nLag; # D.M. version
            end
            # -----------------------------------------------------------------

            # get distance at lags (ll-1, ll, ll+1)
            distLminus1 = d[jb, lMinus1]; # minus:  d( i-b, j-1 )
            distL       = d[ji, ll];      # actual: d( i-1, j   )
            distLplus1  = d[jb, lPlus1];  # plus:   d( i-b, j+1 )

            if ji != jb # equation 10 in Hale (2013)
                for kb = ji:-iInc:jb+iInc # sum errors over i-1:i-b+1
                    distLminus1 = distLminus1 + err[kb, lMinus1];
                    distLplus1  = distLplus1  + err[kb, lPlus1];
                end
            end

            # equation 6 (if b=1) or 10 (if b>1) in Hale (2013) after treating boundaries
            d[ii, ll] = err[ii, ll] + min(distLminus1, distL, distLplus1);
        end
    end
    return d
end


"""

    backtrackDistanceFunction(dir, d, err, lmin, b)

# Arguments
- `dir::Int64`: side to start minimization (dir > 0 ⟹ forward in time, dir ≦ 0 ⟹ backward in time)
- `d::Array{Float64,2}`: 2D distance function
- `err:Array{Float64,2}`: 2D error function
- `lmin::Int64`: minimum lag to search over
- `b::Int64`: strain limit (b∈Ints, b≥1)

# Returns
- `stbar::AbstractArray`: vector of integer shifts subject to |u(i)-u(i-1)| <= 1/b

See equation 2 in Hale, 2013.
Original by Di Yang and modified from DTWDT.jl by Kurama Okubo
"""
function backtrackDistanceFunction(dir::Int64, d::Array{Float64,2}, err::Array{Float64,2}, lmin::Int64, b::Int64)
    nSample = size(d,1); # number of samples
    nLag    = size(d,2); # number of lags
    stbar   = zeros(Int64, nSample); # allocate

    #--------------------------------------------------------------------------
    # Setup indices based on forward or backward accumulation direction
    #--------------------------------------------------------------------------
    if dir > 0            # FORWARD
        iBegin = 1;       # start index
        iEnd   = nSample; # end index
        iInc   = 1;       # increment
    else                  # BACKWARD
        iBegin = nSample; # start index
        iEnd   = 1;       # stop index
        iInc   = -1;      # increment
    end
    #--------------------------------------------------------------------------
    # start from the end (front or back)
    ll0 = argmin(d[iBegin,:]); # find minimum accumulated distance at front or back depending on 'dir'
    stbar[iBegin] = ll0 + lmin - 1; # absolute value of integer shift
    #--------------------------------------------------------------------------
    # move through all time samples in forward or backward direction
    ii = iBegin;

    while ii != iEnd
        if ii == iBegin
            ll = ll0;
        else
            ll = ll_next
        end

        # min/max for edges/boundaries
        ji = max( 1, min(nSample, ii + iInc) );
        jb = max( 1, min(nSample, ii + iInc * b) );

        # -----------------------------------------------------------------
        # check limits on lag indices

        lMinus1 = ll - 1; # lag at l-1

        if lMinus1 < 1 # check lag index is greater than 1
            lMinus1 = 1; # make lag = first lag
        end

        lPlus1 = ll + 1; # lag at l+1

        if lPlus1 > nLag # check lag index less than max lag
            lPlus1 = nLag; # D.M. and D.Y. version
        end
        # -----------------------------------------------------------------
        # get distance at lags (ll-1, ll, ll+1)
        distLminus1 = d[jb, lMinus1]; # minus:  d( i-b, j-1 )
        distL       = d[ji, ll];      # actual: d( i-1, j   )
        distLplus1  = d[jb, lPlus1];  # plus:   d( i-b, j+1 )

        if ji != jb # equation 10 in Hale (2013)
            for kb = ji:iInc:jb-iInc # sum errors over i-1:i-b+1
                distLminus1 = distLminus1 + err[kb, lMinus1];
                distLplus1  = distLplus1  + err[kb, lPlus1];
            end
        end

        dl = min(distLminus1, distL, distLplus1); # update minimum distance to previous sample

        if ( dl != distL ) # then ll != ll and we check forward and backward
            if ( dl == distLminus1 )
                global ll_next = lMinus1;
            else # ( dl == lPlus1 )
                global ll_next = lPlus1;
            end
        else
            ll_next = ll
        end

        # assume ii = ii - 1
        ii += iInc; # previous time sample

        stbar[ii] = ll_next + lmin - 1; # absolute integer of lag
        # now move to correct time index, if smoothing difference over many
        # time samples using 'b'

        if ( ll_next == lMinus1 || ll_next == lPlus1 ) # check edges to see about b values
            if ( ji != jb ) # if b>1 then need to move more steps
                for kb = ji:iInc:jb - iInc
                    ii = ii + iInc; # move from i-1:i-b-1
                    stbar[ii] = ll_next + lmin - 1; # constant lag over that time
                end
            end
        end
    end

    return stbar
end


"""

    computeDTWerror(Aerr, u, lag0)

Compute the accumulated error along the warping path for DTW

# Arguments
- `Aerr::Array{Float64,2}`: error matrix (see eqn. 13 in Hale, 2013)
- `u::Array{Int64,1}`: warping function
- `lag0::Int64`: value of minimum lag

# Returns
- `error::AbstractArray`: accumulatd error along warping path

Original by Dylan Mikesell and modified from DTWDT.jl by Kurama Okubo
"""
function computeDTWerror(Aerr::Array{Float64,2}, u::Array{Int64,1}, lag0::Int)
    npts = length(u);

    if size(Aerr,1) != npts
        error("Funny things with dimensions of error matrix: check inputs.")
    end

    error = 0; # initialize

    # accumulate error
    for ii = 1:npts
        idx = lag0 + 1 + u[ii]; # index of lag
        error += Aerr[ii,idx]; # sum error
    end

    return error
end
