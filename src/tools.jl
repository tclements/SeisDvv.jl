export dvv_lstsq

"""
    dvv_lstsq(time_axis, dt, w)

Linear regression of phase shifts to time to give dt/t=-dv/v

# Arguments
- `time_axis::AbstractArray`: Array of lag times
- `dt::AbstractArray`: Time shifts for each lag time
- `w::AbstractArray`: Weights for the linear regression

# Returns
- `m::Float64`: dt/t for current correlation
- `em::Float64`: Error for calculation of `m`
- `a::Float64`: Intercept for regression calculation
- `ea::Float64`: Error on intercept
- `m0::Float64`: dt/t for current correlation with no intercept
- `em0::Float64`: Error for calculation of `m0`
"""
function dvv_lstsq(time_axis::AbstractArray, dt::AbstractArray; w::AbstractArray=ones(length(time_axis)))
    # regress data using least squares
    # force line through origin
    model0 = glm(@formula(Y ~0 + X),DataFrame(X=time_axis,Y=dt),Normal(),
                IdentityLink(),wts=w)
    # allow nonzero intercept
    model = glm(@formula(Y ~ X),DataFrame(X=time_axis,Y=dt),Normal(),
                IdentityLink(),wts=w)

    # parameters for the regression with nonzero intercept
    a,m = -coef(model).*100
    ea, em = stderror(model).*100
    # parameters for the regression through the origin
    m0 = -coef(model0)[1]*100
    em0 = stderror(model0)[1]*100

    return m, em, a, ea, m0, em0
end
