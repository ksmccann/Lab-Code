# To get these just do `Pkg.add("Parameters")` etc
using Parameters # for the `@unpack` feature for having nice Mathematica like parameters
using ForwardDiff # autodiff library
using NLsolve # root solver library

function model(t, y, par::Dict)
    @unpack r, K, aRC, aCP, aRP, eRC, eCP, eRP, mC, mP = par
    R, C, P = y # just to make the formula's look nicer, could do y[1] etc
    yprime = similar(y) # make an array of the same type and size to y
    yprime[1] = r * R * (1 - R / K) - aRC * R * C - aRP * R * P
    yprime[2] = eRC * aRC * R * C - aCP * C * P - mC * C
    yprime[3] = eCP * aCP * C * P + eRP * aRP * R * P - mP * P
    return yprime
end

# Get a Jacobian using Automatic Differentiation!
# Note: I am giving a time of 0.0 since this is only needed for the ODE solver and not as
# an input for the jacobion, it coud be any value as or diffeq doesn't depend on time
#
# the `y -> model(0.0, y, par)` is just so that you can supply a given set of parameters
# but then create an input function (the y -> blah, is a nameless function) for the
# autodiff package which just wants a function that takes a single vector argument (in this
# case `y`)
cmat(eq, par) = ForwardDiff.jacobian(y -> model(0.0, y, par), eq)

# pick an example parameter set, a Dict is just an array like object that makes keys and values,
# in this case I am mapping "symbols" (the :name notation) to values, a key value pair is
# given by key => value notation, kind of like Mathematica.
# This is nice as you can then index the "array" using the parameter names, like par1[:r],
# instead of having to remember which index is r
par1 = Dict(
    :r => 2.0,
    :K => 3.0,
    :aRC => 1.1,
    :aCP => 0.7,
    :aRP => 0.1,
    :eRC => 1.0,
    :eCP => 1.0,
    :eRP => 1.0,
    :mC => 0.7,
    :mP => 0.7
    )

# The `not_in_place` is just a strange thing for the root solver, to basically say you
# are not using fortran like subroutines (where you pass the output array in as an argument)
eq = nlsolve(not_in_place(y -> model(0.0, y, par1)), [1.5, 1.5, 1.5]).zero

# Sweet sweet eigenvalues!
eigvals(cmat(eq, par1))
