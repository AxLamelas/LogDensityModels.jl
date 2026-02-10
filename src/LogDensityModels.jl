module LogDensityModels

export flatten, Descriptor
export FlattenedModel,TransformedModel,ConditionedModel,
TransformedConditionedModel, DistributionModel, CombinedModel, TaskLocalModel

import LogDensityProblems as LD
using ModelFlatten
using ModelFlatten: Fixed
using TaskLocalValues


abstract type AbstractLogDensityModel end

function unwrap(::AbstractLogDensityModel) end

iswrapped(m::AbstractLogDensityModel) = m !== unwrap(m)

transform(m::AbstractLogDensityModel,x) = if iswrapped(m)
  transform(unwrap(m),x)
else
  throw(error("Model does not have a transform!"))
end

inverse(m::AbstractLogDensityModel,x) = if iswrapped(m)
  inverse(unwrap(m),x)
else
  throw(error("Model does not have an inverse transform!"))
end

get_priors(m::AbstractLogDensityModel) = if iswrapped(m)
  get_priors(unwrap(m))
else
  throw(error("Model does not have associated priors"))
end

function unwrap(m) end

Base.show(io::IO,m::AbstractLogDensityModel) = if iswrapped(m)
  print(io,string(Base.nameof(typeof(m)))*"(")
  show(io,unwrap(m))
  print(io,")")
else
  print(io,string(Base.nameof(typeof(m))))
end

include("flattened.jl")
include("transformed.jl")
include("conditioned/flattened.jl")
include("conditioned/transformed.jl")
include("tasklocal.jl")
include("distribution.jl")
include("scaled.jl")
include("combined.jl")


end # module LogDensityModels
