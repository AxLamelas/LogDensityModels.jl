module LogDensityModels

export flatten, Descriptor
export FlattenedModel,TransformedModel,ConditionedModel, ScaledModel,
DistributionModel, CombinedModel, TaskLocalModel

import LogDensityProblems as LD
using ModelFlatten
using ModelFlatten: Fixed
using TaskLocalValues


abstract type AbstractLogDensityModel end

# By default the model is not wrapped
unwrap(m::AbstractLogDensityModel) = m
iswrapped(m::AbstractLogDensityModel) = m !== unwrap(m)

# Default transformations -> identity
local_transform(::AbstractLogDensityModel,x) = x
local_inverse(::AbstractLogDensityModel,x) = x
transform(m::AbstractLogDensityModel,x) = if iswrapped(m)
	local_transform(m,transform(unwrap(m),x))
else
	local_transform(m,x)
end

inverse(m::AbstractLogDensityModel,x) = if iswrapped(m)
	inverse(unwrap(m),local_inverse(m,x))
else
	local_inverse(m,x)
end

transform(m,x) = x
inverse(m,x) = x


get_param_distribution(m::AbstractLogDensityModel) = if iswrapped(m)
  get_param_distribution(unwrap(m))
else
  throw(error("Model does not have an associated parameter distribution."))
end

Base.show(io::IO,m::AbstractLogDensityModel) = if iswrapped(m)
  print(io,string(Base.nameof(typeof(m)))*"(")
  show(io,unwrap(m))
  print(io,")")
else
  print(io,string(Base.nameof(typeof(m))))
end

include("flattened.jl")
include("transformed.jl")
include("conditioned.jl")
include("tasklocal.jl")
include("distribution.jl")
include("scaled.jl")
include("combined.jl")


end # module LogDensityModels
