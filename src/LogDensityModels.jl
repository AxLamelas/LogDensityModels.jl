module LogDensityModels

export flatten, Descriptor
export FlattenedModel,TransformedModel,ConditionedModel,
TransformedConditionedModel, DistributionModel, CombinedModel

import LogDensityProblems as LD
using ModelFlatten
using ModelFlatten: Fixed

include("flattened.jl")
include("transformed.jl")
include("conditioned/flattened.jl")
include("conditioned/transformed.jl")

# TODO: rename or remove the need for get_priors as it does not make sense for all models
function get_priors(m) end
function unwrap(m) end

struct DistributionModel{D}
  dist::D
end

function LD.logdensity(m::DistributionModel,x)
  T = flat_eltype(x)
  if !insupport(m.dist,x)
    return typemin(T)
  end
  # Type assert to deal with type instability in Distributions.jl
  return logpdf(m.dist,x)::T
end

function LD.logdensity(m::DistributionModel,x::AbstractVector)
  T = eltype(x)
  if !insupport(m.dist,x)
    return typemin(T)
  end
  # Type assert to deal with type instability in Distributions.jl
  return logpdf(m.dist,x)::T
end

LD.dimension(m::DistributionModel) = length(m.dist)
LD.capabilities(::Type{<:DistributionModel}) = LD.LogDensityOrder{0}()

get_priors(m::DistributionModel) = m.dist
unwrap(m::DistributionModel) = m

struct CombinedModel{M <: Tuple, S<:Tuple}
  models::M
  factors::S
end

function LD.logdensity(c::CombinedModel,x)
  sum(f*LD.logdensity(m,x) for (m,f) in zip(c.models,c.factors))
end

LD.dimension(m::CombinedModel) = LD.dimension(first(m.models))
LD.capabilities(m::Type{<:CombinedModel}) = LD.LogDensityOrder{0}()

unwrap(m::CombinedModel) = m.models

struct ScaledModel{M, T}
  model::M
  scales::Vector{T}
  logdetjac::T
end

function ScaledModel(model,scales)
  @assert LD.dimension(model) == length(scales)
  @assert all(scales .> 0)
  return ScaledModel(model,scales,-sum(log,scales))
end

function LD.logdensity(m::ScaledModel,x)
  # Apply scaling
  for i in eachindex(x)
    x[i] /= m.scales[i]
  end
  ℓ = LD.logdensity(m.model,x)
  # Revert scaling
  for i in eachindex(x)
    x[i] *= m.scales[i]
  end

  return ℓ + m.logdetjac
end

LD.dimension(m::ScaledModel) = LD.dimension(m.model)
LD.capabilities(m::Type{<:ScaledModel}) = LD.LogDensityOrder{0}()

unwrap(m::ScaledModel) = m.model

end # module LogDensityModels
