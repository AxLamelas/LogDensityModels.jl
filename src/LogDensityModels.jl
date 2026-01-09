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

get_priors(m)
unwrap(m)

struct DistributionModel{D}
  dist::D
end

LD.logdensity(m::DistributionModel,x) = logpdf(m.dist,x)
LD.dimension(m::DistributionModel) = length(m.dist)
LD.capabilities(m::Type{<:DistributionModel}) = LD.LogDensityOrder{0}()

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

end # module LogDensityModels
