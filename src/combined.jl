struct CombinedModel{M <: Tuple, S<:Tuple} <: AbstractLogDensityModel
  models::M
  factors::S
end

function LD.logdensity(c::CombinedModel,x)
  sum(f*LD.logdensity(m,x) for (m,f) in zip(c.models,c.factors))
end

LD.dimension(m::CombinedModel) = LD.dimension(first(m.models))
LD.capabilities(m::Type{<:CombinedModel}) = LD.LogDensityOrder{0}()

unwrap(m::CombinedModel) = first(m.models)
