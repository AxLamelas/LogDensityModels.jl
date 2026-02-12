struct CombinedModel{M <: Tuple, S<:Tuple} <: AbstractLogDensityModel
  models::M
  factors::S
end

LD.logdensity(c::CombinedModel,x) = mapreduce(+,c.models,c.factors) do m,f
  f * LD.logdensity(m,x)
end

LD.dimension(m::CombinedModel) = LD.dimension(first(m.models))
LD.capabilities(m::Type{<:CombinedModel}) = LD.LogDensityOrder{0}()

unwrap(m::CombinedModel) = first(m.models)
