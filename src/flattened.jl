struct FlattenedModel{M,D} <: AbstractLogDensityModel
  model::M
  unflatten::D
end

function FlattenedModel(model)
  priors = get_param_distribution(model)
  unflatten = Descriptor(priors)

  return FlattenedModel(model,unflatten)
end

function LD.logdensity(m::FlattenedModel,x)
  θ = m.unflatten(x)
  return LD.logdensity(m.model,θ)
end

LD.dimension(m::FlattenedModel) = length(m.unflatten)
LD.capabilities(::Type{<:FlattenedModel}) = LD.LogDensityOrder{0}()

get_param_distribution(m::FlattenedModel) = product_distribution(flatten(get_param_distribution(m.model)))
unwrap(m::FlattenedModel) = m.model

local_transform(m::FlattenedModel,θ) = flatten(m.unflatten,θ)
local_inverse(m::FlattenedModel,x) = m.unflatten(x)

