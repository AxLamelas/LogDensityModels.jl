struct FlattenedModel{M,D}
  model::M
  unflatten::D
end

function FlattenedModel(model)
  priors = get_priors(model)
  unflatten = Descriptor(priors)

  return FlattenedModel(model,unflatten)
end

function LD.logdensity(m::FlattenedModel,x)
  θ = m.unflatten(x)
  return LD.logdensity(m.model,θ)
end

LD.dimension(m::FlattenedModel) = LD.dimension(m.model)
LD.capabilities(m::Type{<:FlattenedModel}) = LD.LogDensityOrder{0}()

get_priors(m::FlattenedModel) = get_priors(m.model)
unwrap(m::FlattenedModel) = unwrap(m.model)
