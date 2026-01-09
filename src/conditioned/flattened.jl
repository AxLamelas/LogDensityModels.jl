struct ConditionedModel{M,D}
  model::M
  unflatten::D
end

# TODO: switch to Accessors.jl to make the conditioning more flexible

function ConditionedModel(model; kwargs...)
  umodel = unwrap(model)
  priors = (;(k in keys(kwargs) ? k => Fixed(kwargs[k]) : k => v for (k,v) in pairs(get_priors(umodel)))...)
  unflatten = Descriptor(priors)
  return ConditionedModel(model,unflatten)
end

function LD.logdensity(m::ConditionedModel,x)
  return LD.logdensity(m.model,m.unflatten(x))
end

LD.dimension(m::ConditionedModel) = length(m.unflatten)
LD.capabilities(m::ConditionedModel) = LD.LogDensityOrder{0}()

get_priors(m::ConditionedModel) = get_priors(m.model)
unwrap(m::ConditionedModel) = unwrap(m.model)

