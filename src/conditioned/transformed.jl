struct TransformedConditionedModel{M,D,B,I} <: AbstractLogDensityModel
  model::M
  unflatten::D
  transform::B
  inverse::I
end

function TransformedConditionedModel(model; kwargs...)
  umodel = unwrap(model)
  # Only works for top level => change to use accesors and optics
  priors = (;(k in keys(kwargs) ? k => Fixed(kwargs[k]) : k => v for (k,v) in pairs(get_priors(umodel)))...)
  unflatten = Descriptor(priors)
  transform,inverse = setup_transforms(priors)
  return TransformedConditionedModel(model,unflatten,transform,inverse)
end

function LD.logdensity(m::TransformedConditionedModel,x)
  y,ℓ = with_logabsdet_jacobian(m.inverse,x)
  θ = m.unflatten(y)
  return ℓ + LD.logdensity(m.model,θ)
end

LD.dimension(m::TransformedConditionedModel) = m.transform.length_out
LD.capabilities(::TransformedConditionedModel) = LD.LogDensityOrder{0}()

get_priors(m::TransformedConditionedModel) = get_priors(m.model)
unwrap(m::TransformedConditionedModel) = m.model
