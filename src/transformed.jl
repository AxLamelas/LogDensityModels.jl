struct TransformedModel{M,D,B,I} <: AbstractLogDensityModel
  model::M
  unflatten::D
  transform::B
  inverse::I
end

function TransformedModel(model)
  priors = get_priors(model)
  unflatten = Descriptor(priors)
  transform,inverse = setup_transforms(priors)

  return TransformedModel(model,unflatten,transform,inverse)
end

(m::TransformedModel)(x) = m.model(m.unflatten(m.inverse(x)))

function LD.logdensity(m::TransformedModel,x::AbstractVector)
  y,ℓ = with_logabsdet_jacobian(m.inverse,x)
  θ = m.unflatten(y)
  return ℓ + LD.logdensity(m.model,θ)
end

LD.dimension(m::TransformedModel) = if hasproperty(m.transform,:length_out)
  m.transform.length_out
else
  LD.dimension(m.model)
end
LD.capabilities(m::Type{<:TransformedModel}) = LD.LogDensityOrder{0}()

get_priors(m::TransformedModel) = get_priors(m.model)
unwrap(m::TransformedModel) = m.model

transform(m::TransformedModel,x::AbstractVector) = m.transform(x)
transform(m::TransformedModel,θ) = m.transform(flatten(m.unflatten,θ))
inverse(m::TransformedModel,x::AbstractVector) = m.inverse(x)

