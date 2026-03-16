struct TransformedModel{M,B,I} <: AbstractLogDensityModel
  model::M
  transform::B
  inverse::I
end

function TransformedModel(model)
  priors = get_param_distribution(model)
  transform,inverse = setup_transforms(priors)

  return TransformedModel(model,transform,inverse)
end

(m::TransformedModel)(x) = m.model(m.inverse(x))

function LD.logdensity(m::TransformedModel,x::AbstractVector)
  y,ℓ = with_logabsdet_jacobian(m.inverse,x)
  return ℓ + LD.logdensity(m.model,y)
end

LD.dimension(m::TransformedModel) = if hasproperty(m.transform,:length_out)
  m.transform.length_out
else
  LD.dimension(m.model)
end
LD.capabilities(::Type{<:TransformedModel}) = LD.LogDensityOrder{0}()

get_param_distribution(m::TransformedModel) = Bijectors.TransformedDistribution(
	get_param_distribution(unwrap(m)),m.transform)
unwrap(m::TransformedModel) = m.model

local_transform(m::TransformedModel,x) = m.transform(x)
local_inverse(m::TransformedModel,x) = m.inverse(x)

