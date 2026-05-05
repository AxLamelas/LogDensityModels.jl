struct ScaledModel{M, T} <: AbstractLogDensityModel
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
LD.capabilities(::Type{<:ScaledModel}) = LD.LogDensityOrder{0}()

unwrap(m::ScaledModel) = m.model

local_transform(m::ScaledModel,x) = m.scales .* x
local_inverse(m::ScaledModel,x) = x ./ m.scales

get_param_distribution(m::ScaledModel) = 
	DiagAffineDistribution(zeros(length(m.scales)),m.scales,get_param_distribution(m.model))


struct DiagAffineDistribution{T,W<:AbstractVector{T},S<:ValueSupport,D<:MultivariateDistribution{S}} <: MultivariateDistribution{S}
	location::W
	scale::W
	ρ::D
end

function DiagAffineDistribution(
	location::AbstractVector{<: Real},
	scale::AbstractVector{<:Real}, ρ::MultivariateDistribution)

	# Distributions.@check_args DiagAffineDistribution (scale, all(!iszero,scale))
	D = typeof(ρ)
	S = Distributions.value_support(D)
	location,scale = promote(location,scale)
	T = eltype(location)
	W = typeof(location)
	return DiagAffineDistribution{T,W,S,D}(location,scale, ρ)
end

const ContinuousDiagAffineDistribution{T<:Real,W,D<:ContinuousMultivariateDistribution} = DiagAffineDistribution{T,W,Continuous,D}
const DiscreteDiagAffineDistribution{T<:Real,W,D<:DiscreteMultivariateDistribution} = DiagAffineDistribution{T,W,Discrete,D}

Base.eltype(::Type{<:DiagAffineDistribution{T,W,S,D}}) where {T,W,S,D} = promote_type(eltype(D), T)


function Distributions.minimum(d::DiagAffineDistribution) 
	lb = minimum(d.ρ)
	ub = maximum(d.ρ)
	for i in eachindex(lb,ub)
		if d.scale[i]> 0 
			lb[i] = d.location[i] + d.scale[i]*lb[i]
		else
			lb[i] = d.location[i] + d.scale[i]*ub[i]
		end

	end
	return lb
end
function Distributions.maximum(d::DiagAffineDistribution) 
	lb = minimum(d.ρ)
	ub = maximum(d.ρ)
	for i in eachindex(lb,ub)
		if d.scale[i]> 0 
			ub[i] = d.location[i] + d.scale[i]*ub[i]
		else
			ub[i] = d.location[i] + d.scale[i]*lb[i]
		end

	end
	return ub
end

#### Parameters

Distributions.location(d::DiagAffineDistribution) = d.location
Distributions.scale(d::DiagAffineDistribution) = d.scale
Distributions.params(d::DiagAffineDistribution) = (d.location,d.scale,d.ρ)
Distributions.partype(d::DiagAffineDistribution{T}) where {T} = promote_type(partype(d.ρ), T)

#### Statistics

Distributions.mean(d::DiagAffineDistribution) = d.location .+ d.scale .* mean(d.ρ)
Distributions.median(d::DiagAffineDistribution) = d.location .+ d.scale .* median(d.ρ)
Distributions.mode(d::DiagAffineDistribution) = d.location .+ d.scale .* mode(d.ρ)


Distributions.var(d::DiagAffineDistribution) = d.scale.^2 .* var(d.ρ)
Distributions.std(d::DiagAffineDistribution) = abs.(d.scale) .* std(d.ρ)
Distributions.skewness(d::DiagAffineDistribution) = sign.(d.scale) .* skewness(d.ρ)
Distributions.kurtosis(d::DiagAffineDistribution) = kurtosis(d.ρ)

Distributions.isplatykurtic(d::DiagAffineDistribution) = isplatykurtic(d.ρ)
Distributions.isleptokurtic(d::DiagAffineDistribution) = isleptokurtic(d.ρ)
Distributions.ismesokurtic(d::DiagAffineDistribution) = ismesokurtic(d.ρ)

Distributions.entropy(d::ContinuousDiagAffineDistribution) = entropy(d.ρ) + sum(log ∘ abs,d.scale)
Distributions.entropy(d::DiscreteDiagAffineDistribution) = entropy(d.ρ)

Distributions.mgf(d::DiagAffineDistribution,t::AbstractVector{<:Real}) = exp(d.location*t) * mgf(d.ρ,d.scale .* t)

#### Evaluation & Sampling

Distributions.pdf(d::ContinuousDiagAffineDistribution, x::AbstractVector{<:Real}) = pdf(d.ρ,(x.-d.location)./d.scale) ./ abs(d.scale)
Distributions.pdf(d::DiscreteDiagAffineDistribution, x::AbstractVector{<:Real}) = pdf(d.ρ,(x.-d.location)./d.scale)

Distributions.logpdf(d::ContinuousDiagAffineDistribution,x::AbstractVector{<:Real}) = logpdf(d.ρ,(x.-d.location)./d.scale) - sum(log ∘ abs,	d.scale)
Distributions.logpdf(d::DiscreteDiagAffineDistribution, x::AbstractVector{<:Real}) = logpdf(d.ρ,(x.-d.location)./d.scale)

# # CDF methods
#
# for (f, fc) in ((:cdf, :ccdf), (:ccdf, :cdf), (:logcdf, :logccdf), (:logccdf, :logcdf))
#     @eval function $f(d::ContinuousDiagAffineDistribution, x::AbstractVector{<:Real})
#         z = (x - d.location) / d.scale
#         return d.scale > 0 ? $f(d.ρ, z) : $fc(d.ρ, z)
#     end
# end
#
# function Distributions.cdf(d::DiscreteDiagAffineDistribution, x::AbstractVector{<:Real})
#     z = (x - d.location) / d.scale
#     # Have to include probability mass at endpoints
#     return d.scale > 0 ? cdf(d.ρ, z) : (ccdf(d.ρ, z) + pdf(d.ρ, z))
# end
# function Distributions.ccdf(d::DiscreteDiagAffineDistribution, x::AbstractVector{<:Real})
#     z = (x - d.location) / d.scale
#     # Have to exclude probability mass at endpoints
#     return d.scale > 0 ? ccdf(d.ρ, z) : (cdf(d.ρ, z) - pdf(d.ρ, z))
# end
# function Distributions.logcdf(d::DiscreteDiagAffineDistribution, x::AbstractVector{<:Real})
#     z = (x - d.location) / d.scale
#     return d.scale > 0 ? logcdf(d.ρ, z) : logaddexp(logccdf(d.ρ, z), logpdf(d.ρ, z))
# end
# function Distributions.logccdf(d::DiscreteDiagAffineDistribution, x::AbstractVector{<:Real})
#     z = (x - d.location) / d.scale
#     return d.scale > 0 ? logccdf(d.ρ, z) : logsubexp(logcdf(d.ρ, z), logpdf(d.ρ, z))
# end

# Distributions.quantile(d::DiagAffineDistribution, q::AbstractVector{<:Real}) = d.location + d.scale * quantile(d.ρ, d.scale > 0 ? q : 1 - q)

Distributions.rand(rng::AbstractRNG, d::DiagAffineDistribution) = d.location .+ d.scale .* rand(rng, d.ρ)
# Distributions.cf(d::DiagAffineDistribution, t::AbstractVector{<:Real}) = cf(d.ρ,t*d.scale) * exp(1im*t*d.location)
# Distributions.gradlogpdf(d::ContinuousDiagAffineDistribution, x::AbstractVector{<:Real}) = gradlogpdf(d.ρ,(x .- d.location)./d.scale) ./ d.scale

Base.length(d::DiagAffineDistribution) = length(d.location)
Bijectors.bijector(d::ContinuousDiagAffineDistribution) = Bijectors.TruncatedBijector(minimum(d),maximum(d))
