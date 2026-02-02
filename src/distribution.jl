struct DistributionModel{D} <: AbstractLogDensityModel
  dist::D
end

function LD.logdensity(m::DistributionModel,x)
  T = flat_eltype(x)
  if !insupport(m.dist,x)
    return typemin(T)
  end
  # Type assert to deal with type instability in Distributions.jl
  return logpdf(m.dist,x)::T
end

function LD.logdensity(m::DistributionModel,x::AbstractVector)
  T = eltype(x)
  if !insupport(m.dist,x)
    return typemin(T)
  end
  # Type assert to deal with type instability in Distributions.jl
  return logpdf(m.dist,x)::T
end

LD.dimension(m::DistributionModel) = length(m.dist)
LD.capabilities(::Type{<:DistributionModel}) = LD.LogDensityOrder{0}()

get_priors(m::DistributionModel) = m.dist
unwrap(m::DistributionModel) = m
