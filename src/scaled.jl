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
LD.capabilities(m::Type{<:ScaledModel}) = LD.LogDensityOrder{0}()

unwrap(m::ScaledModel) = m.model
