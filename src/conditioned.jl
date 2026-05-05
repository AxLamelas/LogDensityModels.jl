struct ConditionedModel{M,C,F} <: AbstractLogDensityModel
	model::M
	cond::C
	free::F
	dim::Int
end

# TODO: switch to Accessors.jl to make the conditioning more flexible
# TODO: see if the same code can be generalized/adapted to allow conditioning on vector indices as well

function ConditionedModel(model; kwargs...)
	base_dist = get_param_distribution(model)
	for k in keys(kwargs)
		if !(k in keys(base_dist))
			@warn "Conditioning on $(k) but it is not a parameter of the model"
		end
	end
	free = Tuple(k for k in keys(base_dist) if !(k in keys(kwargs)))
	dim = sum(length(base_dist[k]) for k in free)
	return ConditionedModel(model,kwargs,free,dim)
end

function LD.logdensity(m::ConditionedModel,θ)
	return LD.logdensity(m.model,(; θ..., m.cond...))
end

LD.dimension(m::ConditionedModel) = m.dim
LD.capabilities(::Type{ConditionedModel}) = LD.LogDensityOrder{0}()

function get_param_distribution(m::ConditionedModel) 
	base_dist = get_param_distribution(unwrap(m))
	(;(k => base_dist[k] for k in m.free)...)
end

unwrap(m::ConditionedModel) = m.model

