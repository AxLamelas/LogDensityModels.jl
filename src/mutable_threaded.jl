struct MutableThreadSafeModel{T} <: AbstractLogDensityModel
	base_model::T
	model_channel::Channel{T}
  function MutableThreadSafeModel(model)
		model_channel = Channel{typeof(model)}(Threads.nthreads())
		for _ in 1:Threads.nthreads()
			put!(model_channel,deepcopy(model))
		end
		return new{typeof(model)}(model,model_channel)
  end
end

function LD.logdensity(t::MutableThreadSafeModel,x) 
	model = take!(t.model_channel)
	r = LD.logdensity(model,x)
	put!(t.model_channel,model)
	return r
end

function LD.logdensity_and_gradient(t::MutableThreadSafeModel,x) 
	model = take!(t.model_channel)
	r = LD.logdensity_and_gradient(model,x)
	put!(t.model_channel,model)
	return r
end

function LD.logdensity_gradient_and_hessian(t::MutableThreadSafeModel,x) 
	model = take!(t.model_channel)
	r = LD.logdensity_gradient_and_hessian(model,x)
	put!(t.model_channel,model)
	return r
end


LD.dimension(t::MutableThreadSafeModel) = LD.dimension(t.base_model)
LD.capabilities(t::MutableThreadSafeModel) = LD.capabilities(t.base_model)

unwrap(t::MutableThreadSafeModel) = t.base_model
