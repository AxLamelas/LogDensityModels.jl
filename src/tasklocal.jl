struct TaskLocalModel{T} <: AbstractLogDensityModel
  model::T
  function TaskLocalModel(model)
    t = TaskLocalValue{typeof(model)}(Base.Fix1(deepcopy,model))
    return new{typeof(t)}(t)
  end
end

LD.logdensity(t::TaskLocalModel,x) = LD.logdensity(t.model[],x)
LD.logdensity_and_gradient(t::TaskLocalModel,x) = LD.logdensity_and_gradient(t.model[],x)
LD.logdensity_gradient_and_hessian(t::TaskLocalModel,x) = LD.logdensity_gradient_and_hessian(t.model[],x)
LD.dimension(t::TaskLocalModel) = LD.dimension(t.model[])
LD.capabilities(t::TaskLocalModel) = LD.capabilities(t.model[])

unwrap(t::TaskLocalModel) = t.model[]
