"""
    AttractorStates

Attractor dynamics for system state convergence.
"""
module AttractorStates

export AttractorField, compute_gradient, evolve_state

"""
    AttractorField

Potential field defining attractor dynamics.
"""
struct AttractorField
    attractors::Vector{Vector{Float64}}
    strengths::Vector{Float64}
end

"""
    compute_gradient(field::AttractorField, state::Vector{Float64}) -> Vector{Float64}

Compute ∇V(s) for gradient descent: ds/dt = -∇V(s)
"""
function compute_gradient(field::AttractorField, state::Vector{Float64})
    gradient = zeros(Float64, length(state))
    
    for (attractor, strength) in zip(field.attractors, field.strengths)
        diff = attractor .- state
        distance = sqrt(sum(diff .^ 2) + 1e-10)
        
        # Gradient points toward attractor, scaled by strength
        gradient .+= strength .* (diff ./ distance)
    end
    
    return gradient
end

"""
    evolve_state(field::AttractorField, state::Vector{Float64}, dt::Float64) -> Vector{Float64}

Evolve state by timestep dt using gradient descent.
"""
function evolve_state(field::AttractorField, state::Vector{Float64}, dt::Float64)
    gradient = compute_gradient(field, state)
    return state .+ dt .* gradient
end

"""
    find_nearest_attractor(field::AttractorField, state::Vector{Float64}) -> Int

Find index of nearest attractor to current state.
"""
function find_nearest_attractor(field::AttractorField, state::Vector{Float64})
    distances = [sqrt(sum((a .- state) .^ 2)) for a in field.attractors]
    return argmin(distances)
end

end # module AttractorStates
