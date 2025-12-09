"""
    RLPlanning

Reinforcement learning-based planning.
"""
module RLPlanning

export value_iteration, policy_iteration, q_learning, temporal_difference

using ..PlanningTypes

"""
    value_iteration(states, actions, γ, ε) -> Dict{UInt64, Float64}

Compute optimal value function via dynamic programming.
"""
function value_iteration(states::Vector{State}, actions::Vector{Action}, γ::Float64=0.9, ε::Float64=0.01)
    V = Dict{UInt64, Float64}()
    for s in states
        V[s.id] = 0.0
    end
    
    max_iter = 1000
    for iter in 1:max_iter
        Δ = 0.0
        
        for s in states
            v_old = V[s.id]
            
            # Bellman update
            values = Float64[]
            for a in actions
                # Expected return
                q_val = immediate_reward(s, a) + γ * expected_next_value(s, a, V, states)
                push!(values, q_val)
            end
            
            V[s.id] = isempty(values) ? 0.0 : maximum(values)
            Δ = max(Δ, abs(V[s.id] - v_old))
        end
        
        if Δ < ε
            break
        end
    end
    
    V
end

function immediate_reward(s::State, a::Action)
    s.utility - a.cost
end

function expected_next_value(s::State, a::Action, V::Dict{UInt64, Float64}, states::Vector{State})
    # Simplified: assume deterministic transition
    # In full implementation, sum over possible next states
    next_id = s.id + 1
    get(V, next_id, 0.0)
end

"""
    policy_iteration(states, actions, γ) -> Dict{UInt64, Action}

Find optimal policy via policy iteration.
"""
function policy_iteration(states::Vector{State}, actions::Vector{Action}, γ::Float64=0.9)
    # Initialize random policy
    π = Dict{UInt64, Action}()
    for s in states
        π[s.id] = rand(actions)
    end
    
    V = Dict{UInt64, Float64}()
    for s in states
        V[s.id] = 0.0
    end
    
    policy_stable = false
    max_iter = 100
    
    for iter in 1:max_iter
        # Policy evaluation
        V = evaluate_policy(π, states, actions, γ, V)
        
        # Policy improvement
        policy_stable = true
        for s in states
            old_action = π[s.id]
            
            # Find best action
            best_action = old_action
            best_value = -Inf
            
            for a in actions
                q_val = immediate_reward(s, a) + γ * expected_next_value(s, a, V, states)
                if q_val > best_value
                    best_value = q_val
                    best_action = a
                end
            end
            
            π[s.id] = best_action
            if best_action.id != old_action.id
                policy_stable = false
            end
        end
        
        if policy_stable
            break
        end
    end
    
    π
end

function evaluate_policy(π::Dict{UInt64, Action}, states::Vector{State}, actions::Vector{Action}, γ::Float64, V::Dict{UInt64, Float64})
    for _ in 1:20  # Fixed number of iterations
        for s in states
            a = π[s.id]
            V[s.id] = immediate_reward(s, a) + γ * expected_next_value(s, a, V, states)
        end
    end
    V
end

"""
    q_learning(episodes, α, γ, ε) -> Dict

Learn Q-values via temporal difference learning.
"""
function q_learning(episodes::Vector{Vector{Tuple{State, Action, Float64}}}, α::Float64=0.1, γ::Float64=0.9, ε::Float64=0.1)
    Q = Dict{Tuple{UInt64, UInt64}, Float64}()
    
    for episode in episodes
        for (s, a, r) in episode
            key = (s.id, a.id)
            q_old = get(Q, key, 0.0)
            
            # Find max Q for next state (simplified)
            max_q_next = 0.0
            
            # Q-learning update
            Q[key] = q_old + α * (r + γ * max_q_next - q_old)
        end
    end
    
    Q
end

"""
    temporal_difference(trajectory, α, γ) -> Dict

TD(0) learning from trajectory.
"""
function temporal_difference(trajectory::Vector{Tuple{State, Action, Float64}}, α::Float64=0.1, γ::Float64=0.9)
    V = Dict{UInt64, Float64}()
    
    for i in 1:length(trajectory)
        s, a, r = trajectory[i]
        v_old = get(V, s.id, 0.0)
        
        # Get next state value
        v_next = if i < length(trajectory)
            get(V, trajectory[i+1][1].id, 0.0)
        else
            0.0
        end
        
        # TD update
        V[s.id] = v_old + α * (r + γ * v_next - v_old)
    end
    
    V
end

end # module
