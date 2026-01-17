"""
    CostAggregation

Aggregation layer for combining multiple cost signals into
unified metrics for utility optimization.
"""
module CostAggregation

export aggregate_costs, WeightedCost, normalize_costs

"""
    WeightedCost

Single cost metric with associated weight.
"""
struct WeightedCost
    name::Symbol
    value::Float64
    weight::Float64
end

"""
    aggregate_costs(costs::Vector{WeightedCost}) -> Float64

Compute weighted sum: C_total = Σᵢ wᵢ × cᵢ
"""
function aggregate_costs(costs::Vector{WeightedCost})
    isempty(costs) && return 0.0
    sum(c.weight * c.value for c in costs)
end

"""
    normalize_costs(costs::Vector{WeightedCost}) -> Vector{WeightedCost}

Normalize cost values to [0, 1] range for each component.
"""
function normalize_costs(costs::Vector{WeightedCost})
    isempty(costs) && return costs
    
    # Group by name to find min/max per component
    groups = Dict{Symbol, Vector{Float64}}()
    for c in costs
        if !haskey(groups, c.name)
            groups[c.name] = Float64[]
        end
        push!(groups[c.name], c.value)
    end
    
    # Compute normalization factors
    ranges = Dict{Symbol, Tuple{Float64, Float64}}()
    for (name, values) in groups
        min_val = minimum(values)
        max_val = maximum(values)
        ranges[name] = (min_val, max_val)
    end
    
    # Normalize each cost
    normalized = WeightedCost[]
    for c in costs
        min_val, max_val = ranges[c.name]
        normalized_value = if max_val - min_val < 1e-10
            0.0
        else
            (c.value - min_val) / (max_val - min_val)
        end
        push!(normalized, WeightedCost(c.name, normalized_value, c.weight))
    end
    
    return normalized
end

end # module CostAggregation
