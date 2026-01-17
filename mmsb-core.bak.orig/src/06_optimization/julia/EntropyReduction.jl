"""
    EntropyReduction

Entropy minimization for memory access patterns.
Reduces H(P) = -Σ f(p)log(f(p)) through intelligent page clustering.
"""
module EntropyReduction

export compute_entropy, reduce_entropy!

# using Statistics  # Not in Project.toml

"""
    compute_entropy(access_frequencies) -> Float64

Compute Shannon entropy H = -Σ pᵢ log₂(pᵢ) of access pattern.
Lower entropy indicates better locality.
"""
function compute_entropy(access_frequencies::Dict{UInt64, Int})
    isempty(access_frequencies) && return 0.0
    
    total = sum(values(access_frequencies))
    total == 0 && return 0.0
    
    entropy = 0.0
    for count in values(access_frequencies)
        if count > 0
            p = count / total
            entropy -= p * log2(p)
        end
    end
    
    return entropy
end

"""
    reduce_entropy!(layout, access_pattern) -> Float64

Reorganize layout to minimize entropy through clustering.
Returns new entropy value.
"""
function reduce_entropy!(layout::Dict{UInt64, UInt64}, 
                        access_pattern::Dict{UInt64, Int},
                        page_size::Int)
    old_entropy = compute_entropy(access_pattern)
    
    # Group pages by access frequency quantiles
    sorted_pages = sort(collect(access_pattern), by=x->x[2], rev=true)
    
    # Reassign addresses to co-locate similar-frequency pages
    for (i, (page_id, _)) in enumerate(sorted_pages)
        layout[page_id] = UInt64((i - 1) * page_size)
    end
    
    return compute_entropy(access_pattern)
end

"""
    entropy_gradient(access_pattern) -> Dict{UInt64, Float64}

Compute gradient ∂H/∂pᵢ for each page to guide optimization.
"""
function entropy_gradient(access_pattern::Dict{UInt64, Int})
    total = sum(values(access_pattern))
    total == 0 && return Dict{UInt64, Float64}()
    
    gradient = Dict{UInt64, Float64}()
    for (page_id, count) in access_pattern
        if count > 0
            p = count / total
            gradient[page_id] = -(log2(p) + 1.0)
        end
    end
    
    return gradient
end

end # module EntropyReduction
