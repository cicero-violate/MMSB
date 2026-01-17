"""
    EntropyMeasure

Shannon entropy computation for state distribution analysis.

Measures information content and unpredictability in system state,
used to quantify structural complexity and optimization potential.
"""
module EntropyMeasure

export compute_entropy, state_entropy, PageDistribution

"""
    PageDistribution

Distribution over page states for entropy computation.
"""
struct PageDistribution
    counts::Dict{UInt64, Int}
    total::Int
end

function PageDistribution(counts::Dict{UInt64, Int})
    PageDistribution(counts, sum(values(counts)))
end

"""
    compute_entropy(distribution::PageDistribution) -> Float64

Compute Shannon entropy: H(X) = -Σᵢ p(xᵢ) log₂(p(xᵢ))
"""
function compute_entropy(dist::PageDistribution)
    dist.total == 0 && return 0.0
    
    entropy = 0.0
    for count in values(dist.counts)
        count == 0 && continue
        
        p = count / dist.total
        entropy -= p * log2(p)
    end
    
    return entropy
end

"""
    state_entropy(access_pattern::Dict) -> Float64

Compute entropy of access pattern distribution.
"""
function state_entropy(access_pattern::Dict{Tuple{UInt64, UInt64}, Int})
    isempty(access_pattern) && return 0.0
    
    total = sum(values(access_pattern))
    entropy = 0.0
    
    for freq in values(access_pattern)
        freq == 0 && continue
        p = freq / total
        entropy -= p * log2(p)
    end
    
    return entropy
end

"""
    entropy_reduction(old_entropy, new_entropy) -> Float64

Compute relative entropy reduction: ΔH = (H_old - H_new) / H_old
"""
function entropy_reduction(old_entropy::Float64, new_entropy::Float64)
    old_entropy == 0.0 && return 0.0
    return (old_entropy - new_entropy) / old_entropy
end

end # module EntropyMeasure
