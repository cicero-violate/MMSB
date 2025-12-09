"""
    AdaptiveLayout

Adaptive memory layout orchestration for Layer 5.

Coordinates page reordering, clustering, and locality optimization
to minimize cache misses and improve memory access patterns.
"""
module AdaptiveLayout

export LayoutState, optimize_layout!, compute_locality_score, PageId, PhysAddr

# Type aliases for Layer 5
const PageId = UInt64
const PhysAddr = UInt64

"""
    LayoutState

Current memory layout state with placement mapping and metrics.
"""
mutable struct LayoutState
    placement::Dict{PageId, PhysAddr}
    page_size::Int
    last_optimization::Float64
    locality_score::Float64
end

function LayoutState(page_size::Int)
    LayoutState(Dict{PageId, PhysAddr}(), page_size, 0.0, 0.0)
end

"""
    optimize_layout!(state::LayoutState, access_pattern)

Reorder pages to minimize locality cost based on access patterns.
Returns improvement ratio (< 1 means improvement).
"""
function optimize_layout!(state::LayoutState, access_pattern::Dict{Tuple{PageId, PageId}, Int})
    old_score = compute_locality_score(state, access_pattern)
    
    # Extract all pages
    pages = collect(keys(state.placement))
    isempty(pages) && return 1.0
    
    # Sort by hotness (total co-access frequency)
    hotness = Dict{PageId, Int}()
    for ((p1, p2), freq) in access_pattern
        hotness[p1] = get(hotness, p1, 0) + freq
        hotness[p2] = get(hotness, p2, 0) + freq
    end
    
    sort!(pages, by = p -> get(hotness, p, 0), rev=true)
    
    # Reassign addresses sequentially
    for (i, page_id) in enumerate(pages)
        state.placement[page_id] = UInt64((i - 1) * state.page_size)
    end
    
    new_score = compute_locality_score(state, access_pattern)
    state.locality_score = new_score
    state.last_optimization = time()
    
    return new_score / (old_score + 1e-10)
end

"""
    compute_locality_score(state, access_pattern) -> Float64

Compute weighted distance cost: Σ_{(p₁,p₂)} freq(p₁,p₂) × distance(p₁,p₂)
"""
function compute_locality_score(state::LayoutState, access_pattern::Dict{Tuple{PageId, PageId}, Int})
    cost = 0.0
    for ((p1, p2), freq) in access_pattern
        addr1 = get(state.placement, p1, nothing)
        addr2 = get(state.placement, p2, nothing)
        
        if !isnothing(addr1) && !isnothing(addr2)
            distance = abs(Int(addr1) - Int(addr2)) ÷ state.page_size
            cost += distance * freq
        end
    end
    return cost
end

end # module AdaptiveLayout
