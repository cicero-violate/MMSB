# src/02_runtime/PageAllocator.jl
"""
PageAllocator - Page creation and lifecycle management

Handles allocation, initialization, and deallocation of pages.
Manages page ID assignment and memory allocation on CPU/GPU.
"""
module PageAllocator

using CUDA
using ..PageTypes: Page, PageID, PageLocation, CPU_LOCATION, GPU_LOCATION,
    UNIFIED_LOCATION, is_gpu_page
using ..MMSBStateTypes: MMSBState, _reserve_page_id_unlocked!
using ..GraphTypes: get_children, get_parents, remove_dependency!
using ..ErrorTypes: PageNotFoundError, GPUMemoryError, UnsupportedLocationError

export create_page!, delete_page!, migrate_page!, resize_page!, clone_page

"""
Create and register a page with the requested location.
"""
function create_page!(state::MMSBState, size::Int64, location::PageLocation;
                      metadata::Dict{Symbol,Any}=Dict{Symbol,Any}())::Page
    page = Page(PageID(0), size, location)
    if !isempty(metadata)
        page.metadata = Dict{Symbol,Any}(metadata)
    end
    lock(state.lock) do
        page.id = _reserve_page_id_unlocked!(state)
        state.pages[page.id] = page
    end
    return page
end

create_cpu_page!(state::MMSBState, size::Int64)::Page =
    create_page!(state, size, CPU_LOCATION)

function create_gpu_page!(state::MMSBState, size::Int64)::Page
    CUDA.functional() || throw(GPUMemoryError("CUDA is not available for GPU page creation"))
    return create_page!(state, size, GPU_LOCATION)
end

"""
Delete a page and all of its dependencies.
"""
function delete_page!(state::MMSBState, page_id::PageID)
    parents = get_parents(state.graph, page_id)
    children = get_children(state.graph, page_id)
    for (parent, _) in parents
        remove_dependency!(state.graph, parent, page_id)
    end
    for (child, _) in children
        remove_dependency!(state.graph, page_id, child)
    end
    removed = lock(state.lock) do
        page = get(state.pages, page_id, nothing)
        page === nothing && throw(PageNotFoundError(UInt64(page_id), "delete_page!"))
        delete!(state.pages, page_id)
        page
    end
    if is_gpu_page(removed)
        CUDA.unsafe_free!(removed.data)
        CUDA.unsafe_free!(removed.mask)
    end
    return nothing
end

"""
Migrate a page between CPU/GPU/Unified memory.
"""
function migrate_page!(state::MMSBState, page_id::PageID, 
                       target_location::PageLocation)::Page
    lock(state.lock) do
        page = get(state.pages, page_id, nothing)
        page === nothing && throw(PageNotFoundError(UInt64(page_id), "migrate_page!"))
        page.location == target_location && return page
        if target_location == CPU_LOCATION
            page.data = Vector{UInt8}(page.data)
            page.mask = Vector{Bool}(page.mask)
        elseif target_location in (GPU_LOCATION, UNIFIED_LOCATION)
            CUDA.functional() || throw(GPUMemoryError("CUDA unavailable for GPU migration"))
            page.data = CuArray{UInt8}(page.data)
            page.mask = CuArray{Bool}(page.mask)
        else
            throw(UnsupportedLocationError(string(target_location), "migrate_page!"))
        end
        page.location = target_location
        return page
    end
end

"""
Resize a page while preserving contents.
"""
function resize_page!(state::MMSBState, page_id::PageID, new_size::Int64)::Page
    lock(state.lock) do
        page = get(state.pages, page_id, nothing)
        page === nothing && throw(PageNotFoundError(UInt64(page_id), "resize_page!"))
        old_data = Vector{UInt8}(page.data)
        old_mask = Vector{Bool}(page.mask)
        new_data = zeros(UInt8, new_size)
        new_mask = falses(new_size)
        copylen = min(page.size, new_size)
        new_data[1:copylen] .= old_data[1:copylen]
        new_mask[1:copylen] .= old_mask[1:copylen]
        page.size = new_size
        page.data = is_gpu_page(page) ? CuArray(new_data) : new_data
        page.mask = is_gpu_page(page) ? CuArray(new_mask) : new_mask
        return page
    end
end

"""
Allocate mask/data arrays appropriate for the location.
"""
function allocate_page_arrays(size::Int64, location::PageLocation)
    page = Page(0, size, location)
    return (page.mask, page.data)
end

"""
Create a deep copy of an existing page with a new ID.
"""
function clone_page(state::MMSBState, page_id::PageID)::Page
    lock(state.lock) do
        source = get(state.pages, page_id, nothing)
        source === nothing && throw(PageNotFoundError(UInt64(page_id), "clone_page"))
        new_id = _reserve_page_id_unlocked!(state)
        clone = Page(new_id, source.size, source.location)
        clone.epoch = source.epoch
        clone.mask .= source.mask
        clone.data .= source.data
        clone.metadata = Dict{Symbol,Any}(source.metadata)
        state.pages[new_id] = clone
        return clone
    end
end

end # module PageAllocator
