# src/02_runtime/PageAllocator.jl
"""
PageAllocator - Page creation and lifecycle management

Handles allocation, initialization, and deallocation of pages.
Manages page ID assignment and memory allocation on CPU/GPU.
"""
module PageAllocator

using ..PageTypes: Page, PageID, PageLocation, CPU_LOCATION, GPU_LOCATION,
    UNIFIED_LOCATION, initialize!, activate!, deactivate!
using ..MMSBStateTypes: MMSBState, _reserve_page_id_unlocked!
using ..GraphTypes: get_children, get_parents, remove_dependency!
using ..ErrorTypes: PageNotFoundError
using ..FFIWrapper

export create_page!, delete_page!, migrate_page!, resize_page!, clone_page, allocator_handle

const _ALLOCATOR = Ref{FFIWrapper.RustAllocatorHandle}(FFIWrapper.RustAllocatorHandle(C_NULL))

function _allocator_handle()
    handle = _ALLOCATOR[]
    if handle.ptr == C_NULL
        handle = FFIWrapper.rust_allocator_new(Int32(CPU_LOCATION))
        _ALLOCATOR[] = handle
    end
    handle
end

allocator_handle() = _allocator_handle()

"""
Create and register a page with the requested location.
"""
function create_page!(state::MMSBState, size::Int64, location::PageLocation;
                      metadata::Dict{Symbol,Any}=Dict{Symbol,Any}())::Page
    page = lock(state.lock) do
        id = _reserve_page_id_unlocked!(state)
        # FFI call only passes scalar values and returns a handle; no Julia buffers
        # are exposed to Rust by pointer here, so no GC.@preserve is required.
        handle = FFIWrapper.rust_allocator_allocate(_allocator_handle(), UInt64(id), size, Int32(location))
        Page(handle, id, location, size; metadata=Dict{Symbol,Any}(metadata))
    end
    lock(state.lock) do
        state.pages[page.id] = page
    end
    # T6: Initialize page state after allocation
    initialize!(page)
    activate!(page)  # Pages are immediately usable after creation
    return page
end

create_cpu_page!(state::MMSBState, size::Int64)::Page =
    create_page!(state, size, CPU_LOCATION)

create_gpu_page!(state::MMSBState, size::Int64)::Page =
    create_page!(state, size, GPU_LOCATION)

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
    lock(state.lock) do
        page = get(state.pages, page_id, nothing)
        page === nothing && throw(PageNotFoundError(UInt64(page_id), "delete_page!"))
        delete!(state.pages, page_id)
    end
    # Release only passes the allocator handle and page id (scalar); no Julia
    # object memory is passed by pointer, so GC.@preserve is not needed here.
    FFIWrapper.rust_allocator_release!(_allocator_handle(), UInt64(page_id))
    return nothing
end

"""
Migrate a page between CPU/GPU/Unified memory.
"""
function migrate_page!(state::MMSBState, page_id::PageID, 
                       target_location::PageLocation)::Page
    page = lock(state.lock) do
        existing = get(state.pages, page_id, nothing)
        existing === nothing && throw(PageNotFoundError(UInt64(page_id), "migrate_page!"))
        existing
    end
    page.location = target_location
    page
end

"""
Resize a page while preserving contents.
"""
function resize_page!(state::MMSBState, page_id::PageID, new_size::Int64)::Page
    page = lock(state.lock) do
        existing = get(state.pages, page_id, nothing)
        existing === nothing && throw(PageNotFoundError(UInt64(page_id), "resize_page!"))
        existing
    end
    page.size = new_size
    page
end

"""
Allocate mask/data arrays appropriate for the location.
"""
function allocate_page_arrays(size::Int64, location::PageLocation)
    error("allocate_page_arrays is not available in the Rust-backed runtime")
end

"""
Create a deep copy of an existing page with a new ID.
"""
function clone_page(state::MMSBState, page_id::PageID)::Page
    error("clone_page is not available in the Rust-backed runtime")
end

end # module PageAllocator
