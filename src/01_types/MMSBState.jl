# src/types/MMSBState.jl
"""
MMSBState - Global state management structure

Central registry for all pages, transaction log, and dependency graph.
This is the "brain" of the MMSB system.
"""
module MMSBStateTypes

using Base: C_NULL
using ..PageTypes: Page, PageID
using ..DeltaTypes: Delta, DeltaID
using ..GraphTypes: ShadowPageGraph
using ..FFIWrapper

export MMSBState, MMSBConfig, allocate_page_id!, allocate_delta_id!,
       get_page, register_page!
export _reserve_page_id_unlocked!  # internal helper for allocator

"""
    MMSBConfig

Configuration parameters for MMSB runtime.
"""
mutable struct MMSBConfig
    enable_logging::Bool
    enable_gpu::Bool
    enable_instrumentation::Bool
    page_size_default::Int64
    max_tlog_size::Int64
    checkpoint_interval::Int64
    tlog_path::String
    
    function MMSBConfig(; enable_logging=true,
                        enable_gpu=true,
                        enable_instrumentation=false,
                        page_size_default::Int64=4096,
                        max_tlog_size::Int64=10_000,
                        checkpoint_interval::Int64=60,
                        tlog_path::String=joinpath(pwd(), "mmsb.tlog"))
        return new(enable_logging, enable_gpu, enable_instrumentation,
                   page_size_default, max_tlog_size, checkpoint_interval, tlog_path)
    end
end

"""
    MMSBState

Global MMSB state containing all pages, logs, and metadata.
"""
mutable struct MMSBState
    pages::Dict{PageID, Page}
    tlog_handle::FFIWrapper.RustTLogHandle
    allocator_handle::FFIWrapper.RustAllocatorHandle
    graph::ShadowPageGraph
    next_page_id::Ref{PageID}
    next_delta_id::Ref{DeltaID}
    config::MMSBConfig
    lock::ReentrantLock
    
    function MMSBState(config::MMSBConfig)
        handle = FFIWrapper.rust_tlog_new(config.tlog_path)
        handle.ptr == C_NULL && error("Failed to initialize Rust TLog at $(config.tlog_path)")
        allocator = FFIWrapper.rust_allocator_new()
        allocator.ptr == C_NULL && error("Failed to initialize Rust allocator")
        state = new(
            Dict{PageID, Page}(),
            handle,
            allocator,
            ShadowPageGraph(),
            Ref{PageID}(PageID(1)),
            Ref{DeltaID}(DeltaID(1)),
            config,
            ReentrantLock(),
        )
        finalizer(state) do st
            st.tlog_handle.ptr != C_NULL && FFIWrapper.rust_tlog_free!(st.tlog_handle)
            st.allocator_handle.ptr != C_NULL && FFIWrapper.rust_allocator_free!(st.allocator_handle)
        end
        return state
    end
    
    MMSBState() = MMSBState(MMSBConfig())
end

"""
    allocate_page_id!(state::MMSBState) -> PageID

Thread-safe page ID allocation.
"""
function allocate_page_id!(state::MMSBState)::PageID
    lock(state.lock) do
        return _reserve_page_id_unlocked!(state)
    end
end

"""
Internal helper that assumes `state.lock` is already held.
"""
@inline function _reserve_page_id_unlocked!(state::MMSBState)::PageID
    id = state.next_page_id[]
    state.next_page_id[] = PageID(id + 1)
    return id
end

"""
    allocate_delta_id!(state::MMSBState) -> DeltaID

Thread-safe delta ID allocation.
"""
function allocate_delta_id!(state::MMSBState)::DeltaID
    lock(state.lock) do
        id = state.next_delta_id[]
        state.next_delta_id[] = DeltaID(id + 1)
        return id
    end
end

"""
    get_page(state::MMSBState, id::PageID) -> Union{Page, Nothing}

Retrieve page by ID, returns nothing if not found.
"""
function get_page(state::MMSBState, id::PageID)::Union{Page, Nothing}
    lock(state.lock) do
        return get(state.pages, id, nothing)
    end
end

"""
    register_page!(state::MMSBState, page::Page)

Add page to registry.
"""
function register_page!(state::MMSBState, page::Page)
    lock(state.lock) do
        state.pages[page.id] = page
    end
end

end # module MMSBStateTypes
