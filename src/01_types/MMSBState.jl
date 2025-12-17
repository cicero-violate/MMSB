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
       get_page, register_page!, reset!
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
    next_delta_id::Threads.Atomic{UInt64}
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
            Threads.Atomic{UInt64}(1),
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
    # T2.3: Lock-free atomic allocation
    id = Threads.atomic_add!(state.next_delta_id, UInt64(1))
    return DeltaID(id)
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

"""
    reset!(state::MMSBState)

Reset state to initial conditions, clearing all mutable components.

Clears in order:
1. pages dictionary
2. shadow page graph
3. next_page_id counter
4. next_delta_id atomic counter
5. transaction log in-memory entries
6. propagation buffers (via StateManagement)

**Thread Safety**: Caller must ensure no concurrent operations.
**Performance**: Target < 1μs for reset operation.
"""
function reset!(state::MMSBState)
    lock(state.lock) do
        # 1. Clear pages
        empty!(state.pages)
        
        # 2. Reset graph
        state.graph = ShadowPageGraph()
        
        # 3. Reset page ID counter
        state.next_page_id[] = PageID(1)
        
        # 4. Reset delta ID counter (atomic)
        state.next_delta_id[] = UInt64(1)
        
        # 5. Clear allocator (frees Rust-side memory)
        FFIWrapper.rust_allocator_clear!(state.allocator_handle)
        
        # 6. Clear TLog in-memory entries (file untouched)
        FFIWrapper.rust_tlog_clear_entries!(state.tlog_handle)
    end
    return nothing
end

end # module MMSBStateTypes
