# Structure Group: src/01_types

## File: MMSB/src/01_types/Errors.jl

- Layer(s): 01_types
- Language coverage: Julia (8)
- Element types: Function (1), Module (1), Struct (6)
- Total elements: 8

### Elements

- [Julia | Module] `ErrorTypes` (line 8, pub)
- [Julia | Struct] `PageNotFoundError` (line 15, pub)
  - Signature: `struct PageNotFoundError <: MMSBError`
- [Julia | Struct] `InvalidDeltaError` (line 20, pub)
  - Signature: `struct InvalidDeltaError <: MMSBError`
- [Julia | Struct] `GPUMemoryError` (line 25, pub)
  - Signature: `struct GPUMemoryError <: MMSBError`
- [Julia | Struct] `SerializationError` (line 29, pub)
  - Signature: `struct SerializationError <: MMSBError`
- [Julia | Struct] `GraphCycleError` (line 33, pub)
  - Signature: `struct GraphCycleError <: MMSBError`
- [Julia | Struct] `UnsupportedLocationError` (line 38, pub)
  - Signature: `struct UnsupportedLocationError <: MMSBError`
- [Julia | Function] `Base.showerror` (line 46, pub)
  - Signature: `Base.showerror(io::IO, err::InvalidDeltaError)`

## File: MMSB/src/01_types/MMSBState.jl

- Layer(s): 01_types
- Language coverage: Julia (10)
- Element types: Function (7), Module (1), Struct (2)
- Total elements: 10

### Elements

- [Julia | Module] `MMSBStateTypes` (line 8, pub)
- [Julia | Struct] `MMSBConfig` (line 25, pub)
  - Signature: `mutable struct MMSBConfig`
- [Julia | Function] `MMSBConfig` (line 34, pub)
  - Signature: `MMSBConfig(; enable_logging`
  - Calls: joinpath, pwd
- [Julia | Struct] `MMSBState` (line 51, pub)
  - Signature: `mutable struct MMSBState`
- [Julia | Function] `MMSBState` (line 61, pub)
  - Signature: `MMSBState(config::MMSBConfig)`
  - Calls: DeltaID, FFIWrapper.rust_allocator_free!, FFIWrapper.rust_allocator_new, FFIWrapper.rust_tlog_free!, FFIWrapper.rust_tlog_new, PageID, ReentrantLock, ShadowPageGraph, error, finalizer, new
- [Julia | Function] `MMSBState` (line 83, pub)
  - Signature: `MMSBState()`
  - Calls: MMSBConfig, MMSBState
- [Julia | Function] `allocate_page_id!` (line 91, pub)
  - Signature: `allocate_page_id!(state::MMSBState)::PageID`
  - Calls: _reserve_page_id_unlocked!, lock
- [Julia | Function] `allocate_delta_id!` (line 111, pub)
  - Signature: `allocate_delta_id!(state::MMSBState)::DeltaID`
  - Calls: DeltaID, lock
- [Julia | Function] `get_page` (line 124, pub)
  - Signature: `get_page(state::MMSBState, id::PageID)::Union{Page, Nothing}`
  - Calls: get, lock
- [Julia | Function] `register_page!` (line 135, pub)
  - Signature: `register_page!(state::MMSBState, page::Page)`
  - Calls: lock

