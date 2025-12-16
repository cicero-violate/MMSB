# Structure Group: src/API.jl

## File: MMSB/src/API.jl

- Layer(s): root
- Language coverage: Julia (8)
- Element types: Function (7), Module (1)
- Total elements: 8

### Elements

- [Julia | Module] `API` (line 10, pub)
- [Julia | Function] `mmsb_start` (line 30, pub)
  - Signature: `mmsb_start(; enable_gpu::Bool`
- [Julia | Function] `mmsb_stop` (line 44, pub)
  - Signature: `mmsb_stop(state::MMSBState; checkpoint_path::Union{Nothing,String}`
- [Julia | Function] `_resolve_location` (line 52, pub)
  - Signature: `_resolve_location(location::Symbol)`
  - Calls: CUDA.functional, GPUMemoryError, UnsupportedLocationError, string, throw
- [Julia | Function] `create_page` (line 68, pub)
  - Signature: `create_page(state::MMSBState; size::Integer, location::Symbol`
- [Julia | Function] `update_page` (line 88, pub)
  - Signature: `update_page(state::MMSBState, page_id::PageID, bytes::AbstractVector{UInt8}; source::Symbol`
- [Julia | Function] `length` (line 92, pub)
  - Signature: `length(bytes)`
  - Calls: InvalidDeltaError, UInt64, throw
- [Julia | Function] `query_page` (line 113, pub)
  - Signature: `query_page(state::MMSBState, page_id::PageID)::Vector{UInt8}`
  - Calls: PageNotFoundError, UInt64, get_page, read_page, throw

