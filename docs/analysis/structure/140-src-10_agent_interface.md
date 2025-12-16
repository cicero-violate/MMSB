# Structure Group: src/10_agent_interface

## File: MMSB/src/10_agent_interface/AgentProtocol.jl

- Layer(s): 10_agent_interface
- Language coverage: Julia (3)
- Element types: Function (1), Module (1), Struct (1)
- Total elements: 3

### Elements

- [Julia | Module] `AgentProtocol` (line 4, pub)
- [Julia | Struct] `AgentAction` (line 14, pub)
  - Signature: `struct AgentAction`
- [Julia | Function] `observe` (line 25, pub)
  - Signature: `observe end`
  - Calls: act!, plan

## File: MMSB/src/10_agent_interface/BaseHook.jl

- Layer(s): 10_agent_interface
- Language coverage: Julia (6)
- Element types: Function (5), Module (1)
- Total elements: 6

### Elements

- [Julia | Module] `BaseHooks` (line 8, pub)
- [Julia | Function] `enable_base_hooks!` (line 37, pub)
  - Signature: `enable_base_hooks!(state::MMSBState)`
  - Calls: MMSB.hook_invoke, invoke
- [Julia | Function] `disable_base_hooks!` (line 74, pub)
  - Signature: `disable_base_hooks!()`
  - Calls: empty!
- [Julia | Function] `hook_invoke` (line 107, pub)
  - Signature: `hook_invoke(f, types::Type, args, kwargs)`
  - Calls: time_ns
- [Julia | Function] `hook_setfield!` (line 135, pub)
  - Signature: `hook_setfield!(obj, field::Symbol, value)`
  - Calls: objectid, time_ns, typeof
- [Julia | Function] `hook_getfield` (line 160, pub)
  - Signature: `hook_getfield(obj, field::Symbol)`
  - Calls: objectid, time_ns, typeof

## File: MMSB/src/10_agent_interface/CompilerHooks.jl

- Layer(s): 10_agent_interface
- Language coverage: Julia (16)
- Element types: Function (14), Module (1), Struct (1)
- Total elements: 16

### Elements

- [Julia | Module] `CompilerHooks` (line 11, pub)
- [Julia | Struct] `MMSBInterpreter` (line 37, pub)
  - Signature: `mutable struct MMSBInterpreter <: AbstractInterpreter`
- [Julia | Function] `MMSBInterpreter` (line 42, pub)
  - Signature: `MMSBInterpreter(state::MMSBState; world::UInt`
  - Calls: Base.get_world_counter
- [Julia | Function] `Core.Compiler.InferenceParams` (line 52, pub)
  - Signature: `Core.Compiler.InferenceParams(interp::MMSBInterpreter)`
- [Julia | Function] `Core.Compiler.OptimizationParams` (line 56, pub)
  - Signature: `Core.Compiler.OptimizationParams(interp::MMSBInterpreter)`
- [Julia | Function] `Core.Compiler.get_world_counter` (line 60, pub)
  - Signature: `Core.Compiler.get_world_counter(interp::MMSBInterpreter)`
- [Julia | Function] `Core.Compiler.get_inference_cache` (line 64, pub)
  - Signature: `Core.Compiler.get_inference_cache(interp::MMSBInterpreter)`
- [Julia | Function] `Core.Compiler.code_cache` (line 68, pub)
  - Signature: `Core.Compiler.code_cache(interp::MMSBInterpreter)`
- [Julia | Function] `Core.Compiler.typeinf` (line 84, pub)
  - Signature: `Core.Compiler.typeinf(interp::MMSBInterpreter, frame::InferenceState)`
  - Calls: create_inference_pages!, invoke, log_inference_result!, log_inference_start!
- [Julia | Function] `Core.Compiler.abstract_call_method` (line 117, pub)
  - Signature: `Core.Compiler.abstract_call_method( interp::MMSBInterpreter, method::Method, sig, sparams::Core.SimpleVector, hardlim...`
  - Calls: invoke, log_method_call!, update_call_graph!
- [Julia | Function] `Core.Compiler.optimize` (line 159, pub)
  - Signature: `Core.Compiler.optimize( interp::MMSBInterpreter, opt::Core.Compiler.OptimizationState, params::OptimizationParams, re...`
  - Calls: copy, create_optimization_delta!, invoke
- [Julia | Function] `log_inference_start!` (line 194, pub)
  - Signature: `log_inference_start!(state::MMSBState, frame::InferenceState)`
  - Calls: time_ns
- [Julia | Function] `log_inference_result!` (line 212, pub)
  - Signature: `log_inference_result!(state::MMSBState, frame::InferenceState, result)`
  - Calls: time_ns
- [Julia | Function] `create_inference_pages!` (line 235, pub)
  - Signature: `create_inference_pages!(state::MMSBState, frame::InferenceState)`
  - Calls: copyto!, create_page!, length, serialize_codeinfo
- [Julia | Function] `enable_compiler_hooks!` (line 259, pub)
  - Signature: `enable_compiler_hooks!(state::MMSBState)`
- [Julia | Function] `disable_compiler_hooks!` (line 273, pub)
  - Signature: `disable_compiler_hooks!()`

## File: MMSB/src/10_agent_interface/CoreHooks.jl

- Layer(s): 10_agent_interface
- Language coverage: Julia (5)
- Element types: Function (4), Module (1)
- Total elements: 5

### Elements

- [Julia | Module] `CoreHooks` (line 8, pub)
- [Julia | Function] `enable_core_hooks!` (line 32, pub)
  - Signature: `enable_core_hooks!(state::MMSBState)`
  - Calls: enabled
- [Julia | Function] `disable_core_hooks!` (line 49, pub)
  - Signature: `disable_core_hooks!()`
- [Julia | Function] `hook_codeinfo_creation` (line 72, pub)
  - Signature: `hook_codeinfo_creation(mi::Core.MethodInstance, ci::Core.CodeInfo)`
  - Calls: create_ir_page!, time_ns
- [Julia | Function] `hook_methodinstance` (line 103, pub)
  - Signature: `hook_methodinstance(mi::Core.MethodInstance)`
  - Calls: time_ns

## File: MMSB/src/10_agent_interface/InstrumentationManager.jl

- Layer(s): 10_agent_interface
- Language coverage: Julia (6)
- Element types: Function (4), Module (1), Struct (1)
- Total elements: 6

### Elements

- [Julia | Module] `InstrumentationManager` (line 8, pub)
- [Julia | Struct] `InstrumentationConfig` (line 23, pub)
  - Signature: `mutable struct InstrumentationConfig`
- [Julia | Function] `InstrumentationConfig` (line 33, pub)
  - Signature: `InstrumentationConfig()`
- [Julia | Function] `enable_instrumentation!` (line 51, pub)
  - Signature: `enable_instrumentation!(state::MMSBState, config::InstrumentationConfig)`
  - Calls: BaseHooks.enable_base_hooks!, CompilerHooks.enable_compiler_hooks!, CoreHooks.enable_core_hooks!
- [Julia | Function] `disable_instrumentation!` (line 77, pub)
  - Signature: `disable_instrumentation!(state::MMSBState)`
  - Calls: BaseHooks.disable_base_hooks!, CompilerHooks.disable_compiler_hooks!, CoreHooks.disable_core_hooks!
- [Julia | Function] `configure_instrumentation!` (line 100, pub)
  - Signature: `configure_instrumentation!(state::MMSBState, config::InstrumentationConfig)`
  - Calls: disable_instrumentation!, enable_instrumentation!

## File: MMSB/src/10_agent_interface/checkpoint_api.jl

- Layer(s): 10_agent_interface
- Language coverage: Julia (4)
- Element types: Function (3), Module (1)
- Total elements: 4

### Elements

- [Julia | Module] `CheckpointAPI` (line 4, pub)
- [Julia | Function] `create_checkpoint` (line 11, pub)
  - Signature: `create_checkpoint(state::MMSBState, name::String)::String`
  - Calls: checkpoint_log!, time_ns
- [Julia | Function] `restore_checkpoint` (line 17, pub)
  - Signature: `restore_checkpoint(state::MMSBState, path::String)`
  - Calls: load_checkpoint!
- [Julia | Function] `list_checkpoints` (line 21, pub)
  - Signature: `list_checkpoints(state::MMSBState)::Vector{String}`

## File: MMSB/src/10_agent_interface/event_subscription.jl

- Layer(s): 10_agent_interface
- Language coverage: Julia (5)
- Element types: Function (3), Module (1), Struct (1)
- Total elements: 5

### Elements

- [Julia | Module] `EventSubscription` (line 4, pub)
- [Julia | Struct] `Subscription` (line 19, pub)
  - Signature: `mutable struct Subscription`
- [Julia | Function] `subscribe_to_events` (line 29, pub)
  - Signature: `subscribe_to_events(types::Vector{EventType}, callback::Function)::UInt64`
  - Calls: Set, Subscription
- [Julia | Function] `unsubscribe` (line 36, pub)
  - Signature: `unsubscribe(id::UInt64)`
  - Calls: haskey
- [Julia | Function] `emit_event` (line 40, pub)
  - Signature: `emit_event(event_type::EventType, data::Any)`
  - Calls: sub.callback, values

