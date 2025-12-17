 Count  Overhead File                                            Line Function
 =====  ======== ====                                            ==== ========
     1         1 @MMSB/src/ffi/FFIWrapper.jl                      14 _check_rust_error(context::String)
     1         1 @MMSB/src/ffi/FFIWrapper.jl                     327 rust_delta_source(handle::Main.MMSBBenchmarks.MMSB.FFIWrapper.RustDeltaHandle)
     1         1 @MMSB/src/ffi/FFIWrapper.jl                     436 rust_get_last_error
     1         0 @MMSB/src/ffi/FFIWrapper.jl                     331 rust_delta_source(handle::Main.MMSBBenchmarks.MMSB.FFIWrapper.RustDeltaHandle)
     1         0 @Base/array.jl                                  935 getindex
     1         0 @Base/abstractarray.jl                          699 checkbounds
     1         0 @Base/abstractarray.jl                          689 checkbounds
     1         0 @Base/abstractarray.jl                          757 checkindex
     1         0 @Base/range.jl                                  688 isempty
     1         1 @MMSB/src/ffi/FFIWrapper.jl                     318 rust_delta_source(handle::Main.MMSBBenchmarks.MMSB.FFIWrapper.RustDeltaHandle)
     1         0 @MMSB/src/01_page/Delta.jl                       31 (::Main.MMSBBenchmarks.MMSB.DeltaTypes.var"#5#6")(d::Main.MMSBBenchmarks.MMSB.DeltaTypes.Delta)
     1         0 @MMSB/src/ffi/FFIWrapper.jl                     154 rust_delta_free!
     1         0 @Base/stat.jl                                   457 isfile
     1         1 @Base/cmem.jl                                     ? rust_delta_mask(handle::Main.MMSBBenchmarks.MMSB.FFIWrapper.RustDeltaHandle)
     1         0 @MMSB/src/01_page/ReplayEngine.jl                59 replay_to_epoch(state::Main.MMSBBenchmarks.MMSB.MMSBStateTypes.MMSBState, target_epoch::UInt32)
     1         0 @MMSB/src/01_page/ReplayEngine.jl                23 _blank_state_like
     1         1 @MMSB/src/ffi/FFIWrapper.jl                     228 rust_tlog_new(path::String)
     1         1 @MMSB/src/03_dag/EventSystem.jl                  78 emit_event!(::Main.MMSBBenchmarks.MMSB.MMSBStateTypes.MMSBState, ::Main.MMSBBenchmarks.MMSB.EventSystem.EventType, ::UInt64, ::Vararg{Any})
     1         1 @Base/multimedia.jl                              49 MIME
     1         1 @Base/operators.jl                              321 !=(x::UInt8, y::UInt8)
     1         1 @Base/array.jl                                    ? recompute_page!(state::Main.MMSBBenchmarks.MMSB.MMSBStateTypes.MMSBState, page_id::UInt64)
     1         0 @Base/Enums.jl                                   61 show(io::IOContext{IOBuffer}, ::MIME{Symbol("text/plain")}, x::Main.MMSBBenchmarks.MMSB.EventSystem.EventType)
     1         0 @Base/show.jl                                  1285 show
     1         0 @Base/intfuncs.jl                               990 string
     1         0 @Base/intfuncs.jl                              1000 string(n::Int32; base::Int64, pad::Int64)
     1         0 @Base/intfuncs.jl                               917 dec(x::UInt32, pad::Int64, neg::Bool)
     1         0 @Base/intfuncs.jl                               809 ndigits
     1         0 @Base/intfuncs.jl                               809 #ndigits#402
     1         0 @Base/intfuncs.jl                               770 ndigits0z
     1         0 @Base/intfuncs.jl                               712 ndigits0zpb(x::UInt32, b::Int64)
     1         0 @Base/intfuncs.jl                               666 bit_ndigits0z
     1         0 @MMSB/src/02_semiring/DeltaRouter.jl             59 create_delta(state::Main.MMSBBenchmarks.MMSB.MMSBStateTypes.MMSBState, page_id::UInt64, mask::Vector{Bool}, data::Vector{UInt8}; source::Symbol, i…
     1         0 @MMSB/src/01_types/MMSBState.jl                 112 allocate_delta_id!(state::Main.MMSBBenchmarks.MMSB.MMSBStateTypes.MMSBState)
     1         1 @MMSB/src/01_types/MMSBState.jl                 114 (::Main.MMSBBenchmarks.MMSB.MMSBStateTypes.var"#allocate_delta_id!##0#allocate_delta_id!##1"{Main.MMSBBenchmarks.MMSB.MMSBStateTypes.MMSBState})()
     1         0 @MMSB/src/01_page/Delta.jl                       30 Main.MMSBBenchmarks.MMSB.DeltaTypes.Delta(id::UInt64, page_id::UInt64, epoch::UInt32, mask::Vector{UInt8}, payload::Vector{UInt8}, source::Symbol;…
     1         1 @Base/gcutils.jl                                 86 finalizer
     1         1 @Base/essentials.jl                             917 getindex(A::Vector{UInt8}, i::Int64)
     1         0 @MMSB/src/01_page/Delta.jl                       27 Main.MMSBBenchmarks.MMSB.DeltaTypes.Delta(id::UInt64, page_id::UInt64, epoch::UInt32, mask::Vector{UInt8}, payload::Vector{UInt8}, source::Symbol;…
     1         0 @MMSB/src/ffi/FFIWrapper.jl                     140 rust_delta_new
     1         0 @MMSB/src/ffi/FFIWrapper.jl                     149 rust_delta_new(delta_id::UInt64, page_id::UInt64, epoch::UInt32, mask::Vector{UInt8}, payload::Vector{UInt8}, source::Symbol; is_sparse::Bool)
     1         1 @Base/util.jl                                    73 with_output_color(f::Function, color::Symbol, io::IOContext{IOBuffer}, args::String; bold::Bool, italic::Bool, underline::Bool, blink::Bool, rever…
     1         0 @Base/logging/ConsoleLogger.jl                   74 default_metafmt(level::Base.CoreLogging.LogLevel, _module::Any, group::Any, id::Any, file::Any, line::Any)
     1         0 @Base/strings/io.jl                             193 string
     1         0 @Base/strings/io.jl                             151 print_to_string(xs::Base.CoreLogging.LogLevel)
     1         0 @Base/strings/io.jl                              35 print(io::IOBuffer, x::Base.CoreLogging.LogLevel)
     1         0 @Base/logging/logging.jl                        181 show(io::IOBuffer, level::Base.CoreLogging.LogLevel)
     1         0 @Base/strings/io.jl                             248 print
     1         0 @Base/strings/io.jl                             246 write
     1         0 @Base/iobuffer.jl                               831 unsafe_write(to::IOBuffer, p::Ptr{UInt8}, nb::UInt64)
     1         0 @Base/iobuffer.jl                               857 _unsafe_write
     1         0 @Base/genericmemory.jl                          260 setindex!
     1         1 @Base/genericmemory.jl                          253 _setindex!
     1         0 @MMSB/src/ffi/FFIWrapper.jl                     243 rust_tlog_append!
     1         1 @Base/strings/string.jl                          73 String
     1         0 @MMSB/src/04_propagation/PropagationEngine.jl   210 propagate_change!(state::Main.MMSBBenchmarks.MMSB.MMSBStateTypes.MMSBState, changed_pages::Vector{UInt64}, mode::Main.MMSBBenchmarks.MMSB.Propagat…
     1         0 @MMSB/src/06_utility/Monitoring.jl               46 track_propagation_latency!
     1         0 @Base/dict.jl                                   503 get
     1         1 @Base/dict.jl                                   237 ht_keyindex(h::Dict{Main.MMSBBenchmarks.MMSB.MMSBStateTypes.MMSBState, Tuple{Int64, UInt64}}, key::Main.MMSBBenchmarks.MMSB.MMSBStateTypes.MMSBSta…
     1         1 @Base/boot.jl                                  1031 Pair(a::Symbol, b::Type)
     1         0 @Base/show.jl                                   335 IOContext
     1         1 @Base/pair.jl                                    51 getindex
     1         0 @Base/logging/ConsoleLogger.jl                  139 handle_message(logger::Base.CoreLogging.ConsoleLogger, level::Base.CoreLogging.LogLevel, message::Any, _module::Any, group::Any, id::Any, filepath…
     1         0 @Base/stream.jl                                 577 displaysize_
     1         1 @Base/stream.jl                                 603 displaysize(io::Base.TTY)
     1         0 @Base/util.jl                                   106 with_output_color(::Function, ::Symbol, ::IOContext{IOBuffer}, ::String, ::Vararg{String}; bold::Bool, italic::Bool, underline::Bool, blink::Bool,…
     1         0 @Base/strings/util.jl                           731 iterate
     1         0 @Base/strings/util.jl                           732 iterate
     1         0 @Base/stream.jl                                1081 uv_write(s::Base.TTY, p::Ptr{UInt8}, n::UInt64)
     1         0 @Base/task.jl                                  1199 wait()
     1         0 @Base/task.jl                                  1187 poptask(W::Base.IntrusiveLinkedListSynchronized{Task})
     1         0 @Base/stream.jl                                1200 uv_writecb_task(req::Ptr{Nothing}, status::Int32)
     1         0 @Base/task.jl                                  1027 schedule
     1         0 @Base/task.jl                                  1040 schedule(t::Task, arg::Any; error::Bool)
     1         1 @Base/task.jl                                   976 enq_work(t::Task)
     1         0 @Base/Enums.jl                                   59 show(io::IOContext{IOBuffer}, ::MIME{Symbol("text/plain")}, x::Main.MMSBBenchmarks.MMSB.EventSystem.EventType)
     1         0 @Base/show.jl                                   968 show(io::IOContext{IOBuffer}, x::Type)
     1         0 @Base/show.jl                                   971 _show_type(io::IOContext{IOBuffer}, x::Type)
     1         0 @Base/show.jl                                  1078 show_type_name(io::IOContext{IOBuffer}, tn::Core.TypeName)
     1         0 @Base/show.jl                                  1044 is_global_function(tn::Core.TypeName, globname::Symbol)
     1         1 @Base/runtime_internals.jl                      418 isconst
     1         0 @Base/reflection.jl                            1276 #invokelatest_gr#232
     1         1 @Compiler/src/ssair/slot2ssa.jl                 101 fixemup!(slot_filter::Compiler.var"#rename_uses!##0#rename_uses!##1", rename_slot::Compiler.var"#rename_uses!##2#rename_uses!##3"{Vector{Pair{Any,…
     1         1 @Compiler/src/optimize.jl                       492 finish(interp::Compiler.NativeInterpreter, opt::Compiler.OptimizationState{Compiler.NativeInterpreter}, ir::Compiler.IRCode, caller::Compiler.Infe…
     1         0 @Compiler/src/tfuncs.jl                        2572 getfield_effects(𝕃::Compiler.InferenceLattice{Compiler.ConditionalsLattice{Compiler.PartialsLattice{Compiler.ConstsLattice}}}, argtypes::Vector{An…
     1         1 @Compiler/src/typeutils.jl                      360 is_immutable_argtype(argtype::Any)
     1         0 @Compiler/src/ssair/passes.jl                  2219 adce_pass!(ir::Compiler.IRCode, inlining::Compiler.InliningState{Compiler.NativeInterpreter})
     1         0 @Compiler/src/ssair/passes.jl                  2036 mark_phi_cycles!(compact::Compiler.IncrementalCompact, safe_phis::IdSet{Int64}, phi::Int64)
     1         1 @Base/idset.jl                                   60 push!(s::IdSet{Int64}, x::Any)
     1         0 @Compiler/src/ssair/passes.jl                   305 walk_to_defs(compact::Compiler.IncrementalCompact, defssa::Any, typeconstraint::Any, predecessors::typeof(Compiler.phi_or_ifelse_predecessors), 𝕃ₒ…
     1         0 @Compiler/src/ssair/ir.jl                       738 CFGTransformState!(blocks::Vector{Compiler.BasicBlock}, allow_cfg_transforms::Bool)
     1         0 @Compiler/src/ssair/slot2ssa.jl                 738 construct_ssa!(ci::Core.CodeInfo, ir::Compiler.IRCode, sv::Compiler.OptimizationState{Compiler.NativeInterpreter}, domtree::Compiler.GenericDomTre…
     1         0 @Compiler/src/ssair/ir.jl                       297 resize!(stmts::Compiler.InstructionStream, len::Int64)
     1         1 @Base/array.jl                                 1127 _growend!
     1         0 @Base/strings/util.jl                           897 split
     1         0 @Base/strings/util.jl                           899 #split#428
     1         0 @Base/array.jl                                  728 collect
     1         0 @Base/array.jl                                  738 _collect(cont::UnitRange{Int64}, itr::Base.SplitIterator{String, Base.Fix2{typeof(isequal), Char}}, ::Base.HasEltype, isz::Base.SizeUnknown)
     1         0 @Base/array.jl                                  660 _similar_for
     1         0 @Base/abstractarray.jl                          824 similar
     1         0 @Base/abstractarray.jl                          832 similar
     1         0 @Compiler/src/typeinfer.jl                     1253 typeinf_ext(interp::Compiler.NativeInterpreter, mi::Core.MethodInstance, source_mode::UInt8)
     1         1 @Compiler/src/inferencestate.jl                 462 (::Compiler.ComputeTryCatch{Compiler.SimpleHandler})(code::Vector{Any}, bbs::Nothing)
     1         0 @Compiler/src/inferencestate.jl                 348 Compiler.InferenceState(result::Compiler.InferenceResult, src::Core.CodeInfo, cache_mode::UInt8, interp::Compiler.NativeInterpreter)
     1         1 @Compiler/src/ssair/ir.jl                      1798 process_newnode!(compact::Compiler.IncrementalCompact, new_idx::Int64, new_node_entry::Compiler.Instruction, new_node_info::Compiler.NewNodeInfo, …
     1         0 @Compiler/src/ssair/ir.jl                      1939 iterate_compact(compact::Compiler.IncrementalCompact)
     1         0 @Compiler/src/abstractinterpretation.jl         183 (::Compiler.var"#handle1#abstract_call_gf_by_type##1"{Int64, Compiler.Future{Compiler.MethodCallResult}, Int64, Vector{Union{Nothing, Core.CodeIns…
     1         1 @Compiler/src/typelattice.jl                      ? widenwrappedconditional
     1         0 @Compiler/src/ssair/inlining.jl                1406 handle_call!
     1         1 @Base/array.jl                                    ? handle_cases!(todo::Vector{Pair{Int64, Any}}, ir::Compiler.IRCode, idx::Int64, stmt::Expr, atype::Any, cases::Vector{Compiler.InliningCase}, handl…
     1         1 @Compiler/src/ssair/ir.jl                      1948 iterate_compact(compact::Compiler.IncrementalCompact)
     1         0 @Compiler/src/ssair/ir.jl                       102 compute_basic_blocks(stmts::Vector{Any})
     1         0 @Compiler/src/ssair/basicblock.jl                25 BasicBlock
     1         0 @Compiler/src/inferencestate.jl                 333 Compiler.InferenceState(result::Compiler.InferenceResult, src::Core.CodeInfo, cache_mode::UInt8, interp::Compiler.NativeInterpreter)
     1         0 @Compiler/src/ssair/ir.jl                       117 compute_basic_blocks(stmts::Vector{Any})
     1         0 @Compiler/src/ssair/ir.jl                       652 _advance(stmt::Any, op::Int64)
     1         0 @Base/abstractset.jl                            216 setdiff
     1         0 @Base/bitset.jl                                 303 setdiff!
     1         0 @Base/bitset.jl                                 181 _matched_map!
     1         1 @Base/bitset.jl                                 189 _matched_map!(f::Base.var"#setdiff!##0#setdiff!##1", a1::Vector{UInt64}, b1::Int64, a2::Vector{UInt64}, b2::Int64, left_false_is_false::Bool, righ…
     1         0 @Compiler/src/typeutils.jl                       54 argtypes_to_type(argtypes::Vector{Any})
     1         0 @Base/reducedim.jl                              992 all
     1         0 @Base/reducedim.jl                              992 #all#756
     1         0 @Base/anyall.jl                                 197 _all
     1         0 @Compiler/src/typeutils.jl                       54 #argtypes_to_type##4
     1         0 @Compiler/src/typeutils.jl                      116 valid_as_lattice(x::Any, astag::Bool)
     1         0 @Base/runtime_internals.jl                      789 isstructtype
     1         0 @Base/runtime_internals.jl                      804 isprimitivetype
     1         0 @Compiler/src/abstractinterpretation.jl        4199 typeinf_local(interp::Compiler.NativeInterpreter, frame::Compiler.InferenceState, nextresult::Compiler.CurrentState)
     1         0 @Compiler/src/ssair/passes.jl                   306 walk_to_defs(compact::Compiler.IncrementalCompact, defssa::Any, typeconstraint::Any, predecessors::typeof(Compiler.phi_or_ifelse_predecessors), 𝕃ₒ…
     1         1 @Compiler/src/ssair/ir.jl                      1456 process_node!(compact::Compiler.IncrementalCompact, result_idx::Int64, inst::Compiler.Instruction, idx::Int64, processed_idx::Int64, active_bb::In…
     1         0 @Compiler/src/typeinfer.jl                      676 record_slot_assign!(sv::Compiler.InferenceState)
     1         0 @Compiler/src/ssair/passes.jl                   412 lift_leaves(compact::Compiler.IncrementalCompact, field::Int64, leaves::Vector{Any}, 𝕃ₒ::Compiler.PartialsLattice{Compiler.ConstsLattice})
     1         0 @Compiler/src/abstractinterpretation.jl        3676 scan_specified_partitions(query::typeof(Compiler.abstract_eval_partition_load), walk_binding_partition::typeof(Compiler.walk_binding_partition), i…
     1         1 @Base/runtime_internals.jl                      231 lookup_binding_partition
     1         0 @Compiler/src/ssair/passes.jl                     6 is_known_call(x::Any, func::Any, ir::Compiler.IncrementalCompact)
     1         1 @Compiler/src/utilities.jl                      216 singleton_type
     1         1 @Compiler/src/ssair/inlining.jl                   ? ssa_substitute_op!(insert_node!::Compiler.InsertBefore{Compiler.IncrementalCompact}, subst_inst::Compiler.Instruction, val::Any, ssa_substitute::C…
     1         0 @Compiler/src/ssair/ir.jl                      1352 renumber_ssa2!(stmt::Any, ssanums::Vector{Any}, used_ssas::Vector{Int64}, new_new_used_ssas::Vector{Int64}, late_fixup::Vector{Int64}, result_idx:…
     1         0 @Compiler/src/ssair/inlining.jl                 542 ir_inline_unionsplit!(compact::Compiler.IncrementalCompact, idx::Int64, argexprs::Vector{Any}, union_split::Compiler.UnionSplit, boundscheck::Symb…
     1         0 @Base/runtime_internals.jl                     1156 fieldcount
     1         0 @Base/runtime_internals.jl                     1129 datatype_fieldcount
     1         1 @Base/essentials.jl                               ? isvatuple
     1         0 @Compiler/src/ssair/inlining.jl                 581 ir_inline_unionsplit!(compact::Compiler.IncrementalCompact, idx::Int64, argexprs::Vector{Any}, union_split::Compiler.UnionSplit, boundscheck::Symb…
     1         0 @Compiler/src/ssair/slot2ssa.jl                  29 scan_entry!(result::Vector{Compiler.SlotInfo}, idx::Int64, stmt::Any)
     1         1 @Compiler/src/abstractinterpretation.jl         170 (::Compiler.var"#handle1#abstract_call_gf_by_type##1"{Int64, Compiler.Future{Compiler.MethodCallResult}, Int64, Vector{Union{Nothing, Core.CodeIns…
     1         1 @Base/array.jl                                 2227 reverse!(v::Vector{Any}, start::Int64, stop::Int64)
     1         0 @Base/expr.jl                                    81 copy(c::Core.CodeInfo)
     1         0 @Base/expr.jl                                    68 copy_exprs(x::Any)
     1         0 @Base/expr.jl                                    45 copy(x::Core.PhiNode)
     1         0 @Compiler/src/ssair/inlining.jl                 437 ir_inline_item!(compact::Compiler.IncrementalCompact, idx::Int64, argexprs::Vector{Any}, item::Compiler.InliningTodo, boundscheck::Symbol, todo_bb…
     1         1 @Compiler/src/ssair/ir.jl                      1152 setindex!(compact::Compiler.IncrementalCompact, v::Any, idx::Int64)
     1         0 @Compiler/src/abstractinterpretation.jl        1054 maybe_get_const_prop_profitable(interp::Compiler.NativeInterpreter, result::Compiler.MethodCallResult, f::Any, arginfo::Compiler.ArgInfo, si::Comp…
     1         0 @Compiler/src/abstractinterpretation.jl        3004 abstract_eval_value
     1         0 @Compiler/src/inferencestate.jl                1064 merge_effects!
     1         0 @Compiler/src/effects.jl                        276 merge_effects
     1         1 @Compiler/src/effects.jl                        289 merge_effectbits
     1         0 @Compiler/src/ssair/slot2ssa.jl                 140 fixemup!(slot_filter::Compiler.var"#rename_uses!##0#rename_uses!##1", rename_slot::Compiler.var"#rename_uses!##2#rename_uses!##3"{Vector{Pair{Any,…
     1         0 @Compiler/src/ssair/passes.jl                  1339 sroa_pass!(ir::Compiler.IRCode, inlining::Compiler.InliningState{Compiler.NativeInterpreter})
     1         1 @Compiler/src/ssair/ir.jl                         ? compute_basic_blocks(stmts::Vector{Any})
     1         0 @Compiler/src/ssair/slot2ssa.jl                 233 iterated_dominance_frontier(cfg::Compiler.CFG, liveness::Compiler.BlockLiveness, domtree::Compiler.GenericDomTree{false})
     1         0 @Compiler/src/ssair/ir.jl                       774 Compiler.IncrementalCompact(code::Compiler.IRCode, cfg_transform::Compiler.CFGTransformState)
     1         0 @Compiler/src/ssair/ir.jl                      1551 process_node!(compact::Compiler.IncrementalCompact, result_idx::Int64, inst::Compiler.Instruction, idx::Int64, processed_idx::Int64, active_bb::In…
     1         0 @Compiler/src/abstractinterpretation.jl        3043 abstract_call(interp::Compiler.NativeInterpreter, arginfo::Compiler.ArgInfo, sstate::Compiler.StatementState, sv::Compiler.InferenceState)
     1         0 @Compiler/src/inferencestate.jl                1177 Compiler.Future{Any}(f::Compiler.var"#abstract_call##0#abstract_call##1", prev::Compiler.Future{Compiler.CallMeta}, interp::Compiler.NativeInterpr…
     1         1 @Compiler/src/inferencestate.jl                1152 getindex
     1         1 @Compiler/src/ssair/ir.jl                      1481 process_node!(compact::Compiler.IncrementalCompact, result_idx::Int64, inst::Compiler.Instruction, idx::Int64, processed_idx::Int64, active_bb::In…
     1         0 @Compiler/src/optimize.jl                       833 scan_inconsistency!(inst::Compiler.Instruction, sv::Compiler.PostOptAnalysisState)
     1         0 @Compiler/src/optimize.jl                       701 iscall_with_boundscheck(stmt::Any, sv::Compiler.PostOptAnalysisState)
     1         1 @Compiler/src/abstractinterpretation.jl        3604 walk_binding_partition(imported_binding::Core.Binding, partition::Core.BindingPartition, world::UInt64)
     1         0 @Compiler/src/abstractinterpretation.jl        3037 abstract_call(interp::Compiler.NativeInterpreter, arginfo::Compiler.ArgInfo, sstate::Compiler.StatementState, sv::Compiler.InferenceState)
     1         0 @Compiler/src/abstractinterpretation.jl          12 call_result_unused
     1         0 @Base/bitset.jl                                 291 isempty
     1         0 @Compiler/src/inferencestate.jl                 478 (::Compiler.ComputeTryCatch{Compiler.TryCatchFrame})(code::Vector{Any}, bbs::Nothing)
     1         0 @Compiler/src/ssair/legacy.jl                    14 inflate_ir!(ci::Core.CodeInfo, mi::Core.MethodInstance)
     1         1 @Compiler/src/inferenceresult.jl                  ? most_general_argtypes(method::Method, specTypes::Any)
     1         0 @Compiler/src/abstractinterpretation.jl        4352 typeinf_local(interp::Compiler.NativeInterpreter, frame::Compiler.InferenceState, nextresult::Compiler.CurrentState)
     1         0 @Compiler/src/abstractinterpretation.jl        4118 update_exc_bestguess!(interp::Compiler.NativeInterpreter, exct::Any, frame::Compiler.InferenceState)
     1         0 @Compiler/src/abstractinterpretation.jl        4149 update_cycle_worklists!(callback::Compiler.var"#update_exc_bestguess!##0#update_exc_bestguess!##1", frame::Compiler.InferenceState)
     1         0 @Base/array.jl                                  901 iterate
     1         1 @Base/int.jl                                    519 <
     1         1 @Compiler/src/ssair/ir.jl                       531 _useref_getindex(stmt::Any, op::Int64)
     1         1 @Base/Base_compiler.jl                            ? findall(sig::Type, table::Compiler.CachedMethodTable{Compiler.InternalMethodTable}; limit::Int64)
     1         0 @Compiler/src/abstractinterpretation.jl        4546 typeinf(interp::Compiler.NativeInterpreter, frame::Compiler.InferenceState)
     1         1 @Compiler/src/utilities.jl                      364 _time_ns
     1         0 @Compiler/src/abstractinterpretation.jl        3061 abstract_eval_call
     1         0 @Compiler/src/inferencestate.jl                1182 Compiler.Future{Compiler.RTEffects}(f::Compiler.var"#abstract_eval_call##0#abstract_eval_call##1", prev::Compiler.Future{Compiler.CallMeta}, inter…
     1         0 @Base/array.jl                                 1297 push!
     1         1 @Base/array.jl                                 1008 __safe_setindex!
     1         1 @Compiler/src/abstractinterpretation.jl        1283 semi_concrete_eval_call(interp::Compiler.NativeInterpreter, mi::Core.MethodInstance, result::Compiler.MethodCallResult, arginfo::Compiler.ArgInfo,…
     1         0 @Compiler/src/typeinfer.jl                     1424 add_codeinsts_to_jit!(interp::Compiler.NativeInterpreter, ci::Core.CodeInstance, source_mode::UInt8)
     1         0 @Compiler/src/typeinfer.jl                     1339 collectinvokes!
     1         0 @Compiler/src/typeinfer.jl                     1346 #collectinvokes!#166
     1         0 @Compiler/src/ssair/passes.jl                   435 lift_leaves(compact::Compiler.IncrementalCompact, field::Int64, leaves::Vector{Any}, 𝕃ₒ::Compiler.PartialsLattice{Compiler.ConstsLattice})
     1         0 @Compiler/src/ssair/passes.jl                   513 lift_arg!(compact::Compiler.IncrementalCompact, leaf::Any, cache_key::Any, stmt::Expr, argidx::Int64, lifted_leaves::IdDict{Any, Union{Nothing, Co…
     1         1 @Base/iddict.jl                                  86 setindex!
     1         0 @Compiler/src/tfuncs.jl                         355 egal_tfunc(𝕃::Compiler.InferenceLattice{Compiler.ConditionalsLattice{Compiler.PartialsLattice{Compiler.ConstsLattice}}}, x::Any, y::Any)
     1         0 @Compiler/src/tfuncs.jl                         375 egal_tfunc
     1         0 @Compiler/src/tfuncs.jl                         355 egal_tfunc
     1         0 @Compiler/src/tfuncs.jl                         384 egal_tfunc
     1         0 @Compiler/src/tfuncs.jl                         387 egal_tfunc(::Compiler.JLTypeLattice, x::Any, y::Any)
     1         0 @Compiler/src/ssair/slot2ssa.jl                 130 fixemup!(slot_filter::Compiler.var"#rename_uses!##0#rename_uses!##1", rename_slot::Compiler.var"#rename_uses!##2#rename_uses!##3"{Vector{Pair{Any,…
     1         0 @Compiler/src/abstractinterpretation.jl         277 (::Compiler.var"#infercalls#abstract_call_gf_by_type##0"{Compiler.ArgInfo, Compiler.StmtInfo, Compiler.CallInferenceState, Compiler.Future{Compile…
     1         0 @Compiler/src/abstractinterpretation.jl         425 from_interprocedural!(interp::Compiler.NativeInterpreter, rt::Any, sv::Compiler.InferenceState, arginfo::Compiler.ArgInfo, maybecondinfo::Any)
     1         0 @Compiler/src/typelattice.jl                    279 is_lattice_bool
     1         0 @Compiler/src/inferencestate.jl                 337 Compiler.InferenceState(result::Compiler.InferenceResult, src::Core.CodeInfo, cache_mode::UInt8, interp::Compiler.NativeInterpreter)
     1         0 @Base/bitset.jl                                  38 union!(s::BitSet, itr::Int64)
     1         1 @Base/bitset.jl                                 121 _growend0!
     1         0 @Compiler/src/abstractinterpretation.jl         323 (::Compiler.var"#infercalls2#abstract_call_gf_by_type##2"{Compiler.SafeBox{Int64}, Bool, Compiler.ArgInfo, Vector{Compiler.MethodMatchTarget}})(in…
     1         0 @Compiler/src/typeinfer.jl                      905 typeinf_edge(interp::Compiler.NativeInterpreter, method::Method, atype::Any, sparams::Core.SimpleVector, caller::Compiler.InferenceState, edgecycl…
     1         1 @Base/runtime_internals.jl                     1588 #specialize_method#8
     1         1 @Compiler/src/ssair/slot2ssa.jl                 857 construct_ssa!(ci::Core.CodeInfo, ir::Compiler.IRCode, sv::Compiler.OptimizationState{Compiler.NativeInterpreter}, domtree::Compiler.GenericDomTre…
     1         0 @Compiler/src/ssair/ir.jl                       145 compute_basic_blocks(stmts::Vector{Any})
     1         0 @Compiler/src/ssair/slot2ssa.jl                 519 compute_live_ins(cfg::Compiler.CFG, defs::Vector{Int64}, uses::Vector{Int64})
     1         1 @Compiler/src/optimize.jl                       238 ir_to_codeinf!
     1         0 @Compiler/src/tfuncs.jl                        2575 getfield_effects(𝕃::Compiler.InferenceLattice{Compiler.ConditionalsLattice{Compiler.PartialsLattice{Compiler.ConstsLattice}}}, argtypes::Vector{An…
     1         1 @Compiler/src/tfuncs.jl                           ? getfield_boundscheck(argtypes::Vector{Any})
     1         1 @Compiler/src/tfuncs.jl                         107 instanceof_tfunc(t::Any, astag::Bool, troot::Core.Const)
     1         0 @Compiler/src/optimize.jl                      1329 statement_cost(ex::Expr, line::Int64, src::Compiler.IRCode, sptypes::Vector{Compiler.VarState}, params::Compiler.OptimizationParams)
     1         1 @Compiler/src/optimize.jl                       229 ir_to_codeinf!
     1         1 @Compiler/src/ssair/ir.jl                      1450 process_node!(compact::Compiler.IncrementalCompact, result_idx::Int64, inst::Compiler.Instruction, idx::Int64, processed_idx::Int64, active_bb::In…
     1         0 @Compiler/src/abstractinterpretation.jl        1040 maybe_get_const_prop_profitable(interp::Compiler.NativeInterpreter, result::Compiler.MethodCallResult, f::Any, arginfo::Compiler.ArgInfo, si::Comp…
     1         1 @Compiler/src/abstractinterpretation.jl        1076 const_prop_rettype_heuristic
     1         0 @Compiler/src/optimize.jl                      1342 statement_cost(ex::Expr, line::Int64, src::Compiler.IRCode, sptypes::Vector{Compiler.VarState}, params::Compiler.OptimizationParams)
     1         1 @Base/tuple.jl                                  162 indexed_iterate
     1         1 @Compiler/src/abstractinterpretation.jl           ? abstract_call(interp::Compiler.NativeInterpreter, arginfo::Compiler.ArgInfo, si::Compiler.StmtInfo, sv::Compiler.InferenceState, max_methods::Int6…
     1         1 @Compiler/src/optimize.jl                       452 argextype(x::Any, src::Compiler.IncrementalCompact, sptypes::Vector{Compiler.VarState}, slottypes::Vector{Any})
     1         0 @Compiler/src/typeinfer.jl                      669 record_slot_assign!(sv::Compiler.InferenceState)
     1         0 @Compiler/src/ssair/passes.jl                  1398 sroa_pass!(ir::Compiler.IRCode, inlining::Compiler.InliningState{Compiler.NativeInterpreter})
     1         1 @Compiler/src/abstractinterpretation.jl         275 (::Compiler.var"#infercalls#abstract_call_gf_by_type##0"{Compiler.ArgInfo, Compiler.StmtInfo, Compiler.CallInferenceState, Compiler.Future{Compile…
     1         0 @Compiler/src/ssair/slot2ssa.jl                 246 iterated_dominance_frontier(cfg::Compiler.CFG, liveness::Compiler.BlockLiveness, domtree::Compiler.GenericDomTree{false})
     1         1 @Compiler/src/ssair/ir.jl                      1274 process_phinode_values(old_values::Vector{Any}, late_fixup::Vector{Int64}, already_inserted::Compiler.var"#did_already_insert#already_inserted_ssa…
     1         0 @Compiler/src/ssair/passes.jl                  1400 sroa_pass!(ir::Compiler.IRCode, inlining::Compiler.InliningState{Compiler.NativeInterpreter})
     1         1 @Compiler/src/ssair/ir.jl                      1460 process_node!(compact::Compiler.IncrementalCompact, result_idx::Int64, inst::Compiler.Instruction, idx::Int64, processed_idx::Int64, active_bb::In…
     1         0 @Compiler/src/typeinfer.jl                      236 finish_cycle(::Compiler.NativeInterpreter, frames::Vector{Union{Compiler.IRInterpretationState, Compiler.InferenceState}}, cycleid::Int64, time_be…
     1         0 @Compiler/src/stmtinfo.jl                        62 _add_edges_impl(edges::Vector{Any}, info::Compiler.MethodMatchInfo, mi_edge::Bool)
     1         0 @Compiler/src/methodtable.jl                     10 length
     1         1 @Compiler/src/ssair/slot2ssa.jl                 649 construct_ssa!(ci::Core.CodeInfo, ir::Compiler.IRCode, sv::Compiler.OptimizationState{Compiler.NativeInterpreter}, domtree::Compiler.GenericDomTre…
     1         1 @Compiler/src/inferencestate.jl                1174 Compiler.Future{Any}(f::Compiler.var"#abstract_call##0#abstract_call##1", prev::Compiler.Future{Compiler.CallMeta}, interp::Compiler.NativeInterpr…
     1         1 @Compiler/src/inferencestate.jl                1153 setindex!(f::Compiler.Future{Compiler.RTEffects}, v::Compiler.RTEffects)
     1         1 @Base/Base.jl                                    12 Pair
     1         0 @Compiler/src/ssair/domtree.jl                  270 compute_domtree_nodes!(domtree::Compiler.GenericDomTree{false})
     1         0 @Compiler/src/ssair/irinterp.jl                 351 (::Compiler.var"#218#219"{Nothing, Compiler.NativeInterpreter, Compiler.IRInterpretationState, Compiler.var"#check_ret!#217"{Vector{Int64}}, BitSe…
     1         0 @Base/bitset.jl                                 330 in
     1         0 @Base/bitset.jl                                  64 _bits_getindex
     1         0 @Base/bitarray.jl                               119 _div64
     1         0 @Base/int.jl                                    540 >>
     1         1 @Base/int.jl                                    533 >>
     1         0 @Compiler/src/ssair/passes.jl                  1411 sroa_pass!(ir::Compiler.IRCode, inlining::Compiler.InliningState{Compiler.NativeInterpreter})
     1         0 @Compiler/src/ssair/passes.jl                     4 is_known_call(x::Any, func::Any, ir::Compiler.IncrementalCompact)
     1         0 @Compiler/src/ssair/slot2ssa.jl                 509 compute_live_ins(cfg::Compiler.CFG, defs::Vector{Int64}, uses::Vector{Int64})
     1         0 @Compiler/src/ssair/ir.jl                        44 block_for_inst
     1         0 @Compiler/src/abstractinterpretation.jl        3039 abstract_call(interp::Compiler.NativeInterpreter, arginfo::Compiler.ArgInfo, sstate::Compiler.StatementState, sv::Compiler.InferenceState)
     1         0 @Compiler/src/inferencestate.jl                1053 add_curr_ssaflag!
     1         1 @Base/essentials.jl                               ? getindex
     1         1 @Base/array.jl                                    ? domsort_ssa!(ir::Compiler.IRCode, domtree::Compiler.GenericDomTree{false})
     1         0 @Compiler/src/stmtinfo.jl                        76 _add_edges_impl(edges::Vector{Any}, info::Compiler.MethodMatchInfo, mi_edge::Bool)
     1         0 @Compiler/src/stmtinfo.jl                       135 add_one_edge!
     1         0 @Compiler/src/ssair/slot2ssa.jl                  28 scan_entry!(result::Vector{Compiler.SlotInfo}, idx::Int64, stmt::Any)
     1         0 @Compiler/src/ssair/passes.jl                  1403 sroa_pass!(ir::Compiler.IRCode, inlining::Compiler.InliningState{Compiler.NativeInterpreter})
     1         0 @Compiler/src/ssair/passes.jl                   571 lift_comparison!(::typeof(===), compact::Compiler.IncrementalCompact, idx::Int64, stmt::Expr, 𝕃ₒ::Compiler.PartialsLattice{Compiler.ConstsLattice})
     1         0 @Compiler/src/ssair/passes.jl                   608 lift_comparison_leaves!(tfunc::typeof(Compiler.egal_tfunc), compact::Compiler.IncrementalCompact, val::Any, cmp::Any, idx::Int64, 𝕃ₒ::Compiler.Par…
     1         0 @Compiler/src/ssair/passes.jl                   342 walk_to_defs(compact::Compiler.IncrementalCompact, defssa::Any, typeconstraint::Any, predecessors::typeof(Compiler.phi_or_ifelse_predecessors), 𝕃ₒ…
     1         0 @Base/array.jl                                 1126 _growend!
     1         0 @Compiler/src/abstractinterpretation.jl        4057 update_bbstate!
     1         1 @Compiler/src/typelimits.jl                     493 tmerge(lattice::Compiler.ConditionalsLattice{Compiler.PartialsLattice{Compiler.ConstsLattice}}, typea::Any, typeb::Any)
     1         1 @Compiler/src/utilities.jl                      127 retrieve_code_info(mi::Core.MethodInstance, world::UInt64)
     1         0 @Compiler/src/ssair/inlining.jl                 127 cfg_inline_item!(ir::Compiler.IRCode, idx::Int64, todo::Compiler.InliningTodo, state::Compiler.CFGInliningState, from_unionsplit::Bool)
     1         0 @Compiler/src/ssair/inlining.jl                 114 inline_into_block!(state::Compiler.CFGInliningState, block::Int64)
     1         0 @Base/array.jl                                 1365 _append!(a::Vector{Compiler.BasicBlock}, ::Base.HasShape{1}, iter::Base.Generator{Vector{Compiler.BasicBlock}, Compiler.var"#inline_into_block!##0…
     1         0 none                                              ? #inline_into_block!##0
     1         0 @Compiler/src/ssair/basicblock.jl                32 copy
     1         0 @Compiler/src/ssair/slot2ssa.jl                 682 construct_ssa!(ci::Core.CodeInfo, ir::Compiler.IRCode, sv::Compiler.OptimizationState{Compiler.NativeInterpreter}, domtree::Compiler.GenericDomTre…
     1         1 @Compiler/src/optimize.jl                       968 ipo_dataflow_analysis!(interp::Compiler.NativeInterpreter, opt::Compiler.OptimizationState{Compiler.NativeInterpreter}, ir::Compiler.IRCode, resul…
     1         1 @Base/abstractarray.jl                          914 copy!(dst::Vector{Compiler.DomTreeNode}, src::Vector{Compiler.DomTreeNode})
     1         0 @Compiler/src/ssair/ir.jl                       568 _useref_getindex(stmt::Any, op::Int64)
     1         0 @Base/array.jl                                  249 isassigned
     1         1 @Base/genericmemory.jl                          113 isassigned
     1         1 @Compiler/src/abstractinterpretation.jl         168 (::Compiler.var"#handle1#abstract_call_gf_by_type##1"{Int64, Compiler.Future{Compiler.MethodCallResult}, Int64, Vector{Union{Nothing, Core.CodeIns…
     1         0 @Compiler/src/inferencestate.jl                1215 doworkloop(interp::Compiler.NativeInterpreter, sv::Compiler.InferenceState)
     1         0 @Base/array.jl                                 2228 reverse!
     1         0 @Base/array.jl                                 2229 reverse!(v::Vector{Any}, start::Int64, stop::Int64)
     1         0 @Compiler/src/ssair/slot2ssa.jl                 736 construct_ssa!(ci::Core.CodeInfo, ir::Compiler.IRCode, sv::Compiler.OptimizationState{Compiler.NativeInterpreter}, domtree::Compiler.GenericDomTre…
     1         0 @Compiler/src/abstractinterpretation.jl        1320 const_prop_call(interp::Compiler.NativeInterpreter, mi::Core.MethodInstance, result::Compiler.MethodCallResult, arginfo::Compiler.ArgInfo, sv::Com…
     1         1 @Compiler/src/inferenceresult.jl                186 cache_lookup(𝕃::Compiler.InferenceLattice{Compiler.ConditionalsLattice{Compiler.PartialsLattice{Compiler.ConstsLattice}}}, mi::Core.MethodInstance…
     1         1 @Base/Base_compiler.jl                           54 getproperty(x::Core.MethodInstance, f::Symbol)
     1         0 @Compiler/src/tfuncs.jl                        2306 _builtin_nothrow(𝕃::Compiler.PartialsLattice{Compiler.ConstsLattice}, f::Core.Builtin, argtypes::Vector{Any}, rt::Any)
     1         0 @Compiler/src/tfuncs.jl                        2225 memoryref_builtin_common_nothrow(argtypes::Vector{Any})
     1         0 @Compiler/src/abstractlattice.jl                303 ⊑
     1         0 @Compiler/src/optimize.jl                       327 stmt_effect_flags(𝕃ₒ::Compiler.PartialsLattice{Compiler.ConstsLattice}, stmt::Any, rt::Any, src::Compiler.IRCode)
     1         1 @Compiler/src/abstractinterpretation.jl        3602 walk_binding_partition(imported_binding::Core.Binding, partition::Core.BindingPartition, world::UInt64)
     1         0 @Compiler/src/ssair/passes.jl                  1569 sroa_pass!(ir::Compiler.IRCode, inlining::Compiler.InliningState{Compiler.NativeInterpreter})
     1         0 @Compiler/src/abstractinterpretation.jl        4233 typeinf_local(interp::Compiler.NativeInterpreter, frame::Compiler.InferenceState, nextresult::Compiler.CurrentState)
     1         1 @Compiler/src/typelattice.jl                    520 ⊑(lattice::Compiler.ConstsLattice, a::Any, b::Any)
     1         0 @Compiler/src/abstractinterpretation.jl        4308 typeinf_local(interp::Compiler.NativeInterpreter, frame::Compiler.InferenceState, nextresult::Compiler.CurrentState)
     1         0 @Compiler/src/abstractinterpretation.jl        4074 update_bestguess!(interp::Compiler.NativeInterpreter, frame::Compiler.InferenceState, currstate::Vector{Compiler.VarState}, rt::Any)
     1         0 @Compiler/src/abstractinterpretation.jl        3890 widenreturn
     1         0 @Compiler/src/abstractinterpretation.jl        3894 widenreturn
     1         0 @Compiler/src/abstractinterpretation.jl        3947 widenreturn(𝕃ᵢ::Compiler.ConditionalsLattice{Compiler.PartialsLattice{Compiler.ConstsLattice}}, rt::Any, info::Compiler.BestguessInfo{Compiler.Nat…
     1         0 @Compiler/src/abstractinterpretation.jl        3990 widenreturn
     1         1 @Compiler/src/abstractinterpretation.jl        4011 widenreturn_partials(𝕃ᵢ::Compiler.PartialsLattice{Compiler.ConstsLattice}, rt::Any, info::Compiler.BestguessInfo{Compiler.NativeInterpreter})
     1         0 @Compiler/src/inferencestate.jl                 340 Compiler.InferenceState(result::Compiler.InferenceResult, src::Core.CodeInfo, cache_mode::UInt8, interp::Compiler.NativeInterpreter)
     1         0 @Compiler/src/utilities.jl                      284 find_ssavalue_uses(body::Vector{Any}, nvals::Int64)
     1         0 @Compiler/src/utilities.jl                      300 find_ssavalue_uses!(uses::Vector{BitSet}, e::Expr, line::Int64)
     1         0 @Base/bitset.jl                                 119 _growend0!
     1         0 @Compiler/src/ssair/inlining.jl                 161 cfg_inline_item!(ir::Compiler.IRCode, idx::Int64, todo::Compiler.InliningTodo, state::Compiler.CFGInliningState, from_unionsplit::Bool)
     1         0 @Base/array.jl                                 1352 append!
     1         0 @Compiler/src/ssair/passes.jl                  2138 adce_pass!(ir::Compiler.IRCode, inlining::Compiler.InliningState{Compiler.NativeInterpreter})
     1         0 @Compiler/src/ssair/domtree.jl                  238 GenericDomTree
     1         0 @Compiler/src/ssair/domtree.jl                   90 Compiler.DFSTree(n_blocks::Int64)
     1         0 @Base/array.jl                                  591 zeros
     1         0 @Base/array.jl                                  595 zeros
     1         0 @Compiler/src/typeinfer.jl                      959 typeinf_edge(interp::Compiler.NativeInterpreter, method::Method, atype::Any, sparams::Core.SimpleVector, caller::Compiler.InferenceState, edgecycl…
     1         0 @Compiler/src/types.jl                          128 InferenceResult
     1         1 @Compiler/src/inferenceresult.jl                128 most_general_argtypes(method::Method, specTypes::Any)
     1         1 @Base/int.jl                                      ? process_node!(compact::Compiler.IncrementalCompact, result_idx::Int64, inst::Compiler.Instruction, idx::Int64, processed_idx::Int64, active_bb::In…
     1         1 @Compiler/src/tfuncs.jl                        2763 builtin_tfunction(interp::Compiler.NativeInterpreter, f::Any, argtypes::Vector{Any}, sv::Compiler.IRInterpretationState)
     1         0 @Compiler/src/ssair/slot2ssa.jl                 424 domsort_ssa!(ir::Compiler.IRCode, domtree::Compiler.GenericDomTree{false})
     1         0 @Compiler/src/ssair/ir.jl                       347 setindex!
     1         1 @Compiler/src/abstractinterpretation.jl        2686 abstract_call_known(interp::Compiler.NativeInterpreter, f::Any, arginfo::Compiler.ArgInfo, si::Compiler.StmtInfo, sv::Compiler.InferenceState, max…
     1         0 @Compiler/src/ssair/slot2ssa.jl                 399 domsort_ssa!(ir::Compiler.IRCode, domtree::Compiler.GenericDomTree{false})
     1         1 @Compiler/src/ssair/ir.jl                      1868 iterate_compact(compact::Compiler.IncrementalCompact)
     1         0 @Compiler/src/abstractinterpretation.jl         242 (::Compiler.var"#handle1#abstract_call_gf_by_type##1"{Int64, Compiler.Future{Compiler.MethodCallResult}, Int64, Vector{Union{Nothing, Core.CodeIns…
     1         0 @Compiler/src/abstractinterpretation.jl         559 conditional_argtype(𝕃ᵢ::Compiler.InferenceLattice{Compiler.ConditionalsLattice{Compiler.PartialsLattice{Compiler.ConstsLattice}}}, rt::Any, sig::A…
     1         0 @Compiler/src/typelattice.jl                    665 tmeet(lattice::Compiler.InferenceLattice{Compiler.ConditionalsLattice{Compiler.PartialsLattice{Compiler.ConstsLattice}}}, v::Any, t::Type)
     1         0 @Compiler/src/typelattice.jl                    652 tmeet
     1         0 @Compiler/src/typelattice.jl                    632 tmeet(lattice::Compiler.PartialsLattice{Compiler.ConstsLattice}, v::Any, t::Type)
     1         1 @Compiler/src/typelattice.jl                    642 tmeet(lattice::Compiler.ConstsLattice, v::Any, t::Type)
     1         0 @Compiler/src/ssair/inlining.jl                1452 semiconcrete_result_item(result::Compiler.SemiConcreteResult, info::Compiler.CallInfo, flag::UInt32, state::Compiler.InliningState{Compiler.Native…
     1         0 @Compiler/src/ssair/inlining.jl                  70 add_inlining_edge!
     1         1 @Base/array.jl                                    ? add_inlining_edge!(edges::Vector{Any}, edge::Core.CodeInstance)
     1         0 @Compiler/src/inferencestate.jl                 473 (::Compiler.ComputeTryCatch{Compiler.SimpleHandler})(code::Vector{Any}, bbs::Nothing)
     1         1 @Base/range.jl                                    ? _growend0!
     1         0 @Compiler/src/ssair/inlining.jl                1353 compute_inlining_cases(info::Compiler.CallInfo, flag::UInt32, sig::Compiler.Signature, state::Compiler.InliningState{Compiler.NativeInterpreter})
     1         0 @Compiler/src/ssair/inlining.jl                1321 info_effects(result::Any, match::Core.MethodMatch, state::Compiler.InliningState{Compiler.NativeInterpreter})
     1         0 @Compiler/src/ssair/inlining.jl                1082 inline_apply!(todo::Vector{Pair{Int64, Any}}, ir::Compiler.IRCode, idx::Int64, stmt::Expr, sig::Compiler.Signature, state::Compiler.InliningState{…
     1         1 @Compiler/src/ssair/ir.jl                       318 getindex
     1         0 @Compiler/src/ssair/inlining.jl                 633 batch_inline!(ir::Compiler.IRCode, todo::Vector{Pair{Int64, Any}}, propagate_inbounds::Bool, interp::Compiler.NativeInterpreter)
     1         1 @Compiler/src/ssair/inlining.jl                 261 finish_cfg_inline!(state::Compiler.CFGInliningState)
     1         0 @Compiler/src/tfuncs.jl                        2336 _builtin_nothrow(𝕃::Compiler.InferenceLattice{Compiler.ConditionalsLattice{Compiler.PartialsLattice{Compiler.ConstsLattice}}}, f::Core.Builtin, ar…
     1         0 @Compiler/src/tfuncs.jl                        1514 fieldtype_nothrow(𝕃::Compiler.InferenceLattice{Compiler.ConditionalsLattice{Compiler.PartialsLattice{Compiler.ConstsLattice}}}, s0::Any, name::Any)
     1         0 @Compiler/src/abstractinterpretation.jl        1049 maybe_get_const_prop_profitable(interp::Compiler.NativeInterpreter, result::Compiler.MethodCallResult, f::Any, arginfo::Compiler.ArgInfo, si::Comp…
     1         1 @Compiler/src/abstractinterpretation.jl        1197 const_prop_function_heuristic(interp::Compiler.NativeInterpreter, f::Any, arginfo::Compiler.ArgInfo, all_overridden::Bool, sv::Compiler.InferenceS…
     1         0 @Compiler/src/ssair/inlining.jl                1272 process_simple!(todo::Vector{Pair{Int64, Any}}, ir::Compiler.IRCode, idx::Int64, flag::UInt32, state::Compiler.InliningState{Compiler.NativeInterp…
     1         1 @Compiler/src/ssair/inlining.jl                1151 is_builtin
     1         0 @Compiler/src/ssair/legacy.jl                    46 inflate_ir!(ci::Core.CodeInfo, sptypes::Vector{Compiler.VarState}, argtypes::Vector{Any})
     1         0 @Compiler/src/ssair/ir.jl                      1589 process_node!(compact::Compiler.IncrementalCompact, result_idx::Int64, inst::Compiler.Instruction, idx::Int64, processed_idx::Int64, active_bb::In…
     1         1 @Compiler/src/ssair/ir.jl                       591 _useref_setindex!(stmt::Any, op::Int64, v::Any)
     1         1 @Compiler/src/ssair/ir.jl                       265 Compiler.InstructionStream(len::Int64)
     1         0 @Compiler/src/ssair/ir.jl                       605 _useref_setindex!(stmt::Any, op::Int64, v::Any)
     1         1 @Base/essentials.jl                             925 setindex!
     1         0 @Compiler/src/ssair/slot2ssa.jl                 528 compute_live_ins(cfg::Compiler.CFG, defs::Vector{Int64}, uses::Vector{Int64})
     1         0 @Base/array.jl                                 1371 _append!(a::Vector{Int64}, ::Base.SizeUnknown, iter::Base.Iterators.Filter{Compiler.var"#compute_live_ins##2#compute_live_ins##3"{Vector{Int64}}, …
     1         0 @Base/iterators.jl                              538 iterate
     1         0 @Compiler/src/ssair/slot2ssa.jl                 528 #compute_live_ins##2
     1         1 @Base/operators.jl                                ? in
     1         0 @Compiler/src/ssair/ir.jl                      1563 process_node!(compact::Compiler.IncrementalCompact, result_idx::Int64, inst::Compiler.Instruction, idx::Int64, processed_idx::Int64, active_bb::In…
     1         1 @Compiler/src/abstractinterpretation.jl        1953 abstract_call_builtin(interp::Compiler.NativeInterpreter, f::Core.Builtin, ::Compiler.ArgInfo, sv::Compiler.InferenceState)
     1         0 @Compiler/src/ssair/passes.jl                  1408 sroa_pass!(ir::Compiler.IRCode, inlining::Compiler.InliningState{Compiler.NativeInterpreter})
     1         0 @Compiler/src/ssair/passes.jl                    12 is_known_invoke_or_call(x::Any, func::Any, ir::Compiler.IncrementalCompact)
     1         1 @Compiler/src/ssair/slot2ssa.jl                  30 scan_entry!(result::Vector{Compiler.SlotInfo}, idx::Int64, stmt::Any)
     1         0 @Compiler/src/ssair/inlining.jl                1137 inline_apply!(todo::Vector{Pair{Int64, Any}}, ir::Compiler.IRCode, idx::Int64, stmt::Expr, sig::Compiler.Signature, state::Compiler.InliningState{…
     1         0 @Compiler/src/ssair/inlining.jl                 736 rewrite_apply_exprargs!(todo::Vector{Pair{Int64, Any}}, ir::Compiler.IRCode, idx::Int64, stmt::Expr, argtypes::Vector{Any}, arginfos::Vector{Union…
     1         0 @Compiler/src/ssair/ir.jl                       294 resize!(stmts::Compiler.InstructionStream, len::Int64)
     1         1 @Compiler/src/utilities.jl                      248 ssamap(f::Compiler.var"#renumber_ssa!##0#renumber_ssa!##1"{Vector{Core.SSAValue}, Bool}, stmt::Any)
     1         0 @Compiler/src/abstractinterpretation.jl         232 (::Compiler.var"#handle1#abstract_call_gf_by_type##1"{Int64, Compiler.Future{Compiler.MethodCallResult}, Int64, Vector{Union{Nothing, Core.CodeIns…
     1         0 @Compiler/src/abstractlattice.jl                291 #tmerge##0
     1         1 @Compiler/src/typelimits.jl                     486 tmerge(lattice::Compiler.InferenceLattice{Compiler.InterConditionalsLattice{Compiler.PartialsLattice{Compiler.ConstsLattice}}}, typea::Any, typeb:…
     1         1 @Compiler/src/optimize.jl                      1357 statement_cost(ex::Expr, line::Int64, src::Compiler.IRCode, sptypes::Vector{Compiler.VarState}, params::Compiler.OptimizationParams)
     1         0 @Compiler/src/abstractinterpretation.jl        4382 typeinf_local(interp::Compiler.NativeInterpreter, frame::Compiler.InferenceState, nextresult::Compiler.CurrentState)
     1         0 @Compiler/src/inferencestate.jl                 802 record_ssa_assign!(𝕃ᵢ::Compiler.InferenceLattice{Compiler.ConditionalsLattice{Compiler.PartialsLattice{Compiler.ConstsLattice}}}, ssa_id::Int64, n…
     1         0 @Base/bitset.jl                                 334 iterate
     1         0 @Base/int.jl                                    524 ==
     1         0 @Compiler/src/ssair/legacy.jl                    18 inflate_ir!(ci::Core.CodeInfo, mi::Core.MethodInstance)
     1         0 @Base/array.jl                                  937 getindex(A::Vector{Any}, I::UnitRange{UInt64})
     1         1 @Base/boot.jl                                     ? similar
     1         0 @Compiler/src/ssair/slot2ssa.jl                  75 new_to_regular(stmt::Any, new_offset::Int64)
     1         1 @Compiler/src/inferencestate.jl                   ? (::Compiler.ComputeTryCatch{Compiler.TryCatchFrame})(code::Vector{Any}, bbs::Nothing)
     1         0 @Compiler/src/ssair/inlining.jl                1622 assemble_inline_todo!(ir::Compiler.IRCode, state::Compiler.InliningState{Compiler.NativeInterpreter})
     1         0 @Compiler/src/abstractinterpretation.jl         318 (::Compiler.var"#infercalls2#abstract_call_gf_by_type##2"{Compiler.SafeBox{Int64}, Bool, Compiler.ArgInfo, Vector{Compiler.MethodMatchTarget}})(in…
     1         1 @Compiler/src/utilities.jl                      161 get_compileable_sig
     1         0 @Compiler/src/ssair/passes.jl                  2107 adce_pass!(ir::Compiler.IRCode, inlining::Compiler.InliningState{Compiler.NativeInterpreter})
     1         1 @Compiler/src/ssair/ir.jl                      1514 process_node!(compact::Compiler.IncrementalCompact, result_idx::Int64, inst::Compiler.Instruction, idx::Int64, processed_idx::Int64, active_bb::In…
     1         0 @Compiler/src/ssair/passes.jl                  1435 sroa_pass!(ir::Compiler.IRCode, inlining::Compiler.InliningState{Compiler.NativeInterpreter})
     1         1 @Base/runtime_internals.jl                     1112 argument_datatype(t::Any)
     1         1 @Compiler/src/abstractinterpretation.jl        3674 scan_specified_partitions(query::typeof(Compiler.abstract_eval_partition_load), walk_binding_partition::typeof(Compiler.walk_binding_partition), i…
     1         0 @Compiler/src/ssair/inlining.jl                 679 batch_inline!(ir::Compiler.IRCode, todo::Vector{Pair{Int64, Any}}, propagate_inbounds::Bool, interp::Compiler.NativeInterpreter)
     1         0 @Compiler/src/ssair/ir.jl                      1154 setindex!(compact::Compiler.IncrementalCompact, v::Any, idx::Int64)
     1         1 @Compiler/src/ssair/ir.jl                      1120 setindex!
     1         1 @Compiler/src/ssair/inlining.jl                1779 ssa_substitute_op!(insert_node!::Compiler.InsertBefore{Compiler.IncrementalCompact}, subst_inst::Compiler.Instruction, val::Any, ssa_substitute::C…
     1         0 @Compiler/src/typeinfer.jl                      566 finishinfer!(me::Compiler.InferenceState, interp::Compiler.NativeInterpreter, cycleid::Int64)
     1         1 @Compiler/src/typeinfer.jl                      322 cache_result!(interp::Compiler.NativeInterpreter, result::Compiler.InferenceResult, ci::Core.CodeInstance)
     1         0 @Compiler/src/ssair/inlining.jl                 667 batch_inline!(ir::Compiler.IRCode, todo::Vector{Pair{Int64, Any}}, propagate_inbounds::Bool, interp::Compiler.NativeInterpreter)
     1         0 @Compiler/src/ssair/ir.jl                       119 compute_basic_blocks(stmts::Vector{Any})
     1         1 @Base/array.jl                                 1153 (::Base.var"#_growend!##0#_growend!##1"{Vector{Int64}, Int64, Int64, Int64, Int64, Int64, Memory{Int64}, MemoryRef{Int64}})()
     1         0 @Compiler/src/ssair/slot2ssa.jl                  52 renumber_ssa(stmt::Core.SSAValue, ssanums::Vector{Core.SSAValue}, new_ssa::Bool)
     1         0 @Compiler/src/ssair/irinterp.jl                 230 reprocess_instruction!(interp::Compiler.NativeInterpreter, inst::Compiler.Instruction, idx::Int64, bb::Int64, irsv::Compiler.IRInterpretationState)
     1         1 @Compiler/src/ssair/ir.jl                        94 compute_basic_blocks(stmts::Vector{Any})
     1         1 @Base/essentials.jl                               ? compute_live_ins(cfg::Compiler.CFG, defs::Vector{Int64}, uses::Vector{Int64})
     1         1 @Compiler/src/ssair/irinterp.jl                  56 abstract_call(interp::Compiler.NativeInterpreter, arginfo::Compiler.ArgInfo, sstate::Compiler.StatementState, irsv::Compiler.IRInterpretationState)
     1         0 @Compiler/src/optimize.jl                      1115 convert_to_ircode(ci::Core.CodeInfo, sv::Compiler.OptimizationState{Compiler.NativeInterpreter})
     1         0 @Base/expr.jl                                    66 copy_exprs(x::Any)
     1         0 @Base/expr.jl                                    41 copy(e::Expr)
     1         0 @Base/expr.jl                                    74 copy_exprargs(x::Vector{Any})
     1         0 @Compiler/src/abstractinterpretation.jl         142 abstract_call_gf_by_type(interp::Compiler.NativeInterpreter, func::Any, arginfo::Compiler.ArgInfo, si::Compiler.StmtInfo, atype::Any, sv::Compiler…
     1         1 @Compiler/src/abstractinterpretation.jl         104 CallInferenceState
     1         0 @Compiler/src/ssair/slot2ssa.jl                 850 construct_ssa!(ci::Core.CodeInfo, ir::Compiler.IRCode, sv::Compiler.OptimizationState{Compiler.NativeInterpreter}, domtree::Compiler.GenericDomTre…
     1         0 @Base/array.jl                                  540 fill
     1         0 @Base/array.jl                                  542 fill
     1         1 @Compiler/src/optimize.jl                      1048 run_passes_ipo_safe(ci::Core.CodeInfo, sv::Compiler.OptimizationState{Compiler.NativeInterpreter}, optimize_until::Nothing)
     1         1 @Compiler/src/abstractinterpretation.jl        3684 scan_specified_partitions(query::typeof(Compiler.abstract_eval_partition_load), walk_binding_partition::typeof(Compiler.walk_binding_partition), i…
     1         0 @Compiler/src/abstractinterpretation.jl        4054 update_bbstate!
     1         1 @Base/genericmemory.jl                          129 unsafe_copyto!
     1         1 @Compiler/src/abstractinterpretation.jl        3669 scan_specified_partitions(query::typeof(Compiler.abstract_eval_partition_load), walk_binding_partition::typeof(Compiler.walk_binding_partition), i…
     1         0 @Compiler/src/ssair/ir.jl                       581 is_relevant_expr
     1         0 @Base/tuple.jl                                  681 in
     1         1 @Base/tuple.jl                                  677 sym_in(x::Symbol, itr::NTuple{20, Symbol})
     1         0 @Compiler/src/ssair/inlining.jl                1072 call_sig(ir::Compiler.IRCode, stmt::Expr)
     1         0 @Compiler/src/abstractinterpretation.jl        3011 collect_argtypes(interp::Compiler.NativeInterpreter, ea::Vector{Any}, sstate::Compiler.StatementState, sv::Compiler.IRInterpretationState)
     1         0 @Compiler/src/ssair/ir.jl                      1553 process_node!(compact::Compiler.IncrementalCompact, result_idx::Int64, inst::Compiler.Instruction, idx::Int64, processed_idx::Int64, active_bb::In…
     1         0 @Base/expr.jl                                  1663 quoted
     1         1 @Base/expr.jl                                  1658 is_self_quoting
     1         0 @Base/expr.jl                                    87 copy(c::Core.CodeInfo)
     1         0 @Base/array.jl                                  354 copy
     1         0 @Compiler/src/abstractinterpretation.jl        4213 typeinf_local(interp::Compiler.NativeInterpreter, frame::Compiler.InferenceState, nextresult::Compiler.CurrentState)
     1         0 @Compiler/src/typeinfer.jl                      491 finishinfer!(me::Compiler.InferenceState, interp::Compiler.NativeInterpreter, cycleid::Int64)
     1         1 @Compiler/src/typeinfer.jl                      401 adjust_effects(sv::Compiler.InferenceState)
     1         1 @Compiler/src/ssair/slot2ssa.jl                  50 renumber_ssa(stmt::Core.SSAValue, ssanums::Vector{Core.SSAValue}, new_ssa::Bool)
     1         0 @Compiler/src/utilities.jl                      243 ssamap(f::Compiler.var"#renumber_ssa!##0#renumber_ssa!##1"{Vector{Core.SSAValue}, Bool}, stmt::Any)
     1         1 @Compiler/src/typeutils.jl                      103 valid_as_lattice(x::Any, astag::Bool)
     1         1 @Compiler/src/typeinfer.jl                      982 (::Compiler.var"#get_infer_result#typeinf_edge##0"{Method, Bool, Bool, Compiler.InferenceResult, Compiler.Future{Compiler.MethodCallResult}})(inte…
     1         0 @Compiler/src/inferencestate.jl                 877 Compiler.IRInterpretationState(interp::Compiler.NativeInterpreter, codeinst::Core.CodeInstance, mi::Core.MethodInstance, argtypes::Vector{Any}, wo…
     1         0 @Compiler/src/ssair/legacy.jl                    59 inflate_ir
     1         0 @Compiler/src/ssair/legacy.jl                    44 inflate_ir!(ci::Core.CodeInfo, sptypes::Vector{Compiler.VarState}, argtypes::Vector{Any})
     1         0 @Compiler/src/ssair/ir.jl                       202 Compiler.DebugInfoStream(def::Nothing, di::Core.DebugInfo, nstmts::Int64)
     1         1 @Base/boot.jl                                   593 memoryref
     1         0 @Compiler/src/ssair/inlining.jl                 339 ir_prepare_inlining!(insert_node!::Compiler.InsertHere, inline_target::Compiler.IncrementalCompact, ir::Compiler.IRCode, spec_info::Compiler.SpecI…
     1         0 @Compiler/src/ssair/inlining.jl                 464 fix_va_argexprs!(insert_node!::Compiler.InsertHere, inline_target::Compiler.IncrementalCompact, argexprs::Vector{Any}, nargs_def::Int64, line_idx:…
     1         0 @Compiler/src/ssair/ir.jl                      2166 InsertHere
     1         0 @Compiler/src/ssair/ir.jl                      1051 insert_node_here!
     1         0 @Compiler/src/ssair/ir.jl                      1060 insert_node_here!(compact::Compiler.IncrementalCompact, newinst::Compiler.NewInstruction, reverse_affinity::Bool)
     1         0 @Compiler/src/ssair/ir.jl                       972 recompute_newinst_flag
     1         0 @Compiler/src/tfuncs.jl                        2611 builtin_effects(𝕃::Compiler.InferenceLattice{Compiler.ConditionalsLattice{Compiler.PartialsLattice{Compiler.ConstsLattice}}}, f::Core.Builtin, arg…
     1         1 @Compiler/src/utilities.jl                       17 contains_is
     1         0 @Compiler/src/ssair/inlining.jl                1290 process_simple!(todo::Vector{Pair{Int64, Any}}, ir::Compiler.IRCode, idx::Int64, flag::UInt32, state::Compiler.InliningState{Compiler.NativeInterp…
     1         0 @Compiler/src/optimize.jl                       337 stmt_effect_flags(𝕃ₒ::Compiler.PartialsLattice{Compiler.ConstsLattice}, stmt::Any, rt::Any, src::Compiler.IRCode)
     1         0 @Compiler/src/ssair/inlining.jl                1455 semiconcrete_result_item(result::Compiler.SemiConcreteResult, info::Compiler.CallInfo, flag::UInt32, state::Compiler.InliningState{Compiler.Native…
     1         0 @Compiler/src/ssair/inlining.jl                  26 InliningTodo
     1         0 @Compiler/src/ssair/inlining.jl                1661 linear_inline_eligible
     1         1 @Base/range.jl                                    ? fixemup!(slot_filter::Compiler.var"#rename_uses!##0#rename_uses!##1", rename_slot::Compiler.var"#rename_uses!##2#rename_uses!##3"{Vector{Pair{Any,…
     1         0 @Compiler/src/ssair/ir.jl                      1114 kill_current_uses!(compact::Compiler.IncrementalCompact, stmt::Any)
     1         0 @Compiler/src/optimize.jl                      1353 statement_cost(ex::Expr, line::Int64, src::Compiler.IRCode, sptypes::Vector{Compiler.VarState}, params::Compiler.OptimizationParams)
     1         1 @Compiler/src/optimize.jl                       426 argextype(x::Any, src::Compiler.IRCode, sptypes::Vector{Compiler.VarState}, slottypes::Vector{Any})
     1         0 @Compiler/src/typeinfer.jl                      672 record_slot_assign!(sv::Compiler.InferenceState)
     1         1 @Compiler/src/ssair/inlining.jl                1230 process_simple!(todo::Vector{Pair{Int64, Any}}, ir::Compiler.IRCode, idx::Int64, flag::UInt32, state::Compiler.InliningState{Compiler.NativeInterp…
     1         0 @Compiler/src/tfuncs.jl                        1581 fieldtype_tfunc(𝕃::Compiler.InferenceLattice{Compiler.ConditionalsLattice{Compiler.PartialsLattice{Compiler.ConstsLattice}}}, s0::Any, name::Any)
     1         0 @Compiler/src/tfuncs.jl                        1646 _fieldtype_tfunc(𝕃::Compiler.InferenceLattice{Compiler.ConditionalsLattice{Compiler.PartialsLattice{Compiler.ConstsLattice}}}, s::Any, name::Any, …
     1         0 @Compiler/src/abstractlattice.jl                307 tmerge
     1         0 @Compiler/src/typelimits.jl                     490 tmerge
     1         0 @Compiler/src/typelimits.jl                     530 tmerge(lattice::Compiler.ConditionalsLattice{Compiler.PartialsLattice{Compiler.ConstsLattice}}, typea::Any, typeb::Any)
     1         0 @Compiler/src/typelimits.jl                     647 tmerge(lattice::Compiler.PartialsLattice{Compiler.ConstsLattice}, typea::Any, typeb::Any)
     1         0 @Compiler/src/typelimits.jl                     405 tmerge_fast_path
     1         1 @Compiler/src/ssair/ir.jl                      2136 compact!(code::Compiler.IRCode, allow_cfg_transforms::Bool)
     1         1 @Compiler/src/abstractinterpretation.jl        3871 abstract_eval_basic_statement
     1         0 @Compiler/src/abstractinterpretation.jl        4187 typeinf_local(interp::Compiler.NativeInterpreter, frame::Compiler.InferenceState, nextresult::Compiler.CurrentState)
     1         0 @Compiler/src/abstractinterpretation.jl        3842 abstract_eval_basic_statement
     1         0 @Compiler/src/typeinfer.jl                      126 finish!(interp::Compiler.NativeInterpreter, caller::Compiler.InferenceState, validation_world::UInt64, time_before::UInt64)
     1         1 @Base/array.jl                                 1486 resize!(a::Vector{Symbol}, nl_::Int64)
     1         0 @Compiler/src/ssair/inlining.jl                 414 ir_inline_item!(compact::Compiler.IncrementalCompact, idx::Int64, argexprs::Vector{Any}, item::Compiler.InliningTodo, boundscheck::Symbol, todo_bb…
     1         1 @Compiler/src/ssair/ir.jl                      1103 kill_current_use!(compact::Compiler.IncrementalCompact, val::Any)
     1         1 @Compiler/src/abstractinterpretation.jl        2691 abstract_call_known(interp::Compiler.NativeInterpreter, f::Any, arginfo::Compiler.ArgInfo, si::Compiler.StmtInfo, sv::Compiler.InferenceState, max…
     1         0 @Compiler/src/abstractinterpretation.jl        1048 maybe_get_const_prop_profitable(interp::Compiler.NativeInterpreter, result::Compiler.MethodCallResult, f::Any, arginfo::Compiler.ArgInfo, si::Comp…
     1         0 @Compiler/src/abstractinterpretation.jl        1155 is_all_overridden(interp::Compiler.NativeInterpreter, ::Compiler.ArgInfo, sv::Compiler.InferenceState)
     1         0 @Compiler/src/abstractlattice.jl                239 is_forwardable_argtype
     1         1 @Compiler/src/abstractlattice.jl                242 is_forwardable_argtype
     1         1 @Base/tuple.jl                                  673 sym_in(x::Symbol, itr::NTuple{20, Symbol})
     1         0 @Compiler/src/abstractinterpretation.jl        3003 abstract_eval_value
     1         0 @Compiler/src/abstractinterpretation.jl        3729 abstract_eval_globalref
     1         1 @Compiler/src/inferencestate.jl                 989 update_valid_age!
     1         1 @Compiler/src/ssair/ir.jl                       662 iterate
     1         0 @Compiler/src/ssair/ir.jl                       794 Compiler.IncrementalCompact(parent::Compiler.IncrementalCompact, code::Compiler.IRCode, result_offset::Int64)
     1         0 @Compiler/src/ssair/inlining.jl                 637 batch_inline!(ir::Compiler.IRCode, todo::Vector{Pair{Int64, Any}}, propagate_inbounds::Bool, interp::Compiler.NativeInterpreter)
     1         0 @Compiler/src/ssair/ir.jl                       780 Compiler.IncrementalCompact(code::Compiler.IRCode, cfg_transform::Compiler.CFGTransformState)
     1         0 @Compiler/src/ssair/inlining.jl                1069 call_sig(ir::Compiler.IRCode, stmt::Expr)
     1         1 @Compiler/src/types.jl                          507 add_edges!(edges::Vector{Any}, info::Compiler.CallInfo)
     1         0 @Compiler/src/ssair/inlining.jl                 327 ir_prepare_inlining!(insert_node!::Compiler.InsertHere, inline_target::Compiler.IncrementalCompact, ir::Compiler.IRCode, spec_info::Compiler.SpecI…
     1         0 @Compiler/src/inferencestate.jl                 578 should_insert_coverage
     1         0 @Compiler/src/inferencestate.jl                 581 should_instrument
     1         0 @Compiler/src/utilities.jl                      332 instrumentation_enabled(m::Module, only_if_affects_optimizer::Bool)
     1         0 @Base/options.jl                                 75 JLOptions
     1         1 @Base/pointer.jl                                151 unsafe_load
     1         0 @Compiler/src/ssair/passes.jl                  1259 sroa_pass!(ir::Compiler.IRCode, inlining::Compiler.InliningState{Compiler.NativeInterpreter})
     1         0 @Compiler/src/ssair/ir.jl                       779 Compiler.IncrementalCompact(code::Compiler.IRCode, cfg_transform::Compiler.CFGTransformState)
     1         0 @Compiler/src/abstractinterpretation.jl        4327 typeinf_local(interp::Compiler.NativeInterpreter, frame::Compiler.InferenceState, nextresult::Compiler.CurrentState)
     1         0 @Base/bitset.jl                                 105 _setint!
     1         0 @Base/bitset.jl                                 128 _growbeg0!
     1         1 @Base/range.jl                                  921 iterate
     1         0 @Compiler/src/inferencestate.jl                1156 setindex!(f::Compiler.Future{Compiler.RTEffects}, v::Compiler.RTEffects)
     1         0 @Base/refvalue.jl                                60 setindex!
     1         1 @Base/Base_compiler.jl                           58 setproperty!
     2         2 @Base/stat.jl                                   198 stat(path::String)
     2         2 @Base/stat.jl                                    60 StatStruct
     2         0 @Base/genericmemory.jl                          208 fill!
     2         2 @Base/cmem.jl                                    42 memset
     2         0 @Base/operators.jl                              425 >
     2         2 @Base/int.jl                                     83 <
     2         0 @Base/pointer.jl                                 73 unsafe_convert
     2         2 @Base/pointer.jl                                 30 convert
     2         0 @MMSB/src/ffi/FFIWrapper.jl                     325 rust_delta_source(handle::Main.MMSBBenchmarks.MMSB.FFIWrapper.RustDeltaHandle)
     2         0 @Base/logging/ConsoleLogger.jl                  182 handle_message(logger::Base.CoreLogging.ConsoleLogger, level::Base.CoreLogging.LogLevel, message::Any, _module::Any, group::Any, id::Any, filepath…
     2         2 @Base/util.jl                                    78 with_output_color(::Function, ::Symbol, ::IOContext{IOBuffer}, ::String, ::Vararg{String}; bold::Bool, italic::Bool, underline::Bool, blink::Bool,…
     2         0 @Base/strings/search.jl                          55 findnext(pred::Base.Fix2{typeof(isequal), Char}, s::String, i::Int64)
     2         2 @Base/strings/search.jl                          93 _search
     2         0 @Base/array.jl                                  986 setindex!
     2         1 @Base/array.jl                                  991 _setindex!
     2         0 @Base/logging/ConsoleLogger.jl                  180 handle_message(logger::Base.CoreLogging.ConsoleLogger, level::Base.CoreLogging.LogLevel, message::Any, _module::Any, group::Any, id::Any, filepath…
     2         0 @Base/logging/ConsoleLogger.jl                  150 handle_message(logger::Base.CoreLogging.ConsoleLogger, level::Base.CoreLogging.LogLevel, message::Any, _module::Any, group::Any, id::Any, filepath…
     2         0 @MMSB/src/02_semiring/DeltaRouter.jl             61 create_delta(state::Main.MMSBBenchmarks.MMSB.MMSBStateTypes.MMSBState, page_id::UInt64, mask::Vector{Bool}, data::Vector{UInt8}; source::Symbol, i…
     2         0 @MMSB/src/ffi/FFIWrapper.jl                      98 rust_page_epoch
     2         2 @Base/logging/ConsoleLogger.jl                  166 handle_message(logger::Base.CoreLogging.ConsoleLogger, level::Base.CoreLogging.LogLevel, message::Any, _module::Any, group::Any, id::Any, filepath…
     2         0 @Compiler/src/tfuncs.jl                        2614 builtin_effects(𝕃::Compiler.InferenceLattice{Compiler.ConditionalsLattice{Compiler.PartialsLattice{Compiler.ConstsLattice}}}, f::Core.Builtin, arg…
     2         0 @Compiler/src/inferencestate.jl                 872 Compiler.IRInterpretationState(interp::Compiler.NativeInterpreter, codeinst::Core.CodeInstance, mi::Core.MethodInstance, argtypes::Vector{Any}, wo…
     2         0 @Compiler/src/ssair/passes.jl                  1483 sroa_pass!(ir::Compiler.IRCode, inlining::Compiler.InliningState{Compiler.NativeInterpreter})
     2         0 @Compiler/src/ssair/ir.jl                      2137 compact!(code::Compiler.IRCode, allow_cfg_transforms::Bool)
     2         0 @Base/boot.jl                                   670 Array
     2         0 @Compiler/src/ssair/ir.jl                       690 insert_node!
     2         0 @Compiler/src/ssair/ir.jl                       684 insert_node!(ir::Compiler.IRCode, pos::Core.SSAValue, newinst::Compiler.NewInstruction, attach_after::Bool)
     2         0 @Compiler/src/ssair/ir.jl                       370 add_inst!
     2         0 @Compiler/src/ssair/ir.jl                       310 Instruction
     2         0 @Compiler/src/ssair/ir.jl                       280 add_new_idx!
     2         0 @Base/array.jl                                 1479 resize!(a::Vector{UInt32}, nl_::Int64)
     2         0 @Compiler/src/ssair/slot2ssa.jl                 568 construct_ssa!(ci::Core.CodeInfo, ir::Compiler.IRCode, sv::Compiler.OptimizationState{Compiler.NativeInterpreter}, domtree::Compiler.GenericDomTre…
     2         0 @Compiler/src/ssair/ir.jl                       339 setindex!
     2         2 @Base/essentials.jl                             919 getindex
     2         0 @Compiler/src/optimize.jl                       239 ir_to_codeinf!
     2         2 @Compiler/src/optimize.jl                       247 widen_all_consts!(src::Core.CodeInfo)
     2         0 @Compiler/src/ssair/ir.jl                       736 CFGTransformState!(blocks::Vector{Compiler.BasicBlock}, allow_cfg_transforms::Bool)
     2         0 @Compiler/src/ssair/domtree.jl                  266 compute_domtree_nodes!(domtree::Compiler.GenericDomTree{false})
     2         0 @Compiler/src/ssair/ir.jl                        37 block_for_inst
     2         0 @Compiler/src/sort.jl                            68 searchsortedfirst
     2         0 @Compiler/src/sort.jl                            68 #searchsortedfirst#1
     2         0 @Compiler/src/sort.jl                            66 searchsortedfirst
     2         2 @Base/int.jl                                      ? searchsortedfirst
     2         0 @Compiler/src/ssair/slot2ssa.jl                 149 fixemup!(slot_filter::Compiler.var"#rename_uses!##0#rename_uses!##1", rename_slot::Compiler.var"#rename_uses!##2#rename_uses!##3"{Vector{Pair{Any,…
     2         0 @Compiler/src/ssair/slot2ssa.jl                 837 construct_ssa!(ci::Core.CodeInfo, ir::Compiler.IRCode, sv::Compiler.OptimizationState{Compiler.NativeInterpreter}, domtree::Compiler.GenericDomTre…
     2         2 @Base/essentials.jl                               ? _useref_setindex!(stmt::Any, op::Int64, v::Any)
     2         0 @Base/runtime_internals.jl                     1595 hasintersect
     2         2 @Base/runtime_internals.jl                     1007 typeintersect
     2         0 @Compiler/src/optimize.jl                       514 finish(interp::Compiler.NativeInterpreter, opt::Compiler.OptimizationState{Compiler.NativeInterpreter}, ir::Compiler.IRCode, caller::Compiler.Infe…
     2         0 @Compiler/src/ssair/passes.jl                  1486 sroa_pass!(ir::Compiler.IRCode, inlining::Compiler.InliningState{Compiler.NativeInterpreter})
     2         0 @Compiler/src/ssair/passes.jl                  1332 sroa_pass!(ir::Compiler.IRCode, inlining::Compiler.InliningState{Compiler.NativeInterpreter})
     2         0 @Compiler/src/ssair/inlining.jl                 665 batch_inline!(ir::Compiler.IRCode, todo::Vector{Pair{Int64, Any}}, propagate_inbounds::Bool, interp::Compiler.NativeInterpreter)
     2         0 @Compiler/src/ssair/ir.jl                      1882 iterate_compact(compact::Compiler.IncrementalCompact)
     2         0 @Compiler/src/typeutils.jl                      381 is_mutation_free_type
     2         0 @Base/runtime_internals.jl                      878 ismutationfree
     2         0 @Base/runtime_internals.jl                      864 datatype_ismutationfree
     2         0 @Compiler/src/abstractinterpretation.jl        3672 scan_specified_partitions(query::typeof(Compiler.abstract_eval_partition_load), walk_binding_partition::typeof(Compiler.walk_binding_partition), i…
     2         0 @Base/runtime_internals.jl                      240 convert
     2         0 @Compiler/src/ssair/inlining.jl                 966 retrieve_ir_for_inlining
     2         0 @Base/expr.jl                                    74 copy_exprargs
     2         0 @Base/runtime_internals.jl                     1591 specialize_method
     2         0 @Base/runtime_internals.jl                     1592 #specialize_method#9
     2         2 @Base/runtime_internals.jl                     1586 #specialize_method#8
     2         0 @Compiler/src/ssair/inlining.jl                1246 process_simple!(todo::Vector{Pair{Int64, Any}}, ir::Compiler.IRCode, idx::Int64, flag::UInt32, state::Compiler.InliningState{Compiler.NativeInterp…
     2         0 @Compiler/src/optimize.jl                       346 stmt_effect_flags(𝕃ₒ::Compiler.PartialsLattice{Compiler.ConstsLattice}, stmt::Any, rt::Any, src::Compiler.IRCode)
     2         0 @Compiler/src/optimize.jl                       270 new_expr_effect_flags
     2         0 @Compiler/src/optimize.jl                       273 new_expr_effect_flags(𝕃ₒ::Compiler.PartialsLattice{Compiler.ConstsLattice}, args::Vector{Any}, src::Compiler.IRCode, pattern_match::Nothing)
     2         1 @Compiler/src/abstractinterpretation.jl        3055 abstract_eval_call
     2         0 @Compiler/src/abstractinterpretation.jl        3013 collect_argtypes(interp::Compiler.NativeInterpreter, ea::Vector{Any}, sstate::Compiler.StatementState, sv::Compiler.InferenceState)
     2         2 @Compiler/src/tfuncs.jl                        2771 builtin_tfunction(interp::Compiler.NativeInterpreter, f::Any, argtypes::Vector{Any}, sv::Compiler.IRInterpretationState)
     2         0 @Compiler/src/ssair/ir.jl                       268 Compiler.InstructionStream(len::Int64)
     2         0 @Compiler/src/tfuncs.jl                        2827 builtin_tfunction(interp::Compiler.NativeInterpreter, f::Any, argtypes::Vector{Any}, sv::Compiler.InferenceState)
     2         2 @Base/tuple.jl                                   33 getindex
     2         2 @Compiler/src/ssair/ir.jl                      1343 renumber_ssa2!(stmt::Any, ssanums::Vector{Any}, used_ssas::Vector{Int64}, new_new_used_ssas::Vector{Int64}, late_fixup::Vector{Int64}, result_idx:…
     2         2 @Compiler/src/ssair/ir.jl                       650 _advance(stmt::Any, op::Int64)
     2         1 @Compiler/src/ssair/ir.jl                      1632 process_node!(compact::Compiler.IncrementalCompact, result_idx::Int64, inst::Compiler.Instruction, idx::Int64, processed_idx::Int64, active_bb::In…
     2         0 @Compiler/src/inferencestate.jl                 338 Compiler.InferenceState(result::Compiler.InferenceResult, src::Core.CodeInfo, cache_mode::UInt8, interp::Compiler.NativeInterpreter)
     2         0 @Compiler/src/typeinfer.jl                      736 type_annotate!(interp::Compiler.NativeInterpreter, sv::Compiler.InferenceState)
     2         2 @Compiler/src/abstractinterpretation.jl         126 abstract_call_gf_by_type(interp::Compiler.NativeInterpreter, func::Any, arginfo::Compiler.ArgInfo, si::Compiler.StmtInfo, atype::Any, sv::Compiler…
     2         0 @Compiler/src/inferenceresult.jl                  5 matching_cache_argtypes
     2         0 @Compiler/src/ssair/passes.jl                  1402 sroa_pass!(ir::Compiler.IRCode, inlining::Compiler.InliningState{Compiler.NativeInterpreter})
     2         0 @Compiler/src/abstractinterpretation.jl        2663 abstract_call_known(interp::Compiler.NativeInterpreter, f::Any, arginfo::Compiler.ArgInfo, si::Compiler.StmtInfo, sv::Compiler.InferenceState, max…
     2         0 @Compiler/src/abstractinterpretation.jl        2421 abstract_eval_getglobal(interp::Compiler.NativeInterpreter, sv::Compiler.InferenceState, saw_latestworld::Bool, argtypes::Vector{Any})
     2         0 @Compiler/src/abstractinterpretation.jl        2393 abstract_eval_getglobal(interp::Compiler.NativeInterpreter, sv::Compiler.InferenceState, saw_latestworld::Bool, M::Any, s::Any)
     2         2 @Compiler/src/ssair/ir.jl                      1863 iterate
     2         2 @Compiler/src/abstractinterpretation.jl        3385 abstract_eval_statement_expr(interp::Compiler.NativeInterpreter, e::Expr, sstate::Compiler.StatementState, sv::Compiler.InferenceState)
     2         0 @Compiler/src/abstractinterpretation.jl        1039 maybe_get_const_prop_profitable(interp::Compiler.NativeInterpreter, result::Compiler.MethodCallResult, f::Any, arginfo::Compiler.ArgInfo, si::Comp…
     2         2 @Compiler/src/abstractinterpretation.jl        1162 force_const_prop(interp::Compiler.NativeInterpreter, f::Any, method::Method)
     2         0 @Compiler/src/ssair/slot2ssa.jl                 473 domsort_ssa!(ir::Compiler.IRCode, domtree::Compiler.GenericDomTree{false})
     2         2 @Compiler/src/typelattice.jl                    514 ⊑(lattice::Compiler.ConstsLattice, a::Any, b::Any)
     2         0 @Base/bitset.jl                                  33 BitSet
     2         2 @Base/essentials.jl                               ? ssa_substitute_op!(insert_node!::Compiler.InsertBefore{Compiler.IncrementalCompact}, subst_inst::Compiler.Instruction, val::Any, ssa_substitute::C…
     2         0 @Compiler/src/abstractinterpretation.jl         331 (::Compiler.var"#infercalls#abstract_call_gf_by_type##0"{Compiler.ArgInfo, Compiler.StmtInfo, Compiler.CallInferenceState, Compiler.Future{Compile…
     2         0 @Base/bitset.jl                                  19 BitSet
     2         2 @Compiler/src/abstractinterpretation.jl        2644 abstract_call_known(interp::Compiler.NativeInterpreter, f::Any, arginfo::Compiler.ArgInfo, si::Compiler.StmtInfo, sv::Compiler.IRInterpretationSta…
     2         2 @Compiler/src/abstractinterpretation.jl           ? scan_specified_partitions(query::typeof(Compiler.abstract_eval_partition_load), walk_binding_partition::typeof(Compiler.walk_binding_partition), i…
     2         0 @Compiler/src/ssair/slot2ssa.jl                 892 construct_ssa!(ci::Core.CodeInfo, ir::Compiler.IRCode, sv::Compiler.OptimizationState{Compiler.NativeInterpreter}, domtree::Compiler.GenericDomTre…
     2         0 @Compiler/src/utilities.jl                      247 ssamap(f::Compiler.var"#renumber_ssa!##0#renumber_ssa!##1"{Vector{Core.SSAValue}, Bool}, stmt::Any)
     2         2 @Compiler/src/ssair/ir.jl                         ? _advance(stmt::Any, op::Int64)
     2         0 @Compiler/src/ssair/inlining.jl                 326 ir_prepare_inlining!(insert_node!::Compiler.InsertHere, inline_target::Compiler.IncrementalCompact, ir::Compiler.IRCode, spec_info::Compiler.SpecI…
     2         2 @Compiler/src/ssair/inlining.jl                 313 ir_inline_linetable!
     2         0 @Compiler/src/ssair/inlining.jl                 683 batch_inline!(ir::Compiler.IRCode, todo::Vector{Pair{Int64, Any}}, propagate_inbounds::Bool, interp::Compiler.NativeInterpreter)
     2         0 @Compiler/src/ssair/passes.jl                  1421 sroa_pass!(ir::Compiler.IRCode, inlining::Compiler.InliningState{Compiler.NativeInterpreter})
     2         2 @Compiler/src/ssair/inlining.jl                1366 compute_inlining_cases(info::Compiler.CallInfo, flag::UInt32, sig::Compiler.Signature, state::Compiler.InliningState{Compiler.NativeInterpreter})
     2         0 @Compiler/src/stmtinfo.jl                       202 add_edges_impl(edges::Vector{Any}, info::Compiler.ConstCallInfo)
     2         0 @Compiler/src/types.jl                          511 add_edges!
     2         0 @Compiler/src/stmtinfo.jl                        47 add_edges_impl
     2         0 @Compiler/src/stmtinfo.jl                        49 _add_edges_impl
     2         0 @Compiler/src/inferencestate.jl                1183 (::Compiler.var"#72#73"{Compiler.var"#abstract_eval_call##0#abstract_eval_call##1", Compiler.Future{Compiler.RTEffects}, Base.RefValue{Compiler.Ca…
     2         1 @Compiler/src/abstractinterpretation.jl        3609 walk_binding_partition(imported_binding::Core.Binding, partition::Core.BindingPartition, world::UInt64)
     2         0 @Compiler/src/ssair/ir.jl                       707 CFGTransformState!(blocks::Vector{Compiler.BasicBlock}, allow_cfg_transforms::Bool)
     2         0 @Compiler/src/abstractinterpretation.jl        3819 abstract_eval_basic_statement
     2         0 @Base/array.jl                                 1296 push!
     2         1 @Compiler/src/ssair/ir.jl                       644 userefs
     2         0 @Compiler/src/abstractinterpretation.jl        4392 typeinf_local(interp::Compiler.NativeInterpreter, frame::Compiler.InferenceState, nextresult::Compiler.CurrentState)
     2         2 @Compiler/src/ssair/ir.jl                      1339 renumber_ssa2!(stmt::Any, ssanums::Vector{Any}, used_ssas::Vector{Int64}, new_new_used_ssas::Vector{Int64}, late_fixup::Vector{Int64}, result_idx:…
     2         0 @Compiler/src/ssair/inlining.jl                 630 batch_inline!(ir::Compiler.IRCode, todo::Vector{Pair{Int64, Any}}, propagate_inbounds::Bool, interp::Compiler.NativeInterpreter)
     2         0 @Base/array.jl                                 1357 append!
     2         0 @Base/array.jl                                  355 copy
     2         0 @Compiler/src/optimize.jl                       842 scan_inconsistency!(inst::Compiler.Instruction, sv::Compiler.PostOptAnalysisState)
     2         0 @Compiler/src/ssair/slot2ssa.jl                 236 iterated_dominance_frontier(cfg::Compiler.CFG, liveness::Compiler.BlockLiveness, domtree::Compiler.GenericDomTree{false})
     2         0 @Compiler/src/ssair/inlining.jl                1264 process_simple!(todo::Vector{Pair{Int64, Any}}, ir::Compiler.IRCode, idx::Int64, flag::UInt32, state::Compiler.InliningState{Compiler.NativeInterp…
     2         0 @Compiler/src/optimize.jl                       338 stmt_effect_flags(𝕃ₒ::Compiler.PartialsLattice{Compiler.ConstsLattice}, stmt::Any, rt::Any, src::Compiler.IRCode)
     2         0 @Compiler/src/tfuncs.jl                        2667 builtin_effects(𝕃::Compiler.PartialsLattice{Compiler.ConstsLattice}, f::Core.Builtin, argtypes::Vector{Any}, rt::Any)
     2         0 @Compiler/src/tfuncs.jl                        2756 builtin_nothrow
     2         2 @Compiler/src/ssair/ir.jl                         ? _useref_getindex(stmt::Any, op::Int64)
     2         0 @Compiler/src/ssair/inlining.jl                1254 process_simple!(todo::Vector{Pair{Int64, Any}}, ir::Compiler.IRCode, idx::Int64, flag::UInt32, state::Compiler.InliningState{Compiler.NativeInterp…
     2         0 @Compiler/src/ssair/inlining.jl                1454 semiconcrete_result_item(result::Compiler.SemiConcreteResult, info::Compiler.CallInfo, flag::UInt32, state::Compiler.InliningState{Compiler.Native…
     2         0 @Compiler/src/ssair/inlining.jl                 979 retrieve_ir_for_inlining
     2         2 @Compiler/src/ssair/ir.jl                       207 DebugInfo
     2         0 @Compiler/src/ssair/inlining.jl                 375 ir_inline_item!(compact::Compiler.IncrementalCompact, idx::Int64, argexprs::Vector{Any}, item::Compiler.InliningTodo, boundscheck::Symbol, todo_bb…
     2         0 @Compiler/src/ssair/inlining.jl                 381 ir_inline_item!(compact::Compiler.IncrementalCompact, idx::Int64, argexprs::Vector{Any}, item::Compiler.InliningTodo, boundscheck::Symbol, todo_bb…
     2         0 @Compiler/src/utilities.jl                      245 ssamap(f::Compiler.var"#renumber_ssa!##0#renumber_ssa!##1"{Vector{Core.SSAValue}, Bool}, stmt::Any)
     2         0 @Compiler/src/ssair/slot2ssa.jl                  62 #renumber_ssa!##0
     2         2 @Compiler/src/optimize.jl                       423 argextype(x::Any, src::Compiler.IRCode, sptypes::Vector{Compiler.VarState}, slottypes::Vector{Any})
     2         0 @Compiler/src/abstractinterpretation.jl        3055 abstract_eval_call(interp::Compiler.NativeInterpreter, e::Expr, sstate::Compiler.StatementState, sv::Compiler.IRInterpretationState)
     3         0 @Base/boot.jl                                   690 Symbol(s::String)
     3         3 @Base/boot.jl                                   683 _Symbol
     3         0 @Base/operators.jl                              321 !=
     3         0 @Base/strings/cstring.jl                         85 unsafe_convert
     3         3 @Base/strings/cstring.jl                         78 containsnul
     3         3 @Base/namedtuple.jl                             346 merge
     3         0 @Base/stream.jl                                1069 uv_write(s::Base.TTY, p::Ptr{UInt8}, n::UInt64)
     3         3 @Base/stream.jl                                1114 uv_write_async(s::Base.TTY, p::Ptr{UInt8}, n::UInt64)
     3         0 @MMSB/src/01_page/Delta.jl                       28 Main.MMSBBenchmarks.MMSB.DeltaTypes.Delta(id::UInt64, page_id::UInt64, epoch::UInt32, mask::Vector{UInt8}, payload::Vector{UInt8}, source::Symbol;…
     3         3 @Base/Base_compiler.jl                           54 getproperty
     3         0 @Base/multimedia.jl                              47 show
     3         0 @Base/show.jl                                  1472 show
     3         0 @Base/show.jl                                  1439 show_delim_array
     3         1 @Base/show.jl                                  1454 show_delim_array(io::IOContext{IOBuffer}, itr::Tuple{UInt64}, op::Char, delim::Char, cl::Char, delim_one::Bool, i1::Int64, n::Int64)
     3         3 @Base/boot.jl                                   499 Box
     3         0 @Compiler/src/abstractinterpretation.jl        2685 abstract_call_known(interp::Compiler.NativeInterpreter, f::Any, arginfo::Compiler.ArgInfo, si::Compiler.StmtInfo, sv::Compiler.IRInterpretationSta…
     3         0 @Compiler/src/abstractinterpretation.jl        1262 semi_concrete_eval_call(interp::Compiler.NativeInterpreter, mi::Core.MethodInstance, result::Compiler.MethodCallResult, arginfo::Compiler.ArgInfo,…
     3         0 @Compiler/src/ssair/passes.jl                   190 collect_leaves
     3         0 @Compiler/src/optimize.jl                       230 ir_to_codeinf!
     3         0 @Compiler/src/ssair/domtree.jl                  260 update_domtree!
     3         3 @Base/int.jl                                     87 +
     3         0 @Compiler/src/ssair/inlining.jl                 385 ir_inline_item!(compact::Compiler.IncrementalCompact, idx::Int64, argexprs::Vector{Any}, item::Compiler.InliningTodo, boundscheck::Symbol, todo_bb…
     3         0 @Compiler/src/ssair/ir.jl                       639 setindex!
     3         0 @Compiler/src/typeinfer.jl                      721 type_annotate!(interp::Compiler.NativeInterpreter, sv::Compiler.InferenceState)
     3         3 @Compiler/src/typelattice.jl                    688 widenconst(c::Core.Const)
     3         0 @Compiler/src/ssair/inlining.jl                1064 call_sig(ir::Compiler.IRCode, stmt::Expr)
     3         0 @Compiler/src/optimize.jl                      1304 slot2reg
     3         0 @Compiler/src/ssair/slot2ssa.jl                  45 scan_slot_def_use(nargs::Int64, ci::Core.CodeInfo, code::Vector{Any})
     3         0 @Compiler/src/ssair/inlining.jl                1306 handle_any_const_result!(cases::Vector{Compiler.InliningCase}, result::Any, match::Core.MethodMatch, argtypes::Vector{Any}, info::Compiler.CallInf…
     3         0 @Compiler/src/ssair/inlining.jl                1423 handle_const_prop_result!
     3         0 @Compiler/src/ssair/inlining.jl                1430 handle_const_prop_result!(cases::Vector{Compiler.InliningCase}, result::Compiler.ConstPropResult, match::Core.MethodMatch, info::Compiler.CallInfo…
     3         0 @Base/runtime_internals.jl                     1575 specialize_method
     3         2 @Compiler/src/tfuncs.jl                         105 instanceof_tfunc(t::Any, astag::Bool)
     3         3 @Base/expr.jl                                    38 isexpr
     3         0 @Compiler/src/ssair/ir.jl                      2140 compact!(code::Compiler.IRCode, allow_cfg_transforms::Bool)
     3         0 @Compiler/src/ssair/ir.jl                      2110 finish
     3         0 @Compiler/src/ssair/passes.jl                  1324 sroa_pass!(ir::Compiler.IRCode, inlining::Compiler.InliningState{Compiler.NativeInterpreter})
     3         0 @Compiler/src/abstractinterpretation.jl        3673 scan_specified_partitions(query::typeof(Compiler.abstract_eval_partition_load), walk_binding_partition::typeof(Compiler.walk_binding_partition), i…
     3         0 @Compiler/src/cicache.jl                         35 max_world
     3         3 @Compiler/src/cicache.jl                         32 last
     3         0 @Compiler/src/ssair/passes.jl                  1318 sroa_pass!(ir::Compiler.IRCode, inlining::Compiler.InliningState{Compiler.NativeInterpreter})
     3         0 @Compiler/src/typeinfer.jl                      652 compute_edges!(sv::Compiler.InferenceState)
     3         1 @Compiler/src/types.jl                          511 add_edges!(edges::Vector{Any}, info::Compiler.CallInfo)
     3         0 @Compiler/src/optimize.jl                       979 ipo_dataflow_analysis!(interp::Compiler.NativeInterpreter, opt::Compiler.OptimizationState{Compiler.NativeInterpreter}, ir::Compiler.IRCode, resul…
     3         0 @Compiler/src/optimize.jl                       869 (::Compiler.ScanStmt)(inst::Compiler.Instruction, lstmt::Int64, bb::Int64)
     3         3 @Base/essentials.jl                              11 length
     3         3 @Base/essentials.jl                             926 setindex!
     3         1 @Compiler/src/typeutils.jl                       52 (::Compiler.var"#argtypes_to_type##0#argtypes_to_type##1")(a::Any)
     3         3 @Compiler/src/typelattice.jl                    688 widenconst
     3         0 @MMSB/benchmark/benchmarks.jl                   146 _full_system_benchmark!()
     3         0 @MMSB/benchmark/benchmarks.jl                    58 _populate_pages!
     3         0 @Base/array.jl                                  790 collect(itr::Base.Generator{UnitRange{Int64}, Main.MMSBBenchmarks.var"#_populate_pages!##0#_populate_pages!##1"{Main.MMSBBenchmarks.MMSB.MMSBState…
     3         0 none                                              ? #_populate_pages!##0
     3         0 @MMSB/src/API.jl                                 68 create_page
     3         0 @MMSB/src/API.jl                                 73 create_page(state::Main.MMSBBenchmarks.MMSB.MMSBStateTypes.MMSBState; size::Int64, location::Symbol, metadata::Dict{Symbol, Any})
     3         0 @MMSB/src/00_physical/PageAllocator.jl           40 create_cpu_page!
     3         0 @MMSB/src/00_physical/PageAllocator.jl           22 create_page!(state::Main.MMSBBenchmarks.MMSB.MMSBStateTypes.MMSBState, size::Int64, location::Main.MMSBBenchmarks.MMSB.PageTypes.PageLocation)
     3         0 @MMSB/src/00_physical/PageAllocator.jl           24 #create_page!#1
     3         0 @MMSB/src/00_physical/PageAllocator.jl           25 (::Main.MMSBBenchmarks.MMSB.PageAllocator.var"#4#5"{Dict{Symbol, Any}, Main.MMSBBenchmarks.MMSB.MMSBStateTypes.MMSBState, Int64, Main.MMSBBenchmar…
     3         3 @MMSB/src/01_types/MMSBState.jl                 102 _reserve_page_id_unlocked!
     3         1 @Compiler/src/tfuncs.jl                        2831 builtin_tfunction(interp::Compiler.NativeInterpreter, f::Any, argtypes::Vector{Any}, sv::Compiler.InferenceState)
     3         0 @Base/bitset.jl                                 102 _setint!
     3         0 @Compiler/src/typeinfer.jl                      247 finish_cycle(::Compiler.NativeInterpreter, frames::Vector{Union{Compiler.IRInterpretationState, Compiler.InferenceState}}, cycleid::Int64, time_be…
     3         0 @Compiler/src/abstractinterpretation.jl        2975 abstract_eval_special_value(interp::Compiler.NativeInterpreter, e::Any, sstate::Compiler.StatementState, sv::Compiler.InferenceState)
     3         0 @Compiler/src/typeinfer.jl                      113 finish!(interp::Compiler.NativeInterpreter, caller::Compiler.InferenceState, validation_world::UInt64, time_before::UInt64)
     3         3 @Compiler/src/typeinfer.jl                      642 store_backedges(caller::Core.CodeInstance, edges::Core.SimpleVector)
     3         0 @Compiler/src/ssair/inlining.jl                 968 retrieve_ir_for_inlining
     3         0 @Compiler/src/ssair/slot2ssa.jl                 887 construct_ssa!(ci::Core.CodeInfo, ir::Compiler.IRCode, sv::Compiler.OptimizationState{Compiler.NativeInterpreter}, domtree::Compiler.GenericDomTre…
     3         0 @Compiler/src/ssair/ir.jl                      1115 kill_current_uses!(compact::Compiler.IncrementalCompact, stmt::Any)
     4         0 @Base/promotion.jl                              487 ==
     4         0 @Base/io.jl                                     837 write(s::Base.TTY, A::Vector{UInt8})
     4         0 @Base/io.jl                                     803 unsafe_write
     4         0 @Base/stream.jl                                1154 unsafe_write(s::Base.TTY, p::Ptr{UInt8}, n::UInt64)
     4         0 @Base/util.jl                                   141 printstyled
     4         0 @Base/util.jl                                   141 #printstyled#849
     4         0 @Base/util.jl                                    73 with_output_color
     4         0 @MMSB/src/02_semiring/DeltaRouter.jl             34 route_delta!(state::Main.MMSBBenchmarks.MMSB.MMSBStateTypes.MMSBState, delta::Main.MMSBBenchmarks.MMSB.DeltaTypes.Delta; propagate::Bool)
     4         0 @Base/lock.jl                                   335 lock(f::Main.MMSBBenchmarks.MMSB.MMSBStateTypes.var"#allocate_delta_id!##0#allocate_delta_id!##1"{Main.MMSBBenchmarks.MMSB.MMSBStateTypes.MMSBStat…
     4         4 @Compiler/src/abstractinterpretation.jl        3679 scan_specified_partitions(query::typeof(Compiler.abstract_eval_partition_load), walk_binding_partition::typeof(Compiler.walk_binding_partition), i…
     4         0 @Compiler/src/inferencestate.jl                 470 ComputeTryCatch
     4         0 @Compiler/src/ssair/passes.jl                  2147 adce_pass!(ir::Compiler.IRCode, inlining::Compiler.InliningState{Compiler.NativeInterpreter})
     4         0 @Compiler/src/ssair/ir.jl                       319 getindex
     4         0 @Compiler/src/typeinfer.jl                       98 finish!(interp::Compiler.NativeInterpreter, caller::Compiler.InferenceState, validation_world::UInt64, time_before::UInt64)
     4         0 @Compiler/src/methodtable.jl                    111 findall(sig::Type, table::Compiler.CachedMethodTable{Compiler.InternalMethodTable}; limit::Int64)
     4         0 @Compiler/src/ssair/domtree.jl                  242 construct_domtree(blocks::Vector{Compiler.BasicBlock})
     4         0 @Compiler/src/ssair/inlining.jl                1404 handle_call!
     4         0 @Compiler/src/ssair/inlining.jl                1836 ssa_substitute_op!(insert_node!::Compiler.InsertBefore{Compiler.IncrementalCompact}, subst_inst::Compiler.Instruction, val::Any, ssa_substitute::C…
     4         0 @Compiler/src/ssair/inlining.jl                 417 ir_inline_item!(compact::Compiler.IncrementalCompact, idx::Int64, argexprs::Vector{Any}, item::Compiler.InliningTodo, boundscheck::Symbol, todo_bb…
     4         0 @Compiler/src/ssair/ir.jl                      1341 renumber_ssa2!(stmt::Any, ssanums::Vector{Any}, used_ssas::Vector{Int64}, new_new_used_ssas::Vector{Int64}, late_fixup::Vector{Int64}, result_idx:…
     4         4 @Compiler/src/ssair/ir.jl                       655 _advance(stmt::Any, op::Int64)
     4         1 @Compiler/src/ssair/passes.jl                  1406 sroa_pass!(ir::Compiler.IRCode, inlining::Compiler.InliningState{Compiler.NativeInterpreter})
     4         0 @Compiler/src/typeutils.jl                       52 argtypes_to_type(argtypes::Vector{Any})
     4         1 @Compiler/src/utilities.jl                       24 anymap
     4         0 @Compiler/src/ssair/slot2ssa.jl                 639 construct_ssa!(ci::Core.CodeInfo, ir::Compiler.IRCode, sv::Compiler.OptimizationState{Compiler.NativeInterpreter}, domtree::Compiler.GenericDomTre…
     4         0 @Compiler/src/ssair/ir.jl                      2079 simple_dce!
     4         4 @Compiler/src/ssair/ir.jl                      2086 simple_dce!(callback::Function, compact::Compiler.IncrementalCompact)
     4         0 @Compiler/src/typeinfer.jl                      499 finishinfer!(me::Compiler.InferenceState, interp::Compiler.NativeInterpreter, cycleid::Int64)
     4         0 @Compiler/src/ssair/inlining.jl                 438 ir_inline_item!(compact::Compiler.IncrementalCompact, idx::Int64, argexprs::Vector{Any}, item::Compiler.InliningTodo, boundscheck::Symbol, todo_bb…
     4         0 @Compiler/src/optimize.jl                      1003 optimize(interp::Compiler.NativeInterpreter, opt::Compiler.OptimizationState{Compiler.NativeInterpreter}, caller::Compiler.InferenceResult)
     4         0 @Compiler/src/abstractinterpretation.jl        3680 scan_specified_partitions(query::typeof(Compiler.abstract_eval_partition_load), walk_binding_partition::typeof(Compiler.walk_binding_partition), i…
     4         0 @Compiler/src/abstractinterpretation.jl        3728 abstract_eval_globalref
     4         0 @Base/generator.jl                               48 iterate
     4         0 @Compiler/src/typelattice.jl                    374 ⊑(lattice::Compiler.InferenceLattice{Compiler.InterConditionalsLattice{Compiler.PartialsLattice{Compiler.ConstsLattice}}}, a::Any, b::Any)
     4         0 @Compiler/src/typelattice.jl                    416 ⊑
     4         0 @Base/bitset.jl                                 255 push!
     4         0 @Base/array.jl                                 1286 push!
     4         0 @Base/array.jl                                 1289 _push!
     4         0 @Compiler/src/ssair/slot2ssa.jl                 603 construct_ssa!(ci::Core.CodeInfo, ir::Compiler.IRCode, sv::Compiler.OptimizationState{Compiler.NativeInterpreter}, domtree::Compiler.GenericDomTre…
     4         0 @Compiler/src/ssair/slot2ssa.jl                 494 compute_live_ins
     4         0 @Compiler/src/optimize.jl                       528 finish(interp::Compiler.NativeInterpreter, opt::Compiler.OptimizationState{Compiler.NativeInterpreter}, ir::Compiler.IRCode, caller::Compiler.Infe…
     4         0 @Compiler/src/optimize.jl                      1452 inline_cost
     4         0 @Compiler/src/optimize.jl                      1430 statement_or_branch_cost
     4         0 @Compiler/src/ssair/slot2ssa.jl                  61 renumber_ssa!
     4         0 @Compiler/src/ssair/inlining.jl                 364 ir_inline_item!(compact::Compiler.IncrementalCompact, idx::Int64, argexprs::Vector{Any}, item::Compiler.InliningTodo, boundscheck::Symbol, todo_bb…
     4         0 @Compiler/src/abstractinterpretation.jl        4531 typeinf(interp::Compiler.NativeInterpreter, frame::Compiler.InferenceState)
     4         0 @Compiler/src/ssair/ir.jl                       543 _useref_getindex(stmt::Any, op::Int64)
     4         0 @Compiler/src/ssair/inlining.jl                1304 handle_any_const_result!(cases::Vector{Compiler.InliningCase}, result::Any, match::Core.MethodMatch, argtypes::Vector{Any}, info::Compiler.CallInf…
     4         0 @Compiler/src/ssair/inlining.jl                1462 handle_semi_concrete_result!(cases::Vector{Compiler.InliningCase}, result::Compiler.SemiConcreteResult, match::Core.MethodMatch, info::Compiler.Ca…
     4         0 @Compiler/src/ssair/ir.jl                      1154 setindex!
     4         0 @Compiler/src/ssair/ir.jl                      1122 setindex!(compact::Compiler.IncrementalCompact, v::Any, ssa::Core.SSAValue)
     5         5 @MMSB/src/ffi/FFIWrapper.jl                     341 rust_delta_mask(handle::Main.MMSBBenchmarks.MMSB.FFIWrapper.RustDeltaHandle)
     5         0 @MMSB/src/02_semiring/DeltaRouter.jl             63 create_delta(state::Main.MMSBBenchmarks.MMSB.MMSBStateTypes.MMSBState, page_id::UInt64, mask::Vector{Bool}, data::Vector{UInt8}; source::Symbol, i…
     5         0 @MMSB/src/01_page/Delta.jl                       24 Delta
     5         0 @Base/logging/ConsoleLogger.jl                  157 handle_message(logger::Base.CoreLogging.ConsoleLogger, level::Base.CoreLogging.LogLevel, message::Any, _module::Any, group::Any, id::Any, filepath…
     5         0 @Base/iterators.jl                              287 iterate
     5         5 @Base/iterators.jl                              311 keys
     5         0 @Compiler/src/abstractinterpretation.jl        1342 const_prop_call(interp::Compiler.NativeInterpreter, mi::Core.MethodInstance, result::Compiler.MethodCallResult, arginfo::Compiler.ArgInfo, sv::Com…
     5         0 @Base/array.jl                                  673 _array_for
     5         0 @Base/array.jl                                  670 _array_for
     5         0 @Base/abstractarray.jl                          866 similar
     5         0 @Base/abstractarray.jl                          867 similar
     5         0 @Compiler/src/ssair/passes.jl                  2105 adce_pass!(ir::Compiler.IRCode, inlining::Compiler.InliningState{Compiler.NativeInterpreter})
     5         0 @Compiler/src/ssair/inlining.jl                 959 retrieve_ir_for_inlining(cached_result::Core.CodeInstance, src::String)
     5         0 @Compiler/src/ssair/legacy.jl                    24 inflate_ir!(ci::Core.CodeInfo, sptypes::Vector{Compiler.VarState}, argtypes::Vector{Any})
     5         0 @Base/array.jl                                  405 getindex
     5         0 @Compiler/src/typelattice.jl                    525 ⊑(lattice::Compiler.ConstsLattice, a::Any, b::Any)
     5         5 @Compiler/src/abstractlattice.jl                153 ⊑
     5         0 @Compiler/src/typeinfer.jl                      515 finishinfer!(me::Compiler.InferenceState, interp::Compiler.NativeInterpreter, cycleid::Int64)
     5         5 @Compiler/src/ssair/ir.jl                       530 _useref_getindex(stmt::Any, op::Int64)
     5         0 @Compiler/src/ssair/inlining.jl                1250 process_simple!(todo::Vector{Pair{Int64, Any}}, ir::Compiler.IRCode, idx::Int64, flag::UInt32, state::Compiler.InliningState{Compiler.NativeInterp…
     5         0 @Compiler/src/abstractinterpretation.jl        3682 scan_specified_partitions(query::typeof(Compiler.abstract_eval_partition_load), walk_binding_partition::typeof(Compiler.walk_binding_partition), i…
     5         0 @Compiler/src/abstractinterpretation.jl        3641 abstract_eval_partition_load
     5         0 @Compiler/src/typeutils.jl                      379 is_mutation_free_argtype
     5         0 @Compiler/src/ssair/inlining.jl                 875 resolve_todo(mi::Core.MethodInstance, result::Compiler.InferenceResult, info::Compiler.CallInfo, flag::UInt32, state::Compiler.InliningState{Compi…
     5         0 @Compiler/src/ssair/inlining.jl                1215 add_inst_flag!
     5         0 @Compiler/src/ssair/inlining.jl                1218 add_inst_flag!
     5         0 @Compiler/src/ssair/ir.jl                      1353 renumber_ssa2!(stmt::Any, ssanums::Vector{Any}, used_ssas::Vector{Int64}, new_new_used_ssas::Vector{Int64}, late_fixup::Vector{Int64}, result_idx:…
     5         0 @Compiler/src/ssair/slot2ssa.jl                 894 construct_ssa!(ci::Core.CodeInfo, ir::Compiler.IRCode, sv::Compiler.OptimizationState{Compiler.NativeInterpreter}, domtree::Compiler.GenericDomTre…
     6         0 @Base/logging/ConsoleLogger.jl                   57 showvalue(io::IOContext{IOBuffer}, msg::Tuple{UInt64})
     6         0 @Base/multimedia.jl                             123 show(io::IOContext{IOBuffer}, m::String, x::Tuple{UInt64})
     6         0 @Compiler/src/ssair/slot2ssa.jl                 773 construct_ssa!(ci::Core.CodeInfo, ir::Compiler.IRCode, sv::Compiler.OptimizationState{Compiler.NativeInterpreter}, domtree::Compiler.GenericDomTre…
     6         0 @Compiler/src/ssair/slot2ssa.jl                 160 rename_uses!
     6         0 @Compiler/src/ssair/inlining.jl                 377 ir_inline_item!(compact::Compiler.IncrementalCompact, idx::Int64, argexprs::Vector{Any}, item::Compiler.InliningTodo, boundscheck::Symbol, todo_bb…
     6         0 @Compiler/src/inferencestate.jl                 607 InferenceState
     6         0 @Compiler/src/inferencestate.jl                 605 Compiler.InferenceState(result::Compiler.InferenceResult, cache_mode::UInt8, interp::Compiler.NativeInterpreter)
     6         0 @Compiler/src/typeinfer.jl                      130 finish!(interp::Compiler.NativeInterpreter, caller::Compiler.InferenceState, validation_world::UInt64, time_before::UInt64)
     6         6 @Compiler/src/typeinfer.jl                      313 maybe_compress_codeinfo
     6         0 @Compiler/src/abstractinterpretation.jl        2681 abstract_call_known(interp::Compiler.NativeInterpreter, f::Any, arginfo::Compiler.ArgInfo, si::Compiler.StmtInfo, sv::Compiler.InferenceState, max…
     6         0 @Compiler/src/abstractinterpretation.jl        2470 abstract_eval_get_binding_type
     6         0 @Compiler/src/abstractinterpretation.jl        2443 abstract_eval_get_binding_type(interp::Compiler.NativeInterpreter, sv::Compiler.InferenceState, M::Any, s::Any)
     6         5 @Compiler/src/abstractinterpretation.jl        3700 scan_specified_partitions(query::Compiler.var"#abstract_eval_get_binding_type##0#abstract_eval_get_binding_type##1", walk_binding_partition::typeo…
     6         0 @Compiler/src/ssair/ir.jl                       653 _advance(stmt::Any, op::Int64)
     6         1 @Compiler/src/ssair/ir.jl                       578 getindex
     6         0 @Compiler/src/abstractinterpretation.jl         884 abstract_call_method_with_const_args(interp::Compiler.NativeInterpreter, result::Compiler.MethodCallResult, f::Any, arginfo::Compiler.ArgInfo, si:…
     6         0 @Compiler/src/optimize.jl                       386 recompute_effects_flags
     6         6 @Compiler/src/typeutils.jl                       55 argtypes_to_type(argtypes::Vector{Any})
     6         1 @Compiler/src/ssair/slot2ssa.jl                  62 renumber_ssa!
     6         0 @Base/array.jl                                 1148 (::Base.var"#_growend!##0#_growend!##1"{Vector{Int64}, Int64, Int64, Int64, Int64, Int64, Memory{Int64}, MemoryRef{Int64}})()
     6         0 @Base/array.jl                                 1067 array_new_memory
     6         0 @Compiler/src/abstractinterpretation.jl        2886 abstract_call(interp::Compiler.NativeInterpreter, arginfo::Compiler.ArgInfo, si::Compiler.StmtInfo, sv::Compiler.InferenceState, max_methods::Int6…
     6         0 @Compiler/src/abstractinterpretation.jl        2876 abstract_call_unknown(interp::Compiler.NativeInterpreter, ft::Any, arginfo::Compiler.ArgInfo, si::Compiler.StmtInfo, sv::Compiler.InferenceState, …
     7         0 @Base/logging/ConsoleLogger.jl                  194 handle_message(logger::Base.CoreLogging.ConsoleLogger, level::Base.CoreLogging.LogLevel, message::Any, _module::Any, group::Any, id::Any, filepath…
     7         2 @Base/lock.jl                                   376 macro expansion
     7         7 @MMSB/src/ffi/FFIWrapper.jl                     242 rust_tlog_append!
     7         0 @Compiler/src/optimize.jl                      1004 optimize(interp::Compiler.NativeInterpreter, opt::Compiler.OptimizationState{Compiler.NativeInterpreter}, caller::Compiler.InferenceResult)
     7         0 @Compiler/src/ssair/irinterp.jl                  58 abstract_call(interp::Compiler.NativeInterpreter, arginfo::Compiler.ArgInfo, sstate::Compiler.StatementState, irsv::Compiler.IRInterpretationState)
     7         0 @Compiler/src/abstractinterpretation.jl        2882 abstract_call(interp::Compiler.NativeInterpreter, arginfo::Compiler.ArgInfo, si::Compiler.StmtInfo, sv::Compiler.IRInterpretationState)
     7         0 @Compiler/src/ssair/passes.jl                  1548 sroa_pass!(ir::Compiler.IRCode, inlining::Compiler.InliningState{Compiler.NativeInterpreter})
     7         0 @Compiler/src/ssair/legacy.jl                    20 inflate_ir!(ci::Core.CodeInfo, mi::Core.MethodInstance)
     7         0 @MMSB/benchmark/benchmarks.jl                   142 _full_system_benchmark!()
     7         0 @MMSB/benchmark/benchmarks.jl                    43 _start_state
     7         0 @MMSB/benchmark/benchmarks.jl                    44 #_start_state#1
     7         0 @MMSB/src/API.jl                                 30 mmsb_start
     7         0 @MMSB/src/API.jl                                 34 #mmsb_start#1
     7         0 @MMSB/src/ffi/FFIWrapper.jl                     229 rust_tlog_new(path::String)
     7         0 @Base/array.jl                                 1131 _growend!
     8         8 @Base/promotion.jl                              637 ==
     8         0 @MMSB/src/01_types/MMSBState.jl                  62 Main.MMSBBenchmarks.MMSB.MMSBStateTypes.MMSBState(config::Main.MMSBBenchmarks.MMSB.MMSBStateTypes.MMSBConfig)
     8         0 @MMSB/src/02_semiring/DeltaRouter.jl             47 kwcall(::@NamedTuple{source::Symbol}, ::typeof(Main.MMSBBenchmarks.MMSB.DeltaRouter.create_delta), state::Main.MMSBBenchmarks.MMSB.MMSBStateTypes.…
     8         0 @MMSB/src/02_semiring/DeltaRouter.jl             36 route_delta!(state::Main.MMSBBenchmarks.MMSB.MMSBStateTypes.MMSBState, delta::Main.MMSBBenchmarks.MMSB.DeltaTypes.Delta; propagate::Bool)
     8         0 @MMSB/src/01_page/TLog.jl                        40 append_to_log!
     8         0 @MMSB/src/01_page/TLog.jl                        30 _with_rust_errors(f::Main.MMSBBenchmarks.MMSB.TLog.var"#append_to_log!##0#append_to_log!##1"{Main.MMSBBenchmarks.MMSB.MMSBStateTypes.MMSBState, Ma…
     8         0 @MMSB/src/01_page/TLog.jl                        42 #append_to_log!##0
     8         0 @Compiler/src/abstractinterpretation.jl        3060 abstract_eval_call(interp::Compiler.NativeInterpreter, e::Expr, sstate::Compiler.StatementState, sv::Compiler.IRInterpretationState)
     8         0 @Compiler/src/abstractinterpretation.jl        2648 abstract_call_known(interp::Compiler.NativeInterpreter, f::Any, arginfo::Compiler.ArgInfo, si::Compiler.StmtInfo, sv::Compiler.InferenceState, max…
     8         0 @Compiler/src/abstractinterpretation.jl        1872 abstract_apply(interp::Compiler.NativeInterpreter, argtypes::Vector{Any}, si::Compiler.StmtInfo, sv::Compiler.InferenceState, max_methods::Int64)
     8         0 @Compiler/src/abstractinterpretation.jl        1852 (::Compiler.var"#infercalls#abstract_apply##0"{Compiler.StmtInfo, Int64, Compiler.Future{Compiler.CallMeta}, Compiler.UnionSplitApplyCallInfo, Vec…
     8         1 @Compiler/src/ssair/ir.jl                       807 IncrementalCompact
     8         8 @Compiler/src/utilities.jl                      128 retrieve_code_info(mi::Core.MethodInstance, world::UInt64)
     8         0 @Compiler/src/typelattice.jl                    503 ⊑(lattice::Compiler.PartialsLattice{Compiler.ConstsLattice}, a::Any, b::Any)
     8         0 @Compiler/src/ssair/ir.jl                       660 iterate
     8         0 @Compiler/src/abstractinterpretation.jl        2781 abstract_call_known(interp::Compiler.NativeInterpreter, f::Any, arginfo::Compiler.ArgInfo, si::Compiler.StmtInfo, sv::Compiler.InferenceState, max…
     8         0 @Compiler/src/abstractinterpretation.jl        2683 abstract_call_known(interp::Compiler.NativeInterpreter, f::Any, arginfo::Compiler.ArgInfo, si::Compiler.StmtInfo, sv::Compiler.IRInterpretationSta…
     8         0 @Compiler/src/abstractinterpretation.jl        1983 abstract_call_builtin(interp::Compiler.NativeInterpreter, f::Core.Builtin, ::Compiler.ArgInfo, sv::Compiler.IRInterpretationState)
     9         0 @MMSB/src/ffi/FFIWrapper.jl                     359 rust_delta_payload(handle::Main.MMSBBenchmarks.MMSB.FFIWrapper.RustDeltaHandle)
     9         0 @Compiler/src/inferencestate.jl                 602 Compiler.InferenceState(result::Compiler.InferenceResult, cache_mode::UInt8, interp::Compiler.NativeInterpreter)
     9         0 @Compiler/src/typeinfer.jl                      963 typeinf_edge(interp::Compiler.NativeInterpreter, method::Method, atype::Any, sparams::Core.SimpleVector, caller::Compiler.InferenceState, edgecycl…
     9         0 @Compiler/src/optimize.jl                       414 argextype
    10        10 @MMSB/src/ffi/FFIWrapper.jl                     355 rust_delta_payload(handle::Main.MMSBBenchmarks.MMSB.FFIWrapper.RustDeltaHandle)
    10         0 @MMSB/src/ffi/FFIWrapper.jl                     339 rust_delta_mask(handle::Main.MMSBBenchmarks.MMSB.FFIWrapper.RustDeltaHandle)
    10         0 @Compiler/src/ssair/irinterp.jl                 154 reprocess_instruction!(interp::Compiler.NativeInterpreter, inst::Compiler.Instruction, idx::Int64, bb::Int64, irsv::Compiler.IRInterpretationState)
    10         0 @Compiler/src/abstractinterpretation.jl        3705 scan_leaf_partitions
    10         0 @Compiler/src/typeinfer.jl                      199 finish_nocycle(::Compiler.NativeInterpreter, frame::Compiler.InferenceState, time_before::UInt64)
    11         7 @MMSB/src/04_propagation/PropagationEngine.jl   320 recompute_page!(state::Main.MMSBBenchmarks.MMSB.MMSBStateTypes.MMSBState, page_id::UInt64)
    11        11 @Base/logging/ConsoleLogger.jl                  113 handle_message(logger::Base.CoreLogging.ConsoleLogger, level::Base.CoreLogging.LogLevel, message::Any, _module::Any, group::Any, id::Any, filepath…
    11         8 @MMSB/src/04_propagation/PropagationEngine.jl   316 recompute_page!(state::Main.MMSBBenchmarks.MMSB.MMSBStateTypes.MMSBState, page_id::UInt64)
    11         0 @Compiler/src/abstractinterpretation.jl         167 (::Compiler.var"#infercalls#abstract_call_gf_by_type##0"{Compiler.ArgInfo, Compiler.StmtInfo, Compiler.CallInferenceState, Compiler.Future{Compile…
    11         0 @Compiler/src/methodtable.jl                    105 findall(sig::Type, table::Compiler.CachedMethodTable{Compiler.InternalMethodTable}; limit::Int64)
    12         3 @MMSB/src/04_propagation/PropagationEngine.jl   326 recompute_page!(state::Main.MMSBBenchmarks.MMSB.MMSBStateTypes.MMSBState, page_id::UInt64)
    12         0 @Compiler/src/ssair/irinterp.jl                 364 (::Compiler.var"#218#219"{Nothing, Compiler.NativeInterpreter, Compiler.IRInterpretationState, Compiler.var"#check_ret!#217"{Vector{Int64}}, BitSe…
    12         0 @Compiler/src/ssair/inlining.jl                 958 retrieve_ir_for_inlining(cached_result::Core.CodeInstance, src::String)
    12         1 @Compiler/src/abstractinterpretation.jl         735 abstract_call_method(interp::Compiler.NativeInterpreter, method::Method, sig::Any, sparams::Core.SimpleVector, hardlimit::Bool, si::Compiler.StmtI…
    12         9 @MMSB/src/03_dag/EventSystem.jl                  83 emit_event!(::Main.MMSBBenchmarks.MMSB.MMSBStateTypes.MMSBState, ::Main.MMSBBenchmarks.MMSB.EventSystem.EventType, ::UInt64, ::Vararg{Any})
    13         0 @Base/array.jl                                  939 getindex
    13         0 @Base/array.jl                                  299 copyto!
    13         0 @Base/array.jl                                  308 _copyto_impl!
    13        12 @Base/boot.jl                                   648 Array
    14         1 @MMSB/src/ffi/FFIWrapper.jl                     163 rust_delta_apply!
    14         0 @Base/genericmemory.jl                          125 unsafe_copyto!
    14        14 @Base/cmem.jl                                    28 memmove
    14         0 @Compiler/src/abstractinterpretation.jl        1265 semi_concrete_eval_call(interp::Compiler.NativeInterpreter, mi::Core.MethodInstance, result::Compiler.MethodCallResult, arginfo::Compiler.ArgInfo,…
    14         0 @Compiler/src/ssair/irinterp.jl                 318 ir_abstract_constant_propagation
    14         1 @Compiler/src/ssair/irinterp.jl                 332 ir_abstract_constant_propagation(interp::Compiler.NativeInterpreter, irsv::Compiler.IRInterpretationState; externally_refined::Nothing)
    14        13 @Base/runtime_internals.jl                     1424 _uncompressed_ir(codeinst::Core.CodeInstance, s::String)
    14         0 @Compiler/src/ssair/inlining.jl                1618 assemble_inline_todo!(ir::Compiler.IRCode, state::Compiler.InliningState{Compiler.NativeInterpreter})
    15         1 @Compiler/src/typeinfer.jl                      205 finish_nocycle(::Compiler.NativeInterpreter, frame::Compiler.InferenceState, time_before::UInt64)
    15         0 @Compiler/src/methodtable.jl                     70 findall
    15         0 @Compiler/src/methodtable.jl                     70 #findall#5
    15         0 @Compiler/src/methodtable.jl                     97 _findall
    15        15 @Base/runtime_internals.jl                     1410 _methods_by_ftype
    16        16 @MMSB/src/ffi/FFIWrapper.jl                     162 rust_delta_apply!
    16         0 @MMSB/src/ffi/FFIWrapper.jl                     358 rust_delta_payload(handle::Main.MMSBBenchmarks.MMSB.FFIWrapper.RustDeltaHandle)
    16         0 @MMSB/src/ffi/FFIWrapper.jl                     298 rust_delta_epoch
    16         0 @MMSB/src/ffi/FFIWrapper.jl                     284 rust_delta_id
    16         0 @MMSB/src/ffi/FFIWrapper.jl                     305 rust_delta_is_sparse
    16        16 @Base/essentials.jl                             920 getindex
    16         0 @Compiler/src/ssair/irinterp.jl                 286 scan!(callback::Compiler.var"#218#219"{Nothing, Compiler.NativeInterpreter, Compiler.IRInterpretationState, Compiler.var"#check_ret!#217"{Vector{I…
    16         0 @Compiler/src/abstractinterpretation.jl         120 abstract_call_gf_by_type(interp::Compiler.NativeInterpreter, func::Any, arginfo::Compiler.ArgInfo, si::Compiler.StmtInfo, atype::Any, sv::Compiler…
    16         0 @Compiler/src/abstractinterpretation.jl         342 find_method_matches
    16         0 @Compiler/src/abstractinterpretation.jl         348 #find_method_matches#129
    16         0 @Compiler/src/abstractinterpretation.jl         386 find_simple_method_matches(interp::Compiler.NativeInterpreter, atype::Any, max_methods::Int64)
    16         0 @Compiler/src/methodtable.jl                    102 findall
    16         1 @Compiler/src/ssair/ir.jl                       661 iterate
    16         1 @Compiler/src/ssair/ir.jl                      1526 process_node!(compact::Compiler.IncrementalCompact, result_idx::Int64, inst::Compiler.Instruction, idx::Int64, processed_idx::Int64, active_bb::In…
    17        10 @Base/stat.jl                                    93 Base.Filesystem.StatStruct(desc::String, buf::Memory{UInt8}, ioerrno::Int32)
    17         0 @Compiler/src/ssair/inlining.jl                 873 resolve_todo(mi::Core.MethodInstance, result::Nothing, info::Compiler.CallInfo, flag::UInt32, state::Compiler.InliningState{Compiler.NativeInterpr…
    18         0 @MMSB/src/ffi/FFIWrapper.jl                     349 rust_delta_payload(handle::Main.MMSBBenchmarks.MMSB.FFIWrapper.RustDeltaHandle)
    18        18 @MMSB/src/ffi/FFIWrapper.jl                     263 rust_tlog_reader_next
    18         0 @Compiler/src/abstractinterpretation.jl         892 abstract_call_method_with_const_args(interp::Compiler.NativeInterpreter, result::Compiler.MethodCallResult, f::Any, arginfo::Compiler.ArgInfo, si:…
    18         0 @Compiler/src/ssair/passes.jl                     5 is_known_call(x::Any, func::Any, ir::Compiler.IncrementalCompact)
    19         0 @MMSB/src/ffi/FFIWrapper.jl                     344 rust_delta_mask(handle::Main.MMSBBenchmarks.MMSB.FFIWrapper.RustDeltaHandle)
    19         0 @Compiler/src/ssair/inlining.jl                1309 handle_any_const_result!(cases::Vector{Compiler.InliningCase}, result::Any, match::Core.MethodMatch, argtypes::Vector{Any}, info::Compiler.CallInf…
    19         0 @Compiler/src/ssair/inlining.jl                1409 handle_match!
    19         0 @Compiler/src/ssair/inlining.jl                1417 handle_match!(cases::Vector{Compiler.InliningCase}, match::Core.MethodMatch, argtypes::Vector{Any}, info::Compiler.CallInfo, flag::UInt32, state::…
    19         0 @Compiler/src/ssair/inlining.jl                 921 analyze_method!
    19         0 @Compiler/src/ssair/inlining.jl                 954 analyze_method!(match::Core.MethodMatch, argtypes::Vector{Any}, info::Compiler.CallInfo, flag::UInt32, state::Compiler.InliningState{Compiler.Nati…
    20         0 @MMSB/src/ffi/FFIWrapper.jl                     161 rust_delta_apply!
    20         0 @MMSB/src/ffi/FFIWrapper.jl                     335 rust_delta_mask(handle::Main.MMSBBenchmarks.MMSB.FFIWrapper.RustDeltaHandle)
    20         0 @Compiler/src/optimize.jl                       417 argextype
    20         1 @Compiler/src/optimize.jl                       418 argextype
    22         0 @MMSB/src/ffi/FFIWrapper.jl                     307 rust_delta_is_sparse
    22         1 @MMSB/src/ffi/FFIWrapper.jl                     286 rust_delta_id
    23         0 @MMSB/src/ffi/FFIWrapper.jl                     382 rust_delta_intent_metadata(handle::Main.MMSBBenchmarks.MMSB.FFIWrapper.RustDeltaHandle)
    23         0 @MMSB/src/ffi/FFIWrapper.jl                     351 rust_delta_payload(handle::Main.MMSBBenchmarks.MMSB.FFIWrapper.RustDeltaHandle)
    23        16 @Base/logging/ConsoleLogger.jl                  162 handle_message(logger::Base.CoreLogging.ConsoleLogger, level::Base.CoreLogging.LogLevel, message::Any, _module::Any, group::Any, id::Any, filepath…
    23         0 @Compiler/src/optimize.jl                       453 argextype(x::Any, src::Compiler.IncrementalCompact, sptypes::Vector{Compiler.VarState}, slottypes::Vector{Any})
    23         0 @Compiler/src/abstractinterpretation.jl        3588 abstract_eval_globalref_type
    23         0 @Compiler/src/abstractinterpretation.jl        3718 abstract_load_all_consistent_leaf_partitions
    23         4 @Compiler/src/abstractinterpretation.jl        3703 scan_leaf_partitions
    23         0 @Compiler/src/abstractinterpretation.jl         874 abstract_call_method_with_const_args(interp::Compiler.NativeInterpreter, result::Compiler.MethodCallResult, f::Any, arginfo::Compiler.ArgInfo, si:…
    23        21 @Compiler/src/abstractinterpretation.jl        1008 concrete_eval_call(interp::Compiler.NativeInterpreter, f::Any, result::Compiler.MethodCallResult, arginfo::Compiler.ArgInfo, sv::Compiler.Inferenc…
    25        22 @Base/logging/logging.jl                        407 macro expansion
    26         1 @MMSB/src/ffi/FFIWrapper.jl                     337 rust_delta_mask(handle::Main.MMSBBenchmarks.MMSB.FFIWrapper.RustDeltaHandle)
    26         0 @MMSB/src/ffi/FFIWrapper.jl                     353 rust_delta_payload(handle::Main.MMSBBenchmarks.MMSB.FFIWrapper.RustDeltaHandle)
    26         0 @MMSB/src/ffi/FFIWrapper.jl                     380 rust_delta_intent_metadata(handle::Main.MMSBBenchmarks.MMSB.FFIWrapper.RustDeltaHandle)
    26         0 @Compiler/src/ssair/inlining.jl                1369 compute_inlining_cases(info::Compiler.CallInfo, flag::UInt32, sig::Compiler.Signature, state::Compiler.InliningState{Compiler.NativeInterpreter})
    26         0 @Compiler/src/ssair/inlining.jl                1297 handle_any_const_result!
    26         0 @Compiler/src/ssair/ir.jl                      1794 process_newnode!(compact::Compiler.IncrementalCompact, new_idx::Int64, new_node_entry::Compiler.Instruction, new_node_info::Compiler.NewNodeInfo, …
    27         0 @MMSB/src/ffi/FFIWrapper.jl                     262 rust_tlog_reader_next
    27         1 @MMSB/src/ffi/FFIWrapper.jl                     300 rust_delta_epoch
    27         1 @Compiler/src/ssair/inlining.jl                 663 batch_inline!(ir::Compiler.IRCode, todo::Vector{Pair{Int64, Any}}, propagate_inbounds::Bool, interp::Compiler.NativeInterpreter)
    27         0 @Compiler/src/ssair/ir.jl                      1455 process_node!(compact::Compiler.IncrementalCompact, result_idx::Int64, inst::Compiler.Instruction, idx::Int64, processed_idx::Int64, active_bb::In…
    27        24 @Compiler/src/ssair/passes.jl                   363 already_inserted_ssa
    27         0 @Compiler/src/ssair/ir.jl                      1919 iterate_compact(compact::Compiler.IncrementalCompact)
    29         0 @Compiler/src/ssair/ir.jl                      1940 iterate_compact(compact::Compiler.IncrementalCompact)
    29         0 @Compiler/src/ssair/inlining.jl                1401 handle_call!
    30        14 @Base/stat.jl                                   193 stat(path::String)
    34         0 @Compiler/src/ssair/inlining.jl                1652 assemble_inline_todo!(ir::Compiler.IRCode, state::Compiler.InliningState{Compiler.NativeInterpreter})
    35         0 @Compiler/src/optimize.jl                      1306 slot2reg
    36         0 @Base/stat.jl                                   191 stat(path::String)
    37         0 @MMSB/src/01_page/TLog.jl                        79 (::Main.MMSBBenchmarks.MMSB.TLog.var"#5#6"{Nothing, Nothing, Nothing, Nothing})(reader::Main.MMSBBenchmarks.MMSB.FFIWrapper.RustTLogReaderHandle)
    37         0 @MMSB/src/01_page/TLog.jl                        80 (::Main.MMSBBenchmarks.MMSB.TLog.var"#5#6"{Nothing, Nothing, Nothing, Nothing})(reader::Main.MMSBBenchmarks.MMSB.FFIWrapper.RustTLogReaderHandle)
    37         0 @Compiler/src/ssair/inlining.jl                  79 ssa_inlining_pass!
    38         0 @Base/baseext.jl                                 23 Array
    38         0 @MMSB/src/01_page/Delta.jl                       41 Main.MMSBBenchmarks.MMSB.DeltaTypes.Delta(handle::Main.MMSBBenchmarks.MMSB.FFIWrapper.RustDeltaHandle)
    38         0 @MMSB/src/01_page/Delta.jl                       38 Main.MMSBBenchmarks.MMSB.DeltaTypes.Delta(handle::Main.MMSBBenchmarks.MMSB.FFIWrapper.RustDeltaHandle)
    39         0 @MMSB/src/ffi/FFIWrapper.jl                     291 rust_delta_page_id
    39         0 @Compiler/src/ssair/ir.jl                      2139 compact!(code::Compiler.IRCode, allow_cfg_transforms::Bool)
    40         0 @MMSB/src/ffi/FFIWrapper.jl                     293 rust_delta_page_id
    41         0 @MMSB/src/ffi/FFIWrapper.jl                     321 rust_delta_source(handle::Main.MMSBBenchmarks.MMSB.FFIWrapper.RustDeltaHandle)
    42         0 @MMSB/src/01_page/Delta.jl                       39 Main.MMSBBenchmarks.MMSB.DeltaTypes.Delta(handle::Main.MMSBBenchmarks.MMSB.FFIWrapper.RustDeltaHandle)
    42         0 @MMSB/src/ffi/FFIWrapper.jl                     312 rust_delta_timestamp
    42         0 @MMSB/src/ffi/FFIWrapper.jl                     330 rust_delta_source(handle::Main.MMSBBenchmarks.MMSB.FFIWrapper.RustDeltaHandle)
    43         0 @MMSB/src/01_page/Delta.jl                       40 Main.MMSBBenchmarks.MMSB.DeltaTypes.Delta(handle::Main.MMSBBenchmarks.MMSB.FFIWrapper.RustDeltaHandle)
    45         0 @Compiler/src/ssair/ir.jl                      2137 compact!
    46         0 @MMSB/src/01_page/ReplayEngine.jl                65 replay_to_epoch(state::Main.MMSBBenchmarks.MMSB.MMSBStateTypes.MMSBState, target_epoch::UInt32)
    46         0 @MMSB/src/01_page/ReplayEngine.jl                50 _apply_delta!
    46         0 @MMSB/src/ffi/FFIWrapper.jl                     319 rust_delta_source(handle::Main.MMSBBenchmarks.MMSB.FFIWrapper.RustDeltaHandle)
    49         0 @MMSB/src/01_page/Delta.jl                       46 Main.MMSBBenchmarks.MMSB.DeltaTypes.Delta(handle::Main.MMSBBenchmarks.MMSB.FFIWrapper.RustDeltaHandle)
    49         0 @MMSB/src/ffi/FFIWrapper.jl                     314 rust_delta_timestamp
    49         0 @Compiler/src/ssair/inlining.jl                  76 ssa_inlining_pass!
    50         0 @MMSB/src/ffi/FFIWrapper.jl                     264 rust_tlog_reader_next
    51         0 @MMSB/src/01_page/Delta.jl                       42 Main.MMSBBenchmarks.MMSB.DeltaTypes.Delta(handle::Main.MMSBBenchmarks.MMSB.FFIWrapper.RustDeltaHandle)
    54         0 @MMSB/src/04_propagation/PropagationEngine.jl   249 _handle_data_dependency!(state::Main.MMSBBenchmarks.MMSB.MMSBStateTypes.MMSBState, page_id::UInt64, mode::Main.MMSBBenchmarks.MMSB.PropagationEngi…
    61         0 @Compiler/src/ssair/ir.jl                      1862 iterate
    61         3 @Compiler/src/inferencestate.jl                1205 doworkloop(interp::Compiler.NativeInterpreter, sv::Compiler.InferenceState)
    65         3 @Compiler/src/abstractinterpretation.jl        4491 typeinf(interp::Compiler.NativeInterpreter, frame::Compiler.InferenceState)
    66         0 @MMSB/src/01_page/TLog.jl                        81 (::Main.MMSBBenchmarks.MMSB.TLog.var"#5#6"{Nothing, Nothing, Nothing, Nothing})(reader::Main.MMSBBenchmarks.MMSB.FFIWrapper.RustTLogReaderHandle)
    71         0 @MMSB/src/01_page/Delta.jl                       43 Main.MMSBBenchmarks.MMSB.DeltaTypes.Delta(handle::Main.MMSBBenchmarks.MMSB.FFIWrapper.RustDeltaHandle)
    78         0 @MMSB/src/02_semiring/DeltaRouter.jl             40 route_delta!(state::Main.MMSBBenchmarks.MMSB.MMSBStateTypes.MMSBState, delta::Main.MMSBBenchmarks.MMSB.DeltaTypes.Delta; propagate::Bool)
    85         0 @Compiler/src/abstractinterpretation.jl        1348 const_prop_call(interp::Compiler.NativeInterpreter, mi::Core.MethodInstance, result::Compiler.MethodCallResult, arginfo::Compiler.ArgInfo, sv::Com…
    86        68 @Base/logging/ConsoleLogger.jl                  144 handle_message(logger::Base.CoreLogging.ConsoleLogger, level::Base.CoreLogging.LogLevel, message::Any, _module::Any, group::Any, id::Any, filepath…
    89         0 @Compiler/src/abstractinterpretation.jl         898 abstract_call_method_with_const_args(interp::Compiler.NativeInterpreter, result::Compiler.MethodCallResult, f::Any, arginfo::Compiler.ArgInfo, si:…
    91        91 @Compiler/src/typeinfer.jl                     1436 add_codeinsts_to_jit!(interp::Compiler.NativeInterpreter, ci::Core.CodeInstance, source_mode::UInt8)
    92         0 @Compiler/src/typeinfer.jl                     1443 typeinf_ext_toplevel
    95         0 @MMSB/src/01_page/TLog.jl                        75 (::Main.MMSBBenchmarks.MMSB.TLog.var"#5#6"{Nothing, Nothing, Nothing, Nothing})(reader::Main.MMSBBenchmarks.MMSB.FFIWrapper.RustTLogReaderHandle)
   102         0 @MMSB/src/01_page/Delta.jl                       45 Main.MMSBBenchmarks.MMSB.DeltaTypes.Delta(handle::Main.MMSBBenchmarks.MMSB.FFIWrapper.RustDeltaHandle)
   103         0 @Compiler/src/abstractinterpretation.jl         252 (::Compiler.var"#infercalls#abstract_call_gf_by_type##0"{Compiler.ArgInfo, Compiler.StmtInfo, Compiler.CallInferenceState, Compiler.Future{Compile…
   115         0 @Compiler/src/abstractinterpretation.jl         338 abstract_call_gf_by_type(interp::Compiler.NativeInterpreter, func::Any, arginfo::Compiler.ArgInfo, si::Compiler.StmtInfo, atype::Any, sv::Compiler…
   123         0 @Compiler/src/abstractinterpretation.jl         178 (::Compiler.var"#handle1#abstract_call_gf_by_type##1"{Int64, Compiler.Future{Compiler.MethodCallResult}, Int64, Vector{Union{Nothing, Core.CodeIns…
   123         0 @Compiler/src/abstractinterpretation.jl         868 abstract_call_method_with_const_args
   128         0 @Compiler/src/abstractinterpretation.jl        2782 abstract_call_known(interp::Compiler.NativeInterpreter, f::Any, arginfo::Compiler.ArgInfo, si::Compiler.StmtInfo, sv::Compiler.InferenceState, max…
   130         0 @MMSB/src/04_propagation/PropagationEngine.jl   327 recompute_page!(state::Main.MMSBBenchmarks.MMSB.MMSBStateTypes.MMSBState, page_id::UInt64)
   130         0 @MMSB/src/02_semiring/DeltaRouter.jl             29 route_delta!(state::Main.MMSBBenchmarks.MMSB.MMSBStateTypes.MMSBState, delta::Main.MMSBBenchmarks.MMSB.DeltaTypes.Delta)
   148         1 @Compiler/src/abstractinterpretation.jl        2889 abstract_call(interp::Compiler.NativeInterpreter, arginfo::Compiler.ArgInfo, si::Compiler.StmtInfo, sv::Compiler.InferenceState, max_methods::Int6…
   150         0 @Compiler/src/abstractinterpretation.jl        3042 abstract_call(interp::Compiler.NativeInterpreter, arginfo::Compiler.ArgInfo, sstate::Compiler.StatementState, sv::Compiler.InferenceState)
   150         0 @Compiler/src/abstractinterpretation.jl        2882 abstract_call
   151         1 @MMSB/src/04_propagation/PropagationEngine.jl   251 _handle_data_dependency!(state::Main.MMSBBenchmarks.MMSB.MMSBStateTypes.MMSBState, page_id::UInt64, mode::Main.MMSBBenchmarks.MMSB.PropagationEngi…
   153         0 @Compiler/src/abstractinterpretation.jl        3060 abstract_eval_call
   155         0 @Compiler/src/abstractinterpretation.jl        3389 abstract_eval_statement_expr(interp::Compiler.NativeInterpreter, e::Expr, sstate::Compiler.StatementState, sv::Compiler.InferenceState)
   156         0 @Compiler/src/abstractinterpretation.jl        3835 abstract_eval_basic_statement
   159         0 @Compiler/src/abstractinterpretation.jl        4342 typeinf_local(interp::Compiler.NativeInterpreter, frame::Compiler.InferenceState, nextresult::Compiler.CurrentState)
   159         0 @Compiler/src/abstractinterpretation.jl        3792 abstract_eval_basic_statement
   169       134 @Base/logging/ConsoleLogger.jl                  149 handle_message(logger::Base.CoreLogging.ConsoleLogger, level::Base.CoreLogging.LogLevel, message::Any, _module::Any, group::Any, id::Any, filepath…
   169         0 @Compiler/src/abstractinterpretation.jl        4500 typeinf(interp::Compiler.NativeInterpreter, frame::Compiler.InferenceState)
   195         0 @MMSB/src/02_semiring/DeltaRouter.jl             39 route_delta!(state::Main.MMSBBenchmarks.MMSB.MMSBStateTypes.MMSBState, delta::Main.MMSBBenchmarks.MMSB.DeltaTypes.Delta; propagate::Bool)
   195         0 @MMSB/src/02_semiring/DeltaRouter.jl             91 propagate_change!(state::Main.MMSBBenchmarks.MMSB.MMSBStateTypes.MMSBState, changed_page_id::UInt64)
   195         0 @MMSB/src/04_propagation/PropagationEngine.jl   200 propagate_change!(state::Main.MMSBBenchmarks.MMSB.MMSBStateTypes.MMSBState, changed_page_id::UInt64)
   195         0 @MMSB/src/04_propagation/PropagationEngine.jl   209 propagate_change!(state::Main.MMSBBenchmarks.MMSB.MMSBStateTypes.MMSBState, changed_pages::Vector{UInt64}, mode::Main.MMSBBenchmarks.MMSB.Propagat…
   195         0 @MMSB/src/04_propagation/PropagationEngine.jl   230 _execute_command_buffer!(state::Main.MMSBBenchmarks.MMSB.MMSBStateTypes.MMSBState, commands::Dict{UInt64, Set{Main.MMSBBenchmarks.MMSB.GraphTypes.…
   195         0 @MMSB/src/04_propagation/PropagationEngine.jl   237 _apply_edges!(state::Main.MMSBBenchmarks.MMSB.MMSBStateTypes.MMSBState, page_id::UInt64, edges::Set{Main.MMSBBenchmarks.MMSB.GraphTypes.EdgeType},…
   220         1 @Compiler/src/optimize.jl                      1013 run_passes_ipo_safe(ci::Core.CodeInfo, sv::Compiler.OptimizationState{Compiler.NativeInterpreter}, optimize_until::Nothing)
   221         0 @Compiler/src/optimize.jl                      1002 optimize(interp::Compiler.NativeInterpreter, opt::Compiler.OptimizationState{Compiler.NativeInterpreter}, caller::Compiler.InferenceResult)
   221         0 @Compiler/src/optimize.jl                      1027 run_passes_ipo_safe
   230         1 @Compiler/src/typeinfer.jl                      202 finish_nocycle(::Compiler.NativeInterpreter, frame::Compiler.InferenceState, time_before::UInt64)
   248         0 @Base/array.jl                                  937 getindex
   248         0 @Base/abstractarray.jl                          822 similar
   248         0 @Base/array.jl                                  377 similar
   252         0 @MMSB/src/ffi/FFIWrapper.jl                     345 rust_delta_mask(handle::Main.MMSBBenchmarks.MMSB.FFIWrapper.RustDeltaHandle)
   255         0 @Compiler/src/abstractinterpretation.jl        4507 typeinf(interp::Compiler.NativeInterpreter, frame::Compiler.InferenceState)
   256         0 @Base/boot.jl                                   660 Array
   295         0 @Base/boot.jl                                   647 Array
   310         0 @Base/logging/ConsoleLogger.jl                  110 kwcall(::NamedTuple, ::typeof(Base.CoreLogging.handle_message), logger::Base.CoreLogging.ConsoleLogger, level::Base.CoreLogging.LogLevel, message:…
   332         0 @MMSB/src/01_page/Delta.jl                       44 Main.MMSBBenchmarks.MMSB.DeltaTypes.Delta(handle::Main.MMSBBenchmarks.MMSB.FFIWrapper.RustDeltaHandle)
   336       335 @Base/boot.jl                                   588 GenericMemory
   402         0 @MMSB/src/ffi/FFIWrapper.jl                     435 rust_get_last_error
   403         0 @MMSB/src/ffi/RustErrors.jl                      37 check_rust_error
   405         2 @MMSB/src/ffi/RustErrors.jl                      63 (::Main.MMSBBenchmarks.MMSB.RustErrors.var"#4#5")(context::String)
   411         0 @Compiler/src/typeinfer.jl                     1259 typeinf_ext(interp::Compiler.NativeInterpreter, mi::Core.MethodInstance, source_mode::UInt8)
   412         0 @Compiler/src/typeinfer.jl                     1442 typeinf_ext_toplevel
   418        12 @MMSB/src/ffi/FFIWrapper.jl                      15 _check_rust_error(context::String)
   502         0 @Compiler/src/typeinfer.jl                     1451 typeinf_ext_toplevel(mi::Core.MethodInstance, world::UInt64, source_mode::UInt8, trim_mode::UInt8)
   618       615 @Base/stat.jl                                   192 stat(path::String)
   643         0 @MMSB/src/02_semiring/DeltaRouter.jl             38 route_delta!(state::Main.MMSBBenchmarks.MMSB.MMSBStateTypes.MMSBState, delta::Main.MMSBBenchmarks.MMSB.DeltaTypes.Delta; propagate::Bool)
   691         0 @MMSB/src/ffi/FFIWrapper.jl                      71 ensure_rust_artifacts
   691         0 @MMSB/src/ffi/FFIWrapper.jl                      68 rust_artifacts_available
   691         3 @Base/stat.jl                                   587 isfile
   702       320 @Base/reflection.jl                            1282 #invokelatest_gr#232
   703         0 @Base/reflection.jl                            1274 invokelatest_gr
   705         0 @Base/logging/logging.jl                        432 handle_message_nothrow(logger::Any, level::Any, msg::Any, _module::Any, group::Any, id::Any, file::Any, line::Any; kwargs...)
   717         0 @Base/logging/logging.jl                        417 macro expansion
   717        11 @Base/logging/logging.jl                        429 kwcall(::NamedTuple, ::typeof(Base.CoreLogging.handle_message_nothrow), logger::Any, level::Any, msg::Any, _module::Any, group::Any, id::Any, file…
   742         0 @MMSB/src/03_dag/EventSystem.jl                 130 log_event!(state::Main.MMSBBenchmarks.MMSB.MMSBStateTypes.MMSBState, event_type::Main.MMSBBenchmarks.MMSB.EventSystem.EventType, data::Tuple{UInt6…
   762        16 @MMSB/src/03_dag/EventSystem.jl                  93 emit_event!(::Main.MMSBBenchmarks.MMSB.MMSBStateTypes.MMSBState, ::Main.MMSBBenchmarks.MMSB.EventSystem.EventType, ::UInt64, ::Vararg{Any})
   768         1 @MMSB/src/01_page/TLog.jl                        89 (::Main.MMSBBenchmarks.MMSB.TLog.var"#5#6"{Nothing, Nothing, Nothing, Nothing})(reader::Main.MMSBBenchmarks.MMSB.FFIWrapper.RustTLogReaderHandle)
   824         0 @MMSB/benchmark/benchmarks.jl                   152 _full_system_benchmark!()
   824         0 @MMSB/src/API.jl                                 88 update_page
   824         0 @MMSB/src/API.jl                                104 update_page(state::Main.MMSBBenchmarks.MMSB.MMSBStateTypes.MMSBState, page_id::UInt64, bytes::Vector{UInt8}; source::Symbol)
   824         0 @MMSB/src/02_semiring/DeltaRouter.jl             29 route_delta!
  1003         0 @MMSB/src/01_page/ReplayEngine.jl                61 replay_to_epoch(state::Main.MMSBBenchmarks.MMSB.MMSBStateTypes.MMSBState, target_epoch::UInt32)
  1003         0 @MMSB/src/01_page/ReplayEngine.jl                55 _all_deltas
  1003         0 @MMSB/src/01_page/TLog.jl                        66 query_log
  1003         0 @MMSB/src/01_page/TLog.jl                        71 #query_log#3
  1003         0 @MMSB/src/01_page/TLog.jl                        60 _iterate_log(f::Main.MMSBBenchmarks.MMSB.TLog.var"#5#6"{Nothing, Nothing, Nothing, Nothing}, path::String)
  1050         0 @MMSB/benchmark/benchmarks.jl                   157 _full_system_benchmark!()
  3566         0 @Base/client.jl                                 550 _start()
  3566         0 @Base/client.jl                                 283 exec_options(opts::Base.JLOptions)
  3566      1323 @Base/boot.jl                                   489 eval(m::Module, e::Any)
Total snapshots: 3904. Utilization: 100% across all threads and tasks. Use the `groupby` kwarg to break down by thread and/or task.
