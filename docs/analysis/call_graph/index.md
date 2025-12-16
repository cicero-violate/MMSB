# Call Graph Analysis

This document shows the **interprocedural call graph** - which functions call which other functions.

> **Note:** This is NOT a control flow graph (CFG). CFG shows intraprocedural control flow (branches, loops) within individual functions.

## Call Graph Statistics

- Total functions: 592
- Total function calls: 754
- Maximum call depth: 9
- Leaf functions (no outgoing calls): 303

## Call Graph Visualization

```mermaid
graph TD
    MMSB_build_rs__main["MMSB/build.rs::main"]
    MMSB_src_ffi_rs__set_last_error["MMSB/src/ffi.rs::set_last_error"]
    MMSB_src_ffi_rs__log_error_code["MMSB/src/ffi.rs::log_error_code"]
    MMSB_src_ffi_rs__mmsb_error_is_retryable["MMSB/src/ffi.rs::mmsb_error_is_retryable"]
    MMSB_src_ffi_rs__mmsb_error_is_fatal["MMSB/src/ffi.rs::mmsb_error_is_fatal"]
    MMSB_src_ffi_rs__mmsb_get_last_error["MMSB/src/ffi.rs::mmsb_get_last_error"]
    MMSB_src_ffi_rs__convert_location["MMSB/src/ffi.rs::convert_location"]
    MMSB_src_ffi_rs__mask_from_bytes["MMSB/src/ffi.rs::mask_from_bytes"]
    MMSB_src_ffi_rs__vec_from_ptr["MMSB/src/ffi.rs::vec_from_ptr"]
    MMSB_src_ffi_rs__slice_from_ptr["MMSB/src/ffi.rs::slice_from_ptr"]
    MMSB_src_ffi_rs__mmsb_page_read["MMSB/src/ffi.rs::mmsb_page_read"]
    MMSB_src_ffi_rs__mmsb_page_epoch["MMSB/src/ffi.rs::mmsb_page_epoch"]
    MMSB_src_ffi_rs__mmsb_page_write_masked["MMSB/src/ffi.rs::mmsb_page_write_masked"]
    MMSB_src_ffi_rs__mmsb_page_metadata_size["MMSB/src/ffi.rs::mmsb_page_metadata_size"]
    MMSB_src_ffi_rs__mmsb_page_metadata_export["MMSB/src/ffi.rs::mmsb_page_metadata_export"]
    MMSB_src_ffi_rs__mmsb_page_metadata_import["MMSB/src/ffi.rs::mmsb_page_metadata_import"]
    MMSB_src_ffi_rs__mmsb_delta_new["MMSB/src/ffi.rs::mmsb_delta_new"]
    MMSB_src_ffi_rs__mmsb_delta_free["MMSB/src/ffi.rs::mmsb_delta_free"]
    MMSB_src_ffi_rs__mmsb_delta_apply["MMSB/src/ffi.rs::mmsb_delta_apply"]
    MMSB_src_ffi_rs__mmsb_delta_id["MMSB/src/ffi.rs::mmsb_delta_id"]
    MMSB_src_ffi_rs__mmsb_delta_page_id["MMSB/src/ffi.rs::mmsb_delta_page_id"]
    MMSB_src_ffi_rs__mmsb_delta_epoch["MMSB/src/ffi.rs::mmsb_delta_epoch"]
    MMSB_src_ffi_rs__mmsb_delta_is_sparse["MMSB/src/ffi.rs::mmsb_delta_is_sparse"]
    MMSB_src_ffi_rs__mmsb_delta_timestamp["MMSB/src/ffi.rs::mmsb_delta_timestamp"]
    MMSB_src_ffi_rs__mmsb_delta_source_len["MMSB/src/ffi.rs::mmsb_delta_source_len"]
    MMSB_src_ffi_rs__mmsb_delta_copy_source["MMSB/src/ffi.rs::mmsb_delta_copy_source"]
    MMSB_src_ffi_rs__mmsb_delta_mask_len["MMSB/src/ffi.rs::mmsb_delta_mask_len"]
    MMSB_src_ffi_rs__mmsb_delta_copy_mask["MMSB/src/ffi.rs::mmsb_delta_copy_mask"]
    MMSB_src_ffi_rs__mmsb_delta_payload_len["MMSB/src/ffi.rs::mmsb_delta_payload_len"]
    MMSB_src_ffi_rs__mmsb_delta_copy_payload["MMSB/src/ffi.rs::mmsb_delta_copy_payload"]
    MMSB_src_ffi_rs__mmsb_delta_set_intent_metadata["MMSB/src/ffi.rs::mmsb_delta_set_intent_metadata"]
    MMSB_src_ffi_rs__mmsb_delta_intent_metadata_len["MMSB/src/ffi.rs::mmsb_delta_intent_metadata_len"]
    MMSB_src_ffi_rs__mmsb_delta_copy_intent_metadata["MMSB/src/ffi.rs::mmsb_delta_copy_intent_metadata"]
    MMSB_src_ffi_rs__mmsb_tlog_new["MMSB/src/ffi.rs::mmsb_tlog_new"]
    MMSB_src_ffi_rs__mmsb_tlog_free["MMSB/src/ffi.rs::mmsb_tlog_free"]
    MMSB_src_ffi_rs__mmsb_tlog_append["MMSB/src/ffi.rs::mmsb_tlog_append"]
    MMSB_src_ffi_rs__mmsb_checkpoint_write["MMSB/src/ffi.rs::mmsb_checkpoint_write"]
    MMSB_src_ffi_rs__mmsb_checkpoint_load["MMSB/src/ffi.rs::mmsb_checkpoint_load"]
    MMSB_src_ffi_rs__mmsb_tlog_reader_new["MMSB/src/ffi.rs::mmsb_tlog_reader_new"]
    MMSB_src_ffi_rs__mmsb_tlog_reader_free["MMSB/src/ffi.rs::mmsb_tlog_reader_free"]
    MMSB_src_ffi_rs__mmsb_tlog_reader_next["MMSB/src/ffi.rs::mmsb_tlog_reader_next"]
    MMSB_src_ffi_rs__mmsb_tlog_summary["MMSB/src/ffi.rs::mmsb_tlog_summary"]
    MMSB_src_ffi_rs__mmsb_allocator_new["MMSB/src/ffi.rs::mmsb_allocator_new"]
    MMSB_src_ffi_rs__mmsb_allocator_free["MMSB/src/ffi.rs::mmsb_allocator_free"]
    MMSB_src_ffi_rs__mmsb_allocator_allocate["MMSB/src/ffi.rs::mmsb_allocator_allocate"]
    MMSB_src_ffi_rs__mmsb_allocator_release["MMSB/src/ffi.rs::mmsb_allocator_release"]
    MMSB_src_ffi_rs__mmsb_allocator_get_page["MMSB/src/ffi.rs::mmsb_allocator_get_page"]
    MMSB_src_ffi_rs__mmsb_allocator_page_count["MMSB/src/ffi.rs::mmsb_allocator_page_count"]
    MMSB_src_ffi_rs__mmsb_allocator_list_pages["MMSB/src/ffi.rs::mmsb_allocator_list_pages"]
    MMSB_src_ffi_rs__mmsb_semiring_tropical_fold_add["MMSB/src/ffi.rs::mmsb_semiring_tropical_fold_add"]
    MMSB_src_ffi_rs__mmsb_semiring_tropical_fold_mul["MMSB/src/ffi.rs::mmsb_semiring_tropical_fold_mul"]
    MMSB_src_ffi_rs__mmsb_semiring_tropical_accumulate["MMSB/src/ffi.rs::mmsb_semiring_tropical_accumulate"]
    MMSB_src_ffi_rs__mmsb_semiring_boolean_fold_add["MMSB/src/ffi.rs::mmsb_semiring_boolean_fold_add"]
    MMSB_src_ffi_rs__mmsb_semiring_boolean_fold_mul["MMSB/src/ffi.rs::mmsb_semiring_boolean_fold_mul"]
    MMSB_src_ffi_rs__mmsb_semiring_boolean_accumulate["MMSB/src/ffi.rs::mmsb_semiring_boolean_accumulate"]
    MMSB_tests_delta_validation_rs__dense_delta["MMSB/tests/delta_validation.rs::dense_delta"]
    MMSB_tests_delta_validation_rs__validates_dense_lengths["MMSB/tests/delta_validation.rs::validates_dense_lengths"]
    MMSB_tests_delta_validation_rs__rejects_mismatched_dense_lengths["MMSB/tests/delta_validation.rs::rejects_mismatched_dense_lengths"]
    MMSB_tests_examples_basic_rs__example_page_allocation["MMSB/tests/examples_basic.rs::example_page_allocation"]
    MMSB_tests_examples_basic_rs__example_delta_operations["MMSB/tests/examples_basic.rs::example_delta_operations"]
    MMSB_tests_examples_basic_rs__example_checkpoint["MMSB/tests/examples_basic.rs::example_checkpoint"]
    MMSB_tests_mmsb_tests_rs__read_page["MMSB/tests/mmsb_tests.rs::read_page"]
    MMSB_tests_mmsb_tests_rs__test_page_info_metadata_roundtrip["MMSB/tests/mmsb_tests.rs::test_page_info_metadata_roundtrip"]
    MMSB_tests_mmsb_tests_rs__test_page_snapshot_and_restore["MMSB/tests/mmsb_tests.rs::test_page_snapshot_and_restore"]
    MMSB_tests_mmsb_tests_rs__test_thread_safe_allocator["MMSB/tests/mmsb_tests.rs::test_thread_safe_allocator"]
    MMSB_tests_mmsb_tests_rs__test_gpu_delta_kernels["MMSB/tests/mmsb_tests.rs::test_gpu_delta_kernels"]
    MMSB_tests_mmsb_tests_rs__test_checkpoint_log_and_restore["MMSB/tests/mmsb_tests.rs::test_checkpoint_log_and_restore"]
    MMSB_tests_mmsb_tests_rs__test_invalid_page_deletion_is_safe["MMSB/tests/mmsb_tests.rs::test_invalid_page_deletion_is_safe"]
    MMSB_tests_mmsb_tests_rs__test_sparse_delta_application["MMSB/tests/mmsb_tests.rs::test_sparse_delta_application"]
    MMSB_tests_mmsb_tests_rs__test_dense_delta_application["MMSB/tests/mmsb_tests.rs::test_dense_delta_application"]
    MMSB_tests_mmsb_tests_rs__test_api_public_interface["MMSB/tests/mmsb_tests.rs::test_api_public_interface"]
    MMSB_tests_week27_31_integration_rs__test_allocator_cpu_gpu_latency["MMSB/tests/week27_31_integration.rs::test_allocator_cpu_gpu_latency"]
    MMSB_tests_week27_31_integration_rs__test_semiring_operations_tropical["MMSB/tests/week27_31_integration.rs::test_semiring_operations_tropical"]
    MMSB_tests_week27_31_integration_rs__test_delta_merge_simd["MMSB/tests/week27_31_integration.rs::test_delta_merge_simd"]
    MMSB_tests_week27_31_integration_rs__test_lockfree_allocator["MMSB/tests/week27_31_integration.rs::test_lockfree_allocator"]
    MMSB_tests_week27_31_integration_rs__test_propagation_queue["MMSB/tests/week27_31_integration.rs::test_propagation_queue"]
    MMSB_tests_week27_31_integration_rs__test_cpu_features["MMSB/tests/week27_31_integration.rs::test_cpu_features"]
    MMSB_src_02_semiring_semiring_ops_rs__fold_add["MMSB/src/02_semiring/semiring_ops.rs::fold_add"]
    MMSB_src_02_semiring_semiring_ops_rs__fold_mul["MMSB/src/02_semiring/semiring_ops.rs::fold_mul"]
    MMSB_src_02_semiring_semiring_ops_rs__accumulate["MMSB/src/02_semiring/semiring_ops.rs::accumulate"]
    MMSB_src_05_adaptive_locality_optimizer_rs__test_locality_optimizer["MMSB/src/05_adaptive/locality_optimizer.rs::test_locality_optimizer"]
    MMSB_src_05_adaptive_memory_layout_rs__test_memory_layout_creation["MMSB/src/05_adaptive/memory_layout.rs::test_memory_layout_creation"]
    MMSB_src_05_adaptive_memory_layout_rs__test_locality_cost_empty["MMSB/src/05_adaptive/memory_layout.rs::test_locality_cost_empty"]
    MMSB_src_05_adaptive_memory_layout_rs__test_optimize_layout["MMSB/src/05_adaptive/memory_layout.rs::test_optimize_layout"]
    MMSB_src_05_adaptive_page_clustering_rs__test_page_clustering["MMSB/src/05_adaptive/page_clustering.rs::test_page_clustering"]
    MMSB_src_06_utility_cpu_features_rs__cpu_has_avx2["MMSB/src/06_utility/cpu_features.rs::cpu_has_avx2"]
    MMSB_src_06_utility_cpu_features_rs__cpu_has_avx512["MMSB/src/06_utility/cpu_features.rs::cpu_has_avx512"]
    MMSB_src_06_utility_cpu_features_rs__cpu_has_sse42["MMSB/src/06_utility/cpu_features.rs::cpu_has_sse42"]
    MMSB_src_06_utility_telemetry_rs__test_telemetry_basic["MMSB/src/06_utility/telemetry.rs::test_telemetry_basic"]
    MMSB_src_06_utility_telemetry_rs__test_cache_hit_rate["MMSB/src/06_utility/telemetry.rs::test_cache_hit_rate"]
    MMSB_src_06_utility_telemetry_rs__test_reset["MMSB/src/06_utility/telemetry.rs::test_reset"]
    MMSB_src_00_physical_allocator_rs__test_page_info_metadata_roundtrip["MMSB/src/00_physical/allocator.rs::test_page_info_metadata_roundtrip"]
    MMSB_src_00_physical_allocator_rs__test_unified_page["MMSB/src/00_physical/allocator.rs::test_unified_page"]
    MMSB_src_00_physical_allocator_rs__test_checkpoint_roundtrip_in_memory["MMSB/src/00_physical/allocator.rs::test_checkpoint_roundtrip_in_memory"]
    MMSB_src_01_page_checkpoint_rs__write_checkpoint["MMSB/src/01_page/checkpoint.rs::write_checkpoint"]
    MMSB_src_01_page_checkpoint_rs__load_checkpoint["MMSB/src/01_page/checkpoint.rs::load_checkpoint"]
    MMSB_src_01_page_delta_rs__now_ns["MMSB/src/01_page/delta.rs::now_ns"]
    MMSB_src_01_page_delta_merge_rs__merge_deltas["MMSB/src/01_page/delta_merge.rs::merge_deltas"]
    MMSB_src_01_page_delta_merge_rs__merge_dense_avx2["MMSB/src/01_page/delta_merge.rs::merge_dense_avx2"]
    MMSB_src_01_page_delta_merge_rs__merge_dense_avx512["MMSB/src/01_page/delta_merge.rs::merge_dense_avx512"]
    MMSB_src_01_page_delta_merge_rs__merge_dense_simd["MMSB/src/01_page/delta_merge.rs::merge_dense_simd"]
    MMSB_src_01_page_delta_validation_rs__validate_delta["MMSB/src/01_page/delta_validation.rs::validate_delta"]
    MMSB_src_01_page_page_rs__read_u32["MMSB/src/01_page/page.rs::read_u32"]
    MMSB_src_01_page_page_rs__read_bytes["MMSB/src/01_page/page.rs::read_bytes"]
    MMSB_src_01_page_page_rs__allocate_zeroed["MMSB/src/01_page/page.rs::allocate_zeroed"]
    MMSB_src_01_page_simd_mask_rs__generate_mask["MMSB/src/01_page/simd_mask.rs::generate_mask"]
    MMSB_src_01_page_tlog_rs__summary["MMSB/src/01_page/tlog.rs::summary"]
    MMSB_src_01_page_tlog_rs__serialize_frame["MMSB/src/01_page/tlog.rs::serialize_frame"]
    MMSB_src_01_page_tlog_rs__read_frame["MMSB/src/01_page/tlog.rs::read_frame"]
    MMSB_src_01_page_tlog_rs__validate_header["MMSB/src/01_page/tlog.rs::validate_header"]
    MMSB_src_01_page_tlog_compression_rs__encode_rle["MMSB/src/01_page/tlog_compression.rs::encode_rle"]
    MMSB_src_01_page_tlog_compression_rs__decode_rle["MMSB/src/01_page/tlog_compression.rs::decode_rle"]
    MMSB_src_01_page_tlog_compression_rs__bitpack_mask["MMSB/src/01_page/tlog_compression.rs::bitpack_mask"]
    MMSB_src_01_page_tlog_compression_rs__bitunpack_mask["MMSB/src/01_page/tlog_compression.rs::bitunpack_mask"]
    MMSB_src_01_page_tlog_compression_rs__compress_delta_mask["MMSB/src/01_page/tlog_compression.rs::compress_delta_mask"]
    MMSB_src_01_page_tlog_compression_rs__compact["MMSB/src/01_page/tlog_compression.rs::compact"]
    MMSB_src_01_page_tlog_replay_rs__apply_log["MMSB/src/01_page/tlog_replay.rs::apply_log"]
    MMSB_src_01_page_tlog_serialization_rs__read_log["MMSB/src/01_page/tlog_serialization.rs::read_log"]
    MMSB_src_03_dag_cycle_detection_rs__has_cycle["MMSB/src/03_dag/cycle_detection.rs::has_cycle"]
    MMSB_src_03_dag_cycle_detection_rs__dfs["MMSB/src/03_dag/cycle_detection.rs::dfs"]
    MMSB_src_03_dag_shadow_graph_traversal_rs__topological_sort["MMSB/src/03_dag/shadow_graph_traversal.rs::topological_sort"]
    MMSB_src_04_propagation_propagation_fastpath_rs__passthrough["MMSB/src/04_propagation/propagation_fastpath.rs::passthrough"]
    MMSB_src_04_propagation_sparse_message_passing_rs__enqueue_sparse["MMSB/src/04_propagation/sparse_message_passing.rs::enqueue_sparse"]
    MMSB_src_API_jl__mmsb_start["MMSB/src/API.jl::mmsb_start"]
    MMSB_src_API_jl__mmsb_stop["MMSB/src/API.jl::mmsb_stop"]
    MMSB_src_API_jl___resolve_location["MMSB/src/API.jl::_resolve_location"]
    MMSB_src_API_jl__create_page["MMSB/src/API.jl::create_page"]
    MMSB_src_API_jl__update_page["MMSB/src/API.jl::update_page"]
    MMSB_src_API_jl__length["MMSB/src/API.jl::length"]
    MMSB_src_API_jl__query_page["MMSB/src/API.jl::query_page"]
    MMSB_benchmark_benchmarks_jl___start_state["MMSB/benchmark/benchmarks.jl::_start_state"]
    MMSB_benchmark_benchmarks_jl___stop_state_["MMSB/benchmark/benchmarks.jl::_stop_state!"]
    MMSB_benchmark_benchmarks_jl___page["MMSB/benchmark/benchmarks.jl::_page"]
    MMSB_benchmark_benchmarks_jl___populate_pages_["MMSB/benchmark/benchmarks.jl::_populate_pages!"]
    MMSB_benchmark_benchmarks_jl___seed_pages_["MMSB/benchmark/benchmarks.jl::_seed_pages!"]
    MMSB_benchmark_benchmarks_jl___replay_sequence_["MMSB/benchmark/benchmarks.jl::_replay_sequence!"]
    MMSB_benchmark_benchmarks_jl___stress_updates_["MMSB/benchmark/benchmarks.jl::_stress_updates!"]
    MMSB_benchmark_benchmarks_jl___link_chain_["MMSB/benchmark/benchmarks.jl::_link_chain!"]
    MMSB_benchmark_benchmarks_jl___checkpoint["MMSB/benchmark/benchmarks.jl::_checkpoint"]
    MMSB_benchmark_benchmarks_jl___measure_ns["MMSB/benchmark/benchmarks.jl::_measure_ns"]
    MMSB_benchmark_benchmarks_jl___graph_fixture["MMSB/benchmark/benchmarks.jl::_graph_fixture"]
    MMSB_benchmark_benchmarks_jl___graph_bfs["MMSB/benchmark/benchmarks.jl::_graph_bfs"]
    MMSB_benchmark_benchmarks_jl___build_batch_deltas["MMSB/benchmark/benchmarks.jl::_build_batch_deltas"]
    MMSB_benchmark_benchmarks_jl___full_system_benchmark_["MMSB/benchmark/benchmarks.jl::_full_system_benchmark!"]
    MMSB_benchmark_benchmarks_jl___trial_to_dict["MMSB/benchmark/benchmarks.jl::_trial_to_dict"]
    MMSB_benchmark_benchmarks_jl___select_suite["MMSB/benchmark/benchmarks.jl::_select_suite"]
    MMSB_benchmark_benchmarks_jl___collect_instrumentation_report["MMSB/benchmark/benchmarks.jl::_collect_instrumentation_report"]
    MMSB_benchmark_benchmarks_jl___to_mutable["MMSB/benchmark/benchmarks.jl::_to_mutable"]
    MMSB_benchmark_benchmarks_jl__run_benchmarks["MMSB/benchmark/benchmarks.jl::run_benchmarks"]
    MMSB_benchmark_benchmarks_jl__compare_with_baseline["MMSB/benchmark/benchmarks.jl::compare_with_baseline"]
    MMSB_benchmark_helpers_jl___format_time["MMSB/benchmark/helpers.jl::_format_time"]
    MMSB_benchmark_helpers_jl___format_bytes["MMSB/benchmark/helpers.jl::_format_bytes"]
    MMSB_benchmark_helpers_jl__analyze_results["MMSB/benchmark/helpers.jl::analyze_results"]
    MMSB_benchmark_helpers_jl__check_performance_targets["MMSB/benchmark/helpers.jl::check_performance_targets"]
    MMSB_src_ffi_FFIWrapper_jl__register_error_hook["MMSB/src/ffi/FFIWrapper.jl::register_error_hook"]
    MMSB_src_ffi_FFIWrapper_jl___check_rust_error["MMSB/src/ffi/FFIWrapper.jl::_check_rust_error"]
    MMSB_src_ffi_FFIWrapper_jl__rust_artifacts_available["MMSB/src/ffi/FFIWrapper.jl::rust_artifacts_available"]
    MMSB_src_ffi_FFIWrapper_jl__ensure_rust_artifacts["MMSB/src/ffi/FFIWrapper.jl::ensure_rust_artifacts"]
    MMSB_src_ffi_FFIWrapper_jl__rust_page_read_["MMSB/src/ffi/FFIWrapper.jl::rust_page_read!"]
    MMSB_src_ffi_FFIWrapper_jl__rust_page_epoch["MMSB/src/ffi/FFIWrapper.jl::rust_page_epoch"]
    MMSB_src_ffi_FFIWrapper_jl__rust_page_metadata_blob["MMSB/src/ffi/FFIWrapper.jl::rust_page_metadata_blob"]
    MMSB_src_ffi_FFIWrapper_jl__rust_page_metadata_import_["MMSB/src/ffi/FFIWrapper.jl::rust_page_metadata_import!"]
    MMSB_src_ffi_FFIWrapper_jl__rust_page_write_masked_["MMSB/src/ffi/FFIWrapper.jl::rust_page_write_masked!"]
    MMSB_src_ffi_FFIWrapper_jl__rust_delta_new["MMSB/src/ffi/FFIWrapper.jl::rust_delta_new"]
    MMSB_src_ffi_FFIWrapper_jl__rust_delta_free_["MMSB/src/ffi/FFIWrapper.jl::rust_delta_free!"]
    MMSB_src_ffi_FFIWrapper_jl__rust_delta_apply_["MMSB/src/ffi/FFIWrapper.jl::rust_delta_apply!"]
    MMSB_src_ffi_FFIWrapper_jl__rust_allocator_new["MMSB/src/ffi/FFIWrapper.jl::rust_allocator_new"]
    MMSB_src_ffi_FFIWrapper_jl__rust_allocator_free_["MMSB/src/ffi/FFIWrapper.jl::rust_allocator_free!"]
    MMSB_src_ffi_FFIWrapper_jl__rust_allocator_allocate["MMSB/src/ffi/FFIWrapper.jl::rust_allocator_allocate"]
    MMSB_src_ffi_FFIWrapper_jl__rust_allocator_release_["MMSB/src/ffi/FFIWrapper.jl::rust_allocator_release!"]
    MMSB_src_ffi_FFIWrapper_jl__rust_allocator_get_page["MMSB/src/ffi/FFIWrapper.jl::rust_allocator_get_page"]
    MMSB_src_ffi_FFIWrapper_jl__rust_tlog_new["MMSB/src/ffi/FFIWrapper.jl::rust_tlog_new"]
    MMSB_src_ffi_FFIWrapper_jl__rust_tlog_free_["MMSB/src/ffi/FFIWrapper.jl::rust_tlog_free!"]
    MMSB_src_ffi_FFIWrapper_jl__rust_tlog_append_["MMSB/src/ffi/FFIWrapper.jl::rust_tlog_append!"]
    MMSB_src_ffi_FFIWrapper_jl__rust_tlog_reader_new["MMSB/src/ffi/FFIWrapper.jl::rust_tlog_reader_new"]
    MMSB_src_ffi_FFIWrapper_jl__rust_tlog_reader_free_["MMSB/src/ffi/FFIWrapper.jl::rust_tlog_reader_free!"]
    MMSB_src_ffi_FFIWrapper_jl__rust_tlog_reader_next["MMSB/src/ffi/FFIWrapper.jl::rust_tlog_reader_next"]
    MMSB_src_ffi_FFIWrapper_jl__rust_tlog_summary["MMSB/src/ffi/FFIWrapper.jl::rust_tlog_summary"]
    MMSB_src_ffi_FFIWrapper_jl__rust_delta_id["MMSB/src/ffi/FFIWrapper.jl::rust_delta_id"]
    MMSB_src_ffi_FFIWrapper_jl__rust_delta_page_id["MMSB/src/ffi/FFIWrapper.jl::rust_delta_page_id"]
    MMSB_src_ffi_FFIWrapper_jl__rust_delta_epoch["MMSB/src/ffi/FFIWrapper.jl::rust_delta_epoch"]
    MMSB_src_ffi_FFIWrapper_jl__rust_delta_is_sparse["MMSB/src/ffi/FFIWrapper.jl::rust_delta_is_sparse"]
    MMSB_src_ffi_FFIWrapper_jl__rust_delta_timestamp["MMSB/src/ffi/FFIWrapper.jl::rust_delta_timestamp"]
    MMSB_src_ffi_FFIWrapper_jl__rust_delta_source["MMSB/src/ffi/FFIWrapper.jl::rust_delta_source"]
    MMSB_src_ffi_FFIWrapper_jl__rust_delta_mask["MMSB/src/ffi/FFIWrapper.jl::rust_delta_mask"]
    MMSB_src_ffi_FFIWrapper_jl__rust_delta_payload["MMSB/src/ffi/FFIWrapper.jl::rust_delta_payload"]
    MMSB_src_ffi_FFIWrapper_jl__rust_delta_set_intent_metadata_["MMSB/src/ffi/FFIWrapper.jl::rust_delta_set_intent_metadata!"]
    MMSB_src_ffi_FFIWrapper_jl__rust_delta_intent_metadata["MMSB/src/ffi/FFIWrapper.jl::rust_delta_intent_metadata"]
    MMSB_src_ffi_FFIWrapper_jl__rust_checkpoint_write_["MMSB/src/ffi/FFIWrapper.jl::rust_checkpoint_write!"]
    MMSB_src_ffi_FFIWrapper_jl__rust_checkpoint_load_["MMSB/src/ffi/FFIWrapper.jl::rust_checkpoint_load!"]
    MMSB_src_ffi_FFIWrapper_jl__rust_allocator_page_infos["MMSB/src/ffi/FFIWrapper.jl::rust_allocator_page_infos"]
    MMSB_src_ffi_FFIWrapper_jl__rust_allocator_acquire_page["MMSB/src/ffi/FFIWrapper.jl::rust_allocator_acquire_page"]
    MMSB_src_ffi_FFIWrapper_jl__rust_get_last_error["MMSB/src/ffi/FFIWrapper.jl::rust_get_last_error"]
    MMSB_src_ffi_FFIWrapper_jl__rust_semiring_tropical_fold_add["MMSB/src/ffi/FFIWrapper.jl::rust_semiring_tropical_fold_add"]
    MMSB_src_ffi_FFIWrapper_jl__rust_semiring_tropical_fold_mul["MMSB/src/ffi/FFIWrapper.jl::rust_semiring_tropical_fold_mul"]
    MMSB_src_ffi_FFIWrapper_jl__rust_semiring_tropical_accumulate["MMSB/src/ffi/FFIWrapper.jl::rust_semiring_tropical_accumulate"]
    MMSB_src_ffi_FFIWrapper_jl__rust_semiring_boolean_fold_add["MMSB/src/ffi/FFIWrapper.jl::rust_semiring_boolean_fold_add"]
    MMSB_src_ffi_FFIWrapper_jl__rust_semiring_boolean_fold_mul["MMSB/src/ffi/FFIWrapper.jl::rust_semiring_boolean_fold_mul"]
    MMSB_src_ffi_FFIWrapper_jl__rust_semiring_boolean_accumulate["MMSB/src/ffi/FFIWrapper.jl::rust_semiring_boolean_accumulate"]
    MMSB_src_ffi_FFIWrapper_jl__isnull["MMSB/src/ffi/FFIWrapper.jl::isnull"]
    MMSB_src_ffi_RustErrors_jl__Base_showerror["MMSB/src/ffi/RustErrors.jl::Base.showerror"]
    MMSB_src_ffi_RustErrors_jl__check_rust_error["MMSB/src/ffi/RustErrors.jl::check_rust_error"]
    MMSB_src_ffi_RustErrors_jl___default_message["MMSB/src/ffi/RustErrors.jl::_default_message"]
    MMSB_src_ffi_RustErrors_jl__translate_error["MMSB/src/ffi/RustErrors.jl::translate_error"]
    MMSB_src_00_physical_DeviceFallback_jl__has_gpu_support["MMSB/src/00_physical/DeviceFallback.jl::has_gpu_support"]
    MMSB_src_00_physical_DeviceFallback_jl__CPUPropagationQueue["MMSB/src/00_physical/DeviceFallback.jl::CPUPropagationQueue"]
    MMSB_src_00_physical_DeviceFallback_jl__fallback_to_cpu["MMSB/src/00_physical/DeviceFallback.jl::fallback_to_cpu"]
    MMSB_src_00_physical_DeviceSync_jl__create_gpu_command_buffer["MMSB/src/00_physical/DeviceSync.jl::create_gpu_command_buffer"]
    MMSB_src_00_physical_DeviceSync_jl__enqueue_propagation_command["MMSB/src/00_physical/DeviceSync.jl::enqueue_propagation_command"]
    MMSB_src_00_physical_DeviceSync_jl__wait_gpu_queue["MMSB/src/00_physical/DeviceSync.jl::wait_gpu_queue"]
    MMSB_src_00_physical_DeviceSync_jl__sync_page_to_gpu_["MMSB/src/00_physical/DeviceSync.jl::sync_page_to_gpu!"]
    MMSB_src_00_physical_DeviceSync_jl__sync_page_to_cpu_["MMSB/src/00_physical/DeviceSync.jl::sync_page_to_cpu!"]
    MMSB_src_00_physical_DeviceSync_jl__sync_bidirectional_["MMSB/src/00_physical/DeviceSync.jl::sync_bidirectional!"]
    MMSB_src_00_physical_DeviceSync_jl__ensure_page_on_device_["MMSB/src/00_physical/DeviceSync.jl::ensure_page_on_device!"]
    MMSB_src_00_physical_DeviceSync_jl__async_sync_page_to_gpu_["MMSB/src/00_physical/DeviceSync.jl::async_sync_page_to_gpu!"]
    MMSB_src_00_physical_DeviceSync_jl__batch_sync_to_gpu_["MMSB/src/00_physical/DeviceSync.jl::batch_sync_to_gpu!"]
    MMSB_src_00_physical_DeviceSync_jl__batch_sync_to_cpu_["MMSB/src/00_physical/DeviceSync.jl::batch_sync_to_cpu!"]
    MMSB_src_00_physical_DeviceSync_jl__get_sync_statistics["MMSB/src/00_physical/DeviceSync.jl::get_sync_statistics"]
    MMSB_src_00_physical_DeviceSync_jl__prefetch_pages_to_gpu_["MMSB/src/00_physical/DeviceSync.jl::prefetch_pages_to_gpu!"]
    MMSB_src_00_physical_GPUKernels_jl__delta_merge_kernel_["MMSB/src/00_physical/GPUKernels.jl::delta_merge_kernel!"]
    MMSB_src_00_physical_GPUKernels_jl__launch_delta_merge_["MMSB/src/00_physical/GPUKernels.jl::launch_delta_merge!"]
    MMSB_src_00_physical_GPUKernels_jl__page_copy_kernel_["MMSB/src/00_physical/GPUKernels.jl::page_copy_kernel!"]
    MMSB_src_00_physical_GPUKernels_jl__launch_page_copy_["MMSB/src/00_physical/GPUKernels.jl::launch_page_copy!"]
    MMSB_src_00_physical_GPUKernels_jl__page_zero_kernel_["MMSB/src/00_physical/GPUKernels.jl::page_zero_kernel!"]
    MMSB_src_00_physical_GPUKernels_jl__launch_page_zero_["MMSB/src/00_physical/GPUKernels.jl::launch_page_zero!"]
    MMSB_src_00_physical_GPUKernels_jl__page_compare_kernel_["MMSB/src/00_physical/GPUKernels.jl::page_compare_kernel!"]
    MMSB_src_00_physical_GPUKernels_jl__launch_page_compare_["MMSB/src/00_physical/GPUKernels.jl::launch_page_compare!"]
    MMSB_src_00_physical_GPUKernels_jl__sparse_delta_apply_kernel_["MMSB/src/00_physical/GPUKernels.jl::sparse_delta_apply_kernel!"]
    MMSB_src_00_physical_GPUKernels_jl__launch_sparse_delta_apply_["MMSB/src/00_physical/GPUKernels.jl::launch_sparse_delta_apply!"]
    MMSB_src_00_physical_GPUKernels_jl__compute_optimal_kernel_config["MMSB/src/00_physical/GPUKernels.jl::compute_optimal_kernel_config"]
    MMSB_src_00_physical_PageAllocator_jl__create_page_["MMSB/src/00_physical/PageAllocator.jl::create_page!"]
    MMSB_src_00_physical_PageAllocator_jl__delete_page_["MMSB/src/00_physical/PageAllocator.jl::delete_page!"]
    MMSB_src_00_physical_PageAllocator_jl__migrate_page_["MMSB/src/00_physical/PageAllocator.jl::migrate_page!"]
    MMSB_src_00_physical_PageAllocator_jl__resize_page_["MMSB/src/00_physical/PageAllocator.jl::resize_page!"]
    MMSB_src_00_physical_PageAllocator_jl__allocate_page_arrays["MMSB/src/00_physical/PageAllocator.jl::allocate_page_arrays"]
    MMSB_src_00_physical_PageAllocator_jl__clone_page["MMSB/src/00_physical/PageAllocator.jl::clone_page"]
    MMSB_src_00_physical_UnifiedMemory_jl__GPUMemoryPool["MMSB/src/00_physical/UnifiedMemory.jl::GPUMemoryPool"]
    MMSB_src_00_physical_UnifiedMemory_jl__allocate_from_pool["MMSB/src/00_physical/UnifiedMemory.jl::allocate_from_pool"]
    MMSB_src_00_physical_UnifiedMemory_jl__deallocate_to_pool["MMSB/src/00_physical/UnifiedMemory.jl::deallocate_to_pool"]
    MMSB_src_00_physical_UnifiedMemory_jl__get_pool_stats["MMSB/src/00_physical/UnifiedMemory.jl::get_pool_stats"]
    MMSB_src_00_physical_UnifiedMemory_jl__is_unified_memory_available["MMSB/src/00_physical/UnifiedMemory.jl::is_unified_memory_available"]
    MMSB_src_00_physical_UnifiedMemory_jl__create_unified_page_["MMSB/src/00_physical/UnifiedMemory.jl::create_unified_page!"]
    MMSB_src_00_physical_UnifiedMemory_jl__prefetch_unified_to_gpu_["MMSB/src/00_physical/UnifiedMemory.jl::prefetch_unified_to_gpu!"]
    MMSB_src_00_physical_UnifiedMemory_jl__prefetch_unified_to_cpu_["MMSB/src/00_physical/UnifiedMemory.jl::prefetch_unified_to_cpu!"]
    MMSB_src_00_physical_UnifiedMemory_jl__adaptive_prefetch_distance["MMSB/src/00_physical/UnifiedMemory.jl::adaptive_prefetch_distance"]
    MMSB_src_00_physical_UnifiedMemory_jl__set_preferred_location_["MMSB/src/00_physical/UnifiedMemory.jl::set_preferred_location!"]
    MMSB_src_00_physical_UnifiedMemory_jl__convert_to_unified_["MMSB/src/00_physical/UnifiedMemory.jl::convert_to_unified!"]
    MMSB_src_00_physical_UnifiedMemory_jl__enable_read_mostly_hint_["MMSB/src/00_physical/UnifiedMemory.jl::enable_read_mostly_hint!"]
    MMSB_src_00_physical_UnifiedMemory_jl__disable_read_mostly_hint_["MMSB/src/00_physical/UnifiedMemory.jl::disable_read_mostly_hint!"]
    MMSB_src_01_page_Delta_jl__Delta["MMSB/src/01_page/Delta.jl::Delta"]
    MMSB_src_01_page_Delta_jl__Delta["MMSB/src/01_page/Delta.jl::Delta"]
    MMSB_src_01_page_Delta_jl__new_delta_handle["MMSB/src/01_page/Delta.jl::new_delta_handle"]
    MMSB_src_01_page_Delta_jl__apply_delta_["MMSB/src/01_page/Delta.jl::apply_delta!"]
    MMSB_src_01_page_Delta_jl__dense_data["MMSB/src/01_page/Delta.jl::dense_data"]
    MMSB_src_01_page_Delta_jl__serialize_delta["MMSB/src/01_page/Delta.jl::serialize_delta"]
    MMSB_src_01_page_Delta_jl__deserialize_delta["MMSB/src/01_page/Delta.jl::deserialize_delta"]
    MMSB_src_01_page_Delta_jl__set_intent_metadata_["MMSB/src/01_page/Delta.jl::set_intent_metadata!"]
    MMSB_src_01_page_Delta_jl__intent_metadata["MMSB/src/01_page/Delta.jl::intent_metadata"]
    MMSB_src_01_page_Delta_jl___encode_metadata_value["MMSB/src/01_page/Delta.jl::_encode_metadata_value"]
    MMSB_src_01_page_Delta_jl___encode_metadata_dict["MMSB/src/01_page/Delta.jl::_encode_metadata_dict"]
    MMSB_src_01_page_Delta_jl___escape_metadata_string["MMSB/src/01_page/Delta.jl::_escape_metadata_string"]
    MMSB_src_01_page_Delta_jl__merge_deltas_simd_["MMSB/src/01_page/Delta.jl::merge_deltas_simd!"]
    MMSB_src_01_page_Delta_jl___decode_metadata["MMSB/src/01_page/Delta.jl::_decode_metadata"]
    MMSB_src_01_page_Delta_jl___parse_metadata_value["MMSB/src/01_page/Delta.jl::_parse_metadata_value"]
    MMSB_src_01_page_Delta_jl___parse_metadata_object["MMSB/src/01_page/Delta.jl::_parse_metadata_object"]
    MMSB_src_01_page_Delta_jl___parse_metadata_array["MMSB/src/01_page/Delta.jl::_parse_metadata_array"]
    MMSB_src_01_page_Delta_jl___parse_metadata_string["MMSB/src/01_page/Delta.jl::_parse_metadata_string"]
    MMSB_src_01_page_Delta_jl___parse_metadata_number["MMSB/src/01_page/Delta.jl::_parse_metadata_number"]
    MMSB_src_01_page_Delta_jl___skip_ws["MMSB/src/01_page/Delta.jl::_skip_ws"]
    MMSB_src_01_page_Delta_jl___consume["MMSB/src/01_page/Delta.jl::_consume"]
    MMSB_src_01_page_Delta_jl___peek["MMSB/src/01_page/Delta.jl::_peek"]
    MMSB_src_01_page_Page_jl__Page["MMSB/src/01_page/Page.jl::Page"]
    MMSB_src_01_page_Page_jl__is_gpu_page["MMSB/src/01_page/Page.jl::is_gpu_page"]
    MMSB_src_01_page_Page_jl__is_cpu_page["MMSB/src/01_page/Page.jl::is_cpu_page"]
    MMSB_src_01_page_Page_jl__page_size_bytes["MMSB/src/01_page/Page.jl::page_size_bytes"]
    MMSB_src_01_page_Page_jl__initialize_["MMSB/src/01_page/Page.jl::initialize!"]
    MMSB_src_01_page_Page_jl__activate_["MMSB/src/01_page/Page.jl::activate!"]
    MMSB_src_01_page_Page_jl__deactivate_["MMSB/src/01_page/Page.jl::deactivate!"]
    MMSB_src_01_page_Page_jl__read_page["MMSB/src/01_page/Page.jl::read_page"]
    MMSB_src_01_page_Page_jl___apply_metadata_["MMSB/src/01_page/Page.jl::_apply_metadata!"]
    MMSB_src_01_page_Page_jl___encode_metadata_dict["MMSB/src/01_page/Page.jl::_encode_metadata_dict"]
    MMSB_src_01_page_Page_jl___coerce_metadata_value["MMSB/src/01_page/Page.jl::_coerce_metadata_value"]
    MMSB_src_01_page_Page_jl___decode_metadata_blob["MMSB/src/01_page/Page.jl::_decode_metadata_blob"]
    MMSB_src_01_page_Page_jl__metadata_from_blob["MMSB/src/01_page/Page.jl::metadata_from_blob"]
    MMSB_src_01_page_ReplayEngine_jl___blank_state_like["MMSB/src/01_page/ReplayEngine.jl::_blank_state_like"]
    MMSB_src_01_page_ReplayEngine_jl___apply_delta_["MMSB/src/01_page/ReplayEngine.jl::_apply_delta!"]
    MMSB_src_01_page_ReplayEngine_jl___all_deltas["MMSB/src/01_page/ReplayEngine.jl::_all_deltas"]
    MMSB_src_01_page_ReplayEngine_jl__replay_to_epoch["MMSB/src/01_page/ReplayEngine.jl::replay_to_epoch"]
    MMSB_src_01_page_ReplayEngine_jl__replay_to_timestamp["MMSB/src/01_page/ReplayEngine.jl::replay_to_timestamp"]
    MMSB_src_01_page_ReplayEngine_jl__replay_from_checkpoint["MMSB/src/01_page/ReplayEngine.jl::replay_from_checkpoint"]
    MMSB_src_01_page_ReplayEngine_jl__replay_page_history["MMSB/src/01_page/ReplayEngine.jl::replay_page_history"]
    MMSB_src_01_page_ReplayEngine_jl__verify_state_consistency["MMSB/src/01_page/ReplayEngine.jl::verify_state_consistency"]
    MMSB_src_01_page_ReplayEngine_jl__replay_with_predicate["MMSB/src/01_page/ReplayEngine.jl::replay_with_predicate"]
    MMSB_src_01_page_ReplayEngine_jl__incremental_replay_["MMSB/src/01_page/ReplayEngine.jl::incremental_replay!"]
    MMSB_src_01_page_ReplayEngine_jl__compute_diff["MMSB/src/01_page/ReplayEngine.jl::compute_diff"]
    MMSB_src_01_page_TLog_jl__compress_delta_mask["MMSB/src/01_page/TLog.jl::compress_delta_mask"]
    MMSB_src_01_page_TLog_jl___with_rust_errors["MMSB/src/01_page/TLog.jl::_with_rust_errors"]
    MMSB_src_01_page_TLog_jl__append_to_log_["MMSB/src/01_page/TLog.jl::append_to_log!"]
    MMSB_src_01_page_TLog_jl__log_summary["MMSB/src/01_page/TLog.jl::log_summary"]
    MMSB_src_01_page_TLog_jl___iterate_log["MMSB/src/01_page/TLog.jl::_iterate_log"]
    MMSB_src_01_page_TLog_jl__query_log["MMSB/src/01_page/TLog.jl::query_log"]
    MMSB_src_01_page_TLog_jl__get_deltas_for_page["MMSB/src/01_page/TLog.jl::get_deltas_for_page"]
    MMSB_src_01_page_TLog_jl__get_deltas_in_range["MMSB/src/01_page/TLog.jl::get_deltas_in_range"]
    MMSB_src_01_page_TLog_jl__compute_log_statistics["MMSB/src/01_page/TLog.jl::compute_log_statistics"]
    MMSB_src_01_page_TLog_jl__replay_log["MMSB/src/01_page/TLog.jl::replay_log"]
    MMSB_src_01_page_TLog_jl__checkpoint_log_["MMSB/src/01_page/TLog.jl::checkpoint_log!"]
    MMSB_src_01_page_TLog_jl__load_checkpoint_["MMSB/src/01_page/TLog.jl::load_checkpoint!"]
    MMSB_src_01_page_TLog_jl___refresh_pages_["MMSB/src/01_page/TLog.jl::_refresh_pages!"]
    MMSB_src_01_types_Errors_jl__Base_showerror["MMSB/src/01_types/Errors.jl::Base.showerror"]
    MMSB_src_01_types_MMSBState_jl__MMSBConfig["MMSB/src/01_types/MMSBState.jl::MMSBConfig"]
    MMSB_src_01_types_MMSBState_jl__MMSBState["MMSB/src/01_types/MMSBState.jl::MMSBState"]
    MMSB_src_01_types_MMSBState_jl__MMSBState["MMSB/src/01_types/MMSBState.jl::MMSBState"]
    MMSB_src_01_types_MMSBState_jl__allocate_page_id_["MMSB/src/01_types/MMSBState.jl::allocate_page_id!"]
    MMSB_src_01_types_MMSBState_jl__allocate_delta_id_["MMSB/src/01_types/MMSBState.jl::allocate_delta_id!"]
    MMSB_src_01_types_MMSBState_jl__get_page["MMSB/src/01_types/MMSBState.jl::get_page"]
    MMSB_src_01_types_MMSBState_jl__register_page_["MMSB/src/01_types/MMSBState.jl::register_page!"]
    MMSB_src_02_semiring_DeltaRouter_jl__route_delta_["MMSB/src/02_semiring/DeltaRouter.jl::route_delta!"]
    MMSB_src_02_semiring_DeltaRouter_jl__create_delta["MMSB/src/02_semiring/DeltaRouter.jl::create_delta"]
    MMSB_src_02_semiring_DeltaRouter_jl__length["MMSB/src/02_semiring/DeltaRouter.jl::length"]
    MMSB_src_02_semiring_DeltaRouter_jl__batch_route_deltas_["MMSB/src/02_semiring/DeltaRouter.jl::batch_route_deltas!"]
    MMSB_src_02_semiring_DeltaRouter_jl__propagate_change_["MMSB/src/02_semiring/DeltaRouter.jl::propagate_change!"]
    MMSB_src_02_semiring_DeltaRouter_jl__propagate_change_["MMSB/src/02_semiring/DeltaRouter.jl::propagate_change!"]
    MMSB_src_02_semiring_Semiring_jl__tropical_semiring["MMSB/src/02_semiring/Semiring.jl::tropical_semiring"]
    MMSB_src_02_semiring_Semiring_jl__boolean_semiring["MMSB/src/02_semiring/Semiring.jl::boolean_semiring"]
    MMSB_src_02_semiring_Semiring_jl___FLOAT_BUF["MMSB/src/02_semiring/Semiring.jl::_FLOAT_BUF"]
    MMSB_src_02_semiring_Semiring_jl___bool_buf["MMSB/src/02_semiring/Semiring.jl::_bool_buf"]
    MMSB_src_02_semiring_Semiring_jl__tropical_fold_add["MMSB/src/02_semiring/Semiring.jl::tropical_fold_add"]
    MMSB_src_02_semiring_Semiring_jl__tropical_fold_mul["MMSB/src/02_semiring/Semiring.jl::tropical_fold_mul"]
    MMSB_src_02_semiring_Semiring_jl__tropical_accumulate["MMSB/src/02_semiring/Semiring.jl::tropical_accumulate"]
    MMSB_src_02_semiring_Semiring_jl__boolean_fold_add["MMSB/src/02_semiring/Semiring.jl::boolean_fold_add"]
    MMSB_src_02_semiring_Semiring_jl__boolean_fold_mul["MMSB/src/02_semiring/Semiring.jl::boolean_fold_mul"]
    MMSB_src_02_semiring_Semiring_jl__boolean_accumulate["MMSB/src/02_semiring/Semiring.jl::boolean_accumulate"]
    MMSB_src_02_semiring_SemiringConfig_jl__build_semiring["MMSB/src/02_semiring/SemiringConfig.jl::build_semiring"]
    MMSB_src_03_dag_DependencyGraph_jl__add_edge_["MMSB/src/03_dag/DependencyGraph.jl::add_edge!"]
    MMSB_src_03_dag_DependencyGraph_jl__remove_edge_["MMSB/src/03_dag/DependencyGraph.jl::remove_edge!"]
    MMSB_src_03_dag_DependencyGraph_jl__has_edge["MMSB/src/03_dag/DependencyGraph.jl::has_edge"]
    MMSB_src_03_dag_DependencyGraph_jl__get_children["MMSB/src/03_dag/DependencyGraph.jl::get_children"]
    MMSB_src_03_dag_DependencyGraph_jl__get_parents["MMSB/src/03_dag/DependencyGraph.jl::get_parents"]
    MMSB_src_03_dag_DependencyGraph_jl__find_descendants["MMSB/src/03_dag/DependencyGraph.jl::find_descendants"]
    MMSB_src_03_dag_DependencyGraph_jl__find_ancestors["MMSB/src/03_dag/DependencyGraph.jl::find_ancestors"]
    MMSB_src_03_dag_DependencyGraph_jl__detect_cycles["MMSB/src/03_dag/DependencyGraph.jl::detect_cycles"]
    MMSB_src_03_dag_DependencyGraph_jl__dfs_cycle_detect["MMSB/src/03_dag/DependencyGraph.jl::dfs_cycle_detect"]
    MMSB_src_03_dag_DependencyGraph_jl__topological_order["MMSB/src/03_dag/DependencyGraph.jl::topological_order"]
    MMSB_src_03_dag_DependencyGraph_jl__reverse_postorder["MMSB/src/03_dag/DependencyGraph.jl::reverse_postorder"]
    MMSB_src_03_dag_DependencyGraph_jl__compute_closure["MMSB/src/03_dag/DependencyGraph.jl::compute_closure"]
    MMSB_src_03_dag_EventSystem_jl__EventSubscription["MMSB/src/03_dag/EventSystem.jl::EventSubscription"]
    MMSB_src_03_dag_EventSystem_jl__emit_event_["MMSB/src/03_dag/EventSystem.jl::emit_event!"]
    MMSB_src_03_dag_EventSystem_jl__subscribe_["MMSB/src/03_dag/EventSystem.jl::subscribe!"]
    MMSB_src_03_dag_EventSystem_jl__unsubscribe_["MMSB/src/03_dag/EventSystem.jl::unsubscribe!"]
    MMSB_src_03_dag_EventSystem_jl__log_event_["MMSB/src/03_dag/EventSystem.jl::log_event!"]
    MMSB_src_03_dag_EventSystem_jl__clear_subscriptions_["MMSB/src/03_dag/EventSystem.jl::clear_subscriptions!"]
    MMSB_src_03_dag_EventSystem_jl__get_subscription_count["MMSB/src/03_dag/EventSystem.jl::get_subscription_count"]
    MMSB_src_03_dag_EventSystem_jl__create_debug_subscriber["MMSB/src/03_dag/EventSystem.jl::create_debug_subscriber"]
    MMSB_src_03_dag_EventSystem_jl__create_logging_subscriber["MMSB/src/03_dag/EventSystem.jl::create_logging_subscriber"]
    MMSB_src_03_dag_EventSystem_jl___serialize_event["MMSB/src/03_dag/EventSystem.jl::_serialize_event"]
    MMSB_src_03_dag_EventSystem_jl__log_event_to_page_["MMSB/src/03_dag/EventSystem.jl::log_event_to_page!"]
    MMSB_src_03_dag_GraphDSL_jl__node["MMSB/src/03_dag/GraphDSL.jl::node"]
    MMSB_src_03_dag_ShadowPageGraph_jl__ShadowPageGraph["MMSB/src/03_dag/ShadowPageGraph.jl::ShadowPageGraph"]
    MMSB_src_03_dag_ShadowPageGraph_jl___ensure_vertex_["MMSB/src/03_dag/ShadowPageGraph.jl::_ensure_vertex!"]
    MMSB_src_03_dag_ShadowPageGraph_jl__add_dependency_["MMSB/src/03_dag/ShadowPageGraph.jl::add_dependency!"]
    MMSB_src_03_dag_ShadowPageGraph_jl__remove_dependency_["MMSB/src/03_dag/ShadowPageGraph.jl::remove_dependency!"]
    MMSB_src_03_dag_ShadowPageGraph_jl__get_children["MMSB/src/03_dag/ShadowPageGraph.jl::get_children"]
    MMSB_src_03_dag_ShadowPageGraph_jl__get_parents["MMSB/src/03_dag/ShadowPageGraph.jl::get_parents"]
    MMSB_src_03_dag_ShadowPageGraph_jl___dfs_has_cycle["MMSB/src/03_dag/ShadowPageGraph.jl::_dfs_has_cycle"]
    MMSB_src_03_dag_ShadowPageGraph_jl__has_cycle["MMSB/src/03_dag/ShadowPageGraph.jl::has_cycle"]
    MMSB_src_03_dag_ShadowPageGraph_jl___all_vertices["MMSB/src/03_dag/ShadowPageGraph.jl::_all_vertices"]
    MMSB_src_03_dag_ShadowPageGraph_jl__topological_sort["MMSB/src/03_dag/ShadowPageGraph.jl::topological_sort"]
    MMSB_src_04_propagation_PropagationEngine_jl__enable_graph_capture["MMSB/src/04_propagation/PropagationEngine.jl::enable_graph_capture"]
    MMSB_src_04_propagation_PropagationEngine_jl__disable_graph_capture["MMSB/src/04_propagation/PropagationEngine.jl::disable_graph_capture"]
    MMSB_src_04_propagation_PropagationEngine_jl__replay_cuda_graph["MMSB/src/04_propagation/PropagationEngine.jl::replay_cuda_graph"]
    MMSB_src_04_propagation_PropagationEngine_jl__batch_route_deltas_["MMSB/src/04_propagation/PropagationEngine.jl::batch_route_deltas!"]
    MMSB_src_04_propagation_PropagationEngine_jl___buffer["MMSB/src/04_propagation/PropagationEngine.jl::_buffer"]
    MMSB_src_04_propagation_PropagationEngine_jl__register_recompute_fn_["MMSB/src/04_propagation/PropagationEngine.jl::register_recompute_fn!"]
    MMSB_src_04_propagation_PropagationEngine_jl__register_passthrough_recompute_["MMSB/src/04_propagation/PropagationEngine.jl::register_passthrough_recompute!"]
    MMSB_src_04_propagation_PropagationEngine_jl__queue_recomputation_["MMSB/src/04_propagation/PropagationEngine.jl::queue_recomputation!"]
    MMSB_src_04_propagation_PropagationEngine_jl__propagate_change_["MMSB/src/04_propagation/PropagationEngine.jl::propagate_change!"]
    MMSB_src_04_propagation_PropagationEngine_jl__propagate_change_["MMSB/src/04_propagation/PropagationEngine.jl::propagate_change!"]
    MMSB_src_04_propagation_PropagationEngine_jl___aggregate_children["MMSB/src/04_propagation/PropagationEngine.jl::_aggregate_children"]
    MMSB_src_04_propagation_PropagationEngine_jl___execute_command_buffer_["MMSB/src/04_propagation/PropagationEngine.jl::_execute_command_buffer!"]
    MMSB_src_04_propagation_PropagationEngine_jl___apply_edges_["MMSB/src/04_propagation/PropagationEngine.jl::_apply_edges!"]
    MMSB_src_04_propagation_PropagationEngine_jl___handle_data_dependency_["MMSB/src/04_propagation/PropagationEngine.jl::_handle_data_dependency!"]
    MMSB_src_04_propagation_PropagationEngine_jl___collect_descendants["MMSB/src/04_propagation/PropagationEngine.jl::_collect_descendants"]
    MMSB_src_04_propagation_PropagationEngine_jl__schedule_propagation_["MMSB/src/04_propagation/PropagationEngine.jl::schedule_propagation!"]
    MMSB_src_04_propagation_PropagationEngine_jl__execute_propagation_["MMSB/src/04_propagation/PropagationEngine.jl::execute_propagation!"]
    MMSB_src_04_propagation_PropagationEngine_jl__recompute_page_["MMSB/src/04_propagation/PropagationEngine.jl::recompute_page!"]
    MMSB_src_04_propagation_PropagationEngine_jl__mark_page_stale_["MMSB/src/04_propagation/PropagationEngine.jl::mark_page_stale!"]
    MMSB_src_04_propagation_PropagationEngine_jl__schedule_gpu_sync_["MMSB/src/04_propagation/PropagationEngine.jl::schedule_gpu_sync!"]
    MMSB_src_04_propagation_PropagationEngine_jl__invalidate_compilation_["MMSB/src/04_propagation/PropagationEngine.jl::invalidate_compilation!"]
    MMSB_src_04_propagation_PropagationEngine_jl__topological_order_subset["MMSB/src/04_propagation/PropagationEngine.jl::topological_order_subset"]
    MMSB_src_04_propagation_PropagationScheduler_jl__schedule_["MMSB/src/04_propagation/PropagationScheduler.jl::schedule!"]
    MMSB_src_04_propagation_TransactionIsolation_jl__begin_transaction["MMSB/src/04_propagation/TransactionIsolation.jl::begin_transaction"]
    MMSB_src_04_propagation_TransactionIsolation_jl__commit_transaction["MMSB/src/04_propagation/TransactionIsolation.jl::commit_transaction"]
    MMSB_src_04_propagation_TransactionIsolation_jl__rollback_transaction["MMSB/src/04_propagation/TransactionIsolation.jl::rollback_transaction"]
    MMSB_src_04_propagation_TransactionIsolation_jl__with_transaction["MMSB/src/04_propagation/TransactionIsolation.jl::with_transaction"]
    MMSB_src_05_adaptive_AdaptiveLayout_jl__LayoutState["MMSB/src/05_adaptive/AdaptiveLayout.jl::LayoutState"]
    MMSB_src_05_adaptive_AdaptiveLayout_jl__optimize_layout_["MMSB/src/05_adaptive/AdaptiveLayout.jl::optimize_layout!"]
    MMSB_src_05_adaptive_AdaptiveLayout_jl__compute_locality_score["MMSB/src/05_adaptive/AdaptiveLayout.jl::compute_locality_score"]
    MMSB_src_05_adaptive_EntropyReduction_jl__compute_entropy["MMSB/src/05_adaptive/EntropyReduction.jl::compute_entropy"]
    MMSB_src_05_adaptive_EntropyReduction_jl__reduce_entropy_["MMSB/src/05_adaptive/EntropyReduction.jl::reduce_entropy!"]
    MMSB_src_05_adaptive_EntropyReduction_jl__entropy_gradient["MMSB/src/05_adaptive/EntropyReduction.jl::entropy_gradient"]
    MMSB_src_05_adaptive_GraphRewriting_jl__rewrite_dag_["MMSB/src/05_adaptive/GraphRewriting.jl::rewrite_dag!"]
    MMSB_src_05_adaptive_GraphRewriting_jl__can_reorder["MMSB/src/05_adaptive/GraphRewriting.jl::can_reorder"]
    MMSB_src_05_adaptive_GraphRewriting_jl__compute_edge_cost["MMSB/src/05_adaptive/GraphRewriting.jl::compute_edge_cost"]
    MMSB_src_05_adaptive_LocalityAnalysis_jl__analyze_locality["MMSB/src/05_adaptive/LocalityAnalysis.jl::analyze_locality"]
    MMSB_src_05_adaptive_LocalityAnalysis_jl__compute_reuse_distance["MMSB/src/05_adaptive/LocalityAnalysis.jl::compute_reuse_distance"]
    MMSB_src_06_utility_CostAggregation_jl__aggregate_costs["MMSB/src/06_utility/CostAggregation.jl::aggregate_costs"]
    MMSB_src_06_utility_CostAggregation_jl__normalize_costs["MMSB/src/06_utility/CostAggregation.jl::normalize_costs"]
    MMSB_src_06_utility_ErrorRecovery_jl__exponential_backoff["MMSB/src/06_utility/ErrorRecovery.jl::exponential_backoff"]
    MMSB_src_06_utility_ErrorRecovery_jl__is_retryable_error["MMSB/src/06_utility/ErrorRecovery.jl::is_retryable_error"]
    MMSB_src_06_utility_ErrorRecovery_jl__is_fatal_error["MMSB/src/06_utility/ErrorRecovery.jl::is_fatal_error"]
    MMSB_src_06_utility_ErrorRecovery_jl__retry_with_backoff["MMSB/src/06_utility/ErrorRecovery.jl::retry_with_backoff"]
    MMSB_src_06_utility_MemoryPressure_jl__LRUTracker["MMSB/src/06_utility/MemoryPressure.jl::LRUTracker"]
    MMSB_src_06_utility_MemoryPressure_jl__record_access["MMSB/src/06_utility/MemoryPressure.jl::record_access"]
    MMSB_src_06_utility_MemoryPressure_jl__evict_lru_pages["MMSB/src/06_utility/MemoryPressure.jl::evict_lru_pages"]
    MMSB_src_06_utility_Monitoring_jl__track_delta_latency_["MMSB/src/06_utility/Monitoring.jl::track_delta_latency!"]
    MMSB_src_06_utility_Monitoring_jl__track_propagation_latency_["MMSB/src/06_utility/Monitoring.jl::track_propagation_latency!"]
    MMSB_src_06_utility_Monitoring_jl__compute_graph_depth["MMSB/src/06_utility/Monitoring.jl::compute_graph_depth"]
    MMSB_src_06_utility_Monitoring_jl___dfs_depth["MMSB/src/06_utility/Monitoring.jl::_dfs_depth"]
    MMSB_src_06_utility_Monitoring_jl__get_stats["MMSB/src/06_utility/Monitoring.jl::get_stats"]
    MMSB_src_06_utility_Monitoring_jl__print_stats["MMSB/src/06_utility/Monitoring.jl::print_stats"]
    MMSB_src_06_utility_Monitoring_jl__reset_stats_["MMSB/src/06_utility/Monitoring.jl::reset_stats!"]
    MMSB_src_06_utility_cost_functions_jl__compute_cache_cost["MMSB/src/06_utility/cost_functions.jl::compute_cache_cost"]
    MMSB_src_06_utility_cost_functions_jl__compute_memory_cost["MMSB/src/06_utility/cost_functions.jl::compute_memory_cost"]
    MMSB_src_06_utility_cost_functions_jl__compute_latency_cost["MMSB/src/06_utility/cost_functions.jl::compute_latency_cost"]
    MMSB_src_06_utility_cost_functions_jl__from_telemetry["MMSB/src/06_utility/cost_functions.jl::from_telemetry"]
    MMSB_src_06_utility_entropy_measure_jl__PageDistribution["MMSB/src/06_utility/entropy_measure.jl::PageDistribution"]
    MMSB_src_06_utility_entropy_measure_jl__compute_entropy["MMSB/src/06_utility/entropy_measure.jl::compute_entropy"]
    MMSB_src_06_utility_entropy_measure_jl__state_entropy["MMSB/src/06_utility/entropy_measure.jl::state_entropy"]
    MMSB_src_06_utility_entropy_measure_jl__entropy_reduction["MMSB/src/06_utility/entropy_measure.jl::entropy_reduction"]
    MMSB_src_06_utility_utility_engine_jl__UtilityState["MMSB/src/06_utility/utility_engine.jl::UtilityState"]
    MMSB_src_06_utility_utility_engine_jl__compute_utility["MMSB/src/06_utility/utility_engine.jl::compute_utility"]
    MMSB_src_06_utility_utility_engine_jl__update_utility_["MMSB/src/06_utility/utility_engine.jl::update_utility!"]
    MMSB_src_06_utility_utility_engine_jl__utility_trend["MMSB/src/06_utility/utility_engine.jl::utility_trend"]
    MMSB_src_07_intention_IntentionTypes_jl__IntentionState["MMSB/src/07_intention/IntentionTypes.jl::IntentionState"]
    MMSB_src_07_intention_UpsertPlan_jl__validate_plan["MMSB/src/07_intention/UpsertPlan.jl::validate_plan"]
    MMSB_src_07_intention_attractor_states_jl__compute_gradient["MMSB/src/07_intention/attractor_states.jl::compute_gradient"]
    MMSB_src_07_intention_attractor_states_jl__evolve_state["MMSB/src/07_intention/attractor_states.jl::evolve_state"]
    MMSB_src_07_intention_attractor_states_jl__find_nearest_attractor["MMSB/src/07_intention/attractor_states.jl::find_nearest_attractor"]
    MMSB_src_07_intention_goal_emergence_jl__utility_gradient["MMSB/src/07_intention/goal_emergence.jl::utility_gradient"]
    MMSB_src_07_intention_goal_emergence_jl__detect_goals["MMSB/src/07_intention/goal_emergence.jl::detect_goals"]
    MMSB_src_07_intention_intent_lowering_jl__mask_to_bytes["MMSB/src/07_intention/intent_lowering.jl::mask_to_bytes"]
    MMSB_src_07_intention_intent_lowering_jl__lower_intent_to_deltaspec["MMSB/src/07_intention/intent_lowering.jl::lower_intent_to_deltaspec"]
    MMSB_src_07_intention_intent_lowering_jl__execute_upsert_plan_["MMSB/src/07_intention/intent_lowering.jl::execute_upsert_plan!"]
    MMSB_src_07_intention_intention_engine_jl__form_intention["MMSB/src/07_intention/intention_engine.jl::form_intention"]
    MMSB_src_07_intention_intention_engine_jl__evaluate_intention["MMSB/src/07_intention/intention_engine.jl::evaluate_intention"]
    MMSB_src_07_intention_intention_engine_jl__select_best_intention["MMSB/src/07_intention/intention_engine.jl::select_best_intention"]
    MMSB_src_07_intention_structural_preferences_jl__evaluate_preference["MMSB/src/07_intention/structural_preferences.jl::evaluate_preference"]
    MMSB_src_07_intention_structural_preferences_jl__apply_preferences["MMSB/src/07_intention/structural_preferences.jl::apply_preferences"]
    MMSB_src_08_reasoning_ReasoningTypes_jl__ReasoningState["MMSB/src/08_reasoning/ReasoningTypes.jl::ReasoningState"]
    MMSB_src_08_reasoning_constraint_propagation_jl__propagate_constraints["MMSB/src/08_reasoning/constraint_propagation.jl::propagate_constraints"]
    MMSB_src_08_reasoning_constraint_propagation_jl__forward_propagate["MMSB/src/08_reasoning/constraint_propagation.jl::forward_propagate"]
    MMSB_src_08_reasoning_constraint_propagation_jl__backward_propagate["MMSB/src/08_reasoning/constraint_propagation.jl::backward_propagate"]
    MMSB_src_08_reasoning_dependency_inference_jl__infer_dependencies["MMSB/src/08_reasoning/dependency_inference.jl::infer_dependencies"]
    MMSB_src_08_reasoning_dependency_inference_jl__analyze_edge_type["MMSB/src/08_reasoning/dependency_inference.jl::analyze_edge_type"]
    MMSB_src_08_reasoning_dependency_inference_jl__compute_dependency_strength["MMSB/src/08_reasoning/dependency_inference.jl::compute_dependency_strength"]
    MMSB_src_08_reasoning_dependency_inference_jl__count_paths["MMSB/src/08_reasoning/dependency_inference.jl::count_paths"]
    MMSB_src_08_reasoning_dependency_inference_jl__analyze_flow["MMSB/src/08_reasoning/dependency_inference.jl::analyze_flow"]
    MMSB_src_08_reasoning_dependency_inference_jl__compute_critical_path["MMSB/src/08_reasoning/dependency_inference.jl::compute_critical_path"]
    MMSB_src_08_reasoning_logic_engine_jl__deduce["MMSB/src/08_reasoning/logic_engine.jl::deduce"]
    MMSB_src_08_reasoning_logic_engine_jl__abduce["MMSB/src/08_reasoning/logic_engine.jl::abduce"]
    MMSB_src_08_reasoning_logic_engine_jl__induce["MMSB/src/08_reasoning/logic_engine.jl::induce"]
    MMSB_src_08_reasoning_logic_engine_jl__unify_constraints["MMSB/src/08_reasoning/logic_engine.jl::unify_constraints"]
    MMSB_src_08_reasoning_pattern_formation_jl__find_patterns["MMSB/src/08_reasoning/pattern_formation.jl::find_patterns"]
    MMSB_src_08_reasoning_pattern_formation_jl__extract_subgraphs["MMSB/src/08_reasoning/pattern_formation.jl::extract_subgraphs"]
    MMSB_src_08_reasoning_pattern_formation_jl__grow_subgraph["MMSB/src/08_reasoning/pattern_formation.jl::grow_subgraph"]
    MMSB_src_08_reasoning_pattern_formation_jl__extract_subgraph_signature["MMSB/src/08_reasoning/pattern_formation.jl::extract_subgraph_signature"]
    MMSB_src_08_reasoning_pattern_formation_jl__extract_edges["MMSB/src/08_reasoning/pattern_formation.jl::extract_edges"]
    MMSB_src_08_reasoning_pattern_formation_jl__match_pattern["MMSB/src/08_reasoning/pattern_formation.jl::match_pattern"]
    MMSB_src_08_reasoning_reasoning_engine_jl__initialize_reasoning["MMSB/src/08_reasoning/reasoning_engine.jl::initialize_reasoning"]
    MMSB_src_08_reasoning_reasoning_engine_jl__reason_over_dag["MMSB/src/08_reasoning/reasoning_engine.jl::reason_over_dag"]
    MMSB_src_08_reasoning_reasoning_engine_jl__perform_inference["MMSB/src/08_reasoning/reasoning_engine.jl::perform_inference"]
    MMSB_src_08_reasoning_rule_evaluation_jl__evaluate_rules["MMSB/src/08_reasoning/rule_evaluation.jl::evaluate_rules"]
    MMSB_src_08_reasoning_rule_evaluation_jl__apply_rule["MMSB/src/08_reasoning/rule_evaluation.jl::apply_rule"]
    MMSB_src_08_reasoning_rule_evaluation_jl__create_default_rules["MMSB/src/08_reasoning/rule_evaluation.jl::create_default_rules"]
    MMSB_src_08_reasoning_structural_inference_jl__infer_from_structure["MMSB/src/08_reasoning/structural_inference.jl::infer_from_structure"]
    MMSB_src_08_reasoning_structural_inference_jl__derive_constraints["MMSB/src/08_reasoning/structural_inference.jl::derive_constraints"]
    MMSB_src_08_reasoning_structural_inference_jl__check_consistency["MMSB/src/08_reasoning/structural_inference.jl::check_consistency"]
    MMSB_src_09_planning_PlanningTypes_jl__PlanningState["MMSB/src/09_planning/PlanningTypes.jl::PlanningState"]
    MMSB_src_09_planning_decision_graphs_jl__build_decision_graph["MMSB/src/09_planning/decision_graphs.jl::build_decision_graph"]
    MMSB_src_09_planning_decision_graphs_jl__expand_graph_["MMSB/src/09_planning/decision_graphs.jl::expand_graph!"]
    MMSB_src_09_planning_decision_graphs_jl__apply_action_simple["MMSB/src/09_planning/decision_graphs.jl::apply_action_simple"]
    MMSB_src_09_planning_decision_graphs_jl__find_optimal_path["MMSB/src/09_planning/decision_graphs.jl::find_optimal_path"]
    MMSB_src_09_planning_decision_graphs_jl__prune_graph["MMSB/src/09_planning/decision_graphs.jl::prune_graph"]
    MMSB_src_09_planning_goal_decomposition_jl__decompose_goal["MMSB/src/09_planning/goal_decomposition.jl::decompose_goal"]
    MMSB_src_09_planning_goal_decomposition_jl__create_subgoal_hierarchy["MMSB/src/09_planning/goal_decomposition.jl::create_subgoal_hierarchy"]
    MMSB_src_09_planning_goal_decomposition_jl__order_subgoals["MMSB/src/09_planning/goal_decomposition.jl::order_subgoals"]
    MMSB_src_09_planning_goal_decomposition_jl__score_subgoal["MMSB/src/09_planning/goal_decomposition.jl::score_subgoal"]
    MMSB_src_09_planning_goal_decomposition_jl__estimate_achievability["MMSB/src/09_planning/goal_decomposition.jl::estimate_achievability"]
    MMSB_src_09_planning_optimization_planning_jl__optimize_plan["MMSB/src/09_planning/optimization_planning.jl::optimize_plan"]
    MMSB_src_09_planning_optimization_planning_jl__extract_parameters["MMSB/src/09_planning/optimization_planning.jl::extract_parameters"]
    MMSB_src_09_planning_optimization_planning_jl__compute_gradient["MMSB/src/09_planning/optimization_planning.jl::compute_gradient"]
    MMSB_src_09_planning_optimization_planning_jl__norm["MMSB/src/09_planning/optimization_planning.jl::norm"]
    MMSB_src_09_planning_optimization_planning_jl__reconstruct_plan["MMSB/src/09_planning/optimization_planning.jl::reconstruct_plan"]
    MMSB_src_09_planning_optimization_planning_jl__gradient_descent_planning["MMSB/src/09_planning/optimization_planning.jl::gradient_descent_planning"]
    MMSB_src_09_planning_optimization_planning_jl__evaluate_action_sequence["MMSB/src/09_planning/optimization_planning.jl::evaluate_action_sequence"]
    MMSB_src_09_planning_optimization_planning_jl__compute_sequence_gradient["MMSB/src/09_planning/optimization_planning.jl::compute_sequence_gradient"]
    MMSB_src_09_planning_optimization_planning_jl__actions_from_params["MMSB/src/09_planning/optimization_planning.jl::actions_from_params"]
    MMSB_src_09_planning_optimization_planning_jl__prepare_for_enzyme["MMSB/src/09_planning/optimization_planning.jl::prepare_for_enzyme"]
    MMSB_src_09_planning_planning_engine_jl__create_plan["MMSB/src/09_planning/planning_engine.jl::create_plan"]
    MMSB_src_09_planning_planning_engine_jl__execute_planning["MMSB/src/09_planning/planning_engine.jl::execute_planning"]
    MMSB_src_09_planning_planning_engine_jl__replan["MMSB/src/09_planning/planning_engine.jl::replan"]
    MMSB_src_09_planning_rl_planning_jl__value_iteration["MMSB/src/09_planning/rl_planning.jl::value_iteration"]
    MMSB_src_09_planning_rl_planning_jl__immediate_reward["MMSB/src/09_planning/rl_planning.jl::immediate_reward"]
    MMSB_src_09_planning_rl_planning_jl__expected_next_value["MMSB/src/09_planning/rl_planning.jl::expected_next_value"]
    MMSB_src_09_planning_rl_planning_jl__policy_iteration["MMSB/src/09_planning/rl_planning.jl::policy_iteration"]
    MMSB_src_09_planning_rl_planning_jl__evaluate_policy["MMSB/src/09_planning/rl_planning.jl::evaluate_policy"]
    MMSB_src_09_planning_rl_planning_jl__q_learning["MMSB/src/09_planning/rl_planning.jl::q_learning"]
    MMSB_src_09_planning_rl_planning_jl__temporal_difference["MMSB/src/09_planning/rl_planning.jl::temporal_difference"]
    MMSB_src_09_planning_rollout_simulation_jl__simulate_plan["MMSB/src/09_planning/rollout_simulation.jl::simulate_plan"]
    MMSB_src_09_planning_rollout_simulation_jl__parallel_rollout["MMSB/src/09_planning/rollout_simulation.jl::parallel_rollout"]
    MMSB_src_09_planning_rollout_simulation_jl__evaluate_outcome["MMSB/src/09_planning/rollout_simulation.jl::evaluate_outcome"]
    MMSB_src_09_planning_search_algorithms_jl__astar_search["MMSB/src/09_planning/search_algorithms.jl::astar_search"]
    MMSB_src_09_planning_search_algorithms_jl__mcts_search["MMSB/src/09_planning/search_algorithms.jl::mcts_search"]
    MMSB_src_09_planning_search_algorithms_jl__MCTSNode["MMSB/src/09_planning/search_algorithms.jl::MCTSNode"]
    MMSB_src_09_planning_search_algorithms_jl__select_node["MMSB/src/09_planning/search_algorithms.jl::select_node"]
    MMSB_src_09_planning_search_algorithms_jl__best_uct_child["MMSB/src/09_planning/search_algorithms.jl::best_uct_child"]
    MMSB_src_09_planning_search_algorithms_jl__expand_node["MMSB/src/09_planning/search_algorithms.jl::expand_node"]
    MMSB_src_09_planning_search_algorithms_jl__simulate["MMSB/src/09_planning/search_algorithms.jl::simulate"]
    MMSB_src_09_planning_search_algorithms_jl__backpropagate["MMSB/src/09_planning/search_algorithms.jl::backpropagate"]
    MMSB_src_09_planning_search_algorithms_jl__compute_heuristic["MMSB/src/09_planning/search_algorithms.jl::compute_heuristic"]
    MMSB_src_09_planning_search_algorithms_jl__can_apply["MMSB/src/09_planning/search_algorithms.jl::can_apply"]
    MMSB_src_09_planning_search_algorithms_jl__apply_action["MMSB/src/09_planning/search_algorithms.jl::apply_action"]
    MMSB_src_09_planning_search_algorithms_jl__is_terminal["MMSB/src/09_planning/search_algorithms.jl::is_terminal"]
    MMSB_src_09_planning_search_algorithms_jl__reconstruct_plan["MMSB/src/09_planning/search_algorithms.jl::reconstruct_plan"]
    MMSB_src_09_planning_search_algorithms_jl__extract_plan_from_mcts["MMSB/src/09_planning/search_algorithms.jl::extract_plan_from_mcts"]
    MMSB_src_09_planning_strategy_generation_jl__generate_strategies["MMSB/src/09_planning/strategy_generation.jl::generate_strategies"]
    MMSB_src_09_planning_strategy_generation_jl__hierarchical_planning["MMSB/src/09_planning/strategy_generation.jl::hierarchical_planning"]
    MMSB_src_09_planning_strategy_generation_jl__select_strategy["MMSB/src/09_planning/strategy_generation.jl::select_strategy"]
    MMSB_src_09_planning_strategy_generation_jl__adapt_strategy["MMSB/src/09_planning/strategy_generation.jl::adapt_strategy"]
    MMSB_src_10_agent_interface_AgentProtocol_jl__observe["MMSB/src/10_agent_interface/AgentProtocol.jl::observe"]
    MMSB_src_10_agent_interface_BaseHook_jl__enable_base_hooks_["MMSB/src/10_agent_interface/BaseHook.jl::enable_base_hooks!"]
    MMSB_src_10_agent_interface_BaseHook_jl__disable_base_hooks_["MMSB/src/10_agent_interface/BaseHook.jl::disable_base_hooks!"]
    MMSB_src_10_agent_interface_BaseHook_jl__hook_invoke["MMSB/src/10_agent_interface/BaseHook.jl::hook_invoke"]
    MMSB_src_10_agent_interface_BaseHook_jl__hook_setfield_["MMSB/src/10_agent_interface/BaseHook.jl::hook_setfield!"]
    MMSB_src_10_agent_interface_BaseHook_jl__hook_getfield["MMSB/src/10_agent_interface/BaseHook.jl::hook_getfield"]
    MMSB_src_10_agent_interface_CompilerHooks_jl__MMSBInterpreter["MMSB/src/10_agent_interface/CompilerHooks.jl::MMSBInterpreter"]
    MMSB_src_10_agent_interface_CompilerHooks_jl__Core_Compiler_InferenceParams["MMSB/src/10_agent_interface/CompilerHooks.jl::Core.Compiler.InferenceParams"]
    MMSB_src_10_agent_interface_CompilerHooks_jl__Core_Compiler_OptimizationParams["MMSB/src/10_agent_interface/CompilerHooks.jl::Core.Compiler.OptimizationParams"]
    MMSB_src_10_agent_interface_CompilerHooks_jl__Core_Compiler_get_world_counter["MMSB/src/10_agent_interface/CompilerHooks.jl::Core.Compiler.get_world_counter"]
    MMSB_src_10_agent_interface_CompilerHooks_jl__Core_Compiler_get_inference_cache["MMSB/src/10_agent_interface/CompilerHooks.jl::Core.Compiler.get_inference_cache"]
    MMSB_src_10_agent_interface_CompilerHooks_jl__Core_Compiler_code_cache["MMSB/src/10_agent_interface/CompilerHooks.jl::Core.Compiler.code_cache"]
    MMSB_src_10_agent_interface_CompilerHooks_jl__Core_Compiler_typeinf["MMSB/src/10_agent_interface/CompilerHooks.jl::Core.Compiler.typeinf"]
    MMSB_src_10_agent_interface_CompilerHooks_jl__Core_Compiler_abstract_call_method["MMSB/src/10_agent_interface/CompilerHooks.jl::Core.Compiler.abstract_call_method"]
    MMSB_src_10_agent_interface_CompilerHooks_jl__Core_Compiler_optimize["MMSB/src/10_agent_interface/CompilerHooks.jl::Core.Compiler.optimize"]
    MMSB_src_10_agent_interface_CompilerHooks_jl__log_inference_start_["MMSB/src/10_agent_interface/CompilerHooks.jl::log_inference_start!"]
    MMSB_src_10_agent_interface_CompilerHooks_jl__log_inference_result_["MMSB/src/10_agent_interface/CompilerHooks.jl::log_inference_result!"]
    MMSB_src_10_agent_interface_CompilerHooks_jl__create_inference_pages_["MMSB/src/10_agent_interface/CompilerHooks.jl::create_inference_pages!"]
    MMSB_src_10_agent_interface_CompilerHooks_jl__enable_compiler_hooks_["MMSB/src/10_agent_interface/CompilerHooks.jl::enable_compiler_hooks!"]
    MMSB_src_10_agent_interface_CompilerHooks_jl__disable_compiler_hooks_["MMSB/src/10_agent_interface/CompilerHooks.jl::disable_compiler_hooks!"]
    MMSB_src_10_agent_interface_CoreHooks_jl__enable_core_hooks_["MMSB/src/10_agent_interface/CoreHooks.jl::enable_core_hooks!"]
    MMSB_src_10_agent_interface_CoreHooks_jl__disable_core_hooks_["MMSB/src/10_agent_interface/CoreHooks.jl::disable_core_hooks!"]
    MMSB_src_10_agent_interface_CoreHooks_jl__hook_codeinfo_creation["MMSB/src/10_agent_interface/CoreHooks.jl::hook_codeinfo_creation"]
    MMSB_src_10_agent_interface_CoreHooks_jl__hook_methodinstance["MMSB/src/10_agent_interface/CoreHooks.jl::hook_methodinstance"]
    MMSB_src_10_agent_interface_InstrumentationManager_jl__InstrumentationConfig["MMSB/src/10_agent_interface/InstrumentationManager.jl::InstrumentationConfig"]
    MMSB_src_10_agent_interface_InstrumentationManager_jl__enable_instrumentation_["MMSB/src/10_agent_interface/InstrumentationManager.jl::enable_instrumentation!"]
    MMSB_src_10_agent_interface_InstrumentationManager_jl__disable_instrumentation_["MMSB/src/10_agent_interface/InstrumentationManager.jl::disable_instrumentation!"]
    MMSB_src_10_agent_interface_InstrumentationManager_jl__configure_instrumentation_["MMSB/src/10_agent_interface/InstrumentationManager.jl::configure_instrumentation!"]
    MMSB_src_10_agent_interface_checkpoint_api_jl__create_checkpoint["MMSB/src/10_agent_interface/checkpoint_api.jl::create_checkpoint"]
    MMSB_src_10_agent_interface_checkpoint_api_jl__restore_checkpoint["MMSB/src/10_agent_interface/checkpoint_api.jl::restore_checkpoint"]
    MMSB_src_10_agent_interface_checkpoint_api_jl__list_checkpoints["MMSB/src/10_agent_interface/checkpoint_api.jl::list_checkpoints"]
    MMSB_src_10_agent_interface_event_subscription_jl__subscribe_to_events["MMSB/src/10_agent_interface/event_subscription.jl::subscribe_to_events"]
    MMSB_src_10_agent_interface_event_subscription_jl__unsubscribe["MMSB/src/10_agent_interface/event_subscription.jl::unsubscribe"]
    MMSB_src_10_agent_interface_event_subscription_jl__emit_event["MMSB/src/10_agent_interface/event_subscription.jl::emit_event"]
    MMSB_src_11_agents_AgentTypes_jl__push_memory_["MMSB/src/11_agents/AgentTypes.jl::push_memory!"]
    MMSB_src_11_agents_enzyme_integration_jl__gradient_descent_step_["MMSB/src/11_agents/enzyme_integration.jl::gradient_descent_step!"]
    MMSB_src_11_agents_enzyme_integration_jl__autodiff_loss["MMSB/src/11_agents/enzyme_integration.jl::autodiff_loss"]
    MMSB_src_11_agents_hybrid_agent_jl__observe["MMSB/src/11_agents/hybrid_agent.jl::observe"]
    MMSB_src_11_agents_hybrid_agent_jl__symbolic_step_["MMSB/src/11_agents/hybrid_agent.jl::symbolic_step!"]
    MMSB_src_11_agents_hybrid_agent_jl__neural_step_["MMSB/src/11_agents/hybrid_agent.jl::neural_step!"]
    MMSB_src_11_agents_lux_models_jl__create_value_network["MMSB/src/11_agents/lux_models.jl::create_value_network"]
    MMSB_src_11_agents_lux_models_jl__create_policy_network["MMSB/src/11_agents/lux_models.jl::create_policy_network"]
    MMSB_src_11_agents_planning_agent_jl__observe["MMSB/src/11_agents/planning_agent.jl::observe"]
    MMSB_src_11_agents_planning_agent_jl__generate_plan["MMSB/src/11_agents/planning_agent.jl::generate_plan"]
    MMSB_src_11_agents_planning_agent_jl__execute_plan_step["MMSB/src/11_agents/planning_agent.jl::execute_plan_step"]
    MMSB_src_11_agents_rl_agent_jl__RLAgent["MMSB/src/11_agents/rl_agent.jl::RLAgent"]
    MMSB_src_11_agents_rl_agent_jl__observe["MMSB/src/11_agents/rl_agent.jl::observe"]
    MMSB_src_11_agents_rl_agent_jl__compute_reward["MMSB/src/11_agents/rl_agent.jl::compute_reward"]
    MMSB_src_11_agents_rl_agent_jl__train_step_["MMSB/src/11_agents/rl_agent.jl::train_step!"]
    MMSB_src_11_agents_symbolic_agent_jl__SymbolicAgent["MMSB/src/11_agents/symbolic_agent.jl::SymbolicAgent"]
    MMSB_src_11_agents_symbolic_agent_jl__observe["MMSB/src/11_agents/symbolic_agent.jl::observe"]
    MMSB_src_11_agents_symbolic_agent_jl__infer_rules_["MMSB/src/11_agents/symbolic_agent.jl::infer_rules!"]
    MMSB_src_11_agents_symbolic_agent_jl__apply_rule["MMSB/src/11_agents/symbolic_agent.jl::apply_rule"]
    MMSB_src_12_applications_financial_modeling_jl__compute_value["MMSB/src/12_applications/financial_modeling.jl::compute_value"]
    MMSB_src_12_applications_financial_modeling_jl__rebalance_["MMSB/src/12_applications/financial_modeling.jl::rebalance!"]
    MMSB_src_12_applications_llm_tools_jl__query_llm["MMSB/src/12_applications/llm_tools.jl::query_llm"]
    MMSB_src_12_applications_llm_tools_jl__store_llm_response["MMSB/src/12_applications/llm_tools.jl::store_llm_response"]
    MMSB_src_12_applications_memory_driven_reasoning_jl__reason_over_memory["MMSB/src/12_applications/memory_driven_reasoning.jl::reason_over_memory"]
    MMSB_src_12_applications_memory_driven_reasoning_jl__temporal_reasoning["MMSB/src/12_applications/memory_driven_reasoning.jl::temporal_reasoning"]
    MMSB_src_12_applications_multi_agent_system_jl__register_agent_["MMSB/src/12_applications/multi_agent_system.jl::register_agent!"]
    MMSB_src_12_applications_multi_agent_system_jl__coordinate_step_["MMSB/src/12_applications/multi_agent_system.jl::coordinate_step!"]
    MMSB_src_12_applications_world_simulation_jl__add_entity_["MMSB/src/12_applications/world_simulation.jl::add_entity!"]
    MMSB_src_12_applications_world_simulation_jl__simulate_step_["MMSB/src/12_applications/world_simulation.jl::simulate_step!"]
    MMSB_src_ffi_rs__mmsb_page_read --> MMSB_src_ffi_rs__set_last_error
    MMSB_src_ffi_rs__mmsb_page_read --> MMSB_src_ffi_rs__set_last_error
    MMSB_src_ffi_rs__mmsb_page_epoch --> MMSB_src_ffi_rs__set_last_error
    MMSB_src_ffi_rs__mmsb_page_write_masked --> MMSB_src_ffi_rs__set_last_error
    MMSB_src_ffi_rs__mmsb_page_write_masked --> MMSB_src_ffi_rs__mask_from_bytes
    MMSB_src_ffi_rs__mmsb_page_write_masked --> MMSB_src_ffi_rs__vec_from_ptr
    MMSB_src_ffi_rs__mmsb_page_write_masked --> MMSB_src_ffi_rs__set_last_error
    MMSB_src_ffi_rs__mmsb_page_metadata_size --> MMSB_src_ffi_rs__set_last_error
    MMSB_src_ffi_rs__mmsb_page_metadata_export --> MMSB_src_ffi_rs__set_last_error
    MMSB_src_ffi_rs__mmsb_page_metadata_import --> MMSB_src_ffi_rs__set_last_error
    MMSB_src_ffi_rs__mmsb_page_metadata_import --> MMSB_src_ffi_rs__set_last_error
    MMSB_src_ffi_rs__mmsb_delta_new --> MMSB_src_ffi_rs__mask_from_bytes
    MMSB_src_ffi_rs__mmsb_delta_new --> MMSB_src_ffi_rs__vec_from_ptr
    MMSB_src_ffi_rs__mmsb_delta_apply --> MMSB_src_ffi_rs__set_last_error
    MMSB_src_ffi_rs__mmsb_delta_apply --> MMSB_src_ffi_rs__set_last_error
    MMSB_src_ffi_rs__mmsb_delta_id --> MMSB_src_ffi_rs__set_last_error
    MMSB_src_ffi_rs__mmsb_delta_page_id --> MMSB_src_ffi_rs__set_last_error
    MMSB_src_ffi_rs__mmsb_delta_epoch --> MMSB_src_ffi_rs__set_last_error
    MMSB_src_ffi_rs__mmsb_delta_is_sparse --> MMSB_src_ffi_rs__set_last_error
    MMSB_src_ffi_rs__mmsb_delta_timestamp --> MMSB_src_ffi_rs__set_last_error
    MMSB_src_ffi_rs__mmsb_delta_source_len --> MMSB_src_ffi_rs__set_last_error
    MMSB_src_ffi_rs__mmsb_delta_copy_source --> MMSB_src_ffi_rs__set_last_error
    MMSB_src_ffi_rs__mmsb_delta_mask_len --> MMSB_src_ffi_rs__set_last_error
    MMSB_src_ffi_rs__mmsb_delta_copy_mask --> MMSB_src_ffi_rs__set_last_error
    MMSB_src_ffi_rs__mmsb_delta_payload_len --> MMSB_src_ffi_rs__set_last_error
    MMSB_src_ffi_rs__mmsb_delta_copy_payload --> MMSB_src_ffi_rs__set_last_error
    MMSB_src_ffi_rs__mmsb_delta_set_intent_metadata --> MMSB_src_ffi_rs__set_last_error
    MMSB_src_ffi_rs__mmsb_delta_set_intent_metadata --> MMSB_src_ffi_rs__set_last_error
    MMSB_src_ffi_rs__mmsb_delta_intent_metadata_len --> MMSB_src_ffi_rs__set_last_error
    MMSB_src_ffi_rs__mmsb_delta_copy_intent_metadata --> MMSB_src_ffi_rs__set_last_error
    MMSB_src_ffi_rs__mmsb_tlog_new --> MMSB_src_ffi_rs__set_last_error
    MMSB_src_ffi_rs__mmsb_tlog_new --> MMSB_src_ffi_rs__set_last_error
    MMSB_src_ffi_rs__mmsb_tlog_new --> MMSB_src_ffi_rs__set_last_error
    MMSB_src_ffi_rs__mmsb_tlog_new --> MMSB_src_ffi_rs__log_error_code
    MMSB_src_ffi_rs__mmsb_tlog_append --> MMSB_src_ffi_rs__set_last_error
    MMSB_src_ffi_rs__mmsb_tlog_append --> MMSB_src_ffi_rs__set_last_error
    MMSB_src_ffi_rs__mmsb_tlog_append --> MMSB_src_ffi_rs__log_error_code
    MMSB_src_ffi_rs__mmsb_checkpoint_write --> MMSB_src_ffi_rs__set_last_error
    MMSB_src_ffi_rs__mmsb_checkpoint_write --> MMSB_src_ffi_rs__set_last_error
    MMSB_src_ffi_rs__mmsb_checkpoint_write --> MMSB_src_01_page_checkpoint_rs__write_checkpoint
    MMSB_src_ffi_rs__mmsb_checkpoint_write --> MMSB_benchmark_benchmarks_jl___checkpoint
    MMSB_src_ffi_rs__mmsb_checkpoint_write --> MMSB_src_ffi_rs__set_last_error
    MMSB_src_ffi_rs__mmsb_checkpoint_load --> MMSB_src_ffi_rs__set_last_error
    MMSB_src_ffi_rs__mmsb_checkpoint_load --> MMSB_src_ffi_rs__set_last_error
    MMSB_src_ffi_rs__mmsb_checkpoint_load --> MMSB_src_01_page_checkpoint_rs__load_checkpoint
    MMSB_src_ffi_rs__mmsb_checkpoint_load --> MMSB_benchmark_benchmarks_jl___checkpoint
    MMSB_src_ffi_rs__mmsb_checkpoint_load --> MMSB_src_ffi_rs__set_last_error
    MMSB_src_ffi_rs__mmsb_tlog_reader_new --> MMSB_src_ffi_rs__set_last_error
    MMSB_src_ffi_rs__mmsb_tlog_reader_new --> MMSB_src_ffi_rs__set_last_error
    MMSB_src_ffi_rs__mmsb_tlog_reader_new --> MMSB_src_ffi_rs__set_last_error
    MMSB_src_ffi_rs__mmsb_tlog_reader_new --> MMSB_src_ffi_rs__log_error_code
    MMSB_src_ffi_rs__mmsb_tlog_reader_next --> MMSB_src_ffi_rs__set_last_error
    MMSB_src_ffi_rs__mmsb_tlog_reader_next --> MMSB_src_ffi_rs__set_last_error
    MMSB_src_ffi_rs__mmsb_tlog_reader_next --> MMSB_src_ffi_rs__log_error_code
    MMSB_src_ffi_rs__mmsb_tlog_summary --> MMSB_src_ffi_rs__set_last_error
    MMSB_src_ffi_rs__mmsb_tlog_summary --> MMSB_src_ffi_rs__set_last_error
    MMSB_src_ffi_rs__mmsb_tlog_summary --> MMSB_src_01_page_tlog_rs__summary
    MMSB_src_ffi_rs__mmsb_tlog_summary --> MMSB_src_ffi_rs__set_last_error
    MMSB_src_ffi_rs__mmsb_tlog_summary --> MMSB_src_ffi_rs__log_error_code
    MMSB_src_ffi_rs__mmsb_allocator_allocate --> MMSB_src_ffi_rs__set_last_error
    MMSB_src_ffi_rs__mmsb_allocator_allocate --> MMSB_src_ffi_rs__set_last_error
    MMSB_src_ffi_rs__mmsb_allocator_allocate --> MMSB_src_ffi_rs__set_last_error
    MMSB_src_ffi_rs__mmsb_allocator_release --> MMSB_src_ffi_rs__set_last_error
    MMSB_src_ffi_rs__mmsb_allocator_get_page --> MMSB_src_ffi_rs__set_last_error
    MMSB_src_ffi_rs__mmsb_allocator_get_page --> MMSB_benchmark_benchmarks_jl___page
    MMSB_src_ffi_rs__mmsb_allocator_get_page --> MMSB_src_ffi_rs__set_last_error
    MMSB_src_ffi_rs__mmsb_allocator_page_count --> MMSB_src_ffi_rs__set_last_error
    MMSB_src_ffi_rs__mmsb_allocator_list_pages --> MMSB_src_ffi_rs__set_last_error
    MMSB_src_ffi_rs__mmsb_semiring_tropical_fold_add --> MMSB_src_ffi_rs__slice_from_ptr
    MMSB_src_ffi_rs__mmsb_semiring_tropical_fold_add --> MMSB_src_02_semiring_semiring_ops_rs__fold_add
    MMSB_src_ffi_rs__mmsb_semiring_tropical_fold_mul --> MMSB_src_ffi_rs__slice_from_ptr
    MMSB_src_ffi_rs__mmsb_semiring_tropical_fold_mul --> MMSB_src_02_semiring_semiring_ops_rs__fold_mul
    MMSB_src_ffi_rs__mmsb_semiring_tropical_accumulate --> MMSB_src_02_semiring_semiring_ops_rs__accumulate
    MMSB_src_ffi_rs__mmsb_semiring_boolean_fold_add --> MMSB_src_ffi_rs__slice_from_ptr
    MMSB_src_ffi_rs__mmsb_semiring_boolean_fold_add --> MMSB_src_02_semiring_semiring_ops_rs__fold_add
    MMSB_src_ffi_rs__mmsb_semiring_boolean_fold_mul --> MMSB_src_ffi_rs__slice_from_ptr
    MMSB_src_ffi_rs__mmsb_semiring_boolean_fold_mul --> MMSB_src_02_semiring_semiring_ops_rs__fold_mul
    MMSB_src_ffi_rs__mmsb_semiring_boolean_accumulate --> MMSB_src_02_semiring_semiring_ops_rs__accumulate
    MMSB_tests_delta_validation_rs__validates_dense_lengths --> MMSB_tests_delta_validation_rs__dense_delta
    MMSB_tests_delta_validation_rs__rejects_mismatched_dense_lengths --> MMSB_tests_delta_validation_rs__dense_delta
    MMSB_tests_mmsb_tests_rs__test_page_snapshot_and_restore --> MMSB_benchmark_benchmarks_jl___page
    MMSB_tests_mmsb_tests_rs__test_checkpoint_log_and_restore --> MMSB_src_01_page_checkpoint_rs__write_checkpoint
    MMSB_tests_mmsb_tests_rs__test_checkpoint_log_and_restore --> MMSB_benchmark_benchmarks_jl___checkpoint
    MMSB_tests_mmsb_tests_rs__test_checkpoint_log_and_restore --> MMSB_src_01_page_checkpoint_rs__load_checkpoint
    MMSB_tests_mmsb_tests_rs__test_checkpoint_log_and_restore --> MMSB_benchmark_benchmarks_jl___checkpoint
    MMSB_tests_mmsb_tests_rs__test_checkpoint_log_and_restore --> MMSB_benchmark_benchmarks_jl___page
    MMSB_tests_week27_31_integration_rs__test_delta_merge_simd --> MMSB_src_01_page_delta_merge_rs__merge_deltas
    MMSB_tests_week27_31_integration_rs__test_lockfree_allocator --> MMSB_src_06_utility_Monitoring_jl__get_stats
    MMSB_src_00_physical_allocator_rs__test_checkpoint_roundtrip_in_memory --> MMSB_benchmark_benchmarks_jl___page
    MMSB_src_01_page_delta_merge_rs__merge_dense_simd --> MMSB_src_01_page_delta_merge_rs__merge_dense_avx512
    MMSB_src_01_page_delta_merge_rs__merge_dense_simd --> MMSB_src_01_page_delta_merge_rs__merge_dense_avx2
    MMSB_src_01_page_tlog_rs__summary --> MMSB_src_01_page_tlog_rs__validate_header
    MMSB_src_01_page_tlog_rs__summary --> MMSB_src_01_page_tlog_rs__read_frame
    MMSB_src_01_page_tlog_compression_rs__compress_delta_mask --> MMSB_src_01_page_tlog_compression_rs__encode_rle
    MMSB_src_01_page_tlog_compression_rs__compress_delta_mask --> MMSB_src_01_page_tlog_compression_rs__bitpack_mask
    MMSB_src_03_dag_cycle_detection_rs__has_cycle --> MMSB_src_03_dag_cycle_detection_rs__dfs
    MMSB_src_03_dag_cycle_detection_rs__has_cycle --> MMSB_src_03_dag_cycle_detection_rs__dfs
    MMSB_src_03_dag_cycle_detection_rs__dfs --> MMSB_src_03_dag_cycle_detection_rs__dfs
    MMSB_src_API_jl__query_page --> MMSB_benchmark_benchmarks_jl___page
    MMSB_src_API_jl__query_page --> MMSB_src_01_types_MMSBState_jl__get_page
    MMSB_src_API_jl__query_page --> MMSB_tests_mmsb_tests_rs__read_page
    MMSB_src_API_jl__query_page --> MMSB_benchmark_benchmarks_jl___page
    MMSB_src_API_jl__query_page --> MMSB_src_01_page_Page_jl__read_page
    MMSB_benchmark_benchmarks_jl___stop_state_ --> MMSB_src_API_jl__mmsb_stop
    MMSB_benchmark_benchmarks_jl___populate_pages_ --> MMSB_src_API_jl__create_page
    MMSB_benchmark_benchmarks_jl___populate_pages_ --> MMSB_benchmark_benchmarks_jl___page
    MMSB_benchmark_benchmarks_jl___seed_pages_ --> MMSB_src_API_jl__update_page
    MMSB_benchmark_benchmarks_jl___seed_pages_ --> MMSB_benchmark_benchmarks_jl___page
    MMSB_benchmark_benchmarks_jl___seed_pages_ --> MMSB_benchmark_benchmarks_jl___page
    MMSB_benchmark_benchmarks_jl___replay_sequence_ --> MMSB_src_API_jl__update_page
    MMSB_benchmark_benchmarks_jl___replay_sequence_ --> MMSB_benchmark_benchmarks_jl___page
    MMSB_benchmark_benchmarks_jl___stress_updates_ --> MMSB_src_API_jl__update_page
    MMSB_benchmark_benchmarks_jl___stress_updates_ --> MMSB_benchmark_benchmarks_jl___page
    MMSB_benchmark_benchmarks_jl___stress_updates_ --> MMSB_src_API_jl__length
    MMSB_benchmark_benchmarks_jl___stress_updates_ --> MMSB_src_02_semiring_DeltaRouter_jl__length
    MMSB_benchmark_benchmarks_jl___link_chain_ --> MMSB_src_03_dag_ShadowPageGraph_jl__add_dependency_
    MMSB_benchmark_benchmarks_jl___link_chain_ --> MMSB_src_04_propagation_PropagationEngine_jl__register_passthrough_recompute_
    MMSB_benchmark_benchmarks_jl___link_chain_ --> MMSB_src_API_jl__length
    MMSB_benchmark_benchmarks_jl___link_chain_ --> MMSB_src_02_semiring_DeltaRouter_jl__length
    MMSB_benchmark_benchmarks_jl___checkpoint --> MMSB_src_01_page_TLog_jl__checkpoint_log_
    MMSB_benchmark_benchmarks_jl___graph_fixture --> MMSB_src_03_dag_ShadowPageGraph_jl__ShadowPageGraph
    MMSB_benchmark_benchmarks_jl___graph_fixture --> MMSB_src_03_dag_ShadowPageGraph_jl__add_dependency_
    MMSB_benchmark_benchmarks_jl___graph_bfs --> MMSB_src_03_dag_DependencyGraph_jl__get_children
    MMSB_benchmark_benchmarks_jl___graph_bfs --> MMSB_src_03_dag_ShadowPageGraph_jl__get_children
    MMSB_benchmark_benchmarks_jl___graph_bfs --> MMSB_src_API_jl__length
    MMSB_benchmark_benchmarks_jl___graph_bfs --> MMSB_src_02_semiring_DeltaRouter_jl__length
    MMSB_benchmark_benchmarks_jl___build_batch_deltas --> MMSB_src_02_semiring_DeltaRouter_jl__create_delta
    MMSB_benchmark_benchmarks_jl___full_system_benchmark_ --> MMSB_src_API_jl__update_page
    MMSB_benchmark_benchmarks_jl___full_system_benchmark_ --> MMSB_benchmark_benchmarks_jl___page
    MMSB_benchmark_benchmarks_jl___full_system_benchmark_ --> MMSB_src_03_dag_ShadowPageGraph_jl__add_dependency_
    MMSB_benchmark_benchmarks_jl___full_system_benchmark_ --> MMSB_src_01_types_MMSBState_jl__MMSBState
    MMSB_benchmark_benchmarks_jl___full_system_benchmark_ --> MMSB_src_01_types_MMSBState_jl__MMSBState
    MMSB_benchmark_benchmarks_jl___full_system_benchmark_ --> MMSB_src_04_propagation_PropagationEngine_jl__register_passthrough_recompute_
    MMSB_benchmark_benchmarks_jl___full_system_benchmark_ --> MMSB_src_01_page_ReplayEngine_jl__replay_to_epoch
    MMSB_benchmark_benchmarks_jl___full_system_benchmark_ --> MMSB_src_01_page_TLog_jl__load_checkpoint_
    MMSB_benchmark_benchmarks_jl___full_system_benchmark_ --> MMSB_benchmark_benchmarks_jl___checkpoint
    MMSB_benchmark_benchmarks_jl___full_system_benchmark_ --> MMSB_benchmark_benchmarks_jl___populate_pages_
    MMSB_benchmark_benchmarks_jl___full_system_benchmark_ --> MMSB_benchmark_benchmarks_jl___start_state
    MMSB_benchmark_benchmarks_jl___full_system_benchmark_ --> MMSB_benchmark_benchmarks_jl___stop_state_
    MMSB_benchmark_benchmarks_jl___full_system_benchmark_ --> MMSB_src_API_jl__length
    MMSB_benchmark_benchmarks_jl___full_system_benchmark_ --> MMSB_src_02_semiring_DeltaRouter_jl__length
    MMSB_benchmark_benchmarks_jl___collect_instrumentation_report --> MMSB_src_API_jl__update_page
    MMSB_benchmark_benchmarks_jl___collect_instrumentation_report --> MMSB_benchmark_benchmarks_jl___page
    MMSB_benchmark_benchmarks_jl___collect_instrumentation_report --> MMSB_src_03_dag_ShadowPageGraph_jl__add_dependency_
    MMSB_benchmark_benchmarks_jl___collect_instrumentation_report --> MMSB_src_06_utility_Monitoring_jl__get_stats
    MMSB_benchmark_benchmarks_jl___collect_instrumentation_report --> MMSB_src_06_utility_Monitoring_jl__reset_stats_
    MMSB_benchmark_benchmarks_jl___collect_instrumentation_report --> MMSB_src_04_propagation_PropagationEngine_jl__register_passthrough_recompute_
    MMSB_benchmark_benchmarks_jl___collect_instrumentation_report --> MMSB_src_02_semiring_semiring_ops_rs__fold_add
    MMSB_benchmark_benchmarks_jl___collect_instrumentation_report --> MMSB_src_02_semiring_Semiring_jl__boolean_fold_add
    MMSB_benchmark_benchmarks_jl___collect_instrumentation_report --> MMSB_src_02_semiring_semiring_ops_rs__fold_add
    MMSB_benchmark_benchmarks_jl___collect_instrumentation_report --> MMSB_src_02_semiring_Semiring_jl__tropical_fold_add
    MMSB_benchmark_benchmarks_jl___collect_instrumentation_report --> MMSB_benchmark_benchmarks_jl___measure_ns
    MMSB_benchmark_benchmarks_jl___collect_instrumentation_report --> MMSB_benchmark_benchmarks_jl___page
    MMSB_benchmark_benchmarks_jl___collect_instrumentation_report --> MMSB_benchmark_benchmarks_jl___start_state
    MMSB_benchmark_benchmarks_jl___collect_instrumentation_report --> MMSB_benchmark_benchmarks_jl___stop_state_
    MMSB_benchmark_benchmarks_jl___to_mutable --> MMSB_benchmark_benchmarks_jl___to_mutable
    MMSB_benchmark_helpers_jl__analyze_results --> MMSB_benchmark_helpers_jl___format_bytes
    MMSB_benchmark_helpers_jl__analyze_results --> MMSB_benchmark_helpers_jl___format_time
    MMSB_benchmark_helpers_jl__check_performance_targets --> MMSB_benchmark_helpers_jl___format_time
    MMSB_src_ffi_FFIWrapper_jl__ensure_rust_artifacts --> MMSB_src_ffi_FFIWrapper_jl__rust_artifacts_available
    MMSB_src_ffi_FFIWrapper_jl__rust_page_read_ --> MMSB_src_ffi_FFIWrapper_jl__ensure_rust_artifacts
    MMSB_src_ffi_FFIWrapper_jl__rust_page_read_ --> MMSB_src_API_jl__length
    MMSB_src_ffi_FFIWrapper_jl__rust_page_read_ --> MMSB_src_02_semiring_DeltaRouter_jl__length
    MMSB_src_ffi_FFIWrapper_jl__rust_page_epoch --> MMSB_src_ffi_FFIWrapper_jl___check_rust_error
    MMSB_src_ffi_FFIWrapper_jl__rust_page_epoch --> MMSB_src_ffi_RustErrors_jl__check_rust_error
    MMSB_src_ffi_FFIWrapper_jl__rust_page_epoch --> MMSB_src_ffi_FFIWrapper_jl__ensure_rust_artifacts
    MMSB_src_ffi_FFIWrapper_jl__rust_page_metadata_blob --> MMSB_src_ffi_FFIWrapper_jl___check_rust_error
    MMSB_src_ffi_FFIWrapper_jl__rust_page_metadata_blob --> MMSB_src_ffi_RustErrors_jl__check_rust_error
    MMSB_src_ffi_FFIWrapper_jl__rust_page_metadata_blob --> MMSB_src_ffi_FFIWrapper_jl__ensure_rust_artifacts
    MMSB_src_ffi_FFIWrapper_jl__rust_page_metadata_import_ --> MMSB_src_ffi_FFIWrapper_jl__ensure_rust_artifacts
    MMSB_src_ffi_FFIWrapper_jl__rust_page_metadata_import_ --> MMSB_src_API_jl__length
    MMSB_src_ffi_FFIWrapper_jl__rust_page_metadata_import_ --> MMSB_src_02_semiring_DeltaRouter_jl__length
    MMSB_src_ffi_FFIWrapper_jl__rust_delta_free_ --> MMSB_src_ffi_FFIWrapper_jl___check_rust_error
    MMSB_src_ffi_FFIWrapper_jl__rust_delta_free_ --> MMSB_src_ffi_RustErrors_jl__check_rust_error
    MMSB_src_ffi_FFIWrapper_jl__rust_delta_free_ --> MMSB_src_ffi_FFIWrapper_jl__ensure_rust_artifacts
    MMSB_src_ffi_FFIWrapper_jl__rust_delta_apply_ --> MMSB_src_ffi_FFIWrapper_jl___check_rust_error
    MMSB_src_ffi_FFIWrapper_jl__rust_delta_apply_ --> MMSB_src_ffi_RustErrors_jl__check_rust_error
    MMSB_src_ffi_FFIWrapper_jl__rust_delta_apply_ --> MMSB_src_ffi_FFIWrapper_jl__ensure_rust_artifacts
    MMSB_src_ffi_FFIWrapper_jl__rust_allocator_new --> MMSB_src_ffi_FFIWrapper_jl___check_rust_error
    MMSB_src_ffi_FFIWrapper_jl__rust_allocator_new --> MMSB_src_ffi_RustErrors_jl__check_rust_error
    MMSB_src_ffi_FFIWrapper_jl__rust_allocator_new --> MMSB_src_ffi_FFIWrapper_jl__ensure_rust_artifacts
    MMSB_src_ffi_FFIWrapper_jl__rust_allocator_free_ --> MMSB_src_ffi_FFIWrapper_jl___check_rust_error
    MMSB_src_ffi_FFIWrapper_jl__rust_allocator_free_ --> MMSB_src_ffi_RustErrors_jl__check_rust_error
    MMSB_src_ffi_FFIWrapper_jl__rust_allocator_free_ --> MMSB_src_ffi_FFIWrapper_jl__ensure_rust_artifacts
    MMSB_src_ffi_FFIWrapper_jl__rust_allocator_release_ --> MMSB_src_ffi_FFIWrapper_jl___check_rust_error
    MMSB_src_ffi_FFIWrapper_jl__rust_allocator_release_ --> MMSB_src_ffi_RustErrors_jl__check_rust_error
    MMSB_src_ffi_FFIWrapper_jl__rust_allocator_release_ --> MMSB_src_ffi_FFIWrapper_jl__ensure_rust_artifacts
    MMSB_src_ffi_FFIWrapper_jl__rust_allocator_get_page --> MMSB_src_ffi_FFIWrapper_jl___check_rust_error
    MMSB_src_ffi_FFIWrapper_jl__rust_allocator_get_page --> MMSB_src_ffi_RustErrors_jl__check_rust_error
    MMSB_src_ffi_FFIWrapper_jl__rust_allocator_get_page --> MMSB_src_ffi_FFIWrapper_jl__ensure_rust_artifacts
    MMSB_src_ffi_FFIWrapper_jl__rust_tlog_new --> MMSB_src_ffi_FFIWrapper_jl___check_rust_error
    MMSB_src_ffi_FFIWrapper_jl__rust_tlog_new --> MMSB_src_ffi_RustErrors_jl__check_rust_error
    MMSB_src_ffi_FFIWrapper_jl__rust_tlog_new --> MMSB_src_ffi_FFIWrapper_jl__ensure_rust_artifacts
    MMSB_src_ffi_FFIWrapper_jl__rust_tlog_free_ --> MMSB_src_ffi_FFIWrapper_jl___check_rust_error
    MMSB_src_ffi_FFIWrapper_jl__rust_tlog_free_ --> MMSB_src_ffi_RustErrors_jl__check_rust_error
    MMSB_src_ffi_FFIWrapper_jl__rust_tlog_free_ --> MMSB_src_ffi_FFIWrapper_jl__ensure_rust_artifacts
    MMSB_src_ffi_FFIWrapper_jl__rust_tlog_append_ --> MMSB_src_ffi_FFIWrapper_jl___check_rust_error
    MMSB_src_ffi_FFIWrapper_jl__rust_tlog_append_ --> MMSB_src_ffi_RustErrors_jl__check_rust_error
    MMSB_src_ffi_FFIWrapper_jl__rust_tlog_append_ --> MMSB_src_ffi_FFIWrapper_jl__ensure_rust_artifacts
    MMSB_src_ffi_FFIWrapper_jl__rust_tlog_reader_new --> MMSB_src_ffi_FFIWrapper_jl___check_rust_error
    MMSB_src_ffi_FFIWrapper_jl__rust_tlog_reader_new --> MMSB_src_ffi_RustErrors_jl__check_rust_error
    MMSB_src_ffi_FFIWrapper_jl__rust_tlog_reader_new --> MMSB_src_ffi_FFIWrapper_jl__ensure_rust_artifacts
    MMSB_src_ffi_FFIWrapper_jl__rust_tlog_reader_free_ --> MMSB_src_ffi_FFIWrapper_jl___check_rust_error
    MMSB_src_ffi_FFIWrapper_jl__rust_tlog_reader_free_ --> MMSB_src_ffi_RustErrors_jl__check_rust_error
    MMSB_src_ffi_FFIWrapper_jl__rust_tlog_reader_free_ --> MMSB_src_ffi_FFIWrapper_jl__ensure_rust_artifacts
    MMSB_src_ffi_FFIWrapper_jl__rust_tlog_reader_next --> MMSB_src_ffi_FFIWrapper_jl___check_rust_error
    MMSB_src_ffi_FFIWrapper_jl__rust_tlog_reader_next --> MMSB_src_ffi_RustErrors_jl__check_rust_error
    MMSB_src_ffi_FFIWrapper_jl__rust_tlog_reader_next --> MMSB_src_ffi_FFIWrapper_jl__ensure_rust_artifacts
    MMSB_src_ffi_FFIWrapper_jl__rust_tlog_summary --> MMSB_src_ffi_FFIWrapper_jl___check_rust_error
    MMSB_src_ffi_FFIWrapper_jl__rust_tlog_summary --> MMSB_src_ffi_RustErrors_jl__check_rust_error
    MMSB_src_ffi_FFIWrapper_jl__rust_tlog_summary --> MMSB_src_ffi_FFIWrapper_jl__ensure_rust_artifacts
    MMSB_src_ffi_FFIWrapper_jl__rust_delta_id --> MMSB_src_ffi_FFIWrapper_jl___check_rust_error
    MMSB_src_ffi_FFIWrapper_jl__rust_delta_id --> MMSB_src_ffi_RustErrors_jl__check_rust_error
    MMSB_src_ffi_FFIWrapper_jl__rust_delta_id --> MMSB_src_ffi_FFIWrapper_jl__ensure_rust_artifacts
    MMSB_src_ffi_FFIWrapper_jl__rust_delta_page_id --> MMSB_src_ffi_FFIWrapper_jl___check_rust_error
    MMSB_src_ffi_FFIWrapper_jl__rust_delta_page_id --> MMSB_src_ffi_RustErrors_jl__check_rust_error
    MMSB_src_ffi_FFIWrapper_jl__rust_delta_page_id --> MMSB_src_ffi_FFIWrapper_jl__ensure_rust_artifacts
    MMSB_src_ffi_FFIWrapper_jl__rust_delta_epoch --> MMSB_src_ffi_FFIWrapper_jl___check_rust_error
    MMSB_src_ffi_FFIWrapper_jl__rust_delta_epoch --> MMSB_src_ffi_RustErrors_jl__check_rust_error
    MMSB_src_ffi_FFIWrapper_jl__rust_delta_epoch --> MMSB_src_ffi_FFIWrapper_jl__ensure_rust_artifacts
    MMSB_src_ffi_FFIWrapper_jl__rust_delta_is_sparse --> MMSB_src_ffi_FFIWrapper_jl___check_rust_error
    MMSB_src_ffi_FFIWrapper_jl__rust_delta_is_sparse --> MMSB_src_ffi_RustErrors_jl__check_rust_error
    MMSB_src_ffi_FFIWrapper_jl__rust_delta_is_sparse --> MMSB_src_ffi_FFIWrapper_jl__ensure_rust_artifacts
    MMSB_src_ffi_FFIWrapper_jl__rust_delta_timestamp --> MMSB_src_ffi_FFIWrapper_jl___check_rust_error
    MMSB_src_ffi_FFIWrapper_jl__rust_delta_timestamp --> MMSB_src_ffi_RustErrors_jl__check_rust_error
    MMSB_src_ffi_FFIWrapper_jl__rust_delta_timestamp --> MMSB_src_ffi_FFIWrapper_jl__ensure_rust_artifacts
    MMSB_src_ffi_FFIWrapper_jl__rust_delta_source --> MMSB_src_ffi_FFIWrapper_jl___check_rust_error
    MMSB_src_ffi_FFIWrapper_jl__rust_delta_source --> MMSB_src_ffi_RustErrors_jl__check_rust_error
    MMSB_src_ffi_FFIWrapper_jl__rust_delta_source --> MMSB_src_ffi_FFIWrapper_jl__ensure_rust_artifacts
    MMSB_src_ffi_FFIWrapper_jl__rust_delta_mask --> MMSB_src_ffi_FFIWrapper_jl___check_rust_error
    MMSB_src_ffi_FFIWrapper_jl__rust_delta_mask --> MMSB_src_ffi_RustErrors_jl__check_rust_error
    MMSB_src_ffi_FFIWrapper_jl__rust_delta_mask --> MMSB_src_ffi_FFIWrapper_jl__ensure_rust_artifacts
    MMSB_src_ffi_FFIWrapper_jl__rust_delta_payload --> MMSB_src_ffi_FFIWrapper_jl___check_rust_error
    MMSB_src_ffi_FFIWrapper_jl__rust_delta_payload --> MMSB_src_ffi_RustErrors_jl__check_rust_error
    MMSB_src_ffi_FFIWrapper_jl__rust_delta_payload --> MMSB_src_ffi_FFIWrapper_jl__ensure_rust_artifacts
    MMSB_src_ffi_FFIWrapper_jl__rust_delta_set_intent_metadata_ --> MMSB_src_ffi_FFIWrapper_jl___check_rust_error
    MMSB_src_ffi_FFIWrapper_jl__rust_delta_set_intent_metadata_ --> MMSB_src_ffi_RustErrors_jl__check_rust_error
    MMSB_src_ffi_FFIWrapper_jl__rust_delta_set_intent_metadata_ --> MMSB_src_ffi_FFIWrapper_jl__ensure_rust_artifacts
    MMSB_src_ffi_FFIWrapper_jl__rust_delta_set_intent_metadata_ --> MMSB_src_API_jl__length
    MMSB_src_ffi_FFIWrapper_jl__rust_delta_set_intent_metadata_ --> MMSB_src_02_semiring_DeltaRouter_jl__length
    MMSB_src_ffi_FFIWrapper_jl__rust_delta_intent_metadata --> MMSB_src_ffi_FFIWrapper_jl___check_rust_error
    MMSB_src_ffi_FFIWrapper_jl__rust_delta_intent_metadata --> MMSB_src_ffi_RustErrors_jl__check_rust_error
    MMSB_src_ffi_FFIWrapper_jl__rust_delta_intent_metadata --> MMSB_src_ffi_FFIWrapper_jl__ensure_rust_artifacts
    MMSB_src_ffi_FFIWrapper_jl__rust_checkpoint_write_ --> MMSB_src_ffi_FFIWrapper_jl___check_rust_error
    MMSB_src_ffi_FFIWrapper_jl__rust_checkpoint_write_ --> MMSB_src_ffi_RustErrors_jl__check_rust_error
    MMSB_src_ffi_FFIWrapper_jl__rust_checkpoint_write_ --> MMSB_src_ffi_FFIWrapper_jl__ensure_rust_artifacts
    MMSB_src_ffi_FFIWrapper_jl__rust_checkpoint_load_ --> MMSB_src_ffi_FFIWrapper_jl___check_rust_error
    MMSB_src_ffi_FFIWrapper_jl__rust_checkpoint_load_ --> MMSB_src_ffi_RustErrors_jl__check_rust_error
    MMSB_src_ffi_FFIWrapper_jl__rust_checkpoint_load_ --> MMSB_src_ffi_FFIWrapper_jl__ensure_rust_artifacts
    MMSB_src_ffi_FFIWrapper_jl__rust_allocator_page_infos --> MMSB_src_ffi_FFIWrapper_jl___check_rust_error
    MMSB_src_ffi_FFIWrapper_jl__rust_allocator_page_infos --> MMSB_src_ffi_RustErrors_jl__check_rust_error
    MMSB_src_ffi_FFIWrapper_jl__rust_allocator_page_infos --> MMSB_src_ffi_FFIWrapper_jl__ensure_rust_artifacts
    MMSB_src_ffi_FFIWrapper_jl__rust_allocator_acquire_page --> MMSB_src_ffi_FFIWrapper_jl___check_rust_error
    MMSB_src_ffi_FFIWrapper_jl__rust_allocator_acquire_page --> MMSB_src_ffi_RustErrors_jl__check_rust_error
    MMSB_src_ffi_FFIWrapper_jl__rust_allocator_acquire_page --> MMSB_src_ffi_FFIWrapper_jl__ensure_rust_artifacts
    MMSB_src_ffi_FFIWrapper_jl__rust_get_last_error --> MMSB_src_ffi_FFIWrapper_jl__ensure_rust_artifacts
    MMSB_src_ffi_FFIWrapper_jl__rust_semiring_tropical_fold_add --> MMSB_src_ffi_FFIWrapper_jl__ensure_rust_artifacts
    MMSB_src_ffi_FFIWrapper_jl__rust_semiring_tropical_fold_add --> MMSB_src_API_jl__length
    MMSB_src_ffi_FFIWrapper_jl__rust_semiring_tropical_fold_add --> MMSB_src_02_semiring_DeltaRouter_jl__length
    MMSB_src_ffi_FFIWrapper_jl__rust_semiring_tropical_fold_mul --> MMSB_src_ffi_FFIWrapper_jl__ensure_rust_artifacts
    MMSB_src_ffi_FFIWrapper_jl__rust_semiring_tropical_fold_mul --> MMSB_src_API_jl__length
    MMSB_src_ffi_FFIWrapper_jl__rust_semiring_tropical_fold_mul --> MMSB_src_02_semiring_DeltaRouter_jl__length
    MMSB_src_ffi_FFIWrapper_jl__rust_semiring_tropical_accumulate --> MMSB_src_ffi_FFIWrapper_jl___check_rust_error
    MMSB_src_ffi_FFIWrapper_jl__rust_semiring_tropical_accumulate --> MMSB_src_ffi_RustErrors_jl__check_rust_error
    MMSB_src_ffi_FFIWrapper_jl__rust_semiring_tropical_accumulate --> MMSB_src_ffi_FFIWrapper_jl__ensure_rust_artifacts
    MMSB_src_ffi_FFIWrapper_jl__rust_semiring_boolean_fold_add --> MMSB_src_ffi_FFIWrapper_jl__ensure_rust_artifacts
    MMSB_src_ffi_FFIWrapper_jl__rust_semiring_boolean_fold_add --> MMSB_src_API_jl__length
    MMSB_src_ffi_FFIWrapper_jl__rust_semiring_boolean_fold_add --> MMSB_src_02_semiring_DeltaRouter_jl__length
    MMSB_src_ffi_FFIWrapper_jl__rust_semiring_boolean_fold_mul --> MMSB_src_ffi_FFIWrapper_jl__ensure_rust_artifacts
    MMSB_src_ffi_FFIWrapper_jl__rust_semiring_boolean_fold_mul --> MMSB_src_API_jl__length
    MMSB_src_ffi_FFIWrapper_jl__rust_semiring_boolean_fold_mul --> MMSB_src_02_semiring_DeltaRouter_jl__length
    MMSB_src_ffi_FFIWrapper_jl__rust_semiring_boolean_accumulate --> MMSB_src_ffi_FFIWrapper_jl___check_rust_error
    MMSB_src_ffi_FFIWrapper_jl__rust_semiring_boolean_accumulate --> MMSB_src_ffi_RustErrors_jl__check_rust_error
    MMSB_src_ffi_FFIWrapper_jl__rust_semiring_boolean_accumulate --> MMSB_src_ffi_FFIWrapper_jl__ensure_rust_artifacts
    MMSB_src_ffi_RustErrors_jl__check_rust_error --> MMSB_src_ffi_FFIWrapper_jl__rust_get_last_error
    MMSB_src_00_physical_DeviceFallback_jl__CPUPropagationQueue --> MMSB_src_00_physical_DeviceFallback_jl__CPUPropagationQueue
    MMSB_src_00_physical_DeviceSync_jl__sync_bidirectional_ --> MMSB_src_00_physical_DeviceSync_jl__sync_page_to_gpu_
    MMSB_src_00_physical_DeviceSync_jl__ensure_page_on_device_ --> MMSB_src_00_physical_DeviceSync_jl__sync_page_to_cpu_
    MMSB_src_00_physical_DeviceSync_jl__ensure_page_on_device_ --> MMSB_src_00_physical_DeviceSync_jl__sync_page_to_gpu_
    MMSB_src_00_physical_DeviceSync_jl__batch_sync_to_gpu_ --> MMSB_src_00_physical_DeviceSync_jl__sync_page_to_gpu_
    MMSB_src_00_physical_DeviceSync_jl__batch_sync_to_cpu_ --> MMSB_src_00_physical_DeviceSync_jl__sync_page_to_cpu_
    MMSB_src_00_physical_DeviceSync_jl__prefetch_pages_to_gpu_ --> MMSB_src_00_physical_DeviceSync_jl__batch_sync_to_gpu_
    MMSB_src_00_physical_DeviceSync_jl__prefetch_pages_to_gpu_ --> MMSB_benchmark_benchmarks_jl___page
    MMSB_src_00_physical_DeviceSync_jl__prefetch_pages_to_gpu_ --> MMSB_src_01_types_MMSBState_jl__get_page
    MMSB_src_00_physical_GPUKernels_jl__delta_merge_kernel_ --> MMSB_src_API_jl__length
    MMSB_src_00_physical_GPUKernels_jl__delta_merge_kernel_ --> MMSB_src_02_semiring_DeltaRouter_jl__length
    MMSB_src_00_physical_GPUKernels_jl__launch_delta_merge_ --> MMSB_src_00_physical_GPUKernels_jl__compute_optimal_kernel_config
    MMSB_src_00_physical_GPUKernels_jl__launch_delta_merge_ --> MMSB_src_00_physical_GPUKernels_jl__delta_merge_kernel_
    MMSB_src_00_physical_GPUKernels_jl__launch_delta_merge_ --> MMSB_src_API_jl__length
    MMSB_src_00_physical_GPUKernels_jl__launch_delta_merge_ --> MMSB_src_02_semiring_DeltaRouter_jl__length
    MMSB_src_00_physical_GPUKernels_jl__launch_page_copy_ --> MMSB_src_00_physical_GPUKernels_jl__compute_optimal_kernel_config
    MMSB_src_00_physical_GPUKernels_jl__launch_page_copy_ --> MMSB_src_00_physical_GPUKernels_jl__page_copy_kernel_
    MMSB_src_00_physical_GPUKernels_jl__page_zero_kernel_ --> MMSB_src_API_jl__length
    MMSB_src_00_physical_GPUKernels_jl__page_zero_kernel_ --> MMSB_src_02_semiring_DeltaRouter_jl__length
    MMSB_src_00_physical_GPUKernels_jl__launch_page_zero_ --> MMSB_src_00_physical_GPUKernels_jl__compute_optimal_kernel_config
    MMSB_src_00_physical_GPUKernels_jl__launch_page_zero_ --> MMSB_src_API_jl__length
    MMSB_src_00_physical_GPUKernels_jl__launch_page_zero_ --> MMSB_src_02_semiring_DeltaRouter_jl__length
    MMSB_src_00_physical_GPUKernels_jl__launch_page_zero_ --> MMSB_src_00_physical_GPUKernels_jl__page_zero_kernel_
    MMSB_src_00_physical_GPUKernels_jl__page_compare_kernel_ --> MMSB_src_API_jl__length
    MMSB_src_00_physical_GPUKernels_jl__page_compare_kernel_ --> MMSB_src_02_semiring_DeltaRouter_jl__length
    MMSB_src_00_physical_GPUKernels_jl__launch_page_compare_ --> MMSB_src_00_physical_GPUKernels_jl__compute_optimal_kernel_config
    MMSB_src_00_physical_GPUKernels_jl__launch_page_compare_ --> MMSB_src_API_jl__length
    MMSB_src_00_physical_GPUKernels_jl__launch_page_compare_ --> MMSB_src_02_semiring_DeltaRouter_jl__length
    MMSB_src_00_physical_GPUKernels_jl__launch_page_compare_ --> MMSB_src_00_physical_GPUKernels_jl__page_compare_kernel_
    MMSB_src_00_physical_GPUKernels_jl__launch_sparse_delta_apply_ --> MMSB_src_00_physical_GPUKernels_jl__compute_optimal_kernel_config
    MMSB_src_00_physical_GPUKernels_jl__launch_sparse_delta_apply_ --> MMSB_src_API_jl__length
    MMSB_src_00_physical_GPUKernels_jl__launch_sparse_delta_apply_ --> MMSB_src_02_semiring_DeltaRouter_jl__length
    MMSB_src_00_physical_GPUKernels_jl__launch_sparse_delta_apply_ --> MMSB_src_00_physical_GPUKernels_jl__sparse_delta_apply_kernel_
    MMSB_src_00_physical_PageAllocator_jl__delete_page_ --> MMSB_src_03_dag_DependencyGraph_jl__get_children
    MMSB_src_00_physical_PageAllocator_jl__delete_page_ --> MMSB_src_03_dag_ShadowPageGraph_jl__get_children
    MMSB_src_00_physical_PageAllocator_jl__delete_page_ --> MMSB_src_03_dag_DependencyGraph_jl__get_parents
    MMSB_src_00_physical_PageAllocator_jl__delete_page_ --> MMSB_src_03_dag_ShadowPageGraph_jl__get_parents
    MMSB_src_00_physical_PageAllocator_jl__delete_page_ --> MMSB_src_03_dag_ShadowPageGraph_jl__remove_dependency_
    MMSB_src_00_physical_UnifiedMemory_jl__GPUMemoryPool --> MMSB_src_00_physical_UnifiedMemory_jl__GPUMemoryPool
    MMSB_src_00_physical_UnifiedMemory_jl__create_unified_page_ --> MMSB_src_01_page_Page_jl__Page
    MMSB_src_00_physical_UnifiedMemory_jl__create_unified_page_ --> MMSB_src_01_types_MMSBState_jl__allocate_page_id_
    MMSB_src_00_physical_UnifiedMemory_jl__convert_to_unified_ --> MMSB_src_00_physical_DeviceSync_jl__sync_page_to_cpu_
    MMSB_src_01_page_Delta_jl__Delta --> MMSB_src_ffi_FFIWrapper_jl__rust_delta_epoch
    MMSB_src_01_page_Delta_jl__Delta --> MMSB_src_ffi_FFIWrapper_jl__rust_delta_free_
    MMSB_src_01_page_Delta_jl__Delta --> MMSB_src_ffi_FFIWrapper_jl__rust_delta_id
    MMSB_src_01_page_Delta_jl__Delta --> MMSB_src_ffi_FFIWrapper_jl__rust_delta_intent_metadata
    MMSB_src_01_page_Delta_jl__Delta --> MMSB_src_01_page_Delta_jl__intent_metadata
    MMSB_src_01_page_Delta_jl__Delta --> MMSB_src_ffi_FFIWrapper_jl__rust_delta_is_sparse
    MMSB_src_01_page_Delta_jl__Delta --> MMSB_src_ffi_FFIWrapper_jl__rust_delta_mask
    MMSB_src_01_page_Delta_jl__Delta --> MMSB_src_ffi_FFIWrapper_jl__rust_delta_page_id
    MMSB_src_01_page_Delta_jl__Delta --> MMSB_src_ffi_FFIWrapper_jl__rust_delta_payload
    MMSB_src_01_page_Delta_jl__Delta --> MMSB_src_ffi_FFIWrapper_jl__rust_delta_source
    MMSB_src_01_page_Delta_jl__Delta --> MMSB_src_ffi_FFIWrapper_jl__rust_delta_timestamp
    MMSB_src_01_page_Delta_jl__apply_delta_ --> MMSB_src_ffi_FFIWrapper_jl__rust_delta_apply_
    MMSB_src_01_page_Delta_jl__dense_data --> MMSB_src_API_jl__length
    MMSB_src_01_page_Delta_jl__dense_data --> MMSB_src_02_semiring_DeltaRouter_jl__length
    MMSB_src_01_page_Delta_jl__deserialize_delta --> MMSB_src_01_page_Delta_jl__Delta
    MMSB_src_01_page_Delta_jl__deserialize_delta --> MMSB_src_01_page_Delta_jl__Delta
    MMSB_src_01_page_Delta_jl__set_intent_metadata_ --> MMSB_src_ffi_FFIWrapper_jl__rust_delta_set_intent_metadata_
    MMSB_src_01_page_Delta_jl__set_intent_metadata_ --> MMSB_src_01_page_Delta_jl__set_intent_metadata_
    MMSB_src_01_page_Delta_jl__set_intent_metadata_ --> MMSB_src_01_page_Delta_jl___encode_metadata_dict
    MMSB_src_01_page_Delta_jl__set_intent_metadata_ --> MMSB_src_01_page_Page_jl___encode_metadata_dict
    MMSB_src_01_page_Delta_jl___encode_metadata_value --> MMSB_src_01_page_Delta_jl___encode_metadata_dict
    MMSB_src_01_page_Delta_jl___encode_metadata_value --> MMSB_src_01_page_Page_jl___encode_metadata_dict
    MMSB_src_01_page_Delta_jl___encode_metadata_value --> MMSB_src_01_page_Delta_jl___encode_metadata_value
    MMSB_src_01_page_Delta_jl___encode_metadata_value --> MMSB_src_01_page_Delta_jl___escape_metadata_string
    MMSB_src_01_page_Delta_jl___encode_metadata_dict --> MMSB_src_01_page_Delta_jl___encode_metadata_value
    MMSB_src_01_page_Delta_jl___encode_metadata_dict --> MMSB_src_01_page_Delta_jl___escape_metadata_string
    MMSB_src_01_page_Delta_jl__merge_deltas_simd_ --> MMSB_src_API_jl__length
    MMSB_src_01_page_Delta_jl__merge_deltas_simd_ --> MMSB_src_02_semiring_DeltaRouter_jl__length
    MMSB_src_01_page_Delta_jl___decode_metadata --> MMSB_src_01_page_Delta_jl___parse_metadata_value
    MMSB_src_01_page_Delta_jl___parse_metadata_value --> MMSB_src_01_page_Delta_jl___parse_metadata_array
    MMSB_src_01_page_Delta_jl___parse_metadata_value --> MMSB_src_01_page_Delta_jl___parse_metadata_number
    MMSB_src_01_page_Delta_jl___parse_metadata_value --> MMSB_src_01_page_Delta_jl___parse_metadata_object
    MMSB_src_01_page_Delta_jl___parse_metadata_value --> MMSB_src_01_page_Delta_jl___parse_metadata_string
    MMSB_src_01_page_Delta_jl___parse_metadata_value --> MMSB_src_01_page_Delta_jl___peek
    MMSB_src_01_page_Delta_jl___parse_metadata_value --> MMSB_src_01_page_Delta_jl___skip_ws
    MMSB_src_01_page_Delta_jl___parse_metadata_object --> MMSB_src_01_page_Delta_jl___consume
    MMSB_src_01_page_Delta_jl___parse_metadata_object --> MMSB_src_01_page_Delta_jl___parse_metadata_string
    MMSB_src_01_page_Delta_jl___parse_metadata_object --> MMSB_src_01_page_Delta_jl___parse_metadata_value
    MMSB_src_01_page_Delta_jl___parse_metadata_object --> MMSB_src_01_page_Delta_jl___peek
    MMSB_src_01_page_Delta_jl___parse_metadata_object --> MMSB_src_01_page_Delta_jl___skip_ws
    MMSB_src_01_page_Delta_jl___parse_metadata_array --> MMSB_src_01_page_Delta_jl___consume
    MMSB_src_01_page_Delta_jl___parse_metadata_array --> MMSB_src_01_page_Delta_jl___parse_metadata_value
    MMSB_src_01_page_Delta_jl___parse_metadata_array --> MMSB_src_01_page_Delta_jl___peek
    MMSB_src_01_page_Delta_jl___parse_metadata_array --> MMSB_src_01_page_Delta_jl___skip_ws
    MMSB_src_01_page_Delta_jl___parse_metadata_string --> MMSB_src_01_page_Delta_jl___consume
    MMSB_src_01_page_Delta_jl___consume --> MMSB_src_01_page_Delta_jl___peek
    MMSB_src_01_page_Page_jl__read_page --> MMSB_src_ffi_FFIWrapper_jl__rust_page_read_
    MMSB_src_01_page_Page_jl___apply_metadata_ --> MMSB_src_ffi_FFIWrapper_jl__rust_page_metadata_import_
    MMSB_src_01_page_Page_jl___apply_metadata_ --> MMSB_src_01_page_Delta_jl___encode_metadata_dict
    MMSB_src_01_page_Page_jl___apply_metadata_ --> MMSB_src_01_page_Page_jl___encode_metadata_dict
    MMSB_src_01_page_Page_jl___encode_metadata_dict --> MMSB_src_01_page_Page_jl___coerce_metadata_value
    MMSB_src_01_page_Page_jl___encode_metadata_dict --> MMSB_src_API_jl__length
    MMSB_src_01_page_Page_jl___encode_metadata_dict --> MMSB_src_02_semiring_DeltaRouter_jl__length
    MMSB_src_01_page_Page_jl__metadata_from_blob --> MMSB_src_01_page_Page_jl___decode_metadata_blob
    MMSB_src_01_page_ReplayEngine_jl___blank_state_like --> MMSB_src_ffi_FFIWrapper_jl__rust_allocator_allocate
    MMSB_src_01_page_ReplayEngine_jl___blank_state_like --> MMSB_src_01_types_MMSBState_jl__MMSBConfig
    MMSB_src_01_page_ReplayEngine_jl___blank_state_like --> MMSB_src_01_types_MMSBState_jl__MMSBState
    MMSB_src_01_page_ReplayEngine_jl___blank_state_like --> MMSB_src_01_types_MMSBState_jl__MMSBState
    MMSB_src_01_page_ReplayEngine_jl___blank_state_like --> MMSB_src_01_page_Page_jl__Page
    MMSB_src_01_page_ReplayEngine_jl___blank_state_like --> MMSB_src_01_page_Page_jl__activate_
    MMSB_src_01_page_ReplayEngine_jl___blank_state_like --> MMSB_src_01_page_Page_jl__initialize_
    MMSB_src_01_page_ReplayEngine_jl___blank_state_like --> MMSB_src_01_types_MMSBState_jl__register_page_
    MMSB_src_01_page_ReplayEngine_jl___apply_delta_ --> MMSB_src_ffi_FFIWrapper_jl__rust_delta_apply_
    MMSB_src_01_page_ReplayEngine_jl___all_deltas --> MMSB_src_01_page_TLog_jl__query_log
    MMSB_src_01_page_ReplayEngine_jl__replay_to_epoch --> MMSB_src_01_page_ReplayEngine_jl___all_deltas
    MMSB_src_01_page_ReplayEngine_jl__replay_to_epoch --> MMSB_src_01_page_Delta_jl__apply_delta_
    MMSB_src_01_page_ReplayEngine_jl__replay_to_epoch --> MMSB_src_01_page_ReplayEngine_jl___apply_delta_
    MMSB_src_01_page_ReplayEngine_jl__replay_to_epoch --> MMSB_src_01_page_ReplayEngine_jl___blank_state_like
    MMSB_src_01_page_ReplayEngine_jl__replay_to_epoch --> MMSB_benchmark_benchmarks_jl___page
    MMSB_src_01_page_ReplayEngine_jl__replay_to_epoch --> MMSB_src_01_types_MMSBState_jl__get_page
    MMSB_src_01_page_ReplayEngine_jl__replay_to_timestamp --> MMSB_src_01_page_ReplayEngine_jl___all_deltas
    MMSB_src_01_page_ReplayEngine_jl__replay_to_timestamp --> MMSB_src_01_page_Delta_jl__apply_delta_
    MMSB_src_01_page_ReplayEngine_jl__replay_to_timestamp --> MMSB_src_01_page_ReplayEngine_jl___apply_delta_
    MMSB_src_01_page_ReplayEngine_jl__replay_to_timestamp --> MMSB_src_01_page_ReplayEngine_jl___blank_state_like
    MMSB_src_01_page_ReplayEngine_jl__replay_to_timestamp --> MMSB_benchmark_benchmarks_jl___page
    MMSB_src_01_page_ReplayEngine_jl__replay_to_timestamp --> MMSB_src_01_types_MMSBState_jl__get_page
    MMSB_src_01_page_ReplayEngine_jl__replay_page_history --> MMSB_src_ffi_FFIWrapper_jl__rust_page_epoch
    MMSB_src_01_page_ReplayEngine_jl__replay_page_history --> MMSB_src_ffi_FFIWrapper_jl__rust_page_write_masked_
    MMSB_src_01_page_ReplayEngine_jl__replay_page_history --> MMSB_src_01_page_Page_jl__Page
    MMSB_src_01_page_ReplayEngine_jl__replay_page_history --> MMSB_src_01_page_TLog_jl__query_log
    MMSB_src_01_page_ReplayEngine_jl__replay_page_history --> MMSB_src_01_page_Delta_jl__apply_delta_
    MMSB_src_01_page_ReplayEngine_jl__replay_page_history --> MMSB_src_01_page_ReplayEngine_jl___apply_delta_
    MMSB_src_01_page_ReplayEngine_jl__replay_page_history --> MMSB_src_01_page_Page_jl__activate_
    MMSB_src_01_page_ReplayEngine_jl__replay_page_history --> MMSB_benchmark_benchmarks_jl___page
    MMSB_src_01_page_ReplayEngine_jl__replay_page_history --> MMSB_src_01_types_MMSBState_jl__get_page
    MMSB_src_01_page_ReplayEngine_jl__replay_page_history --> MMSB_src_01_page_Page_jl__initialize_
    MMSB_src_01_page_ReplayEngine_jl__replay_page_history --> MMSB_tests_mmsb_tests_rs__read_page
    MMSB_src_01_page_ReplayEngine_jl__replay_page_history --> MMSB_benchmark_benchmarks_jl___page
    MMSB_src_01_page_ReplayEngine_jl__replay_page_history --> MMSB_src_01_page_Page_jl__read_page
    MMSB_src_01_page_ReplayEngine_jl__verify_state_consistency --> MMSB_benchmark_benchmarks_jl___page
    MMSB_src_01_page_ReplayEngine_jl__verify_state_consistency --> MMSB_src_01_types_MMSBState_jl__get_page
    MMSB_src_01_page_ReplayEngine_jl__verify_state_consistency --> MMSB_tests_mmsb_tests_rs__read_page
    MMSB_src_01_page_ReplayEngine_jl__verify_state_consistency --> MMSB_benchmark_benchmarks_jl___page
    MMSB_src_01_page_ReplayEngine_jl__verify_state_consistency --> MMSB_src_01_page_Page_jl__read_page
    MMSB_src_01_page_ReplayEngine_jl__verify_state_consistency --> MMSB_src_01_page_ReplayEngine_jl__replay_to_epoch
    MMSB_src_01_page_ReplayEngine_jl__replay_with_predicate --> MMSB_src_01_page_ReplayEngine_jl___all_deltas
    MMSB_src_01_page_ReplayEngine_jl__incremental_replay_ --> MMSB_src_01_page_Delta_jl__apply_delta_
    MMSB_src_01_page_ReplayEngine_jl__incremental_replay_ --> MMSB_src_01_page_ReplayEngine_jl___apply_delta_
    MMSB_src_01_page_ReplayEngine_jl__incremental_replay_ --> MMSB_benchmark_benchmarks_jl___page
    MMSB_src_01_page_ReplayEngine_jl__incremental_replay_ --> MMSB_src_01_types_MMSBState_jl__get_page
    MMSB_src_01_page_TLog_jl__append_to_log_ --> MMSB_src_ffi_FFIWrapper_jl__rust_tlog_append_
    MMSB_src_01_page_TLog_jl__append_to_log_ --> MMSB_src_01_page_TLog_jl___with_rust_errors
    MMSB_src_01_page_TLog_jl__log_summary --> MMSB_src_01_page_tlog_rs__summary
    MMSB_src_01_page_TLog_jl__log_summary --> MMSB_src_ffi_FFIWrapper_jl__rust_tlog_summary
    MMSB_src_01_page_TLog_jl__log_summary --> MMSB_src_01_page_TLog_jl__log_summary
    MMSB_src_01_page_TLog_jl__log_summary --> MMSB_src_01_page_TLog_jl___with_rust_errors
    MMSB_src_01_page_TLog_jl___iterate_log --> MMSB_src_ffi_FFIWrapper_jl__rust_tlog_reader_free_
    MMSB_src_01_page_TLog_jl___iterate_log --> MMSB_src_ffi_FFIWrapper_jl__rust_tlog_reader_new
    MMSB_src_01_page_TLog_jl__get_deltas_for_page --> MMSB_src_01_page_TLog_jl__query_log
    MMSB_src_01_page_TLog_jl__get_deltas_in_range --> MMSB_src_API_jl__length
    MMSB_src_01_page_TLog_jl__get_deltas_in_range --> MMSB_src_02_semiring_DeltaRouter_jl__length
    MMSB_src_01_page_TLog_jl__get_deltas_in_range --> MMSB_src_01_page_TLog_jl__query_log
    MMSB_src_01_page_TLog_jl__compute_log_statistics --> MMSB_src_01_page_tlog_rs__summary
    MMSB_src_01_page_TLog_jl__compute_log_statistics --> MMSB_src_01_page_TLog_jl__log_summary
    MMSB_src_01_page_TLog_jl__replay_log --> MMSB_src_01_page_ReplayEngine_jl__replay_to_epoch
    MMSB_src_01_page_TLog_jl__checkpoint_log_ --> MMSB_src_ffi_FFIWrapper_jl__rust_checkpoint_write_
    MMSB_src_01_page_TLog_jl__checkpoint_log_ --> MMSB_src_01_page_TLog_jl___with_rust_errors
    MMSB_src_01_page_TLog_jl__load_checkpoint_ --> MMSB_src_ffi_FFIWrapper_jl__rust_checkpoint_load_
    MMSB_src_01_page_TLog_jl__load_checkpoint_ --> MMSB_src_01_page_TLog_jl___with_rust_errors
    MMSB_src_01_page_TLog_jl___refresh_pages_ --> MMSB_benchmark_benchmarks_jl___page
    MMSB_src_01_page_TLog_jl___refresh_pages_ --> MMSB_src_ffi_FFIWrapper_jl__rust_allocator_acquire_page
    MMSB_src_01_page_TLog_jl___refresh_pages_ --> MMSB_src_ffi_FFIWrapper_jl__rust_allocator_page_infos
    MMSB_src_01_page_TLog_jl___refresh_pages_ --> MMSB_src_01_page_Page_jl__Page
    MMSB_src_01_page_TLog_jl___refresh_pages_ --> MMSB_src_01_page_Page_jl__activate_
    MMSB_src_01_page_TLog_jl___refresh_pages_ --> MMSB_src_01_page_Page_jl__initialize_
    MMSB_src_01_page_TLog_jl___refresh_pages_ --> MMSB_src_01_page_Page_jl__metadata_from_blob
    MMSB_src_01_types_MMSBState_jl__MMSBState --> MMSB_src_ffi_FFIWrapper_jl__rust_allocator_free_
    MMSB_src_01_types_MMSBState_jl__MMSBState --> MMSB_src_ffi_FFIWrapper_jl__rust_allocator_new
    MMSB_src_01_types_MMSBState_jl__MMSBState --> MMSB_src_ffi_FFIWrapper_jl__rust_tlog_free_
    MMSB_src_01_types_MMSBState_jl__MMSBState --> MMSB_src_ffi_FFIWrapper_jl__rust_tlog_new
    MMSB_src_01_types_MMSBState_jl__MMSBState --> MMSB_src_03_dag_ShadowPageGraph_jl__ShadowPageGraph
    MMSB_src_01_types_MMSBState_jl__MMSBState --> MMSB_src_01_types_MMSBState_jl__MMSBConfig
    MMSB_src_01_types_MMSBState_jl__MMSBState --> MMSB_src_01_types_MMSBState_jl__MMSBState
    MMSB_src_01_types_MMSBState_jl__MMSBState --> MMSB_src_01_types_MMSBState_jl__MMSBState
    MMSB_src_02_semiring_DeltaRouter_jl__length --> MMSB_src_API_jl__length
    MMSB_src_02_semiring_DeltaRouter_jl__length --> MMSB_src_02_semiring_DeltaRouter_jl__length
    MMSB_src_02_semiring_DeltaRouter_jl__batch_route_deltas_ --> MMSB_benchmark_benchmarks_jl___page
    MMSB_src_02_semiring_DeltaRouter_jl__batch_route_deltas_ --> MMSB_src_01_types_MMSBState_jl__get_page
    MMSB_src_02_semiring_DeltaRouter_jl__batch_route_deltas_ --> MMSB_src_02_semiring_DeltaRouter_jl__propagate_change_
    MMSB_src_02_semiring_DeltaRouter_jl__batch_route_deltas_ --> MMSB_src_02_semiring_DeltaRouter_jl__propagate_change_
    MMSB_src_02_semiring_DeltaRouter_jl__batch_route_deltas_ --> MMSB_src_04_propagation_PropagationEngine_jl__propagate_change_
    MMSB_src_02_semiring_DeltaRouter_jl__batch_route_deltas_ --> MMSB_src_04_propagation_PropagationEngine_jl__propagate_change_
    MMSB_src_02_semiring_DeltaRouter_jl__batch_route_deltas_ --> MMSB_src_02_semiring_DeltaRouter_jl__route_delta_
    MMSB_src_02_semiring_DeltaRouter_jl__propagate_change_ --> MMSB_src_02_semiring_DeltaRouter_jl__propagate_change_
    MMSB_src_02_semiring_DeltaRouter_jl__propagate_change_ --> MMSB_src_02_semiring_DeltaRouter_jl__propagate_change_
    MMSB_src_02_semiring_DeltaRouter_jl__propagate_change_ --> MMSB_src_04_propagation_PropagationEngine_jl__propagate_change_
    MMSB_src_02_semiring_DeltaRouter_jl__propagate_change_ --> MMSB_src_04_propagation_PropagationEngine_jl__propagate_change_
    MMSB_src_02_semiring_DeltaRouter_jl__propagate_change_ --> MMSB_src_02_semiring_DeltaRouter_jl__propagate_change_
    MMSB_src_02_semiring_DeltaRouter_jl__propagate_change_ --> MMSB_src_02_semiring_DeltaRouter_jl__propagate_change_
    MMSB_src_02_semiring_DeltaRouter_jl__propagate_change_ --> MMSB_src_04_propagation_PropagationEngine_jl__propagate_change_
    MMSB_src_02_semiring_DeltaRouter_jl__propagate_change_ --> MMSB_src_04_propagation_PropagationEngine_jl__propagate_change_
    MMSB_src_02_semiring_Semiring_jl___bool_buf --> MMSB_src_API_jl__length
    MMSB_src_02_semiring_Semiring_jl___bool_buf --> MMSB_src_02_semiring_DeltaRouter_jl__length
    MMSB_src_02_semiring_Semiring_jl__tropical_fold_add --> MMSB_src_02_semiring_semiring_ops_rs__fold_add
    MMSB_src_02_semiring_Semiring_jl__tropical_fold_add --> MMSB_src_ffi_FFIWrapper_jl__rust_semiring_tropical_fold_add
    MMSB_src_02_semiring_Semiring_jl__tropical_fold_add --> MMSB_src_02_semiring_Semiring_jl__tropical_fold_add
    MMSB_src_02_semiring_Semiring_jl__tropical_fold_add --> MMSB_src_02_semiring_Semiring_jl___FLOAT_BUF
    MMSB_src_02_semiring_Semiring_jl__tropical_fold_mul --> MMSB_src_02_semiring_semiring_ops_rs__fold_mul
    MMSB_src_02_semiring_Semiring_jl__tropical_fold_mul --> MMSB_src_ffi_FFIWrapper_jl__rust_semiring_tropical_fold_mul
    MMSB_src_02_semiring_Semiring_jl__tropical_fold_mul --> MMSB_src_02_semiring_Semiring_jl__tropical_fold_mul
    MMSB_src_02_semiring_Semiring_jl__tropical_fold_mul --> MMSB_src_02_semiring_Semiring_jl___FLOAT_BUF
    MMSB_src_02_semiring_Semiring_jl__tropical_accumulate --> MMSB_src_02_semiring_semiring_ops_rs__accumulate
    MMSB_src_02_semiring_Semiring_jl__tropical_accumulate --> MMSB_src_ffi_FFIWrapper_jl__rust_semiring_tropical_accumulate
    MMSB_src_02_semiring_Semiring_jl__tropical_accumulate --> MMSB_src_02_semiring_Semiring_jl__tropical_accumulate
    MMSB_src_02_semiring_Semiring_jl__boolean_fold_add --> MMSB_src_02_semiring_semiring_ops_rs__fold_add
    MMSB_src_02_semiring_Semiring_jl__boolean_fold_add --> MMSB_src_ffi_FFIWrapper_jl__rust_semiring_boolean_fold_add
    MMSB_src_02_semiring_Semiring_jl__boolean_fold_add --> MMSB_src_02_semiring_Semiring_jl__boolean_fold_add
    MMSB_src_02_semiring_Semiring_jl__boolean_fold_add --> MMSB_src_02_semiring_Semiring_jl___bool_buf
    MMSB_src_02_semiring_Semiring_jl__boolean_fold_mul --> MMSB_src_02_semiring_semiring_ops_rs__fold_mul
    MMSB_src_02_semiring_Semiring_jl__boolean_fold_mul --> MMSB_src_ffi_FFIWrapper_jl__rust_semiring_boolean_fold_mul
    MMSB_src_02_semiring_Semiring_jl__boolean_fold_mul --> MMSB_src_02_semiring_Semiring_jl__boolean_fold_mul
    MMSB_src_02_semiring_Semiring_jl__boolean_fold_mul --> MMSB_src_02_semiring_Semiring_jl___bool_buf
    MMSB_src_02_semiring_Semiring_jl__boolean_accumulate --> MMSB_src_02_semiring_semiring_ops_rs__accumulate
    MMSB_src_02_semiring_Semiring_jl__boolean_accumulate --> MMSB_src_ffi_FFIWrapper_jl__rust_semiring_boolean_accumulate
    MMSB_src_02_semiring_Semiring_jl__boolean_accumulate --> MMSB_src_02_semiring_Semiring_jl__boolean_accumulate
    MMSB_src_03_dag_DependencyGraph_jl__find_descendants --> MMSB_src_03_dag_DependencyGraph_jl__get_children
    MMSB_src_03_dag_DependencyGraph_jl__find_descendants --> MMSB_src_03_dag_ShadowPageGraph_jl__get_children
    MMSB_src_03_dag_DependencyGraph_jl__find_ancestors --> MMSB_src_03_dag_DependencyGraph_jl__get_parents
    MMSB_src_03_dag_DependencyGraph_jl__find_ancestors --> MMSB_src_03_dag_ShadowPageGraph_jl__get_parents
    MMSB_src_03_dag_DependencyGraph_jl__detect_cycles --> MMSB_src_03_dag_DependencyGraph_jl__dfs_cycle_detect
    MMSB_src_03_dag_DependencyGraph_jl__dfs_cycle_detect --> MMSB_src_03_dag_DependencyGraph_jl__dfs_cycle_detect
    MMSB_src_03_dag_DependencyGraph_jl__dfs_cycle_detect --> MMSB_src_03_dag_DependencyGraph_jl__get_children
    MMSB_src_03_dag_DependencyGraph_jl__dfs_cycle_detect --> MMSB_src_03_dag_ShadowPageGraph_jl__get_children
    MMSB_src_03_dag_DependencyGraph_jl__topological_order --> MMSB_src_03_dag_DependencyGraph_jl__get_children
    MMSB_src_03_dag_DependencyGraph_jl__topological_order --> MMSB_src_03_dag_ShadowPageGraph_jl__get_children
    MMSB_src_03_dag_DependencyGraph_jl__topological_order --> MMSB_src_API_jl__length
    MMSB_src_03_dag_DependencyGraph_jl__topological_order --> MMSB_src_02_semiring_DeltaRouter_jl__length
    MMSB_src_03_dag_DependencyGraph_jl__reverse_postorder --> MMSB_src_03_dag_DependencyGraph_jl__get_children
    MMSB_src_03_dag_DependencyGraph_jl__reverse_postorder --> MMSB_src_03_dag_ShadowPageGraph_jl__get_children
    MMSB_src_03_dag_DependencyGraph_jl__compute_closure --> MMSB_src_03_dag_DependencyGraph_jl__find_descendants
    MMSB_src_03_dag_EventSystem_jl__get_subscription_count --> MMSB_src_API_jl__length
    MMSB_src_03_dag_EventSystem_jl__get_subscription_count --> MMSB_src_02_semiring_DeltaRouter_jl__length
    MMSB_src_03_dag_EventSystem_jl__create_logging_subscriber --> MMSB_src_03_dag_EventSystem_jl__log_event_to_page_
    MMSB_src_03_dag_EventSystem_jl__create_logging_subscriber --> MMSB_src_03_dag_EventSystem_jl__subscribe_
    MMSB_src_03_dag_EventSystem_jl__log_event_to_page_ --> MMSB_src_03_dag_EventSystem_jl___serialize_event
    MMSB_src_03_dag_EventSystem_jl__log_event_to_page_ --> MMSB_benchmark_benchmarks_jl___page
    MMSB_src_03_dag_EventSystem_jl__log_event_to_page_ --> MMSB_src_01_types_MMSBState_jl__get_page
    MMSB_src_03_dag_EventSystem_jl__log_event_to_page_ --> MMSB_src_API_jl__length
    MMSB_src_03_dag_EventSystem_jl__log_event_to_page_ --> MMSB_src_02_semiring_DeltaRouter_jl__length
    MMSB_src_03_dag_EventSystem_jl__log_event_to_page_ --> MMSB_tests_mmsb_tests_rs__read_page
    MMSB_src_03_dag_EventSystem_jl__log_event_to_page_ --> MMSB_benchmark_benchmarks_jl___page
    MMSB_src_03_dag_EventSystem_jl__log_event_to_page_ --> MMSB_src_01_page_Page_jl__read_page
    MMSB_src_03_dag_ShadowPageGraph_jl__add_dependency_ --> MMSB_src_03_dag_ShadowPageGraph_jl___ensure_vertex_
    MMSB_src_03_dag_ShadowPageGraph_jl__add_dependency_ --> MMSB_src_03_dag_cycle_detection_rs__has_cycle
    MMSB_src_03_dag_ShadowPageGraph_jl__add_dependency_ --> MMSB_src_03_dag_ShadowPageGraph_jl__has_cycle
    MMSB_src_03_dag_ShadowPageGraph_jl__add_dependency_ --> MMSB_src_03_dag_ShadowPageGraph_jl__remove_dependency_
    MMSB_src_03_dag_ShadowPageGraph_jl___dfs_has_cycle --> MMSB_src_03_dag_cycle_detection_rs__has_cycle
    MMSB_src_03_dag_ShadowPageGraph_jl___dfs_has_cycle --> MMSB_src_03_dag_ShadowPageGraph_jl___dfs_has_cycle
    MMSB_src_03_dag_ShadowPageGraph_jl___dfs_has_cycle --> MMSB_src_03_dag_ShadowPageGraph_jl__has_cycle
    MMSB_src_03_dag_ShadowPageGraph_jl__has_cycle --> MMSB_src_03_dag_cycle_detection_rs__has_cycle
    MMSB_src_03_dag_ShadowPageGraph_jl__has_cycle --> MMSB_src_03_dag_ShadowPageGraph_jl___dfs_has_cycle
    MMSB_src_03_dag_ShadowPageGraph_jl__has_cycle --> MMSB_src_03_dag_ShadowPageGraph_jl__has_cycle
    MMSB_src_03_dag_ShadowPageGraph_jl__topological_sort --> MMSB_src_03_dag_ShadowPageGraph_jl___all_vertices
    MMSB_src_03_dag_ShadowPageGraph_jl__topological_sort --> MMSB_src_API_jl__length
    MMSB_src_03_dag_ShadowPageGraph_jl__topological_sort --> MMSB_src_02_semiring_DeltaRouter_jl__length
    MMSB_src_04_propagation_PropagationEngine_jl__batch_route_deltas_ --> MMSB_src_02_semiring_DeltaRouter_jl__route_delta_
    MMSB_src_04_propagation_PropagationEngine_jl__register_recompute_fn_ --> MMSB_benchmark_benchmarks_jl___page
    MMSB_src_04_propagation_PropagationEngine_jl__register_recompute_fn_ --> MMSB_src_01_types_MMSBState_jl__get_page
    MMSB_src_04_propagation_PropagationEngine_jl__queue_recomputation_ --> MMSB_src_04_propagation_PropagationEngine_jl___buffer
    MMSB_src_04_propagation_PropagationEngine_jl___aggregate_children --> MMSB_src_03_dag_DependencyGraph_jl__get_children
    MMSB_src_04_propagation_PropagationEngine_jl___aggregate_children --> MMSB_src_03_dag_ShadowPageGraph_jl__get_children
    MMSB_src_04_propagation_PropagationEngine_jl___execute_command_buffer_ --> MMSB_src_04_propagation_PropagationEngine_jl___apply_edges_
    MMSB_src_04_propagation_PropagationEngine_jl___apply_edges_ --> MMSB_src_04_propagation_PropagationEngine_jl___handle_data_dependency_
    MMSB_src_04_propagation_PropagationEngine_jl___apply_edges_ --> MMSB_src_04_propagation_PropagationEngine_jl__invalidate_compilation_
    MMSB_src_04_propagation_PropagationEngine_jl___apply_edges_ --> MMSB_src_04_propagation_PropagationEngine_jl__mark_page_stale_
    MMSB_src_04_propagation_PropagationEngine_jl___apply_edges_ --> MMSB_src_04_propagation_PropagationEngine_jl__schedule_gpu_sync_
    MMSB_src_04_propagation_PropagationEngine_jl___handle_data_dependency_ --> MMSB_src_03_dag_EventSystem_jl__emit_event_
    MMSB_src_04_propagation_PropagationEngine_jl___handle_data_dependency_ --> MMSB_src_04_propagation_PropagationEngine_jl__queue_recomputation_
    MMSB_src_04_propagation_PropagationEngine_jl___handle_data_dependency_ --> MMSB_src_04_propagation_PropagationEngine_jl__recompute_page_
    MMSB_src_04_propagation_PropagationEngine_jl___collect_descendants --> MMSB_src_03_dag_DependencyGraph_jl__get_children
    MMSB_src_04_propagation_PropagationEngine_jl___collect_descendants --> MMSB_src_03_dag_ShadowPageGraph_jl__get_children
    MMSB_src_04_propagation_PropagationEngine_jl__schedule_propagation_ --> MMSB_src_04_propagation_PropagationEngine_jl___collect_descendants
    MMSB_src_04_propagation_PropagationEngine_jl__schedule_propagation_ --> MMSB_src_04_propagation_PropagationEngine_jl__queue_recomputation_
    MMSB_src_04_propagation_PropagationEngine_jl__schedule_propagation_ --> MMSB_src_04_propagation_PropagationEngine_jl__topological_order_subset
    MMSB_src_04_propagation_PropagationEngine_jl__execute_propagation_ --> MMSB_src_04_propagation_PropagationEngine_jl___buffer
    MMSB_src_04_propagation_PropagationEngine_jl__execute_propagation_ --> MMSB_src_04_propagation_PropagationEngine_jl__recompute_page_
    MMSB_src_04_propagation_PropagationEngine_jl__recompute_page_ --> MMSB_benchmark_benchmarks_jl___page
    MMSB_src_04_propagation_PropagationEngine_jl__recompute_page_ --> MMSB_src_01_types_MMSBState_jl__get_page
    MMSB_src_04_propagation_PropagationEngine_jl__recompute_page_ --> MMSB_src_API_jl__length
    MMSB_src_04_propagation_PropagationEngine_jl__recompute_page_ --> MMSB_src_02_semiring_DeltaRouter_jl__length
    MMSB_src_04_propagation_PropagationEngine_jl__recompute_page_ --> MMSB_tests_mmsb_tests_rs__read_page
    MMSB_src_04_propagation_PropagationEngine_jl__recompute_page_ --> MMSB_benchmark_benchmarks_jl___page
    MMSB_src_04_propagation_PropagationEngine_jl__recompute_page_ --> MMSB_src_01_page_Page_jl__read_page
    MMSB_src_04_propagation_PropagationEngine_jl__mark_page_stale_ --> MMSB_src_03_dag_EventSystem_jl__emit_event_
    MMSB_src_04_propagation_PropagationEngine_jl__mark_page_stale_ --> MMSB_benchmark_benchmarks_jl___page
    MMSB_src_04_propagation_PropagationEngine_jl__mark_page_stale_ --> MMSB_src_01_types_MMSBState_jl__get_page
    MMSB_src_04_propagation_PropagationEngine_jl__schedule_gpu_sync_ --> MMSB_src_03_dag_EventSystem_jl__emit_event_
    MMSB_src_04_propagation_PropagationEngine_jl__schedule_gpu_sync_ --> MMSB_benchmark_benchmarks_jl___page
    MMSB_src_04_propagation_PropagationEngine_jl__schedule_gpu_sync_ --> MMSB_src_01_types_MMSBState_jl__get_page
    MMSB_src_04_propagation_PropagationEngine_jl__invalidate_compilation_ --> MMSB_src_03_dag_EventSystem_jl__emit_event_
    MMSB_src_04_propagation_PropagationEngine_jl__invalidate_compilation_ --> MMSB_benchmark_benchmarks_jl___page
    MMSB_src_04_propagation_PropagationEngine_jl__invalidate_compilation_ --> MMSB_src_01_types_MMSBState_jl__get_page
    MMSB_src_04_propagation_PropagationEngine_jl__topological_order_subset --> MMSB_src_03_dag_DependencyGraph_jl__get_children
    MMSB_src_04_propagation_PropagationEngine_jl__topological_order_subset --> MMSB_src_03_dag_ShadowPageGraph_jl__get_children
    MMSB_src_04_propagation_TransactionIsolation_jl__with_transaction --> MMSB_src_04_propagation_TransactionIsolation_jl__begin_transaction
    MMSB_src_04_propagation_TransactionIsolation_jl__with_transaction --> MMSB_src_04_propagation_TransactionIsolation_jl__commit_transaction
    MMSB_src_04_propagation_TransactionIsolation_jl__with_transaction --> MMSB_src_04_propagation_TransactionIsolation_jl__rollback_transaction
    MMSB_src_05_adaptive_AdaptiveLayout_jl__LayoutState --> MMSB_src_05_adaptive_AdaptiveLayout_jl__LayoutState
    MMSB_src_05_adaptive_AdaptiveLayout_jl__optimize_layout_ --> MMSB_src_05_adaptive_AdaptiveLayout_jl__compute_locality_score
    MMSB_src_05_adaptive_EntropyReduction_jl__reduce_entropy_ --> MMSB_src_05_adaptive_EntropyReduction_jl__compute_entropy
    MMSB_src_05_adaptive_EntropyReduction_jl__reduce_entropy_ --> MMSB_src_06_utility_entropy_measure_jl__compute_entropy
    MMSB_src_05_adaptive_GraphRewriting_jl__rewrite_dag_ --> MMSB_src_05_adaptive_GraphRewriting_jl__can_reorder
    MMSB_src_05_adaptive_GraphRewriting_jl__rewrite_dag_ --> MMSB_src_05_adaptive_GraphRewriting_jl__compute_edge_cost
    MMSB_src_05_adaptive_GraphRewriting_jl__rewrite_dag_ --> MMSB_src_API_jl__length
    MMSB_src_05_adaptive_GraphRewriting_jl__rewrite_dag_ --> MMSB_src_02_semiring_DeltaRouter_jl__length
    MMSB_src_06_utility_MemoryPressure_jl__LRUTracker --> MMSB_src_06_utility_MemoryPressure_jl__LRUTracker
    MMSB_src_06_utility_MemoryPressure_jl__record_access --> MMSB_src_06_utility_MemoryPressure_jl__LRUTracker
    MMSB_src_06_utility_MemoryPressure_jl__evict_lru_pages --> MMSB_src_06_utility_MemoryPressure_jl__LRUTracker
    MMSB_src_06_utility_MemoryPressure_jl__evict_lru_pages --> MMSB_src_API_jl__length
    MMSB_src_06_utility_MemoryPressure_jl__evict_lru_pages --> MMSB_src_02_semiring_DeltaRouter_jl__length
    MMSB_src_06_utility_Monitoring_jl__compute_graph_depth --> MMSB_src_06_utility_Monitoring_jl___dfs_depth
    MMSB_src_06_utility_Monitoring_jl___dfs_depth --> MMSB_src_06_utility_Monitoring_jl___dfs_depth
    MMSB_src_06_utility_Monitoring_jl__get_stats --> MMSB_src_01_page_tlog_rs__summary
    MMSB_src_06_utility_Monitoring_jl__get_stats --> MMSB_src_ffi_FFIWrapper_jl__rust_tlog_summary
    MMSB_src_06_utility_Monitoring_jl__get_stats --> MMSB_src_01_page_TLog_jl__log_summary
    MMSB_src_06_utility_Monitoring_jl__get_stats --> MMSB_src_06_utility_Monitoring_jl__compute_graph_depth
    MMSB_src_06_utility_Monitoring_jl__get_stats --> MMSB_src_API_jl__length
    MMSB_src_06_utility_Monitoring_jl__get_stats --> MMSB_src_02_semiring_DeltaRouter_jl__length
    MMSB_src_06_utility_Monitoring_jl__print_stats --> MMSB_src_06_utility_Monitoring_jl__get_stats
    MMSB_src_06_utility_cost_functions_jl__from_telemetry --> MMSB_src_06_utility_cost_functions_jl__compute_cache_cost
    MMSB_src_06_utility_cost_functions_jl__from_telemetry --> MMSB_src_06_utility_cost_functions_jl__compute_latency_cost
    MMSB_src_06_utility_cost_functions_jl__from_telemetry --> MMSB_src_06_utility_cost_functions_jl__compute_memory_cost
    MMSB_src_06_utility_entropy_measure_jl__PageDistribution --> MMSB_src_06_utility_entropy_measure_jl__PageDistribution
    MMSB_src_06_utility_utility_engine_jl__update_utility_ --> MMSB_src_06_utility_utility_engine_jl__compute_utility
    MMSB_src_06_utility_utility_engine_jl__update_utility_ --> MMSB_src_API_jl__length
    MMSB_src_06_utility_utility_engine_jl__update_utility_ --> MMSB_src_02_semiring_DeltaRouter_jl__length
    MMSB_src_06_utility_utility_engine_jl__utility_trend --> MMSB_src_API_jl__length
    MMSB_src_06_utility_utility_engine_jl__utility_trend --> MMSB_src_02_semiring_DeltaRouter_jl__length
    MMSB_src_07_intention_IntentionTypes_jl__IntentionState --> MMSB_src_07_intention_IntentionTypes_jl__IntentionState
    MMSB_src_07_intention_UpsertPlan_jl__validate_plan --> MMSB_src_API_jl__length
    MMSB_src_07_intention_UpsertPlan_jl__validate_plan --> MMSB_src_02_semiring_DeltaRouter_jl__length
    MMSB_src_07_intention_attractor_states_jl__compute_gradient --> MMSB_src_API_jl__length
    MMSB_src_07_intention_attractor_states_jl__compute_gradient --> MMSB_src_02_semiring_DeltaRouter_jl__length
    MMSB_src_07_intention_attractor_states_jl__evolve_state --> MMSB_src_07_intention_attractor_states_jl__compute_gradient
    MMSB_src_07_intention_attractor_states_jl__evolve_state --> MMSB_src_09_planning_optimization_planning_jl__compute_gradient
    MMSB_src_07_intention_goal_emergence_jl__utility_gradient --> MMSB_src_API_jl__length
    MMSB_src_07_intention_goal_emergence_jl__utility_gradient --> MMSB_src_02_semiring_DeltaRouter_jl__length
    MMSB_src_07_intention_goal_emergence_jl__detect_goals --> MMSB_src_07_intention_goal_emergence_jl__utility_gradient
    MMSB_src_07_intention_intent_lowering_jl__lower_intent_to_deltaspec --> MMSB_src_07_intention_intent_lowering_jl__mask_to_bytes
    MMSB_src_07_intention_intent_lowering_jl__lower_intent_to_deltaspec --> MMSB_src_07_intention_UpsertPlan_jl__validate_plan
    MMSB_src_07_intention_intention_engine_jl__form_intention --> MMSB_src_06_utility_utility_engine_jl__utility_trend
    MMSB_src_07_intention_intention_engine_jl__evaluate_intention --> MMSB_src_API_jl__length
    MMSB_src_07_intention_intention_engine_jl__evaluate_intention --> MMSB_src_02_semiring_DeltaRouter_jl__length
    MMSB_src_07_intention_intention_engine_jl__select_best_intention --> MMSB_src_07_intention_intention_engine_jl__evaluate_intention
    MMSB_src_07_intention_structural_preferences_jl__apply_preferences --> MMSB_src_07_intention_structural_preferences_jl__evaluate_preference
    MMSB_src_08_reasoning_ReasoningTypes_jl__ReasoningState --> MMSB_src_08_reasoning_ReasoningTypes_jl__ReasoningState
    MMSB_src_08_reasoning_constraint_propagation_jl__forward_propagate --> MMSB_src_08_reasoning_constraint_propagation_jl__propagate_constraints
    MMSB_src_08_reasoning_dependency_inference_jl__infer_dependencies --> MMSB_src_08_reasoning_dependency_inference_jl__analyze_edge_type
    MMSB_src_08_reasoning_dependency_inference_jl__infer_dependencies --> MMSB_src_08_reasoning_dependency_inference_jl__compute_dependency_strength
    MMSB_src_08_reasoning_dependency_inference_jl__compute_dependency_strength --> MMSB_src_08_reasoning_dependency_inference_jl__count_paths
    MMSB_src_08_reasoning_dependency_inference_jl__count_paths --> MMSB_src_08_reasoning_dependency_inference_jl__count_paths
    MMSB_src_08_reasoning_dependency_inference_jl__analyze_flow --> MMSB_src_08_reasoning_dependency_inference_jl__compute_critical_path
    MMSB_src_08_reasoning_logic_engine_jl__induce --> MMSB_src_API_jl__length
    MMSB_src_08_reasoning_logic_engine_jl__induce --> MMSB_src_02_semiring_DeltaRouter_jl__length
    MMSB_src_08_reasoning_pattern_formation_jl__extract_subgraphs --> MMSB_src_08_reasoning_pattern_formation_jl__grow_subgraph
    MMSB_src_08_reasoning_pattern_formation_jl__extract_subgraphs --> MMSB_src_API_jl__length
    MMSB_src_08_reasoning_pattern_formation_jl__extract_subgraphs --> MMSB_src_02_semiring_DeltaRouter_jl__length
    MMSB_src_08_reasoning_pattern_formation_jl__grow_subgraph --> MMSB_src_API_jl__length
    MMSB_src_08_reasoning_pattern_formation_jl__grow_subgraph --> MMSB_src_02_semiring_DeltaRouter_jl__length
    MMSB_src_08_reasoning_pattern_formation_jl__extract_subgraph_signature --> MMSB_src_API_jl__length
    MMSB_src_08_reasoning_pattern_formation_jl__extract_subgraph_signature --> MMSB_src_02_semiring_DeltaRouter_jl__length
    MMSB_src_08_reasoning_pattern_formation_jl__match_pattern --> MMSB_src_08_reasoning_pattern_formation_jl__extract_subgraph_signature
    MMSB_src_08_reasoning_pattern_formation_jl__match_pattern --> MMSB_src_08_reasoning_pattern_formation_jl__grow_subgraph
    MMSB_src_08_reasoning_pattern_formation_jl__match_pattern --> MMSB_src_API_jl__length
    MMSB_src_08_reasoning_pattern_formation_jl__match_pattern --> MMSB_src_02_semiring_DeltaRouter_jl__length
    MMSB_src_08_reasoning_reasoning_engine_jl__initialize_reasoning --> MMSB_src_08_reasoning_dependency_inference_jl__infer_dependencies
    MMSB_src_08_reasoning_reasoning_engine_jl__initialize_reasoning --> MMSB_src_08_reasoning_pattern_formation_jl__find_patterns
    MMSB_src_08_reasoning_reasoning_engine_jl__initialize_reasoning --> MMSB_src_08_reasoning_ReasoningTypes_jl__ReasoningState
    MMSB_src_08_reasoning_reasoning_engine_jl__initialize_reasoning --> MMSB_src_08_reasoning_rule_evaluation_jl__create_default_rules
    MMSB_src_08_reasoning_reasoning_engine_jl__reason_over_dag --> MMSB_src_08_reasoning_constraint_propagation_jl__forward_propagate
    MMSB_src_08_reasoning_reasoning_engine_jl__reason_over_dag --> MMSB_src_08_reasoning_pattern_formation_jl__match_pattern
    MMSB_src_08_reasoning_reasoning_engine_jl__reason_over_dag --> MMSB_src_08_reasoning_rule_evaluation_jl__evaluate_rules
    MMSB_src_08_reasoning_reasoning_engine_jl__reason_over_dag --> MMSB_src_08_reasoning_structural_inference_jl__derive_constraints
    MMSB_src_08_reasoning_reasoning_engine_jl__perform_inference --> MMSB_src_08_reasoning_logic_engine_jl__deduce
    MMSB_src_08_reasoning_rule_evaluation_jl__create_default_rules --> MMSB_src_API_jl__length
    MMSB_src_08_reasoning_rule_evaluation_jl__create_default_rules --> MMSB_src_02_semiring_DeltaRouter_jl__length
    MMSB_src_08_reasoning_structural_inference_jl__infer_from_structure --> MMSB_src_API_jl__length
    MMSB_src_08_reasoning_structural_inference_jl__infer_from_structure --> MMSB_src_02_semiring_DeltaRouter_jl__length
    MMSB_src_08_reasoning_structural_inference_jl__derive_constraints --> MMSB_src_08_reasoning_structural_inference_jl__infer_from_structure
    MMSB_src_08_reasoning_structural_inference_jl__check_consistency --> MMSB_src_API_jl__length
    MMSB_src_08_reasoning_structural_inference_jl__check_consistency --> MMSB_src_02_semiring_DeltaRouter_jl__length
    MMSB_src_09_planning_PlanningTypes_jl__PlanningState --> MMSB_src_09_planning_PlanningTypes_jl__PlanningState
    MMSB_src_09_planning_decision_graphs_jl__build_decision_graph --> MMSB_src_09_planning_decision_graphs_jl__expand_graph_
    MMSB_src_09_planning_decision_graphs_jl__expand_graph_ --> MMSB_src_09_planning_decision_graphs_jl__apply_action_simple
    MMSB_src_09_planning_decision_graphs_jl__expand_graph_ --> MMSB_src_09_planning_decision_graphs_jl__expand_graph_
    MMSB_src_09_planning_decision_graphs_jl__find_optimal_path --> MMSB_src_API_jl__length
    MMSB_src_09_planning_decision_graphs_jl__find_optimal_path --> MMSB_src_02_semiring_DeltaRouter_jl__length
    MMSB_src_09_planning_goal_decomposition_jl__order_subgoals --> MMSB_src_09_planning_goal_decomposition_jl__score_subgoal
    MMSB_src_09_planning_goal_decomposition_jl__score_subgoal --> MMSB_src_09_planning_goal_decomposition_jl__estimate_achievability
    MMSB_src_09_planning_optimization_planning_jl__optimize_plan --> MMSB_src_07_intention_attractor_states_jl__compute_gradient
    MMSB_src_09_planning_optimization_planning_jl__optimize_plan --> MMSB_src_09_planning_optimization_planning_jl__compute_gradient
    MMSB_src_09_planning_optimization_planning_jl__optimize_plan --> MMSB_src_09_planning_optimization_planning_jl__extract_parameters
    MMSB_src_09_planning_optimization_planning_jl__optimize_plan --> MMSB_src_09_planning_optimization_planning_jl__norm
    MMSB_src_09_planning_optimization_planning_jl__optimize_plan --> MMSB_src_09_planning_optimization_planning_jl__reconstruct_plan
    MMSB_src_09_planning_optimization_planning_jl__optimize_plan --> MMSB_src_09_planning_search_algorithms_jl__reconstruct_plan
    MMSB_src_09_planning_optimization_planning_jl__compute_gradient --> MMSB_src_API_jl__length
    MMSB_src_09_planning_optimization_planning_jl__compute_gradient --> MMSB_src_02_semiring_DeltaRouter_jl__length
    MMSB_src_09_planning_optimization_planning_jl__compute_sequence_gradient --> MMSB_src_09_planning_optimization_planning_jl__evaluate_action_sequence
    MMSB_src_09_planning_optimization_planning_jl__prepare_for_enzyme --> MMSB_src_API_jl__length
    MMSB_src_09_planning_optimization_planning_jl__prepare_for_enzyme --> MMSB_src_02_semiring_DeltaRouter_jl__length
    MMSB_src_09_planning_planning_engine_jl__create_plan --> MMSB_src_09_planning_optimization_planning_jl__optimize_plan
    MMSB_src_09_planning_planning_engine_jl__create_plan --> MMSB_src_09_planning_strategy_generation_jl__generate_strategies
    MMSB_src_09_planning_planning_engine_jl__create_plan --> MMSB_src_09_planning_strategy_generation_jl__select_strategy
    MMSB_src_09_planning_planning_engine_jl__execute_planning --> MMSB_src_09_planning_goal_decomposition_jl__decompose_goal
    MMSB_src_09_planning_planning_engine_jl__execute_planning --> MMSB_src_09_planning_rollout_simulation_jl__simulate_plan
    MMSB_src_09_planning_planning_engine_jl__execute_planning --> MMSB_src_09_planning_search_algorithms_jl__mcts_search
    MMSB_src_09_planning_planning_engine_jl__execute_planning --> MMSB_src_09_planning_planning_engine_jl__create_plan
    MMSB_src_09_planning_planning_engine_jl__execute_planning --> MMSB_src_API_jl__length
    MMSB_src_09_planning_planning_engine_jl__execute_planning --> MMSB_src_02_semiring_DeltaRouter_jl__length
    MMSB_src_09_planning_planning_engine_jl__replan --> MMSB_src_09_planning_rollout_simulation_jl__simulate_plan
    MMSB_src_09_planning_planning_engine_jl__replan --> MMSB_src_09_planning_search_algorithms_jl__astar_search
    MMSB_src_09_planning_planning_engine_jl__replan --> MMSB_src_09_planning_strategy_generation_jl__generate_strategies
    MMSB_src_09_planning_rl_planning_jl__evaluate_policy --> MMSB_src_09_planning_rl_planning_jl__expected_next_value
    MMSB_src_09_planning_rl_planning_jl__evaluate_policy --> MMSB_src_09_planning_rl_planning_jl__immediate_reward
    MMSB_src_09_planning_rollout_simulation_jl__simulate_plan --> MMSB_src_09_planning_search_algorithms_jl__apply_action
    MMSB_src_09_planning_rollout_simulation_jl__simulate_plan --> MMSB_src_09_planning_search_algorithms_jl__can_apply
    MMSB_src_09_planning_rollout_simulation_jl__evaluate_outcome --> MMSB_src_API_jl__length
    MMSB_src_09_planning_rollout_simulation_jl__evaluate_outcome --> MMSB_src_02_semiring_DeltaRouter_jl__length
    MMSB_src_09_planning_search_algorithms_jl__MCTSNode --> MMSB_src_09_planning_search_algorithms_jl__MCTSNode
    MMSB_src_09_planning_search_algorithms_jl__select_node --> MMSB_src_09_planning_search_algorithms_jl__best_uct_child
    MMSB_src_09_planning_search_algorithms_jl__expand_node --> MMSB_src_09_planning_search_algorithms_jl__MCTSNode
    MMSB_src_09_planning_search_algorithms_jl__expand_node --> MMSB_src_09_planning_search_algorithms_jl__apply_action
    MMSB_src_09_planning_search_algorithms_jl__expand_node --> MMSB_src_09_planning_search_algorithms_jl__can_apply
    MMSB_src_09_planning_search_algorithms_jl__simulate --> MMSB_src_09_planning_search_algorithms_jl__apply_action
    MMSB_src_09_planning_search_algorithms_jl__simulate --> MMSB_src_09_planning_search_algorithms_jl__can_apply
    MMSB_src_09_planning_search_algorithms_jl__extract_plan_from_mcts --> MMSB_src_09_planning_search_algorithms_jl__best_uct_child
    MMSB_src_09_planning_strategy_generation_jl__generate_strategies --> MMSB_src_09_planning_search_algorithms_jl__astar_search
    MMSB_src_09_planning_strategy_generation_jl__generate_strategies --> MMSB_src_09_planning_search_algorithms_jl__mcts_search
    MMSB_src_09_planning_strategy_generation_jl__generate_strategies --> MMSB_src_09_planning_strategy_generation_jl__hierarchical_planning
    MMSB_src_09_planning_strategy_generation_jl__hierarchical_planning --> MMSB_src_09_planning_goal_decomposition_jl__decompose_goal
    MMSB_src_09_planning_strategy_generation_jl__hierarchical_planning --> MMSB_src_09_planning_search_algorithms_jl__apply_action
    MMSB_src_09_planning_strategy_generation_jl__hierarchical_planning --> MMSB_src_09_planning_search_algorithms_jl__astar_search
    MMSB_src_10_agent_interface_BaseHook_jl__enable_base_hooks_ --> MMSB_src_10_agent_interface_BaseHook_jl__hook_invoke
    MMSB_src_10_agent_interface_CompilerHooks_jl__Core_Compiler_typeinf --> MMSB_src_10_agent_interface_CompilerHooks_jl__create_inference_pages_
    MMSB_src_10_agent_interface_CompilerHooks_jl__Core_Compiler_typeinf --> MMSB_src_10_agent_interface_CompilerHooks_jl__log_inference_result_
    MMSB_src_10_agent_interface_CompilerHooks_jl__Core_Compiler_typeinf --> MMSB_src_10_agent_interface_CompilerHooks_jl__log_inference_start_
    MMSB_src_10_agent_interface_CompilerHooks_jl__create_inference_pages_ --> MMSB_src_00_physical_PageAllocator_jl__create_page_
    MMSB_src_10_agent_interface_CompilerHooks_jl__create_inference_pages_ --> MMSB_src_API_jl__length
    MMSB_src_10_agent_interface_CompilerHooks_jl__create_inference_pages_ --> MMSB_src_02_semiring_DeltaRouter_jl__length
    MMSB_src_10_agent_interface_InstrumentationManager_jl__enable_instrumentation_ --> MMSB_src_10_agent_interface_BaseHook_jl__enable_base_hooks_
    MMSB_src_10_agent_interface_InstrumentationManager_jl__enable_instrumentation_ --> MMSB_src_10_agent_interface_CompilerHooks_jl__enable_compiler_hooks_
    MMSB_src_10_agent_interface_InstrumentationManager_jl__enable_instrumentation_ --> MMSB_src_10_agent_interface_CoreHooks_jl__enable_core_hooks_
    MMSB_src_10_agent_interface_InstrumentationManager_jl__disable_instrumentation_ --> MMSB_src_10_agent_interface_BaseHook_jl__disable_base_hooks_
    MMSB_src_10_agent_interface_InstrumentationManager_jl__disable_instrumentation_ --> MMSB_src_10_agent_interface_CompilerHooks_jl__disable_compiler_hooks_
    MMSB_src_10_agent_interface_InstrumentationManager_jl__disable_instrumentation_ --> MMSB_src_10_agent_interface_CoreHooks_jl__disable_core_hooks_
    MMSB_src_10_agent_interface_InstrumentationManager_jl__configure_instrumentation_ --> MMSB_src_10_agent_interface_InstrumentationManager_jl__disable_instrumentation_
    MMSB_src_10_agent_interface_InstrumentationManager_jl__configure_instrumentation_ --> MMSB_src_10_agent_interface_InstrumentationManager_jl__enable_instrumentation_
    MMSB_src_10_agent_interface_checkpoint_api_jl__create_checkpoint --> MMSB_src_01_page_TLog_jl__checkpoint_log_
    MMSB_src_10_agent_interface_checkpoint_api_jl__restore_checkpoint --> MMSB_src_01_page_TLog_jl__load_checkpoint_
    MMSB_src_11_agents_AgentTypes_jl__push_memory_ --> MMSB_src_API_jl__length
    MMSB_src_11_agents_AgentTypes_jl__push_memory_ --> MMSB_src_02_semiring_DeltaRouter_jl__length
    MMSB_src_11_agents_enzyme_integration_jl__autodiff_loss --> MMSB_src_API_jl__length
    MMSB_src_11_agents_enzyme_integration_jl__autodiff_loss --> MMSB_src_02_semiring_DeltaRouter_jl__length
    MMSB_src_11_agents_hybrid_agent_jl__observe --> MMSB_src_10_agent_interface_AgentProtocol_jl__observe
    MMSB_src_11_agents_hybrid_agent_jl__observe --> MMSB_src_11_agents_hybrid_agent_jl__observe
    MMSB_src_11_agents_hybrid_agent_jl__observe --> MMSB_src_11_agents_planning_agent_jl__observe
    MMSB_src_11_agents_hybrid_agent_jl__observe --> MMSB_src_11_agents_rl_agent_jl__observe
    MMSB_src_11_agents_hybrid_agent_jl__observe --> MMSB_src_11_agents_symbolic_agent_jl__observe
    MMSB_src_11_agents_hybrid_agent_jl__symbolic_step_ --> MMSB_src_08_reasoning_rule_evaluation_jl__apply_rule
    MMSB_src_11_agents_hybrid_agent_jl__symbolic_step_ --> MMSB_src_11_agents_symbolic_agent_jl__apply_rule
    MMSB_src_11_agents_hybrid_agent_jl__neural_step_ --> MMSB_src_11_agents_rl_agent_jl__train_step_
    MMSB_src_11_agents_rl_agent_jl__observe --> MMSB_src_API_jl__length
    MMSB_src_11_agents_rl_agent_jl__observe --> MMSB_src_02_semiring_DeltaRouter_jl__length
    MMSB_src_11_agents_rl_agent_jl__train_step_ --> MMSB_src_11_agents_rl_agent_jl__compute_reward
    MMSB_src_11_agents_rl_agent_jl__train_step_ --> MMSB_src_10_agent_interface_AgentProtocol_jl__observe
    MMSB_src_11_agents_rl_agent_jl__train_step_ --> MMSB_src_11_agents_hybrid_agent_jl__observe
    MMSB_src_11_agents_rl_agent_jl__train_step_ --> MMSB_src_11_agents_planning_agent_jl__observe
    MMSB_src_11_agents_rl_agent_jl__train_step_ --> MMSB_src_11_agents_rl_agent_jl__observe
    MMSB_src_11_agents_rl_agent_jl__train_step_ --> MMSB_src_11_agents_symbolic_agent_jl__observe
    MMSB_src_11_agents_rl_agent_jl__train_step_ --> MMSB_src_11_agents_AgentTypes_jl__push_memory_
    MMSB_src_11_agents_symbolic_agent_jl__SymbolicAgent --> MMSB_src_11_agents_symbolic_agent_jl__SymbolicAgent
    MMSB_src_12_applications_multi_agent_system_jl__coordinate_step_ --> MMSB_src_10_agent_interface_AgentProtocol_jl__observe
    MMSB_src_12_applications_multi_agent_system_jl__coordinate_step_ --> MMSB_src_11_agents_hybrid_agent_jl__observe
    MMSB_src_12_applications_multi_agent_system_jl__coordinate_step_ --> MMSB_src_11_agents_planning_agent_jl__observe
    MMSB_src_12_applications_multi_agent_system_jl__coordinate_step_ --> MMSB_src_11_agents_rl_agent_jl__observe
    MMSB_src_12_applications_multi_agent_system_jl__coordinate_step_ --> MMSB_src_11_agents_symbolic_agent_jl__observe
    MMSB_src_12_applications_world_simulation_jl__add_entity_ --> MMSB_src_01_types_MMSBState_jl__allocate_page_id_
```
