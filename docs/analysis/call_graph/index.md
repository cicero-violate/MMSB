# Call Graph Analysis

This document shows the **interprocedural call graph** - which functions call which other functions.

> **Note:** This is NOT a control flow graph (CFG). CFG shows intraprocedural control flow (branches, loops) within individual functions.

## Call Graph Statistics

- Total functions: 123
- Total function calls: 89
- Maximum call depth: 3
- Leaf functions (no outgoing calls): 74

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
    MMSB_src_ffi_rs__mmsb_checkpoint_write --> MMSB_src_ffi_rs__set_last_error
    MMSB_src_ffi_rs__mmsb_checkpoint_load --> MMSB_src_ffi_rs__set_last_error
    MMSB_src_ffi_rs__mmsb_checkpoint_load --> MMSB_src_ffi_rs__set_last_error
    MMSB_src_ffi_rs__mmsb_checkpoint_load --> MMSB_src_01_page_checkpoint_rs__load_checkpoint
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
    MMSB_tests_mmsb_tests_rs__test_checkpoint_log_and_restore --> MMSB_src_01_page_checkpoint_rs__write_checkpoint
    MMSB_tests_mmsb_tests_rs__test_checkpoint_log_and_restore --> MMSB_src_01_page_checkpoint_rs__load_checkpoint
    MMSB_tests_week27_31_integration_rs__test_delta_merge_simd --> MMSB_src_01_page_delta_merge_rs__merge_deltas
    MMSB_src_01_page_delta_merge_rs__merge_dense_simd --> MMSB_src_01_page_delta_merge_rs__merge_dense_avx512
    MMSB_src_01_page_delta_merge_rs__merge_dense_simd --> MMSB_src_01_page_delta_merge_rs__merge_dense_avx2
    MMSB_src_01_page_tlog_rs__summary --> MMSB_src_01_page_tlog_rs__validate_header
    MMSB_src_01_page_tlog_rs__summary --> MMSB_src_01_page_tlog_rs__read_frame
    MMSB_src_01_page_tlog_compression_rs__compress_delta_mask --> MMSB_src_01_page_tlog_compression_rs__encode_rle
    MMSB_src_01_page_tlog_compression_rs__compress_delta_mask --> MMSB_src_01_page_tlog_compression_rs__bitpack_mask
    MMSB_src_03_dag_cycle_detection_rs__has_cycle --> MMSB_src_03_dag_cycle_detection_rs__dfs
    MMSB_src_03_dag_cycle_detection_rs__has_cycle --> MMSB_src_03_dag_cycle_detection_rs__dfs
    MMSB_src_03_dag_cycle_detection_rs__dfs --> MMSB_src_03_dag_cycle_detection_rs__dfs
```
