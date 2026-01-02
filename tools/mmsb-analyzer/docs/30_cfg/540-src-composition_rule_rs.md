# CFG Group: src/composition_rule.rs

## Function: `check_conflicts`

- File: src/composition_rule.rs
- Branches: 0
- Loops: 0
- Nodes: 11
- Edges: 10

```mermaid
flowchart TD
    check_conflicts_0["ENTRY"]
    check_conflicts_1["for file in & signature . writes . files { if let Some (& prior_index) = stat..."]
    check_conflicts_2["for module_write in & signature . writes . modules { if let Some (& prior_ind..."]
    check_conflicts_3["for import_write in & signature . writes . imports { let key = (import_write ..."]
    check_conflicts_4["for re_export_write in & signature . writes . re_exports { let key = (re_expo..."]
    check_conflicts_5["for visibility_write in & signature . writes . visibility_modifiers { let key..."]
    check_conflicts_6["for read_path in & signature . reads . paths { if let Some (& prior_index) = ..."]
    check_conflicts_7["let current_invariants = collect_invariants_touched (& signature . invariant_touchpoints)"]
    check_conflicts_8["for invariant in current_invariants { if let Some (prior_indices) = state . i..."]
    check_conflicts_9["None"]
    check_conflicts_10["EXIT"]
    check_conflicts_0 --> check_conflicts_1
    check_conflicts_1 --> check_conflicts_2
    check_conflicts_2 --> check_conflicts_3
    check_conflicts_3 --> check_conflicts_4
    check_conflicts_4 --> check_conflicts_5
    check_conflicts_5 --> check_conflicts_6
    check_conflicts_6 --> check_conflicts_7
    check_conflicts_7 --> check_conflicts_8
    check_conflicts_8 --> check_conflicts_9
    check_conflicts_9 --> check_conflicts_10
```

## Function: `collect_invariants_touched`

- File: src/composition_rule.rs
- Branches: 5
- Loops: 0
- Nodes: 29
- Edges: 33

```mermaid
flowchart TD
    collect_invariants_touched_0["ENTRY"]
    collect_invariants_touched_1["let mut invariants = Vec :: new ()"]
    collect_invariants_touched_2["if touchpoints . i1_module_coherence"]
    collect_invariants_touched_3["THEN BB"]
    collect_invariants_touched_4["invariants . push (InvariantType :: I1ModuleCoherence)"]
    collect_invariants_touched_5["EMPTY ELSE"]
    collect_invariants_touched_6["IF JOIN"]
    collect_invariants_touched_7["if touchpoints . i2_dependency_direction"]
    collect_invariants_touched_8["THEN BB"]
    collect_invariants_touched_9["invariants . push (InvariantType :: I2DependencyDirection)"]
    collect_invariants_touched_10["EMPTY ELSE"]
    collect_invariants_touched_11["IF JOIN"]
    collect_invariants_touched_12["if touchpoints . visibility_law"]
    collect_invariants_touched_13["THEN BB"]
    collect_invariants_touched_14["invariants . push (InvariantType :: VisibilityLaw)"]
    collect_invariants_touched_15["EMPTY ELSE"]
    collect_invariants_touched_16["IF JOIN"]
    collect_invariants_touched_17["if touchpoints . re_export_law"]
    collect_invariants_touched_18["THEN BB"]
    collect_invariants_touched_19["invariants . push (InvariantType :: ReExportLaw)"]
    collect_invariants_touched_20["EMPTY ELSE"]
    collect_invariants_touched_21["IF JOIN"]
    collect_invariants_touched_22["if touchpoints . test_topology_law"]
    collect_invariants_touched_23["THEN BB"]
    collect_invariants_touched_24["invariants . push (InvariantType :: TestTopologyLaw)"]
    collect_invariants_touched_25["EMPTY ELSE"]
    collect_invariants_touched_26["IF JOIN"]
    collect_invariants_touched_27["invariants"]
    collect_invariants_touched_28["EXIT"]
    collect_invariants_touched_0 --> collect_invariants_touched_1
    collect_invariants_touched_1 --> collect_invariants_touched_2
    collect_invariants_touched_2 --> collect_invariants_touched_3
    collect_invariants_touched_3 --> collect_invariants_touched_4
    collect_invariants_touched_2 --> collect_invariants_touched_5
    collect_invariants_touched_4 --> collect_invariants_touched_6
    collect_invariants_touched_5 --> collect_invariants_touched_6
    collect_invariants_touched_6 --> collect_invariants_touched_7
    collect_invariants_touched_7 --> collect_invariants_touched_8
    collect_invariants_touched_8 --> collect_invariants_touched_9
    collect_invariants_touched_7 --> collect_invariants_touched_10
    collect_invariants_touched_9 --> collect_invariants_touched_11
    collect_invariants_touched_10 --> collect_invariants_touched_11
    collect_invariants_touched_11 --> collect_invariants_touched_12
    collect_invariants_touched_12 --> collect_invariants_touched_13
    collect_invariants_touched_13 --> collect_invariants_touched_14
    collect_invariants_touched_12 --> collect_invariants_touched_15
    collect_invariants_touched_14 --> collect_invariants_touched_16
    collect_invariants_touched_15 --> collect_invariants_touched_16
    collect_invariants_touched_16 --> collect_invariants_touched_17
    collect_invariants_touched_17 --> collect_invariants_touched_18
    collect_invariants_touched_18 --> collect_invariants_touched_19
    collect_invariants_touched_17 --> collect_invariants_touched_20
    collect_invariants_touched_19 --> collect_invariants_touched_21
    collect_invariants_touched_20 --> collect_invariants_touched_21
    collect_invariants_touched_21 --> collect_invariants_touched_22
    collect_invariants_touched_22 --> collect_invariants_touched_23
    collect_invariants_touched_23 --> collect_invariants_touched_24
    collect_invariants_touched_22 --> collect_invariants_touched_25
    collect_invariants_touched_24 --> collect_invariants_touched_26
    collect_invariants_touched_25 --> collect_invariants_touched_26
    collect_invariants_touched_26 --> collect_invariants_touched_27
    collect_invariants_touched_27 --> collect_invariants_touched_28
```

## Function: `compose_batch`

- File: src/composition_rule.rs
- Branches: 0
- Loops: 0
- Nodes: 5
- Edges: 4

```mermaid
flowchart TD
    compose_batch_0["ENTRY"]
    compose_batch_1["let mut state = ComposedEffectState :: empty ()"]
    compose_batch_2["for (index , signature) in batch . iter () . enumerate () { if let Some (conf..."]
    compose_batch_3["CompositionResult :: Admissible { action_count : batch . len () , final_state..."]
    compose_batch_4["EXIT"]
    compose_batch_0 --> compose_batch_1
    compose_batch_1 --> compose_batch_2
    compose_batch_2 --> compose_batch_3
    compose_batch_3 --> compose_batch_4
```

## Function: `compose_into_state`

- File: src/composition_rule.rs
- Branches: 0
- Loops: 0
- Nodes: 16
- Edges: 15

```mermaid
flowchart TD
    compose_into_state_0["ENTRY"]
    compose_into_state_1["for file in & signature . writes . files { state . files_written . insert (fi..."]
    compose_into_state_2["for module_write in & signature . writes . modules { state . modules_written ..."]
    compose_into_state_3["for import_write in & signature . writes . imports { let key = (import_write ..."]
    compose_into_state_4["for re_export_write in & signature . writes . re_exports { let key = (re_expo..."]
    compose_into_state_5["for visibility_write in & signature . writes . visibility_modifiers { let key..."]
    compose_into_state_6["for path in & signature . reads . paths { state . files_read . insert (path ...."]
    compose_into_state_7["for symbol in & signature . reads . symbols { state . symbols_read . insert (..."]
    compose_into_state_8["let invariants = collect_invariants_touched (& signature . invariant_touchpoints)"]
    compose_into_state_9["for invariant in invariants { state . invariants_touched . entry (invariant) ..."]
    compose_into_state_10["state . executor_surfaces . requires_import_repair |= signature . executor_su..."]
    compose_into_state_11["state . executor_surfaces . requires_module_shim |= signature . executor_surf..."]
    compose_into_state_12["state . executor_surfaces . requires_re_export_enforcement |= signature . exe..."]
    compose_into_state_13["state . executor_surfaces . requires_verification_gate |= signature . executo..."]
    compose_into_state_14["state . action_count += 1"]
    compose_into_state_15["EXIT"]
    compose_into_state_0 --> compose_into_state_1
    compose_into_state_1 --> compose_into_state_2
    compose_into_state_2 --> compose_into_state_3
    compose_into_state_3 --> compose_into_state_4
    compose_into_state_4 --> compose_into_state_5
    compose_into_state_5 --> compose_into_state_6
    compose_into_state_6 --> compose_into_state_7
    compose_into_state_7 --> compose_into_state_8
    compose_into_state_8 --> compose_into_state_9
    compose_into_state_9 --> compose_into_state_10
    compose_into_state_10 --> compose_into_state_11
    compose_into_state_11 --> compose_into_state_12
    compose_into_state_12 --> compose_into_state_13
    compose_into_state_13 --> compose_into_state_14
    compose_into_state_14 --> compose_into_state_15
```

## Function: `create_test_signature`

- File: src/composition_rule.rs
- Branches: 0
- Loops: 0
- Nodes: 3
- Edges: 2

```mermaid
flowchart TD
    create_test_signature_0["ENTRY"]
    create_test_signature_1["EffectSignature { schema_version : SCHEMA_VERSION . to_string () , action_typ..."]
    create_test_signature_2["EXIT"]
    create_test_signature_0 --> create_test_signature_1
    create_test_signature_1 --> create_test_signature_2
```

## Function: `test_abort_happens_at_first_conflict`

- File: src/composition_rule.rs
- Branches: 0
- Loops: 0
- Nodes: 7
- Edges: 6

```mermaid
flowchart TD
    test_abort_happens_at_first_conflict_0["ENTRY"]
    test_abort_happens_at_first_conflict_1["let sig_a = create_test_signature ('action_a' , vec ! [PathBuf :: from ('file1.rs')] , In..."]
    test_abort_happens_at_first_conflict_2["let sig_b = create_test_signature ('action_b' , vec ! [PathBuf :: from ('file1.rs')] , In..."]
    test_abort_happens_at_first_conflict_3["let sig_c = create_test_signature ('action_c' , vec ! [PathBuf :: from ('file1.rs')] , In..."]
    test_abort_happens_at_first_conflict_4["let result = compose_batch (& [sig_a , sig_b , sig_c])"]
    test_abort_happens_at_first_conflict_5["match result { CompositionResult :: Inadmissible { first_failure_index , fail..."]
    test_abort_happens_at_first_conflict_6["EXIT"]
    test_abort_happens_at_first_conflict_0 --> test_abort_happens_at_first_conflict_1
    test_abort_happens_at_first_conflict_1 --> test_abort_happens_at_first_conflict_2
    test_abort_happens_at_first_conflict_2 --> test_abort_happens_at_first_conflict_3
    test_abort_happens_at_first_conflict_3 --> test_abort_happens_at_first_conflict_4
    test_abort_happens_at_first_conflict_4 --> test_abort_happens_at_first_conflict_5
    test_abort_happens_at_first_conflict_5 --> test_abort_happens_at_first_conflict_6
```

## Function: `test_empty_batch_is_admissible`

- File: src/composition_rule.rs
- Branches: 0
- Loops: 0
- Nodes: 4
- Edges: 3

```mermaid
flowchart TD
    test_empty_batch_is_admissible_0["ENTRY"]
    test_empty_batch_is_admissible_1["let result = compose_batch (& [])"]
    test_empty_batch_is_admissible_2["match result { CompositionResult :: Admissible { action_count , .. } => { ass..."]
    test_empty_batch_is_admissible_3["EXIT"]
    test_empty_batch_is_admissible_0 --> test_empty_batch_is_admissible_1
    test_empty_batch_is_admissible_1 --> test_empty_batch_is_admissible_2
    test_empty_batch_is_admissible_2 --> test_empty_batch_is_admissible_3
```

## Function: `test_executor_surface_accumulation`

- File: src/composition_rule.rs
- Branches: 0
- Loops: 0
- Nodes: 8
- Edges: 7

```mermaid
flowchart TD
    test_executor_surface_accumulation_0["ENTRY"]
    test_executor_surface_accumulation_1["let mut sig1 = create_test_signature ('action1' , vec ! [PathBuf :: from ('file1.rs')] , Inv..."]
    test_executor_surface_accumulation_2["sig1 . executor_surface . requires_import_repair = true"]
    test_executor_surface_accumulation_3["let mut sig2 = create_test_signature ('action2' , vec ! [PathBuf :: from ('file2.rs')] , Inv..."]
    test_executor_surface_accumulation_4["sig2 . executor_surface . requires_verification_gate = true"]
    test_executor_surface_accumulation_5["let result = compose_batch (& [sig1 , sig2])"]
    test_executor_surface_accumulation_6["match result { CompositionResult :: Admissible { final_state , .. } => { asse..."]
    test_executor_surface_accumulation_7["EXIT"]
    test_executor_surface_accumulation_0 --> test_executor_surface_accumulation_1
    test_executor_surface_accumulation_1 --> test_executor_surface_accumulation_2
    test_executor_surface_accumulation_2 --> test_executor_surface_accumulation_3
    test_executor_surface_accumulation_3 --> test_executor_surface_accumulation_4
    test_executor_surface_accumulation_4 --> test_executor_surface_accumulation_5
    test_executor_surface_accumulation_5 --> test_executor_surface_accumulation_6
    test_executor_surface_accumulation_6 --> test_executor_surface_accumulation_7
```

## Function: `test_file_write_conflict_aborts`

- File: src/composition_rule.rs
- Branches: 0
- Loops: 0
- Nodes: 6
- Edges: 5

```mermaid
flowchart TD
    test_file_write_conflict_aborts_0["ENTRY"]
    test_file_write_conflict_aborts_1["let sig1 = create_test_signature ('action1' , vec ! [PathBuf :: from ('file1.rs')] , Inv..."]
    test_file_write_conflict_aborts_2["let sig2 = create_test_signature ('action2' , vec ! [PathBuf :: from ('file1.rs')] , Inv..."]
    test_file_write_conflict_aborts_3["let result = compose_batch (& [sig1 , sig2])"]
    test_file_write_conflict_aborts_4["match result { CompositionResult :: Inadmissible { first_failure_index , fail..."]
    test_file_write_conflict_aborts_5["EXIT"]
    test_file_write_conflict_aborts_0 --> test_file_write_conflict_aborts_1
    test_file_write_conflict_aborts_1 --> test_file_write_conflict_aborts_2
    test_file_write_conflict_aborts_2 --> test_file_write_conflict_aborts_3
    test_file_write_conflict_aborts_3 --> test_file_write_conflict_aborts_4
    test_file_write_conflict_aborts_4 --> test_file_write_conflict_aborts_5
```

## Function: `test_invariant_overlap_aborts`

- File: src/composition_rule.rs
- Branches: 0
- Loops: 0
- Nodes: 6
- Edges: 5

```mermaid
flowchart TD
    test_invariant_overlap_aborts_0["ENTRY"]
    test_invariant_overlap_aborts_1["let sig1 = create_test_signature ('action1' , vec ! [PathBuf :: from ('file1.rs')] , Inv..."]
    test_invariant_overlap_aborts_2["let sig2 = create_test_signature ('action2' , vec ! [PathBuf :: from ('file2.rs')] , Inv..."]
    test_invariant_overlap_aborts_3["let result = compose_batch (& [sig1 , sig2])"]
    test_invariant_overlap_aborts_4["match result { CompositionResult :: Inadmissible { first_failure_index , conf..."]
    test_invariant_overlap_aborts_5["EXIT"]
    test_invariant_overlap_aborts_0 --> test_invariant_overlap_aborts_1
    test_invariant_overlap_aborts_1 --> test_invariant_overlap_aborts_2
    test_invariant_overlap_aborts_2 --> test_invariant_overlap_aborts_3
    test_invariant_overlap_aborts_3 --> test_invariant_overlap_aborts_4
    test_invariant_overlap_aborts_4 --> test_invariant_overlap_aborts_5
```

## Function: `test_non_overlapping_actions_are_admissible`

- File: src/composition_rule.rs
- Branches: 0
- Loops: 0
- Nodes: 6
- Edges: 5

```mermaid
flowchart TD
    test_non_overlapping_actions_are_admissible_0["ENTRY"]
    test_non_overlapping_actions_are_admissible_1["let sig1 = create_test_signature ('action1' , vec ! [PathBuf :: from ('file1.rs')] , Inv..."]
    test_non_overlapping_actions_are_admissible_2["let sig2 = create_test_signature ('action2' , vec ! [PathBuf :: from ('file2.rs')] , Inv..."]
    test_non_overlapping_actions_are_admissible_3["let result = compose_batch (& [sig1 , sig2])"]
    test_non_overlapping_actions_are_admissible_4["match result { CompositionResult :: Admissible { action_count , .. } => { ass..."]
    test_non_overlapping_actions_are_admissible_5["EXIT"]
    test_non_overlapping_actions_are_admissible_0 --> test_non_overlapping_actions_are_admissible_1
    test_non_overlapping_actions_are_admissible_1 --> test_non_overlapping_actions_are_admissible_2
    test_non_overlapping_actions_are_admissible_2 --> test_non_overlapping_actions_are_admissible_3
    test_non_overlapping_actions_are_admissible_3 --> test_non_overlapping_actions_are_admissible_4
    test_non_overlapping_actions_are_admissible_4 --> test_non_overlapping_actions_are_admissible_5
```

## Function: `test_order_sensitivity`

- File: src/composition_rule.rs
- Branches: 0
- Loops: 0
- Nodes: 10
- Edges: 9

```mermaid
flowchart TD
    test_order_sensitivity_0["ENTRY"]
    test_order_sensitivity_1["let file_path = PathBuf :: from ('file1.rs')"]
    test_order_sensitivity_2["let mut sig_read = create_test_signature ('reader' , vec ! [] , InvariantTouchpoints { i1_module..."]
    test_order_sensitivity_3["sig_read . reads . paths . insert (file_path . clone ())"]
    test_order_sensitivity_4["let sig_write = create_test_signature ('writer' , vec ! [file_path . clone ()] , InvariantTou..."]
    test_order_sensitivity_5["let result1 = compose_batch (& [sig_read . clone () , sig_write . clone ()])"]
    test_order_sensitivity_6["macro assert"]
    test_order_sensitivity_7["let result2 = compose_batch (& [sig_write , sig_read])"]
    test_order_sensitivity_8["macro assert"]
    test_order_sensitivity_9["EXIT"]
    test_order_sensitivity_0 --> test_order_sensitivity_1
    test_order_sensitivity_1 --> test_order_sensitivity_2
    test_order_sensitivity_2 --> test_order_sensitivity_3
    test_order_sensitivity_3 --> test_order_sensitivity_4
    test_order_sensitivity_4 --> test_order_sensitivity_5
    test_order_sensitivity_5 --> test_order_sensitivity_6
    test_order_sensitivity_6 --> test_order_sensitivity_7
    test_order_sensitivity_7 --> test_order_sensitivity_8
    test_order_sensitivity_8 --> test_order_sensitivity_9
```

## Function: `test_read_after_write_aborts`

- File: src/composition_rule.rs
- Branches: 0
- Loops: 0
- Nodes: 8
- Edges: 7

```mermaid
flowchart TD
    test_read_after_write_aborts_0["ENTRY"]
    test_read_after_write_aborts_1["let file_path = PathBuf :: from ('file1.rs')"]
    test_read_after_write_aborts_2["let sig1 = create_test_signature ('action1' , vec ! [file_path . clone ()] , InvariantTo..."]
    test_read_after_write_aborts_3["let mut sig2 = create_test_signature ('action2' , vec ! [PathBuf :: from ('file2.rs')] , Inv..."]
    test_read_after_write_aborts_4["sig2 . reads . paths . insert (file_path . clone ())"]
    test_read_after_write_aborts_5["let result = compose_batch (& [sig1 , sig2])"]
    test_read_after_write_aborts_6["match result { CompositionResult :: Inadmissible { first_failure_index , conf..."]
    test_read_after_write_aborts_7["EXIT"]
    test_read_after_write_aborts_0 --> test_read_after_write_aborts_1
    test_read_after_write_aborts_1 --> test_read_after_write_aborts_2
    test_read_after_write_aborts_2 --> test_read_after_write_aborts_3
    test_read_after_write_aborts_3 --> test_read_after_write_aborts_4
    test_read_after_write_aborts_4 --> test_read_after_write_aborts_5
    test_read_after_write_aborts_5 --> test_read_after_write_aborts_6
    test_read_after_write_aborts_6 --> test_read_after_write_aborts_7
```

## Function: `test_single_action_is_admissible`

- File: src/composition_rule.rs
- Branches: 0
- Loops: 0
- Nodes: 5
- Edges: 4

```mermaid
flowchart TD
    test_single_action_is_admissible_0["ENTRY"]
    test_single_action_is_admissible_1["let sig = create_test_signature ('action1' , vec ! [PathBuf :: from ('file1.rs')] , Inv..."]
    test_single_action_is_admissible_2["let result = compose_batch (& [sig])"]
    test_single_action_is_admissible_3["match result { CompositionResult :: Admissible { action_count , .. } => { ass..."]
    test_single_action_is_admissible_4["EXIT"]
    test_single_action_is_admissible_0 --> test_single_action_is_admissible_1
    test_single_action_is_admissible_1 --> test_single_action_is_admissible_2
    test_single_action_is_admissible_2 --> test_single_action_is_admissible_3
    test_single_action_is_admissible_3 --> test_single_action_is_admissible_4
```

