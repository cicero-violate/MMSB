# Structure Group: src/07_intention

## File: MMSB/src/07_intention/IntentionTypes.jl

- Layer(s): 07_intention
- Language coverage: Julia (5)
- Element types: Function (1), Module (1), Struct (3)
- Total elements: 5

### Elements

- [Julia | Module] `IntentionTypes` (line 6, pub)
- [Julia | Struct] `Intention` (line 17, pub)
  - Signature: `struct Intention`
- [Julia | Struct] `Goal` (line 31, pub)
  - Signature: `struct Goal`
- [Julia | Struct] `IntentionState` (line 44, pub)
  - Signature: `mutable struct IntentionState`
- [Julia | Function] `IntentionState` (line 51, pub)
  - Signature: `IntentionState()`
  - Calls: IntentionState, time

## File: MMSB/src/07_intention/UpsertPlan.jl

- Layer(s): 07_intention
- Language coverage: Julia (4)
- Element types: Function (1), Module (1), Struct (2)
- Total elements: 4

### Elements

- [Julia | Module] `UpsertPlan` (line 1, pub)
- [Julia | Struct] `DeltaSpec` (line 5, pub)
  - Signature: `struct DeltaSpec`
- [Julia | Struct] `UpsertPlan` (line 11, pub)
  - Signature: `struct UpsertPlan`
- [Julia | Function] `validate_plan` (line 18, pub)
  - Signature: `validate_plan(plan::UpsertPlan)`
  - Calls: error, isempty, length

## File: MMSB/src/07_intention/attractor_states.jl

- Layer(s): 07_intention
- Language coverage: Julia (5)
- Element types: Function (3), Module (1), Struct (1)
- Total elements: 5

### Elements

- [Julia | Module] `AttractorStates` (line 6, pub)
- [Julia | Struct] `AttractorField` (line 15, pub)
  - Signature: `struct AttractorField`
- [Julia | Function] `compute_gradient` (line 25, pub)
  - Signature: `compute_gradient(field::AttractorField, state::Vector{Float64})`
  - Calls: length, sqrt, sum, zeros, zip
- [Julia | Function] `evolve_state` (line 44, pub)
  - Signature: `evolve_state(field::AttractorField, state::Vector{Float64}, dt::Float64)`
  - Calls: compute_gradient
- [Julia | Function] `find_nearest_attractor` (line 54, pub)
  - Signature: `find_nearest_attractor(field::AttractorField, state::Vector{Float64})`
  - Calls: argmin, sqrt, sum

## File: MMSB/src/07_intention/goal_emergence.jl

- Layer(s): 07_intention
- Language coverage: Julia (3)
- Element types: Function (2), Module (1)
- Total elements: 3

### Elements

- [Julia | Module] `GoalEmergence` (line 6, pub)
- [Julia | Function] `utility_gradient` (line 17, pub)
  - Signature: `utility_gradient(utility_history::Vector{Float64})`
  - Calls: length, max, sum
- [Julia | Function] `detect_goals` (line 38, pub)
  - Signature: `detect_goals(utility_state, threshold::Float64)`
  - Calls: Dict, Goal, UInt64, abs, push!, utility_gradient

## File: MMSB/src/07_intention/intent_lowering.jl

- Layer(s): 07_intention
- Language coverage: Julia (4)
- Element types: Function (3), Module (1)
- Total elements: 4

### Elements

- [Julia | Module] `IntentLowering` (line 6, pub)
- [Julia | Function] `mask_to_bytes` (line 17, pub)
  - Signature: `mask_to_bytes(mask::Vector{Bool})`
- [Julia | Function] `lower_intent_to_deltaspec` (line 21, pub)
  - Signature: `lower_intent_to_deltaspec(plan::UpsertPlan)`
  - Calls: mask_to_bytes, validate_plan
- [Julia | Function] `execute_upsert_plan!` (line 29, pub)
  - Signature: `execute_upsert_plan!(state::MMSBState, plan::UpsertPlan; source::Symbol`

## File: MMSB/src/07_intention/intention_engine.jl

- Layer(s): 07_intention
- Language coverage: Julia (4)
- Element types: Function (3), Module (1)
- Total elements: 4

### Elements

- [Julia | Module] `IntentionEngine` (line 6, pub)
- [Julia | Function] `form_intention` (line 18, pub)
  - Signature: `form_intention(utility_state, layout_state, id::UInt64)`
  - Calls: UtilityEngine.utility_trend
- [Julia | Function] `evaluate_intention` (line 50, pub)
  - Signature: `evaluate_intention(intention::Intention, current_utility::Float64)`
  - Calls: Float64, length
- [Julia | Function] `select_best_intention` (line 66, pub)
  - Signature: `select_best_intention(intentions::Vector{Intention}, utility::Float64)`
  - Calls: argmax, evaluate_intention, isempty

## File: MMSB/src/07_intention/structural_preferences.jl

- Layer(s): 07_intention
- Language coverage: Julia (4)
- Element types: Function (2), Module (1), Struct (1)
- Total elements: 4

### Elements

- [Julia | Module] `StructuralPreferences` (line 6, pub)
- [Julia | Struct] `Preference` (line 15, pub)
  - Signature: `struct Preference`
- [Julia | Function] `evaluate_preference` (line 26, pub)
  - Signature: `evaluate_preference(pref::Preference, state)`
  - Calls: pref.constraint
- [Julia | Function] `apply_preferences` (line 40, pub)
  - Signature: `apply_preferences(prefs::Vector{Preference}, state)`
  - Calls: evaluate_preference, sum

