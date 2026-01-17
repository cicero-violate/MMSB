# mmsb-intent Architecture

## Module Purpose

$$\text{mmsb-intent}: \text{Intent} \rightarrow \{\text{Allow}, \text{RequireJudgment}\}$$

Determines which operations can bypass the judgment ritual based on classification and policy rules.

## Mathematical Model

### Variables

Let:
- $I$ = Intent content (patch, command, operation description)
- $C$ = Set of intent classes (formatting, refactor, structural, etc.)
- $M$ = IntentMetadata extracted from $I$
- $P$ = ScopePolicy (rules defining what's allowed)
- $D$ = PolicyDecision (Allow or RequireJudgment)

### Classification Function

$$\text{classify}: I \rightarrow M$$

where $M = \{C, \text{paths}, \text{tools}, n_{\text{files}}, n_{\text{lines}}\}$

### Policy Evaluation

$$\text{evaluate}: M \times P \rightarrow D$$

$$D = \begin{cases}
\text{Allow} & \text{if } \text{violations}(M, P) = \emptyset \\
\text{RequireJudgment}(V) & \text{otherwise}
\end{cases}$$

where $V = \text{violations}(M, P)$

### Violation Rules

$$\text{violations}(M, P) = V_c \cup V_p \cup V_t \cup V_s$$

$$V_c = \{v \mid c \in M.C \land c \notin P.\text{allowed\_classes}\}$$
$$V_p = \{v \mid p \in M.\text{paths} \land (p \notin P.\text{allowed} \lor p \in P.\text{forbidden})\}$$
$$V_t = \{v \mid t \in M.\text{tools} \land (t \notin P.\text{allowed} \lor t \in P.\text{forbidden})\}$$
$$V_s = \{v \mid M.n_{\text{files}} > P.\text{max\_files} \lor M.n_{\text{lines}} > P.\text{max\_lines}\}$$

## Component Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    mmsb-intent Library                   │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌────────────────┐         ┌─────────────────┐        │
│  │ IntentClassifier│────────>│ IntentMetadata  │        │
│  └────────────────┘         └─────────────────┘        │
│         │                            │                  │
│         │ analyze(content)           │ .classes         │
│         │ extract_paths()            │ .paths           │
│         │ estimate_complexity()      │ .tools           │
│         │                            │ .files_touched   │
│         v                            │ .diff_lines      │
│                                      v                  │
│                            ┌──────────────────┐         │
│  ┌──────────────┐          │ PolicyEvaluator  │         │
│  │ ScopePolicy  │─────────>│                  │         │
│  │              │          │ evaluate(M, P)   │         │
│  │ .allowed_*   │          │       │          │         │
│  │ .forbidden_* │          │       v          │         │
│  │ .max_*       │          │ PolicyDecision   │         │
│  └──────────────┘          └──────────────────┘         │
│         ^                                                │
│         │                                                │
│  ┌──────────────┐                                       │
│  │ ScopeManager │                                       │
│  │              │                                       │
│  │ load/save    │                                       │
│  └──────────────┘                                       │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

## Integration Flow

```
User Intent (patch/command)
         │
         v
    ┌─────────────────────┐
    │ IntentClassifier    │
    │ - Parse content     │
    │ - Extract metadata  │
    │ - Classify type     │
    └─────────────────────┘
         │
         v
    IntentMetadata
         │
         v
    ┌─────────────────────┐         ┌──────────────┐
    │ PolicyEvaluator     │<────────│ ScopePolicy  │
    │ - Check classes     │         │ (from file)  │
    │ - Check paths       │         └──────────────┘
    │ - Check tools       │
    │ - Check limits      │
    └─────────────────────┘
         │
         v
    PolicyDecision
         │
         ├─> Allow ──────────────────> Execute directly
         │                              (bypass judgment)
         │
         └─> RequireJudgment(V) ────> mmsb-judgment ritual
                                       (human authorization)
```

## Data Flow Example

```rust
// 1. Load intent
let intent = fs::read_to_string("operation.patch")?;

// 2. Classify
let mut metadata = IntentClassifier::classify(&intent);
let paths = IntentClassifier::extract_paths(&intent);
let (files, lines) = IntentClassifier::estimate_complexity(&intent);
metadata = metadata.with_paths(paths)
                   .with_files_touched(files)
                   .with_diff_lines(lines);

// 3. Load policy
let policy = ScopeManager::load_from_file("policy.json")?;

// 4. Evaluate
let evaluator = PolicyEvaluator::new(policy);
let decision = evaluator.evaluate(&metadata);

// 5. Branch on decision
match decision {
    PolicyDecision::Allow => {
        // Safe operation, execute directly
        execute_intent(&intent)?;
    }
    PolicyDecision::RequireJudgment(violations) => {
        // Requires human judgment
        println!("Judgment required:");
        for v in violations {
            println!("  - {}", v.description());
        }
        
        // Invoke judgment ritual
        let token = invoke_judgment_ritual(&intent)?;
        execute_with_token(&intent, token)?;
    }
}
```

## Policy Schema

```json
{
  "schema": "intent_policy.v1",
  "scope_id": "safe_operations",
  "allowed_classes": ["formatting", "lint_fix", "documentation"],
  "allowed_paths": ["src/**/*.rs", "tests/**/*.rs"],
  "forbidden_paths": ["migrations/**", "infra/**"],
  "allowed_tools": ["rustfmt", "clippy"],
  "forbidden_tools": ["shell_runner"],
  "max_files_touched": 50,
  "max_diff_lines": 2000,
  "version": 1
}
```

## Design Invariants

1. **Safe by Default**: Unknown/unclassified intents → RequireJudgment
2. **Explicit Allowlist**: Must be in allowed set (not just absent from forbidden)
3. **Any Violation = Judgment**: Single rule violation triggers judgment requirement
4. **Immutable After Decision**: PolicyDecision is final, no retries without new intent
5. **Auditable**: Every RequireJudgment includes complete violation list

## Extension Points

### Custom Classifiers

```rust
pub trait IntentClassifierExt {
    fn classify_custom(&self, content: &str) -> Vec<IntentClass>;
}

// ML-based classifier
impl IntentClassifierExt for MLClassifier {
    fn classify_custom(&self, content: &str) -> Vec<IntentClass> {
        self.model.predict(content)
    }
}
```

### Policy Composition

```rust
pub struct CompositePolicy {
    policies: Vec<ScopePolicy>,
    mode: CompositionMode,
}

pub enum CompositionMode {
    AllMustAllow,  // Strict: all policies must allow
    AnyCanAllow,   // Permissive: any policy allowing is sufficient
}
```

### Custom Violation Types

```rust
pub enum PolicyViolation {
    // Core violations
    ClassNotAllowed(String),
    PathNotAllowed(String),
    // ... existing variants ...
    
    // Extensible
    Custom {
        rule_id: String,
        description: String,
        severity: ViolationSeverity,
    },
}
```

## Testing Strategy

- **Unit**: Each component tested independently
- **Integration**: Classify → Evaluate pipeline
- **Property**: Invariant checking (safe by default, etc.)
- **Regression**: Known intents with expected decisions

## Future Work

- [ ] ML-based classification (train on historical judgments)
- [ ] Policy templates library
- [ ] Intent pattern matching DSL
- [ ] Policy testing framework
- [ ] Performance optimization (caching, lazy evaluation)
- [ ] Policy versioning and migration
