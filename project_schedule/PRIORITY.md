| Node                                                    | Rating (R_i) | Order (O_i) | Priority     |
| ------------------------------------------------------- | ------------ | ----------- | ------------ |
| MMSB Core (page model, delta routing, memory semantics) | 9.8          | 1           | **Critical** |
| Canonical Delta IR                                      | 9.6          | 2           | **Critical** |
| Query / Mutate / Upsert Core                            | 9.5          | 3           | **Critical** |
| Intent Persistence (intent ≠ execution)                 | 9.7          | 4           | **Critical** |
| Deterministic Replay / Fork Engine                      | 9.2          | 5           | **High**     |
| Time-Aware Query (state over time)                      | 8.9          | 6           | **High**     |
| Idempotent Mutation Semantics                           | 8.6          | 7           | **High**     |
| Error-as-Data Pipeline                                  | 8.2          | 8           | Medium       |
| Single Global Invariant (enforced everywhere)           | 9.3          | 9           | **High**     |
| Cognitive Surface Reduction (API discipline)            | 8.8          | 10          | **High**     |
