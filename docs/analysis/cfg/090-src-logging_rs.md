# CFG Group: src/logging.rs

## Function: `diagnostics_enabled`

- File: MMSB/src/logging.rs
- Branches: 0
- Loops: 0
- Nodes: 4
- Edges: 3

```mermaid
flowchart TD
    diagnostics_enabled_0["ENTRY"]
    diagnostics_enabled_1["item"]
    diagnostics_enabled_2["* ENABLED . get_or_init (| | match std :: env :: var ('MMSB_FFI_DEBUG') { Ok ..."]
    diagnostics_enabled_3["EXIT"]
    diagnostics_enabled_0 --> diagnostics_enabled_1
    diagnostics_enabled_1 --> diagnostics_enabled_2
    diagnostics_enabled_2 --> diagnostics_enabled_3
```

## Function: `is_enabled`

- File: MMSB/src/logging.rs
- Branches: 0
- Loops: 0
- Nodes: 3
- Edges: 2

```mermaid
flowchart TD
    is_enabled_0["ENTRY"]
    is_enabled_1["diagnostics_enabled ()"]
    is_enabled_2["EXIT"]
    is_enabled_0 --> is_enabled_1
    is_enabled_1 --> is_enabled_2
```

