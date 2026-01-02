# Warning Hygiene

Action: Review only. Detection-only report.

- Source: `MMSB/tools/mmsb-analyzer/docs/cargo_warnings.txt`
- Warnings parsed: 3

## Warning Types

- `mmsb-analyzer` (example "dogfood_admission") generated 1 warning (run `cargo fix --example "dogfood_admission" -p mmsb-analyzer` to apply 1 suggestion) (1 occurrences)
- variable does not need to be mutable (2 occurrences)

## Warning Details

- `examples/run_batch_admission.rs:98` → variable does not need to be mutable
- `examples/dogfood_admission.rs:94` → variable does not need to be mutable
- `unknown` → `mmsb-analyzer` (example "dogfood_admission") generated 1 warning (run `cargo fix --example "dogfood_admission" -p mmsb-analyzer` to apply 1 suggestion)

