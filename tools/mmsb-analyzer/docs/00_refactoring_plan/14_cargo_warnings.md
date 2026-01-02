## Cargo Warnings

Action: address compiler warnings before major refactors.
Note: captured from cargo check/test outputs.

```text
Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.31s
warning: variable does not need to be mutable
  --> examples/run_batch_admission.rs:98:9
   |
98 |     let mut modules_touched = BTreeSet::new();
   |         ----^^^^^^^^^^^^^^^
   |         |
   |         help: remove this `mut`
   |
   = note: `#[warn(unused_mut)]` (part of `#[warn(unused)]`) on by default

warning: variable does not need to be mutable
  --> examples/dogfood_admission.rs:94:9
   |
94 |     let mut modules_touched = BTreeSet::new();
   |         ----^^^^^^^^^^^^^^^
   |         |
   |         help: remove this `mut`
   |
   = note: `#[warn(unused_mut)]` (part of `#[warn(unused)]`) on by default

warning: `mmsb-analyzer` (example "run_batch_admission") generated 1 warning (run `cargo fix --example "run_batch_admission" -p mmsb-analyzer` to apply 1 suggestion)
warning: `mmsb-analyzer` (example "dogfood_admission") generated 1 warning (run `cargo fix --example "dogfood_admission" -p mmsb-analyzer` to apply 1 suggestion)
    Finished `test` profile [unoptimized + debuginfo] target(s) in 0.19s
  Executable unittests src/360_lib.rs (target/debug/deps/mmsb_analyzer-16aa7e17a552ead4)
  Executable unittests src/340_main.rs (target/debug/deps/mmsb_analyzer-635e46ccd5cc173a)
  Executable tests/ci_admission_gate.rs (target/debug/deps/ci_admission_gate-c143fb113a7214ab)
```

