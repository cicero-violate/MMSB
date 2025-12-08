# v0.2.0-rust-alpha Release Plan

## Blocking Criteria
- Rust checkpoints+replay validated across allocator/log combinations.
- GPU validation suite green on CI.
- `cargo build --release` + `julia --project test/runtests.jl` green on CI.
- Docs updated (README, Architecture, SerializationSpec, API changes).

## Checklist
1. [ ] Finalize replay/changelog documentation.
2. [ ] Capture performance baselines vs. v0.1.x.
3. [ ] Draft migration guide for Julia users.
4. [ ] Tag `v0.2.0-rust-alpha` once CI green.
5. [ ] Announce release + publish binaries.
