# Rust Core Agent Playbook
## Charter
- Maintain Rust allocator/page/delta implementations.
- Instrument and patch `src/ffi.rs` plus supporting modules.
- Ensure `libmmsb_core.so` stays in sync with Julia expectations.

## Tool Belt
- `cargo build --release && cp target/release/libmmsb_core.so .`
- `cargo test` or targeted `cargo test <module>` for Rust-only coverage.
- `RUSTFLAGS="-Z sanitizer=address" cargo +nightly test` (sanity when needed).
- `rg -n "" src/*.rs` for quick code navigation.

## Standard Runbooks
1. Review latest diagnostics evidence (`DIAGNOSTICS.md`, logs).
2. Modify Rust sources; keep notes on invariants touched.
3. Rebuild + copy `.so`.
4. Notify Diagnostics/QA agents before/after running Julia tests.
5. Log changes in `TASK_LOG.md` (Task IDs T0.x, P8.4, etc.).

## Reporting
- Summaries or design notes can live in `pending_work/notes/` (link from `TASK_LOG.md`).
- Update `DAG_PRIORITY.md` when Rust tasks move state.

## Open Questions / TODOs
- Add more Rust unit tests (allocator, page read/write).
- Document expected log signatures per failure mode.
