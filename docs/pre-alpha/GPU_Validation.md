# GPU Validation Track

## Goals
- Prove pointer stability for Rust-managed pages when mirrored on CUDA devices.
- Document host/device synchronization flow with the new checkpoint + replay pipeline.
- Establish conformance tests for: unified memory migration, CUDA stream async copies, and GPU delta application.

## Immediate Tasks
1. Instrument Rust allocator to expose device residency metadata per page.
2. Build a tiny validation kernel that reads/writes Rust pages via FFI pointers.
3. Record expected behaviors (epoch bumping, mask updates) when GPU writes occur.

## Deliverables
- Test harness under `test/gpu_validation.jl` gating CUDA + Rust coherence.
- Rust-side integration tests asserting device registry invariants.
- Documentation describing failure modes and mitigations.
