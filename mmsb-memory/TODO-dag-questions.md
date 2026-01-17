## Questions for Clarification

1. **materialization/ placement**: Currently classified as KEEP. The `pure_materialize.rs` constructs views from deltas, which is "what state becomes" not "how it's executed". Correct?
2. **page/ and delta/ allocation**: Need to verify if these contain physical allocation code or only logical definitions. Should I audit these next?
3. **MemoryView trait location**: Should this go in `mmsb-memory/src/memory_view.rs` or as a separate trait in `mmsb-events` or `mmsb-proof` for protocol formalization?
4. **numbered directory deletion**: Should numbered directories be removed in Phase 1, or after Phase 10 to avoid disrupting current structure during refactor?
5. **External dependency audit scope**: Phase 2.4 defers external reference checking. Should this happen before or after Phase 4's substrate move?
