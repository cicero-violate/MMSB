# Structure Group: src/logging.rs

## File: MMSB/src/logging.rs

- Layer(s): root
- Language coverage: Rust (2)
- Element types: Function (2)
- Total elements: 2

### Elements

- [Rust | Function] `diagnostics_enabled` (line 0, priv)
  - Signature: `fn diagnostics_enabled () -> bool { static ENABLED : OnceLock < bool > = OnceLock :: new () ; * ENABLED . get_or_init...`
  - Calls: OnceLock::new, get_or_init, std::env::var
- [Rust | Function] `is_enabled` (line 0, pub(crate))
  - Signature: `pub (crate) fn is_enabled () -> bool { diagnostics_enabled () } . sig`
  - Calls: diagnostics_enabled

