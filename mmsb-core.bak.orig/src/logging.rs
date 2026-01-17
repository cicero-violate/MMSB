use std::sync::OnceLock;

fn diagnostics_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| match std::env::var("MMSB_FFI_DEBUG") {
        Ok(value) => matches!(value.as_str(), "1" | "true" | "TRUE"),
        Err(_) => false,
    })
}

pub(crate) fn is_enabled() -> bool {
    diagnostics_enabled()
}

#[macro_export]
macro_rules! ffi_debug {
    ($($arg:tt)*) => {{
        if $crate::logging::is_enabled() {
            eprintln!($($arg)*);
        }
    }};
}
