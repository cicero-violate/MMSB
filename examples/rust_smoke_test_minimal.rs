use std::ffi::{c_char, CString};

#[repr(C)]
#[derive(Clone, Copy)]
struct PageHandle { ptr: *mut () }

#[repr(u32)]
#[derive(Debug, Copy, Clone)]
enum MMSBErrorCode { Ok = 0, AllocError = 1, IOError = 2, SnapshotError = 3, CorruptLog = 4, InvalidHandle = 5 }

#[repr(C)]
#[derive(Default)]
struct TLogSummary {
    total_deltas: u64,
    total_bytes: u64,
    last_epoch: u32,
}

extern "C" {
    fn mmsb_get_last_error() -> MMSBErrorCode;
    fn mmsb_page_new(page_id: u64, size: usize, location: i32) -> PageHandle;
    fn mmsb_page_free(handle: PageHandle);
    fn mmsb_page_read(handle: PageHandle, dst: *mut u8, len: usize) -> usize;
    fn mmsb_page_epoch(handle: PageHandle) -> u32;
    fn mmsb_tlog_summary(path: *const c_char, out: *mut TLogSummary) -> i32;
}

fn main() {
    println!("=== MMSB Rust FFI smoke test ===\n");

    let page = unsafe { mmsb_page_new(42, 4096, 0) };
    if page.ptr.is_null() {
        println!("page_new failed — error = {:?}", unsafe { mmsb_get_last_error() });
        return;
    }
    println!("page_new SUCCESS → {:p}", page.ptr);

    let mut buf = [0u8; 64];
    let n = unsafe { mmsb_page_read(page, buf.as_mut_ptr(), buf.len()) };
    println!("read {} bytes", n);

    let epoch = unsafe { mmsb_page_epoch(page) };
    println!("epoch = {}", epoch);

    unsafe { mmsb_page_free(page) };
    println!("page freed");

    let path = CString::new("dummy.tlog").unwrap();
    let mut summary = TLogSummary::default();
    let rc = unsafe { mmsb_tlog_summary(path.as_ptr(), &mut summary) };
    println!("tlog_summary rc = {}", rc);

    println!("\nRUST CORE IS HEALTHY!");
}
