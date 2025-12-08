use crate::runtime::allocator::{PageAllocator, PageAllocatorConfig};
use crate::runtime::checkpoint;
use crate::runtime::tlog::{TransactionLog, TransactionLogReader};
use crate::types::{Delta, DeltaID, Epoch, Page, PageError, PageID, PageLocation, Source};
use std::cell::RefCell;
use std::cmp::min;
use std::ffi::{CStr, CString};
use std::io::ErrorKind;
use std::os::raw::c_char;
use std::panic;
use std::ptr;
use std::thread_local;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct PageHandle {
    pub ptr: *mut Page,
}

impl PageHandle {
    fn null() -> Self {
        Self {
            ptr: ptr::null_mut(),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct DeltaHandle {
    pub ptr: *mut Delta,
}

impl DeltaHandle {
    fn null() -> Self {
        Self {
            ptr: ptr::null_mut(),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct AllocatorHandle {
    pub ptr: *mut PageAllocator,
}

impl AllocatorHandle {
    fn null() -> Self {
        Self {
            ptr: ptr::null_mut(),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct TLogHandle {
    pub ptr: *mut TransactionLog,
}

impl TLogHandle {
    fn null() -> Self {
        Self {
            ptr: ptr::null_mut(),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct TLogReaderHandle {
    pub ptr: *mut TransactionLogReader,
}

impl TLogReaderHandle {
    fn null() -> Self {
        Self {
            ptr: ptr::null_mut(),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MMSBErrorCode {
    Ok = 0,
    AllocError = 1,
    IOError = 2,
    SnapshotError = 3,
    CorruptLog = 4,
    InvalidHandle = 5,
}

thread_local! {
    static TLS_LAST_ERROR: RefCell<MMSBErrorCode> = RefCell::new(MMSBErrorCode::Ok);
    static TLS_PAGE_METADATA: RefCell<Vec<Vec<u8>>> = RefCell::new(Vec::new());
}

fn set_last_error(code: MMSBErrorCode) {
    TLS_LAST_ERROR.with(|cell| {
        *cell.borrow_mut() = code;
    });
}

fn log_error_code(err: &std::io::Error) -> MMSBErrorCode {
    if err.kind() == ErrorKind::InvalidData {
        MMSBErrorCode::CorruptLog
    } else {
        MMSBErrorCode::IOError
    }
}

#[no_mangle]
pub extern "C" fn mmsb_get_last_error() -> MMSBErrorCode {
    TLS_LAST_ERROR.with(|cell| {
        let mut guard = cell.borrow_mut();
        let code = *guard;
        *guard = MMSBErrorCode::Ok;
        code
    })
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct TLogSummary {
    pub total_deltas: u64,
    pub total_bytes: u64,
    pub last_epoch: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct PageInfoABI {
    pub page_id: u64,
    pub size: usize,
    pub location: i32,
    pub epoch: u32,
    pub metadata_ptr: *const u8,
    pub metadata_len: usize,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct EpochABI {
    pub value: u32,
}

impl From<EpochABI> for Epoch {
    fn from(value: EpochABI) -> Self {
        Epoch(value.value)
    }
}

fn convert_location(tag: i32) -> Result<PageLocation, PageError> {
    PageLocation::from_tag(tag)
}

fn mask_from_bytes(ptr: *const u8, len: usize) -> Vec<bool> {
    if ptr.is_null() || len == 0 {
        return Vec::new();
    }
    unsafe { std::slice::from_raw_parts(ptr, len) }
        .iter()
        .map(|v| *v != 0)
        .collect()
}

fn vec_from_ptr(ptr: *const u8, len: usize) -> Vec<u8> {
    if ptr.is_null() || len == 0 {
        return Vec::new();
    }
    unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec()
}

#[no_mangle]
pub extern "C" fn mmsb_page_read(handle: PageHandle, dst: *mut u8, len: usize) -> usize {
    eprintln!("=== mmsb_page_read START ===");
    eprintln!("  handle.ptr = {:p}", handle.ptr);
    eprintln!("  dst = {:p}", dst);
    eprintln!("  requested len = {}", len);

    if handle.ptr.is_null() {
        eprintln!("  ERROR: Invalid page handle (null)");
        set_last_error(MMSBErrorCode::InvalidHandle);
        return 0;
    }

    // Special case: len == 0 → just return page size (used by tests)
    if len == 0 {
        let page = unsafe { &*handle.ptr };
        let size = page.size();
        eprintln!("  Returning page size: {} (len=0 query)", size);
        return size;
    }

    // Normal case: dst must not be null
    if dst.is_null() {
        eprintln!("  ERROR: dst is null but len > 0");
        set_last_error(MMSBErrorCode::InvalidHandle);
        return 0;
    }

    let page = unsafe { &*handle.ptr };
    let page_size = page.size();
    let bytes_to_copy = len.min(page_size);

    eprintln!("  page size = {}, copying {} bytes", page_size, bytes_to_copy);

    let src_slice = page.data_slice();
    unsafe {
        std::ptr::copy_nonoverlapping(src_slice.as_ptr(), dst, bytes_to_copy);
    }

    eprintln!("=== mmsb_page_read END (success, {} bytes) ===", bytes_to_copy);
    bytes_to_copy
}

#[no_mangle]
pub extern "C" fn mmsb_page_epoch(handle: PageHandle) -> u32 {
    if handle.ptr.is_null() {
        set_last_error(MMSBErrorCode::InvalidHandle);
        return 0;
    }
    let page = unsafe { &*handle.ptr };
    page.epoch().0
}

#[no_mangle]
pub extern "C" fn mmsb_page_write_masked(
    handle: PageHandle,
    mask_ptr: *const u8,
    mask_len: usize,
    payload_ptr: *const u8,
    payload_len: usize,
    is_sparse: u8,
    epoch: EpochABI,
) -> i32 {
    if handle.ptr.is_null() {
        set_last_error(MMSBErrorCode::InvalidHandle);
        return -1;
    }
    let mask = mask_from_bytes(mask_ptr, mask_len);
    let payload = vec_from_ptr(payload_ptr, payload_len);
    let delta = Delta {
        delta_id: DeltaID(0),
        page_id: unsafe { (*handle.ptr).id },
        epoch: epoch.into(),
        mask,
        payload,
        is_sparse: is_sparse != 0,
        timestamp: 0,
        source: Source("page_write_masked".into()),
    };
    let page = unsafe { &mut *handle.ptr };
    match page.apply_delta(&delta) {
        Ok(_) => 0,
        Err(_) => {
            set_last_error(MMSBErrorCode::AllocError);
            -1
        }
    }
}

#[no_mangle]
pub extern "C" fn mmsb_page_metadata_size(handle: PageHandle) -> usize {
    if handle.ptr.is_null() {
        set_last_error(MMSBErrorCode::InvalidHandle);
        return 0;
    }
    let page = unsafe { &*handle.ptr };
    page.metadata_blob().len()
}

#[no_mangle]
pub extern "C" fn mmsb_page_metadata_export(handle: PageHandle, dst: *mut u8, len: usize) -> usize {
    if handle.ptr.is_null() || dst.is_null() {
        set_last_error(MMSBErrorCode::InvalidHandle);
        return 0;
    }
    let page = unsafe { &*handle.ptr };
    let blob = page.metadata_blob();
    let copy_len = len.min(blob.len());
    unsafe {
        std::ptr::copy_nonoverlapping(blob.as_ptr(), dst, copy_len);
    }
    copy_len
}

#[no_mangle]
pub extern "C" fn mmsb_page_metadata_import(handle: PageHandle, src: *const u8, len: usize) -> i32 {
    if handle.ptr.is_null() || src.is_null() {
        set_last_error(MMSBErrorCode::InvalidHandle);
        return -1;
    }
    let page = unsafe { &mut *handle.ptr };
    let blob = unsafe { std::slice::from_raw_parts(src, len) };
    match page.set_metadata_blob(blob) {
        Ok(_) => 0,
        Err(_) => {
            set_last_error(MMSBErrorCode::AllocError);
            -1
        }
    }
}

#[no_mangle]
pub extern "C" fn mmsb_delta_new(
    delta_id: u64,
    page_id: u64,
    epoch: EpochABI,
    mask_ptr: *const u8,
    mask_len: usize,
    payload_ptr: *const u8,
    payload_len: usize,
    is_sparse: u8,
    source_ptr: *const c_char,
) -> DeltaHandle {
    let mask = mask_from_bytes(mask_ptr, mask_len);
    let payload = vec_from_ptr(payload_ptr, payload_len);
    let source = if source_ptr.is_null() {
        "ffi_delta_new".to_string()
    } else {
        unsafe { CStr::from_ptr(source_ptr) }
            .to_string_lossy()
            .to_string()
    };
    let delta = Delta {
        delta_id: DeltaID(delta_id),
        page_id: PageID(page_id),
        epoch: epoch.into(),
        mask,
        payload,
        is_sparse: is_sparse != 0,
        timestamp: 0,
        source: Source(source),
    };
    let boxed = Box::new(delta);
    DeltaHandle {
        ptr: Box::into_raw(boxed),
    }
}

#[no_mangle]
pub extern "C" fn mmsb_delta_free(handle: DeltaHandle) {
    if !handle.ptr.is_null() {
        unsafe {
            drop(Box::from_raw(handle.ptr));
        }
    }
}

#[no_mangle]
pub extern "C" fn mmsb_delta_apply(page: PageHandle, delta: DeltaHandle) -> i32 {
    if page.ptr.is_null() || delta.ptr.is_null() {
        set_last_error(MMSBErrorCode::InvalidHandle);
        return -1;
    }
    let page_ref = unsafe { &mut *page.ptr };
    let delta_ref = unsafe { &*delta.ptr };
    match page_ref.apply_delta(delta_ref) {
        Ok(_) => 0,
        Err(_) => {
            set_last_error(MMSBErrorCode::AllocError);
            -1
        }
    }
}

#[no_mangle]
pub extern "C" fn mmsb_delta_id(handle: DeltaHandle) -> u64 {
    if handle.ptr.is_null() {
        set_last_error(MMSBErrorCode::InvalidHandle);
        return 0;
    }
    unsafe { (*handle.ptr).delta_id.0 }
}

#[no_mangle]
pub extern "C" fn mmsb_delta_page_id(handle: DeltaHandle) -> u64 {
    if handle.ptr.is_null() {
        set_last_error(MMSBErrorCode::InvalidHandle);
        return 0;
    }
    unsafe { (*handle.ptr).page_id.0 }
}

#[no_mangle]
pub extern "C" fn mmsb_delta_epoch(handle: DeltaHandle) -> u32 {
    if handle.ptr.is_null() {
        set_last_error(MMSBErrorCode::InvalidHandle);
        return 0;
    }
    unsafe { (*handle.ptr).epoch.0 }
}

#[no_mangle]
pub extern "C" fn mmsb_delta_is_sparse(handle: DeltaHandle) -> u8 {
    if handle.ptr.is_null() {
        set_last_error(MMSBErrorCode::InvalidHandle);
        return 0;
    }
    unsafe { (*handle.ptr).is_sparse as u8 }
}

#[no_mangle]
pub extern "C" fn mmsb_delta_timestamp(handle: DeltaHandle) -> u64 {
    if handle.ptr.is_null() {
        set_last_error(MMSBErrorCode::InvalidHandle);
        return 0;
    }
    unsafe { (*handle.ptr).timestamp }
}

#[no_mangle]
pub extern "C" fn mmsb_delta_source_len(handle: DeltaHandle) -> usize {
    if handle.ptr.is_null() {
        set_last_error(MMSBErrorCode::InvalidHandle);
        return 0;
    }
    unsafe { (&(*handle.ptr).source.0).len() }
}

#[no_mangle]
pub extern "C" fn mmsb_delta_copy_source(handle: DeltaHandle, dst: *mut u8, len: usize) -> usize {
    if handle.ptr.is_null() || dst.is_null() {
        set_last_error(MMSBErrorCode::InvalidHandle);
        return 0;
    }
    let source = unsafe { &(*handle.ptr).source.0 };
    let bytes = source.as_bytes();
    let copy_len = min(bytes.len(), len);
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst, copy_len);
    }
    copy_len
}

#[no_mangle]
pub extern "C" fn mmsb_delta_mask_len(handle: DeltaHandle) -> usize {
    if handle.ptr.is_null() {
        set_last_error(MMSBErrorCode::InvalidHandle);
        return 0;
    }
    unsafe { (*handle.ptr).mask.len() }
}

#[no_mangle]
pub extern "C" fn mmsb_delta_copy_mask(handle: DeltaHandle, dst: *mut u8, len: usize) -> usize {
    if handle.ptr.is_null() || dst.is_null() {
        set_last_error(MMSBErrorCode::InvalidHandle);
        return 0;
    }
    let mask = unsafe { &(*handle.ptr).mask };
    let copy_len = min(mask.len(), len);
    for (idx, flag) in mask.iter().enumerate().take(copy_len) {
        unsafe {
            *dst.add(idx) = if *flag { 1 } else { 0 };
        }
    }
    copy_len
}

#[no_mangle]
pub extern "C" fn mmsb_delta_payload_len(handle: DeltaHandle) -> usize {
    if handle.ptr.is_null() {
        set_last_error(MMSBErrorCode::InvalidHandle);
        return 0;
    }
    unsafe { (*handle.ptr).payload.len() }
}

#[no_mangle]
pub extern "C" fn mmsb_delta_copy_payload(handle: DeltaHandle, dst: *mut u8, len: usize) -> usize {
    if handle.ptr.is_null() || dst.is_null() {
        set_last_error(MMSBErrorCode::InvalidHandle);
        return 0;
    }
    let payload = unsafe { &(*handle.ptr).payload };
    let copy_len = min(payload.len(), len);
    unsafe {
        std::ptr::copy_nonoverlapping(payload.as_ptr(), dst, copy_len);
    }
    copy_len
}

#[no_mangle]
pub extern "C" fn mmsb_tlog_new(path: *const c_char) -> TLogHandle {
    if path.is_null() {
        set_last_error(MMSBErrorCode::InvalidHandle);
        return TLogHandle::null();
    }
    let c_str = unsafe { CStr::from_ptr(path) };
    let Ok(path_str) = c_str.to_str() else {
        set_last_error(MMSBErrorCode::InvalidHandle);
        return TLogHandle::null();
    };
    let owned = path_str.to_owned();
    match TransactionLog::new(owned) {
        Ok(log) => TLogHandle {
            ptr: Box::into_raw(Box::new(log)),
        },
        Err(err) => {
            set_last_error(log_error_code(&err));
            TLogHandle::null()
        }
    }
}

#[no_mangle]
pub extern "C" fn mmsb_tlog_free(handle: TLogHandle) {
    if !handle.ptr.is_null() {
        unsafe {
            drop(Box::from_raw(handle.ptr));
        }
    }
}

#[no_mangle]
pub extern "C" fn mmsb_tlog_append(handle: TLogHandle, delta: DeltaHandle) -> i32 {
    if handle.ptr.is_null() || delta.ptr.is_null() {
        set_last_error(MMSBErrorCode::InvalidHandle);
        return -1;
    }
    let log = unsafe { &*handle.ptr };
    let delta_ref = unsafe { &*delta.ptr };
    match log.append(delta_ref.clone()) {
        Ok(_) => 0,
        Err(err) => {
            set_last_error(log_error_code(&err));
            -1
        }
    }
}

#[no_mangle]
pub extern "C" fn mmsb_checkpoint_write(
    allocator: AllocatorHandle,
    log: TLogHandle,
    path: *const c_char,
) -> i32 {
    if allocator.ptr.is_null() || log.ptr.is_null() || path.is_null() {
        set_last_error(MMSBErrorCode::InvalidHandle);
        return -1;
    }
    let allocator_ref = unsafe { &*allocator.ptr };
    let log_ref = unsafe { &*log.ptr };
    let path_str = unsafe { CStr::from_ptr(path) }
        .to_string_lossy()
        .to_string();
    match checkpoint::write_checkpoint(allocator_ref, log_ref, path_str) {
        Ok(_) => 0,
        Err(_) => {
            set_last_error(MMSBErrorCode::SnapshotError);
            -1
        }
    }
}

#[no_mangle]
pub extern "C" fn mmsb_checkpoint_load(
    allocator: AllocatorHandle,
    log: TLogHandle,
    path: *const c_char,
) -> i32 {
    if allocator.ptr.is_null() || log.ptr.is_null() || path.is_null() {
        set_last_error(MMSBErrorCode::InvalidHandle);
        return -1;
    }
    let allocator_ref = unsafe { &*allocator.ptr };
    let log_ref = unsafe { &*log.ptr };
    let path_str = unsafe { CStr::from_ptr(path) }
        .to_string_lossy()
        .to_string();
    match checkpoint::load_checkpoint(allocator_ref, log_ref, path_str) {
        Ok(_) => 0,
        Err(_) => {
            set_last_error(MMSBErrorCode::SnapshotError);
            -1
        }
    }
}

// src/ffi.rs — ONLY THESE TWO FUNCTIONS

#[no_mangle]
pub extern "C" fn mmsb_tlog_reader_new(path: *const c_char) -> TLogReaderHandle {
    if path.is_null() {
        set_last_error(MMSBErrorCode::InvalidHandle);
        return TLogReaderHandle::null();
    }
    let path_str = match unsafe { CStr::from_ptr(path).to_str() } {
        Ok(s) => s.to_owned(),
        Err(_) => {
            set_last_error(MMSBErrorCode::IOError);
            return TLogReaderHandle::null();
        }
    };

    match TransactionLogReader::open(path_str) {
        Ok(reader) => TLogReaderHandle {
            ptr: Box::into_raw(Box::new(reader)),
        },
        Err(err) => {
            set_last_error(log_error_code(&err));
            TLogReaderHandle::null()
        }
    }
}



#[no_mangle]
pub extern "C" fn mmsb_tlog_reader_free(handle: TLogReaderHandle) {
    if !handle.ptr.is_null() {
        unsafe {
            drop(Box::from_raw(handle.ptr));
        }
    }
}

#[no_mangle]
pub extern "C" fn mmsb_tlog_reader_next(handle: TLogReaderHandle) -> DeltaHandle {
    if handle.ptr.is_null() {
        set_last_error(MMSBErrorCode::InvalidHandle);
        return DeltaHandle::null();
    }

    let reader = unsafe { &mut *handle.ptr };
    match reader.next() {
        Ok(Some(delta)) => {
            let boxed = Box::new(delta);
            DeltaHandle {
                ptr: Box::into_raw(boxed),
            }
        }
        Ok(None) => DeltaHandle::null(),
        Err(err) => {
            set_last_error(log_error_code(&err));
            DeltaHandle::null()
        }
    }
}

#[no_mangle]
pub extern "C" fn mmsb_tlog_summary(path: *const c_char, out: *mut TLogSummary) -> i32 {
    if path.is_null() || out.is_null() {
        set_last_error(MMSBErrorCode::InvalidHandle);
        return -1;
    }

    let path_str = match unsafe { CStr::from_ptr(path).to_str() } {
        Ok(s) => s.to_owned(),
        Err(_) => {
            set_last_error(MMSBErrorCode::IOError);
            return -1;
        }
    };

    let log = match TransactionLog::new(&path_str) {
        Ok(log) => log,
        Err(e) => {
            set_last_error(log_error_code(&e));
            return -1;
        }
    };

    let summary = log.summary();

    unsafe {
        (*out).total_deltas = summary.total_deltas;
        (*out).total_bytes = summary.total_bytes;
        (*out).last_epoch = summary.last_epoch;
    }

    0
}

// ──────────────────────────────────────────────────────────────
//  PageAllocator FFI – complete and bulletproof
// ──────────────────────────────────────────────────────────────
#[no_mangle]
pub extern "C" fn mmsb_allocator_new() -> AllocatorHandle {
    eprintln!("=== mmsb_allocator_new START ===");
    let config = PageAllocatorConfig::default();
    let allocator = PageAllocator::new(config);
    let boxed = Box::new(allocator);
    let ptr = Box::into_raw(boxed);
    eprintln!("   Allocator created at {:p}", ptr);
    eprintln!("=== mmsb_allocator_new END ===");
    AllocatorHandle { ptr }
}

#[no_mangle]
pub extern "C" fn mmsb_allocator_free(handle: AllocatorHandle) {
    if !handle.ptr.is_null() {
        eprintln!("Freeing allocator at {:p}", handle.ptr);
        unsafe {
            drop(Box::from_raw(handle.ptr));
        }
    }
}

#[no_mangle]
pub extern "C" fn mmsb_allocator_allocate(
    handle: AllocatorHandle,
    page_id_hint: u64,
    size: usize,
    location: i32,
) -> PageHandle {
    eprintln!("=== mmsb_allocator_allocate START ===");
    eprintln!("  handle = {:p}, page_id_hint = {}, size = {}", handle.ptr, page_id_hint, size);

    if handle.ptr.is_null() {
        set_last_error(MMSBErrorCode::InvalidHandle);
        return PageHandle::null();
    }
    if size == 0 {
        set_last_error(MMSBErrorCode::AllocError);
        return PageHandle::null();
    }

    let allocator = unsafe { &*handle.ptr };
    let loc = match PageLocation::from_tag(location) {
        Ok(loc) => Some(loc),
        Err(_) => None,
    };

    match allocator.allocate_raw(PageID(page_id_hint), size, loc) {
        Ok(page_ptr) => {
            eprintln!("   Allocation SUCCESS → page at {:p}", page_ptr);
            eprintln!("=== mmsb_allocator_allocate END (success) ===");
            PageHandle { ptr: page_ptr }
        }
        Err(_) => {
            eprintln!("   Allocation FAILED");
            set_last_error(MMSBErrorCode::AllocError);
            PageHandle::null()
        }
    }
}

#[no_mangle]
pub extern "C" fn mmsb_allocator_release(handle: AllocatorHandle, page_id: u64) {
    if handle.ptr.is_null() {
        set_last_error(MMSBErrorCode::InvalidHandle);
        return;
    }
    let allocator = unsafe { &*handle.ptr };
    allocator.release(PageID(page_id));
}

#[no_mangle]
pub extern "C" fn mmsb_allocator_get_page(handle: AllocatorHandle, page_id: u64) -> PageHandle {
    if handle.ptr.is_null() {
        set_last_error(MMSBErrorCode::InvalidHandle);
        return PageHandle::null();
    }
    let allocator = unsafe { &*handle.ptr };
    match allocator.acquire_page(PageID(page_id)) {
        Some(ptr) => PageHandle { ptr },
        None => {
            set_last_error(MMSBErrorCode::InvalidHandle);
            PageHandle::null()
        }
    }
}

#[no_mangle]
pub extern "C" fn mmsb_allocator_page_count(handle: AllocatorHandle) -> usize {
    if handle.ptr.is_null() {
        set_last_error(MMSBErrorCode::InvalidHandle);
        return 0;
    }
    let allocator = unsafe { &*handle.ptr };
    allocator.len()
}

#[no_mangle]
pub extern "C" fn mmsb_allocator_list_pages(
    handle: AllocatorHandle,
    out_infos: *mut PageInfoABI,
    max_count: usize,
) -> usize {
    if handle.ptr.is_null() || out_infos.is_null() {
        set_last_error(MMSBErrorCode::InvalidHandle);
        return 0;
    }
    let allocator = unsafe { &*handle.ptr };
    let infos = allocator.page_infos();

    let count = max_count.min(infos.len());
    TLS_PAGE_METADATA.with(|storage| {
        let mut storage = storage.borrow_mut();
        storage.clear();
        storage.extend(infos.iter().take(count).map(|info| info.metadata.clone()));

        for (i, info) in infos.iter().take(count).enumerate() {
            let blob = &storage[i];
            let (ptr, len) = if blob.is_empty() {
                (std::ptr::null(), 0)
            } else {
                (blob.as_ptr(), blob.len())
            };
            unsafe {
                *out_infos.add(i) = PageInfoABI {
                    page_id: info.page_id.0,
                    size: info.size,
                    location: info.location as i32,
                    epoch: info.epoch,
                    metadata_ptr: ptr,
                    metadata_len: len,
                };
            }
        }
    });
    count
}
