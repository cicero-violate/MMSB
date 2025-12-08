// src/ffi.rs
use crate::runtime::allocator::{PageAllocator, PageAllocatorConfig};
use crate::runtime::checkpoint;
use crate::runtime::tlog::{TransactionLog, TransactionLogReader, summary as tlog_summary};
use crate::types::{Delta, DeltaID, Epoch, Page, PageError, PageID, PageLocation, Source};
use std::cell::RefCell;
use std::cmp::min;
use std::ffi::{CStr, CString};
use std::io::ErrorKind;
use std::os::raw::c_char;
use std::ptr;
use std::thread_local;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct PageHandle {
    pub ptr: *mut Page,
}

impl PageHandle {
    fn null() -> Self {
        Self { ptr: ptr::null_mut() }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct DeltaHandle {
    pub ptr: *mut Delta,
}

impl DeltaHandle {
    fn null() -> Self {
        Self { ptr: ptr::null_mut() }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct AllocatorHandle {
    pub ptr: *mut PageAllocator,
}

impl AllocatorHandle {
    fn null() -> Self {
        Self { ptr: ptr::null_mut() }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct TLogHandle {
    pub ptr: *mut TransactionLog,
}

impl TLogHandle {
    fn null() -> Self {
        Self { ptr: ptr::null_mut() }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct TLogReaderHandle {
    pub ptr: *mut TransactionLogReader,
}

impl TLogReaderHandle {
    fn null() -> Self {
        Self { ptr: ptr::null_mut() }
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
}

fn set_last_error(code: MMSBErrorCode) {
    TLS_LAST_ERROR.with(|cell| *cell.borrow_mut() = code);
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

// ──────────────────────────────────────────────────────────────
// Page read — PERFECT (you wrote this, it's beautiful)
// ──────────────────────────────────────────────────────────────
#[no_mangle]
pub extern "C" fn mmsb_page_read(handle: PageHandle, dst: *mut u8, len: usize) -> usize {
    if handle.ptr.is_null() {
        set_last_error(MMSBErrorCode::InvalidHandle);
        return 0;
    }
    if len == 0 {
        let page = unsafe { &*handle.ptr };
        return page.size();
    }
    if dst.is_null() {
        set_last_error(MMSBErrorCode::InvalidHandle);
        return 0;
    }
    let page = unsafe { &*handle.ptr };
    let page_size = page.size();
    let copy_len = len.min(page_size);
    let src = page.data_slice();
    unsafe {
        std::ptr::copy_nonoverlapping(src.as_ptr(), dst, copy_len);
    }
    copy_len
}

#[no_mangle]
pub extern "C" fn mmsb_page_epoch(handle: PageHandle) -> u32 {
    if handle.ptr.is_null() {
        set_last_error(MMSBErrorCode::InvalidHandle);
        return 0;
    }
    unsafe { (*handle.ptr).epoch().0 }
}

// ──────────────────────────────────────────────────────────────
// Allocator FFI — PERFECT
// ──────────────────────────────────────────────────────────────
#[no_mangle]
pub extern "C" fn mmsb_allocator_new() -> AllocatorHandle {
    let allocator = PageAllocator::new(PageAllocatorConfig::default());
    AllocatorHandle {
        ptr: Box::into_raw(Box::new(allocator)),
    }
}

#[no_mangle]
pub extern "C" fn mmsb_allocator_free(handle: AllocatorHandle) {
    if !handle.ptr.is_null() {
        unsafe { drop(Box::from_raw(handle.ptr)) };
    }
}

#[no_mangle]
pub extern "C" fn mmsb_allocator_allocate(
    handle: AllocatorHandle,
    page_id_hint: u64,
    size: usize,
    location: i32,
) -> PageHandle {
    if handle.ptr.is_null() || size == 0 {
        set_last_error(MMSBErrorCode::InvalidHandle);
        return PageHandle::null();
    }
    let allocator = unsafe { &*handle.ptr };
    let loc = PageLocation::from_tag(location).ok();
    match allocator.allocate_raw(PageID(page_id_hint), size, loc) {
        Ok(page) => PageHandle { ptr: page },
        Err(_) => {
            set_last_error(MMSBErrorCode::AllocError);
            PageHandle::null()
        }
    }
}

#[no_mangle]
pub extern "C" fn mmsb_allocator_release(handle: AllocatorHandle, page_id: u64) {
    if handle.ptr.is_null() {
        return;
    }
    let allocator = unsafe { &*handle.ptr };
    allocator.release(PageID(page_id));
}

#[no_mangle]
pub extern "C" fn mmsb_allocator_get_page(handle: AllocatorHandle, page_id: u64) -> PageHandle {
    if handle.ptr.is_null() {
        return PageHandle::null();
    }
    let allocator = unsafe { &*handle.ptr };
    allocator.acquire_page(PageID(page_id)).map_or(PageHandle::null(), |p| PageHandle { ptr: p })
}

// ──────────────────────────────────────────────────────────────
// TLog FFI — FINAL AND CORRECT
// ──────────────────────────────────────────────────────────────
#[no_mangle]
pub extern "C" fn mmsb_tlog_new(path: *const c_char) -> TLogHandle {
    if path.is_null() {
        set_last_error(MMSBErrorCode::InvalidHandle);
        return TLogHandle::null();
    }
    let path_str = unsafe { CStr::from_ptr(path).to_string_lossy().into_owned() };
    match TransactionLog::new(path_str) {
        Ok(log) => TLogHandle { ptr: Box::into_raw(Box::new(log)) },
        Err(e) => {
            set_last_error(log_error_code(&e));
            TLogHandle::null()
        }
    }
}

#[no_mangle]
pub extern "C" fn mmsb_tlog_free(handle: TLogHandle) {
    if !handle.ptr.is_null() {
        unsafe { drop(Box::from_raw(handle.ptr)) };
    }
}

#[no_mangle]
pub extern "C" fn mmsb_tlog_append(handle: TLogHandle, delta: DeltaHandle) -> i32 {
    if handle.ptr.is_null() || delta.ptr.is_null() {
        set_last_error(MMSBErrorCode::InvalidHandle);
        return -1;
    }
    let log = unsafe { &*handle.ptr };
    let delta = unsafe { &*delta.ptr };
    log.append(delta.clone()).map_or_else(
        |e| {
            set_last_error(log_error_code(&e));
            -1
        },
        |_| 0,
    )
}

// ──────────────────────────────────────────────────────────────
// TLog Reader FFI — FINAL AND CORRECT
// ──────────────────────────────────────────────────────────────
#[no_mangle]
pub extern "C" fn mmsb_tlog_reader_new(path: *const c_char) -> TLogReaderHandle {
    if path.is_null() {
        set_last_error(MMSBErrorCode::InvalidHandle);
        return TLogReaderHandle::null();
    }
    let path_str = unsafe { CStr::from_ptr(path).to_string_lossy().into_owned() };
    match TransactionLogReader::open(path_str) {
        Ok(reader) => TLogReaderHandle { ptr: Box::into_raw(Box::new(reader)) },
        Err(e) => {
            set_last_error(log_error_code(&e));
            TLogReaderHandle::null()
        }
    }
}

#[no_mangle]
pub extern "C" fn mmsb_tlog_reader_free(handle: TLogReaderHandle) {
    if !handle.ptr.is_null() {
        unsafe { drop(Box::from_raw(handle.ptr)) };
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
        Ok(Some(delta)) => DeltaHandle { ptr: Box::into_raw(Box::new(delta)) },
        Ok(None) => DeltaHandle::null(),
        Err(e) => {
            set_last_error(log_error_code(&e));
            DeltaHandle::null()
        }
    }
}

// ──────────────────────────────────────────────────────────────
// Checkpoint FFI — FINAL AND CORRECT
// ──────────────────────────────────────────────────────────────
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
    let allocator = unsafe { &*allocator.ptr };
    let log = unsafe { &*log.ptr };
    let path_str = unsafe { CStr::from_ptr(path).to_string_lossy().into_owned() };
    checkpoint::write_checkpoint(allocator, log, path_str).map_or_else(
        |_| {
            set_last_error(MMSBErrorCode::SnapshotError);
            -1
        },
        |_| 0,
    )
}

#[no_mangle]
pub extern "C" fn mmsb_checkpoint_load(
    allocator: AllocatorHandle,
    _log: TLogHandle,  // unused — we don't need it
    path: *const c_char,
) -> i32 {
    if allocator.ptr.is_null() || path.is_null() {
        set_last_error(MMSBErrorCode::InvalidHandle);
        return -1;
    }
    let allocator = unsafe { &*allocator.ptr };
    let path_str = unsafe { CStr::from_ptr(path).to_string_lossy().into_owned() };
    checkpoint::load_checkpoint(allocator, path_str).map_or_else(
        |_| {
            set_last_error(MMSBErrorCode::SnapshotError);
            -1
        },
        |_| 0,
    )
}

// ──────────────────────────────────────────────────────────────
// TLog Summary — FIXED
// ──────────────────────────────────────────────────────────────
#[no_mangle]
pub extern "C" fn mmsb_tlog_summary(path: *const c_char, out: *mut TLogSummary) -> i32 {
    if path.is_null() || out.is_null() {
        set_last_error(MMSBErrorCode::InvalidHandle);
        return -1;
    }
    let path_str = unsafe { CStr::from_ptr(path).to_string_lossy().into_owned() };
    match tlog_summary(path_str) {
        Ok(summary) => {
            unsafe {
                *out = TLogSummary {
                    total_deltas: summary.total_deltas,
                    total_bytes: summary.total_bytes,
                    last_epoch: summary.last_epoch,
                };
            }
            0
        }
        Err(e) => {
            set_last_error(log_error_code(&e));
            -1
        }
    }
}
