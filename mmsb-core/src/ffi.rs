use crate::ffi_debug;
use crate::page::checkpoint;
use crate::page::tlog::{TransactionLog, TransactionLogReader};
use crate::page::{Delta, DeltaID, Epoch, Page, PageAllocator, PageAllocatorConfig, PageError, PageID, PageLocation, Source};
use crate::semiring::{
    accumulate, fold_add, fold_mul, BooleanSemiring, Semiring, TropicalSemiring,
};
use crate::utility::{MmsbAdmissionProof, MmsbExecutionProof};
use mmsb_judgment::JudgmentToken;
use serde_json::json;
use std::cell::RefCell;
use std::cmp::min;
use std::ffi::CStr;
use std::io::ErrorKind;
use std::os::raw::c_char;
use std::ptr;
use std::slice;
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
#[derive(Clone, Copy)]
pub struct JudgmentTokenHandle {
    pub ptr: *const JudgmentToken,
}

impl JudgmentTokenHandle {
    fn null() -> Self {
        Self { ptr: ptr::null() }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct AdmissionProofInput {
    pub delta_hash: *const c_char,
    pub version: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ExecutionProofInput {
    pub delta_hash: *const c_char,
    pub version: u32,
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
    GPUError = 6,
    CompressionError = 7,
    ChecksumMismatch = 8,
    TransactionConflict = 9,
    MemoryPressure = 10,
    NetworkError = 11,
}

impl MMSBErrorCode {
    pub fn is_retryable(&self) -> bool {
        matches!(self, 
            MMSBErrorCode::IOError | 
            MMSBErrorCode::NetworkError |
            MMSBErrorCode::MemoryPressure
        )
    }
    
    pub fn is_fatal(&self) -> bool {
        matches!(self,
            MMSBErrorCode::CorruptLog |
            MMSBErrorCode::ChecksumMismatch |
            MMSBErrorCode::InvalidHandle
        )
    }
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
pub extern "C" fn mmsb_error_is_retryable(code: MMSBErrorCode) -> bool {
    code.is_retryable()
}

#[no_mangle]
pub extern "C" fn mmsb_error_is_fatal(code: MMSBErrorCode) -> bool {
    code.is_fatal()
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

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct SemiringPairF64 {
    pub add: f64,
    pub mul: f64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct SemiringPairBool {
    pub add: u8,
    pub mul: u8,
}

fn mask_from_bytes(ptr: *const u8, len: usize, payload_len: usize, is_sparse: bool) -> Vec<bool> {
    if ptr.is_null() || len == 0 {
        return Vec::new();
    }
    let bytes = unsafe { std::slice::from_raw_parts(ptr, len) };

    // Legacy callers sometimes passed one byte per entry (length == payload_len).
    // Treat input as packed only when it clearly encodes bitfields:
    //  - sparse deltas always pack bits so `mask` counts entries.
    //  - dense payloads produced via `(size + 7) / 8` satisfy `len * 8 == payload_len`.
    let treat_as_packed = is_sparse || (!is_sparse && len.saturating_mul(8) == payload_len);
    if !treat_as_packed {
        return bytes.iter().map(|v| *v != 0).collect();
    }

    let mut mask = Vec::with_capacity(len.saturating_mul(8));
    for byte in bytes {
        for bit in 0..8 {
            mask.push(byte & (1 << bit) != 0);
        }
    }
    if !is_sparse && payload_len > 0 && mask.len() > payload_len {
        mask.truncate(payload_len);
    }
    mask
}

fn vec_from_ptr(ptr: *const u8, len: usize) -> Vec<u8> {
    if ptr.is_null() || len == 0 {
        return Vec::new();
    }
    unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec()
}

fn slice_from_ptr<'a, T>(ptr: *const T, len: usize) -> &'a [T] {
    if ptr.is_null() || len == 0 {
        &[]
    } else {
        unsafe { slice::from_raw_parts(ptr, len) }
    }
}

#[no_mangle]
pub extern "C" fn mmsb_page_read(handle: PageHandle, dst: *mut u8, len: usize) -> usize {
    ffi_debug!("=== mmsb_page_read START ===");
    ffi_debug!("  handle.ptr = {:p}", handle.ptr);
    ffi_debug!("  dst = {:p}", dst);
    ffi_debug!("  requested len = {}", len);

    if handle.ptr.is_null() {
        ffi_debug!("  ERROR: Invalid page handle (null)");
        set_last_error(MMSBErrorCode::InvalidHandle);
        return 0;
    }

    // Special case: len == 0 → just return page size (used by tests)
    if len == 0 {
        let page = unsafe { &*handle.ptr };
        let size = page.size();
        ffi_debug!("  Returning page size: {} (len=0 query)", size);
        return size;
    }

    // Normal case: dst must not be null
    if dst.is_null() {
        ffi_debug!("  ERROR: dst is null but len > 0");
        set_last_error(MMSBErrorCode::InvalidHandle);
        return 0;
    }

    let page = unsafe { &*handle.ptr };
    let page_size = page.size();
    let bytes_to_copy = len.min(page_size);

    ffi_debug!("  page size = {}, copying {} bytes", page_size, bytes_to_copy);

    let src_slice = page.data_slice();
    unsafe {
        std::ptr::copy_nonoverlapping(src_slice.as_ptr(), dst, bytes_to_copy);
    }

    ffi_debug!("=== mmsb_page_read END (success, {} bytes) ===", bytes_to_copy);
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
    let sparse = is_sparse != 0;
    let mask = mask_from_bytes(mask_ptr, mask_len, payload_len, sparse);
    let payload = vec_from_ptr(payload_ptr, payload_len);
    let delta = Delta {
        delta_id: DeltaID(0),
        page_id: unsafe { (*handle.ptr).id },
        epoch: epoch.into(),
        mask,
        payload,
        is_sparse: sparse,
        timestamp: 0,
        source: Source("page_write_masked".into()),
        intent_metadata: None,
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
    let sparse = is_sparse != 0;
    let mask = mask_from_bytes(mask_ptr, mask_len, payload_len, sparse);
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
        is_sparse: sparse,
        timestamp: 0,
        source: Source(source),
        intent_metadata: None,
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
pub extern "C" fn mmsb_delta_set_intent_metadata(
    handle: DeltaHandle,
    metadata_ptr: *const u8,
    metadata_len: usize,
) -> i32 {
    if handle.ptr.is_null() {
        set_last_error(MMSBErrorCode::InvalidHandle);
        return -1;
    }
    if metadata_ptr.is_null() || metadata_len == 0 {
        unsafe {
            (*handle.ptr).intent_metadata = None;
        }
        return 0;
    }
    let bytes = unsafe { std::slice::from_raw_parts(metadata_ptr, metadata_len) };
    match std::str::from_utf8(bytes) {
        Ok(value) => {
            unsafe {
                (*handle.ptr).intent_metadata = Some(value.to_string());
            }
            0
        }
        Err(_) => {
            set_last_error(MMSBErrorCode::InvalidHandle);
            -1
        }
    }
}

#[no_mangle]
pub extern "C" fn mmsb_delta_intent_metadata_len(handle: DeltaHandle) -> usize {
    if handle.ptr.is_null() {
        set_last_error(MMSBErrorCode::InvalidHandle);
        return 0;
    }
    unsafe {
        (*handle.ptr)
            .intent_metadata
            .as_ref()
            .map(|value| value.len())
            .unwrap_or(0)
    }
}

#[no_mangle]
pub extern "C" fn mmsb_delta_copy_intent_metadata(
    handle: DeltaHandle,
    dst: *mut u8,
    len: usize,
) -> usize {
    if handle.ptr.is_null() || dst.is_null() {
        set_last_error(MMSBErrorCode::InvalidHandle);
        return 0;
    }
    let metadata = unsafe { &(*handle.ptr).intent_metadata };
    let Some(value) = metadata else {
        return 0;
    };
    let bytes = value.as_bytes();
    let copy_len = min(bytes.len(), len);
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst, copy_len);
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
pub extern "C" fn mmsb_tlog_append(
    handle: TLogHandle,
    token: JudgmentTokenHandle,
    admission: AdmissionProofInput,
    execution: ExecutionProofInput,
    delta: DeltaHandle,
) -> i32 {
    if handle.ptr.is_null() || delta.ptr.is_null() || token.ptr.is_null() {
        set_last_error(MMSBErrorCode::InvalidHandle);
        return -1;
    }
    if admission.delta_hash.is_null() || execution.delta_hash.is_null() {
        set_last_error(MMSBErrorCode::InvalidHandle);
        return -1;
    }
    let admission_hash = unsafe { CStr::from_ptr(admission.delta_hash) }
        .to_str()
        .ok()
        .map(|s| s.to_string());
    let execution_hash = unsafe { CStr::from_ptr(execution.delta_hash) }
        .to_str()
        .ok()
        .map(|s| s.to_string());
    let (Some(admission_hash), Some(execution_hash)) = (admission_hash, execution_hash) else {
        set_last_error(MMSBErrorCode::InvalidHandle);
        return -1;
    };

    let log = unsafe { &*handle.ptr };
    let delta_ref = unsafe { &*delta.ptr };
    let token_ref = unsafe { &*token.ptr };

    let admission_proof = MmsbAdmissionProof {
        version: admission.version,
        delta_hash: admission_hash.clone(),
        conversation_id: "ffi".to_string(),
        message_id: "ffi".to_string(),
        suffix: "0".to_string(),
        intent_hash: admission_hash,
        approved: true,
        command: Vec::new(),
        cwd: None,
        env: None,
        epoch: 0,
    };
    let execution_proof = MmsbExecutionProof {
        version: execution.version,
        delta_hash: execution_hash,
        tool_call_id: "ffi".to_string(),
        tool_name: "ffi".to_string(),
        output: json!({}),
        epoch: 0,
    };

    match crate::propagation::tick_orchestrator::request_commit(
        log,
        token_ref,
        &admission_proof,
        &execution_proof,
        delta_ref.clone(),
    ) {
        Ok(_) => 0,
        Err(err) => {
            set_last_error(log_error_code(&err));
            -1
        }
    }
}

#[no_mangle]
pub extern "C" fn mmsb_tlog_clear_entries(handle: TLogHandle) {
    if handle.ptr.is_null() {
        set_last_error(MMSBErrorCode::InvalidHandle);
        return;
    }
    let log = unsafe { &*handle.ptr };
    log.clear_entries();
}

// src/ffi.rs — mmsb_checkpoint_write
#[no_mangle]
pub extern "C" fn mmsb_checkpoint_write(
    allocator: AllocatorHandle,
    log: TLogHandle,
    path: *const c_char,
) -> i32 {
    eprintln!("\n=== mmsb_checkpoint_write CALLED ===");
    eprintln!("   allocator.ptr = {:p}", allocator.ptr);
    eprintln!("   log.ptr       = {:p}", log.ptr);

    let path_str_display = if path.is_null() {
        "<null>"
    } else {
        match unsafe { CStr::from_ptr(path).to_str() } {
            Ok(s) => s,
            Err(_) => "<invalid utf-8>",
        }
    };
    eprintln!("   path          = {path_str_display}");

    if allocator.ptr.is_null() || log.ptr.is_null() || path.is_null() {
        eprintln!("   INVALID HANDLE → returning -1");
        set_last_error(MMSBErrorCode::InvalidHandle);
        return -1;
    }

    let allocator_ref = unsafe { &*allocator.ptr };
    let log_ref = unsafe { &*log.ptr };
    let path_str = match unsafe { CStr::from_ptr(path).to_str() } {
        Ok(s) => s.to_owned(),
        Err(_) => {
            set_last_error(MMSBErrorCode::IOError);
            return -1;
        }
    };

    eprintln!("   Calling checkpoint::write_checkpoint(...)");
    match checkpoint::write_checkpoint(allocator_ref, log_ref, path_str) {
        Ok(_) => {
            eprintln!("   checkpoint::write_checkpoint() → SUCCESS");
            0
        }
        Err(e) => {
            eprintln!("   checkpoint::write_checkpoint() → FAILED: {e:?}");
            set_last_error(MMSBErrorCode::SnapshotError);
            -1
        }
    }
}

// src/ffi.rs — mmsb_checkpoint_load
#[no_mangle]
pub extern "C" fn mmsb_checkpoint_load(
    allocator: AllocatorHandle,
    log: TLogHandle,
    path: *const c_char,
) -> i32 {
    eprintln!("\n=== mmsb_checkpoint_load CALLED ===");
    eprintln!("   allocator.ptr = {:p}", allocator.ptr);
    eprintln!("   log.ptr       = {:p}", log.ptr);

    let path_str_display = if path.is_null() {
        "<null>"
    } else {
        match unsafe { CStr::from_ptr(path).to_str() } {
            Ok(s) => s,
            Err(_) => "<invalid utf-8>",
        }
    };
    eprintln!("   path          = {path_str_display}");

    if allocator.ptr.is_null() || log.ptr.is_null() || path.is_null() {
        eprintln!("   INVALID HANDLE → returning -1");
        set_last_error(MMSBErrorCode::InvalidHandle);
        return -1;
    }

    let allocator_ref = unsafe { &*allocator.ptr };
    let log_ref = unsafe { &*log.ptr };
    let path_str = match unsafe { CStr::from_ptr(path).to_str() } {
        Ok(s) => s.to_owned(),
        Err(_) => {
            set_last_error(MMSBErrorCode::IOError);
            return -1;
        }
    };

    eprintln!("   Calling checkpoint::load_checkpoint(...)");
    match checkpoint::load_checkpoint(allocator_ref, log_ref, path_str) {
        Ok(_) => {
            eprintln!("   checkpoint::load_checkpoint() → SUCCESS");
            0
        }
        Err(e) => {
            eprintln!("   checkpoint::load_checkpoint() → FAILED: {e:?}");
            set_last_error(MMSBErrorCode::SnapshotError);
            -1
        }
    }
}

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
        Ok(s) => s,
        Err(_) => {
            set_last_error(MMSBErrorCode::IOError);
            return -1;
        }
    };

    match crate::page::tlog::summary(path_str) {
        Ok(summary) => {
            unsafe {
                (*out).total_deltas = summary.total_deltas;
                (*out).total_bytes = summary.total_bytes;
                (*out).last_epoch = summary.last_epoch;
            }
            0
        }
        Err(e) => {
            set_last_error(log_error_code(&e));
            -1
        }
    }
}

// ──────────────────────────────────────────────────────────────
//  PageAllocator FFI – complete and bulletproof
// ──────────────────────────────────────────────────────────────
#[no_mangle]
pub extern "C" fn mmsb_allocator_new() -> AllocatorHandle {
    ffi_debug!("=== mmsb_allocator_new START ===");
    let config = PageAllocatorConfig::default();
    let allocator = PageAllocator::new(config);
    let boxed = Box::new(allocator);
    let ptr = Box::into_raw(boxed);
    ffi_debug!("   Allocator created at {:p}", ptr);
    ffi_debug!("=== mmsb_allocator_new END ===");
    AllocatorHandle { ptr }
}

#[no_mangle]
pub extern "C" fn mmsb_allocator_clear(handle: AllocatorHandle) {
    if handle.ptr.is_null() {
        set_last_error(MMSBErrorCode::InvalidHandle);
        return;
    }
    ffi_debug!("Clearing allocator at {:p}", handle.ptr);
    let alloc = unsafe { &*handle.ptr };
    alloc.clear();
}

#[no_mangle]
pub extern "C" fn mmsb_allocator_free(handle: AllocatorHandle) {
    if !handle.ptr.is_null() {
        ffi_debug!("Freeing allocator at {:p}", handle.ptr);
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
    ffi_debug!("=== mmsb_allocator_allocate START ===");
    ffi_debug!("  handle = {:p}, page_id_hint = {}, size = {}", handle.ptr, page_id_hint, size);

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
                ffi_debug!("   Allocation SUCCESS → page at {:p}", page_ptr);
                ffi_debug!("=== mmsb_allocator_allocate END (success) ===");
                PageHandle { ptr: page_ptr }
            }
            Err(_) => {
                ffi_debug!("   Allocation FAILED");
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

#[no_mangle]
pub extern "C" fn mmsb_semiring_tropical_fold_add(values: *const f64, len: usize) -> f64 {
    let semiring = TropicalSemiring;
    if len == 0 {
        return semiring.zero();
    }
    let slice = slice_from_ptr(values, len);
    fold_add(&semiring, slice.iter().copied())
}

#[no_mangle]
pub extern "C" fn mmsb_semiring_tropical_fold_mul(values: *const f64, len: usize) -> f64 {
    let semiring = TropicalSemiring;
    if len == 0 {
        return semiring.one();
    }
    let slice = slice_from_ptr(values, len);
    fold_mul(&semiring, slice.iter().copied())
}

#[no_mangle]
pub extern "C" fn mmsb_semiring_tropical_accumulate(left: f64, right: f64) -> SemiringPairF64 {
    let semiring = TropicalSemiring;
    let (add, mul) = accumulate(&semiring, &left, &right);
    SemiringPairF64 { add, mul }
}

#[no_mangle]
pub extern "C" fn mmsb_semiring_boolean_fold_add(values: *const u8, len: usize) -> u8 {
    let semiring = BooleanSemiring;
    if len == 0 {
        return semiring.zero() as u8;
    }
    let slice = slice_from_ptr(values, len);
    let iter = slice.iter().map(|v| *v != 0);
    let result = fold_add(&semiring, iter);
    result as u8
}

#[no_mangle]
pub extern "C" fn mmsb_semiring_boolean_fold_mul(values: *const u8, len: usize) -> u8 {
    let semiring = BooleanSemiring;
    if len == 0 {
        return semiring.one() as u8;
    }
    let slice = slice_from_ptr(values, len);
    let iter = slice.iter().map(|v| *v != 0);
    let result = fold_mul(&semiring, iter);
    result as u8
}

#[no_mangle]
pub extern "C" fn mmsb_semiring_boolean_accumulate(left: u8, right: u8) -> SemiringPairBool {
    let semiring = BooleanSemiring;
    let (add, mul) = accumulate(&semiring, &(left != 0), &(right != 0));
    SemiringPairBool {
        add: add as u8,
        mul: mul as u8,
    }
}
