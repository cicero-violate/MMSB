#ifndef MMSB_H
#define MMSB_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    void *ptr;
} PageHandle;

typedef struct {
    void *ptr;
} DeltaHandle;

typedef struct {
    void *ptr;
} AllocatorHandle;

typedef struct {
    void *ptr;
} TLogHandle;

typedef struct {
    uint32_t value;
} Epoch;

typedef struct {
    uint64_t page_id;
    size_t size;
    int32_t location;
    uint32_t epoch;
    const uint8_t *metadata_ptr;
    size_t metadata_len;
} PageInfo;

typedef enum {
    MMSB_ERROR_OK = 0,
    MMSB_ERROR_ALLOC = 1,
    MMSB_ERROR_IO = 2,
    MMSB_ERROR_SNAPSHOT = 3,
    MMSB_ERROR_CORRUPT_LOG = 4,
    MMSB_ERROR_INVALID_HANDLE = 5,
} MMSBErrorCode;

MMSBErrorCode mmsb_get_last_error(void);

PageHandle mmsb_page_new(uint64_t page_id, size_t size, int32_t location);
void mmsb_page_free(PageHandle handle);
size_t mmsb_page_read(PageHandle handle, uint8_t *dst, size_t len);
uint32_t mmsb_page_epoch(PageHandle handle);
int32_t mmsb_page_write_masked(PageHandle handle,
                               const uint8_t *mask,
                               size_t mask_len,
                               const uint8_t *payload,
                               size_t payload_len,
                               uint8_t is_sparse,
                               Epoch epoch);
size_t mmsb_page_metadata_size(PageHandle handle);
size_t mmsb_page_metadata_export(PageHandle handle, uint8_t *dst, size_t len);
int32_t mmsb_page_metadata_import(PageHandle handle, const uint8_t *src, size_t len);

DeltaHandle mmsb_delta_new(uint64_t delta_id,
                           uint64_t page_id,
                           Epoch epoch,
                           const uint8_t *mask,
                           size_t mask_len,
                           const uint8_t *payload,
                           size_t payload_len,
                           uint8_t is_sparse,
                           const char *source);
void mmsb_delta_free(DeltaHandle handle);
int32_t mmsb_delta_apply(PageHandle page, DeltaHandle delta);
uint64_t mmsb_delta_id(DeltaHandle handle);
uint64_t mmsb_delta_page_id(DeltaHandle handle);
uint32_t mmsb_delta_epoch(DeltaHandle handle);
uint8_t mmsb_delta_is_sparse(DeltaHandle handle);
uint64_t mmsb_delta_timestamp(DeltaHandle handle);
size_t mmsb_delta_source_len(DeltaHandle handle);
size_t mmsb_delta_copy_source(DeltaHandle handle, uint8_t *dst, size_t len);
size_t mmsb_delta_mask_len(DeltaHandle handle);
size_t mmsb_delta_copy_mask(DeltaHandle handle, uint8_t *dst, size_t len);
size_t mmsb_delta_payload_len(DeltaHandle handle);
size_t mmsb_delta_copy_payload(DeltaHandle handle, uint8_t *dst, size_t len);

AllocatorHandle mmsb_allocator_new(int32_t default_location);
void mmsb_allocator_free(AllocatorHandle handle);
PageHandle mmsb_allocator_alloc(AllocatorHandle handle,
                                uint64_t page_id,
                                size_t size,
                                int32_t location);
void mmsb_allocator_release(AllocatorHandle handle, uint64_t page_id);
size_t mmsb_allocator_page_count(AllocatorHandle handle);
size_t mmsb_allocator_list_pages(AllocatorHandle handle, PageInfo *out_infos, size_t len);
PageHandle mmsb_allocator_get_page(AllocatorHandle handle, uint64_t page_id);

TLogHandle mmsb_tlog_new(const char *path);
void mmsb_tlog_free(TLogHandle handle);
int32_t mmsb_tlog_append(TLogHandle handle, DeltaHandle delta);
int32_t mmsb_checkpoint_write(AllocatorHandle allocator, TLogHandle log, const char *path);
int32_t mmsb_checkpoint_load(AllocatorHandle allocator, TLogHandle log, const char *path);

typedef struct {
    uint64_t total_deltas;
    uint64_t total_bytes;
    uint32_t last_epoch;
} TLogSummary;

typedef struct {
    void *ptr;
} TLogReaderHandle;

TLogReaderHandle mmsb_tlog_reader_new(const char *path);
void mmsb_tlog_reader_free(TLogReaderHandle handle);
DeltaHandle mmsb_tlog_reader_next(TLogReaderHandle handle);
int32_t mmsb_tlog_summary(const char *path, TLogSummary *out);

#ifdef __cplusplus
}
#endif

#endif // MMSB_H
