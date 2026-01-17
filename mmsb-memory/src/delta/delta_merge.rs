use crate::page::{Delta, DeltaError};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Merge deltas using SIMD when available
pub fn merge_deltas(first: &Delta, second: &Delta) -> Result<Delta, DeltaError> {
    first.merge(second)
}

/// SIMD-optimized dense delta merge using AVX2
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn merge_dense_avx2(
    data_a: &[u8],
    mask_a: &[bool],
    data_b: &[u8],
    mask_b: &[bool],
    out_data: &mut [u8],
    out_mask: &mut [bool],
) {
    let len = data_a.len().min(data_b.len());
    let mut i = 0;

    // Process 32 bytes at a time with AVX2
    while i + 32 <= len {
        let va = _mm256_loadu_si256(data_a.as_ptr().add(i) as *const __m256i);
        let vb = _mm256_loadu_si256(data_b.as_ptr().add(i) as *const __m256i);
        
        // Load masks (convert bool to byte for SIMD)
        let mut mask_a_bytes = [0u8; 32];
        let mut mask_b_bytes = [0u8; 32];
        for j in 0..32 {
            mask_a_bytes[j] = mask_a[i + j] as u8 * 0xFF;
            mask_b_bytes[j] = mask_b[i + j] as u8 * 0xFF;
        }
        
        let ma = _mm256_loadu_si256(mask_a_bytes.as_ptr() as *const __m256i);
        let mb = _mm256_loadu_si256(mask_b_bytes.as_ptr() as *const __m256i);
        
        // Select: where mask_b is set, take data_b, else take data_a
        let result = _mm256_blendv_epi8(va, vb, mb);
        _mm256_storeu_si256(out_data.as_mut_ptr().add(i) as *mut __m256i, result);
        
        // Merge masks with OR
        let mask_result = _mm256_or_si256(ma, mb);
        let mut mask_out_bytes = [0u8; 32];
        _mm256_storeu_si256(mask_out_bytes.as_mut_ptr() as *mut __m256i, mask_result);
        
        for j in 0..32 {
            out_mask[i + j] = mask_out_bytes[j] != 0;
        }
        
        i += 32;
    }

    // Scalar fallback for remainder
    while i < len {
        if mask_b[i] {
            out_data[i] = data_b[i];
            out_mask[i] = true;
        } else {
            out_data[i] = data_a[i];
            out_mask[i] = mask_a[i];
        }
        i += 1;
    }
}

/// SIMD-optimized dense delta merge using AVX-512
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn merge_dense_avx512(
    data_a: &[u8],
    mask_a: &[bool],
    data_b: &[u8],
    mask_b: &[bool],
    out_data: &mut [u8],
    out_mask: &mut [bool],
) {
    let len = data_a.len().min(data_b.len());
    let mut i = 0;

    // Process 64 bytes at a time with AVX-512
    while i + 64 <= len {
        let va = _mm512_loadu_si512(data_a.as_ptr().add(i) as *const __m512i);
        let vb = _mm512_loadu_si512(data_b.as_ptr().add(i) as *const __m512i);
        
        // Load masks
        let mut mask_b_bytes = [0u8; 64];
        for j in 0..64 {
            mask_b_bytes[j] = mask_b[i + j] as u8 * 0xFF;
        }
        
        let mb = _mm512_loadu_si512(mask_b_bytes.as_ptr() as *const __m512i);
        
        // Create mask for blending
        let blend_mask = _mm512_test_epi8_mask(mb, mb);
        let result = _mm512_mask_blend_epi8(blend_mask, va, vb);
        _mm512_storeu_si512(out_data.as_mut_ptr().add(i) as *mut __m512i, result);
        
        // Update output masks
        for j in 0..64 {
            out_mask[i + j] = mask_a[i + j] || mask_b[i + j];
        }
        
        i += 64;
    }

    // Scalar fallback
    while i < len {
        if mask_b[i] {
            out_data[i] = data_b[i];
            out_mask[i] = true;
        } else {
            out_data[i] = data_a[i];
            out_mask[i] = mask_a[i];
        }
        i += 1;
    }
}

/// Dispatch to appropriate SIMD implementation
pub fn merge_dense_simd(
    data_a: &[u8],
    mask_a: &[bool],
    data_b: &[u8],
    mask_b: &[bool],
    out_data: &mut [u8],
    out_mask: &mut [bool],
) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            unsafe {
                merge_dense_avx512(data_a, mask_a, data_b, mask_b, out_data, out_mask);
            }
            return;
        }
        
        if is_x86_feature_detected!("avx2") {
            unsafe {
                merge_dense_avx2(data_a, mask_a, data_b, mask_b, out_data, out_mask);
            }
            return;
        }
    }
    
    // Scalar fallback
    let len = data_a.len().min(data_b.len());
    for i in 0..len {
        if mask_b[i] {
            out_data[i] = data_b[i];
            out_mask[i] = true;
        } else {
            out_data[i] = data_a[i];
            out_mask[i] = mask_a[i];
        }
    }
}
use crate::delta::delta::Delta;
use crate::types::DeltaError;
