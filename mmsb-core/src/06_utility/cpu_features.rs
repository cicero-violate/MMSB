use std::sync::OnceLock;

#[derive(Debug, Clone, Copy)]
pub struct CpuFeatures {
    pub avx2: bool,
    pub avx512f: bool,
    pub sse42: bool,
    pub bmi2: bool,
}

static CPU_FEATURES: OnceLock<CpuFeatures> = OnceLock::new();

impl CpuFeatures {
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            Self {
                avx2: is_x86_feature_detected!("avx2"),
                avx512f: is_x86_feature_detected!("avx512f"),
                sse42: is_x86_feature_detected!("sse4.2"),
                bmi2: is_x86_feature_detected!("bmi2"),
            }
        }
        
        #[cfg(not(target_arch = "x86_64"))]
        {
            Self {
                avx2: false,
                avx512f: false,
                sse42: false,
                bmi2: false,
            }
        }
    }
    
    pub fn get() -> &'static CpuFeatures {
        CPU_FEATURES.get_or_init(Self::detect)
    }
}

#[no_mangle]
pub extern "C" fn cpu_has_avx2() -> bool {
    CpuFeatures::get().avx2
}

#[no_mangle]
pub extern "C" fn cpu_has_avx512() -> bool {
    CpuFeatures::get().avx512f
}

#[no_mangle]
pub extern "C" fn cpu_has_sse42() -> bool {
    CpuFeatures::get().sse42
}
