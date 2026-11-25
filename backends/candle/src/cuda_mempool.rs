#[cfg(feature = "cuda")]
use std::ffi::{c_int, c_void};
#[cfg(feature = "cuda")]
use std::mem::MaybeUninit;

#[cfg(feature = "cuda")]
use anyhow::Context;
#[cfg(feature = "cuda")]
use candle::cuda_backend::cudarc::driver;
#[cfg(feature = "cuda")]
use candle::cuda_backend::cudarc::driver::sys::{
    self, CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED,
    CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
};

const RELEASE_THRESHOLD_ENV: &str = "CUDA_MEMPOOL_RELEASE_THRESHOLD_MB";
const MIB: u64 = 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReleaseThreshold {
    DriverDefault,
    Bytes(u64),
    Unlimited,
}

impl ReleaseThreshold {
    fn from_env_value(raw: Option<&str>) -> Self {
        let Some(raw) = raw.map(str::trim).filter(|s| !s.is_empty()) else {
            return Self::Unlimited;
        };

        if raw.eq_ignore_ascii_case("unlimited") || raw.eq_ignore_ascii_case("max") {
            return Self::Unlimited;
        }

        match raw.parse::<u64>() {
            Ok(0) => Self::DriverDefault,
            Ok(mib) => Self::Bytes(mib.saturating_mul(MIB)),
            Err(_) => {
                tracing::warn!(
                    "Invalid {RELEASE_THRESHOLD_ENV}={raw:?}; expected a non-negative integer in MiB or \"unlimited\", falling back to unlimited"
                );
                Self::Unlimited
            }
        }
    }

    fn as_bytes(self) -> Option<u64> {
        match self {
            Self::DriverDefault => None,
            Self::Bytes(bytes) => Some(bytes),
            Self::Unlimited => Some(u64::MAX),
        }
    }
}

/// Set `CU_MEMPOOL_ATTR_RELEASE_THRESHOLD` on the CUDA device's default memory pool.
///
/// Controlled by `CUDA_MEMPOOL_RELEASE_THRESHOLD_MB`: unset/`"unlimited"`/`"max"` sets
/// `u64::MAX` (retaining only already-allocated buffers without pre-allocating VRAM),
/// `"0"` keeps CUDA's driver default, and a positive integer caps retention in MiB.
#[cfg(feature = "cuda")]
pub fn configure_default_mempool(ordinal: usize) {
    let threshold = match std::env::var(RELEASE_THRESHOLD_ENV) {
        Ok(raw) => ReleaseThreshold::from_env_value(Some(&raw)),
        Err(std::env::VarError::NotPresent) => ReleaseThreshold::Unlimited,
        Err(std::env::VarError::NotUnicode(_)) => {
            tracing::warn!(
                "Invalid non-UTF-8 value for {RELEASE_THRESHOLD_ENV}; falling back to unlimited"
            );
            ReleaseThreshold::Unlimited
        }
    };

    let Some(threshold_bytes) = threshold.as_bytes() else {
        tracing::info!(
            "Keeping CUDA default memory pool release threshold at 0 (driver default) due to {RELEASE_THRESHOLD_ENV}=0"
        );
        return;
    };

    if let Err(err) = set_default_mempool_release_threshold(ordinal, threshold_bytes) {
        tracing::warn!(
            "Failed to configure CUDA default memory pool release threshold on device {ordinal}: {err:?}"
        );
    }
}

#[cfg(feature = "cuda")]
fn set_default_mempool_release_threshold(
    ordinal: usize,
    threshold_bytes: u64,
) -> Result<(), anyhow::Error> {
    driver::result::init().context("cuInit failed")?;
    let ordinal = c_int::try_from(ordinal).context("CUDA device ordinal overflows c_int")?;
    let cu_dev = driver::result::device::get(ordinal).context("cuDeviceGet failed")?;

    let supported = unsafe {
        driver::result::device::get_attribute(cu_dev, CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED)
    }
    .context("cuDeviceGetAttribute(MEMORY_POOLS_SUPPORTED) failed")?;

    if supported == 0 {
        tracing::debug!(
            "CUDA device {ordinal} does not support memory pools; skipping release threshold configuration"
        );
        return Ok(());
    }

    let mut pool = MaybeUninit::uninit();
    unsafe {
        sys::cuDeviceGetDefaultMemPool(pool.as_mut_ptr(), cu_dev)
            .result()
            .context("cuDeviceGetDefaultMemPool failed")?;
    }
    let pool = unsafe { pool.assume_init() };

    let mut value = threshold_bytes;
    unsafe {
        sys::cuMemPoolSetAttribute(
            pool,
            CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
            (&mut value as *mut u64).cast::<c_void>(),
        )
        .result()
        .context("cuMemPoolSetAttribute(RELEASE_THRESHOLD) failed")?;
    }

    if threshold_bytes == u64::MAX {
        tracing::info!(
            "Configured CUDA device {ordinal} default memory pool release threshold to unlimited"
        );
    } else {
        tracing::info!(
            "Configured CUDA device {ordinal} default memory pool release threshold to {} MiB ({threshold_bytes} bytes)",
            threshold_bytes / MIB
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_release_threshold() {
        assert_eq!(
            ReleaseThreshold::from_env_value(None),
            ReleaseThreshold::Unlimited
        );
        assert_eq!(
            ReleaseThreshold::from_env_value(Some("")),
            ReleaseThreshold::Unlimited
        );
        assert_eq!(
            ReleaseThreshold::from_env_value(Some("   ")),
            ReleaseThreshold::Unlimited
        );
        assert_eq!(
            ReleaseThreshold::from_env_value(Some("unlimited")),
            ReleaseThreshold::Unlimited
        );
        assert_eq!(
            ReleaseThreshold::from_env_value(Some("MAX")),
            ReleaseThreshold::Unlimited
        );
        assert_eq!(
            ReleaseThreshold::from_env_value(Some("0")),
            ReleaseThreshold::DriverDefault
        );
        assert_eq!(
            ReleaseThreshold::from_env_value(Some(" 16384 ")),
            ReleaseThreshold::Bytes(16384 * MIB)
        );
        assert_eq!(
            ReleaseThreshold::from_env_value(Some("-1")),
            ReleaseThreshold::Unlimited
        );
        assert_eq!(
            ReleaseThreshold::from_env_value(Some("invalid")),
            ReleaseThreshold::Unlimited
        );
        assert_eq!(
            ReleaseThreshold::from_env_value(Some("18446744073709551615")),
            ReleaseThreshold::Bytes(u64::MAX)
        );
        assert_eq!(ReleaseThreshold::DriverDefault.as_bytes(), None);
        assert_eq!(ReleaseThreshold::Unlimited.as_bytes(), Some(u64::MAX));
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_set_and_readback_default_mempool_release_threshold() -> Result<(), anyhow::Error> {
        if !candle::utils::cuda_is_available() {
            return Ok(());
        }

        driver::result::init().context("cuInit failed")?;
        let cu_dev = driver::result::device::get(0).context("cuDeviceGet failed")?;
        let supported = unsafe {
            driver::result::device::get_attribute(
                cu_dev,
                CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED,
            )
        }
        .context("cuDeviceGetAttribute failed")?;
        if supported == 0 {
            return Ok(());
        }

        let mut pool = MaybeUninit::uninit();
        unsafe {
            sys::cuDeviceGetDefaultMemPool(pool.as_mut_ptr(), cu_dev)
                .result()
                .context("cuDeviceGetDefaultMemPool failed")?;
        }
        let pool = unsafe { pool.assume_init() };

        let read_threshold = || -> Result<u64, anyhow::Error> {
            let mut val: u64 = 0;
            unsafe {
                sys::cuMemPoolGetAttribute(
                    pool,
                    CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                    (&mut val as *mut u64).cast::<c_void>(),
                )
                .result()
                .context("cuMemPoolGetAttribute failed")?;
            }
            Ok(val)
        };

        let original = read_threshold()?;

        struct RestoreGuard {
            pool: sys::CUmemoryPool,
            original: u64,
        }
        impl Drop for RestoreGuard {
            fn drop(&mut self) {
                let mut val = self.original;
                let _ = unsafe {
                    sys::cuMemPoolSetAttribute(
                        self.pool,
                        CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                        (&mut val as *mut u64).cast::<c_void>(),
                    )
                };
            }
        }
        let _guard = RestoreGuard { pool, original };

        let target = 64 * MIB;
        set_default_mempool_release_threshold(0, target)?;
        assert_eq!(read_threshold()?, target);

        set_default_mempool_release_threshold(0, u64::MAX)?;
        assert_eq!(read_threshold()?, u64::MAX);

        Ok(())
    }
}
