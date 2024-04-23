use anyhow::Context;
use candle::cuda_backend::cudarc::driver;
use candle::cuda_backend::cudarc::driver::sys::CUdevice_attribute::{
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
};
use candle::cuda_backend::cudarc::driver::CudaDevice;

pub fn get_compile_compute_cap() -> Result<usize, anyhow::Error> {
    env!("CUDA_COMPUTE_CAP")
        .parse::<usize>()
        .context("Could not retrieve compile time CUDA_COMPUTE_CAP")
}

pub fn get_runtime_compute_cap() -> Result<usize, anyhow::Error> {
    driver::result::init().context("CUDA is not available")?;
    let device = CudaDevice::new(0).context("CUDA is not available")?;
    let major = device
        .attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
        .context("Could not retrieve device compute capability major")?;
    let minor = device
        .attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
        .context("Could not retrieve device compute capability minor")?;
    Ok((major * 10 + minor) as usize)
}

fn compute_cap_matching(runtime_compute_cap: usize, compile_compute_cap: usize) -> bool {
    match (runtime_compute_cap, compile_compute_cap) {
        (75, 75) => true,
        (80..=89, 80) => true,
        (86..=89, 80..=86) => true,
        (89, 89) => true,
        (90, 90) => true,
        (_, _) => false,
    }
}

pub fn compatible_compute_cap() -> Result<bool, anyhow::Error> {
    let compile_compute_cap = get_compile_compute_cap()?;
    let runtime_compute_cap = get_runtime_compute_cap()?;
    Ok(compute_cap_matching(
        runtime_compute_cap,
        compile_compute_cap,
    ))
}

#[cfg(test)]
mod tests {
    use crate::compute_cap::compute_cap_matching;

    #[test]
    fn test_compute_cap() {
        assert!(compute_cap_matching(75, 75));
        assert!(compute_cap_matching(80, 80));
        assert!(compute_cap_matching(86, 86));
        assert!(compute_cap_matching(89, 89));
        assert!(compute_cap_matching(90, 90));

        assert!(compute_cap_matching(86, 80));
        assert!(compute_cap_matching(89, 80));
        assert!(compute_cap_matching(89, 86));

        assert!(!compute_cap_matching(75, 80));
        assert!(!compute_cap_matching(75, 86));
        assert!(!compute_cap_matching(75, 89));
        assert!(!compute_cap_matching(75, 90));

        assert!(!compute_cap_matching(80, 75));
        assert!(!compute_cap_matching(80, 86));
        assert!(!compute_cap_matching(80, 89));
        assert!(!compute_cap_matching(80, 90));

        assert!(!compute_cap_matching(86, 75));
        assert!(!compute_cap_matching(86, 89));
        assert!(!compute_cap_matching(86, 90));

        assert!(!compute_cap_matching(89, 75));
        assert!(!compute_cap_matching(89, 90));

        assert!(!compute_cap_matching(90, 75));
        assert!(!compute_cap_matching(90, 80));
        assert!(!compute_cap_matching(90, 86));
        assert!(!compute_cap_matching(90, 89));
    }
}
