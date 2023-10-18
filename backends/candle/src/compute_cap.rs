use candle::cuda_backend::cudarc::driver::sys::CUdevice_attribute::{
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
};
use candle::cuda_backend::cudarc::driver::CudaDevice;
use lazy_static::lazy_static;

lazy_static! {
    pub static ref RUNTIME_COMPUTE_CAP: usize = {
        let device = CudaDevice::new(0).expect("cuda is not available");
        let major = device
            .attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
            .unwrap();
        let minor = device
            .attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
            .unwrap();
        (major * 10 + minor) as usize
    };
    pub static ref COMPILE_COMPUTE_CAP: usize = env!("CUDA_COMPUTE_CAP").parse::<usize>().unwrap();
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

pub fn incompatible_compute_cap() -> bool {
    let compile_compute_cap = *COMPILE_COMPUTE_CAP;
    let runtime_compute_cap = *RUNTIME_COMPUTE_CAP;
    !compute_cap_matching(runtime_compute_cap, compile_compute_cap)
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