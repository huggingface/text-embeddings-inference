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

pub fn incompatible_compute_cap() -> bool {
    let compile_compute_cap = *COMPILE_COMPUTE_CAP;
    let runtime_compute_cap = *RUNTIME_COMPUTE_CAP;

    ((runtime_compute_cap == 75 || runtime_compute_cap == 90)
        && runtime_compute_cap != compile_compute_cap)
        || (!((80..90).contains(&runtime_compute_cap) && (80..90).contains(&compile_compute_cap)))
}
