use lazy_static::lazy_static;

lazy_static! {
    pub static ref RUNTIME_COMPUTE_CAP: usize = {
        let out = std::process::Command::new("nvidia-smi")
            .arg("--query-gpu=compute_cap")
            .arg("--format=csv")
            .output()
            .unwrap();
        let out = std::str::from_utf8(&out.stdout).unwrap();
        let mut lines = out.lines();
        assert_eq!(lines.next().unwrap(), "compute_cap");
        let cap = lines.next().unwrap().replace('.', "");
        cap.parse::<usize>().unwrap()
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
