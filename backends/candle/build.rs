use anyhow::{bail, Context, Result};

fn main() {
    println!("cargo:rerun-if-env-changed=CUDA_COMPUTE_CAP");
    if let Ok(compute_cap) = set_compute_cap() {
        println!("cargo:rustc-env=CUDA_COMPUTE_CAP={compute_cap}");
    }
}

fn set_compute_cap() -> Result<usize> {
    // Try to parse compute caps from env
    let compute_cap = if let Ok(compute_cap_str) = std::env::var("CUDA_COMPUTE_CAP") {
        compute_cap_str
            .parse::<usize>()
            .context("Could not parse code")?
    } else {
        // Use nvidia-smi to get the current compute cap
        let out = std::process::Command::new("nvidia-smi")
            .arg("--query-gpu=compute_cap")
            .arg("--format=csv")
            .output()
            .context("`nvidia-smi` failed. Ensure that you have CUDA installed and that `nvidia-smi` is in your PATH.")?;
        let out = std::str::from_utf8(&out.stdout).context("stdout is not a utf8 string")?;
        let mut lines = out.lines();
        if lines.next().context("missing line in stdout")? != "compute_cap" {
            bail!("First line should be `compute_cap`");
        }
        let cap = lines
            .next()
            .context("missing line in stdout")?
            .replace('.', "");
        cap.parse::<usize>().context("cannot parse as int")?
    };
    Ok(compute_cap)
}
