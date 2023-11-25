use std::error::Error;
use vergen::EmitBuilder;

fn main() -> Result<(), Box<dyn Error>> {
    // Try to get the git sha from the local git repository
    if EmitBuilder::builder()
        .fail_on_error()
        .git_sha(false)
        .emit()
        .is_err()
    {
        // Unable to get the git sha
        if let Ok(sha) = std::env::var("GIT_SHA") {
            // Set it from an env var
            println!("cargo:rustc-env=VERGEN_GIT_SHA={sha}");
        }
    }

    // Set docker label if present
    if let Ok(label) = std::env::var("DOCKER_LABEL") {
        // Set it from an env var
        println!("cargo:rustc-env=DOCKER_LABEL={label}");
    }

    #[cfg(feature = "grpc")]
    {
        use std::env;
        use std::fs;
        use std::path::PathBuf;

        fs::create_dir("src/grpc/pb").unwrap_or(());

        let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
        tonic_build::configure()
            .build_client(false)
            .build_server(true)
            .file_descriptor_set_path(out_dir.join("descriptor.bin"))
            .out_dir("src/grpc/pb")
            .include_file("mod.rs")
            .compile(&["../proto/tei.proto"], &["../proto"])
            .unwrap_or_else(|e| panic!("protobuf compilation failed: {}", e));
    }

    Ok(())
}
