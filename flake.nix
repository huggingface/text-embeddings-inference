/*
## CPU

```bash
nix develop
cargo build --release
```
## GPU

```bash
nix develop .#cuda
# -F candle-cuda for GPU
# -F dynamic-linking or -F static-linking should be set explicitly because of cudarc
# --no-default-features prevent Cargo from compiling unnecessary, heavy backends
# -F http because of previous flag we need to state it explicitely now
cargo build --release -F candle-cuda -F http -F static-linking --no-default-features
# or
cargo build --release -F candle-cuda -F http -F dynamic-linking --no-default-features
```

## Running with Ubuntu libraries example

```bash
# 1. Create a hidden folder in your current directory
mkdir -p .nvidia-shims
# 2. Symlink only the NVIDIA driver libraries from your host
ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so* .nvidia-shims/
ln -sf /usr/lib/x86_64-linux-gnu/libnvidia-*.so* .nvidia-shims/
# 3. Run
LD_LIBRARY_PATH=$(pwd)/.nvidia-shims ./target/release/text-embeddings-router --port 8888 --model-id nomic-ai/nomic-embed-text-v2-moe
```
*/

{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    { nixpkgs, ... }:
    let
      forAllSystems = nixpkgs.lib.genAttrs [
        "aarch64-linux"
        "x86_64-linux"
        "aarch64-darwin"
      ];
    in
    {
      devShells = forAllSystems (
        system:
        let
          isLinux = nixpkgs.lib.hasSuffix "-linux" system;

          # 1. Base package set (CPU-only)
          pkgs = import nixpkgs { inherit system; };

          # 2. CUDA package set (Linux only)
          pkgsCuda = if isLinux then import nixpkgs {
            inherit system;
            config = {
              allowUnfree = true;
              cudaSupport = true;
            };
          } else pkgs;

          cudaPackages = pkgsCuda.cudaPackages_12_8;

          # 3. Create a unified root containing both shared and static library binaries
          cuda-root = if isLinux then pkgsCuda.symlinkJoin {
            name = "cuda-root";
            paths = [
              cudaPackages.cuda_nvcc
              cudaPackages.cuda_cudart
              (pkgsCuda.lib.getStatic cudaPackages.cuda_cudart) # <-- Safely resolves static files
              cudaPackages.cuda_nvrtc
              (pkgsCuda.lib.getStatic cudaPackages.cuda_nvrtc)  # <-- Safely resolves static files
              cudaPackages.cccl
              cudaPackages.libcurand
              (pkgsCuda.lib.getStatic cudaPackages.libcurand)  # <-- Safely resolves static files
              cudaPackages.libcublas
              (pkgsCuda.lib.getStatic cudaPackages.libcublas)  # <-- Safely resolves static files
              cudaPackages.libcusparse
              (pkgsCuda.lib.getStatic cudaPackages.libcusparse) # <-- Safely resolves static files
            ];
          } else null;

          # 4. Share common shell configurations
          commonShellAttrs = {
            venvDir = "./.venv";
            CFLAGS = "-Wno-error=old-style-definition -Wno-error=implicit-function-declaration";
            RUSTFLAGS = "-C default-linker-libraries=yes";
            postVenvCreation = ''
              unset SOURCE_DATE_EPOCH
            '';
            postShellHook = ''
              unset SOURCE_DATE_EPOCH
            '';
          };
        in
        {
          # --- SHELL 1: CPU-only ---
          default = pkgs.mkShell (commonShellAttrs // {
            buildInputs = with pkgs; [
              rustup
              protobuf
              openssl
              gcc
              pkg-config
            ] ++ pkgs.lib.optionals pkgs.stdenv.isDarwin [
              apple-sdk_15
            ];
          });

          # --- SHELL 2: CUDA-enabled ---
          cuda = (pkgsCuda.mkShell.override { stdenv = cudaPackages.backendStdenv; }) (commonShellAttrs // {
            buildInputs = with pkgsCuda; [
              rustup
              protobuf
              openssl
              pkg-config
            ] 
            ++ (if isLinux then [
              cudaPackages.cuda_nvcc
              cudaPackages.cuda_cudart
              cudaPackages.cuda_nvrtc
              cudaPackages.cccl
              cudaPackages.libcurand
              cudaPackages.libcublas
              cudaPackages.libcusparse
              cudaPackages.cudnn
            ] else [])
            ++ pkgsCuda.lib.optionals pkgsCuda.stdenv.isDarwin [
              apple-sdk_15
            ];

            shellHook = pkgsCuda.lib.optionalString isLinux ''
              export CUDA_PATH="${cuda-root}"
              export CUDA_ROOT="${cuda-root}"
              export CUDA_INCLUDE_DIR="${cuda-root}/include"
            '';
          });
        }
      );
    };
}
