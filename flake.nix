{
  description = "Build a cargo project";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";

    crane.url = "github:ipetkov/crane";

    flake-utils.url = "github:numtide/flake-utils";

    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      crane,
      flake-utils,
      rust-overlay,
      ...
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
          overlays = [ (import rust-overlay) ];
        };

        inherit (pkgs) lib;

        rustToolchainFor =
          p:
          p.rust-bin.stable.latest.default.override {
            # Set the build targets supported by the toolchain,
            # wasm32-unknown-unknown is required for trunk.
            targets = [ "wasm32-unknown-unknown" ];
          };
        craneLib = (crane.mkLib pkgs).overrideToolchain rustToolchainFor;

        # When filtering sources, we want to allow assets other than .rs files
        unfilteredRoot = ./.; # The original, unfiltered source
        src = lib.fileset.toSource {
          root = unfilteredRoot;
          fileset = lib.fileset.unions [
            # Default files from crane (Rust and cargo files)
            (craneLib.fileset.commonCargoSources unfilteredRoot)
            (lib.fileset.fileFilter (
              file:
              lib.any file.hasExt [
                "proto"
              ]
            ) unfilteredRoot)
            # Example of a folder for images, icons, etc
            (lib.fileset.maybeMissing ./assets)
          ];
        };

        # Arguments to be used by both the client and the server
        # When building a workspace with crane, it's a good idea
        # to set "pname" and "version".
        commonArgs = with pkgs; {
          inherit src;
          strictDeps = true;

          doCheck = false;
          nativeBuildInputs = [
            pkg-config
          ];
          buildInputs =
            [
              # Add additional build inputs here
              openssl
              protobuf
            ]
            ++ lib.optionals pkgs.stdenv.isDarwin [
              # Additional darwin specific inputs can be set here
              pkgs.libiconv
            ];
          PROTOC = "${protobuf}/bin/protoc";
        };

        # Native packages

        nativeArgs = commonArgs // {
          pname = "trunk-workspace-native";
        };

        # Build *just* the cargo dependencies, so we can reuse
        # all of that work (e.g. via cachix) when running in CI
        cargoArtifacts = craneLib.buildDepsOnly nativeArgs;

        # Simple JSON API that can be queried by the client
        text-embeddings-router = craneLib.buildPackage (
          nativeArgs
          // {
            inherit cargoArtifacts;
            # The server needs to know where the client's dist dir is to
            # serve it, so we pass it as an environment variable at build time
          }
        );

        # Wasm packages

        # it's not possible to build the server on the
        # wasm32 target, so we only build the client.
        # wasmArgs = commonArgs // {
        #   pname = "trunk-workspace-wasm";
        #   cargoExtraArgs = "--package=client";
        #   CARGO_BUILD_TARGET = "wasm32-unknown-unknown";
        # };

        # cargoArtifactsWasm = craneLib.buildDepsOnly (
        #   wasmArgs
        #   // {
        #     doCheck = false;
        #   }
        # );

        # Build the frontend of the application.
        # This derivation is a directory you can put on a webserver.
        # myClient = craneLib.buildTrunkPackage (wasmArgs // {
        #   pname = "trunk-workspace-client";
        #   cargoArtifacts = cargoArtifactsWasm;
        #   # Trunk expects the current directory to be the crate to compile
        #   preBuild = ''
        #     cd ./client
        #   '';
        #   # After building, move the `dist` artifacts and restore the working directory
        #   postBuild = ''
        #     mv ./dist ..
        #     cd ..
        #   '';
        #   # The version of wasm-bindgen-cli here must match the one from Cargo.lock.
        #   # When updating to a new version replace the hash values with lib.fakeHash,
        #   # then try to do a build, which will fail but will print out the correct value
        #   # for `hash`. Replace the value and then repeat the process but this time the
        #   # printed value will be for the second `hash` below
        #   wasm-bindgen-cli = pkgs.buildWasmBindgenCli rec {
        #     src = pkgs.fetchCrate {
        #       pname = "wasm-bindgen-cli";
        #       version = "0.2.99";
        #       hash = "sha256-1AN2E9t/lZhbXdVznhTcniy+7ZzlaEp/gwLEAucs6EA=";
        #       # hash = lib.fakeHash;
        #     };
        mkl2024 = import ./nix/mkl.nix;

        onnxruntimeGcc13 = pkgs.onnxruntime.override {
          stdenv = pkgs.cudaPackages.backendStdenv;
        };

      in
      #     cargoDeps = pkgs.rustPlatform.fetchCargoVendor {
      #       inherit src;
      #       inherit (src) pname version;
      #       hash = "sha256-HGcqXb2vt6nAvPXBZOJn7nogjIoAgXno2OJBE1trHpc=";
      #       # hash = lib.fakeHash;
      #     };
      #   };
      # });
      {
        checks = {
          # Build the crate as part of `nix flake check` for convenience
          # inherit text-embeddings-router;

          # Run clippy (and deny all warnings) on the crate source,
          # again, reusing the dependency artifacts from above.
          #
          # Note that this is done as a separate derivation so that
          # we can block the CI if there are issues here, but not
          # prevent downstream consumers from building our crate by itself.
          # my-app-clippy = craneLib.cargoClippy (
          #   commonArgs
          #   // {
          #     inherit cargoArtifacts;
          #     cargoClippyExtraArgs = "--all-targets -- --deny warnings";
          #     # Here we don't care about serving the frontend
          #     CLIENT_DIST = "";
          #   }
          # );

          # # Check formatting
          # my-app-fmt = craneLib.cargoFmt commonArgs;
        };

        apps.default = flake-utils.lib.mkApp {
          name = "server";
          drv = text-embeddings-router;
        };

        packages = {
          default = text-embeddings-router;
        };

        devShells.default =
          pkgs.mkShell.override
            {
              stdenv = pkgs.cudaPackages.backendStdenv;
            }
            {

              # Extra inputs can be added here; cargo and rustc are provided by default.
              buildInputs = with pkgs; [
                protobuf
                rustup
                openssl
                pkg-config
                cudaPackages.cudatoolkit
                python3Packages.python
                python3Packages.venvShellHook
                onnxruntimeGcc13
                mkl
              ];
              venvDir = "./.venv";
              LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib:/run/opengl-driver/lib";
              LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib:/run/opengl-driver/lib";
              CUDA_ROOT = "${pkgs.cudaPackages.cudatoolkit}";
            };
      }
    );
}
