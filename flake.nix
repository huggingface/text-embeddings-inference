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
          pkgs = nixpkgs.legacyPackages.${system};
        in
        {
          default = pkgs.mkShell {
            buildInputs =
              with pkgs;
              [
                rustup
                protobuf
                openssl
                gcc
                pkg-config
              ]
              ++ pkgs.lib.optionals pkgs.stdenv.isDarwin [
                apple-sdk_15
              ];
            venvDir = "./.venv";
            # Relax compiler strictness for oniguruma's old C code
            CFLAGS = "-Wno-error=old-style-definition -Wno-error=implicit-function-declaration";
            RUSTFLAGS = "-C default-linker-libraries=yes";
            postVenvCreation = ''
              unset SOURCE_DATE_EPOCH
            '';
            postShellHook = ''
              unset SOURCE_DATE_EPOCH
            '';
          };

        }
      );
    };
}
