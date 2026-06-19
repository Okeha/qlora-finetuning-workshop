{
  description = "Development shell for qlora-finetuning-workshop";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    { nixpkgs, ... }:
    let
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];

      forAllSystems =
        function:
        nixpkgs.lib.genAttrs systems (
          system:
          function {
            pkgs = import nixpkgs { inherit system; };
          }
        );
    in
    {
      devShells = forAllSystems (
        { pkgs }:
        {
          default = pkgs.mkShell {
            packages = with pkgs; [
              git
              gnumake
              python313
              uv

              # Native libraries commonly needed by PyTorch/Transformers wheels.
              openssl
              stdenv.cc.cc.lib
              zlib
            ];

            env = {
              UV_PYTHON = "${pkgs.python313}/bin/python";
              UV_PROJECT_ENVIRONMENT = ".venv";
            };

            shellHook = ''
              # NixOS exposes host GPU driver libraries here; CUDA wheels need
              # libcuda.so/libnvidia-ml.so from this path at runtime.
              export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath [
                pkgs.openssl
                pkgs.stdenv.cc.cc.lib
                pkgs.zlib
              ]}:/run/opengl-driver/lib:''${LD_LIBRARY_PATH:-}
            '';
          };
        }
      );
    };
}
