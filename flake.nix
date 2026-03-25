{
  description = "Minimal flake for summit_sim.";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs";
    flake-parts.url = "github:hercules-ci/flake-parts";
  };

    outputs = { nixpkgs, self, ... } @ inputs: inputs.flake-parts.lib.mkFlake { inherit inputs; } {
        perSystem = { system, pkgs, lib, ... }:
        let
            pkgs = import inputs.nixpkgs {
                inherit system;
                config.allowUnfree = true;
            };
        in {
            devShells.default =
                pkgs.mkShell rec {
                    buildInputs = with pkgs; [
                        uv
                        python312
                        python312Packages.jupyterlab
                        python312Packages.pip
                        ruff
                    ];

                    # Required for building C extensions
                    LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib";
                    # PYTHONPATH is overridden with contents from e.g. poetry */site-package.
                    # We do not want them to be in PYTHONPATH.
                    # Therefore, in ./.envrc PYTHONPATH is set to the _PYTHONPATH defined below
                    # and also in shellHooks (direnv does not load shellHook exports, always).
                    _PYTHONPATH = "${pkgs.python312}/lib/python3.12/site-packages";

                    shellHook = ''
                        # Setup pre-commit
                        pre-commit install
                    '';

                    # Note: .env is NOT sourced here. Docker containers load
                    # environment variables via docker-compose.yml env_file directive.
                    # This prevents AWS credentials and other host env vars from
                    # leaking into the container runtime.
                };
        };
        flake = { };
        systems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
    };
}