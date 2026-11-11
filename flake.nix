{
  description = "Python development shell with uv and a Nix-managed interpreter";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { nixpkgs, ... }:
    let
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "aarch64-darwin"
      ];
      forAllSystems = nixpkgs.lib.genAttrs systems;
    in
    {
      devShells = forAllSystems (system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          python = "${pkgs.python3}/bin/python3";
        in
        {
          default = pkgs.mkShell {
            packages = [
              pkgs.python3
              pkgs.uv
            ];

            PYTHON = python;
            UV_PYTHON = python;
            UV_PYTHON_DOWNLOADS = "never";
          };
        });
    };
}
