# Python development shell

Requires Nix with the `nix-command` and `flakes` experimental features enabled.

```sh
nix develop
uv venv
uv run python --version
```

The shell includes `uv` and Python from the pinned Nixpkgs input. Both `PYTHON`
and uv's `UV_PYTHON` point directly to that Python executable in the Nix store.
`UV_PYTHON_DOWNLOADS=never` prevents uv from downloading another interpreter.
Virtual environments and Python dependencies remain managed by uv; entering the
shell does not create a virtual environment or install dependencies.

Shell outputs are available for x86_64 and aarch64 Linux, and Apple Silicon macOS.
The pinned Nixpkgs no longer supports Intel macOS.

To update the pinned tools, run `nix flake update nixpkgs` and review `flake.lock`.
After changing the flake, run `nix flake check` and repeat the quick start above.
When this directory is put under Git, include `flake.nix` and `flake.lock` in the
repository so Git-backed flake commands can see them.
