# Remember

Find and read messages from past coding-agent sessions. Run the CLI from this
checkout with `PYTHONPATH=src uv run python -m remember.cli`.

## Machine-readable output

Every command accepts `--json` or the equivalent `--format json`:

```sh
PYTHONPATH=src uv run python -m remember.cli sessions --all-projects --json
PYTHONPATH=src uv run python -m remember.cli search "dependency" --format json
PYTHONPATH=src uv run python -m remember.cli read pi:01a1ac0c --json
```

The output options also work before the command. Text remains the default;
`--format text` selects it explicitly. If output options repeat, the last one
wins. `--help` always shows the normal text help.

JSON mode writes one JSON object to stdout, without text headings or summaries.
All responses include a `warnings` array. Warnings that would normally go to
stderr appear in this array instead.

| Command | JSON fields |
| --- | --- |
| `sessions` | `sessions`, `total` (before the limit), `limit` |
| `search` | `sessions` with a `matches` array per session, `total_matches`, `total_sessions` (both before the limit), `limit` |
| `read` | `session`, `messages`, `pagination` |

Session objects include `ref`, the full `id`, `source`, `cwd`, `start`, and `end`.
The `sessions` and `search` commands also include the full opening `request`.
Message objects include `ref`, the full `id`, `role`, `time`, `text`, `truncated`,
and `total_chars`. Times use ISO 8601 with a UTC offset; an unknown message time
is `null`. Text keeps Unicode, newlines, and quotes.

The usual scope, role, `--full`, and limit options apply in JSON mode. Search
returns full matching text, not display snippets. Read respects `--max-chars`
(default: 2000; `0` means unlimited), with truncation described by `truncated`
and `total_chars`, not a hint added to the text.

Read's `pagination` contains `total` messages in the selected view, one-based
`start` and `end` positions for the returned page, and `earlier_ref` and
`later_ref` for adjacent messages. Missing positions or refs are `null`.
Empty results use empty arrays, not a text notice.

Successful calls, including empty results, exit with status 0. Invalid arguments,
unresolved or ambiguous refs, and file access errors exit with status 2. In JSON
mode these return `{"error": {"type": "...", "message": "..."}, "warnings": []}`
on stdout, with no text on stderr. Error types are `usage_error`,
`command_error`, and `io_error`.

## Python development shell

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
