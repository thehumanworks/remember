"""Readers for the saved sessions of each coding agent.

SOURCES maps each source name, as used in refs such as ``pi:01a1ac0c``, to the
module that reads it. Every module implements base.TranscriptSource.
"""

from . import claude_code, codex, github_copilot, pi
from .base import TranscriptSource

SOURCES: dict[str, TranscriptSource] = {
    "pi": pi,
    "codex": codex,
    "claude-code": claude_code,
    "copilot": github_copilot,
}
