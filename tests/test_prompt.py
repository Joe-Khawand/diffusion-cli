"""Tests for the interactive prompt helpers. Pure, no GPU."""

from __future__ import annotations

from diffusion.utils import prompt


def _completions(text: str) -> set[str]:
    from prompt_toolkit.completion import CompleteEvent
    from prompt_toolkit.document import Document

    completer = prompt._slash_completer()
    doc = Document(text, cursor_position=len(text))
    return {c.text for c in completer.get_completions(doc, CompleteEvent())}


def test_completer_suggests_s_commands() -> None:
    assert _completions("/s") == {"/steps", "/size", "/seed"}


def test_completer_suggests_neg() -> None:
    assert _completions("/n") == {"/neg"}


def test_completer_empty_slash_returns_all() -> None:
    assert _completions("/") == set(prompt.SLASH_COMMANDS)


def test_completer_ignores_plain_prompt() -> None:
    # A normal prompt must not trigger the slash-command dropdown.
    assert _completions("Image of a fox") == set()
    assert _completions("") == set()


def test_completer_stops_after_command_word() -> None:
    # Once a space is typed, the command is chosen; no more suggestions.
    assert _completions("/steps ") == set()


def test_history_path_under_cache() -> None:
    assert prompt.HISTORY_PATH.name == "history"
    assert prompt.HISTORY_PATH.parent.name == "diffusion"
