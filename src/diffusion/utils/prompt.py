"""Interactive prompt input backed by ``prompt_toolkit``.

Import-light: ``prompt_toolkit`` is heavy enough to keep out of the ``--help`` path, so
all of its imports live inside functions. The chat REPL builds one session up front and
reads lines with :func:`read_prompt`.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.completion import Completer

# Slash commands offered by the chat REPL, used for tab-completion.
SLASH_COMMANDS = ("/steps", "/size", "/seed", "/neg", "/help", "/exit")

# Where the persistent input history is stored.
HISTORY_PATH = Path.home() / ".cache" / "diffusion" / "history"


def _slash_completer() -> Completer:
    """Build a completer over the chat slash commands.

    Completions are offered ONLY while typing a slash command — i.e. when the line
    begins with ``/`` and no space has been typed yet. Plain prompts get no dropdown.

    Returns
    -------
    prompt_toolkit.completion.Completer
        A completer that suggests :data:`SLASH_COMMANDS`.
    """
    from prompt_toolkit.completion import Completer, Completion

    class _SlashCompleter(Completer):
        def get_completions(self, document, complete_event):
            text = document.text_before_cursor
            # Only a bare "/word" with no space is a command being typed; everything
            # else is a normal prompt and must not trigger the dropdown.
            if not text.startswith("/") or " " in text:
                return
            for cmd in SLASH_COMMANDS:
                if cmd.startswith(text):
                    yield Completion(cmd, start_position=-len(text))

    return _SlashCompleter()


def build_session() -> PromptSession:
    """Build a ``PromptSession`` with history, slash-command completion, and styling.

    History is persisted to ``~/.cache/diffusion/history`` (the parent directory is
    created if needed).

    Returns
    -------
    prompt_toolkit.PromptSession
        A configured session ready for :func:`read_prompt`.
    """
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.styles import Style

    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    style = Style.from_dict({"prompt": "bold #ff8a4c"})
    return PromptSession(
        history=FileHistory(str(HISTORY_PATH)),
        completer=_slash_completer(),
        style=style,
    )


def read_prompt(session: PromptSession) -> str:
    """Read one line from ``session``, stripped of surrounding whitespace.

    ``EOFError`` (Ctrl-D) and ``KeyboardInterrupt`` (Ctrl-C) are allowed to propagate so
    the caller can treat them as "quit".

    Parameters
    ----------
    session : prompt_toolkit.PromptSession
        The session built by :func:`build_session`.

    Returns
    -------
    str
        The entered line with leading/trailing whitespace removed.
    """
    return session.prompt([("class:prompt", "\n› ")]).strip()
