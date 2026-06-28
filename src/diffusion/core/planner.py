"""Claude-powered prompt refinement for the agent command.

Torch-free: only stdlib + anthropic. All LLM communication lives here so the
command module stays focused on orchestration.
"""

from __future__ import annotations

import base64
import json
import re
from typing import TYPE_CHECKING

from diffusion.utils.errors import DiffusionError

if TYPE_CHECKING:
    from pathlib import Path

_MODEL = "claude-sonnet-4-6"

_SYSTEM = """\
You are an expert prompt engineer for diffusion image-generation models.

The user has a *goal* — a natural-language description of the image they want.
Your job is to write and iteratively refine the **diffusion prompt** (and
optional negative prompt) that will produce an image matching that goal.

Rules:
- Write prompts in the terse, comma-separated style that diffusion models
  respond to best (e.g. "cinematic photo, golden hour, shallow depth of field").
- Keep prompts under 200 words.
- The negative prompt should list qualities to avoid (e.g. "blurry, low quality,
  watermark, text").
- When critiquing an image, be specific about what matches or misses the goal.
- Score from 1-10: 1-3 poor, 4-6 decent, 7-8 good, 9-10 excellent match.

Always respond with a JSON object (no markdown fence) containing these fields:
{
  "prompt": "the diffusion prompt",
  "negative_prompt": "things to avoid, or null",
  "critique": "what works and what doesn't (empty string on first plan)",
  "score": 0,
  "reasoning": "why you chose or changed the prompt"
}
"""


class AgentPlanner:
    """Stateful Claude API wrapper for the plan-critique-refine loop."""

    def __init__(self, goal: str, *, model: str = _MODEL) -> None:
        try:
            import anthropic
        except ImportError:
            raise DiffusionError(
                "The 'anthropic' package is required for the agent command.",
                hint="Install it with: pip install anthropic   "
                "or: pip install diffusion-cli[agent]",
            ) from None

        api_key = None
        import os

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise DiffusionError(
                "ANTHROPIC_API_KEY environment variable is not set.",
                hint="Export your key: export ANTHROPIC_API_KEY=sk-ant-...",
            )

        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model
        self._goal = goal
        self._messages: list[dict] = []

    def initial_plan(self) -> tuple[str, str | None]:
        """Ask Claude to craft the first prompt from the goal.

        Returns
        -------
        tuple of (str, str or None)
            The initial (prompt, negative_prompt).
        """
        self._messages = [
            {
                "role": "user",
                "content": f"My goal is: {self._goal}\n\n"
                "Create an initial diffusion prompt to achieve this goal. "
                "Respond with the JSON object only.",
            }
        ]
        result = self._call()
        return result["prompt"], result.get("negative_prompt")

    def critique_and_refine(
        self, image_path: Path, current_prompt: str, iteration: int
    ) -> tuple[str, int, str, str | None]:
        """Send the generated image to Claude for vision critique and refinement.

        Parameters
        ----------
        image_path : Path
            Path to the PNG image from the last iteration.
        current_prompt : str
            The prompt that produced the image.
        iteration : int
            The current iteration number (1-based).

        Returns
        -------
        tuple of (str, int, str, str or None)
            (critique_text, score, new_prompt, new_negative_prompt).
        """
        b64 = _encode_image(image_path)
        self._messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": f"This is iteration {iteration}. The prompt was:\n"
                        f"{current_prompt}\n\n"
                        f"Goal reminder: {self._goal}\n\n"
                        "Critique the image against the goal, score it 1-10, "
                        "and provide a refined prompt. Respond with the JSON object only.",
                    },
                ],
            }
        )
        result = self._call()
        return (
            result.get("critique", ""),
            int(result.get("score", 5)),
            result["prompt"],
            result.get("negative_prompt"),
        )

    def _call(self) -> dict:
        """Make an API call and parse the JSON response."""
        import anthropic

        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=1024,
                system=_SYSTEM,
                messages=self._messages,
            )
        except anthropic.APIError as exc:
            raise DiffusionError(
                f"Claude API error: {exc}",
                hint="Check your API key and network connection.",
            ) from exc

        text = response.content[0].text
        self._messages.append({"role": "assistant", "content": text})
        return _parse_json(text)


def _encode_image(path: Path) -> str:
    """Base64-encode a PNG file for the vision API."""
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _parse_json(text: str) -> dict:
    """Extract a JSON object from Claude's response text."""
    # Try fenced code block first.
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return json.loads(match.group(1))
    # Try raw JSON.
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise DiffusionError(
        "Could not parse Claude's response as JSON.",
        hint=f"Raw response: {text[:200]}",
    )
