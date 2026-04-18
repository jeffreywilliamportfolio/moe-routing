#!/usr/bin/env python3
"""122B-local prompt rendering helpers for prompt TSV generation.

This bundle uses the model-specific chat template in `chat_template.txt` as the
reference. For current routing runs we only need the simplest supported case:
single-turn user input with generation prompt enabled and thinking disabled.

The rendered form here matches that case directly:

<|im_start|>user
...content...
<|im_end|>
<|im_start|>assistant
<think>

</think>

"""

from __future__ import annotations

from pathlib import Path


BUNDLE_DIR = Path(__file__).resolve().parent.parent
CHAT_TEMPLATE_PATH = BUNDLE_DIR / "chat_template.txt"


def render_single_user_no_think(user_text: str) -> str:
    """Render the current 122B single-user no-think prompt form."""
    text = user_text.strip()
    return f"<|im_start|>user\\n{text}<|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n"


def template_reference_path() -> Path:
    return CHAT_TEMPLATE_PATH


__all__ = ["render_single_user_no_think", "template_reference_path"]
