"""Jinja2 templates for the reflection agents' prompts.

Ported from strands-labs/harness-optimizer, ``strands_harness_optimizer/utils/templates.py``. ``load_builtin_template``
is renamed ``load_template`` and ``list_builtin_templates`` ``list_templates``; both read this package's
``templates/`` directory, which holds the templates the optimizers here use.
"""

from __future__ import annotations

import os
from typing import Any, Protocol

from jinja2 import Template
from jinja2.sandbox import SandboxedEnvironment

# Built-in templates live under _reflection/templates/
_TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")

# autoescape is OFF. These templates render prompts for an LLM, not HTML: with
# escaping on, a system prompt or tool description passed in as a variable came out
# with `"` as `&#34;` and `<` as `&lt;`, and the optimizer then reasoned about, and
# wrote back, text that was never in the agent. Sandboxing is unrelated to escaping
# and stays on.
_ENV = SandboxedEnvironment(autoescape=False)


class Renderable(Protocol):
    """Anything a prompt can be rendered from: a jinja ``Template``, or an object wrapping one."""

    def render(self, **kwargs: Any) -> str: ...


def create_template(source: str) -> Template:
    """Create a sandboxed Jinja2 Template from a string.

    Args:
        source: Jinja2 template source string.

    Returns:
        Jinja2 Template object backed by a SandboxedEnvironment.
    """
    return _ENV.from_string(source)


def load_template(path: str) -> Template:
    """Load a built-in Jinja2 template by relative path.

    Args:
        path: Relative path within the built-in templates directory.
            e.g., "skill_library/task_message.jinja"

    Returns:
        Jinja2 Template object backed by a SandboxedEnvironment.

    Raises:
        FileNotFoundError: If the template does not exist.
    """
    full_path = os.path.normpath(os.path.join(_TEMPLATE_DIR, path))
    with open(full_path) as f:
        return create_template(f.read())


def list_templates() -> list[str]:
    """List all available built-in template paths.

    Returns:
        List of relative paths to built-in templates.
    """
    templates = []
    base = os.path.normpath(_TEMPLATE_DIR)
    for root, _, files in os.walk(base):
        for f in files:
            if f.endswith(".jinja"):
                rel = os.path.relpath(os.path.join(root, f), base)
                templates.append(rel)
    return sorted(templates)
