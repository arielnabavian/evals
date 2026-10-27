"""Tool descriptions as tunable text: one ``TextValue`` per tool, starting from the tool's own description.

    @tunable(tools=tool_descriptions(open_document, read_page))
    def create_agent(tools):
        return Agent(tools=tools, ...)

The factory receives copies of the tools, each with the trial's description; the tools themselves are never changed.
The keys are ``tools.open_document``, ``tools.read_page`` (named after the factory's argument).
"""

from __future__ import annotations

import copy
from typing import Any

from .component import Component
from .domains import TextValue

# Bedrock rejects longer tool descriptions, and a longer one has stopped being a contract (the limit harness-optimizer's
# tool-description formula uses).
MAX_CHARS = 800


def tool_name(tool: Any) -> str:
    return getattr(tool, "tool_name", None) or tool.tool_spec["name"]


def description(tool: Any) -> str:
    return str(tool.tool_spec.get("description") or "")


def with_description(tool: Any, text: str) -> Any:
    """A copy of ``tool`` whose description is ``text``."""
    out = copy.copy(tool)
    mcp = getattr(tool, "mcp_tool", None)
    if mcp is not None and hasattr(mcp, "description"):  # an MCP tool builds its spec from mcp_tool.description
        out.mcp_tool = mcp.model_copy(update={"description": text}) if hasattr(mcp, "model_copy") else copy.copy(mcp)
        if not hasattr(mcp, "model_copy"):
            out.mcp_tool.description = text
        return out
    spec = getattr(tool, "_tool_spec", None)
    if isinstance(spec, dict):
        out._tool_spec = {**spec, "description": text}
        return out
    raise TypeError(f"{type(tool).__name__} {tool_name(tool)!r}: its description cannot be set")


class _Described:
    """Builds the tools with their descriptions (the constructor of the tool_descriptions component)."""

    def __init__(self, tools: tuple[Any, ...]) -> None:
        self.tools = tools

    def __call__(self, **descriptions: str) -> list[Any]:
        return [with_description(t, descriptions[tool_name(t)]) for t in self.tools]


def tool_descriptions(*tools: Any, max_chars: int | None = MAX_CHARS) -> Component:
    """One ``TextValue`` per tool, each starting from the tool's own description; the factory gets the tools."""
    names = [tool_name(t) for t in tools]
    if len(set(names)) != len(names):
        raise ValueError(f"two tools have the same name: {names}")
    return Component(
        _Described(tuple(tools)),
        {
            tool_name(t): TextValue(
                default=description(t),
                max_chars=max_chars,
                about=f"the description of the {tool_name(t)} tool: what the model reads to decide when to call it, "
                "how to fill its inputs and how to read its result",
            )
            for t in tools
        },
    )
