"""The four server configurations under test.

Three orthogonal knobs, varied one at a time so each comparison isolates one
decision:

    A -> B   tool count       22 endpoint wrappers vs 6 task-shaped tools
    B -> C   response shaping  raw upstream records vs projected ones
    C -> D   schema docs       bare parameter names vs documented ones

    E -> D   tool count again, with documentation held equal

E exists because the chain above adds documentation only on top of the
6-tool server, so `A -> D` conflates consolidating the tools with documenting
them. E is the baseline's strongest form -- the same 22 wrappers, with the
attribute paths and field names a careful author would supply -- and `E -> D`
is the tool-count comparison with that advantage given to both sides.

Building E was only honest once `granular.py` had parameter documentation to
keep: before that, turning docs on for the 22-tool server moved its rendered
definitions by 1% and added no parameter descriptions at all.

Schema verbosity is applied here rather than in the server, because it is a
property of how tool definitions are rendered to a model, not of the server's
behaviour. Stripping is deliberately crude: drop every `description`, keep the
types. That is what a server written without attention to its own schemas
looks like.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Literal

ToolFamily = Literal["granular", "task_shaped"]


@dataclass(frozen=True)
class Variant:
    name: str
    family: ToolFamily
    shaping: bool
    schema_docs: bool
    isolates: str

    @property
    def label(self) -> str:
        shaping = "on" if self.shaping else "off"
        docs = "on" if self.schema_docs else "off"
        return f"{self.name} ({self.family}, shaping={shaping}, docs={docs})"


VARIANTS: tuple[Variant, ...] = (
    Variant(
        "A", "granular", shaping=False, schema_docs=False, isolates="baseline: 1:1 API wrappers"
    ),
    Variant("B", "task_shaped", shaping=False, schema_docs=False, isolates="vs A: tool count"),
    Variant(
        "C", "task_shaped", shaping=True, schema_docs=False, isolates="vs B: response shaping"
    ),
    Variant(
        "D",
        "task_shaped",
        shaping=True,
        schema_docs=True,
        isolates="vs C: schema documentation",
    ),
    Variant(
        "E",
        "granular",
        shaping=False,
        schema_docs=True,
        isolates="vs D: tool count, documentation held equal",
    ),
)

BY_NAME = {v.name: v for v in VARIANTS}


def strip_descriptions(schema: dict[str, Any]) -> dict[str, Any]:
    """Recursively remove `description` keys from a JSON schema."""
    if not isinstance(schema, dict):
        return schema
    out: dict[str, Any] = {}
    for key, value in schema.items():
        if key == "description":
            continue
        if isinstance(value, dict):
            out[key] = strip_descriptions(value)
        elif isinstance(value, list):
            out[key] = [strip_descriptions(v) if isinstance(v, dict) else v for v in value]
        else:
            out[key] = value
    return out


def render_tool_definitions(tools: list[Any], variant: Variant) -> list[dict[str, Any]]:
    """Tool definitions in Anthropic tool-use format, per the variant's rules."""
    rendered: list[dict[str, Any]] = []
    for tool in tools:
        schema = copy.deepcopy(tool.input_schema or {"type": "object", "properties": {}})
        description = tool.description or ""
        if not variant.schema_docs:
            schema = strip_descriptions(schema)
            # Keep the first sentence only: enough to know what the tool is,
            # without the usage guidance that variant D pays for.
            description = description.split(".")[0].strip()
        rendered.append({"name": tool.name, "description": description, "input_schema": schema})
    return rendered
