"""
Standardized Tool Schema System
================================
Generates OpenAI-compatible tool definitions from any tool in the registry.
Supports BaseTool subclasses, plain objects with execute(), callables.

This enables any LLM supporting function calling (GPT-4, Gemini, Claude, Moonshot)
to discover and call tools using the standard `tools` API format.
"""

import inspect
from typing import Dict, Any, List, Optional, get_type_hints
from dataclasses import dataclass, field
from loguru import logger

from .tool_registry import ToolRegistry


@dataclass
class ToolSchema:
    """Normalized schema for a tool, compatible with OpenAI function calling format."""

    name: str
    description: str
    parameters: Dict[str, Any]
    required: List[str] = field(default_factory=list)


def _extract_pydantic_schema(tool: Any) -> Optional[Dict[str, Any]]:
    """Extract JSON schema from a Pydantic-v2-typed BaseTool."""
    try:
        if hasattr(tool, "args_schema") and tool.args_schema is not None:
            schema = tool.args_schema.model_json_schema()
            properties = schema.get("properties", {})
            required = schema.get("required", [])
            return {
                "type": "object",
                "properties": properties,
                "required": required,
            }
    except Exception:
        pass
    return None


def _extract_function_signature(
    tool: Any, method_name: str = "execute"
) -> Optional[Dict[str, Any]]:
    """Extract schema from a function signature using type hints."""
    method = None
    if hasattr(tool, method_name):
        method = getattr(tool, method_name)
    elif callable(tool):
        method = tool

    if method is None:
        return None

    try:
        sig = inspect.signature(method)
        hints = get_type_hints(method) if hasattr(method, "__annotations__") else {}

        properties = {}
        required = []

        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls", "args", "kwargs"):
                continue

            param_type = hints.get(param_name, str)
            type_name = _type_to_json_schema_type(param_type)
            prop: Dict[str, Any] = {"type": type_name}

            if param.default is not inspect.Parameter.empty:
                prop["default"] = param.default
            else:
                required.append(param_name)

            properties[param_name] = prop

        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }
    except (ValueError, TypeError) as e:
        logger.debug(
            f"Could not extract signature for {getattr(tool, 'name', 'unknown')}: {e}"
        )
        return None


def _type_to_json_schema_type(tp: type) -> str:
    """Map Python types to JSON Schema types."""
    origin = getattr(tp, "__origin__", None)
    if origin is list or origin is List:
        return "array"
    if origin is dict or origin is Dict:
        return "object"
    if tp is str:
        return "string"
    if tp is int:
        return "integer"
    if tp is float:
        return "number"
    if tp is bool:
        return "boolean"
    if tp is type(None):
        return "null"
    return "string"


def _get_tool_description(tool: Any) -> str:
    """Get description from any tool type."""
    if hasattr(tool, "description") and tool.description:
        return tool.description
    doc = inspect.getdoc(tool)
    if doc:
        return doc.split("\n")[0].strip()
    return f"Tool: {getattr(tool, 'name', 'unknown')}"


def get_tool_schema(tool: Any, name: Optional[str] = None) -> ToolSchema:
    """
    Extract a normalized ToolSchema from any tool implementation.

    Supports:
    - langchain_core.tools.BaseTool subclasses (with args_schema)
    - Plain objects with execute() method + name/description attrs
    - Callable functions
    """
    tool_name = name or getattr(
        tool,
        "name",
        tool.__class__.__name__ if hasattr(tool, "__class__") else "unknown",
    )
    description = _get_tool_description(tool)

    # Try Pydantic schema first (BaseTool subclasses)
    schema = _extract_pydantic_schema(tool)

    # Fall back to function signature
    if schema is None:
        schema = _extract_function_signature(tool)

    # Fall back to minimal schema
    if schema is None:
        schema = {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": f"Input for {tool_name}"}
            },
            "required": ["query"],
        }

    return ToolSchema(
        name=tool_name,
        description=description,
        parameters=schema["properties"],
        required=schema.get("required", []),
    )


def format_as_openai_tools(schemas: List[ToolSchema]) -> List[Dict[str, Any]]:
    """
    Format tool schemas into the OpenAI `tools` API format.

    This is the standard format used by GPT-4, Gemini (via OpenAI-compat endpoint),
    Claude (via tool use), and Moonshot AI.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": s.name,
                "description": s.description,
                "parameters": {
                    "type": "object",
                    "properties": s.parameters,
                    "required": s.required,
                },
            },
        }
        for s in schemas
    ]


def get_registry_schemas(registry: ToolRegistry) -> List[ToolSchema]:
    """Extract schemas for all tools in a ToolRegistry."""
    schemas = []
    for name in registry.list_tools():
        tool = registry.get_tool(name)
        if tool is not None:
            try:
                schema = get_tool_schema(tool, name)
                schemas.append(schema)
            except Exception as e:
                logger.warning(f"Failed to build schema for tool '{name}': {e}")
    return schemas


def get_registry_openai_tools(registry: ToolRegistry) -> List[Dict[str, Any]]:
    """Get all tools in OpenAI-compatible format from a ToolRegistry."""
    schemas = get_registry_schemas(registry)
    return format_as_openai_tools(schemas)


__all__ = [
    "ToolSchema",
    "get_tool_schema",
    "format_as_openai_tools",
    "get_registry_schemas",
    "get_registry_openai_tools",
]
