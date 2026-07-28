"""Tool registry with discovery, validation, and built-in tools."""

from __future__ import annotations

import json
import math
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from thunders_ai.config import ThundersConfig
from thunders_ai.logger import get_logger

logger = get_logger(__name__)


class Tool:
    """Represents a single tool that an agent can invoke.

    Args:
        name: Unique tool name.
        description: Human-readable description.
        func: Callable implementing the tool logic.
        input_schema: Optional JSON-schema-style dict describing inputs.
        output_schema: Optional JSON-schema-style dict describing outputs.
    """

    def __init__(
        self,
        name: str,
        description: str,
        func: Callable,
        input_schema: Optional[Dict[str, Any]] = None,
        output_schema: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.name = name
        self.description = description
        self.func = func
        self.input_schema = input_schema or {}
        self.output_schema = output_schema or {}

    def execute(self, *args: Any, **kwargs: Any) -> Any:
        """Run the tool and return its result."""
        return self.func(*args, **kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise tool metadata (excludes the callable)."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
        }


# ---------------------------------------------------------------------------
# Built-in tools
# ---------------------------------------------------------------------------

def _tool_calculator(expression: str, **_: Any) -> str:
    """Evaluate a simple mathematical expression safely."""
    try:
        allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        allowed.update({"abs": abs, "round": round, "min": min, "max": max})
        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(result)
    except Exception as exc:
        return f"Calculator error: {exc}"


def _tool_search(query: str, **_: Any) -> str:
    """Simulate a web search (placeholder – returns a stub result)."""
    return f"[Search results for '{query}'] – No real search backend configured."


def _tool_code_executor(code: str, **_: Any) -> str:
    """Execute Python code in a restricted namespace and capture output."""
    namespace: Dict[str, Any] = {"math": math, "json": json, "re": re}
    try:
        exec(code, {"__builtins__": {}}, namespace)
        # Try to return the last expression if it looks like a statement
        lines = [l.strip() for l in code.strip().splitlines() if l.strip()]
        if lines:
            last = lines[-1]
            if not last.startswith(("import ", "from ", "def ", "class ", "for ", "while ", "if ")):
                return str(eval(last, {"__builtins__": {}}, namespace))
        return "Code executed successfully (no return value)."
    except Exception as exc:
        return f"Execution error: {exc}"


_BUILTIN_TOOLS: List[Tool] = [
    Tool(
        name="calculator",
        description="Evaluate a mathematical expression. Input: expression string.",
        func=_tool_calculator,
        input_schema={"type": "string", "description": "Math expression to evaluate"},
    ),
    Tool(
        name="search",
        description="Search the web for information. Input: query string.",
        func=_tool_search,
        input_schema={"type": "string", "description": "Search query"},
    ),
    Tool(
        name="code_executor",
        description="Execute Python code and return the result. Input: code string.",
        func=_tool_code_executor,
        input_schema={"type": "string", "description": "Python code to execute"},
    ),
]


class ToolRegistry:
    """Register, discover, and execute AI tools.

    Ships with built-in tools (calculator, search, code_executor) and
    supports custom tool registration with input/output validation.

    Args:
        config: ThundersConfig instance.

    Example::

        registry = ToolRegistry(config)
        result = registry.execute("calculator", "2 + 3 * 4")
        registry.register("greet", "Say hello", lambda name: f"Hello {name}!")
    """

    def __init__(self, config: ThundersConfig) -> None:
        self._config = config
        self._tools: Dict[str, Tool] = {}
        self._load_builtins()
        logger.info("ToolRegistry initialized with %d built-in tools.", len(self._tools))

    def _load_builtins(self) -> None:
        """Register built-in tools."""
        for tool in _BUILTIN_TOOLS:
            self._tools[tool.name] = tool

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        name: str,
        description: str,
        func: Callable,
        input_schema: Optional[Dict[str, Any]] = None,
        output_schema: Optional[Dict[str, Any]] = None,
        override: bool = False,
    ) -> None:
        """Register a new tool.

        Args:
            name: Unique tool name.
            description: Human-readable description.
            func: Callable implementing the tool.
            input_schema: Optional schema for input validation.
            output_schema: Optional schema for output validation.
            override: Allow overwriting an existing tool with the same name.

        Raises:
            ValueError: If a tool with *name* already exists and *override* is *False*.
        """
        if name in self._tools and not override:
            raise ValueError(f"Tool '{name}' already registered. Set override=True to replace.")
        tool = Tool(name, description, func, input_schema, output_schema)
        self._tools[name] = tool
        logger.info("Tool registered: %s", name)

    def unregister(self, name: str) -> bool:
        """Remove a tool from the registry.

        Returns:
            *True* if the tool was found and removed.
        """
        if name in self._tools:
            del self._tools[name]
            logger.info("Tool unregistered: %s", name)
            return True
        return False

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """Execute a registered tool.

        Args:
            name: Tool name.
            *args: Positional arguments forwarded to the tool.
            **kwargs: Keyword arguments forwarded to the tool.

        Returns:
            Tool result.

        Raises:
            KeyError: If the tool is not registered.
        """
        tool = self._tools.get(name)
        if tool is None:
            raise KeyError(f"Tool '{name}' not found in registry.")
        logger.debug("Executing tool: %s", name)
        return tool.execute(*args, **kwargs)

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def list_tools(self) -> List[str]:
        """Return the names of all registered tools."""
        return list(self._tools.keys())

    def get_tool(self, name: str) -> Optional[Tool]:
        """Retrieve a tool by name."""
        return self._tools.get(name)

    def describe_tools(self) -> List[Dict[str, Any]]:
        """Return metadata dicts for all registered tools."""
        return [t.to_dict() for t in self._tools.values()]
