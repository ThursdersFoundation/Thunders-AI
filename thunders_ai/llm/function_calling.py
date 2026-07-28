"""Function calling module implementing OpenAI-style structured function invocation."""

from __future__ import annotations

import inspect
import json
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, get_type_hints

from thunders_ai.config import ThundersConfig
from thunders_ai.logger import get_logger

logger = get_logger(__name__)

# Mapping of Python type hints to JSON Schema types
_TYPE_MAP: Dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


class FunctionDefinition:
    """Represents a callable function with its OpenAI-compatible schema.

    Args:
        name: Function name.
        description: Human-readable description.
        parameters: JSON Schema object describing parameters.
        func: The underlying Python callable.
    """

    def __init__(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        func: Callable,
    ) -> None:
        self.name = name
        self.description = description
        self.parameters = parameters
        self.func = func

    def to_openai_spec(self) -> Dict[str, Any]:
        """Return the function definition in OpenAI function-calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class FunctionResult:
    """Result of a function call, structured for LLM consumption."""

    __slots__ = ("name", "result", "error", "success")

    def __init__(
        self,
        name: str,
        result: Any = None,
        error: Optional[str] = None,
    ) -> None:
        self.name = name
        self.result = result
        self.error = error
        self.success = error is None

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the result."""
        payload: Dict[str, Any] = {"name": self.name, "success": self.success}
        if self.success:
            payload["result"] = self.result
        else:
            payload["error"] = self.error
        return payload

    def to_openai_message(self) -> Dict[str, str]:
        """Format as an OpenAI tool-response message."""
        content = json.dumps(self.result) if self.success else json.dumps({"error": self.error})
        return {"role": "tool", "name": self.name, "content": content}


class FunctionCaller:
    """Implements OpenAI-style function calling with signature parsing and validation.

    Registers Python callables, auto-generates JSON Schema from type hints,
    validates arguments, and executes functions safely with structured results.

    Args:
        config: ThundersConfig instance.

    Example::

        fc = FunctionCaller(config)

        def get_weather(city: str, unit: str = "celsius") -> str:
            \"\"\"Get the current weather for a city.\"\"\"
            return f"Sunny in {city}, 22°{unit[0].upper()}"

        fc.register(get_weather)
        result = fc.call("get_weather", {"city": "Paris"})
        print(result.to_dict())
    """

    def __init__(self, config: ThundersConfig) -> None:
        self._config = config
        self._functions: Dict[str, FunctionDefinition] = {}
        logger.info("FunctionCaller initialized.")

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> None:
        """Register a Python function for structured calling.

        The parameter schema is auto-generated from the function's type hints
        and docstring. Parameters without defaults are marked as required.

        Args:
            func: The callable to register.
            name: Override function name (defaults to ``func.__name__``).
            description: Override description (defaults to docstring).

        Raises:
            ValueError: If a function with the same name is already registered.
        """
        fn_name = name or func.__name__
        if fn_name in self._functions:
            raise ValueError(f"Function '{fn_name}' already registered.")

        fn_desc = description or inspect.getdoc(func) or "No description available."
        parameters = self._build_schema(func)

        definition = FunctionDefinition(
            name=fn_name,
            description=fn_desc,
            parameters=parameters,
            func=func,
        )
        self._functions[fn_name] = definition
        logger.info("Function registered: %s", fn_name)

    def unregister(self, name: str) -> bool:
        """Remove a registered function.

        Returns:
            *True* if the function was found and removed.
        """
        if name in self._functions:
            del self._functions[name]
            logger.info("Function unregistered: %s", name)
            return True
        return False

    # ------------------------------------------------------------------
    # Schema generation
    # ------------------------------------------------------------------

    @staticmethod
    def _build_schema(func: Callable) -> Dict[str, Any]:
        """Build a JSON Schema from a function's signature and type hints."""
        sig = inspect.signature(func)
        try:
            hints = get_type_hints(func)
        except Exception:
            hints = {}

        properties: Dict[str, Any] = {}
        required: List[str] = []

        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue

            ptype = hints.get(param_name, str)
            json_type = _TYPE_MAP.get(ptype, "string")

            prop: Dict[str, Any] = {"type": json_type}
            # Extract description from docstring if possible
            doc = inspect.getdoc(func) or ""
            pattern = rf"{param_name}\s*[:–-]\s*(.+?)(?:\n|$)"
            match = re.search(pattern, doc, re.IGNORECASE)
            if match:
                prop["description"] = match.group(1).strip()

            if param.default is inspect.Parameter.empty:
                required.append(param_name)
            else:
                prop["default"] = param.default

            properties[param_name] = prop

        schema: Dict[str, Any] = {
            "type": "object",
            "properties": properties,
        }
        if required:
            schema["required"] = required
        return schema

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_args(schema: Dict[str, Any], args: Dict[str, Any]) -> List[str]:
        """Validate arguments against a JSON Schema (simplified).

        Returns:
            List of validation error strings (empty if valid).
        """
        errors: List[str] = []
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))

        # Check required params
        missing = required - set(args.keys())
        if missing:
            errors.append(f"Missing required parameters: {sorted(missing)}")

        # Check types for provided params
        for key, value in args.items():
            if key not in properties:
                continue
            expected = properties[key].get("type", "string")
            type_checks = {
                "string": lambda v: isinstance(v, str),
                "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
                "number": lambda v: isinstance(v, (int, float)),
                "boolean": lambda v: isinstance(v, bool),
                "array": lambda v: isinstance(v, list),
                "object": lambda v: isinstance(v, dict),
            }
            checker = type_checks.get(expected)
            if checker and not checker(value):
                errors.append(f"Parameter '{key}' expected type '{expected}', got '{type(value).__name__}'")

        return errors

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def call(self, name: str, arguments: Dict[str, Any]) -> FunctionResult:
        """Execute a registered function with validated arguments.

        Args:
            name: Function name.
            arguments: Dict of argument name → value.

        Returns:
            A :class:`FunctionResult` with the outcome.
        """
        definition = self._functions.get(name)
        if definition is None:
            return FunctionResult(name=name, error=f"Function '{name}' not found.")

        # Validate
        errors = self._validate_args(definition.parameters, arguments)
        if errors:
            return FunctionResult(name=name, error="; ".join(errors))

        # Fill defaults for missing optional params
        for prop_name, prop_schema in definition.parameters.get("properties", {}).items():
            if prop_name not in arguments and "default" in prop_schema:
                arguments[prop_name] = prop_schema["default"]

        # Execute safely
        try:
            result = definition.func(**arguments)
            logger.info("Function '%s' called successfully.", name)
            return FunctionResult(name=name, result=result)
        except Exception as exc:
            logger.error("Function '%s' raised %s: %s", name, type(exc).__name__, exc)
            return FunctionResult(name=name, error=f"{type(exc).__name__}: {exc}")

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def list_functions(self) -> List[str]:
        """Return the names of all registered functions."""
        return list(self._functions.keys())

    def get_openai_tools(self) -> List[Dict[str, Any]]:
        """Return all function definitions in OpenAI tool format."""
        return [defn.to_openai_spec() for defn in self._functions.values()]

    def get_definition(self, name: str) -> Optional[FunctionDefinition]:
        """Retrieve a function definition by name."""
        return self._functions.get(name)

    def parse_and_call(self, llm_output: str) -> Optional[FunctionResult]:
        """Parse an LLM output string for a function call and execute it.

        Expected format: ``FUNCTION_CALL: name({...json args...})``

        Args:
            llm_output: Raw LLM output text.

        Returns:
            A :class:`FunctionResult` if a call was parsed, otherwise *None*.
        """
        pattern = r"FUNCTION_CALL:\s*(\w+)\((\{.*?\})\)"
        match = re.search(pattern, llm_output, re.DOTALL)
        if not match:
            return None
        fn_name = match.group(1)
        try:
            args = json.loads(match.group(2))
        except json.JSONDecodeError as exc:
            return FunctionResult(name=fn_name, error=f"Invalid JSON arguments: {exc}")
        return self.call(fn_name, args)
