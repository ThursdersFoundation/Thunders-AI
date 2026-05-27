"""Prompt template system with variable interpolation, few-shot, and chaining."""

from __future__ import annotations

import re
from string import Formatter
from typing import Any, Dict, List, Optional

from thunders_ai.config import ThundersConfig
from thunders_ai.logger import get_logger

logger = get_logger(__name__)


# Built-in templates for common tasks
BUILTIN_TEMPLATES: Dict[str, str] = {
    "summarise": (
        "Summarise the following text concisely:\n\n{text}\n\nSummary:"
    ),
    "translate": (
        "Translate the following text to {target_language}:\n\n{text}\n\nTranslation:"
    ),
    "qa": (
        "Answer the following question based on the context.\n\n"
        "Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    ),
    "code_review": (
        "Review the following code for bugs, style issues, and improvements:\n\n"
        "```{language}\n{code}\n```\n\nReview:"
    ),
    "classify": (
        "Classify the following text into one of these categories: {categories}\n\n"
        "Text: {text}\n\nCategory:"
    ),
    "extract": (
        "Extract {fields} from the following text:\n\n{text}\n\nExtracted:"
    ),
}


class PromptTemplate:
    """Manages prompt templates with variable interpolation and chaining.

    Supports ``{variable}`` style interpolation, few-shot examples,
    prompt chaining, and structural validation. Ships with built-in
    templates for common NLP tasks.

    Args:
        config: ThundersConfig instance.
        template: The prompt template string with ``{variable}`` placeholders.
        name: Optional template name.

    Example::

        pt = PromptTemplate(config, "Summarise: {text}")
        filled = pt.render(text="Long article ...")

        # Few-shot
        pt.add_example({"input": "Hi", "output": "Hello!"})
        result = pt.render(input="Hey")
    """

    _VAR_PATTERN = re.compile(r"\{(\w+)\}")

    def __init__(
        self,
        config: ThundersConfig,
        template: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None:
        self._config = config
        self._template = template or ""
        self._name = name or "unnamed"
        self._examples: List[Dict[str, str]] = []
        self._chain: List["PromptTemplate"] = []
        self._variables: List[str] = self._extract_variables(self._template)
        logger.info("PromptTemplate '%s' created (%d variables).", self._name, len(self._variables))

    # ------------------------------------------------------------------
    # Variable extraction / validation
    # ------------------------------------------------------------------

    @classmethod
    def _extract_variables(cls, template: str) -> List[str]:
        """Extract ``{variable}`` names from a template string."""
        return list(dict.fromkeys(cls._VAR_PATTERN.findall(template)))

    def validate(self, **kwargs: Any) -> List[str]:
        """Check that all required variables are provided.

        Returns:
            List of missing variable names (empty if all present).
        """
        return [v for v in self._variables if v not in kwargs]

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self, **kwargs: Any) -> str:
        """Interpolate variables into the template.

        Args:
            **kwargs: Variable values to fill into the template.

        Returns:
            The fully rendered prompt string.

        Raises:
            ValueError: If required variables are missing.
        """
        missing = self.validate(**kwargs)
        if missing:
            raise ValueError(f"Missing template variables: {missing}")

        # Build few-shot prefix
        prefix = ""
        for example in self._examples:
            for key, val in example.items():
                prefix += f"{key.capitalize()}: {val}\n"
            prefix += "\n"

        rendered = self._template.format(**kwargs)
        return prefix + rendered if prefix else rendered

    # ------------------------------------------------------------------
    # Few-shot examples
    # ------------------------------------------------------------------

    def add_example(self, example: Dict[str, str]) -> "PromptTemplate":
        """Add a few-shot example to the template.

        Args:
            example: Dict mapping variable names to example values.

        Returns:
            ``self`` for fluent chaining.
        """
        self._examples.append(example)
        logger.debug("Example added to template '%s'.", self._name)
        return self

    def set_examples(self, examples: List[Dict[str, str]]) -> "PromptTemplate":
        """Replace the few-shot example list.

        Args:
            examples: New list of example dicts.

        Returns:
            ``self`` for fluent chaining.
        """
        self._examples = list(examples)
        return self

    # ------------------------------------------------------------------
    # Chaining
    # ------------------------------------------------------------------

    def chain(self, next_template: "PromptTemplate") -> "PromptTemplate":
        """Create a new template that chains this template with another.

        The chained template renders this template first, then appends the
        output as the ``previous_output`` variable of the next template.

        Args:
            next_template: The template to chain after this one.

        Returns:
            A new :class:`PromptTemplate` representing the chain.
        """
        combined = self._template + "\n\n" + next_template._template
        chained = PromptTemplate(self._config, combined, name=f"{self._name}_chain")
        chained._variables = self._extract_variables(combined)
        chained._examples = self._examples + next_template._examples
        return chained

    # ------------------------------------------------------------------
    # Built-in templates
    # ------------------------------------------------------------------

    @classmethod
    def from_builtin(cls, config: ThundersConfig, name: str) -> "PromptTemplate":
        """Create a template from a built-in template by name.

        Args:
            config: ThundersConfig instance.
            name: Built-in template name (e.g. ``"summarise"``).

        Returns:
            A new :class:`PromptTemplate`.

        Raises:
            KeyError: If the name is not a known built-in.
        """
        if name not in BUILTIN_TEMPLATES:
            raise KeyError(f"Unknown built-in template: {name!r}. Available: {list(BUILTIN_TEMPLATES)}")
        return cls(config, BUILTIN_TEMPLATES[name], name=name)

    @classmethod
    def list_builtins(cls) -> List[str]:
        """Return the names of all built-in templates."""
        return list(BUILTIN_TEMPLATES.keys())

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def variables(self) -> List[str]:
        """List of template variable names."""
        return list(self._variables)

    @property
    def example_count(self) -> int:
        """Number of few-shot examples."""
        return len(self._examples)

    def __repr__(self) -> str:
        return f"PromptTemplate(name={self._name!r}, vars={self._variables})"
