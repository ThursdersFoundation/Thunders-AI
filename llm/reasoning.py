"""Reasoning engine with chain-of-thought, multi-step, and self-reflection."""

from __future__ import annotations

import re
from enum import Enum
from typing import Any, Dict, List, Optional

from thunders_ai.config import ThundersConfig
from thunders_ai.core.engine import Engine
from thunders_ai.logger import get_logger

logger = get_logger(__name__)


class ReasoningStrategy(str, Enum):
    """Available reasoning strategies."""
    CHAIN_OF_THOUGHT = "chain_of_thought"
    TREE_OF_THOUGHT = "tree_of_thought"
    SELF_REFLECTION = "self_reflection"
    REACT = "react"


class ReasoningStep:
    """A single step in a reasoning trace."""

    __slots__ = ("step_number", "thought", "action", "observation", "confidence")

    def __init__(
        self,
        step_number: int,
        thought: str,
        action: Optional[str] = None,
        observation: Optional[str] = None,
        confidence: float = 1.0,
    ) -> None:
        self.step_number = step_number
        self.thought = thought
        self.action = action
        self.observation = observation
        self.confidence = confidence

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the reasoning step."""
        return {
            "step": self.step_number,
            "thought": self.thought,
            "action": self.action,
            "observation": self.observation,
            "confidence": self.confidence,
        }


class ReasoningTrace:
    """Complete trace of a multi-step reasoning process."""

    def __init__(self, strategy: ReasoningStrategy) -> None:
        self.strategy = strategy
        self.steps: List[ReasoningStep] = []
        self.conclusion: Optional[str] = None

    def add_step(self, step: ReasoningStep) -> None:
        """Append a step to the trace."""
        self.steps.append(step)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the full trace."""
        return {
            "strategy": self.strategy.value,
            "steps": [s.to_dict() for s in self.steps],
            "conclusion": self.conclusion,
        }


class ReasoningEngine:
    """Implements structured reasoning strategies for LLMs.

    Supports chain-of-thought, tree-of-thought, self-reflection, and
    ReAct-style reasoning. Produces detailed reasoning traces for
    interpretability.

    Args:
        config: ThundersConfig instance.
        engine: Engine used for generation.

    Example::

        re = ReasoningEngine(config, engine)
        trace = re.reason("How many tennis balls fit in a bus?", strategy=ReasoningStrategy.CHAIN_OF_THOUGHT)
        print(trace.conclusion)
    """

    # Prompt templates per strategy
    _STRATEGY_PROMPTS: Dict[ReasoningStrategy, str] = {
        ReasoningStrategy.CHAIN_OF_THOUGHT: (
            "Think step-by-step to answer the following question. "
            "Show your reasoning clearly.\n\nQuestion: {query}\n\nReasoning:"
        ),
        ReasoningStrategy.TREE_OF_THOUGHT: (
            "Generate 3 different approaches to solve the problem below. "
            "Evaluate each approach and select the best one.\n\nProblem: {query}\n\nApproaches:"
        ),
        ReasoningStrategy.SELF_REFLECTION: (
            "Answer the question below, then critique your answer for "
            "errors or omissions, and finally provide a revised answer.\n\n"
            "Question: {query}\n\nInitial answer:"
        ),
        ReasoningStrategy.REACT: (
            "Use the ReAct framework (Thought → Action → Observation) to "
            "solve the following. Repeat until you can give a final answer.\n\n"
            "Question: {query}\n\nThought 1:"
        ),
    }

    def __init__(self, config: ThundersConfig, engine: Engine) -> None:
        self._config = config
        self._engine = engine
        self._max_steps: int = getattr(config, "max_reasoning_steps", 8)
        logger.info("ReasoningEngine initialized – max_steps=%d", self._max_steps)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reason(
        self,
        query: str,
        strategy: ReasoningStrategy = ReasoningStrategy.CHAIN_OF_THOUGHT,
        max_steps: Optional[int] = None,
        **kwargs: Any,
    ) -> ReasoningTrace:
        """Execute a reasoning process and return the full trace.

        Args:
            query: The question or problem to reason about.
            strategy: Reasoning strategy to apply.
            max_steps: Override max reasoning steps.
            **kwargs: Extra generation kwargs.

        Returns:
            A :class:`ReasoningTrace` with all steps and a conclusion.
        """
        steps_limit = max_steps or self._max_steps
        trace = ReasoningTrace(strategy)

        if strategy == ReasoningStrategy.CHAIN_OF_THOUGHT:
            self._chain_of_thought(query, trace, steps_limit, **kwargs)
        elif strategy == ReasoningStrategy.TREE_OF_THOUGHT:
            self._tree_of_thought(query, trace, **kwargs)
        elif strategy == ReasoningStrategy.SELF_REFLECTION:
            self._self_reflection(query, trace, **kwargs)
        elif strategy == ReasoningStrategy.REACT:
            self._react(query, trace, steps_limit, **kwargs)

        logger.info(
            "Reasoning complete – strategy=%s, steps=%d",
            strategy.value, len(trace.steps),
        )
        return trace

    # ------------------------------------------------------------------
    # Strategy implementations
    # ------------------------------------------------------------------

    def _chain_of_thought(
        self,
        query: str,
        trace: ReasoningTrace,
        max_steps: int,
        **kwargs: Any,
    ) -> None:
        """Chain-of-thought: generate step-by-step reasoning."""
        prompt = self._STRATEGY_PROMPTS[ReasoningStrategy.CHAIN_OF_THOUGHT].format(query=query)
        raw = self._engine.generate(prompt, max_new_tokens=512, **kwargs)

        # Parse steps from the generated text
        segments = re.split(r"(?:Step \d+|^\d+\.)", raw, flags=re.MULTILINE)
        for i, segment in enumerate(segments[:max_steps]):
            text = segment.strip()
            if not text:
                continue
            trace.add_step(ReasoningStep(step_number=i + 1, thought=text))

        # Extract conclusion (last paragraph)
        paragraphs = [p.strip() for p in raw.split("\n\n") if p.strip()]
        trace.conclusion = paragraphs[-1] if paragraphs else raw.strip()

    def _tree_of_thought(
        self,
        query: str,
        trace: ReasoningTrace,
        **kwargs: Any,
    ) -> None:
        """Tree-of-thought: generate multiple approaches, select best."""
        prompt = self._STRATEGY_PROMPTS[ReasoningStrategy.TREE_OF_THOUGHT].format(query=query)
        raw = self._engine.generate(prompt, max_new_tokens=768, **kwargs)

        approaches = re.split(r"Approach \d+:", raw)
        for i, approach in enumerate(approaches):
            text = approach.strip()
            if not text:
                continue
            trace.add_step(
                ReasoningStep(step_number=i + 1, thought=text, confidence=0.7)
            )

        # Ask the model to select the best approach
        eval_prompt = f"Given these approaches, which is best and why?\n\n{raw}\n\nBest approach:"
        best = self._engine.generate(eval_prompt, max_new_tokens=256, **kwargs)
        trace.conclusion = best.strip()

    def _self_reflection(
        self,
        query: str,
        trace: ReasoningTrace,
        **kwargs: Any,
    ) -> None:
        """Self-reflection: answer, critique, then revise."""
        prompt = self._STRATEGY_PROMPTS[ReasoningStrategy.SELF_REFLECTION].format(query=query)
        raw = self._engine.generate(prompt, max_new_tokens=768, **kwargs)

        sections = re.split(r"(?:Critique:|Revised answer:)", raw)
        labels = ["initial_answer", "critique", "revised_answer"]

        for i, section in enumerate(sections):
            text = section.strip()
            if not text:
                continue
            label = labels[min(i, len(labels) - 1)]
            trace.add_step(
                ReasoningStep(step_number=i + 1, thought=text, action=label)
            )

        trace.conclusion = sections[-1].strip() if len(sections) > 2 else raw.strip()

    def _react(
        self,
        query: str,
        trace: ReasoningTrace,
        max_steps: int,
        **kwargs: Any,
    ) -> None:
        """ReAct: iterative Thought-Action-Observation loop."""
        context = self._STRATEGY_PROMPTS[ReasoningStrategy.REACT].format(query=query)

        for step_num in range(1, max_steps + 1):
            output = self._engine.generate(context, max_new_tokens=256, **kwargs)

            # Parse thought / action / observation
            thought_match = re.search(r"Thought:\s*(.+?)(?:Action:|$)", output, re.DOTALL)
            action_match = re.search(r"Action:\s*(.+?)(?:Observation:|$)", output, re.DOTALL)
            obs_match = re.search(r"Observation:\s*(.+?)$", output, re.DOTALL)

            thought = thought_match.group(1).strip() if thought_match else output.strip()
            action = action_match.group(1).strip() if action_match else None
            observation = obs_match.group(1).strip() if obs_match else None

            trace.add_step(
                ReasoningStep(
                    step_number=step_num,
                    thought=thought,
                    action=action,
                    observation=observation,
                )
            )

            # Detect final answer
            if "final answer" in output.lower() or action is None:
                trace.conclusion = thought
                break

            context += f"\n{output}\nObservation: [simulated]\nThought {step_num + 1}:"

        if trace.conclusion is None and trace.steps:
            trace.conclusion = trace.steps[-1].thought
