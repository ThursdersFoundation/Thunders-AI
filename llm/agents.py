"""Autonomous AI agent with goal-oriented behaviour and tool usage."""

from __future__ import annotations

import time
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from thunders_ai.config import ThundersConfig
from thunders_ai.core.engine import Engine
from thunders_ai.llm.tools import ToolRegistry
from thunders_ai.logger import get_logger

logger = get_logger(__name__)


class AgentState(str, Enum):
    """Agent lifecycle states."""
    IDLE = "idle"
    PLANNING = "planning"
    EXECUTING = "executing"
    OBSERVING = "observing"
    REFLECTING = "reflecting"
    COMPLETED = "completed"
    FAILED = "failed"


class Observation:
    """An observation made by the agent during execution."""

    __slots__ = ("content", "timestamp", "source")

    def __init__(self, content: str, source: str = "environment") -> None:
        self.content = content
        self.timestamp = time.time()
        self.source = source

    def to_dict(self) -> Dict[str, Any]:
        return {"content": self.content, "source": self.source, "timestamp": self.timestamp}


class Agent:
    """Autonomous AI agent that plans and executes multi-step tasks.

    Implements an observe–plan–act loop with tool usage, reflection,
    and optional collaboration with other agents.

    Args:
        config: ThundersConfig instance.
        engine: Engine for LLM generation.
        tool_registry: Optional :class:`~thunders_ai.llm.tools.ToolRegistry`.
        name: Agent name for logging.

    Example::

        agent = Agent(config, engine, name="ResearchBot")
        result = agent.run("Summarise the latest papers on RAG")
    """

    def __init__(
        self,
        config: ThundersConfig,
        engine: Engine,
        tool_registry: Optional[ToolRegistry] = None,
        name: str = "Agent",
    ) -> None:
        self._config = config
        self._engine = engine
        self._tool_registry = tool_registry or ToolRegistry(config)
        self._name = name
        self._state = AgentState.IDLE
        self._goal: Optional[str] = None
        self._plan: List[str] = []
        self._observations: List[Observation] = []
        self._actions_log: List[Dict[str, Any]] = []
        self._max_iterations: int = getattr(config, "agent_max_iterations", 10)
        logger.info("Agent '%s' initialized.", name)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def state(self) -> AgentState:
        """Current agent state."""
        return self._state

    @property
    def name(self) -> str:
        """Agent name."""
        return self._name

    def run(self, goal: str, context: Optional[str] = None) -> str:
        """Execute a goal through the observe–plan–act loop.

        Args:
            goal: The objective to accomplish.
            context: Optional additional context.

        Returns:
            Final result or summary string.
        """
        self._goal = goal
        self._state = AgentState.PLANNING
        self._observations.clear()
        self._actions_log.clear()
        logger.info("Agent '%s' starting goal: %s", self._name, goal)

        # Step 1: Create a plan
        plan_prompt = (
            f"You are an AI agent named {self._name}. "
            f"Create a step-by-step plan to accomplish the following goal:\n"
            f"Goal: {goal}\n"
        )
        if context:
            plan_prompt += f"Context: {context}\n"
        plan_prompt += "\nPlan (numbered steps):"

        plan_text = self._engine.generate(plan_prompt, max_new_tokens=512)
        self._plan = [
            line.strip().lstrip("0123456789.-) ")
            for line in plan_text.split("\n")
            if line.strip() and line.strip()[0].isdigit()
        ]
        if not self._plan:
            self._plan = [goal]

        logger.info("Plan created with %d steps.", len(self._plan))

        # Step 2: Execute the plan iteratively
        self._state = AgentState.EXECUTING
        iteration = 0
        final_result = ""

        while iteration < self._max_iterations and self._state == AgentState.EXECUTING:
            iteration += 1
            obs_summary = self._summarise_observations()

            step_prompt = (
                f"Goal: {self._goal}\n"
                f"Plan: {self._plan}\n"
                f"Observations so far: {obs_summary}\n"
                f"Current step: {self._current_step_description(iteration)}\n\n"
                f"What action should be taken? Provide the action and any tool calls needed."
            )

            action_text = self._engine.generate(step_prompt, max_new_tokens=256)
            self._actions_log.append({
                "iteration": iteration,
                "action": action_text,
                "timestamp": time.time(),
            })

            # Attempt to use tools if mentioned
            observation_text = self._try_use_tools(action_text)
            self._observations.append(Observation(observation_text))

            # Check for completion
            if "task complete" in action_text.lower() or "done" in action_text.lower():
                self._state = AgentState.COMPLETED
                final_result = action_text
                break

            if iteration >= len(self._plan):
                # Reflect on progress
                self._state = AgentState.REFLECTING
                reflection = self._reflect()
                if "complete" in reflection.lower():
                    self._state = AgentState.COMPLETED
                    final_result = reflection
                else:
                    self._state = AgentState.EXECUTING

        if self._state != AgentState.COMPLETED:
            self._state = AgentState.FAILED
            final_result = f"Agent did not complete the goal in {self._max_iterations} iterations."

        logger.info("Agent '%s' finished – state=%s", self._name, self._state.value)
        return final_result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _current_step_description(self, iteration: int) -> str:
        """Return a description of the current plan step."""
        idx = min(iteration - 1, len(self._plan) - 1)
        return self._plan[idx] if self._plan else "No plan step"

    def _summarise_observations(self) -> str:
        """Build a brief summary of all observations so far."""
        if not self._observations:
            return "None yet."
        return " | ".join(o.content[:100] for o in self._observations[-3:])

    def _try_use_tools(self, action_text: str) -> str:
        """Check if the action mentions a tool and try to execute it.

        Returns:
            Observation text (tool result or action text itself).
        """
        available = self._tool_registry.list_tools()
        for tool_name in available:
            if tool_name.lower() in action_text.lower():
                try:
                    result = self._tool_registry.execute(tool_name, action_text)
                    return f"Tool '{tool_name}' result: {result}"
                except Exception as exc:
                    return f"Tool '{tool_name}' error: {exc}"
        return action_text

    def _reflect(self) -> str:
        """Reflect on progress and decide whether to continue."""
        obs_summary = self._summarise_observations()
        reflect_prompt = (
            f"Goal: {self._goal}\n"
            f"Observations: {obs_summary}\n"
            f"Are we done? If yes, summarise the result. If no, suggest next steps."
        )
        reflection = self._engine.generate(reflect_prompt, max_new_tokens=256)
        return reflection.strip()

    # ------------------------------------------------------------------
    # Collaboration
    # ------------------------------------------------------------------

    def delegate(self, other: "Agent", sub_goal: str) -> str:
        """Delegate a sub-goal to another agent.

        Args:
            other: The collaborating agent.
            sub_goal: The sub-task description.

        Returns:
            The other agent's result.
        """
        logger.info("Agent '%s' delegating to '%s': %s", self._name, other.name, sub_goal)
        return other.run(sub_goal)
