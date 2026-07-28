"""LangChain bridge for Thunders AI.

Provides integration with the LangChain ecosystem, enabling
chain creation, agent orchestration, and tool management.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from thunders_ai.logger import get_logger

logger = get_logger(__name__)


class Tool:
    """Represents a callable tool that an agent can use.

    Attributes:
        name: Unique tool name.
        description: Human-readable description.
        func: The underlying callable.
    """

    def __init__(
        self,
        name: str,
        description: str,
        func: Callable[..., Any],
        parameters: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.name = name
        self.description = description
        self.func = func
        self.parameters = parameters or {}
        self.tool_id = f"tool-{uuid.uuid4().hex[:8]}"

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the tool."""
        return self.func(*args, **kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the tool metadata."""
        return {
            "tool_id": self.tool_id,
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


class ChainStep:
    """A single step within a chain execution.

    Attributes:
        name: Step identifier.
        action: The callable or tool to invoke.
        input_key: Key to read from the chain state.
        output_key: Key to write into the chain state.
    """

    def __init__(
        self,
        name: str,
        action: Callable[..., Any],
        input_key: str = "input",
        output_key: str = "output",
    ) -> None:
        self.name = name
        self.action = action
        self.input_key = input_key
        self.output_key = output_key


class LangChainBridge:
    """Bridge between Thunders AI and the LangChain ecosystem.

    Enables creation of chains and agents that combine Thunders AI
    models with LangChain tools, memory, and orchestration patterns.

    Attributes:
        chains: Registered chain definitions.
        agents: Registered agent definitions.
        tools: Registered tool definitions.
    """

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        default_memory_type: str = "buffer",
        verbose: bool = False,
    ) -> None:
        self.llm_client = llm_client
        self.default_memory_type = default_memory_type
        self.verbose = verbose
        self.chains: Dict[str, Dict[str, Any]] = {}
        self.agents: Dict[str, Dict[str, Any]] = {}
        self.tools: Dict[str, Tool] = {}

        logger.info("LangChainBridge initialised: verbose=%s", verbose)

    def create_chain(
        self,
        name: str,
        steps: Optional[List[Dict[str, Any]]] = None,
        memory_type: Optional[str] = None,
        return_intermediate: bool = False,
    ) -> Dict[str, Any]:
        """Create a new chain definition.

        Args:
            name: Chain identifier.
            steps: Ordered list of step configurations.
            memory_type: Memory strategy ('buffer', 'window', 'summary').
            return_intermediate: Whether to include intermediate outputs.

        Returns:
            Chain configuration dict.

        Raises:
            ValueError: If name already exists or steps are invalid.
        """
        if name in self.chains:
            raise ValueError(f"Chain '{name}' already exists")

        chain_steps: List[ChainStep] = []
        for i, step_cfg in enumerate(steps or []):
            step = ChainStep(
                name=step_cfg.get("name", f"step_{i}"),
                action=step_cfg.get("action", lambda x: x),
                input_key=step_cfg.get("input_key", "input"),
                output_key=step_cfg.get("output_key", "output"),
            )
            chain_steps.append(step)

        chain: Dict[str, Any] = {
            "chain_id": f"chain-{uuid.uuid4().hex[:8]}",
            "name": name,
            "steps": chain_steps,
            "memory_type": memory_type or self.default_memory_type,
            "return_intermediate": return_intermediate,
            "created_at": time.time(),
            "run_count": 0,
        }
        self.chains[name] = chain
        logger.info("Chain '%s' created with %d steps", name, len(chain_steps))
        return {
            "chain_id": chain["chain_id"],
            "name": name,
            "step_count": len(chain_steps),
            "memory_type": chain["memory_type"],
        }

    def create_agent(
        self,
        name: str,
        tool_names: Optional[List[str]] = None,
        agent_type: str = "react",
        max_iterations: int = 10,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Create a new agent with access to specified tools.

        Args:
            name: Agent identifier.
            tool_names: Tools the agent may use.
            agent_type: Agent strategy ('react', 'plan-and-execute').
            max_iterations: Maximum reasoning iterations.
            verbose: Enable verbose agent logging.

        Returns:
            Agent configuration dict.

        Raises:
            ValueError: If name already exists.
        """
        if name in self.agents:
            raise ValueError(f"Agent '{name}' already exists")

        tool_names = tool_names or []
        missing = [t for t in tool_names if t not in self.tools]
        if missing:
            logger.warning(
                "Agent '%s' references unregistered tools: %s", name, missing
            )

        agent: Dict[str, Any] = {
            "agent_id": f"agent-{uuid.uuid4().hex[:8]}",
            "name": name,
            "tool_names": tool_names,
            "agent_type": agent_type,
            "max_iterations": max_iterations,
            "verbose": verbose or self.verbose,
            "created_at": time.time(),
            "run_count": 0,
        }
        self.agents[name] = agent
        logger.info("Agent '%s' created: type=%s, tools=%s", name, agent_type, tool_names)
        return {
            "agent_id": agent["agent_id"],
            "name": name,
            "agent_type": agent_type,
            "tools": tool_names,
        }

    def add_tool(
        self,
        name: str,
        description: str,
        func: Callable[..., Any],
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Tool:
        """Register a tool that chains and agents can use.

        Args:
            name: Unique tool name.
            description: Human-readable description.
            func: The underlying callable.
            parameters: JSON-schema-like parameter specification.

        Returns:
            The registered Tool object.

        Raises:
            ValueError: If a tool with this name already exists.
        """
        if name in self.tools:
            raise ValueError(f"Tool '{name}' already exists")

        tool = Tool(
            name=name,
            description=description,
            func=func,
            parameters=parameters,
        )
        self.tools[name] = tool
        logger.info("Tool '%s' registered: %s", name, description[:80])
        return tool

    def run(
        self,
        chain_name: str,
        inputs: Dict[str, Any],
        timeout_seconds: float = 120.0,
    ) -> Dict[str, Any]:
        """Execute a registered chain.

        Args:
            chain_name: Name of the chain to run.
            inputs: Input values for the chain.
            timeout_seconds: Maximum execution time.

        Returns:
            Chain execution result.

        Raises:
            KeyError: If chain_name is not found.
            RuntimeError: If execution fails or times out.
        """
        if chain_name not in self.chains:
            raise KeyError(f"Chain '{chain_name}' not found")

        chain = self.chains[chain_name]
        chain["run_count"] += 1
        logger.info("Running chain '%s' (run #%d)", chain_name, chain["run_count"])

        state: Dict[str, Any] = dict(inputs)
        intermediate: List[Dict[str, Any]] = []
        start_time = time.time()

        try:
            for step in chain["steps"]:
                step_input = state.get(step.input_key, state)
                step_output = step.action(step_input)
                state[step.output_key] = step_output

                intermediate.append({
                    "step": step.name,
                    "input": str(step_input)[:200] if step_input else None,
                    "output": str(step_output)[:200] if step_output else None,
                    "elapsed": round(time.time() - start_time, 4),
                })

                if time.time() - start_time > timeout_seconds:
                    raise RuntimeError(f"Chain '{chain_name}' timed out after {timeout_seconds}s")

        except RuntimeError:
            raise
        except Exception as exc:
            logger.error("Chain '%s' failed at step: %s", chain_name, exc)
            raise RuntimeError(f"Chain execution failed: {exc}") from exc

        result: Dict[str, Any] = {
            "chain_name": chain_name,
            "output": state,
            "elapsed_seconds": round(time.time() - start_time, 4),
        }
        if chain["return_intermediate"]:
            result["intermediate"] = intermediate

        return result

    def run_agent(
        self,
        agent_name: str,
        query: str,
        max_iterations: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Execute a registered agent.

        Args:
            agent_name: Name of the agent to run.
            query: The query or task for the agent.
            max_iterations: Override agent's default max iterations.

        Returns:
            Agent execution result.

        Raises:
            KeyError: If agent_name is not found.
        """
        if agent_name not in self.agents:
            raise KeyError(f"Agent '{agent_name}' not found")

        agent = self.agents[agent_name]
        agent["run_count"] += 1
        max_iter = max_iterations or agent["max_iterations"]

        logger.info("Running agent '%s': query='%s...'", agent_name, query[:60])
        start_time = time.time()

        # Simulate agent reasoning loop
        reasoning_trace: List[Dict[str, Any]] = []
        for i in range(min(max_iter, 3)):
            thought = f"Thought {i + 1}: Analysing query..."
            action = "search" if i == 0 else "reason"
            observation = f"Observation from {action}"
            reasoning_trace.append({
                "iteration": i + 1,
                "thought": thought,
                "action": action,
                "observation": observation,
            })

        result: Dict[str, Any] = {
            "agent_name": agent_name,
            "query": query,
            "answer": f"[Agent response for: {query[:80]}...]",
            "reasoning_trace": reasoning_trace,
            "iterations_used": len(reasoning_trace),
            "elapsed_seconds": round(time.time() - start_time, 4),
        }
        return result
