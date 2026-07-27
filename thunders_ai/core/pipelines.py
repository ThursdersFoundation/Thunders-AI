"""Pipeline module for chaining AI operations with validation and caching."""

from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple

from thunders_ai.config import ThundersConfig
from thunders_ai.logger import get_logger

logger = get_logger(__name__)


class PipelineStep:
    """A single step in a processing pipeline.

    Args:
        name: Human-readable step name.
        func: Callable implementing the step logic.
        validator: Optional callable that validates the step output.
                 Should raise ``ValueError`` on invalid output.
    """

    def __init__(
        self,
        name: str,
        func: Callable[[Any], Any],
        validator: Optional[Callable[[Any], None]] = None,
    ) -> None:
        self.name = name
        self.func = func
        self.validator = validator

    def execute(self, data: Any) -> Any:
        """Run the step function and optionally validate the output."""
        logger.debug("PipelineStep '%s' executing …", self.name)
        result = self.func(data)
        if self.validator is not None:
            self.validator(result)
        return result


class Pipeline:
    """Chain multiple AI operations into a sequential or parallel pipeline.

    Supports custom steps, data validation between steps, result caching,
    and parallel execution of independent steps.

    Args:
        config: ThundersConfig instance.
        name: Optional pipeline name for logging.

    Example::

        pipe = Pipeline(config, "summarise")
        pipe.add_step("tokenize", tokenize_fn)
        pipe.add_step("encode", encode_fn)
        result = pipe.run("Long article text …")
    """

    _MAX_CACHE = 512

    def __init__(
        self,
        config: ThundersConfig,
        name: str = "default",
    ) -> None:
        self._config = config
        self._name = name
        self._steps: List[PipelineStep] = []
        self._parallel_groups: List[List[int]] = []  # groups of step indices
        self._cache: OrderedDict[str, Any] = OrderedDict()
        logger.info("Pipeline '%s' created.", name)

    # ------------------------------------------------------------------
    # Step management
    # ------------------------------------------------------------------

    def add_step(
        self,
        name: str,
        func: Callable[[Any], Any],
        validator: Optional[Callable[[Any], None]] = None,
    ) -> "Pipeline":
        """Append a new step to the pipeline.

        Args:
            name: Step name.
            func: Callable implementing the step.
            validator: Optional output validator.

        Returns:
            ``self`` for fluent chaining.
        """
        step = PipelineStep(name, func, validator)
        self._steps.append(step)
        logger.debug("Step '%s' added to pipeline '%s'.", name, self._name)
        return self

    def add_parallel_group(self, step_names: List[str], funcs: List[Callable]) -> "Pipeline":
        """Add a group of steps that can execute in parallel.

        All steps in the group receive the **same** input data and their
        outputs are merged into a dict keyed by step name.

        Args:
            step_names: Names for each parallel step.
            funcs: Callables, one per step name.

        Returns:
            ``self`` for fluent chaining.
        """
        indices: List[int] = []
        for name, func in zip(step_names, funcs):
            idx = len(self._steps)
            self._steps.append(PipelineStep(name, func))
            indices.append(idx)
        self._parallel_groups.append(indices)
        return self

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(self, data: Any, use_cache: bool = True) -> Any:
        """Execute the pipeline sequentially.

        Args:
            data: Initial input to the first step.
            use_cache: Return cached result if available.

        Returns:
            Output of the final step (or merged dict for parallel groups).

        Raises:
            ValueError: If a validator fails on any step.
        """
        cache_key = self._cache_key(data)
        if use_cache and cache_key in self._cache:
            logger.debug("Pipeline cache hit: %s", self._name)
            return self._cache[cache_key]

        # Determine which step indices belong to parallel groups
        parallel_indices: Dict[int, List[int]] = {}
        for group in self._parallel_groups:
            for idx in group:
                parallel_indices[idx] = group

        current = data
        executed = set()

        for i, step in enumerate(self._steps):
            if i in executed:
                continue

            if i in parallel_indices:
                group = parallel_indices[i]
                current = self._run_parallel_group(group, current)
                executed.update(group)
            else:
                t0 = time.perf_counter()
                current = step.execute(current)
                elapsed = time.perf_counter() - t0
                logger.info(
                    "Step '%s' completed in %.3fs", step.name, elapsed,
                )
                executed.add(i)

        if use_cache:
            self._cache[cache_key] = current
            self._cache.move_to_end(cache_key)
            if len(self._cache) > self._MAX_CACHE:
                self._cache.popitem(last=False)

        return current

    def _run_parallel_group(self, indices: List[int], data: Any) -> Dict[str, Any]:
        """Execute a group of steps in parallel using threads."""
        results: Dict[str, Any] = {}
        with ThreadPoolExecutor(max_workers=len(indices)) as pool:
            futures = {
                pool.submit(self._steps[idx].execute, data): self._steps[idx].name
                for idx in indices
            }
            for future in as_completed(futures):
                name = futures[future]
                results[name] = future.result()
                logger.debug("Parallel step '%s' completed.", name)
        return results

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------

    def _cache_key(self, data: Any) -> str:
        """Produce a deterministic cache key from input data and pipeline structure."""
        step_names = [s.name for s in self._steps]
        raw = json.dumps({"data_hash": hashlib.md5(str(data).encode()).hexdigest(),
                          "steps": step_names}, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()

    def clear_cache(self) -> None:
        """Clear the pipeline result cache."""
        self._cache.clear()
        logger.info("Pipeline '%s' cache cleared.", self._name)

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    @property
    def step_names(self) -> List[str]:
        """Return the ordered list of step names."""
        return [s.name for s in self._steps]

    @property
    def step_count(self) -> int:
        """Return the number of steps in the pipeline."""
        return len(self._steps)

    def __repr__(self) -> str:
        return f"Pipeline(name={self._name!r}, steps={self.step_count})"
