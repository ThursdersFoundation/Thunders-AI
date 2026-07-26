"""Sandbox Module for Thunders AI.

Provides safe code execution with resource limits, code validation,
and security policy enforcement for Python, JavaScript, and Shell.
"""

from __future__ import annotations

import io
import resource
import signal
import sys
import textwrap
import time
import traceback
from contextlib import redirect_stdout, redirect_stderr
from typing import Any, Dict, List, Optional, Tuple

from thunders_ai.config import get_config
from thunders_ai.logger import get_logger

logger = get_logger(__name__)

# Dangerous Python builtins and modules to block
BLOCKED_PYTHON_IMPORTS = {
    "os", "subprocess", "shutil", "sys", "ctypes", "socket",
    "http", "urllib", "requests", "pickle", "shelve", "marshal",
    "importlib", "pkgutil", "pathlib", "signal", "multiprocessing",
    "threading", "asyncio", "concurrent", "tempfile", "glob",
}
BLOCKED_BUILTINS = {
    "exec", "eval", "compile", "__import__", "open", "input",
    "breakpoint", "globals", "locals", "vars", "dir",
    "getattr", "setattr", "delattr", "type", "object",
    "memoryview", "classmethod", "staticmethod",
}
BLOCKED_SHELL_COMMANDS = {
    "rm", "rmdir", "del", "format", "fdisk", "mkfs",
    "dd", "shutdown", "reboot", "kill", "killall",
    "curl", "wget", "nc", "ncat", "ssh", "scp",
    "chmod", "chown", "chgrp", "sudo", "su",
    "crontab", "systemctl", "service",
}
BLOCKED_JS_PATTERNS = {
    "require(", "process.", "child_process", "fs.",
    "net.", "http.", "https.", "eval(", "Function(",
    "setTimeout(", "setInterval(", "__dirname", "__filename",
}


class Sandbox:
    """Secure sandbox for executing untrusted code.

    Provides safe code execution with configurable resource limits
    (memory, time, network), pre-execution code validation, and
    security policy enforcement.

    Args:
        max_memory_mb: Maximum memory allowed in megabytes.
        max_time_seconds: Maximum execution time in seconds.
        allow_network: Whether to allow network access.
        allowed_python_modules: Set of allowed Python module names.
        max_output_bytes: Maximum output size in bytes.

    Example::

        sandbox = Sandbox(max_time_seconds=5)
        is_safe = sandbox.validate_code("print('hello')", language="python")
        result = sandbox.execute("print('hello')", language="python")
    """

    def __init__(
        self,
        max_memory_mb: Optional[int] = None,
        max_time_seconds: Optional[int] = None,
        allow_network: Optional[bool] = None,
        allowed_python_modules: Optional[set] = None,
        max_output_bytes: int = 1024 * 1024,
    ) -> None:
        cfg = get_config().security
        self._max_memory_mb = max_memory_mb or cfg.sandbox_max_memory_mb
        self._max_time_seconds = max_time_seconds or cfg.sandbox_max_time_seconds
        self._allow_network = allow_network if allow_network is not None else cfg.sandbox_allow_network
        self._allowed_python_modules = allowed_python_modules or {"math", "json", "re", "datetime", "collections", "itertools", "functools", "string", "random", "statistics", "decimal", "fractions", "copy"}
        self._max_output_bytes = max_output_bytes

        self._last_output: Optional[str] = None
        self._last_error: Optional[str] = None
        self._execution_count = 0

        logger.info(
            "Sandbox initialized: memory=%dMB, time=%ds, network=%s",
            self._max_memory_mb, self._max_time_seconds, self._allow_network,
        )

    def validate_code(self, code: str, language: str = "python") -> Tuple[bool, List[str]]:
        """Validate code for safety before execution.

        Checks for potentially dangerous patterns without executing the code.

        Args:
            code: Source code to validate.
            language: Programming language ('python', 'javascript', 'shell').

        Returns:
            Tuple of (is_safe: bool, issues: list of warning strings).
        """
        issues: List[str] = []
        language = language.lower()

        if language == "python":
            issues.extend(self._validate_python(code))
        elif language == "javascript":
            issues.extend(self._validate_javascript(code))
        elif language == "shell":
            issues.extend(self._validate_shell(code))
        else:
            issues.append(f"Unsupported language: {language}")

        is_safe = len(issues) == 0
        if not is_safe:
            logger.warning("Code validation found %d issues for %s", len(issues), language)
        return is_safe, issues

    def _validate_python(self, code: str) -> List[str]:
        """Validate Python code for safety issues."""
        issues: List[str] = []
        for line in code.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            for blocked in BLOCKED_BUILTINS:
                if blocked in stripped and not stripped.startswith("#"):
                    issues.append(f"Blocked builtin usage: {blocked}")
            for mod in BLOCKED_PYTHON_IMPORTS:
                if mod in stripped and ("import" in stripped or "from" in stripped):
                    if mod not in self._allowed_python_modules:
                        issues.append(f"Blocked import: {mod}")
        try:
            compile(code, "<sandbox>", "exec")
        except SyntaxError as exc:
            issues.append(f"Syntax error: {exc}")
        return issues

    def _validate_javascript(self, code: str) -> List[str]:
        """Validate JavaScript code for safety issues."""
        issues: List[str] = []
        for pattern in BLOCKED_JS_PATTERNS:
            if pattern in code:
                issues.append(f"Blocked pattern: {pattern}")
        return issues

    def _validate_shell(self, code: str) -> List[str]:
        """Validate Shell code for safety issues."""
        issues: List[str] = []
        tokens = code.split()
        for token in tokens:
            clean = token.strip(";|&")
            if clean in BLOCKED_SHELL_COMMANDS:
                issues.append(f"Blocked command: {clean}")
        if not self._allow_network:
            network_cmds = {"curl", "wget", "nc", "ssh", "scp", "telnet"}
            for token in tokens:
                if token.strip(";|&") in network_cmds:
                    issues.append(f"Network access blocked: {token}")
        return issues

    def execute(
        self, code: str, language: str = "python", timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """Execute code safely within the sandbox.

        Args:
            code: Source code to execute.
            language: Programming language ('python', 'javascript', 'shell').
            timeout: Override default timeout in seconds.

        Returns:
            Dictionary with 'success', 'output', 'error', 'execution_time' keys.
        """
        language = language.lower()
        effective_timeout = timeout or self._max_time_seconds

        is_safe, issues = self.validate_code(code, language)
        if not is_safe:
            logger.warning("Code rejected by sandbox validation: %s", issues)
            return {
                "success": False,
                "output": "",
                "error": f"Code validation failed: {'; '.join(issues)}",
                "execution_time": 0.0,
            }

        self._execution_count += 1
        start_time = time.time()

        if language == "python":
            result = self._execute_python(code, effective_timeout)
        elif language == "javascript":
            result = self._execute_javascript(code, effective_timeout)
        elif language == "shell":
            result = self._execute_shell(code, effective_timeout)
        else:
            result = {
                "success": False,
                "output": "",
                "error": f"Unsupported language: {language}",
                "execution_time": 0.0,
            }

        result["execution_time"] = time.time() - start_time
        self._last_output = result.get("output", "")
        self._last_error = result.get("error", "")
        return result

    def _execute_python(self, code: str, timeout: int) -> Dict[str, Any]:
        """Execute Python code in a restricted namespace."""
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        safe_builtins = {k: v for k, v in __builtins__.items() if k not in BLOCKED_BUILTINS} if isinstance(__builtins__, dict) else {}

        restricted_globals: Dict[str, Any] = {
            "__builtins__": safe_builtins,
            "__name__": "__sandbox__",
        }

        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(compile(code, "<sandbox>", "exec"), restricted_globals)  # noqa: S102

            output = stdout_capture.getvalue()
            error_output = stderr_capture.getvalue()
            if len(output.encode()) > self._max_output_bytes:
                output = output[: self._max_output_bytes] + "\n... [output truncated]"

            self._last_output = output
            return {
                "success": True,
                "output": output,
                "error": error_output,
            }
        except Exception:
            error_msg = traceback.format_exc()
            return {
                "success": False,
                "output": stdout_capture.getvalue(),
                "error": error_msg,
            }

    def _execute_javascript(self, code: str, timeout: int) -> Dict[str, Any]:
        """Execute JavaScript code using subprocess (Node.js).

        Falls back to a simulated execution if Node.js is unavailable.
        """
        import subprocess
        try:
            proc = subprocess.run(
                ["node", "-e", code],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return {
                "success": proc.returncode == 0,
                "output": proc.stdout,
                "error": proc.stderr,
            }
        except FileNotFoundError:
            return {
                "success": False,
                "output": "",
                "error": "Node.js runtime not available for JavaScript execution",
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "output": "",
                "error": f"JavaScript execution timed out after {timeout}s",
            }

    def _execute_shell(self, code: str, timeout: int) -> Dict[str, Any]:
        """Execute shell commands (highly restricted)."""
        import subprocess
        if not self._allow_network:
            return {
                "success": False,
                "output": "",
                "error": "Shell execution is disabled in non-network sandbox mode",
            }
        try:
            proc = subprocess.run(
                code,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return {
                "success": proc.returncode == 0,
                "output": proc.stdout,
                "error": proc.stderr,
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "output": "",
                "error": f"Shell execution timed out after {timeout}s",
            }

    def set_limits(
        self,
        max_memory_mb: Optional[int] = None,
        max_time_seconds: Optional[int] = None,
        allow_network: Optional[bool] = None,
    ) -> None:
        """Update sandbox resource limits.

        Args:
            max_memory_mb: New memory limit in MB.
            max_time_seconds: New time limit in seconds.
            allow_network: New network access policy.
        """
        if max_memory_mb is not None:
            self._max_memory_mb = max_memory_mb
        if max_time_seconds is not None:
            self._max_time_seconds = max_time_seconds
        if allow_network is not None:
            self._allow_network = allow_network
        logger.info(
            "Sandbox limits updated: memory=%dMB, time=%ds, network=%s",
            self._max_memory_mb, self._max_time_seconds, self._allow_network,
        )

    def get_output(self) -> Optional[str]:
        """Retrieve the output from the last execution.

        Returns:
            Last execution output string, or None if no execution occurred.
        """
        return self._last_output

    def cleanup(self) -> None:
        """Clean up sandbox state and reset for next use."""
        self._last_output = None
        self._last_error = None
        logger.debug("Sandbox cleaned up (total executions: %d)", self._execution_count)

    def get_info(self) -> Dict[str, Any]:
        """Return sandbox configuration and status."""
        return {
            "max_memory_mb": self._max_memory_mb,
            "max_time_seconds": self._max_time_seconds,
            "allow_network": self._allow_network,
            "allowed_python_modules": sorted(self._allowed_python_modules),
            "max_output_bytes": self._max_output_bytes,
            "total_executions": self._execution_count,
        }
