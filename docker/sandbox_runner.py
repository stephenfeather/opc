#!/usr/bin/env python3
"""Sandbox Runner - Executes code in isolated Docker container.

This script runs inside the Docker container and:
1. Reads JSON request from stdin
2. Executes code with restricted globals
3. Returns JSON result to stdout

Security features:
- Resource limits (CPU, RAM) via resource module
- Timeout via signal.alarm
- Restricted builtins (no eval, exec, open, etc.)
- Pre-imported safe math libraries only

Input format:
    {"code": "python code", "timeout": 30}

Output format (success):
    {"success": true, "result": <value>, "variables": {"var": <value>}}

Output format (failure):
    {"success": false, "error": "message", "error_type": "TypeError"}
"""

from __future__ import annotations

import json
import resource
import signal
import sys
from typing import Any

import os
import faulthandler
faulthandler.enable(file=open(os.path.expanduser("~/.claude/logs/opc_crash.log"), "a"), all_threads=True)

# Set resource limits
CPU_LIMIT = 30  # seconds
RAM_LIMIT = 512 * 1024 * 1024  # 512 MB

resource.setrlimit(resource.RLIMIT_CPU, (CPU_LIMIT, CPU_LIMIT))
try:
    resource.setrlimit(resource.RLIMIT_AS, (RAM_LIMIT, RAM_LIMIT))
except ValueError:
    # Some systems don't support RLIMIT_AS
    pass


def timeout_handler(signum, frame):
    """Handle execution timeout."""
    raise TimeoutError("Execution timed out")


signal.signal(signal.SIGALRM, timeout_handler)


def serialize_result(obj: Any) -> Any:
    """Serialize result to JSON-compatible format.

    Args:
        obj: Python object to serialize

    Returns:
        JSON-serializable value
    """
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [serialize_result(item) for item in obj]
    if isinstance(obj, dict):
        return {str(k): serialize_result(v) for k, v in obj.items()}
    if hasattr(obj, "tolist"):  # numpy array
        return obj.tolist()
    if hasattr(obj, "__dict__"):
        return str(obj)
    return str(obj)


def execute_code(code: str, timeout: int = 30) -> dict:
    """Execute code in sandbox.

    Args:
        code: Python code to execute
        timeout: Maximum execution time in seconds

    Returns:
        Result dict with success, result/error fields
    """
    signal.alarm(timeout)

    try:
        # Create restricted globals
        restricted_globals = {
            "__builtins__": {
                # Core types
                "print": print,
                "len": len,
                "range": range,
                "enumerate": enumerate,
                "zip": zip,
                "map": map,
                "filter": filter,
                "sorted": sorted,
                "reversed": reversed,
                "list": list,
                "dict": dict,
                "set": set,
                "tuple": tuple,
                "frozenset": frozenset,
                "str": str,
                "int": int,
                "float": float,
                "bool": bool,
                "bytes": bytes,
                "bytearray": bytearray,
                "complex": complex,
                "type": type,
                "isinstance": isinstance,
                "issubclass": issubclass,
                "hasattr": hasattr,
                "getattr": getattr,
                "setattr": setattr,
                "delattr": delattr,
                # Math
                "abs": abs,
                "round": round,
                "pow": pow,
                "min": min,
                "max": max,
                "sum": sum,
                "divmod": divmod,
                # Iteration
                "next": next,
                "iter": iter,
                "all": all,
                "any": any,
                # Other safe builtins
                "repr": repr,
                "hash": hash,
                "id": id,
                "callable": callable,
                "format": format,
                "slice": slice,
                "property": property,
                "staticmethod": staticmethod,
                "classmethod": classmethod,
                "super": super,
                "object": object,
                "True": True,
                "False": False,
                "None": None,
                # Explicitly excluded for security:
                # - eval, exec, compile
                # - open, input
                # - __import__, importlib
                # - globals, locals, vars
                # - exit, quit
            },
        }

        # Import allowed modules
        import numpy as np
        import pandas as pd
        import scipy as sp
        import sympy

        restricted_globals["np"] = np
        restricted_globals["numpy"] = np
        restricted_globals["sp"] = sp
        restricted_globals["scipy"] = sp
        restricted_globals["sympy"] = sympy
        restricted_globals["pd"] = pd
        restricted_globals["pandas"] = pd

        local_vars: dict = {}
        exec(code, restricted_globals, local_vars)

        signal.alarm(0)  # Cancel timeout

        # Extract result
        result = local_vars.get("result", None)

        return {
            "success": True,
            "result": serialize_result(result),
            "variables": {
                k: serialize_result(v) for k, v in local_vars.items() if not k.startswith("_")
            },
        }

    except TimeoutError as e:
        signal.alarm(0)
        return {
            "success": False,
            "error": str(e),
            "error_type": "TimeoutError",
        }
    except MemoryError:
        signal.alarm(0)
        return {
            "success": False,
            "error": "Memory limit exceeded",
            "error_type": "MemoryError",
        }
    except Exception as e:
        signal.alarm(0)
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
        }


def main():
    """Main entry point - read request, execute, return result."""
    try:
        request = json.load(sys.stdin)
    except json.JSONDecodeError as e:
        result = {
            "success": False,
            "error": f"Invalid JSON input: {e}",
            "error_type": "JSONDecodeError",
        }
        json.dump(result, sys.stdout)
        return

    code = request.get("code", "")
    timeout = request.get("timeout", 30)

    result = execute_code(code, timeout)
    json.dump(result, sys.stdout)


if __name__ == "__main__":
    main()
