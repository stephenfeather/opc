"""Regression test for issue #204: math_router used `os` without importing it.

The module's top-level `faulthandler.enable(...)` call references
`os.path.expanduser(...)`, so importing the module raised
`NameError: name 'os' is not defined` before the fix.
"""

import importlib


def test_math_router_imports_without_nameerror():
    """Importing math_router must not raise (os must be imported)."""
    module = importlib.import_module("scripts.cc_math.math_router")
    assert module is not None
    # The symbol the buggy line depended on must be resolvable at module scope.
    assert hasattr(module, "os")
