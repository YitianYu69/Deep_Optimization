import importlib.util
import os
import sys

# Path to the compiled .so file
_so_path = os.path.join(os.path.dirname(__file__), "cpp_extension.so")

# Dynamically load the .so as the package itself
_spec = importlib.util.spec_from_file_location(__name__, _so_path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)

# Replace current package with loaded module
sys.modules[__name__] = _module
