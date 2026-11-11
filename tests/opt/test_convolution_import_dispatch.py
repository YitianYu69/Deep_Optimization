"""Real import binding must not replace lazy convolution launch callables."""

import subprocess
import sys
import textwrap
import unittest


class ConvolutionImportDispatchTests(unittest.TestCase):
    def run_fresh(self, preload):
        # Substitute only CUDA launch implementations. Python's real import
        # machinery still binds their modules on the actual backend package.
        # This reproduces the namespace collision on CPU-only installations.
        script = textwrap.dedent(f"""
            import importlib
            import importlib.abc
            import importlib.util
            from importlib.resources import files
            import sys
            from types import SimpleNamespace
            import torch
            from deepopt.opt.backends import convolution as executor

            prefix = "deepopt.opt.backends.triton.convolution"
            calls = []

            class LaunchLoader(importlib.abc.MetaPathFinder, importlib.abc.Loader):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname in (prefix + ".forward", prefix + ".forward_impl", prefix + ".backward"):
                        return importlib.util.spec_from_loader(fullname, self)

                def create_module(self, spec):
                    return None

                def exec_module(self, module):
                    def launch(name):
                        def call(*args, **kwargs):
                            calls.append((name, args, kwargs))
                            return name
                        return call
                    if module.__name__.endswith(".backward"):
                        module.input_gradient = launch("input_gradient")
                        module.weight_gradient = launch("weight_gradient")
                    else:
                        module.forward = launch("forward")

            sys.meta_path.insert(0, LaunchLoader())
            package = importlib.import_module(prefix)
            implementation = "forward_impl" if files(package).joinpath("forward_impl.py").is_file() else "forward"
            if {preload!r}:
                # Transposed dX and direct kernel imports can load this module
                # before the first ordinary convolution forward.
                importlib.import_module(prefix + "." + implementation)
                importlib.import_module(prefix + ".backward")
            assert "triton" not in sys.modules
            operand = SimpleNamespace(data=torch.empty(1))
            weight, geometry = object(), object()
            for backend in ("triton", "native_fast", "native_precise"):
                for iteration in range(4):
                    assert executor.forward(operand, weight, geometry, backend=backend) == "forward"
                    assert executor.input_gradient(operand, weight, geometry, backend=backend) == "input_gradient"
                    assert executor.weight_gradient(operand, weight, geometry, backend=backend) == "weight_gradient"
                    for name in ("forward", "input_gradient", "weight_gradient"):
                        assert callable(getattr(package, name)), name
            assert len(calls) == 36
            assert all(args == (operand, weight, geometry) for _, args, _ in calls)
            assert "triton" not in sys.modules
        """)
        result = subprocess.run([sys.executable, "-B", "-c", script], text=True, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_repeated_warmup_forward_and_backward_dispatch(self):
        self.run_fresh(False)

    def test_dispatch_after_direct_implementation_imports(self):
        self.run_fresh(True)


if __name__ == "__main__":
    unittest.main()
