"""Installed-only imports reject editable packages and checkout shadowing."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


HELPER = Path(__file__).resolve().parents[2] / "benchmarks/compiler/installed_runtime.py"


class InstalledBackendRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="installed backend ")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.site = self.root / "site-packages"
        package = self.site / "deepopt"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("marker = 'installed'\n")
        metadata = self.site / "deep_optimization-9.0.dist-info"
        metadata.mkdir()
        (metadata / "METADATA").write_text("Metadata-Version: 2.1\nName: deep-optimization\nVersion: 9.0\n")
        (metadata / "RECORD").write_text("deepopt/__init__.py,,\n")
        self.metadata = metadata
        target = self.root / "benchmarks/compiler"
        target.mkdir(parents=True)
        shutil.copyfile(HELPER, target / "installed_runtime.py")
        self.helper = target / "installed_runtime.py"

    def run_script(self, code, extra_paths=()):
        header = f'''import importlib.util, json, os, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location("runtime", {str(self.helper)!r})
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)
root = Path({str(self.root)!r})
'''
        return subprocess.run([sys.executable, "-S", "-c", header + code], cwd=self.root,
            env={**os.environ, "PYTHONPATH": os.pathsep.join(map(str, (*extra_paths, self.site))),
                 "DEEPOPT_QUALIFICATION_INSTALLED_IDENTITY": ""},
            capture_output=True, text=True, timeout=15)

    def test_installed_distribution_wins_over_its_checkout_src(self):
        shadow = self.root / "src/deepopt"
        shadow.mkdir(parents=True)
        (shadow / "__init__.py").write_text("raise AssertionError('checkout source imported')\n")
        result = self.run_script("print(json.dumps(runtime.installed_identity(root)))", (self.root / "src",))
        self.assertEqual(result.returncode, 0, result.stderr)
        identity = json.loads(result.stdout)
        self.assertEqual(identity["version"], "9.0")
        self.assertEqual(identity["package_path"], str((self.site / "deepopt").resolve()))
        self.assertFalse(identity["editable"])

    def test_unrelated_shadow_package_fails_instead_of_qualifying(self):
        shadow = self.root / "foreign/deepopt"
        shadow.mkdir(parents=True)
        (shadow / "__init__.py").write_text("marker = 'shadow'\n")
        result = self.run_script("runtime.installed_identity(root)", (shadow.parent,))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("shadowed", result.stderr)

    def test_editable_distribution_is_rejected(self):
        (self.metadata / "direct_url.json").write_text(json.dumps({"dir_info": {"editable": True}}))
        result = self.run_script("runtime.installed_identity(root)")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Editable", result.stderr)

    def test_missing_wheel_record_is_rejected(self):
        (self.metadata / "RECORD").unlink()
        result = self.run_script("runtime.installed_identity(root)")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("RECORD", result.stderr)

    def test_child_environment_pins_only_the_validated_install_and_helpers(self):
        result = self.run_script('''identity = runtime.installed_identity(root)
env = runtime.child_environment(root, identity, {"PYTHONPATH": "/foreign/src"})
assert "/foreign/src" not in env["PYTHONPATH"]
assert env["PYTHONPATH"].split(os.pathsep)[0] == identity["install_root"]
assert json.loads(env[runtime.IDENTITY_ENV]) == identity
''')
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_installed_file_change_invalidates_parent_identity(self):
        result = self.run_script('''identity = runtime.installed_identity(root)
os.environ[runtime.IDENTITY_ENV] = json.dumps(identity)
Path(identity["import_path"]).write_text("marker = 'changed'\\n")
runtime.installed_identity(root)
''')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("differs from the parent", result.stderr)

    def test_group_reuses_content_hashes_but_revalidates_at_both_boundaries(self):
        result = self.run_script('''from unittest.mock import patch
original = Path.read_bytes
reads = []
def read(path):
    reads.append(str(path))
    return original(path)
with patch.object(Path, "read_bytes", read):
    with runtime.identity_session(root) as group:
        first = runtime.installed_identity(root)
        for _ in range(5):
            assert runtime.installed_identity(root) == first
        assert len(reads) == first["files_hashed"]
        assert group.reuses == 6
assert len(reads) == first["files_hashed"] * 2
''')
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_group_rejects_restored_bytes_and_mtime(self):
        result = self.run_script('''with runtime.identity_session(root) as group:
    path = Path(group.identity["import_path"])
    before, content = path.stat(), path.read_bytes()
    path.write_bytes(content + b"# altered\\n")
    path.write_bytes(content)
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    runtime.installed_identity(root)
''')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("changed inside", result.stderr)

    def test_group_rejects_added_package_file_or_editable_metadata(self):
        for mutation in ("(Path(group.identity['package_path']) / 'new.py').write_text('x=1')",
                         f"Path({str(self.metadata / 'direct_url.json')!r}).write_text('{{\"dir_info\":{{\"editable\":true}}}}')"):
            with self.subTest(mutation=mutation):
                result = self.run_script("with runtime.identity_session(root) as group:\n    " + mutation
                                         + "\n    runtime.installed_identity(root)\n")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("changed inside", result.stderr)
                (self.site / "deepopt/new.py").unlink(missing_ok=True)
                (self.metadata / "direct_url.json").unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
