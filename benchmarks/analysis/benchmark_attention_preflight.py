#!/usr/bin/env python3
"""Qualify installed FA2, changing-input forward/backward graphs and joint packing."""

import argparse
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import runpy
import unittest


RUNTIME_SPEED_PREFLIGHT_REVISION=4


def load_preflight_suite(root,*,runtime_speed=False,convolution_only=False):
    """Keep the focused convolution check independent of FA2/timm tests."""
    def load(name,exported,canonical):
        path=root/exported
        if not path.is_file():path=root.parents[1]/canonical
        spec=importlib.util.spec_from_file_location(name,path)
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module,path
    suite=unittest.TestSuite()
    test_file=extra_test=extra_module=None
    if not convolution_only:
        module,test_file=load('_attention_cuda_preflight','test_flash_attention_backend_cuda.py','tests/cuda/test_flash_attention_backend.py')
        suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))
    if runtime_speed or convolution_only:
        extra_module,extra_test=load('_runtime_speed_cuda_preflight','test_runtime_speed_cuda.py','tests/cuda/test_runtime_speed.py')
        if getattr(extra_module,'RUNTIME_SPEED_PREFLIGHT_REVISION',None)!=RUNTIME_SPEED_PREFLIGHT_REVISION:
            raise RuntimeError('Mixed runtime-speed preflight files: extract the complete updated benchmark ZIP')
        if convolution_only:
            suite.addTest(extra_module.RuntimeSpeedCUDATests('test_dense_and_depthwise_tuning_and_changing_value_replay'))
            test_file=extra_test
        else:suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(extra_module))
        candidates = [('convolution_tuned', 'test_convolution_tuned.py')]
        if runtime_speed:
            candidates += [('speed_activation_candidates', 'test_speed_activation_candidates.py'),
                           ('fp8_preparation_tuning', 'test_fp8_preparation_tuning.py')]
        extra_module.SPEED_CANDIDATE_TEST_FILES = {}
        for name, filename in candidates:
            candidate, path = load('_' + name + '_preflight', filename, 'tests/cuda/' + filename)
            suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(candidate))
            extra_module.SPEED_CANDIDATE_TEST_FILES[filename] = sha256(path.read_bytes()).hexdigest()
    return suite,test_file,extra_test,extra_module


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=Path("attention-preflight.json"))
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument("--runtime-speed", action="store_true", help="Also compile and validate the runtime speed kernels")
    mode.add_argument("--convolution-only", action="store_true", help="Run only dense/depthwise implementation and native-replay checks")
    args = parser.parse_args(argv)
    if __name__ == "__main__":
        lease=Path(__file__).with_name("_gpu_lease.py")
        if not lease.is_file():lease=Path(__file__).resolve().parents[1]/'_gpu_lease.py'
        code = runpy.run_path(str(lease))["ensure"](__file__, argv, args.device)
        if code is not None:
            return code
    import torch
    import deepopt
    if not torch.cuda.is_available() or torch.version.hip is not None:
        raise RuntimeError("Preflight requires an NVIDIA CUDA GPU")
    flash_version=None
    if not args.convolution_only:
        import flash_attn
        flash_version=str(flash_attn.__version__)
        if not flash_version.startswith("2."):
            raise RuntimeError("FA2 preflight requires flash-attn version 2")
        if importlib.util.find_spec("timm") is None:
            raise RuntimeError("Install timm in the active environment for the Trainer/graph integration preflight")
    torch.cuda.set_device(args.device)
    root = Path(__file__).resolve().parent
    suite,test_file,extra_test,extra_module=load_preflight_suite(root,runtime_speed=args.runtime_speed,convolution_only=args.convolution_only)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if extra_module is not None:
        extra_module.CONVOLUTION_ARTIFACT_PATH=args.output.with_name(args.output.stem+'-convolution-operands.pt')
    print(f'Preflight scope={"convolution_only" if args.convolution_only else "runtime_speed" if args.runtime_speed else "attention"}; '
          f'runtime_speed_revision={RUNTIME_SPEED_PREFLIGHT_REVISION if extra_module is not None else "not_requested"}',flush=True)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    library_root = Path(deepopt.__file__).resolve().parent
    library_hash = sha256()
    for source in sorted(library_root.rglob('*.py')):
        library_hash.update(source.relative_to(library_root).as_posix().encode())
        library_hash.update(source.read_bytes())
    minimum = 3 if args.convolution_only else 25 if args.runtime_speed else 10
    report = {"status": "ok" if result.wasSuccessful() and not result.skipped and result.testsRun >= minimum else "error",
              "tests_run": result.testsRun, "skipped": result.skipped,
              "failures": result.failures, "errors": result.errors,
              "torch": str(torch.__version__), "cuda": torch.version.cuda,
              "flash_attn": flash_version, "gpu": torch.cuda.get_device_name(args.device),
              "deepopt": str(Path(deepopt.__file__).resolve())}
    report.update(deepopt_source_sha256=library_hash.hexdigest(),
                  tests_sha256=sha256(test_file.read_bytes()).hexdigest(),
                  cuda_capability=list(torch.cuda.get_device_capability(args.device)))
    report["runtime_speed_tests_sha256"] = sha256(extra_test.read_bytes()).hexdigest() if extra_test is not None else None
    report["scope"]='convolution_only' if args.convolution_only else 'runtime_speed' if args.runtime_speed else 'attention'
    report["runtime_speed_convolution_oracle"] = "exact_ternary_cpu_fp32_and_native_replay" if extra_module is not None else None
    report["runtime_speed_revision"]=RUNTIME_SPEED_PREFLIGHT_REVISION if extra_module is not None else None
    report["candidate_tests_sha256"] = getattr(extra_module, 'SPEED_CANDIDATE_TEST_FILES', {})
    report["convolution_diagnostics"]=extra_module.CONVOLUTION_DIAGNOSTICS if extra_module is not None else []
    report["convolution_operands_artifact"]=(str(extra_module.CONVOLUTION_ARTIFACT_PATH)
        if extra_module is not None and extra_module.CONVOLUTION_ARTIFACT_WRITTEN
           and extra_module.CONVOLUTION_ARTIFACT_PATH.is_file() else None)
    # unittest failure tuples contain TestCase instances; persist portable names.
    for key in ("skipped", "failures", "errors"):
        report[key] = [(str(test), reason) for test, reason in report[key]]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n"); temporary.replace(args.output)
    return int(report["status"] != "ok")


if __name__ == "__main__":
    raise SystemExit(main())
