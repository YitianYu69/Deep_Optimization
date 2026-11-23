"""Small parent receipts with complete diagnostic evidence in separate files.

This module uses only the standard library. Full compiler entries are saved
without filtering; only the parent-facing views deliberately select metadata.
"""

import hashlib
from itertools import islice
import json
import math
import os
from pathlib import Path
import re
import tempfile


MAX_TEXT_CHARS = 2048
MAX_FAILED_NAMES = 128
_MAX_NAME_CHARS = 192
_MAX_ITEMS = 32
_ENCODER = json.JSONEncoder(sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _file_reference(path, relative_to):
    path = Path(path)
    digest, size = hashlib.sha256(), 0
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(data)
            size += len(data)
    return {"path": Path(os.path.relpath(path.resolve(), Path(relative_to).resolve())).as_posix(),
            "sha256": digest.hexdigest(), "bytes": size}


def _small(value, depth=0):
    """Keep selected scalars or short scalar lists, without object reprs."""
    if value is None or type(value) in (bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("Diagnostic summary contains nonfinite metadata")
        return value
    if isinstance(value, str):
        return value[:MAX_TEXT_CHARS]
    if depth < 1 and isinstance(value, (list, tuple)):
        return [_small(item, depth + 1) for item in value[:_MAX_ITEMS]]
    return None


def _selected(value, keys):
    if not isinstance(value, dict):
        return {}
    return {key: _small(value[key]) for key in keys if key in value}


_COMPARISON_KEYS = ("name", "passed", "failure", "shape", "dtype", "elements", "max_abs",
                    "relative_l2", "relative_l2_limit", "rtol", "atol", "exact", "finite",
                    "arithmetic_domain", "bitwise_equal", "status", "key", "module", "phase",
                    "role", "occurrence", "operation", "source_node", "elementwise_passed",
                    "relative_l2_passed", "failure_reasons")
_CONTEXT_KEYS = ("component", "stage", "phase", "route", "role", "update")


def _comparison(value):
    row = _selected(value, _COMPARISON_KEYS)
    if isinstance(value, dict):
        for key in ("expected", "actual"):
            if key in value:
                row[key] = _selected(value[key], ("present", "shape", "dtype"))
        if "context" in value:
            row["context"] = _selected(value["context"], _CONTEXT_KEYS)
    return row


def _compiler_summary(entry):
    if not isinstance(entry, dict):
        raise TypeError("Each diagnostic compiler report entry must be a dictionary")
    row = _selected(entry, ("slot", "compiler_report_scope", "compiler_report_status",
                            "compiler_report_collection_error"))
    report = entry.get("compiler_report")
    if isinstance(report, dict):
        count_keys = ("captured_graphs", "runtime_fallbacks", "triton_regions", "tilelang_regions",
                      "deepopt_operations", "lowered_deepopt_operations", "compiled_deepopt_backward_regions",
                      "native_autodiff_operations", "native_autodiff_regions", "fused_deepopt_regions")
        counts = {key: report[key] for key in count_keys if type(report.get(key)) is int}
        for source, target in (("graphs", "graphs"), ("fusion_candidates", "fusion_candidates"),
                               ("pack_sharing_decisions", "pack_sharing_decisions")):
            if isinstance(report.get(source), (list, tuple)):
                counts[target] = len(report[source])
        row["counts"] = counts
    elif isinstance(entry.get("counts"), dict):
        row["counts"] = {key: value for key, value in islice(entry["counts"].items(), _MAX_ITEMS)
                         if isinstance(key, str) and type(value) is int}
    if entry.get("compiler_report_externalized") is True:
        row["artifact"] = _selected(entry["artifact"], ("path", "sha256", "bytes"))
        row["compiler_report_externalized"] = True
    return row


def externalize_compiler_reports(result, directory):
    """Replace full route entries in ``result`` with verified artifact views.

    Every original entry, including unknown fields and IR, is persisted as one
    compact JSON file. Existing files are accepted only when byte-identical.
    Serialization or filesystem errors leave ``result`` unchanged and propagate
    so the caller can record collection failure without replacing its cause.
    """
    entries = result.get("compiler_reports")
    if entries is None:
        return result
    if not isinstance(entries, list):
        raise TypeError("diagnostic compiler_reports must be a list")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    staged, summaries, created = [], [], []
    try:
        for index, entry in enumerate(entries):
            summary = _compiler_summary(entry)
            if entry.get("compiler_report_externalized") is True:
                artifact = entry.get("artifact", {})
                name = artifact.get("path")
                if not isinstance(name, str) or Path(name).name != name or name in (".", ".."):
                    raise ValueError("External compiler report must reference a relative filename")
                if _file_reference(directory / name, directory) != artifact:
                    raise ValueError("External compiler report identity differs from its artifact")
                summaries.append(summary)
                continue
            slot = entry.get("slot")
            label = re.sub(r"[^A-Za-z0-9_-]", "_", slot)[:48] if isinstance(slot, str) else "route"
            destination = directory / f"compiler-report-{index:02d}-{label or 'route'}.json"
            with tempfile.NamedTemporaryFile(mode="wb", dir=directory, prefix=".compiler-report-",
                                             suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                staged.append((temporary, destination))
                digest, size = hashlib.sha256(), 0
                for piece in _ENCODER.iterencode(entry):
                    data = piece.encode("utf-8")
                    stream.write(data)
                    digest.update(data)
                    size += len(data)
            artifact = {"path": destination.name, "sha256": digest.hexdigest(), "bytes": size}
            if destination.exists() and _file_reference(destination, directory) != artifact:
                raise FileExistsError(f"Preserved compiler report differs: {destination.name}")
            summary.update(artifact=artifact, compiler_report_externalized=True)
            summaries.append(summary)
        for temporary, destination in staged:
            if not destination.exists():
                # An exclusive link publishes the complete file without
                # overwriting a prior capture or exposing a partial JSON file.
                os.link(temporary, destination)
                created.append(destination)
        result["compiler_reports"] = summaries
        return result
    except BaseException:
        for path in created:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        raise
    finally:
        for temporary, _ in staged:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def _oracle_reports(result):
    evidence = result.get("qualification_evidence")
    evidence = evidence if isinstance(evidence, dict) else {}
    trajectory = result.get("trajectory")
    trajectory = trajectory if isinstance(trajectory, dict) else evidence
    reports = [row["oracle"] for row in trajectory.get("updates", ())
               if isinstance(row, dict) and isinstance(row.get("oracle"), dict)]
    current = result.get("oracle_report") or evidence.get("oracle_report") or evidence.get("oracle")
    if isinstance(current, dict):
        reports.append(current)
    return reports


def _comparison_summary(result):
    summary = {"completed": 0, "failure_count": 0, "gradient_count": 0,
               "failed_gradient_count": 0, "unconsumed_count": 0}
    failed_names, first = set(), result.get("comparison") or None
    for report in _oracle_reports(result):
        rows = report.get("comparisons", ())
        rows = rows if isinstance(rows, (list, tuple)) else ()
        summary["completed"] += report.get("completed_comparison_count", len(rows))
        summary["failure_count"] += report.get("failure_count", sum(row.get("passed") is False
                                                   for row in rows if isinstance(row, dict)))
        summary["unconsumed_count"] += report.get("unconsumed_count", 0)
        if not isinstance(first, dict):
            first = report.get("first_failure")
        for row in rows:
            if not isinstance(row, dict):
                continue
            name, failed = row.get("name"), row.get("passed") is False
            gradient = isinstance(name, str) and name.startswith("gradient/")
            summary["gradient_count"] += gradient
            summary["failed_gradient_count"] += gradient and failed
            if failed and isinstance(name, str):
                failed_names.add(name)
                if not isinstance(first, dict):
                    first = row
    names = sorted(failed_names)
    summary.update(failed_names=[name[:_MAX_NAME_CHARS] for name in names[:MAX_FAILED_NAMES]],
                   failed_name_count=len(names), failed_names_truncated=max(0, len(names) - MAX_FAILED_NAMES))
    return summary, _comparison(first)


def _probe_summary(value):
    row = _selected(value, ("scope", "passed", "completed", "status", "deterministic_algorithms",
                            "repeat_stable", "deterministic_blocked", "cuda_execution_observed"))
    if isinstance(value, dict) and isinstance(value.get("error"), str):
        row["error"] = value["error"].partition("\n")[0][:MAX_TEXT_CHARS]
    elif isinstance(value, dict) and isinstance(value.get("error"), dict):
        row["error"] = _selected(value["error"], ("type", "message"))
    return row


def summary_for_parent(result, result_path, output_parent):
    """Return a bounded run row; the canonical case file remains authoritative.

    Unknown fields stay in the case JSON (or full compiler artifact). They are
    deliberately absent here so a new detailed evidence field cannot enlarge
    the parent receipt unexpectedly. Artifact paths are relative to its folder.
    """
    row = _selected(result, ("status", "settings_sha256", "returncode", "source_stable",
                             "boundary_observation_enabled", "candidate_capacity_status",
                             "diagnostic_reporting_error"))
    case = result.get("case")
    row["case"] = _selected(case, ("name", "kind", "observe_boundaries"))
    for slot in ("reference", "candidate"):
        if isinstance(case, dict) and slot in case:
            row["case"][slot] = _selected(case[slot], ("execution", "profile"))
    row["installed"] = _selected(result.get("installed"), ("distribution_name", "version", "package_path",
        "import_path", "install_root", "source_sha256", "files_hashed", "editable"))
    row.update(qualification_eligible=False, timing_eligible=False, cuda_qualified=False)
    if isinstance(result.get("error"), str):
        lines = result["error"].splitlines()
        message = lines[0] if lines else ""
        if message.startswith("BackendCompilerFailed:"):
            cause = next((line.strip() for line in lines[1:] if line.strip()), "")
            message += " " + cause if cause else ""
        row["error"] = message[:MAX_TEXT_CHARS]
    if "failure_context" in result:
        row["failure_context"] = _selected(result["failure_context"], _CONTEXT_KEYS)
    if isinstance(result.get("determinism_blocker"), dict):
        row["determinism_blocker"] = _selected(result["determinism_blocker"], ("error_type", "message"))
    row["comparison_summary"], first = _comparison_summary(result)
    if first:
        row["comparison"] = first
    localization = result.get("localization")
    if isinstance(localization, dict):
        row["localization"] = _selected(localization, ("scope", "enabled", "complete",
            "coverage_complete", "context_succeeded", "global_first_divergence_verified"))
        for key in ("first_tolerance_failure", "first_bitwise_difference"):
            if key in localization:
                row["localization"][key] = _comparison(localization[key]) if localization[key] else None
    row["compiler_reports"] = []
    entries = result.get("compiler_reports", ())
    for entry in entries[:_MAX_ITEMS]:
        report = _compiler_summary(entry)
        if "artifact" in report:
            name = report["artifact"].get("path")
            if not isinstance(name, str) or Path(name).name != name:
                raise ValueError("Compiler artifact reference must be a relative filename")
            report["artifact"]["path"] = Path(os.path.relpath(Path(result_path).parent / name,
                                                             output_parent)).as_posix()
        row["compiler_reports"].append(report)
    row["compiler_report_count"] = len(entries)
    row["compiler_reports_truncated"] = max(0, len(entries) - _MAX_ITEMS)
    if isinstance(result.get("operators"), dict):
        row["operators"] = {name[:_MAX_NAME_CHARS]: _probe_summary(value)
                            for name, value in islice(result["operators"].items(), _MAX_ITEMS)
                            if isinstance(name, str)}
    if isinstance(result.get("repeatability"), dict):
        row["repeatability"] = _probe_summary(result["repeatability"])
        controls = result["repeatability"].get("controls")
        if isinstance(controls, list):
            row["repeatability"]["controls"] = [
                _selected(control, ("name", "completed", "repeat_stable", "deterministic_blocked",
                                    "fixed_inputs_unchanged")) for control in controls[:_MAX_ITEMS]]
    row["result_file"] = _file_reference(result_path, output_parent)
    return row
