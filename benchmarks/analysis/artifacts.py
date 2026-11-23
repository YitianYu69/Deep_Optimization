"""Atomic numerical artifacts and explicit, resumable profiler requirements."""

from fnmatch import fnmatchcase
from hashlib import sha256
from pathlib import Path


def file_identity(path):
    path = Path(path)
    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(2**20):
            digest.update(chunk)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def save_snapshot(torch, values, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(values, temporary)
        identity = file_identity(temporary)
        temporary.replace(path)
        return identity
    finally:
        temporary.unlink(missing_ok=True)


def snapshot_valid(row):
    # Memory receipts prove only a live comparison in their original suite.
    # They never authorize result/core resume in a later invocation.
    if row.get("comparison_input") is not None:
        return False
    if row.get("numerical_snapshot", {}).get("status") == "not_requested":
        return (row.get("measurement_policy", {}).get("numerical_comparison") is False
                and row.get("snapshot") is None and row.get("snapshot_file") is None)
    try:
        path = Path(row["snapshot"])
        expected = row["snapshot_file"]
        return (path.is_file() and path.stat().st_size == expected["bytes"] > 0
                and file_identity(path) == expected)
    except (KeyError, OSError, TypeError):
        return False


def profile_detail(task, options):
    workloads, cases = options.get("trace_workloads"), options.get("trace_cases")
    if ((workloads and not any(fnmatchcase(task["workload"]["name"], p) for p in workloads))
            or cases and task["case"] not in cases):
        return "summary"
    return options.get("profile_detail", "trace")


def trace_directory(directory, task, options):
    directory=Path(directory)
    if options.get("trace_directory"):
        return (Path(options["trace_directory"]).expanduser().resolve()
                / (options.get("resume_environment_sha256") or "local") / directory.parent.parent.name / task["id"])
    return directory


def profiles_valid(row, task, options):
    if options.get("profile_policy")=="selected":
        from .scheduler import profile_options
        options=profile_options(task,options,"selected")
    for field, steps in [("profile", options.get("profile_steps", 0)),
                         ("replay_profile", options.get("profile_replay_steps", 0)
                          if task["settings"]["runtime"]["execution"] == "graph" else 0)]:
        if not steps:
            continue
        data = row.get(field) or {}
        if data.get("steps") != steps or data.get("detail", "trace") != profile_detail(task, options):
            return False
        if profile_detail(task, options) != "summary":
            path = data.get("trace")
            if not path or not Path(path).is_file() or data.get("trace_export_error"):
                return False
    return True
