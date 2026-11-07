"""Copy authoritative packaged TOML files into the checkout's compatibility paths."""

import argparse
from pathlib import Path
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="report drift without writing files")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    source = root / "src" / "deepopt" / "configs"
    destination = root / "configs"
    sources = sorted(source.rglob("*.toml"))
    if not sources:
        parser.error(f"no packaged TOML configurations found in {source}")

    expected = {path.relative_to(source) for path in sources}
    unexpected = sorted(path.relative_to(destination) for path in destination.rglob("*.toml")
                        if path.relative_to(destination) not in expected)
    drift = []
    updated = 0
    for path in sources:
        relative = path.relative_to(source)
        target = destination / relative
        content = path.read_bytes()
        if target.is_file() and target.read_bytes() == content:
            continue
        if args.check:
            drift.append(relative)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            updated += 1

    for relative in drift:
        print(f"out of sync: configs/{relative.as_posix()}", file=sys.stderr)
    for relative in unexpected:
        print(f"no packaged source: configs/{relative.as_posix()} (left untouched)", file=sys.stderr)
    if drift:
        print("Run python scripts/sync_configs.py to update checkout copies.", file=sys.stderr)
    if drift or unexpected:
        return 1
    if args.check:
        print(f"All {len(sources)} checkout configurations match packaged sources.")
    else:
        print(f"Updated {updated} checkout configurations from packaged sources.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
