"""Local atomic text output; callers own directories and distributed routing."""

from collections.abc import Iterable
import os
from pathlib import Path
import tempfile


def atomic_write_text(path: str | Path, content: str | Iterable[str]) -> None:
    """Replace only after successful writing; retain the old file on failure."""
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         delete=False, suffix=".tmp") as file:
            temporary = Path(file.name)
            if isinstance(content, str):
                file.write(content)
            else:
                file.writelines(content)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
