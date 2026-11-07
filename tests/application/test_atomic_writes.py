"""Text replacement preserves durable output when writing or rename fails."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from deepopt._io import atomic_write_text


class AtomicWriteTests(unittest.TestCase):
    def test_text_and_streamed_chunks_replace_utf8_output(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.jsonl"
            atomic_write_text(path, "first\n")
            atomic_write_text(path, iter(("epoch α\n", "epoch β\n")))
            self.assertEqual(path.read_text(encoding="utf-8"), "epoch α\nepoch β\n")
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_failed_chunk_generation_or_replace_keeps_existing_file(self):
        def failed_chunks():
            yield "partial replacement\n"
            raise RuntimeError("serialization failed")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.jsonl"
            path.write_text("original\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "serialization failed"):
                atomic_write_text(path, failed_chunks())
            self.assertEqual(path.read_text(), "original\n")
            self.assertEqual(list(Path(directory).iterdir()), [path])
            with patch("deepopt._io.os.replace", side_effect=OSError("replace failed")), \
                    self.assertRaisesRegex(OSError, "replace failed"):
                atomic_write_text(path, "replacement\n")
            self.assertEqual(path.read_text(), "original\n")
            self.assertEqual(list(Path(directory).iterdir()), [path])


if __name__ == "__main__":
    unittest.main()
