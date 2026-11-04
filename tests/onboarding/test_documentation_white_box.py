"""Keep repository navigation and the runnable example catalog complete."""

from collections import Counter
import ast
from pathlib import Path
import re
import textwrap
import tomllib
import unittest
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[2]


def prose(text):
    return re.sub(r"(^[ \t]*(`{3,}|~{3,}).*?$).*?^[ \t]*\2[ \t]*$", "", text,
                  flags=re.M | re.S)


def links(path):
    for match in re.finditer(r"\[[^\]\n]*\]\(([^\n)]+)\)", prose(path.read_text())):
        target = match.group(1).strip()
        if target.startswith("<"):
            target = target[1:target.index(">")]
        else:
            target = target.split(' "', 1)[0]
        parsed = urlsplit(target)
        if not parsed.scheme and not parsed.netloc:
            yield target, parsed


def anchors(path):
    result, counts = set(), Counter()
    text = prose(path.read_text())
    for heading in re.findall(r"^ {0,3}#{1,6}\s+(.+?)\s*#*\s*$", text, re.M):
        heading = re.sub(r"\[([^]]+)\]\([^)]+\)", r"\1", heading)
        slug = re.sub(r"[^\w\- ]", "", heading.lower()).replace(" ", "-")
        suffix = f"-{counts[slug]}" if counts[slug] else ""
        result.add(slug + suffix)
        counts[slug] += 1
    result.update(re.findall(r'<a\s+(?:id|name)="([^"]+)"', text))
    return result


class DocumentationWhiteBoxTests(unittest.TestCase):
    def test_python_and_toml_documentation_blocks_are_syntactically_valid(self):
        pages = [ROOT / "README.md", ROOT / "examples/README.md", ROOT / "tests/README.md"]
        pages.extend((ROOT / "docs").rglob("*.md"))
        for page in pages:
            blocks = re.findall(r"^```(python|toml)\n(.*?)^```", page.read_text(), re.M | re.S)
            for index, (language, block) in enumerate(blocks, 1):
                with self.subTest(page=str(page.relative_to(ROOT)), block=index):
                    if language == "python":
                        ast.parse(textwrap.dedent(block), filename=f"{page}:{index}")
                    else:
                        tomllib.loads(block)

    def test_local_document_links_and_section_anchors_resolve(self):
        pages = [ROOT / "README.md", ROOT / "examples/README.md", ROOT / "tests/README.md"]
        pages.extend((ROOT / "docs").rglob("*.md"))
        problems = []
        for page in pages:
            for original, target in links(page):
                destination = (page.parent / unquote(target.path)).resolve() if target.path else page
                if not destination.exists():
                    problems.append(f"{page.relative_to(ROOT)}: missing {original}")
                elif target.fragment and destination.suffix == ".md":
                    if unquote(target.fragment) not in anchors(destination):
                        problems.append(f"{page.relative_to(ROOT)}: missing section {original}")
        self.assertEqual(problems, [], "\n".join(problems))

    def test_every_runnable_example_is_linked_from_the_catalog(self):
        catalog = ROOT / "examples/README.md"
        indexed = {(catalog.parent / unquote(target.path)).resolve()
                   for _, target in links(catalog) if target.path}
        missing = [str(path.relative_to(ROOT)) for path in sorted((ROOT / "examples").rglob("*.py"))
                   if not path.name.startswith("_") and path.resolve() not in indexed]
        self.assertEqual(missing, [], "Examples missing from catalog: " + ", ".join(missing))


if __name__ == "__main__":
    unittest.main()
