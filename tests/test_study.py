"""The study guide builds, and everything it cites exists (no stale references)."""

import re
from html.parser import HTMLParser
from pathlib import Path

import pytest

from ace_bench.env import REPO_ROOT
from ace_bench.study import PAGES, SRC, Builder

TESTS = REPO_ROOT / "tests"


class _Links(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.refs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for key, value in attrs:
            if key in ("href", "src") and value:
                self.refs.append(value)


@pytest.fixture(scope="module")
def site(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if not (REPO_ROOT / "results" / "kaggle").exists():
        pytest.skip("saved runs (git-ignored results/) not present, e.g. on Kaggle")
    out = tmp_path_factory.mktemp("study")
    Builder(out).build()
    return out


def test_every_page_is_built(site: Path) -> None:
    for page in PAGES:
        assert (site / f"{page.slug}.html").stat().st_size > 1000


def test_internal_links_resolve(site: Path) -> None:
    for page in site.glob("*.html"):
        parser = _Links()
        parser.feed(page.read_text())
        for ref in parser.refs:
            if ref.startswith(("http://", "https://", "#", "mailto:")):
                continue
            target = (site / ref.split("#")[0]).resolve()
            assert target.exists(), f"{page.name}: broken link {ref}"


def test_cited_tests_exist() -> None:
    """Every ``test_*`` name the pages cite is defined in tests/."""
    defined = {
        m.group(1)
        for path in TESTS.glob("test_*.py")
        for m in re.finditer(r"def (test_\w+)", path.read_text())
    }
    files = {p.stem for p in TESTS.glob("test_*.py")}
    for src in SRC.glob("*.md"):
        for name in set(re.findall(r"\b(test_\w+)", src.read_text())):
            assert name in defined or name in files, f"{src.name}: unknown {name}"
