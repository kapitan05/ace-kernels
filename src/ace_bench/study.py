"""Browser study guide: ``docs/study/src/*.md`` -> ``docs/study/*.html``.

Pages are Markdown plus line directives that pull live material from the repo, so
code excerpts, tables and plots cannot drift from the code and the saved runs (the
build fails on a missing symbol, file or run; ``tests/test_study.py``):

    ::code <path>::<Name>[.<method>] [| note]   a function / class via ``ast``
    ::code <path>:<first>-<last> [| note]       a line range
    ::plot <path or glob> | <caption> | <what it proves>
    ::table <parquet path or glob> | <polars expression on df> | <caption>
                                                (``ladder``: sort key in sprint order)
    ::run <shell command> [| <what to expect>]
    ::check <question> | <answer>
    ::svg <name>                                docs/study/src/diagrams/<name>.svg
    ::walk <path>::<Name> (or <path>:<a>-<b>)   code in chunks, each explained:
    @ <text on the chunk's first line> | <explanation (Markdown)>
    ::end
    ::exec                                      Python run at build time (CPU);
    <python lines>                              the code and its printed output
    ::end                                       are shown (worked examples)

Run: ``uv run python -m ace_bench.study``; open ``docs/study/index.html``.
"""

import argparse
import ast
import contextlib
import html
import io
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import markdown
import polars as pl
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer, get_lexer_for_filename
from pygments.util import ClassNotFound

from ace_bench.env import REPO_ROOT

SRC = REPO_ROOT / "docs" / "study" / "src"
OUT = REPO_ROOT / "docs" / "study"
GITHUB = "https://github.com/kapitan05/ace-kernels/blob"


@dataclass(frozen=True)
class Page:
    slug: str
    title: str
    sprints: str


PAGES = [
    Page("index", "Overview: map of the project", "all"),
    Page("01-loss", "1. The loss and why the original is slow", "S1-S2"),
    Page("02-correctness", "2. Proving 'same as the original'", "S1, S9"),
    Page("03-pytorch", "3. PyTorch: vectorize, handwritten backward", "S3-S4"),
    Page("04-triton", "4. Triton kernels", "S5-S8"),
    Page("05-sprint11", "5. Sprint 11: launches, index, packaging", "S11"),
    Page("06-training", "6. Training: end-to-end, torch.compile", "S8, S11"),
    Page("07-tooling", "7. Tooling: bench, Kaggle, W&B, profiler", "all"),
    Page("evidence", "Evidence: claim -> proof", "all"),
    Page("glossary", "Glossary", "all"),
]


class BuildError(Exception):
    pass


def _commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "main"


def _resolve(pattern: str) -> Path:
    """A repo path or glob (latest match: run folders are timestamped)."""
    matches = sorted(REPO_ROOT.glob(pattern))
    if not matches:
        raise BuildError(f"no file matches {pattern!r}")
    return matches[-1]


def _symbol_lines(path: Path, name: str) -> tuple[int, int]:
    """First (incl. decorators) and last line of ``Name`` or ``Class.method``."""
    tree = ast.parse(path.read_text())
    body: list[ast.stmt] = tree.body
    node: ast.stmt | None = None
    for part in name.split("."):
        node = next(
            (
                n
                for n in body
                if isinstance(n, ast.FunctionDef | ast.ClassDef | ast.AsyncFunctionDef)
                and n.name == part
            ),
            None,
        )
        if node is None:
            raise BuildError(f"{path.relative_to(REPO_ROOT)}: no symbol {name!r}")
        body = node.body if isinstance(node, ast.ClassDef) else []
    assert isinstance(node, ast.FunctionDef | ast.ClassDef | ast.AsyncFunctionDef)
    first = min([node.lineno, *(d.lineno for d in node.decorator_list)])
    assert node.end_lineno is not None
    return first, node.end_lineno


class Builder:
    def __init__(self, out: Path) -> None:
        self.out = out
        self.commit = _commit()
        self.formatter = HtmlFormatter(cssclass="highlight", nowrap=False)
        self.plots: dict[Path, str] = {}

    # --- directives -------------------------------------------------------

    def _locate(self, spec: str) -> tuple[str, int, int, list[str]]:
        """``path::Symbol`` or ``path:first-last`` -> (rel, first, last, lines)."""
        if "::" in spec:
            rel, name = spec.split("::", 1)
            path = REPO_ROOT / rel
            if not path.exists():
                raise BuildError(f"missing file {rel}")
            first, last = _symbol_lines(path, name)
        else:
            m = re.fullmatch(r"(.+):(\d+)-(\d+)", spec)
            if not m:
                raise BuildError(f"bad ::code spec {spec!r}")
            rel, first, last = m.group(1), int(m.group(2)), int(m.group(3))
            path = REPO_ROOT / rel
            if not path.exists():
                raise BuildError(f"missing file {rel}")
        lines = path.read_text().splitlines()
        if last > len(lines):
            raise BuildError(f"{rel}: lines {first}-{last} beyond end ({len(lines)})")
        return rel, first, last, lines

    def _highlight(self, rel: str, lines: list[str], first: int, last: int) -> str:
        formatter = HtmlFormatter(
            cssclass="highlight", linenos="inline", linenostart=first
        )
        try:
            lexer = get_lexer_for_filename(rel)
        except ClassNotFound:
            lexer = PythonLexer()
        return highlight("\n".join(lines[first - 1 : last]), lexer, formatter)

    def _head(self, rel: str, first: int, last: int, label: str = "") -> str:
        url = f"{GITHUB}/{self.commit}/{rel}#L{first}-L{last}"
        return (
            f'<div class="code-head"><span class="path">{label}{html.escape(rel)}'
            f":{first}-{last}</span>"
            f'<a href="{url}" target="_blank" rel="noopener">GitHub</a></div>'
        )

    def code(self, arg: str) -> str:
        spec, _, note = (s.strip() for s in arg.partition("|"))
        rel, first, last, lines = self._locate(spec)
        body = self._highlight(rel, lines, first, last)
        head = self._head(rel, first, last)
        note_html = f'<p class="code-note">{_inline(note)}</p>' if note else ""
        return f'<figure class="code">{head}{body}{note_html}</figure>'

    def walk(self, spec: str, steps: list[str]) -> str:
        """Code split at anchor lines, each chunk with its explanation.

        ``steps``: lines ``@ <text on the chunk's first line> | <explanation>``;
        other lines continue the previous explanation. Anchors are searched in
        order inside the excerpt, so a renamed or moved line fails the build.
        """
        rel, first, last, lines = self._locate(spec.strip())
        parsed: list[tuple[str, str]] = []
        for raw in steps:
            if raw.startswith("@ "):
                anchor, sep, note = raw[2:].partition(" | ")
                if not sep:
                    raise BuildError(f"::walk step needs ' | ': {raw!r}")
                parsed.append((anchor.strip(), note.strip()))
            elif parsed:
                parsed[-1] = (parsed[-1][0], parsed[-1][1] + "\n" + raw)
        if not parsed:
            raise BuildError(f"::walk {spec}: no steps")
        starts: list[int] = []
        pos = first
        for anchor, _ in parsed:
            hit = next(
                (n for n in range(pos, last + 1) if anchor in lines[n - 1]), None
            )
            if hit is None:
                raise BuildError(
                    f"::walk {rel}: anchor {anchor!r} not found after L{pos}"
                )
            starts.append(hit)
            pos = hit + 1
        starts[0] = first  # decorators / signature before the first anchor
        ends = [s - 1 for s in starts[1:]] + [last]
        chunks = "".join(
            f'<div class="step"><div class="step-code">'
            f"{self._highlight(rel, lines, a, b)}</div>"
            f'<div class="step-note">{markdown.markdown(note)}</div></div>'
            for (_, note), a, b in zip(parsed, starts, ends, strict=True)
        )
        head = self._head(rel, first, last, "walk-through: ")
        return f'<figure class="walk">{head}{chunks}</figure>'

    def plot(self, arg: str) -> str:
        parts = [s.strip() for s in arg.split("|")]
        src = _resolve(parts[0])
        name = re.sub(r"[^A-Za-z0-9._-]+", "_", str(src.relative_to(REPO_ROOT)))
        self.plots[src] = name
        caption = _inline(parts[1]) if len(parts) > 1 else ""
        proves = (
            f'<p class="proves"><b>Proves:</b> {_inline(parts[2])}</p>'
            if len(parts) > 2
            else ""
        )
        rel = src.relative_to(REPO_ROOT)
        return (
            f'<figure class="plot"><img src="assets/plots/{name}" alt="{caption}" '
            f'loading="lazy"><figcaption>{caption} <span class="src">'
            f"{html.escape(str(rel))}</span></figcaption>{proves}</figure>"
        )

    def table(self, arg: str) -> str:
        parts = [s.strip() for s in arg.split("|")]
        if len(parts) < 2:
            raise BuildError(f"::table needs '<path> | <expression>': {arg!r}")
        src = _resolve(parts[0])
        df = pl.read_parquet(src)
        try:
            # trusted page sources; ``ladder`` sorts rows in sprint (ablation) order
            names = {"pl": pl, "df": df, "ladder": _ladder_rank()}
            result = eval(parts[1], names)
        except Exception as exc:
            raise BuildError(f"::table {parts[1]!r}: {exc}") from exc
        if not isinstance(result, pl.DataFrame):
            raise BuildError(f"::table expression must give a DataFrame: {parts[1]}")
        caption = _inline(parts[2]) if len(parts) > 2 else ""
        head = "".join(f"<th>{html.escape(c)}</th>" for c in result.columns)
        rows = "".join(
            "<tr>" + "".join(f"<td>{_cell(v)}</td>" for v in row) + "</tr>"
            for row in result.iter_rows()
        )
        rel = html.escape(str(src.relative_to(REPO_ROOT)))
        return (
            f'<figure class="table"><div class="scroll"><table><thead><tr>{head}'
            f"</tr></thead><tbody>{rows}</tbody></table></div><figcaption>{caption} "
            f'<span class="src">{rel}</span></figcaption></figure>'
        )

    def run(self, arg: str) -> str:
        cmd, _, expect = (s.strip() for s in arg.partition("|"))
        exp = f'<p class="expect">{_inline(expect)}</p>' if expect else ""
        return (
            f'<div class="run"><button class="copy" type="button">copy</button>'
            f"<pre><code>$ {html.escape(cmd)}</code></pre>{exp}</div>"
        )

    def check(self, arg: str) -> str:
        question, _, answer = (s.strip() for s in arg.partition("|"))
        if not answer:
            raise BuildError(f"::check needs an answer: {question!r}")
        return (
            f'<details class="check"><summary>Self-check: {_inline(question)}'
            f'</summary><div class="answer">{_inline(answer)}</div></details>'
        )

    def svg(self, arg: str) -> str:
        path = SRC / "diagrams" / f"{arg.strip()}.svg"
        if not path.exists():
            raise BuildError(f"missing diagram {path.name}")
        return f'<figure class="diagram">{path.read_text()}</figure>'

    def exec_block(self, code: str) -> str:
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                exec(compile(code, "<study>", "exec"), {"__name__": "__study__"})
        except Exception as exc:
            raise BuildError(f"::exec failed: {type(exc).__name__}: {exc}") from exc
        src = highlight(code, PythonLexer(), self.formatter)
        out = html.escape(buf.getvalue().rstrip())
        return (
            f'<figure class="exec"><div class="code-head"><span class="path">'
            f"worked example (run at build time, CPU)</span></div>{src}"
            f'<pre class="output">{out}</pre></figure>'
        )

    # --- pages ------------------------------------------------------------

    def render(self, text: str) -> str:
        """Markdown with directives replaced by placeholders, then filled in."""
        blocks: list[str] = []
        lines = text.splitlines()
        md_lines: list[str] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.strip() == "::exec":
                end = next(
                    (
                        j
                        for j in range(i + 1, len(lines))
                        if lines[j].strip() == "::end"
                    ),
                    None,
                )
                if end is None:
                    raise BuildError("::exec without ::end")
                blocks.append(self.exec_block("\n".join(lines[i + 1 : end])))
                md_lines += ["", f"STUDYBLOCK{len(blocks) - 1}X", ""]
                i = end + 1
                continue
            if line.startswith("::walk "):
                end = next(
                    (
                        j
                        for j in range(i + 1, len(lines))
                        if lines[j].strip() == "::end"
                    ),
                    None,
                )
                if end is None:
                    raise BuildError("::walk without ::end")
                blocks.append(self.walk(line[len("::walk ") :], lines[i + 1 : end]))
                md_lines += ["", f"STUDYBLOCK{len(blocks) - 1}X", ""]
                i = end + 1
                continue
            m = re.match(r"::(code|plot|table|run|check|svg)\s+(.*)$", line.strip())
            if m:
                blocks.append(getattr(self, m.group(1))(m.group(2)))
                md_lines += ["", f"STUDYBLOCK{len(blocks) - 1}X", ""]
            else:
                md_lines.append(line)
            i += 1
        body = markdown.markdown(
            "\n".join(md_lines), extensions=["tables", "fenced_code", "toc"]
        )
        for k, block in enumerate(blocks):
            body = body.replace(f"<p>STUDYBLOCK{k}X</p>", block)
            body = body.replace(f"STUDYBLOCK{k}X", block)
        return body

    def page(self, page: Page, body: str) -> str:
        idx = PAGES.index(page)
        nav = "".join(
            f'<li class="{"current" if p is page else ""}"><a href="{p.slug}.html">'
            f"{html.escape(p.title)}</a></li>"
            for p in PAGES
        )
        heads = re.findall(r'<h2 id="([^"]+)">(.*?)</h2>', body)
        toc = "".join(f'<li><a href="#{a}">{t}</a></li>' for a, t in heads)
        prev_link = (
            f'<a href="{PAGES[idx - 1].slug}.html">&larr; {PAGES[idx - 1].title}</a>'
            if idx > 0
            else "<span></span>"
        )
        next_link = (
            f'<a href="{PAGES[idx + 1].slug}.html">{PAGES[idx + 1].title} &rarr;</a>'
            if idx + 1 < len(PAGES)
            else "<span></span>"
        )
        return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(page.title)} | ACE kernels study</title>
<link rel="stylesheet" href="assets/style.css">
<link rel="stylesheet" href="assets/pygments.css">
</head><body>
<aside class="side"><a class="brand" href="index.html">mL1-ACE kernels<br>
<small>study guide</small></a><ol class="pages">{nav}</ol>
<div class="toc"><b>On this page</b><ul>{toc}</ul></div>
<p class="meta">commit {self.commit[:10]}</p></aside>
<main><p class="sprints">Sprints: {page.sprints}</p>{body}
<nav class="pager">{prev_link}{next_link}</nav></main>
<script src="assets/study.js"></script></body></html>
"""

    def build(self) -> list[Path]:
        self.out.mkdir(parents=True, exist_ok=True)
        assets = self.out / "assets"
        (assets / "plots").mkdir(parents=True, exist_ok=True)
        for name in ("style.css", "study.js"):
            shutil.copy(SRC / name, assets / name)
        light = HtmlFormatter(style="friendly").get_style_defs(".highlight")
        dark = HtmlFormatter(style="monokai").get_style_defs(".highlight")
        (assets / "pygments.css").write_text(
            f"{light}\n@media (prefers-color-scheme: dark) {{\n{dark}\n}}\n"
        )
        written = []
        for page in PAGES:
            src = SRC / f"{page.slug}.md"
            if not src.exists():
                raise BuildError(f"missing page source {src.name}")
            path = self.out / f"{page.slug}.html"
            path.write_text(self.page(page, self.render(src.read_text())))
            written.append(path)
        for src_plot, name in self.plots.items():
            shutil.copy(src_plot, assets / "plots" / name)
        figures = SRC.parent / "assets" / "figures"
        if figures.exists() and self.out != OUT:
            shutil.copytree(figures, assets / "figures", dirs_exist_ok=True)
        return written


def _ladder_rank() -> pl.Expr:
    from ace_bench.ablation import LADDER

    order = {impl: k for k, (impl, _, _) in enumerate(LADDER)}
    return pl.col("impl").replace_strict(order, default=99, return_dtype=pl.Int64)


def _inline(text: str) -> str:
    """Inline Markdown (code spans, bold, links) without a wrapping <p>."""
    out = markdown.markdown(text)
    return re.sub(r"^<p>(.*)</p>$", r"\1", out, flags=re.S)


def _cell(value: object) -> str:
    if isinstance(value, float):
        if value != value:
            return "NaN"
        if value != 0 and (abs(value) < 1e-3 or abs(value) >= 1e5):
            return f"{value:.2e}"
        return f"{value:.3g}" if abs(value) < 100 else f"{value:.1f}"
    return html.escape("" if value is None else str(value))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the HTML study guide")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    written = Builder(args.out).build()
    print(f"wrote {len(written)} pages -> {args.out / 'index.html'}")


if __name__ == "__main__":
    main()
