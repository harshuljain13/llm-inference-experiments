#!/usr/bin/env python3
"""One-off: strip comments from repo files (README excluded). Delete this script after run."""
from __future__ import annotations

import ast
import io
import re
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def strip_python_hash(src: str) -> str:
    out: list[tokenize.TokenInfo] = []
    readline = io.StringIO(src).readline
    try:
        for tok in tokenize.generate_tokens(readline):
            if tok.type == tokenize.COMMENT:
                continue
            out.append(tok)
        return tokenize.untokenize(out)
    except (tokenize.TokenError, ValueError):
        return src


class _DocStrip(ast.NodeTransformer):
    @staticmethod
    def _strip_leading_string(body: list[ast.stmt]) -> list[ast.stmt]:
        if not body:
            return body
        first = body[0]
        if not isinstance(first, ast.Expr):
            return body
        v = first.value
        if isinstance(v, ast.Constant) and isinstance(v.value, str):
            return body[1:]
        return body

    def visit_Module(self, node: ast.Module) -> ast.Module:
        node.body = self._strip_leading_string(node.body)
        return self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        self.generic_visit(node)
        node.body = self._strip_leading_string(node.body)
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AsyncFunctionDef:
        self.generic_visit(node)
        node.body = self._strip_leading_string(node.body)
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        self.generic_visit(node)
        node.body = self._strip_leading_string(node.body)
        return node


def strip_python(path: Path) -> None:
    src = path.read_text(encoding="utf-8")
    src = strip_python_hash(src)
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        raise RuntimeError(f"{path}: {e}") from e
    tree = _DocStrip().visit(tree)
    ast.fix_missing_locations(tree)
    out = ast.unparse(tree) + "\n"
    path.write_text(out, encoding="utf-8")


def strip_shell_line(line: str, *, is_first: bool) -> str | None:
    raw = line.rstrip("\n")
    if is_first and raw.startswith("#!"):
        return raw + "\n"
    s = raw.lstrip()
    if not s:
        return "\n"
    if s.startswith("#"):
        return None
    out = []
    i = 0
    in_squote = in_dquote = False
    esc = False
    while i < len(raw):
        c = raw[i]
        if esc:
            out.append(c)
            esc = False
            i += 1
            continue
        if c == "\\" and (in_squote or in_dquote):
            out.append(c)
            esc = True
            i += 1
            continue
        if not in_dquote and c == "'" and not in_squote:
            in_squote = True
            out.append(c)
            i += 1
            continue
        if not in_squote and c == '"' and not in_dquote:
            in_dquote = True
            out.append(c)
            i += 1
            continue
        if in_squote and c == "'":
            in_squote = False
            out.append(c)
            i += 1
            continue
        if in_dquote and c == '"':
            in_dquote = False
            out.append(c)
            i += 1
            continue
        if not in_squote and not in_dquote and c == "#":
            break
        out.append(c)
        i += 1
    line_out = "".join(out).rstrip()
    return (line_out + "\n") if line_out else None


def strip_shell(path: Path) -> None:
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    out: list[str] = []
    for idx, line in enumerate(lines):
        if line.endswith("\r\n"):
            core = line[:-2] + "\n"
            nl = "\r\n"
        else:
            core = line
            nl = "\n" if line.endswith("\n") else ""
        if core.endswith("\n"):
            core = core[:-1]
        stripped = strip_shell_line(core + "\n", is_first=(idx == 0))
        if stripped is None:
            continue
        if stripped.endswith("\n"):
            stripped = stripped[:-1]
        out.append(stripped + (nl if line.endswith("\n") else ""))
    path.write_text("".join(out), encoding="utf-8")


def strip_yamlish(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    out: list[str] = []
    for line in lines:
        s = line.lstrip()
        if s.startswith("#"):
            continue
        line = re.sub(r"\s+#.*$", "", line)
        out.append(line)
    while out and out[-1] == "":
        out.pop()
    path.write_text("\n".join(out) + ("\n" if text.endswith("\n") else ""), encoding="utf-8")


def strip_env_example(path: Path) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    out = [ln for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def main() -> None:
    for rel in ("gateway.py", "crew.py", "lambda_pricing.py"):
        strip_python(ROOT / rel)
    for sh in sorted(ROOT.glob("scripts/**/*.sh")):
        strip_shell(sh)
    for rel in (
        "monitoring/prometheus.yml",
        "monitoring/docker-compose.yml",
        "monitoring/grafana/provisioning/dashboards/dashboards.yaml",
    ):
        strip_yamlish(ROOT / rel)
    strip_env_example(ROOT / ".env.example")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        (ROOT / "_strip_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
