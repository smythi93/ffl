"""Function spans for function-level outcomes."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class FunctionSpan:
    file: str
    name: str
    start: int
    end: int

    @property
    def length(self) -> int:
        return max(1, self.end - self.start + 1)


def _spans_of_source(file_name: str, source: str) -> List[FunctionSpan]:
    tree = ast.parse(source)
    spans: List[FunctionSpan] = []

    class Visitor(ast.NodeVisitor):
        def __init__(self):
            self.stack: List[str] = []

        def _add(self, node):
            start = int(getattr(node, "lineno", 0) or 0)
            end = int(getattr(node, "end_lineno", start) or start)
            qualname = ".".join(self.stack + [node.name])
            spans.append(FunctionSpan(file_name, qualname, start, end))

        def visit_FunctionDef(self, node):
            self._add(node)
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_ClassDef(self, node):
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

    Visitor().visit(tree)
    spans.sort(key=lambda s: (s.start, s.end - s.start))
    return spans


def build_index(base_dir: Path, files: Iterable[str]) -> Dict[str, List[FunctionSpan]]:
    index: Dict[str, List[FunctionSpan]] = {}
    for file_name in sorted(set(files)):
        path = Path(file_name)
        if not path.is_absolute():
            path = base_dir / path
        if path.suffix != ".py" or not path.exists():
            continue
        try:
            index[file_name] = _spans_of_source(
                file_name, path.read_text(encoding="utf-8", errors="ignore")
            )
        except (SyntaxError, ValueError, OSError):
            continue
    return index


def enclosing(
    index: Dict[str, List[FunctionSpan]], file: str, line: int
) -> Optional[FunctionSpan]:
    """Innermost function containing ``line`` (shortest matching span)."""
    candidates = [s for s in index.get(file, []) if s.start <= line <= s.end]
    if not candidates:
        return None
    candidates.sort(key=lambda s: (s.end - s.start, s.start))
    return candidates[0]


def assign(
    index: Dict[str, List[FunctionSpan]], locations: Iterable[Tuple[str, int]]
) -> Tuple[List[FunctionSpan], List[int]]:
    """Return the function table and, per location, its function index (-1 if none)."""
    table: List[FunctionSpan] = []
    lookup: Dict[FunctionSpan, int] = {}
    assignment: List[int] = []
    for file, line in locations:
        span = enclosing(index, file, int(line))
        if span is None:
            assignment.append(-1)
            continue
        if span not in lookup:
            lookup[span] = len(table)
            table.append(span)
        assignment.append(lookup[span])
    return table, assignment
