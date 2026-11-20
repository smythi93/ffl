"""Reference-patch and patch-hunk ground truth.

The registered report distinguishes two patch-based ground-truth sets
(Section "Operational Definitions"):

* *reference faulty lines* -- the exact lines modified by the official
  Tests4Py fix, obtained from ``tests4py.api.get_faulty_lines``;
* *patch-hunk sets* -- the corresponding modified hunk regions, i.e. every
  source line of the buggy revision that falls inside a hunk touched by the
  official fix.

Test files are excluded from both sets, mirroring ``get_faulty_lines``.
"""

from __future__ import annotations

import importlib.resources
from typing import List, Set, Tuple

from unidiff import PatchSet

import tests4py.api as t4p
from tests4py.projects import resources
from tests4py.projects import Project

Loc = Tuple[str, int]


def _patch_text(project: Project) -> str:
    project_resources = importlib.resources.files(
        getattr(getattr(resources, project.project_name), f"bug_{project.bug_id}")
    )
    with importlib.resources.as_file(project_resources.joinpath("fix.patch")) as path:
        with open(path, "r") as fp:
            return fp.read()


def reference_faulty_lines(project: Project) -> Set[Loc]:
    """Narrow target: the statement each fix line belongs to.

    ``get_faulty_lines`` reports the exact lines the fix touches, which an
    execution-based tool cannot always match: a fix on the second line of a
    two-line statement is never reported by the tracer, and a pure insertion has
    no line in the buggy revision at all. The statement scope resolves both.
    """
    return {
        (hunk.file, line)
        for hunk in t4p.get_hunk_faulty_lines(project, scope="statement")
        for line in hunk.lines()
    }


def block_hunk_lines(project: Project) -> Set[Loc]:
    """Broad target: each faulty statement plus its neighbouring statements.

    Replaces the diff-context notion used before. Which lines a diff prints as
    context depends on how the patch was formatted, not on the code, so the two
    targets were defined on different footings. Both are now structural.
    """
    return {
        (hunk.file, line)
        for hunk in t4p.get_hunk_faulty_lines(project, scope="block")
        for line in hunk.lines()
    }


def patch_hunk_lines(project: Project) -> Set[Loc]:
    """All buggy-revision source lines covered by a hunk of the official fix."""
    lines: Set[Loc] = set()
    try:
        text = _patch_text(project)
    except (IOError, AttributeError, ModuleNotFoundError):
        return lines
    for diff in PatchSet(text):
        if project.test_base and diff.path.startswith(str(project.test_base)):
            continue
        if not diff.path.endswith(".py"):
            continue
        for hunk in diff:
            for line in hunk:
                # ``source_line_no`` is defined for context and removed lines,
                # i.e. exactly the lines that exist in the buggy revision.
                if line.source_line_no is None:
                    continue
                if not line.value.strip():
                    continue
                lines.add((diff.path, int(line.source_line_no)))
    return lines


def ground_truth(project: Project) -> Tuple[List[Loc], List[Loc]]:
    """The narrow and broad fault targets, both defined on program structure."""
    reference = sorted(reference_faulty_lines(project))
    hunks = sorted(block_hunk_lines(project) | set(reference))
    return reference, hunks
