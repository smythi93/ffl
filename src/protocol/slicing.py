"""Exploratory: dynamic slicing and failure proximity on the ordered traces.

Not registered and not part of the study's verdict. Both signals come from a
*single* failing run, so unlike every statistical scorer they do not need
several failing tests (72% of the subjects have exactly one).

**Dynamic backward slice.** Each executed line is an *instance*. Instances
depend on

* the instances that last defined the values they use (data dependence: a use
  of ``(name, object id)`` depends on the last definition of the same pair, or
  else on the last definition of the same object under any name, which covers
  argument passing and aliasing; arguments are bound at the call site),
* the latest instance, in the same frame, of their enclosing compound statement
  header (``if``/``for``/``while``/``try``/``with``/``match``; control dependence
  from the source's AST), or on the call site if there is none,
* the last instance of each function they called (return values).

The slicing criterion is what reaches the test: the returning instance of every
call the test made into the subject, and the crash site of every exception that
propagated to the test. Lines in the slice could have influenced the failure;
lines outside it could not (up to the approximations above).

**Failure proximity.** The failure point is the last criterion instance. Lines
are ranked by how recently before it they last executed.

Per failing run the result is, for every executed line, its slice depth (BFS
distance from the criterion, absent if not in the slice) and its recency rank.
"""

from __future__ import annotations

import ast
import gzip
import os
import pickle
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

Location = Tuple[str, int]

_HEADERS = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.With, ast.AsyncWith)
if hasattr(ast, "TryStar"):
    _HEADERS += (ast.TryStar,)
if hasattr(ast, "Match"):
    _HEADERS += (ast.Match,)
_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)


def control_parents(source: str) -> Dict[int, int]:
    """Line -> line of its innermost enclosing compound-statement header,
    within the same function (function boundaries reset the chain)."""
    parents: Dict[int, int] = {}
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return parents

    def visit(node, header: Optional[int]):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, _SCOPES):
                visit(child, None)
                continue
            line = getattr(child, "lineno", None)
            #: Line events carry a statement's first line, and nested
            #: statements get their own (inner) header by recursion.
            if isinstance(child, ast.stmt) and line is not None and header is not None:
                parents[line] = header
            if isinstance(child, _HEADERS):
                visit(child, child.lineno)
            else:
                visit(child, header)

    visit(tree, None)
    return parents


class Sources:
    """Control parents per file of one checkout, parsed on first use."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.cache: Dict[str, Dict[int, int]] = {}

    def parent(self, file: str, line: int) -> Optional[int]:
        table = self.cache.get(file)
        if table is None:
            path = self.root / file
            try:
                table = control_parents(path.read_text(errors="ignore"))
            except OSError:
                table = {}
            self.cache[file] = table
        return table.get(line)


def analyze_events(events, sources: Sources) -> Dict:
    """Slice depths and recency ranks of one failing run."""
    from sflkitlib.events.event import (
        BranchEvent, DefEvent, FunctionEnterEvent, FunctionErrorEvent,
        FunctionExitEvent, LineEvent, UseEvent,
    )

    lines: List[Location] = []          # instance -> location
    deps: List[List[int]] = []          # instance -> instances it depends on
    last_def: Dict[tuple, int] = {}
    last_object: Dict[int, int] = {}
    #: Frame: [current instance, call-site instance, {header line: instance}, last instance]
    frames: List[list] = [[None, None, {}, None]]
    criterion: List[int] = []
    error_origin: Optional[int] = None
    previous_error = False

    for event in events:
        kind = type(event)
        if kind is LineEvent:
            frame = frames[-1]
            location = (event.file, int(event.line))
            index = len(lines)
            lines.append(location)
            parents = []
            header = sources.parent(event.file, int(event.line))
            if header is not None and header in frame[2]:
                parents.append(frame[2][header])
            elif frame[1] is not None:
                parents.append(frame[1])
            deps.append(parents)
            frame[0] = index
            frame[3] = index
            #: This line may itself be a header for later lines.
            frame[2][int(event.line)] = index
            previous_error = False
        elif kind is UseEvent:
            #: Before a callee's first line (argument binding), events belong
            #: to the call site.
            current = frames[-1][0] if frames[-1][0] is not None else frames[-1][1]
            source = last_def.get((event.var, event.var_id))
            if source is None:
                #: The same object bound under another name (argument passing,
                #: aliasing). Identity can over-approximate (interned values),
                #: which is the safe side for a slice.
                source = last_object.get(event.var_id)
            if current is not None and source is not None and source != current:
                deps[current].append(source)
        elif kind is DefEvent:
            current = frames[-1][0] if frames[-1][0] is not None else frames[-1][1]
            if current is not None:
                last_def[(event.var, event.var_id)] = current
                last_object[event.var_id] = current
        elif kind is FunctionEnterEvent:
            frames.append([None, frames[-1][0], {}, None])
            previous_error = False
        elif kind is FunctionExitEvent or kind is FunctionErrorEvent:
            if len(frames) <= 1:
                continue
            callee = frames.pop()
            caller = frames[-1]
            if kind is FunctionErrorEvent:
                if not previous_error:
                    error_origin = callee[3]
                previous_error = True
            else:
                previous_error = False
            if callee[3] is not None and caller[0] is not None:
                deps[caller[0]].append(callee[3])
            if len(frames) == 1:
                #: Returned to the test (or the module level).
                if kind is FunctionErrorEvent:
                    #: An exception reached the test: its origin is the
                    #: criterion, not the frame's last line (nothing returned).
                    if error_origin is not None:
                        criterion.append(error_origin)
                elif callee[3] is not None:
                    criterion.append(callee[3])
        elif kind is BranchEvent:
            pass  # branch direction is implied by the next executed line

    result = {"instances": len(lines), "criterion": len(criterion)}
    if not lines:
        return {**result, "depth": {}, "recency": {}}
    if not criterion:
        #: Nothing returned to the test (e.g. the run was cut off): fall back to
        #: the last executed instance.
        criterion = [len(lines) - 1]

    #: Backward BFS over the dependence graph.
    depth_of = np.full(len(lines), -1, dtype=np.int64)
    queue = deque()
    for c in set(criterion):
        depth_of[c] = 0
        queue.append(c)
    while queue:
        node = queue.popleft()
        for parent in deps[node]:
            if depth_of[parent] < 0:
                depth_of[parent] = depth_of[node] + 1
                queue.append(parent)
    depth: Dict[Location, int] = {}
    for index in np.flatnonzero(depth_of >= 0):
        location = lines[index]
        d = int(depth_of[index])
        if location not in depth or d < depth[location]:
            depth[location] = d

    #: Recency before the failure point (the last criterion instance).
    failure = max(criterion)
    recency: Dict[Location, int] = {}
    rank = 0
    for index in range(failure, -1, -1):
        location = lines[index]
        if location not in recency:
            rank += 1
            recency[location] = rank
    executed = len(set(lines))
    return {**result, "depth": depth, "recency": recency, "executed": executed,
            "slice_lines": len(depth)}


def analyze_trace(job) -> Tuple[str, str]:
    """Analyse one failing event file and cache the result."""
    from sflkit.events.event_file import EventFile
    from sflkit.events.mapping import EventMapping

    event_path, mapping_path, source_root, out_path = (Path(p) for p in job)
    if out_path.exists():
        return str(out_path), "cached"
    try:
        mapping = EventMapping.load_from_file(mapping_path, "")
        with EventFile(event_path, 0, mapping, failing=True) as event_file:
            result = analyze_events(event_file.load(), Sources(source_root))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(".tmp")
        with gzip.open(tmp, "wb", compresslevel=3) as handle:
            pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, out_path)
    except Exception as error:  # noqa: BLE001
        return str(out_path), f"failed: {type(error).__name__}: {error}"
    return str(out_path), "ok"


def trace_jobs(events_root: Path, mappings_root: Path, checkouts: Path, cache_root: Path):
    from protocol.fsm import cache_name

    jobs = []
    for subject in sorted(events_root.glob("*/*")):
        identifier = f"{subject.parent.name}_{subject.name}"
        mapping = mappings_root / f"{identifier}_buggy.json"
        folder = subject / "failing"
        if not mapping.exists() or not (subject / ".complete").exists() or not folder.exists():
            continue
        for trace in folder.iterdir():
            out = cache_root / identifier / cache_name(trace.name)
            jobs.append((trace.stat().st_size, str(trace), str(mapping),
                         str(checkouts / identifier), str(out)))
    jobs.sort(reverse=True)
    return [job[1:] for job in jobs]


def subject_weights(cache: Path, artifact: Dict) -> Tuple[Dict[str, Dict[str, list]], Dict]:
    """Per-line weights from all failing runs of one subject.

    ``slice``       share of failing runs whose slice contains the line;
    ``slice_depth`` mean over failing runs of 1 / (1 + slice depth), 0 outside;
    ``recency``     mean over failing runs of 1 / recency rank;
    ``slice_recency`` recency restricted to the slice.
    """
    results = []
    for path in sorted(cache.glob("*.pkl.gz")):
        with gzip.open(path, "rb") as handle:
            results.append(pickle.load(handle))
    if not results:
        return {}, {}
    lines = artifact["lines"]
    position = {(f, int(l)): i for i, (f, l) in enumerate(zip(lines["file"], lines["line"]))}
    n = len(position)
    member = np.zeros(n)
    depth = np.zeros(n)
    recency = np.zeros(n)
    sliced_recency = np.zeros(n)
    for result in results:
        for location, d in result["depth"].items():
            i = position.get(location)
            if i is not None:
                member[i] += 1.0
                depth[i] += 1.0 / (1.0 + d)
        for location, r in result["recency"].items():
            i = position.get(location)
            if i is not None:
                recency[i] += 1.0 / r
                if location in result["depth"]:
                    sliced_recency[i] += 1.0 / r
    k = float(len(results))
    weights = {}
    for name, values in (("slice", member / k), ("slice_depth", depth / k),
                         ("recency", recency / k), ("slice_recency", sliced_recency / k)):
        values = values / values.max() if values.max() > 0 else values
        weights[name] = {"max": values.tolist(), "mean": values.tolist(), "sum": values.tolist()}
    for name in ("slice", "slice_depth", "recency", "slice_recency"):
        weights[f"tie_{name}"] = weights[name]
    executed = [r.get("executed", 0) for r in results]
    in_slice = [r.get("slice_lines", 0) for r in results]
    stats = {
        "failing_runs": len(results),
        "executed_lines": float(np.mean(executed)),
        "slice_lines": float(np.mean(in_slice)),
        "slice_share": float(np.mean([s / e for s, e in zip(in_slice, executed) if e])),
        "criterion": float(np.mean([r.get("criterion", 0) for r in results])),
    }
    return weights, stats


SOURCES = ("slice", "slice_depth", "recency", "slice_recency")
METHODS = (
    list(SOURCES) + [f"tie_{s}" for s in SOURCES] + [f"lex_{s}" for s in SOURCES]
    + [f"h_feat_{s}" for s in SOURCES]
)
