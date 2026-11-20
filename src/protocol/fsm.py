"""Exploratory: a control-flow automaton over the ordered event traces.

Not registered and not part of the study's verdict. The registered feature
vectors are unordered: they record *which* predicates held during a test, not in
which order the program got there. This module reads the ordered traces and
builds, per subject, a finite-state model of the executions whose states are
executed lines. Its transitions come at three granularities:

``call``  caller line -> entered function (the inter-procedural automaton);
``edge``  line -> next line executed in the same frame (control flow);
``path``  three consecutive lines in the same frame (control flow in context).

A per-frame stack, driven by the function enter, exit and error events, keeps
the line sequence of a caller apart from the lines of the functions it calls.

Each test is reduced to the set of transitions it takes, and transitions are
scored exactly as the post-hoc scorers score feature values:

``fsm_<kind>`` / ``fsm``  Ochiai of taking the transition (one kind / pooled);
``fsm_nn``                nearest-neighbour contrast over the pooled sets;
``fsm_neg``               uniform random scores, the density control.

A transition is mapped to every line it connects, then aggregated per line like
the registered weights. The ``g_{func,block,line}_<source>`` weights feed the
granularity-matched hierarchical descent (call edges select files and
functions, control-flow edges the blocks, paths the lines).
"""

from __future__ import annotations

import gzip
import os
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

KINDS = ("edge", "path", "call")
#: Hierarchy level each transition kind describes.
LEVELS = {"func": "call", "block": "edge", "line": "path"}
SOURCES = ("fsm", "fsm_nn", "fsm_neg")

Location = Tuple[str, int]


def transitions_from_events(events: Iterable) -> Dict[str, Set[tuple]]:
    """The transitions one execution takes, from its ordered events."""
    from sflkitlib.events.event import (
        FunctionEnterEvent, FunctionErrorEvent, FunctionExitEvent, LineEvent,
    )

    edges: Set[tuple] = set()
    paths: Set[tuple] = set()
    calls: Set[tuple] = set()
    #: One frame per active call: [second-to-last line, last line].
    stack: List[list] = [[None, None]]
    for event in events:
        kind = type(event)
        if kind is LineEvent:
            key = (event.file, int(event.line))
            frame = stack[-1]
            last = frame[1]
            if last is not None:
                edges.add((last, key))
                if frame[0] is not None:
                    paths.add((frame[0], last, key))
            frame[0], frame[1] = last, key
        elif kind is FunctionEnterEvent:
            calls.add((stack[-1][1], (event.file, int(event.line))))
            stack.append([None, None])
        elif kind is FunctionExitEvent or kind is FunctionErrorEvent:
            if len(stack) > 1:
                stack.pop()
    return {"edge": edges, "path": paths, "call": calls}


# ---------------------------------------------------------------------------
# Trace decoding (one event file -> one cached transition set)
# ---------------------------------------------------------------------------
_MAPPINGS: Dict[str, object] = {}


def _mapping(path: Path):
    from sflkit.events.mapping import EventMapping

    key = str(path)
    if key not in _MAPPINGS:
        _MAPPINGS.clear()  # one subject at a time per worker is enough
        _MAPPINGS[key] = EventMapping.load_from_file(path, "")
    return _MAPPINGS[key]


def decode_trace(job) -> Tuple[str, str]:
    """Decode one event file into its cached transition sets."""
    from sflkit.events.event_file import EventFile

    event_path, mapping_path, out_path = (Path(p) for p in job)
    if out_path.exists():
        return str(out_path), "cached"
    try:
        with EventFile(event_path, 0, _mapping(mapping_path)) as event_file:
            transitions = transitions_from_events(event_file.load())
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(".tmp")
        with gzip.open(tmp, "wb", compresslevel=3) as handle:
            pickle.dump({k: sorted(v, key=repr) for k, v in transitions.items()}, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, out_path)
    except Exception as error:  # noqa: BLE001
        return str(out_path), f"failed: {type(error).__name__}: {error}"
    return str(out_path), "ok"


def cache_name(trace_name: str) -> str:
    """Cache file name of a trace; test ids can exceed the 255-byte name limit."""
    import hashlib

    if len(trace_name.encode()) <= 180:
        return f"{trace_name}.pkl.gz"
    digest = hashlib.sha1(trace_name.encode()).hexdigest()[:16]
    return f"{trace_name.encode()[:120].decode(errors='ignore')}_{digest}.pkl.gz"


def trace_jobs(events_root: Path, mappings_root: Path, cache_root: Path,
               identifiers: Optional[Set[str]] = None) -> List[Tuple[str, str, str]]:
    """One job per failing or passing event file, largest first."""
    jobs = []
    for subject in sorted(events_root.glob("*/*")):
        identifier = f"{subject.parent.name}_{subject.name}"
        if identifiers is not None and identifier not in identifiers:
            continue
        mapping = mappings_root / f"{identifier}_buggy.json"
        if not mapping.exists() or not (subject / ".complete").exists():
            continue
        for outcome in ("failing", "passing"):
            folder = subject / outcome
            if not folder.exists():
                continue
            for trace in folder.iterdir():
                out = cache_root / identifier / outcome / cache_name(trace.name)
                jobs.append((trace.stat().st_size, str(trace), str(mapping), str(out)))
    jobs.sort(reverse=True)
    return [job[1:] for job in jobs]


# ---------------------------------------------------------------------------
# Scoring (cached transition sets -> per-line weights)
# ---------------------------------------------------------------------------
def load_subject(cache: Path) -> Tuple[List[Dict[str, list]], np.ndarray]:
    runs, labels = [], []
    for outcome, label in (("failing", 1), ("passing", 0)):
        folder = cache / outcome
        if not folder.exists():
            continue
        for path in sorted(folder.glob("*.pkl.gz")):
            with gzip.open(path, "rb") as handle:
                runs.append(pickle.load(handle))
            labels.append(label)
    return runs, np.asarray(labels, dtype=int)


def ochiai(ef: np.ndarray, ep: np.ndarray, n_failing: int) -> np.ndarray:
    denominator = np.sqrt(max(n_failing, 1) * (ef + ep))
    return np.divide(ef, denominator, out=np.zeros_like(ef, dtype=float),
                     where=denominator > 0)


def nearest_neighbour(sets: Sequence[np.ndarray], labels: np.ndarray, n_items: int) -> np.ndarray:
    """Nearest-neighbour contrast on transition sets (cf. ``posthoc``):
    per failing run, the share of its nearest passing runs (smallest symmetric
    difference) that differ from it in each transition; averaged over failing runs."""
    failing = [s for s, l in zip(sets, labels) if l == 1]
    passing = [s for s, l in zip(sets, labels) if l == 0]
    scores = np.zeros(n_items, dtype=float)
    if not failing or not passing:
        return scores
    for f in failing:
        diffs = [np.setxor1d(f, p, assume_unique=True) for p in passing]
        sizes = np.array([len(d) for d in diffs])
        nearest = [d for d, s in zip(diffs, sizes) if s == sizes.min()]
        for d in nearest:
            scores[d] += 1.0 / len(nearest)
    return scores / len(failing)


def subject_weights(cache: Path, artifact: Dict, seed: int) -> Tuple[Dict[str, Dict[str, list]], Dict]:
    """Every ``fsm*`` and ``g_*_fsm*`` weighting of one subject, and model sizes."""
    from protocol.extract import aggregate_weights, normalize

    runs, labels = load_subject(cache)
    if len(runs) == 0 or labels.sum() == 0 or labels.sum() == len(labels):
        return {}, {}
    lines = artifact["lines"]
    position = {(f, int(l)): i for i, (f, l) in enumerate(zip(lines["file"], lines["line"]))}
    n_lines = len(position)
    identity = {i: i for i in range(n_lines)}

    #: Pooled vocabulary of transitions; ``kind_of`` tells which kind each is.
    vocabulary: Dict[tuple, int] = {}
    kind_of: List[str] = []
    locations: List[List[int]] = []
    sets: List[np.ndarray] = []
    for run in runs:
        ids = []
        for kind in KINDS:
            for item in run.get(kind, ()):
                key = (kind, item)
                index = vocabulary.get(key)
                if index is None:
                    index = len(vocabulary)
                    vocabulary[key] = index
                    kind_of.append(kind)
                    locations.append(sorted({position[loc] for loc in item
                                             if loc is not None and loc in position}))
                ids.append(index)
        sets.append(np.unique(np.asarray(ids, dtype=np.int64)))
    n_items = len(vocabulary)
    kinds = np.asarray(kind_of)
    failing = labels == 1
    ef = np.zeros(n_items)
    ep = np.zeros(n_items)
    for s, label in zip(sets, labels):
        (ef if label == 1 else ep)[s] += 1.0
    n_failing = int(failing.sum())

    scores = {
        "fsm": ochiai(ef, ep, n_failing),
        "fsm_nn": nearest_neighbour(sets, labels, n_items),
        "fsm_neg": np.random.default_rng(seed + 7).random(n_items),
    }

    def weigh(values: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict[str, list]:
        values = values if mask is None else np.where(mask, values, 0.0)
        aggregated = aggregate_weights(locations, normalize(values), identity, n_lines)
        return {k: v.tolist() for k, v in aggregated.items()}

    weights: Dict[str, Dict[str, list]] = {}
    for source, values in scores.items():
        weights[source] = weigh(values)
        for level, kind in LEVELS.items():
            weights[f"g_{level}_{source}"] = weigh(values, kinds == kind)
    for kind in KINDS:
        weights[f"fsm_{kind}"] = weigh(scores["fsm"], kinds == kind)
    weights["tie_fsm"] = weights["fsm"]
    weights["tie_fsm_nn"] = weights["fsm_nn"]
    stats = {
        "runs": int(len(labels)), "failing": n_failing,
        **{f"n_{kind}": int((kinds == kind).sum()) for kind in KINDS},
    }
    return weights, stats


METHODS = (
    ["fsm", "fsm_edge", "fsm_path", "fsm_call", "fsm_nn", "fsm_neg", "tie_fsm", "tie_fsm_nn"]
    + [f"h_{mode}_{source}" for mode in ("feat", "blend") for source in SOURCES]
    + [f"h_gran_{source}" for source in SOURCES]
)
