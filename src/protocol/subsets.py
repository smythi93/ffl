"""Exploratory: which subsets of the feature vocabulary carry localization signal?

Not registered. A subset selects features by type and, where a type has internal
structure, by that structure: the comparator of scalar-pair, variable and return
predicates, the operand of return predicates, the direction of a branch, the
form of a condition, the regime of a length predicate and the span of a
definition-use pair. Each subset is scored three ways, always through the
registered mapping, maximum aggregation and blend:

``spectrum``  the feature spectrum computed on the subset alone (univariate, so
              identical to instrumenting only these features);
``rf``        the registered forest's importances restricted to the subset (the
              registered family ablation, at finer granularity);
``neg``       uniform random importances restricted to the subset, the density
              control: a subset that helps only because of how many features it
              puts on a line helps under ``neg`` as well.

Method names are ``s_<source>_<subset>``.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

TYPE_NAMES = {
    0: "line", 1: "branch", 2: "function", 3: "loop", 4: "defuse", 5: "condition",
    6: "scalarpair", 7: "variable", 8: "return", 9: "none", 10: "emptystring",
    11: "asciistring", 12: "digitstring", 13: "specialstring", 14: "emptybytes",
    15: "length", 16: "functionerror",
}
COMPARING = {6: "scalarpair", 7: "variable", 8: "return"}
OPERATORS = ("EQ", "NE", "LT", "LE", "GT", "GE")
VALUE_TYPES = {7, 9, 10, 11, 12, 13, 14, 15}  # features that name a variable at a line
SOURCES = ("spectrum", "rf", "neg")

_OPERATOR = re.compile("|".join(OPERATORS))
_OPERAND = re.compile(r"^[A-Za-z_][\w.\[\]'\"()]*$|^-?[0-9.]+$|^None$")


def _location(name: str) -> Tuple[str, str]:
    parts = name.split(":")
    return parts[1], parts[2]


def split_comparison(name: str, known: Optional[Set[str]] = None) -> Optional[Tuple[str, str, str]]:
    """``(left, operator, right)`` of a comparing predicate, or None if ambiguous.

    Names are ``<left><OP><right>`` without separators, so every occurrence of an
    operator token is a candidate split. A split is valid if both sides look like
    operands. With several valid splits, the one whose left operand is observed
    as a variable at the same line by another feature (``known``) wins.
    """
    predicate = name.split(":", 3)[-1]
    candidates = []
    for match in _OPERATOR.finditer(predicate):
        left, right = predicate[:match.start()], predicate[match.end():]
        if left and right and _OPERAND.match(left) and _OPERAND.match(right):
            candidates.append((left, match.group(0), right))
    if len(candidates) == 1:
        return candidates[0]
    if known:
        preferred = [c for c in candidates if c[0] in known]
        if len(preferred) == 1:
            return preferred[0]
    return None


def condition_form(condition: str) -> str:
    if " and " in condition or " or " in condition:
        return "andor"
    if condition.startswith("not "):
        return "not"
    if re.search(r"\bis\b", condition):
        return "is"
    if re.search(r"\bin\b", condition):
        return "in"
    if re.search(r"==|!=|<=|>=|<|>", condition):
        return "compare"
    return "plain"


def defuse_span(name: str) -> str:
    parts = name.split(":")
    if len(parts) < 5 or parts[1] != parts[3]:
        return "crossfile"
    try:
        span = abs(int(parts[4]) - int(parts[2]))
    except ValueError:
        return "unknown"
    return "sameline" if span == 0 else "near" if span <= 5 else "far"


def subset_labels(types: np.ndarray, names) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    """Boolean feature masks for every subset, and diagnostics of the parsing."""
    names = [str(n) for n in names]
    types = np.asarray(types)
    #: Variables observed at each line by the value features, to disambiguate
    #: comparisons whose operands contain operator tokens.
    variables: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    for name, t in zip(names, types):
        if int(t) in VALUE_TYPES:
            parts = name.split(":")
            if len(parts) >= 4:
                variables[(parts[1], parts[2])].add(
                    re.sub(r"(EQ|NE|LT|LE|GT|GE)-?[0-9.]+$", "", parts[3]))
    sub = np.full(len(names), "", dtype=object)
    stats = {"comparisons": 0, "ambiguous": 0}
    for i, (name, t) in enumerate(zip(names, types)):
        t = int(t)
        if t in COMPARING:
            stats["comparisons"] += 1
            split = split_comparison(name, variables.get(_location(name)))
            if split is None:
                stats["ambiguous"] += 1
                continue
            left, op, right = split
            sub[i] = op.lower()
            if t == 8:
                operand = "none" if right == "None" else "zero" if right == "0" else "other"
                sub[i] = f"{op.lower()}|{operand}"
        elif t == 1:
            sub[i] = name.split(":")[-2]
        elif t == 5:
            sub[i] = condition_form(name.split(":", 3)[-1])
        elif t == 15:
            sub[i] = name.rsplit("_", 1)[-1]  # 0, 1, more
        elif t == 4:
            sub[i] = defuse_span(name)

    masks: Dict[str, np.ndarray] = {}
    for type_id, label in TYPE_NAMES.items():
        masks[label] = types == type_id
    ops = np.array([s.split("|")[0] if s else "" for s in sub])
    for type_id, label in COMPARING.items():
        for op in OPERATORS:
            masks[f"{label}{op.lower()}"] = (types == type_id) & (ops == op.lower())
    operands = np.array([s.split("|")[1] if "|" in s else "" for s in sub])
    for operand in ("none", "zero", "other"):
        masks[f"returnvs{operand}"] = (types == 8) & (operands == operand)
    for direction in ("then", "else"):
        masks[f"branch{direction}"] = (types == 1) & (sub == direction)
    for form in ("compare", "in", "is", "not", "andor", "plain"):
        masks[f"condition{form}"] = (types == 5) & (sub == form)
    for regime in ("0", "1", "more"):
        masks[f"length{regime}"] = (types == 15) & (sub == regime)
    for span in ("sameline", "near", "far", "crossfile"):
        masks[f"defuse{span}"] = (types == 4) & (sub == span)
    return masks, stats


def subset_names() -> List[str]:
    names = list(TYPE_NAMES.values())
    names += [f"{label}{op.lower()}" for label in COMPARING.values() for op in OPERATORS]
    names += [f"returnvs{o}" for o in ("none", "zero", "other")]
    names += [f"branch{d}" for d in ("then", "else")]
    names += [f"condition{f}" for f in ("compare", "in", "is", "not", "andor", "plain")]
    names += [f"length{r}" for r in ("0", "1", "more")]
    names += [f"defuse{s}" for s in ("sameline", "near", "far", "crossfile")]
    return names


def methods(sources=SOURCES) -> List[str]:
    return [f"s_{source}_{subset}" for source in sources for subset in subset_names()]


def subset_weights(artifact, vectors_dir) -> Dict[str, Dict[str, list]]:
    """Line weights (maximum aggregation) of every ``s_<source>_<subset>`` method.

    Computed from the subject's stored feature matrix through the registered
    feature-to-line map, exactly as the registered weights are.
    """
    from pathlib import Path

    from protocol.extract import aggregate_weights, normalize
    from protocol.posthoc import spectrum_scores

    path = Path(vectors_dir) / f"{artifact['subject']}.npz"
    if not path.exists():
        return {}
    with np.load(path, allow_pickle=True) as data:
        matrix, labels = data["matrix"], data["labels"].astype(int)
        types, names = data["types"], data["features"]
        flat, offsets = data["map_flat"], data["map_offsets"]
        sources = {
            "spectrum": spectrum_scores(matrix, labels),
            "rf": np.asarray(data["importance_rf"], dtype=float),
            "neg": np.asarray(data["importance_neg"], dtype=float),
        }
    primary = [flat[offsets[i]:offsets[i + 1]].tolist() for i in range(len(offsets) - 1)]
    n_lines = len(artifact["lines"]["line"])
    identity = {i: i for i in range(n_lines)}
    masks, _ = subset_labels(types, names)
    weights = {}
    for source, scores in sources.items():
        if len(scores) != len(types):
            continue
        for subset, mask in masks.items():
            restricted = np.where(mask, scores, 0.0)
            line = aggregate_weights(primary, normalize(restricted), identity, n_lines)["max"]
            weights[f"s_{source}_{subset}"] = {"max": line.tolist()}
    return weights
