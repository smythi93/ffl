#!/usr/bin/env python3
"""Driver for the pre-registered execution-feature fault-localization study.

Sub-commands
------------
``subject``   run Phases 1-2 for one Tests4Py subject and write its artifact
``run``       run ``subject`` for a list of projects/subjects, with workers
``status``    summarise which artifacts exist and which subjects failed
``collect``   turn artifacts into the tidy per-(subject, formula, method) table
``analyze``   run the pre-registered analysis plan on the tidy table
``tables``    render the LaTeX tables and figures of the manuscript
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Dict, Iterable, List, Optional

sys.path.insert(0, str(Path(__file__).parent / "src"))

import protocol.compat  # noqa: E402,F401  (Tests4Py configuration shim)

ROOT = Path(__file__).parent
ARTIFACTS = ROOT / "artifacts"
LOGS = ROOT / "logs"

#: Tests4Py projects of the study.  ``pandas`` is excluded because its faults
#: are not reproducible with the Tests4Py harness, leaving the 310 subjects of
#: the registered report.
PROJECTS = [
    "ansible", "black", "calculator", "cookiecutter", "expression", "fastapi",
    "httpie", "keras", "luigi", "markup", "matplotlib", "middle", "pysnooper",
    "sanic", "scrapy", "spacy", "thefuck", "tornado", "tqdm", "youtubedl",
]

SHUTTLE_ASSIGNMENT = {
    "se01": ["keras", "youtubedl", "tqdm"],
    "se02": ["matplotlib", "spacy", "scrapy", "thefuck"],
    "se03": ["luigi", "black", "fastapi", "httpie", "sanic"],
    "se04": ["ansible", "tornado", "cookiecutter", "markup", "middle",
             "pysnooper", "calculator", "expression"],
}


def usable_projects(names: Optional[Iterable[str]] = None, bug_id: Optional[int] = None):
    import tests4py.api as t4p
    from tests4py.projects import TestStatus

    selected = list(names) if names else PROJECTS
    out = []
    for name in selected:
        for project in t4p.get_projects(name, bug_id):
            if (
                project.test_status_buggy == TestStatus.FAILING
                and project.test_status_fixed == TestStatus.PASSING
            ):
                out.append(project)
    return out


# ---------------------------------------------------------------------------
# subject
# ---------------------------------------------------------------------------
def run_subject(
    project_name: str,
    bug_id: int,
    keep_events: bool = False,
    force: bool = False,
    with_stability: bool = True,
) -> Dict:
    import tests4py.api as t4p
    from protocol.events import collect_events, events_complete
    from protocol.extract import extract_subject, read_artifact

    projects = [p for p in t4p.get_projects(project_name, bug_id)]
    if not projects:
        return {"subject": f"{project_name}_{bug_id}", "status": "unknown_subject"}
    project = projects[0]
    identifier = project.get_identifier()
    target = ARTIFACTS / f"{identifier}.json.gz"
    if target.exists() and not force:
        try:
            existing = read_artifact(target)
            if existing.get("status") == "ok":
                return {"subject": identifier, "status": "cached"}
        except Exception:
            pass

    LOGS.mkdir(exist_ok=True)
    started = time.time()
    events = ROOT / "sflkit_events" / project.project_name / str(project.bug_id)
    report: Dict[str, object] = {"subject": identifier}
    have_events = events.exists() and events_complete(events)
    if not have_events or force:
        report["events"] = collect_events(project)
        if report["events"].get("status") != "ok":
            report["status"] = report["events"]["status"]
            (LOGS / f"{identifier}.json").write_text(json.dumps(report, indent=1, default=str))
            return report

    checkout = ROOT / "tmp" / identifier
    if not checkout.exists():
        result = t4p.checkout(project)
        if not result.successful:
            report["status"] = "checkout_failed"
            (LOGS / f"{identifier}.json").write_text(json.dumps(report, indent=1, default=str))
            return report

    artifact = extract_subject(project, artifacts_dir=ARTIFACTS, with_stability=with_stability)
    report["status"] = artifact.get("status", "unknown")
    report["error"] = artifact.get("error", "")
    report["seconds"] = time.time() - started
    (LOGS / f"{identifier}.json").write_text(json.dumps(report, indent=1, default=str))

    if not keep_events and artifact.get("status") == "ok":
        shutil.rmtree(events, ignore_errors=True)
        shutil.rmtree(ROOT / "tmp" / f"sfl_{identifier}", ignore_errors=True)
    return report


def _limit_worker_memory():
    """Cap a worker's address space so a runaway subject cannot take the host down.

    Extraction memory scales with trace volume times analysis objects, and a
    single large subject has repeatedly exhausted a 251 GB machine, leaving it
    unable to fork sshd. With a cap the worker raises MemoryError instead, which
    is recorded as a per-subject failure while the host stays responsive.
    """
    gigabytes = float(os.environ.get("FFL_MEM_GB", "0") or 0)
    if gigabytes <= 0:
        return
    import resource

    limit = int(gigabytes * 1024**3)
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        ceiling = limit if hard == resource.RLIM_INFINITY else min(limit, hard)
        resource.setrlimit(resource.RLIMIT_AS, (ceiling, hard))
    except (ValueError, OSError):
        pass


def _subject_worker(args):
    project_name, bug_id, keep_events, force, with_stability = args
    try:
        return run_subject(project_name, bug_id, keep_events, force, with_stability)
    except BaseException as error:  # noqa: BLE001
        return {
            "subject": f"{project_name}_{bug_id}",
            "status": "crashed",
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(limit=5),
        }


def _prepare_worker(job):
    """Checkout and build one subject (no instrumentation, no tests).

    The network-bound part of event collection: run with modest parallelism so
    GitHub and PyPI are not flooded, after which ``run`` finds the checkout with
    its project venv (marked ``.prepared``) and only instruments and executes.
    Transient network failures are retried from a clean checkout.
    """
    import shutil

    import tests4py.api as t4p
    from protocol.events import (
        FALLBACK_CONSTRAINTS, FALLBACK_MARKER, ensure_build_constraints,
        relax_unsupported_pins,
    )

    project_name, bug_id, retries = job
    project = t4p.get_projects(project_name, bug_id)[0]
    identifier = project.get_identifier()
    ensure_build_constraints()
    checkout = ROOT / "tmp" / identifier
    error = ""
    for attempt in range(1, retries + 1):
        if not checkout.exists():
            result = t4p.checkout(project)
            if not result.successful:
                error = f"checkout: {result.raised}"
                shutil.rmtree(checkout, ignore_errors=True)
                time.sleep(10 * attempt)
                continue
        if (checkout / ".prepared").exists():
            return {"subject": identifier, "status": "cached"}
        #: First attempt with the study's constraints; later attempts add the
        #: fallback constraints and record that choice in the checkout.
        fallback = attempt > 1
        if fallback:
            (checkout / FALLBACK_MARKER).write_text("\n".join(FALLBACK_CONSTRAINTS) + "\n")
            #: Pins that the subject's interpreter cannot install at all.
            relax_unsupported_pins(checkout / "tests4py_requirements.txt",
                                   project.python_version)
        ensure_build_constraints(checkout, checkout / "tests4py_requirements.txt", fallback)
        #: Tests4Py caches every installed venv and copies it on later builds
        #: without running pip, so constraints could never reach a venv that was
        #: cached before them. Fallback attempts therefore bypass the cache.
        result = t4p.build(checkout, force=fallback)
        if result.successful:
            (checkout / ".prepared").write_text("checkout and build ok\n")
            return {"subject": identifier, "status": "ok", "attempts": attempt}
        #: A failed build can leave a broken venv inside the checkout; start
        #: the next attempt from a clean checkout.
        error = f"build: {result.raised}"
        #: Keep the checkout but discard its broken venv, so the retry (with the
        #: fallback constraints) rebuilds from scratch.
        for venv in checkout.glob("tests4py_venv*"):
            shutil.rmtree(venv, ignore_errors=True)
        time.sleep(10 * attempt)
    return {"subject": identifier, "status": "failed", "error": str(error)[-300:]}


def command_prepare(arguments) -> None:
    names = arguments.projects.split(",") if arguments.projects else None
    projects = usable_projects(names, arguments.bug_id)
    jobs = [(p.project_name, p.bug_id, arguments.retries) for p in projects]
    print(f"{len(jobs)} subjects, {arguments.workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=arguments.workers) as pool:
        for report in pool.map(_prepare_worker, jobs):
            print(f"{report['subject']:>24}  {report['status']}  {report.get('error', '')}",
                  flush=True)


def command_run(arguments) -> None:
    ARTIFACTS.mkdir(exist_ok=True)
    LOGS.mkdir(exist_ok=True)
    names = arguments.projects.split(",") if arguments.projects else None
    projects = usable_projects(names, arguments.bug_id)
    jobs = [
        (p.project_name, p.bug_id, arguments.keep_events, arguments.force,
         not arguments.no_stability)
        for p in projects
    ]
    print(f"{len(jobs)} subjects, {arguments.workers} workers", flush=True)
    if arguments.workers <= 1:
        for job in jobs:
            report = _subject_worker(job)
            print(f"{report['subject']:>24}  {report['status']}", flush=True)
        return
    # A worker that dies abruptly (a segfault in a native extension, or the OOM
    # killer) breaks the whole pool. Because every subject is idempotent -- an
    # existing artifact is returned as "cached" -- a broken pool is recovered by
    # simply running the job list again with a fresh pool. Passes are bounded so
    # a subject that always kills its worker cannot loop forever; whatever is
    # still missing afterwards is reported by ``status``.
    for attempt in range(1, arguments.max_passes + 1):
        try:
            with ProcessPoolExecutor(max_workers=arguments.workers,
                                     initializer=_limit_worker_memory) as pool:
                futures = {pool.submit(_subject_worker, job): job for job in jobs}
                done = 0
                for future in as_completed(futures):
                    report = future.result()
                    done += 1
                    print(
                        f"[{done}/{len(jobs)}] {report['subject']:>24}  "
                        f"{report['status']} {str(report.get('error',''))[:90]}",
                        flush=True,
                    )
            return
        except BrokenProcessPool as error:
            print(f"!! worker pool broke on pass {attempt}: {error}", flush=True)
            if attempt == arguments.max_passes:
                print("!! giving up; rerun to continue", flush=True)
                raise


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------
def command_status(arguments) -> None:
    from protocol.extract import read_artifact

    expected = {p.get_identifier() for p in usable_projects()}
    have, failed, missing = [], [], []
    for identifier in sorted(expected):
        path = ARTIFACTS / f"{identifier}.json.gz"
        if not path.exists():
            missing.append(identifier)
            continue
        try:
            artifact = read_artifact(path)
        except Exception as error:
            failed.append((identifier, f"CORRUPT: {error}"))
            continue
        if artifact.get("status") == "ok":
            have.append(identifier)
        else:
            failed.append((identifier, artifact.get("error", "")[:120]))
    if getattr(arguments, "repair", False):
        for identifier, _ in failed:
            (ARTIFACTS / f"{identifier}.json.gz").unlink(missing_ok=True)
            (LOGS / f"{identifier}.json").unlink(missing_ok=True)
        print(f"repaired: removed {len(failed)} unusable artifacts")
    print(f"expected {len(expected)}  ok {len(have)}  failed {len(failed)}  missing {len(missing)}")
    for identifier, error in failed:
        print(f"  FAILED  {identifier}: {error}")
    if arguments.verbose:
        for identifier in missing:
            print(f"  MISSING {identifier}")


# ---------------------------------------------------------------------------
# collect
# ---------------------------------------------------------------------------
def _collect_worker(job):
    """One subject's outcome rows, confounders and diagnostics.

    Every sampled outcome is seeded from the subject's own seed, formula, method
    and configuration, so the result does not depend on the order or process in
    which subjects are handled."""
    from protocol.extract import read_artifact
    from protocol.metrics import confounders, subject_rows

    path, methods, config = job
    artifact = read_artifact(path)
    if artifact.get("status") != "ok":
        return None, {"subject": artifact["subject"], "project": artifact["project"],
                      "status": artifact.get("status"), "error": artifact.get("error", "")}, None
    rows = subject_rows(artifact, methods, config)
    meta = {**confounders(artifact), "status": "ok"}
    diagnostics = {"subject": artifact["subject"], **_flatten(artifact["diagnostics"]),
                   **{f"time_{k}": v for k, v in artifact.get("timing", {}).items()}}
    return rows, meta, diagnostics


def command_collect(arguments) -> None:
    import pandas as pd
    from protocol.metrics import Config

    methods = arguments.methods.split(",")
    out_dir = ROOT / "study"
    out_dir.mkdir(exist_ok=True)
    paths = sorted(ARTIFACTS.glob("*.json.gz"))
    rows, meta, diagnostics = [], [], []
    config = Config(
        lam=arguments.lam,
        epsilon=arguments.epsilon,
        aggregation=arguments.aggregation,
        threshold=arguments.threshold,
        repeats=arguments.repeats,
        unmapped=arguments.unmapped,
        product=arguments.product,
    )
    started = time.time()
    jobs = [(path, methods, config) for path in paths]
    if arguments.workers > 1:
        pool = ProcessPoolExecutor(max_workers=arguments.workers)
        results = pool.map(_collect_worker, jobs)
    else:
        pool, results = None, map(_collect_worker, jobs)
    for index, (subject_rows_, subject_meta, subject_diag) in enumerate(results, start=1):
        meta.append(subject_meta)
        if subject_rows_ is not None:
            rows.extend(subject_rows_)
            diagnostics.append(subject_diag)
        if index % 25 == 0:
            print(f"{index}/{len(paths)} ({time.time()-started:.0f}s)", flush=True)
    if pool is not None:
        pool.shutdown()
    #: A subject that never produced an artifact would otherwise be absent from
    #: subjects.csv entirely, so the exclusion count would read zero and the
    #: denominator would silently shrink. Record it as an explicit exclusion.
    seen = {entry.get("subject") for entry in meta}
    for project in usable_projects():
        identifier = project.get_identifier()
        if identifier not in seen:
            meta.append({
                "subject": identifier,
                "project": identifier.rsplit("_", 1)[0],
                "status": "not_completed",
                "error": "did not finish within the compute budget",
            })

    suffix = "" if arguments.tag is None else f"_{arguments.tag}"
    pd.DataFrame(rows).to_csv(out_dir / f"outcomes{suffix}.csv", index=False)
    pd.DataFrame(meta).to_csv(out_dir / f"subjects{suffix}.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(out_dir / f"diagnostics{suffix}.csv", index=False)
    excluded = sum(1 for entry in meta if entry.get("status") != "ok")
    print(f"wrote {len(rows)} outcome rows for {len(diagnostics)} subjects; "
          f"{excluded} excluded")


def _flatten(data: Dict, prefix: str = "") -> Dict:
    flat = {}
    for key, value in data.items():
        if isinstance(value, dict):
            flat.update(_flatten(value, f"{prefix}{key}_"))
        else:
            flat[f"{prefix}{key}"] = value
    return flat


# ---------------------------------------------------------------------------
# analyze / tables
# ---------------------------------------------------------------------------
def _sensitivity_worker(job):
    """Baseline and RF rows of one subject under one grid configuration."""
    from protocol.extract import read_artifact
    from protocol.metrics import subject_rows

    path, config = job
    artifact = read_artifact(path)
    if artifact.get("status") != "ok":
        return []
    return subject_rows(artifact, ["baseline", "rf"], config)


def command_sensitivity(arguments) -> None:
    """Pre-registered sensitivity grids over lambda, epsilon, aggregation, threshold."""
    import pandas as pd
    from protocol.metrics import Config
    from protocol.analysis import PRIMARY_METRICS

    grids = [
        ("$\\lambda$", "lam", [0.15, 0.25, 0.35, 0.45, 0.55]),
        ("$\\varepsilon$", "epsilon", [1e-4, 1e-3, 1e-2]),
        ("aggregation", "aggregation", ["max", "mean", "sum"]),
        ("impact threshold", "threshold", [0.5, 0.6, 0.7, 0.8]),
    ]
    paths = sorted(ARTIFACTS.glob("*.json.gz"))
    settings = [(label, knob, value) for label, knob, values in grids for value in values]
    configs = [Config(**{knob: value, "repeats": arguments.repeats})
               for _, knob, value in settings]
    #: All grid points are submitted at once so that the whole grid shares one
    #: pool; results are regrouped per grid point in submission order.
    jobs = [(path, config) for config in configs for path in paths]
    if arguments.workers > 1:
        with ProcessPoolExecutor(max_workers=arguments.workers) as pool:
            results = list(pool.map(_sensitivity_worker, jobs, chunksize=4))
    else:
        results = [_sensitivity_worker(job) for job in jobs]
    per_config = {
        index: [r for chunk in results[index * len(paths):(index + 1) * len(paths)]
                for r in chunk]
        for index in range(len(configs))
    }
    rows = []
    for index, (label, knob, value) in enumerate(settings):
        frame = pd.DataFrame(per_config[index])
        augmented = frame[frame.method == "rf"]
        baseline = frame[frame.method == "baseline"]
        row = {"knob": label, "value": value}
        # The impact threshold only moves the test-impact target set, so it
        # is reported against the test-impact outcomes; the other knobs move
        # the ranking itself and are reported against the primary outcomes.
        observed = [o for o, _ in PRIMARY_METRICS] + [
            "impact_found@1", "impact_found@5", "impact_found@10",
            "impact_min_rank",
        ]
        for outcome in observed:
            if outcome not in frame.columns:
                continue
            row[outcome] = float(augmented[outcome].mean())
            row[f"baseline_{outcome}"] = float(baseline[outcome].mean())
        rows.append(row)
        print(f"{label} = {value} done", flush=True)
    out = ROOT / "study" / "sensitivity.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(rows, indent=1))
    print(f"wrote {out}")


def command_verify(arguments) -> None:
    """Replicability check: re-extract a seeded sample and compare artifacts."""
    import hashlib
    import random

    from protocol.extract import extract_subject, read_artifact
    from protocol.metrics import subject_rows

    projects = usable_projects()
    available = [
        p for p in projects if (ARTIFACTS / f"{p.get_identifier()}.json.gz").exists()
    ]
    random.Random(42).shuffle(available)
    sample = available[: arguments.n]
    report = []
    scratch = ROOT / "verify"
    scratch.mkdir(exist_ok=True)
    for project in sample:
        identifier = project.get_identifier()
        original = read_artifact(ARTIFACTS / f"{identifier}.json.gz")
        repeated = extract_subject(
            project, artifacts_dir=scratch, vectors_dir=None, with_stability=False
        )
        if repeated.get("status") != "ok":
            report.append({"subject": identifier, "identical": False,
                           "reason": repeated.get("error", "extraction failed")})
            continue
        rows_a = subject_rows(original, ["baseline", "rf"])
        rows_b = subject_rows(repeated, ["baseline", "rf"])
        digest = lambda rows: hashlib.sha256(
            json.dumps(rows, sort_keys=True, default=float).encode()
        ).hexdigest()
        report.append({
            "subject": identifier,
            "identical": digest(rows_a) == digest(rows_b),
            "weights_identical": original["weights"]["rf"]["max"] == repeated["weights"]["rf"]["max"],
        })
        print(report[-1], flush=True)
    out = ROOT / "study" / "verify.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(f"{sum(1 for r in report if r['identical'])}/{len(report)} identical")


#: SFLKit analysis-type ids grouped into the four EFDD feature families.
FEATURE_FAMILIES = {
    "coverage": {0, 1, 2, 3},                 # line, branch, function, loop
    "data flow": {4},                         # definition-use pairs
    "value": {6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    "condition": {5},                         # condition outcomes
    "exception": {16},                        # function error exits
}


def _variant_worker(args):
    """Recompute the exploratory variant weights for one subject."""
    import gzip
    import json as _json

    import numpy as np

    from protocol.extract import (
        aggregate_weights,
        hybrid_importances,
        normalize,
        read_artifact,
        write_artifact,
    )

    identifier, modes, force = args
    artifact_path = ARTIFACTS / f"{identifier}.json.gz"
    vector_path = ROOT / "vectors" / f"{identifier}.npz"
    if not artifact_path.exists() or not vector_path.exists():
        return {"subject": identifier, "status": "missing"}
    artifact = read_artifact(artifact_path)
    if artifact.get("status") != "ok":
        return {"subject": identifier, "status": artifact.get("status", "failed")}
    if not force and not artifact.get("variants_deferred", True):
        return {"subject": identifier, "status": "cached"}
    with np.load(vector_path, allow_pickle=True) as data:
        matrix = data["matrix"]
        labels = data["labels"].astype(int)
        names = [str(x) for x in data["features"]]
        flat = data["map_flat"]
        offsets = data["map_offsets"]
    locations = [
        flat[offsets[i]:offsets[i + 1]].tolist() for i in range(len(offsets) - 1)
    ]
    n_lines = len(artifact["lines"]["line"])
    identity = {i: i for i in range(n_lines)}

    class _Named:
        __slots__ = ("name",)

        def __init__(self, name):
            self.name = name

    features = [_Named(name) for name in names]
    started = time.time()
    for mode in modes:
        try:
            importance = hybrid_importances(matrix, labels, features, mode)
        except BaseException as error:  # noqa: BLE001
            return {"subject": identifier, "status": f"failed: {error}"}
        aggregated = aggregate_weights(locations, normalize(importance), identity, n_lines)
        artifact["weights"][mode] = {k: v.tolist() for k, v in aggregated.items()}
    artifact.setdefault("timing", {})["variants"] = time.time() - started
    artifact["variants_deferred"] = False
    write_artifact(artifact, artifact_path)
    return {"subject": identifier, "status": "ok", "seconds": time.time() - started}


def command_variants(arguments) -> None:
    """Recompute the exploratory ranking variants for every subject at once."""
    modes = ("hybrid", "failfirst")
    identifiers = sorted(
        path.name[: -len(".json.gz")] for path in ARTIFACTS.glob("*.json.gz")
    )
    jobs = [(identifier, modes, arguments.force) for identifier in identifiers]
    print(f"{len(jobs)} subjects, {arguments.workers} workers", flush=True)
    done = 0
    if arguments.workers <= 1:
        for job in jobs:
            report = _variant_worker(job)
            done += 1
            print(f"[{done}/{len(jobs)}] {report['subject']} {report['status']}", flush=True)
        return
    with ProcessPoolExecutor(max_workers=arguments.workers) as pool:
        for report in pool.map(_variant_worker, jobs):
            done += 1
            print(f"[{done}/{len(jobs)}] {report['subject']} {report['status']}", flush=True)


def command_feasibility(arguments) -> None:
    """Aggregate per-subject execution logs into feasibility statistics."""
    import pandas as pd

    rows = []
    for path in sorted(LOGS.glob("*.json")):
        try:
            log = json.loads(path.read_text())
        except ValueError:
            continue
        events = log.get("events", {}) or {}
        times = events.get("time", {}) or {}
        rows.append({
            "subject": log.get("subject", path.stem),
            "status": log.get("status", "unknown"),
            "error": str(log.get("error", ""))[:200],
            "event_status": events.get("status", "cached"),
            "passing_tests": events.get("passing"),
            "failing_tests": events.get("failing"),
            "undefined_tests": events.get("undefined"),
            "time_checkout": times.get("checkout"),
            "time_build": times.get("build"),
            "time_instrument": times.get("instrument"),
            "time_tests": times.get("tests"),
            "seconds_total": log.get("seconds"),
        })
    frame = pd.DataFrame(rows)
    if frame.empty:
        #: The replication package ships study/feasibility.csv but not the logs;
        #: never overwrite the shipped statistics with an empty table.
        print("no logs found; keeping study/feasibility.csv")
        return
    out = ROOT / "study" / "feasibility.csv"
    out.parent.mkdir(exist_ok=True)
    frame.to_csv(out, index=False)
    #: The logs are a per-run diagnostic tier and are not guaranteed complete: a
    #: subject re-run after a failure keeps the failing log, and logs pulled from
    #: the shuttles can be overwritten. Reconcile against the artifacts so a
    #: missing log never reads as a missing subject.
    n_artifacts = len(list(ARTIFACTS.glob("*.json.gz")))
    print(f"{len(frame)} of {n_artifacts} subject runs have a surviving log")
    if len(frame) < n_artifacts:
        print(f"  {n_artifacts - len(frame)} artifacts have no log; "
              f"statuses below describe the logged runs only")
    print(frame["status"].value_counts().to_string())
    numeric = frame.select_dtypes("number")
    if not numeric.empty:
        print(numeric.median(numeric_only=True).round(2).to_string())
    undefined = pd.to_numeric(frame["undefined_tests"], errors="coerce").fillna(0)
    print(f"executions classified undefined (incl. 120 s timeouts): {int(undefined.sum())}"
          f" across {int((undefined > 0).sum())} subjects")


def command_checks(arguments) -> None:
    """Pre-registered data-quality checks over every produced artifact."""
    import numpy as np
    import pandas as pd

    from protocol.extract import read_artifact
    from protocol.formulas import FORMULAS, suspiciousness
    from protocol.metrics import EPSILON, test_impact_indices

    rows = []
    for path in sorted(ARTIFACTS.glob("*.json.gz")):
        artifact = read_artifact(path)
        identifier = artifact["subject"]
        if artifact.get("status") != "ok":
            rows.append({"subject": identifier, "status": artifact.get("status"),
                         "evaluable": False})
            continue
        lines = artifact["lines"]
        n_lines = len(lines["line"])
        weights = np.asarray(artifact["weights"]["rf"]["max"], dtype=float)
        encoding_ok = None
        line_features = aligned = None
        vector_path = ROOT / "vectors" / f"{identifier}.npz"
        if vector_path.exists():
            with np.load(vector_path, allow_pickle=True) as data:
                values = np.unique(data["matrix"])
                labels = np.unique(data["labels"])
                names, types = data["features"], data["types"]
                flat, offsets = data["map_flat"], data["map_offsets"]
            encoding_ok = bool(set(values.tolist()) <= {-1, 0, 1}
                               and set(labels.tolist()) <= {0, 1})
            #: The stored feature-to-line map holds row numbers into this
            #: artifact's line table. A line feature names its own line, so it
            #: must map to exactly that row; anything else means the map and the
            #: table disagree and every weight would land on the wrong line.
            rows_of = list(zip(lines["file"], (int(l) for l in lines["line"])))
            line_features = aligned = 0
            for i in np.flatnonzero(types == 0):
                _, file, line = str(names[i]).rsplit(":", 2)
                mapped = [rows_of[r] for r in flat[offsets[i]:offsets[i + 1]]]
                line_features += 1
                aligned += mapped == [(file, int(line))]
        ef = np.asarray(lines["ef"], dtype=float)
        ep = np.asarray(lines["ep"], dtype=float)
        impact = test_impact_indices(ef, ep)
        applied = EPSILON + (1.0 - EPSILON) * weights
        rows.append({
            "subject": identifier,
            "status": "ok",
            "n_lines": n_lines,
            "weights_complete": bool(len(weights) == n_lines),
            "weights_in_range": bool(applied.min() >= EPSILON - 1e-12
                                     and applied.max() <= 1.0 + 1e-12),
            "encoding_binary_or_tertiary": encoding_ok,
            "line_features": line_features,
            "line_features_aligned": aligned,
            "has_reference_lines": bool(len(artifact["reference_lines"]) > 0),
            "reference_lines_ranked": sum(
                1 for f, l in artifact["reference_lines"]
                if (f, int(l)) in {(a, int(b)) for a, b in zip(lines["file"], lines["line"])}
            ),
            "has_hunk_lines": bool(len(artifact["hunk_lines"]) > 0),
            "impact_set_size": int(len(impact)),
            "has_functions": bool(len(artifact["functions"]) > 0),
            "failing_vectors": artifact["diagnostics"]["n_vectors_failing"],
            "passing_vectors": artifact["diagnostics"]["n_vectors_passing"],
            "variants_deferred": bool(artifact.get("variants_deferred", False)),
        })
        #: The registered blend multiplies the weight into the formula's own
        #: suspiciousness, so a reference line the formula scores at zero cannot
        #: be promoted by any weight. Counting those lines measures the ceiling
        #: on the whole family of methods this protocol can express.
        index = {(f, int(l)): i for i, (f, l) in enumerate(zip(lines["file"], lines["line"]))}
        ranked = [index[(f, int(l))] for f, l in artifact["reference_lines"]
                  if (f, int(l)) in index]
        for formula in FORMULAS:
            key = f"zero_susp_{formula.lower()}"
            if not ranked:
                rows[-1][key] = None
                continue
            scores = suspiciousness(
                formula, ef, ep, artifact["totals"]["failed"], artifact["totals"]["passed"]
            )
            rows[-1][key] = sum(1 for i in ranked if scores[i] <= 0) / len(ranked)
        rows[-1]["n_reference_ranked"] = len(ranked)
        rows[-1]["evaluable"] = bool(
            rows[-1]["weights_complete"] and rows[-1]["weights_in_range"]
            and rows[-1]["has_reference_lines"] and rows[-1]["impact_set_size"] > 0
            and rows[-1]["failing_vectors"] > 0 and rows[-1]["passing_vectors"] > 0
        )
    frame = pd.DataFrame(rows)
    out = ROOT / "study" / "checks.csv"
    out.parent.mkdir(exist_ok=True)
    frame.to_csv(out, index=False)
    ok = frame[frame.status == "ok"] if "status" in frame else frame
    print(f"{len(frame)} artifacts, {int(frame['evaluable'].sum())} evaluable")
    for column in ("weights_complete", "weights_in_range", "encoding_binary_or_tertiary",
                   "has_reference_lines", "has_hunk_lines", "has_functions"):
        if column in ok:
            values = ok[column].dropna()
            print(f"  {column}: {int(values.sum())}/{len(values)}")
    if "impact_set_size" in ok:
        print(f"  empty test-impact sets: {int((ok['impact_set_size'] == 0).sum())}")
    if "reference_lines_ranked" in ok:
        print(f"  subjects with no executed reference line: "
              f"{int((ok['reference_lines_ranked'] == 0).sum())}")


def _permutation_worker(identifier):
    """Post-hoc control: keep the importance *distribution*, shuffle which
    feature carries which importance.

    The pre-registered negative control replaces the learned importances with
    uniform random scores. Because a line's weight is the maximum over the
    features mapped to it, and the maximum of k uniforms grows with k, that
    control also encodes how many features map to a line. Permuting the learned
    importances across features preserves both the distribution and the
    multiplicity, and therefore isolates whether *which* feature carries the
    importance matters at all.
    """
    import numpy as np

    from protocol.extract import (
        aggregate_weights, normalize, read_artifact, write_artifact,
    )
    from protocol.metrics import stable_seed

    artifact_path = ARTIFACTS / f"{identifier}.json.gz"
    vector_path = ROOT / "vectors" / f"{identifier}.npz"
    if not artifact_path.exists() or not vector_path.exists():
        return {"subject": identifier, "status": "missing"}
    artifact = read_artifact(artifact_path)
    if artifact.get("status") != "ok":
        return {"subject": identifier, "status": artifact.get("status", "failed")}
    with np.load(vector_path, allow_pickle=True) as data:
        importance = data["importance_rf"]
        flat, offsets = data["map_flat"], data["map_offsets"]
    locations = [
        flat[offsets[i]:offsets[i + 1]].tolist() for i in range(len(offsets) - 1)
    ]
    rng = np.random.default_rng(stable_seed(artifact.get("seed", 42), "permutation"))
    shuffled = rng.permutation(importance)
    n_lines = len(artifact["lines"]["line"])
    aggregated = aggregate_weights(
        locations, normalize(shuffled), {i: i for i in range(n_lines)}, n_lines
    )
    artifact["weights"]["perm"] = {k: v.tolist() for k, v in aggregated.items()}
    write_artifact(artifact, artifact_path)
    return {"subject": identifier, "status": "ok"}


def command_permcontrol(arguments) -> None:
    identifiers = sorted(
        path.name[: -len(".json.gz")] for path in ARTIFACTS.glob("*.json.gz")
    )
    done = 0
    with ProcessPoolExecutor(max_workers=arguments.workers) as pool:
        for report in pool.map(_permutation_worker, identifiers):
            done += 1
            if report["status"] != "ok":
                print(f"[{done}/{len(identifiers)}] {report['subject']} {report['status']}",
                      flush=True)
    print(f"permutation control written for {done} subjects")


def _family_worker(identifier):
    """Ablate the feature vocabulary: keep one family's importances, zero the rest.

    Execution-feature debugging claims that families beyond coverage carry the
    diagnostic signal. Restricting the weight to a single family and rerunning
    the ranking tests that claim directly at the localization level.
    """
    import numpy as np

    from protocol.extract import (
        aggregate_weights, normalize, read_artifact, write_artifact,
    )

    artifact_path = ARTIFACTS / f"{identifier}.json.gz"
    vector_path = ROOT / "vectors" / f"{identifier}.npz"
    if not artifact_path.exists() or not vector_path.exists():
        return {"subject": identifier, "status": "missing"}
    artifact = read_artifact(artifact_path)
    if artifact.get("status") != "ok":
        return {"subject": identifier, "status": artifact.get("status", "failed")}
    with np.load(vector_path, allow_pickle=True) as data:
        importance = data["importance_rf"]
        types = data["types"]
        flat, offsets = data["map_flat"], data["map_offsets"]
    locations = [
        flat[offsets[i]:offsets[i + 1]].tolist() for i in range(len(offsets) - 1)
    ]
    n_lines = len(artifact["lines"]["line"])
    identity = {i: i for i in range(n_lines)}
    for family, ids in FEATURE_FAMILIES.items():
        restricted = np.where(np.isin(types, list(ids)), importance, 0.0)
        aggregated = aggregate_weights(
            locations, normalize(restricted), identity, n_lines
        )
        key = "fam_" + family.replace(" ", "_")
        artifact["weights"][key] = {k: v.tolist() for k, v in aggregated.items()}
    write_artifact(artifact, artifact_path)
    return {"subject": identifier, "status": "ok"}


def command_family_ablation(arguments) -> None:
    identifiers = sorted(
        path.name[: -len(".json.gz")] for path in ARTIFACTS.glob("*.json.gz")
    )
    done = bad = 0
    with ProcessPoolExecutor(max_workers=arguments.workers) as pool:
        for report in pool.map(_family_worker, identifiers):
            done += 1
            bad += report["status"] != "ok"
    print(f"family ablation written for {done - bad}/{done} subjects")


def command_categories(arguments) -> None:
    """Per-subject importance mass carried by each execution-feature family."""
    import numpy as np
    import pandas as pd

    rows = []
    for path in sorted((ROOT / "vectors").glob("*.npz")):
        with np.load(path, allow_pickle=True) as data:
            types = data["types"]
            importance = data["importance_rf"]
            if importance.size != types.size:
                continue
            total = float(importance.sum())
            row = {"subject": path.stem, "n_features": int(types.size)}
            for family, ids in FEATURE_FAMILIES.items():
                mask = np.isin(types, list(ids))
                row[f"share_{family.replace(' ', '_')}"] = (
                    float(importance[mask].sum() / total) if total > 0 else 0.0
                )
                row[f"count_{family.replace(' ', '_')}"] = int(mask.sum())
            # Separate "the mapping never reaches this line" from "the forest
            # gave this line's features zero importance": the first is a
            # property of the mapping rules, the second of the model.
            flat = data["map_flat"]
            offsets = data["map_offsets"]
            mapped = np.unique(flat) if flat.size else np.zeros(0, dtype=int)
            with_importance = set()
            for index in range(len(offsets) - 1):
                if importance[index] > 0:
                    with_importance.update(
                        flat[offsets[index]:offsets[index + 1]].tolist()
                    )
            artifact_path = ARTIFACTS / f"{path.stem}.json.gz"
            n_lines = 0
            if artifact_path.exists():
                from protocol.extract import read_artifact

                n_lines = len(read_artifact(artifact_path)["lines"]["line"])
            row["n_lines"] = n_lines
            row["lines_mapped"] = int(mapped.size)
            row["lines_mapped_rate"] = float(mapped.size / n_lines) if n_lines else 0.0
            row["lines_with_importance"] = int(len(with_importance))
            # How strongly does a line's weight track the number of features
            # mapped to it? See the discussion of the negative control.
            from scipy.stats import spearmanr

            from protocol.extract import read_artifact as _read

            counts = np.bincount(flat, minlength=n_lines).astype(float) if n_lines else None
            if counts is not None and counts.std() > 0:
                artifact = _read(artifact_path)
                for condition in ("rf", "neg", "perm"):
                    weights = artifact["weights"].get(condition)
                    if weights is None:
                        continue
                    vector = np.asarray(weights["max"], dtype=float)
                    if vector.std() > 0:
                        row[f"density_rho_{condition}"] = float(
                            spearmanr(counts, vector).statistic
                        )
            row["lines_with_importance_rate"] = (
                float(len(with_importance) / n_lines) if n_lines else 0.0
            )
            rows.append(row)
    out = ROOT / "study" / "categories.csv"
    out.parent.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out} ({len(rows)} subjects)")


def command_signprobe(arguments) -> None:
    """Record how the registered blend behaves on formulas with a negative range.

    Naish2 is the only formula in the registered set whose suspiciousness can go
    below zero. The blend multiplies the weight into that score, and multiplying
    a negative value by a small weight moves it UP, so the weighting is inverted
    on exactly those lines. This records the size of that population and measures
    the inversion directly, so the manuscript can cite recorded numbers instead
    of asserted ones.
    """
    import numpy as np
    from scipy.stats import spearmanr

    from protocol.extract import read_artifact
    from protocol.formulas import FORMULAS, suspiciousness
    from protocol.metrics import LAMBDA, EPSILON, blended_scores

    negatives = {f: {"neg": 0, "total": 0, "subjects": 0} for f in FORMULAS}
    rho_pos, rho_neg = [], []
    for path in sorted(ARTIFACTS.glob("*.json.gz")):
        artifact = read_artifact(path)
        if artifact.get("status") != "ok":
            continue
        lines = artifact["lines"]
        ef = np.asarray(lines["ef"], dtype=float)
        ep = np.asarray(lines["ep"], dtype=float)
        tf, tp = artifact["totals"]["failed"], artifact["totals"]["passed"]
        for formula in FORMULAS:
            scores = suspiciousness(formula, ef, ep, tf, tp)
            count = int((scores < 0).sum())
            negatives[formula]["neg"] += count
            negatives[formula]["total"] += scores.size
            negatives[formula]["subjects"] += count > 0

        weights = np.asarray(
            (artifact["weights"].get("rf") or {}).get("max") or [], dtype=float
        )
        base = suspiciousness("Naish2", ef, ep, tf, tp)
        if weights.size != base.size or not weights.any():
            continue
        blended = blended_scores(base, weights, LAMBDA, EPSILON)
        #: Hold the baseline score fixed and ask whether more weight raises the
        #: blended score. Comparing within one score keeps the formula's own
        #: ordering out of the correlation.
        for mask, bucket in ((base > 0, rho_pos), (base < 0, rho_neg)):
            if mask.sum() < 20:
                continue
            values, counts = np.unique(base[mask], return_counts=True)
            target = values[counts.argmax()]
            selected = mask & (base == target)
            if selected.sum() < 20 or len(np.unique(weights[selected])) < 3:
                continue
            rho = spearmanr(weights[selected], blended[selected]).statistic
            if np.isfinite(rho):
                bucket.append(float(rho))

    report = {
        "negative_scores": {
            f: {
                "share": v["neg"] / v["total"] if v["total"] else 0.0,
                "subjects": v["subjects"],
            }
            for f, v in negatives.items()
        },
        "inversion": {
            "rho_positive_median": float(np.median(rho_pos)) if rho_pos else None,
            "rho_negative_median": float(np.median(rho_neg)) if rho_neg else None,
            "n_subjects_positive": len(rho_pos),
            "n_subjects_negative": len(rho_neg),
        },
    }
    target = ROOT / "study" / "sign_probe.json"
    target.write_text(json.dumps(report, indent=1))
    print(f"wrote {target}")


def _posthoc_worker(identifier):
    """Post-hoc feature scorers for one subject (see ``protocol.posthoc``).

    Reads only the stored feature matrix, labels and feature-to-line map, and
    adds one weighting per scorer to the artifact, built exactly like ``rf``.
    Must run after ``reweight``, which rebuilds the weights from scratch.
    """
    import os

    os.environ["FFL_RF_JOBS"] = "1"  # one process per subject; no nested threads
    import numpy as np

    from protocol.extract import aggregate_weights, normalize, read_artifact, write_artifact
    from protocol.posthoc import SCORERS, TIEBREAK

    artifact_path = ARTIFACTS / f"{identifier}.json.gz"
    vector_path = ROOT / "vectors" / f"{identifier}.npz"
    if not artifact_path.exists() or not vector_path.exists():
        return {"subject": identifier, "status": "missing"}
    artifact = read_artifact(artifact_path)
    if artifact.get("status") != "ok":
        return {"subject": identifier, "status": artifact.get("status", "failed")}
    with np.load(vector_path, allow_pickle=True) as data:
        matrix, labels = data["matrix"], data["labels"].astype(int)
        flat, offsets = data["map_flat"], data["map_offsets"]
    primary = [flat[offsets[i]:offsets[i + 1]].tolist() for i in range(len(offsets) - 1)]
    n_lines = len(artifact["lines"]["line"])
    identity = {i: i for i in range(n_lines)}
    seed = int(artifact.get("seed", 42))

    weights = artifact["weights"]
    failed = []
    for name, scorer in SCORERS.items():
        try:
            scores = scorer(matrix, labels, seed)
        except Exception as error:  # noqa: BLE001
            failed.append(f"{name}: {error}")
            scores = np.zeros(matrix.shape[1])
        weights[name] = {k: v.tolist() for k, v in
                         aggregate_weights(primary, normalize(scores), identity, n_lines).items()}
    for name, source in TIEBREAK.items():
        if source in weights:
            weights[name] = weights[source]

    #: Granularity-restricted weights for the exploratory hierarchical descent:
    #: each level is scored only by the feature types that describe that level.
    from protocol.hierarchical import GRANULARITY, GRANULARITY_SOURCES

    with np.load(vector_path, allow_pickle=True) as data:
        types = data["types"]
        neg = data["importance_neg"] if "importance_neg" in data.files else None
    feature_scores = {}
    for source in GRANULARITY_SOURCES:
        if source == "neg":
            if neg is not None and len(neg) == matrix.shape[1]:
                feature_scores[source] = np.asarray(neg, dtype=float)
        else:
            feature_scores[source] = SCORERS[source](matrix, labels, seed)
    for source, scores in feature_scores.items():
        for level, ids in GRANULARITY.items():
            restricted = np.where(np.isin(types, list(ids)), scores, 0.0)
            weights[f"g_{level}_{source}"] = {
                k: v.tolist() for k, v in
                aggregate_weights(primary, normalize(restricted), identity, n_lines).items()
            }
    write_artifact(artifact, artifact_path)
    return {"subject": identifier, "status": "ok", "failed": failed}


def command_posthoc(arguments) -> None:
    identifiers = sorted(path.name[: -len(".json.gz")] for path in ARTIFACTS.glob("*.json.gz"))
    done = 0
    problems = []
    with ProcessPoolExecutor(max_workers=arguments.workers) as pool:
        for report in pool.map(_posthoc_worker, identifiers):
            done += 1
            if report.get("failed"):
                problems.append(report)
            if done % 25 == 0:
                print(f"{done}/{len(identifiers)}", flush=True)
    for report in problems:
        print(f"  {report['subject']}: {'; '.join(report['failed'])}")
    print(f"post-hoc weights for {done} subjects; {len(problems)} with scorer failures")


def _reweight_worker(args):
    """Rebuild every weighting condition of one artifact from its feature vectors.

    The artifact stores per-line weights, which are a lossy aggregate of the
    per-feature importances in the npz. This recomputes them from the npz alone,
    mirroring :func:`protocol.extract.build_artifact` exactly. Two uses: it
    guarantees the artifact agrees with the vectors, and because the only inputs
    are ``importance_*``, ``types`` and the stored feature-to-line map, ground
    truth cannot reach any condition except ``pos``, whose oracle marking is
    part of its definition.
    """
    import numpy as np

    from protocol.extract import (
        AGGREGATIONS, aggregate_weights, normalize, read_artifact, write_artifact,
    )
    from protocol.metrics import stable_seed

    identifier, dry_run = args
    artifact_path = ARTIFACTS / f"{identifier}.json.gz"
    vector_path = ROOT / "vectors" / f"{identifier}.npz"
    if not artifact_path.exists() or not vector_path.exists():
        return {"subject": identifier, "status": "missing"}
    artifact = read_artifact(artifact_path)
    if artifact.get("status") != "ok":
        return {"subject": identifier, "status": artifact.get("status", "failed")}

    with np.load(vector_path, allow_pickle=True) as data:
        imp = {k[len("importance_"):]: data[k] for k in data.files
               if k.startswith("importance_")}
        types = data["types"]
        flat, offsets = data["map_flat"], data["map_offsets"]
        alt_flat, alt_offsets = data["alt_flat"], data["alt_offsets"]

    def unpack(f, o):
        return [f[o[i]:o[i + 1]].tolist() for i in range(len(o) - 1)]

    primary, alternative = unpack(flat, offsets), unpack(alt_flat, alt_offsets)
    n_lines = len(artifact["lines"]["line"])
    identity = {i: i for i in range(n_lines)}

    def agg(locations, values):
        return aggregate_weights(locations, normalize(values), identity, n_lines)

    weights = {}
    for condition in ("rf", "neg", "adaptive", "hybrid", "failfirst"):
        if condition in imp:
            weights[condition] = {k: v.tolist() for k, v in agg(primary, imp[condition]).items()}
    if "rf" in imp:
        weights["alt"] = {k: v.tolist() for k, v in agg(alternative, imp["rf"]).items()}

    #: The positive control is the one condition that legitimately reads ground
    #: truth: it is the primary weighting with the reference lines raised to the
    #: largest weight the pipeline can assign, which makes it dominate ``rf`` per
    #: subject. It is derived from ``importance_rf`` rather than from a separate
    #: oracle-augmented forest, so old vectors carrying ``importance_pos`` and new
    #: ones without it both work.
    if "rf" in imp:
        positive = aggregate_weights(primary, normalize(imp["rf"]), identity, n_lines)
        index = {(f, int(l)): i for i, (f, l)
                 in enumerate(zip(artifact["lines"]["file"], artifact["lines"]["line"]))}
        for aggregation in AGGREGATIONS:
            vector = positive[aggregation]
            for f, l in artifact["reference_lines"]:
                i = index.get((f, int(l)))
                if i is not None:
                    vector[i] = 1.0
        weights["pos"] = {k: v.tolist() for k, v in positive.items()}

    if "rf" in imp:
        rng = np.random.default_rng(stable_seed(artifact.get("seed", 42), "permutation"))
        weights["perm"] = {k: v.tolist()
                           for k, v in agg(primary, rng.permutation(imp["rf"])).items()}
        for family, ids in FEATURE_FAMILIES.items():
            restricted = np.where(np.isin(types, list(ids)), imp["rf"], 0.0)
            key = "fam_" + family.replace(" ", "_")
            weights[key] = {k: v.tolist() for k, v in agg(primary, restricted).items()}

    changed = sum(
        1 for c in weights
        if c not in (artifact.get("weights") or {})
        or not np.allclose(np.asarray(artifact["weights"][c]["max"], float),
                           np.asarray(weights[c]["max"], float), atol=1e-9)
    )
    if not dry_run:
        artifact["weights"] = weights
        artifact["variants_deferred"] = False
        write_artifact(artifact, artifact_path)
    return {"subject": identifier, "status": "ok",
            "conditions": len(weights), "changed": changed}


def command_fsm_traces(arguments) -> None:
    """Decode every ordered trace into its automaton transitions (cached)."""
    from protocol.fsm import decode_trace, trace_jobs

    jobs = trace_jobs(ROOT / "sflkit_events", ROOT / "mappings", ROOT / "fsm_cache")
    print(f"{len(jobs)} traces, {arguments.workers} workers", flush=True)
    started, failed = time.time(), 0
    with ProcessPoolExecutor(max_workers=arguments.workers) as pool:
        for index, (path, status) in enumerate(pool.map(decode_trace, jobs, chunksize=1), 1):
            if status.startswith("failed"):
                failed += 1
                print(f"  {path}: {status}", flush=True)
            if index % 2000 == 0:
                print(f"{index}/{len(jobs)} ({time.time() - started:.0f}s)", flush=True)
    print(f"done: {len(jobs) - failed}/{len(jobs)} decoded ({time.time() - started:.0f}s)")


def _fsm_worker(identifier):
    """Add the automaton weightings to one subject's artifact."""
    from protocol.extract import read_artifact, write_artifact
    from protocol.fsm import subject_weights

    artifact_path = ARTIFACTS / f"{identifier}.json.gz"
    cache = ROOT / "fsm_cache" / identifier
    if not artifact_path.exists() or not cache.exists():
        return {"subject": identifier, "status": "missing"}
    artifact = read_artifact(artifact_path)
    if artifact.get("status") != "ok":
        return {"subject": identifier, "status": artifact.get("status", "failed")}
    weights, stats = subject_weights(cache, artifact, int(artifact.get("seed", 42)))
    if not weights:
        return {"subject": identifier, "status": "no_runs"}
    artifact["weights"].update(weights)
    artifact["fsm_stats"] = stats
    write_artifact(artifact, artifact_path)
    return {"subject": identifier, "status": "ok", **stats}


def command_fsm(arguments) -> None:
    """Exploratory automaton weightings (requires ``fsm-traces``)."""
    import pandas as pd

    identifiers = sorted(p.name[: -len(".json.gz")] for p in ARTIFACTS.glob("*.json.gz"))
    print(f"{len(identifiers)} subjects, {arguments.workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=arguments.workers) as pool:
        reports = list(pool.map(_fsm_worker, identifiers, chunksize=1))
    frame = pd.DataFrame(reports)
    (ROOT / "study").mkdir(exist_ok=True)
    frame.to_csv(ROOT / "study" / "fsm_subjects.csv", index=False)
    print(frame["status"].value_counts().to_string())


def command_slice_traces(arguments) -> None:
    """Slice every failing trace and rank its lines by failure proximity (cached)."""
    from protocol.slicing import analyze_trace, trace_jobs

    jobs = trace_jobs(ROOT / "sflkit_events", ROOT / "mappings", ROOT / "tmp",
                      ROOT / "slice_cache")
    print(f"{len(jobs)} failing traces, {arguments.workers} workers", flush=True)
    started, failed = time.time(), 0
    with ProcessPoolExecutor(max_workers=arguments.workers) as pool:
        for index, (path, status) in enumerate(pool.map(analyze_trace, jobs, chunksize=1), 1):
            if status.startswith("failed"):
                failed += 1
                print(f"  {path}: {status}", flush=True)
            if index % 200 == 0:
                print(f"{index}/{len(jobs)} ({time.time() - started:.0f}s)", flush=True)
    print(f"done: {len(jobs) - failed}/{len(jobs)} analysed ({time.time() - started:.0f}s)")


def _slice_worker(identifier):
    """Add the slicing and proximity weightings to one subject's artifact."""
    from protocol.extract import read_artifact, write_artifact
    from protocol.slicing import subject_weights

    artifact_path = ARTIFACTS / f"{identifier}.json.gz"
    cache = ROOT / "slice_cache" / identifier
    if not artifact_path.exists() or not cache.exists():
        return {"subject": identifier, "status": "missing"}
    artifact = read_artifact(artifact_path)
    if artifact.get("status") != "ok":
        return {"subject": identifier, "status": artifact.get("status", "failed")}
    weights, stats = subject_weights(cache, artifact)
    if not weights:
        return {"subject": identifier, "status": "no_runs"}
    artifact["weights"].update(weights)
    artifact["slice_stats"] = stats
    write_artifact(artifact, artifact_path)
    return {"subject": identifier, "status": "ok", **stats}


def command_slice(arguments) -> None:
    """Exploratory slicing/proximity weightings (requires ``slice-traces``)."""
    import pandas as pd

    identifiers = sorted(p.name[: -len(".json.gz")] for p in ARTIFACTS.glob("*.json.gz"))
    print(f"{len(identifiers)} subjects, {arguments.workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=arguments.workers) as pool:
        reports = list(pool.map(_slice_worker, identifiers, chunksize=1))
    frame = pd.DataFrame(reports)
    (ROOT / "study").mkdir(exist_ok=True)
    frame.to_csv(ROOT / "study" / "slice_subjects.csv", index=False)
    print(frame["status"].value_counts().to_string())


TRACE_WEIGHTS = ("fsm", "g_func_fsm", "g_block_fsm", "g_line_fsm", "tie_fsm",
                 "slice", "tie_slice", "recency", "tie_recency")


def _transplant_worker(job):
    """Copy trace-derived weightings onto one artifact, matched by (file, line).

    The ordered traces were collected again with the newer SFLKit/Tests4Py
    stack. Its artifacts rank (almost) the same lines but carry Tests4Py's newer
    ground truth, so trace-derived signals are evaluated on the study's own
    artifacts instead: lines without trace evidence get weight 0.
    """
    from protocol.extract import read_artifact, write_artifact

    identifier, source_dir = job
    target_path = ARTIFACTS / f"{identifier}.json.gz"
    source_path = Path(source_dir) / f"{identifier}.json.gz"
    if not target_path.exists() or not source_path.exists():
        return {"subject": identifier, "status": "missing"}
    target, source = read_artifact(target_path), read_artifact(source_path)
    if "lines" not in target or "lines" not in source:
        return {"subject": identifier, "status": "no_lines"}
    index = {(f, int(l)): i for i, (f, l) in
             enumerate(zip(source["lines"]["file"], source["lines"]["line"]))}
    mapped = [index.get((f, int(l))) for f, l in
              zip(target["lines"]["file"], target["lines"]["line"])]
    names = [n for n in source["weights"] if n.startswith(TRACE_WEIGHTS)]
    for name in names:
        weights = {}
        for aggregation, values in source["weights"][name].items():
            weights[aggregation] = [0.0 if i is None else float(values[i]) for i in mapped]
        target["weights"][name] = weights
    write_artifact(target, target_path)
    covered = sum(i is not None for i in mapped) / max(len(mapped), 1)
    return {"subject": identifier, "status": "ok", "weights": len(names), "line_coverage": covered}


def command_transplant(arguments) -> None:
    """Add trace-derived weightings from another artifact folder (exploratory)."""
    import pandas as pd

    identifiers = sorted(p.name[: -len(".json.gz")] for p in ARTIFACTS.glob("*.json.gz"))
    jobs = [(i, arguments.source) for i in identifiers]
    with ProcessPoolExecutor(max_workers=arguments.workers) as pool:
        frame = pd.DataFrame(pool.map(_transplant_worker, jobs, chunksize=1))
    (ROOT / "study").mkdir(exist_ok=True)
    frame.to_csv(ROOT / "study" / "transplant.csv", index=False)
    print(frame["status"].value_counts().to_string())
    if "line_coverage" in frame:
        print(f"median line coverage {frame['line_coverage'].median():.3f}, "
              f"min {frame['line_coverage'].min():.3f}")


def _ltr_table_worker(identifier):
    from protocol.extract import read_artifact
    from protocol.ltr import line_table

    artifact = read_artifact(ARTIFACTS / f"{identifier}.json.gz")
    if artifact.get("status") != "ok" or "lines" not in artifact:
        return None
    return line_table(artifact)


def _ltr_write_worker(job):
    import numpy as np

    from protocol.extract import read_artifact, write_artifact

    identifier, scores = job
    path = ARTIFACTS / f"{identifier}.json.gz"
    artifact = read_artifact(path)
    for name, values in scores.items():
        values = np.asarray(values, dtype=float)
        span = values.max() - values.min() if len(values) else 0.0
        values = (values - values.min()) / span if span > 0 else np.zeros_like(values)
        artifact["weights"][name] = {k: values.tolist() for k in ("max", "mean", "sum")}
    write_artifact(artifact, path)
    return identifier


def _ltr_fit_worker(job):
    import os
    import pickle

    os.environ.setdefault("OMP_NUM_THREADS", "8")
    from protocol.ltr import leave_one_project_out

    cache, columns, seed = job
    with open(cache, "rb") as handle:
        table = pickle.load(handle)
    return leave_one_project_out(table, columns, seed=seed)


def command_ltr(arguments) -> None:
    """Cross-subject learning to rank, leave one project out (exploratory)."""
    import numpy as np
    import pandas as pd

    from protocol.ltr import SEED, VARIANTS, feature_columns, leave_one_project_out

    identifiers = sorted(p.name[: -len(".json.gz")] for p in ARTIFACTS.glob("*.json.gz"))
    started = time.time()
    with ProcessPoolExecutor(max_workers=arguments.workers) as pool:
        tables = [t for t in pool.map(_ltr_table_worker, identifiers, chunksize=1) if t is not None]
    groups = tables[0].attrs["groups"]
    table = pd.concat(tables, ignore_index=True)
    print(f"{len(tables)} subjects, {len(table)} lines, {int(table.label.sum())} fault lines, "
          f"{table.project.nunique()} projects ({time.time() - started:.0f}s)", flush=True)
    variants = arguments.variants.split(",") if arguments.variants else list(VARIANTS)
    #: A non-default seed is stored under its own name (``ltr_<variant>_s<seed>``).
    suffix = "" if arguments.seed is None else f"_s{arguments.seed}"
    if arguments.ensemble:
        #: Seed ensemble: the mean out-of-project probability of the pre-fixed
        #: seed and seeds 1-5, so that no single seed decides the ranking.
        import pickle
        import tempfile

        suffix = "_ens"
        seeds = [SEED, 1, 2, 3, 4, 5]
        with tempfile.NamedTemporaryFile(suffix=".pkl", dir=ROOT / "study", delete=False) as handle:
            pickle.dump(table, handle)
            cache = handle.name
        jobs = [(cache, feature_columns(groups, v), seed) for v in variants for seed in seeds]
        with ProcessPoolExecutor(max_workers=min(len(jobs), arguments.workers)) as pool:
            predictions = list(pool.map(_ltr_fit_worker, jobs))
        os.remove(cache)
        for index, variant in enumerate(variants):
            runs = predictions[index * len(seeds):(index + 1) * len(seeds)]
            table[f"ltr_{variant}{suffix}"] = np.mean(runs, axis=0)
            print(f"  {variant}: mean of {len(runs)} seeds ({time.time() - started:.0f}s)", flush=True)
    for variant in ([] if arguments.ensemble else variants):
        columns = feature_columns(groups, variant)
        table[f"ltr_{variant}{suffix}"] = leave_one_project_out(
            table, columns, seed=arguments.seed if arguments.seed is not None else SEED)
        print(f"  {variant}: {len(columns)} inputs ({time.time() - started:.0f}s)", flush=True)
    jobs = [(subject, {f"ltr_{v}{suffix}": rows[f"ltr_{v}{suffix}"].to_numpy() for v in variants})
            for subject, rows in table.groupby("subject", sort=False)]
    with ProcessPoolExecutor(max_workers=arguments.workers) as pool:
        written = list(pool.map(_ltr_write_worker, jobs, chunksize=1))
    print(f"wrote ltr weights for {len(written)} subjects ({time.time() - started:.0f}s)")


#: Trace-derived weightings shipped with the project (``traces/``): the inputs
#: the cross-subject ranker uses, so the raw ordered traces are not needed.
EXPORTED_TRACE_WEIGHTS = ("slice", "slice_depth", "recency", "slice_recency", "fsm", "fsm_nn")


def _export_trace_worker(job):
    import gzip
    import json

    from protocol.extract import read_artifact

    identifier, destination = job
    artifact = read_artifact(ARTIFACTS / f"{identifier}.json.gz")
    if "lines" not in artifact:
        return identifier, "no_lines"
    weights = {name: {"max": artifact["weights"][name]["max"]}
               for name in EXPORTED_TRACE_WEIGHTS if name in artifact["weights"]}
    if not weights:
        return identifier, "no_trace_weights"
    slim = {"subject": identifier,
            "lines": {"file": artifact["lines"]["file"], "line": artifact["lines"]["line"]},
            "weights": weights}
    with gzip.open(Path(destination) / f"{identifier}.json.gz", "wt") as handle:
        json.dump(slim, handle)
    return identifier, "ok"


def command_export_traces(arguments) -> None:
    """Export the trace-derived weightings of every artifact (run where the
    ordered traces were analysed: ``fsm-traces``/``fsm``/``slice-traces``/``slice``)."""
    destination = Path(arguments.destination)
    destination.mkdir(parents=True, exist_ok=True)
    keep = set(Path(arguments.subjects).read_text().split()) if arguments.subjects else None
    identifiers = sorted(p.name[: -len(".json.gz")] for p in ARTIFACTS.glob("*.json.gz"))
    identifiers = [i for i in identifiers if keep is None or i in keep]
    with ProcessPoolExecutor(max_workers=arguments.workers) as pool:
        results = list(pool.map(_export_trace_worker, [(i, destination) for i in identifiers]))
    ok = sum(status == "ok" for _, status in results)
    print(f"exported {ok}/{len(results)} subjects to {destination}")
    for identifier, status in results:
        if status != "ok":
            print(f"  {identifier}: {status}")


def _ltr_robust_worker(job):
    import os
    import pickle

    os.environ.setdefault("OMP_NUM_THREADS", "8")
    from protocol.ltr import (ROBUSTNESS, SEED, feature_columns, leave_one_project_out,
                              quick_metrics, signal_groups)

    cache, config, variant = job
    with open(cache, "rb") as handle:
        table, groups = pickle.load(handle)
    kind, params, seed = ROBUSTNESS[config]
    columns = feature_columns(groups, variant)
    if config == "base" and variant == "full":
        prediction, permuted = leave_one_project_out(table, columns, params, kind, seed or SEED,
                                                     permute=signal_groups(columns))
        reference = quick_metrics(table, prediction)
        importance = {g: {k: reference[k] - v for k, v in quick_metrics(table, p).items()}
                      for g, p in permuted.items()}
        return config, variant, reference, importance
    prediction = leave_one_project_out(table, columns, params, kind, seed or SEED)
    return config, variant, quick_metrics(table, prediction), None


def command_ltr_robustness(arguments) -> None:
    """Robustness of the cross-subject ranker (seeds, hyperparameters, model
    family) and permutation importance on held-out projects, with fast proxy
    metrics (best fault line per subject, midpoint rank among equal scores)."""
    import pickle
    import tempfile

    import pandas as pd

    from protocol.ltr import ROBUSTNESS, ROBUSTNESS_VARIANTS

    identifiers = sorted(p.name[: -len(".json.gz")] for p in ARTIFACTS.glob("*.json.gz"))
    with ProcessPoolExecutor(max_workers=arguments.workers) as pool:
        tables = [t for t in pool.map(_ltr_table_worker, identifiers, chunksize=1) if t is not None]
    groups = tables[0].attrs["groups"]
    table = pd.concat(tables, ignore_index=True)
    (ROOT / "study").mkdir(exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".pkl", dir=ROOT / "study", delete=False) as handle:
        pickle.dump((table, groups), handle)
        cache = handle.name
    jobs = [(cache, c, v) for c in ROBUSTNESS for v in ROBUSTNESS_VARIANTS]
    rows, importance = [], {}
    with ProcessPoolExecutor(max_workers=min(len(jobs), arguments.workers)) as pool:
        for config, variant, metrics, imp in pool.map(_ltr_robust_worker, jobs, chunksize=1):
            rows.append({"config": config, "variant": variant, **metrics})
            importance.update(imp or {})
    os.remove(cache)
    pd.DataFrame(rows).to_csv(ROOT / "study" / "ltr_robustness.csv", index=False)
    pd.DataFrame(importance).T.rename_axis("signal").to_csv(ROOT / "study" / "ltr_importance.csv")
    print(f"robustness: {len(rows)} runs")


def command_reweight(arguments) -> None:
    """Recompute every weighting condition from the persisted feature vectors."""
    identifiers = sorted(p.name[: -len(".json.gz")] for p in ARTIFACTS.glob("*.json.gz"))
    jobs = [(i, arguments.dry_run) for i in identifiers]
    print(f"{len(jobs)} subjects, {arguments.workers} workers"
          f"{' (dry run)' if arguments.dry_run else ''}", flush=True)
    done = bad = drift = 0
    results = (map(_reweight_worker, jobs) if arguments.workers <= 1 else
               ProcessPoolExecutor(max_workers=arguments.workers).map(_reweight_worker, jobs))
    for report in results:
        done += 1
        if report["status"] != "ok":
            bad += 1
            print(f"  {report['subject']}: {report['status']}", flush=True)
        elif report["changed"]:
            drift += 1
    verb = "would change" if arguments.dry_run else "changed"
    print(f"{done - bad}/{done} rebuilt; {verb} weights on {drift} subjects; {bad} failed")


def command_analyze(arguments) -> None:
    from protocol.analysis import run_analysis

    run_analysis(ROOT / "study", tag=arguments.tag)


def command_tables(arguments) -> None:
    from protocol.report import render

    render(ROOT / "study", ROOT / "paper_out")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="collect events and extract artifacts")
    run.add_argument("-p", "--projects", type=str, default=None)
    run.add_argument("-i", "--bug-id", type=int, default=None)
    run.add_argument("-w", "--workers", type=int, default=1)
    run.add_argument("--keep-events", action="store_true")
    run.add_argument("--force", action="store_true")
    run.add_argument("--no-stability", action="store_true")
    run.add_argument("--max-passes", type=int, default=8,
                     help="how often to rebuild the pool after a worker dies")
    run.set_defaults(func=command_run)

    subject = sub.add_parser("subject", help="run a single subject")
    subject.add_argument("-p", "--project", required=True)
    subject.add_argument("-i", "--bug-id", type=int, required=True)
    subject.add_argument("--keep-events", action="store_true")
    subject.add_argument("--force", action="store_true")
    subject.set_defaults(
        func=lambda a: print(
            json.dumps(run_subject(a.project, a.bug_id, a.keep_events, a.force), indent=1, default=str)
        )
    )

    status = sub.add_parser("status", help="artifact inventory")
    status.add_argument("-v", "--verbose", action="store_true")
    status.add_argument("--repair", action="store_true",
                        help="delete corrupt or failed artifacts so they are redone")
    status.set_defaults(func=command_status)

    collect = sub.add_parser("collect", help="build the tidy outcome table")
    collect.add_argument("--methods", type=str,
                         default="baseline,rf,alt,neg,pos,hybrid,failfirst,adaptive")
    collect.add_argument("--lam", type=float, default=0.35)
    collect.add_argument("--epsilon", type=float, default=1e-3)
    collect.add_argument("--aggregation", type=str, default="max")
    collect.add_argument("--threshold", type=float, default=0.6)
    collect.add_argument("--unmapped", type=str, default="floor",
                         choices=["floor", "neutral"],
                         help="post-hoc probe: how lines without importance evidence are treated")
    collect.add_argument("--product", type=str, default="raw",
                         choices=["raw", "prenorm"],
                         help="post-hoc probe: what the weight multiplies. "
                              "'raw' is the registered susp'(l) x w(l); 'prenorm' "
                              "multiplies the normalised score so the operand stays "
                              "non-negative for every formula")
    collect.add_argument("--repeats", type=int, default=10000)
    collect.add_argument("--tag", type=str, default=None)
    collect.add_argument("-w", "--workers", type=int, default=1)
    collect.set_defaults(func=command_collect)

    signprobe = sub.add_parser(
        "signprobe", help="post-hoc: negative-range behaviour of the registered blend")
    signprobe.set_defaults(func=command_signprobe)

    prepare = sub.add_parser("prepare", help="checkout and build subjects (no tests)")
    prepare.add_argument("-p", "--projects", type=str, default=None)
    prepare.add_argument("-i", "--bug-id", type=int, default=None)
    prepare.add_argument("-w", "--workers", type=int, default=8)
    prepare.add_argument("--retries", type=int, default=3)
    prepare.set_defaults(func=command_prepare)

    posthoc = sub.add_parser("posthoc", help="post-hoc feature scorers (exploratory)")
    posthoc.add_argument("-w", "--workers", type=int, default=4)
    posthoc.set_defaults(func=command_posthoc)

    robust = sub.add_parser("ltr-robustness", help="robustness grid of the cross-subject ranker")
    robust.add_argument("-w", "--workers", type=int, default=8)
    robust.set_defaults(func=command_ltr_robustness)
    export = sub.add_parser("export-traces", help="export trace-derived weightings (traces/)")
    export.add_argument("destination", type=str)
    export.add_argument("--subjects", type=str, default=None, help="file listing the subjects to export")
    export.add_argument("-w", "--workers", type=int, default=8)
    export.set_defaults(func=command_export_traces)
    ltr = sub.add_parser("ltr", help="cross-subject learning to rank (exploratory)")
    ltr.add_argument("-w", "--workers", type=int, default=8)
    ltr.add_argument("--variants", type=str, default=None)
    ltr.add_argument("--seed", type=int, default=None)
    ltr.add_argument("--ensemble", action="store_true",
                     help="mean of the pre-fixed seed and seeds 1-5 (stored as ltr_<variant>_ens)")
    ltr.set_defaults(func=command_ltr)
    transplant = sub.add_parser("transplant", help="add trace-derived weights from another artifact folder")
    transplant.add_argument("source", type=str)
    transplant.add_argument("-w", "--workers", type=int, default=8)
    transplant.set_defaults(func=command_transplant)
    slice_traces = sub.add_parser("slice-traces", help="dynamic slices of failing traces")
    slice_traces.add_argument("-w", "--workers", type=int, default=8)
    slice_traces.set_defaults(func=command_slice_traces)
    slice_ = sub.add_parser("slice", help="slicing/proximity weightings (exploratory)")
    slice_.add_argument("-w", "--workers", type=int, default=8)
    slice_.set_defaults(func=command_slice)
    fsm_traces = sub.add_parser("fsm-traces", help="decode ordered traces into automaton transitions")
    fsm_traces.add_argument("-w", "--workers", type=int, default=8)
    fsm_traces.set_defaults(func=command_fsm_traces)
    fsm = sub.add_parser("fsm", help="automaton weightings (exploratory)")
    fsm.add_argument("-w", "--workers", type=int, default=8)
    fsm.set_defaults(func=command_fsm)
    reweight = sub.add_parser(
        "reweight", help="rebuild all weighting conditions from the feature vectors")
    reweight.add_argument("-w", "--workers", type=int, default=4)
    reweight.add_argument("--dry-run", action="store_true",
                          help="report which subjects would change, write nothing")
    reweight.set_defaults(func=command_reweight)

    analyze = sub.add_parser("analyze", help="run the pre-registered analysis plan")
    analyze.add_argument("--tag", type=str, default=None)
    analyze.set_defaults(func=command_analyze)

    sensitivity = sub.add_parser("sensitivity", help="pre-registered sensitivity grids")
    sensitivity.add_argument("--repeats", type=int, default=10000)
    sensitivity.add_argument("-w", "--workers", type=int, default=1)
    sensitivity.set_defaults(func=command_sensitivity)

    verify = sub.add_parser("verify", help="replicability check on a seeded sample")
    verify.add_argument("-n", type=int, default=10)
    verify.set_defaults(func=command_verify)

    variants = sub.add_parser("variants", help="recompute exploratory ranking variants")
    variants.add_argument("-w", "--workers", type=int, default=4)
    variants.add_argument("--force", action="store_true",
                          help="recompute even for subjects that already have them")
    variants.set_defaults(func=command_variants)

    feasibility = sub.add_parser("feasibility", help="runtime and failure statistics")
    feasibility.set_defaults(func=command_feasibility)

    family = sub.add_parser("families", help="per-feature-family weight ablation")
    family.add_argument("-w", "--workers", type=int, default=4)
    family.set_defaults(func=command_family_ablation)

    checks = sub.add_parser("checks", help="pre-registered data-quality checks")
    checks.set_defaults(func=command_checks)

    perm = sub.add_parser("permcontrol", help="post-hoc importance-permutation control")
    perm.add_argument("-w", "--workers", type=int, default=4)
    perm.set_defaults(func=command_permcontrol)

    categories = sub.add_parser("categories", help="feature-family importance shares")
    categories.set_defaults(func=command_categories)

    tables = sub.add_parser("tables", help="render manuscript tables and figures")
    tables.set_defaults(func=command_tables)

    arguments = parser.parse_args(argv)
    arguments.func(arguments)


if __name__ == "__main__":
    main()
