"""Phase 1 event collection with the pre-registered 120 s per-test timeout.

Tests4Py's own ``sflkit_unittest`` helper hard-codes a 60 s per-test timeout,
so the test execution is driven directly through SFLKit's ``PytestRunner``
with the timeout the registered report specifies.  Checkout, build and
instrumentation reuse the Tests4Py API unchanged.
"""

from __future__ import annotations

import protocol.compat  # noqa: F401  (must precede any Tests4Py use)

import os
import re
import shutil
import time
import traceback
from pathlib import Path
import subprocess
from typing import Dict, List, Optional, Set, Tuple

import tests4py.api as t4p
from sflkit.runners import PytestRunner
from tests4py.api.utils import load_project
from tests4py.constants import GLOBAL_PROJECTS
from tests4py.environment import activate_venv, env_on
from tests4py.projects import Project
from tests4py.sfl import instrument
from tests4py.tests.utils import get_pytest_skip

from protocol.instrumentation import create_config

#: Pre-registered per-test timeout (Section "Execution Plan", Phase 1).
TEST_TIMEOUT = 120

#: Written once a trace set is known to be complete.
COMPLETE_MARKER = ".complete"

#: Build-time constraints for the benchmark's editable installs. Tests4Py
#: subjects are pinned to old releases, but pip resolves *build* dependencies
#: fresh, so a modern backend can reject a project that was fine when the
#: benchmark was built: flit_core 4 refuses the ``[tool.flit.metadata]`` table
#: that FastAPI's revisions use. Constraining the build backend keeps those
#: subjects analysable instead of excluding them for a packaging accident.
BUILD_CONSTRAINTS = ("flit_core<4", "exceptiongroup<1.2")

#: Packages that a cached project environment may already carry at a version
#: too new for the interpreter the subject is pinned to. Constraining the build
#: does not help here: ``copy_cached_env`` reproduces an environment that is
#: already broken, so the offending package has to be downgraded in place.
ENVIRONMENT_PINS = ("exceptiongroup<1.2",)

#: Further repairs, tried in order only while collection is still empty: a
#: pytest that is too new for a subject's conftest leaves collection empty as
#: well. pytest >= 3.6 rejects applying ``pytest.fixture`` to an
#: already-decorated function, which httpie_1's conftest does with
#: ``httpbin_ca_bundle``; its original environment shipped pytest 3.2.1.
RUNNER_PINS = (("pytest<8",), ("pytest<3.6",))


#: Applied only when a subject's first build fails (``run_protocol.py prepare``
#: retries with them): numpy 1.15 built from source needs a setuptools that
#: still ships ``distutils.msvccompiler``, subjects pinning an old
#: ``importlib-metadata`` need a setuptools that does not import it (before 60),
#: and uvloop built from
#: source needs a Cython that still supports Python 3.8 (3.1 dropped it). The
#: choice is recorded in the checkout so instrumentation rebuilds reuse it.
FALLBACK_CONSTRAINTS = ("setuptools<60", "Cython<3.1")
FALLBACK_MARKER = ".fallback-constraints"


def _requirement_pins(requirements: Optional[Path]) -> Dict[str, str]:
    """Package name -> exact pinned version (``name==version`` lines only)."""
    if requirements is None or not requirements.exists():
        return {}
    pins = {}
    for line in requirements.read_text().splitlines():
        line = line.split("#", 1)[0].split(";", 1)[0].strip()
        match = re.match(r"^([A-Za-z0-9_.\-]+)\s*==\s*([^\s,]+)$", line)
        if match:
            pins[match.group(1).lower().replace("_", "-")] = match.group(2)
    return pins


def _compatible(constraint: str, pins: Dict[str, str]) -> bool:
    """False only if the subject pins this package at a version the constraint excludes."""
    from packaging.requirements import Requirement

    requirement = Requirement(constraint)
    pinned = pins.get(requirement.name.lower().replace("_", "-"))
    return pinned is None or requirement.specifier.contains(pinned, prereleases=True)


#: Records which pins a fallback build relaxed (see ``relax_unsupported_pins``).
RELAXED_MARKER = ".relaxed-pins"


def _requires_python(name: str, version: str) -> Optional[str]:
    """``Requires-Python`` of one release on PyPI ('' if none, None if unknown)."""
    import json
    import urllib.request

    url = f"https://pypi.org/pypi/{name}/{version}/json"
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                return json.load(response)["info"].get("requires_python") or ""
        except Exception:  # noqa: BLE001
            time.sleep(2 * (attempt + 1))
    return None


def relax_unsupported_pins(requirements: Path, python_version: str) -> List[str]:
    """Unpin exact requirements whose release does not support the interpreter.

    Tests4Py shares one requirements file across the bugs of a project, while
    single bugs may be pinned to an older interpreter (sanic_5: Python 3.7.8
    against pins such as ``flake8==7.1.0``, which require Python >= 3.8). Such a
    pin cannot be installed at all, so the environment never builds. Each one is
    replaced by its bare package name, letting pip pick the newest release that
    supports the interpreter. Pins the interpreter supports are never touched,
    so this cannot change an environment that builds. Returns the relaxed pins,
    which are also recorded next to the requirements file.
    """
    from packaging.specifiers import InvalidSpecifier, SpecifierSet

    lines = requirements.read_text().splitlines()
    relaxed, out = [], []
    for line in lines:
        bare = line.split("#", 1)[0].strip()
        match = re.match(r"^([A-Za-z0-9_.\-]+)\s*==\s*([^\s,;]+)$", bare)
        if match:
            name, version = match.groups()
            requires = _requires_python(name, version)
            try:
                unsupported = bool(requires) and not SpecifierSet(requires).contains(
                    python_version, prereleases=True)
            except InvalidSpecifier:
                unsupported = False
            if unsupported:
                relaxed.append(f"{name}=={version} (Requires-Python {requires})")
                out.append(name)
                continue
        out.append(line)
    if relaxed:
        requirements.write_text("\n".join(out) + "\n")
        (requirements.parent / RELAXED_MARKER).write_text(
            f"Python {python_version}\n" + "\n".join(relaxed) + "\n")
    return relaxed


def ensure_build_constraints(root: Path = Path("."), requirements: Optional[Path] = None,
                             fallback: Optional[bool] = None) -> Path:
    """Point pip at a constraints file for build dependencies.

    ``PIP_CONSTRAINT`` applies to every install, not only to build backends, so
    a constraint that contradicts a version the subject itself pins (e.g.
    ``exceptiongroup<1.2`` against ``exceptiongroup==1.2.2``) would make its
    environment uninstallable; such a constraint is dropped for that subject. A
    constraint the pin satisfies is kept, because it still governs pip's
    isolated build environments. ``fallback`` adds ``FALLBACK_CONSTRAINTS``; by
    default it follows the marker that ``prepare`` leaves in the checkout.
    """
    if fallback is None:
        fallback = (root / FALLBACK_MARKER).exists()
    pins = _requirement_pins(requirements)
    candidates = list(BUILD_CONSTRAINTS) + (list(FALLBACK_CONSTRAINTS) if fallback else [])
    constraints = [c for c in candidates if _compatible(c, pins)]
    path = (root / "pip-constraints.txt").absolute()
    expected = "\n".join(constraints) + "\n"
    if not path.exists() or path.read_text() != expected:
        path.write_text(expected)
    os.environ["PIP_CONSTRAINT"] = str(path)
    return path


def cached_venv(project: Project) -> Path:
    return GLOBAL_PROJECTS / project.project_name / f"venv_{project.bug_id}"


def run_instrumented_tests(work_dir: Path, output: Path) -> Dict[str, int]:
    project = load_project(work_dir, only_project=True)
    environ = activate_venv(work_dir, env_on(project))
    runner = PytestRunner(timeout=TEST_TIMEOUT)
    k = get_pytest_skip(project.skip_tests) if project.skip_tests else None
    runner.run(
        directory=work_dir,
        output=output,
        files=project.relevant_test_files,
        base=project.test_base,
        environ=environ,
        k=k,
    )
    return {
        "passing": len(runner.passing_tests),
        "failing": len(runner.failing_tests),
        "undefined": len(runner.undefined_tests),
    }


def _collects_tests(project: Project, work_dir: Path, environ) -> Tuple[bool, str]:
    """Does pytest actually collect anything in the instrumented tree?"""
    files = [str(f) for f in (project.relevant_test_files or [])]
    result = subprocess.run(
        ["python3", "-m", "pytest", "--collect-only", "-q", *files],
        env=environ, cwd=work_dir, capture_output=True,
    )
    output = (result.stdout + result.stderr).decode(errors="ignore")
    #: Collected node ids are listed as ``path::name``. The summary line is no
    #: signal: pytest < 6 ends ``--collect-only`` with "no tests ran" even when
    #: it collected tests.
    collected = result.returncode == 0 and "::" in output
    return collected, output.strip()[-300:]


def repair_test_environment(project: Project, work_dir: Path) -> Optional[str]:
    """Make sure the instrumented tree can actually collect its tests.

    Tests4Py pins each subject to an old interpreter, but the cached environment
    holds whatever the ecosystem offered when it was built. A dependency that
    later dropped support for that interpreter -- pytest's ``exceptiongroup`` on
    Python 3.8 is the case we hit -- makes collection return nothing, which is
    indistinguishable from a subject that has no tests. We probe collection
    rather than importability, because that is the symptom, and downgrade the
    known offenders only when the probe fails.
    """
    environ = activate_venv(work_dir, env_on(project))
    collected, output = _collects_tests(project, work_dir, environ)
    if collected:
        return None
    subprocess.run(
        ["python3", "-m", "pip", "install", "-q", *ENVIRONMENT_PINS],
        env=environ, cwd=work_dir, capture_output=True,
    )
    collected, output = _collects_tests(project, work_dir, environ)
    if collected:
        return "repaired: " + ", ".join(ENVIRONMENT_PINS)
    applied = list(ENVIRONMENT_PINS)
    for runner_pins in RUNNER_PINS:
        subprocess.run(
            ["python3", "-m", "pip", "install", "-q", *runner_pins],
            env=environ, cwd=work_dir, capture_output=True,
        )
        applied += runner_pins
        collected, output = _collects_tests(project, work_dir, environ)
        if collected:
            return "repaired: " + ", ".join(applied)
    return f"collection still empty after {', '.join(applied)}: {output}"


def collect_events(
    project: Project,
    events_root: Path = Path("sflkit_events"),
    mappings_root: Path = Path("mappings"),
    checkout_root: Path = Path("tmp"),
) -> Dict:
    identifier = project.get_identifier()
    report: Dict[str, object] = {"subject": identifier, "time": {}}
    events = events_root / project.project_name / str(project.bug_id)
    ensure_build_constraints()

    start = time.time()
    checkout = checkout_root / identifier
    if not checkout.exists():
        result = t4p.checkout(project)
        if not result.successful:
            report["status"] = "checkout_failed"
            report["error"] = "".join(traceback.format_exception(result.raised))[-2000:]
            return report
        checkout = Path(result.location)
    report["time"]["checkout"] = time.time() - start
    ensure_build_constraints(checkout, checkout / "tests4py_requirements.txt")

    #: ``run_protocol.py prepare`` builds the project venv inside the checkout
    #: ahead of time and leaves this marker; then collection does not rebuild.
    prepared = (checkout / ".prepared").exists()
    if not cached_venv(project).exists() and not prepared:
        start = time.time()
        result = t4p.build(checkout)
        report["time"]["build"] = time.time() - start
        if not result.successful:
            report["status"] = "build_failed"
            report["error"] = "".join(traceback.format_exception(result.raised))[-2000:]
            return report

    mapping = mappings_root / f"{project}.json"
    mapping.parent.mkdir(parents=True, exist_ok=True)
    instrumented = checkout_root / f"sfl_{identifier}"
    shutil.rmtree(instrumented, ignore_errors=True)
    start = time.time()
    try:
        # ``tests4py.sfl.instrument`` rewrites the sources and then rebuilds the
        # instrumented tree with ``sfl=True``, which installs ``sflkitlib`` and
        # re-runs the project's editable install so the tests import the
        # instrumented code rather than the original checkout.
        instrument(create_config(project, checkout, instrumented, mapping=mapping))
    except BaseException as error:  # noqa: BLE001
        report["status"] = "instrument_failed"
        report["error"] = f"{type(error).__name__}: {error}"
        return report
    report["time"]["instrument"] = time.time() - start

    repair = repair_test_environment(project, instrumented)
    if repair:
        report["environment"] = repair

    shutil.rmtree(events, ignore_errors=True)
    if project.project_name == "ansible":
        # Ansible occasionally imports the uninstrumented checkout.
        shutil.rmtree(checkout, ignore_errors=True)
    start = time.time()
    try:
        report.update(run_instrumented_tests(instrumented, events))
        # A run that produced no usable traces is not a complete trace set. The
        # marker is only written when both outcome classes are present, so an
        # environment failure (for instance a project virtual environment whose
        # pytest no longer imports) is retried and finally reported as a
        # collection failure rather than silently becoming an empty subject.
        if report.get("passing", 0) > 0 and report.get("failing", 0) > 0:
            report["status"] = "ok"
            (events / COMPLETE_MARKER).write_text(
                f"{report['passing']} passing, {report['failing']} failing\n"
            )
        else:
            report["status"] = "no_traces"
            report["error"] = (
                f"instrumented test run produced {report.get('passing', 0)} passing "
                f"and {report.get('failing', 0)} failing traces"
            )
    except BaseException as error:  # noqa: BLE001
        report["status"] = "test_failed"
        report["error"] = f"{type(error).__name__}: {error}"
    report["time"]["tests"] = time.time() - start
    return report


def events_complete(events: Path) -> bool:
    return (events / COMPLETE_MARKER).exists()
