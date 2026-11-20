"""SFLKit instrumentation config for a Tests4Py subject.

Moved here from the ``eval`` package, which held the pre-protocol prototypes.
This is the only piece of that package phase 1 ever needed, and keeping it
beside :mod:`protocol.events` means the protocol pipeline no longer depends on
code the registered report does not describe.
"""

from pathlib import Path
from typing import Optional

import tests4py.api as t4p
from sflkit import Config
from sflkitlib.events import EventType
from tests4py.projects import Project
from tests4py.sfl import get_events_path
from tests4py.sfl.constants import DEFAULT_EXCLUDES

#: Subjects whose instrumentation needs a deviation from the defaults.
#: ``pysnooper`` defines a test function inside a test function and traces it;
#: with instrumentation that inner trace is asserted and does not match, so
#: inner functions are ignored for it. ``calculator`` and ``markup`` do not
#: declare a test base, so one is supplied.
IGNORE_INNER = ("pysnooper",)
DEFAULT_TEST_BASE = ("calculator", "markup")


def create_config(
    project: Project,
    src: Path,
    dst: Path,
    metrics: Optional[str] = None,
    events_path: Optional[Path] = None,
    mapping: Optional[Path] = None,
    only_patched_files: bool = False,
    include_suffix: bool = False,
) -> Config:
    """Build the SFLKit configuration used to instrument ``project``."""
    if only_patched_files:
        includes = t4p.get_patched_files(project)
        excludes = list()
    elif project.included_files:
        includes = project.included_files
        excludes = project.excluded_files
    elif project.excluded_files:
        includes = list()
        excludes = project.excluded_files
    else:
        includes = list()
        excludes = DEFAULT_EXCLUDES
    if project.project_name in DEFAULT_TEST_BASE:
        project.test_base = Path("tests")
    test_files = list({str(file.split("::")[0]) for file in project.test_cases})
    return Config.create(
        path=str(src.absolute()),
        language="python",
        events=",".join([event.name for event in EventType.events()]),
        ignore_inner=str(project.project_name in IGNORE_INNER),
        metrics=metrics or "",
        passing=str(
            get_events_path(
                project=project,
                passing=True,
                events_path=events_path,
                include_suffix=include_suffix,
            )
        ),
        failing=str(
            get_events_path(
                project=project,
                passing=False,
                events_path=events_path,
                include_suffix=include_suffix,
            )
        ),
        working=str(dst.absolute()),
        include='"' + '","'.join(includes) + '"',
        exclude='"' + '","'.join(excludes) + '"',
        test_files='"' + '","'.join(test_files) + '"',
        mapping_path=str(mapping.absolute()) if mapping else "",
    )


__all__ = ["create_config"]
