"""Compatibility shims for the Tests4Py installation on the study machines.

Two adjustments keep the pinned Tests4Py 0.0.13 usable on the shuttles:

* ``~/.t4p/config.ini`` is shared with other studies that use a newer Tests4Py
  release and therefore carries keys 0.0.13 does not accept, so the global
  configuration is read tolerantly;
* subjects are processed by several worker processes at once, so the global
  configuration is never written back.  Writing it is unnecessary here because
  every Tests4Py call in this study passes its working directory explicitly,
  and concurrent writes would race.  Virtual-environment caching is forced on
  so workers reuse the pre-built project environments.
"""

from __future__ import annotations

from tests4py.api import config as _config


def _fixed_load():
    return _config.GlobalConfig(cache=True, last_workdir=None)


def _no_write(self) -> None:  # noqa: ANN001
    return None


def install() -> None:
    _config.GlobalConfig.load = staticmethod(_fixed_load)
    _config.GlobalConfig.write = _no_write


install()
