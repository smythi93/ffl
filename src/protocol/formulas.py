"""The five pre-registered SFL baseline formulas.

The implementations reproduce SFLKit's ``Spectrum`` methods exactly, including
its convention of mapping division-by-zero and NaN results to ``0``.
"""

from __future__ import annotations

import numpy as np

FORMULAS = ("Tarantula", "Ochiai", "DStar", "Naish2", "GP13")


def _clean(values: np.ndarray) -> np.ndarray:
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)


def suspiciousness(name: str, ef: np.ndarray, ep: np.ndarray, tf: int, tp: int) -> np.ndarray:
    ef = np.asarray(ef, dtype=float)
    ep = np.asarray(ep, dtype=float)
    nf = float(tf) - ef  # failing runs that do not observe the line
    with np.errstate(divide="ignore", invalid="ignore"):
        if name == "Tarantula":
            values = (ef / tf) / ((ef / tf) + (ep / tp))
        elif name == "Ochiai":
            values = ef / np.sqrt(float(tf) * (ef + ep))
        elif name == "DStar":
            values = ef**2 / (nf + ep)
        elif name == "Naish2":
            values = ef - ep / (float(tp) + 1.0)
        elif name == "GP13":
            values = ef * (1.0 + 1.0 / (2.0 * ep + ef))
        else:
            raise ValueError(f"unknown formula {name}")
    return _clean(values)


def min_max(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    lo, hi = float(values.min()), float(values.max())
    if hi <= lo:
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)
