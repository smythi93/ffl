import numpy as np
from sflkitlib.events.event import (
    DefEvent, FunctionEnterEvent, FunctionErrorEvent, FunctionExitEvent, LineEvent, UseEvent,
)

from protocol.slicing import Sources, analyze_events, control_parents

SOURCE = """\
def f(a):
    if a:
        b = 1
    elif a is None:
        b = 2
    else:
        b = 3
    c = 4
    return b
"""


class FakeSources(Sources):
    def __init__(self, table):
        self.cache = {"m.py": table}


def L(n):
    return LineEvent("m.py", n, 0)


def D(n, var, vid):
    return DefEvent("m.py", n, 0, var, vid, 0, "int")


def U(n, var, vid):
    return UseEvent("m.py", n, 0, var, vid)


def test_control_parents_follow_the_innermost_header():
    parents = control_parents(SOURCE)
    assert parents[3] == 2 and parents[4] == 2 and parents[5] == 4 and parents[7] == 4
    assert 8 not in parents and 9 not in parents  # top level of the function


def test_slice_keeps_data_and_control_dependences_and_drops_the_rest():
    # test calls f: line 2 (if a), line 3 (b = 1), line 8 (c = 4), line 9 (return b)
    events = [
        FunctionEnterEvent("m.py", 1, 0, "f", 0),
        L(2), U(2, "a", 1),
        L(3), D(3, "b", 11),
        L(8), D(8, "c", 12),
        L(9), U(9, "b", 11),
        FunctionExitEvent("m.py", 1, 0, "f", 0, "tmp"),
    ]
    result = analyze_events(events, FakeSources(control_parents(SOURCE)))
    depth = {line: d for (_, line), d in result["depth"].items()}
    assert depth[9] == 0          # criterion: returned to the test
    assert depth[3] == 1          # defines the returned b
    assert depth[2] == 2          # controls line 3
    assert 8 not in depth         # c never reaches the test
    recency = {line: r for (_, line), r in result["recency"].items()}
    assert recency[9] == 1 and recency[8] == 2


def test_crash_site_is_a_criterion():
    events = [
        FunctionEnterEvent("m.py", 1, 0, "f", 0),
        L(8), D(8, "c", 12),
        FunctionEnterEvent("m.py", 20, 0, "g", 1),
        L(21), U(21, "c", 12),
        FunctionErrorEvent("m.py", 20, 0, "g", 1),
        FunctionErrorEvent("m.py", 1, 0, "f", 0),
    ]
    result = analyze_events(events, FakeSources({}))
    depth = {line: d for (_, line), d in result["depth"].items()}
    assert depth[21] == 0 and depth[8] == 1


def test_arguments_link_callee_uses_to_caller_definitions():
    events = [
        FunctionEnterEvent("m.py", 1, 0, "f", 0),
        L(8), D(8, "x", 42),
        FunctionEnterEvent("m.py", 20, 0, "g", 1),
        D(20, "a", 42),            # parameter bound before g's first line
        L(21), U(21, "a", 42),
        FunctionExitEvent("m.py", 20, 0, "g", 1, "tmp"),
        FunctionExitEvent("m.py", 1, 0, "f", 0, "tmp"),
    ]
    result = analyze_events(events, FakeSources({}))
    depth = {line: d for (_, line), d in result["depth"].items()}
    assert 21 in depth and 8 in depth
