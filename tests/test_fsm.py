import numpy as np
from sflkitlib.events.event import (
    FunctionEnterEvent, FunctionErrorEvent, FunctionExitEvent, LineEvent,
)

from protocol.fsm import nearest_neighbour, ochiai, transitions_from_events


def line(n, file="m.py"):
    return LineEvent(file, n, 0)


def enter(n, file="m.py"):
    return FunctionEnterEvent(file, n, 0, "f", 0)


def exit_(n, file="m.py"):
    return FunctionExitEvent(file, n, 0, "f", 0, "tmp")


def test_frames_separate_caller_and_callee_lines():
    events = [line(1), line(2), enter(10), line(11), line(12), exit_(10), line(3)]
    t = transitions_from_events(events)
    a = lambda n: ("m.py", n)  # noqa: E731
    #: Line 3 follows line 2 in the caller frame, not line 12 of the callee.
    assert t["edge"] == {(a(1), a(2)), (a(11), a(12)), (a(2), a(3))}
    assert t["path"] == {(a(1), a(2), a(3))}
    assert t["call"] == {(a(2), a(10))}


def test_error_exit_pops_the_frame():
    events = [line(1), enter(10), line(11), FunctionErrorEvent("m.py", 10, 0, "f", 0), line(2)]
    t = transitions_from_events(events)
    assert (("m.py", 1), ("m.py", 2)) in t["edge"]
    assert (("m.py", 11), ("m.py", 2)) not in t["edge"]


def test_unbalanced_exit_keeps_the_base_frame():
    t = transitions_from_events([exit_(10), line(1), line(2)])
    assert t["edge"] == {(("m.py", 1), ("m.py", 2))}


def test_ochiai_prefers_failing_only_transitions():
    s = ochiai(np.array([1.0, 1.0, 0.0]), np.array([0.0, 3.0, 2.0]), 1)
    assert s[0] == 1.0 and s[1] == 0.5 and s[2] == 0.0


def test_nearest_neighbour_credits_only_the_nearest_difference():
    sets = [np.array([0, 1, 2]), np.array([0, 1]), np.array([3])]
    labels = np.array([1, 0, 0])
    scores = nearest_neighbour(sets, labels, 4)
    #: The nearest passing run differs from the failing run in item 2 alone.
    assert scores.tolist() == [0.0, 0.0, 1.0, 0.0]
