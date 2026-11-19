"""Optional per-worker progress channel for block readers and writers."""
from contextlib import contextmanager
from contextvars import ContextVar

_sink = ContextVar('work_progress', default=None)


@contextmanager
def reporting(callback):
    previous = None

    def notify(done, total, stage):
        nonlocal previous
        percent = min(99, max(0, int(100 * done / total))) if total else -1
        if previous != (stage, percent):
            previous = stage, percent
            callback(stage, percent)

    token = _sink.set(notify)
    try:
        yield
    finally:
        _sink.reset(token)


def report(done, total, stage):
    sink = _sink.get()
    if sink is not None:
        sink(done, total, stage)
