"""Throttled reporting of completed operator blocks and wall-clock timings."""
from time import perf_counter


class WorkProgress:
    def __init__(self, callback, xp, interval=.25):
        self.callback=callback; self.xp=xp; self.interval=interval
        self.started=perf_counter(); self.iteration_started=self.started
        self.iteration=0; self.trial=0; self.last_emit=0.

    def sync(self):
        if hasattr(self.xp,'cuda'): self.xp.cuda.get_current_stream().synchronize()

    def begin_iteration(self, iteration):
        self.iteration=iteration; self.trial=0; self.iteration_started=perf_counter()

    def begin_stage(self, stage, total):
        if not self.callback: return
        self.sync()
        self.stage=stage; self.total=total; self.stage_started=perf_counter()
        self.update(0, force=True)

    def update(self, completed, force=False):
        if not self.callback: return
        now=perf_counter()
        if not force and completed!=self.total and now-self.last_emit<self.interval: return
        # CUDA submissions are asynchronous: count work only after it has completed.
        self.sync(); now=perf_counter(); self.last_emit=now
        stage_seconds=now-self.stage_started
        remaining=stage_seconds*(self.total-completed)/completed if completed else None
        self.callback(dict(event='work',iteration=self.iteration,trial=self.trial,
            stage=self.stage,completed=completed,total=self.total,
            elapsed_seconds=now-self.started,iteration_seconds=now-self.iteration_started,
            stage_seconds=stage_seconds,stage_remaining_seconds=remaining))


def duration(seconds):
    if seconds is None: return 'оценка появится после замеров'
    seconds=max(0,int(seconds)); hours,remainder=divmod(seconds,3600); minutes,seconds=divmod(remainder,60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'
