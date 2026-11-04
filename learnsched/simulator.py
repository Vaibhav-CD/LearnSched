"""Single-core CPU scheduling simulator.

Tick-based (1 tick = 1 time unit). At each tick: arrivals and I/O
completions join the ready structure, one ready process runs for 1 tick,
waiting time accrues for every other ready process.

Context-switch overhead: when the CPU switches to a *different* process
than the one that ran last, ``cs_overhead`` ticks are consumed as overhead
(no process makes progress, ready processes still accrue waiting time).
With cs_overhead=0 behaviour matches textbook hand calculations exactly.

Policies: fcfs, sjf (non-preemptive), rr, mlfq.
MLFQ here is also the engine driven stepwise by the RL environment.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .process import Process

ALGORITHMS = ("fcfs", "sjf", "rr", "mlfq")


@dataclass
class MLFQConfig:
    """Configurable MLFQ parameters (the PPO action selects among these)."""
    quanta: tuple = (2.0, 4.0, 8.0, float("inf"))  # last queue is FCFS
    boost_interval: float = 50.0  # 0 disables boost

    def label(self) -> str:
        q = "/".join("inf" if x == float("inf") else str(int(x)) for x in self.quanta)
        return f"q={q},boost={int(self.boost_interval) if self.boost_interval else 'off'}"


@dataclass
class Segment:
    pid: int | None  # None = idle, -1 = context-switch overhead
    start: float
    end: float


@dataclass
class SimResult:
    processes: list[Process]
    timeline: list[Segment]
    context_switches: int
    makespan: float
    overhead_time: float = 0.0
    decision_overhead: float = 0.0  # controller (PPO) wall-clock cost, if any
    config_history: list = field(default_factory=list)  # (time, label) for MLFQ changes

    def metrics(self) -> dict:
        from .metrics import compute_metrics
        return compute_metrics(self)


class Simulator:
    """Stateful tick simulator. Can run to completion or step N ticks."""

    def __init__(self, processes: list[Process], algorithm: str = "mlfq",
                 mlfq: MLFQConfig | None = None, rr_quantum: float = 4.0,
                 cs_overhead: float = 0.0):
        if algorithm not in ALGORITHMS:
            raise ValueError(f"unknown algorithm {algorithm!r}, expected {ALGORITHMS}")
        self.algorithm = algorithm
        self.cs_overhead = float(cs_overhead)
        self.rr_quantum = float(rr_quantum)
        self.procs = sorted((p.fresh_copy() for p in processes), key=lambda p: (p.arrival, p.pid))
        self.by_pid = {p.pid: p for p in self.procs}
        self.time = 0.0
        self._future = deque(self.procs)          # not yet arrived, sorted
        self._io: list[Process] = []              # blocked on I/O
        self.ready: list[Process] = []            # fcfs/sjf/rr ready list
        self.queues: list[deque] = []             # mlfq queues
        self.running: Process | None = None
        self.set_mlfq_config(mlfq or MLFQConfig(), record=False)
        self._cs_left = 0.0
        self._last_pid: int | None = None
        self._since_boost = 0.0
        self.timeline: list[Segment] = []
        self.context_switches = 0
        self.overhead_time = 0.0
        self.done_count = 0
        self.config_history: list = []
        self.wait_ticks_window = 0.0
        self.completed_window = 0
        self.cs_window = 0
        self.busy_ticks = 0.0
        self.window_ticks = 0.0

    # -- configuration -------------------------------------------------
    def set_mlfq_config(self, cfg: MLFQConfig, record: bool = True) -> None:
        self.mlfq = cfg
        n = len(cfg.quanta)
        if len(self.queues) != n:  # rebuild, preserving waiting processes top-down
            waiting = [p for q in self.queues for p in q]
            if self.running is not None and self.algorithm == "mlfq":
                waiting.append(self.running)
                self.running = None
            self.queues = [deque() for _ in range(n)]
            for p in waiting:
                p.queue_level = min(p.queue_level, n - 1)
                self.queues[p.queue_level].append(p)
        if record:
            self.config_history.append((self.time, cfg.label()))

    # -- internals -------------------------------------------------------
    def _enqueue(self, p: Process, level: int | None = None) -> None:
        p.quantum_used = 0.0
        if self.algorithm == "mlfq":
            p.queue_level = 0 if level is None else level
            p.quantum_used = 0.0
            self.queues[p.queue_level].append(p)
        else:
            self.ready.append(p)

    def _admit(self) -> None:
        while self._future and self._future[0].arrival <= self.time:
            self._enqueue(self._future.popleft())
        finished_io = [p for p in self._io if p.io_done_at is not None and p.io_done_at <= self.time]
        for p in finished_io:
            self._io.remove(p)
            # I/O-bound (interactive) processes re-enter at the same level:
            # they yielded the CPU voluntarily, so no demotion.
            self._enqueue(p, level=p.queue_level if self.algorithm == "mlfq" else None)

    def _pick(self) -> Process | None:
        if self.algorithm == "fcfs":
            return self.ready.pop(0) if self.ready else None
        if self.algorithm == "sjf":
            if not self.ready:
                return None
            best = min(self.ready, key=lambda p: (p.remaining, p.arrival, p.pid))
            self.ready.remove(best)
            return best
        if self.algorithm == "rr":
            return self.ready.pop(0) if self.ready else None
        for q in self.queues:
            if q:
                return q.popleft()
        return None

    def _quantum_for(self, p: Process) -> float:
        if self.algorithm == "rr":
            return self.rr_quantum
        if self.algorithm == "mlfq":
            return self.mlfq.quanta[p.queue_level]
        return float("inf")

    def _boost(self) -> None:
        if self.algorithm != "mlfq" or not self.mlfq.boost_interval:
            return
        if self._since_boost >= self.mlfq.boost_interval:
            self._since_boost = 0.0
            allp = [p for q in self.queues for p in q]
            self.queues = [deque() for _ in self.mlfq.quanta]
            for p in allp:
                p.queue_level, p.quantum_used = 0, 0.0
                self.queues[0].append(p)

    def _emit(self, pid: int | None, start: float, end: float) -> None:
        if end <= start:
            return
        if self.timeline and self.timeline[-1].pid == pid and self.timeline[-1].end == start:
            self.timeline[-1].end = end
        else:
            self.timeline.append(Segment(pid, start, end))

    @property
    def done(self) -> bool:
        return self.done_count >= len(self.procs)

    def step_ticks(self, n: int) -> None:
        for _ in range(int(n)):
            if self.done:
                break
            self._tick()

    def _accrue_waiting(self) -> None:
        ready_now = (self.ready if self.algorithm != "mlfq"
                     else [p for q in self.queues for p in q])
        for p in ready_now:
            p.waiting += 1.0
        self.wait_ticks_window += len(ready_now)

    def _tick(self) -> None:
        self._admit()
        self._boost()
        start = self.time
        self.window_ticks += 1.0

        if self._cs_left > 0:  # context-switch overhead tick
            self._accrue_waiting()
            self._cs_left -= 1.0
            self.overhead_time += 1.0
            self._emit(-1, start, start + 1.0)
            self.time += 1.0
            self._since_boost += 1.0
            return

        if self.running is None:
            self.running = self._pick()
            if self.running is not None and self._last_pid not in (None, self.running.pid):
                self.context_switches += 1
                self.cs_window += 1
                if self.cs_overhead > 0:
                    # consume overhead starting this tick
                    self._accrue_waiting()
                    self._cs_left = self.cs_overhead
                    self._emit(-1, start, start + 1.0)
                    self.overhead_time += 1.0
                    self._cs_left -= 1.0
                    self.time += 1.0
                    self._since_boost += 1.0
                    return
            if self.running is not None and self.running.first_start is None:
                self.running.first_start = self.time
        # the running process does not accrue waiting this tick
        self._accrue_waiting()

        if self.running is None:  # idle
            self._emit(None, start, start + 1.0)
            self.time += 1.0
            self._since_boost += 1.0
            return

        p = self.running
        if p.first_start is None:
            p.first_start = self.time
        p.remaining -= 1.0
        p.quantum_used += 1.0
        self.busy_ticks += 1.0
        self._last_pid = p.pid
        self._emit(p.pid, start, start + 1.0)
        self.time += 1.0
        self._since_boost += 1.0

        if p.remaining <= 0:  # CPU burst finished
            if p.burst_idx < len(p.cpu_bursts) - 1:
                io = p.io_bursts[p.burst_idx] if p.burst_idx < len(p.io_bursts) else 0.0
                p.burst_idx += 1
                p.remaining = float(p.cpu_bursts[p.burst_idx])
                p.quantum_used = 0.0
                if io > 0:
                    p.io_done_at = self.time + io
                    self._io.append(p)
                else:
                    self._enqueue(p, level=p.queue_level if self.algorithm == "mlfq" else None)
            else:
                p.completion = self.time
                self.done_count += 1
                self.completed_window += 1
            self.running = None
        elif p.quantum_used >= self._quantum_for(p):
            if self.algorithm == "mlfq" and p.queue_level < len(self.queues) - 1:
                p.queue_level += 1  # demotion: used full quantum without blocking
            self._enqueue(p, level=p.queue_level if self.algorithm == "mlfq" else None)
            self.running = None

    def run(self) -> SimResult:
        guard = 0
        while not self.done:
            self._tick()
            guard += 1
            if guard > 10_000_000:
                raise RuntimeError("simulation did not terminate")
        # trailing overhead-only segments do not extend makespan
        return SimResult(processes=self.procs, timeline=self.timeline,
                         context_switches=self.context_switches,
                         makespan=self.time, overhead_time=self.overhead_time,
                         config_history=list(self.config_history))

    # -- observation helpers (RL) ---------------------------------------
    def ready_count(self) -> int:
        if self.algorithm == "mlfq":
            return sum(len(q) for q in self.queues) + (1 if self.running else 0)
        return len(self.ready) + (1 if self.running else 0)

    def reset_window_stats(self) -> None:
        self.wait_ticks_window = 0.0
        self.completed_window = 0
        self.cs_window = 0
        self.busy_ticks = 0.0
        self.window_ticks = 0.0
