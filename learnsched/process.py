"""Process model for LearnSched."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Process:
    """A process with alternating CPU / I/O bursts.

    cpu_bursts[i] is followed by io_bursts[i] (if present).
    len(io_bursts) must be len(cpu_bursts) - 1 or len(cpu_bursts).
    """

    pid: int
    arrival: float
    cpu_bursts: list[float]
    io_bursts: list[float] = field(default_factory=list)
    priority: int = 0

    # runtime state (reset by the simulator via fresh copies)
    burst_idx: int = 0
    remaining: float = field(init=False)
    first_start: float | None = None
    completion: float | None = None
    waiting: float = 0.0  # time spent in ready queue
    io_done_at: float | None = None
    queue_level: int = 0
    quantum_used: float = 0.0
    cpu_received: float = 0.0  # CPU ticks actually served (for fairness so far)

    def __post_init__(self) -> None:
        if not self.cpu_bursts or any(b <= 0 for b in self.cpu_bursts):
            raise ValueError("cpu_bursts must be non-empty and positive")
        if self.arrival < 0:
            raise ValueError("arrival must be >= 0")
        if len(self.io_bursts) not in (0, len(self.cpu_bursts) - 1, len(self.cpu_bursts)):
            raise ValueError("io_bursts length inconsistent with cpu_bursts")
        self.remaining = float(self.cpu_bursts[0])

    @property
    def total_cpu(self) -> float:
        return float(sum(self.cpu_bursts))

    @property
    def total_io(self) -> float:
        return float(sum(self.io_bursts))

    @property
    def turnaround(self) -> float | None:
        return None if self.completion is None else self.completion - self.arrival

    @property
    def response(self) -> float | None:
        return None if self.first_start is None else self.first_start - self.arrival

    def fresh_copy(self) -> "Process":
        return Process(pid=self.pid, arrival=float(self.arrival),
                       cpu_bursts=list(self.cpu_bursts),
                       io_bursts=list(self.io_bursts), priority=self.priority)

    def to_dict(self) -> dict:
        return {"pid": self.pid, "arrival": self.arrival,
                "cpu_bursts": ";".join(str(b) for b in self.cpu_bursts),
                "io_bursts": ";".join(str(b) for b in self.io_bursts),
                "priority": self.priority}

    @staticmethod
    def from_dict(d: dict) -> "Process":
        parse = lambda s: [float(x) for x in str(s).split(";") if str(x).strip() != ""]
        return Process(pid=int(d["pid"]), arrival=float(d["arrival"]),
                       cpu_bursts=parse(d["cpu_bursts"]),
                       io_bursts=parse(d.get("io_bursts", "")),
                       priority=int(d.get("priority", 0)))


def make_processes(specs: list[tuple]) -> list[Process]:
    """Helper for tests: specs are (arrival, cpu_burst) or (arrival, [cpu...], [io...])."""
    out = []
    for i, s in enumerate(specs, start=1):
        if len(s) == 2:
            arrival, burst = s
            out.append(Process(pid=i, arrival=float(arrival), cpu_bursts=[float(burst)]))
        else:
            arrival, cpus, ios = s
            out.append(Process(pid=i, arrival=float(arrival),
                               cpu_bursts=[float(x) for x in cpus],
                               io_bursts=[float(x) for x in ios]))
    return out
