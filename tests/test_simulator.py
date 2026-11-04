"""Manually verifiable scheduling tests (hand-computed expectations)."""
import pytest

from learnsched.process import make_processes
from learnsched.simulator import Simulator, MLFQConfig


def run(algo, specs, **kw):
    return Simulator(make_processes(specs), algorithm=algo, **kw).run()


def by_pid(result):
    return {p.pid: p for p in result.processes}


def test_fcfs_waiting_and_turnaround():
    # P1 arrives 0 runs 0-5; P2 arrives 1 runs 5-8
    r = run("fcfs", [(0, 5), (1, 3)])
    p = by_pid(r)
    assert p[1].waiting == 0 and p[1].turnaround == 5
    assert p[2].waiting == 4 and p[2].turnaround == 7
    assert p[2].response == 4
    assert r.makespan == 8


def test_fcfs_metrics():
    m = run("fcfs", [(0, 5), (1, 3)]).metrics()
    assert m["avg_waiting"] == pytest.approx(2.0)
    assert m["avg_turnaround"] == pytest.approx(6.0)
    assert m["throughput"] == pytest.approx(2 / 8)


def test_sjf_orders_shortest_first_at_same_arrival():
    r = run("sjf", [(0, 6), (0, 2), (0, 4)])
    order = [s.pid for s in r.timeline]
    assert order == [2, 3, 1]
    p = by_pid(r)
    assert (p[2].waiting, p[3].waiting, p[1].waiting) == (0, 2, 6)


def test_sjf_is_nonpreemptive():
    # P2 (short) arrives while P1 runs: P1 still finishes first
    r = run("sjf", [(0, 6), (1, 2)])
    p = by_pid(r)
    assert p[1].completion == 6 and p[2].waiting == 5


def test_rr_alternates_each_quantum():
    r = run("rr", [(0, 4), (0, 4)], rr_quantum=2)
    assert [s.pid for s in r.timeline] == [1, 2, 1, 2]
    assert all(s.end - s.start == 2 for s in r.timeline)
    p = by_pid(r)
    assert p[1].completion == 6 and p[2].completion == 8


def test_rr_quantum_larger_than_burst_matches_fcfs():
    r = run("rr", [(0, 3), (0, 3)], rr_quantum=10)
    assert [s.pid for s in r.timeline] == [1, 2]


def test_io_process_blocks_and_returns():
    # P1: cpu2, io3, cpu2 -> completes at 7; P2 cpu4 fills 2-6
    r = run("fcfs", [(0, [2, 2], [3]), (0, 4)])
    p = by_pid(r)
    assert p[2].completion == 6
    assert p[1].completion == 8  # runs 0-2, io till 5 (cpu busy), resumes 6-8
    assert p[1].turnaround == 8


def test_mlfq_demotes_long_process():
    # Quantum 2 at Q0: two interactive short jobs finish from Q0 while
    # the long job sinks; long job must still complete (no starvation).
    cfg = MLFQConfig(quanta=(2, 4, 8, float("inf")), boost_interval=0)
    r = Simulator(make_processes([(0, 30), (0, 2), (0, 2)]),
                  algorithm="mlfq", mlfq=cfg).run()
    p = by_pid(r)
    assert p[2].completion == 4 and p[3].completion == 6
    assert p[1].completion is not None and p[1].completion == 34


def test_mlfq_boost_prevents_starvation():
    # Continuous stream of short jobs; long job boosted back to Q0.
    specs = [(0, 40)] + [(i * 3, 2) for i in range(12)]
    cfg = MLFQConfig(quanta=(2, 4, 8, float("inf")), boost_interval=10)
    r = Simulator(make_processes(specs), algorithm="mlfq", mlfq=cfg).run()
    assert all(p.completion is not None for p in r.processes)


def test_context_switch_overhead_increases_makespan():
    base = run("rr", [(0, 4), (0, 4)], rr_quantum=2)
    with_cs = run("rr", [(0, 4), (0, 4)], rr_quantum=2, cs_overhead=1)
    assert with_cs.context_switches == 3  # 1->2, 2->1, 1->2
    assert with_cs.makespan == base.makespan + 3
    assert with_cs.overhead_time == 3


def test_idle_gap_when_no_process_arrived():
    r = run("fcfs", [(5, 3)])
    assert r.timeline[0].pid is None and r.makespan == 8


def test_waiting_equals_turnaround_minus_service_for_no_io():
    r = run("rr", [(0, 5), (2, 3), (4, 4)], rr_quantum=2)
    for p in r.processes:
        assert p.waiting == pytest.approx(p.turnaround - p.total_cpu)
