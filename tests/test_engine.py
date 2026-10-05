"""Unit tests for the simulator, the delay model and the gate engine: no database."""
import numpy as np
import pytest

from ap import engine, world

W = world.build()
S = world.DEMO_STORM
NOW = world.NOW


def test_world_is_deterministic_and_feasible():
    again = world.build.__wrapped__(17)
    assert again["plan"] == W["plan"] and again["conns"] == W["conns"] and len(W["flights"]) > 250
    free = {}
    for i, g in enumerate(W["plan"]):                                                      # the timetable's plan has no two aircraft on a gate at once
        f = W["flights"][i]
        assert world.compatible(f, W["gates"][g]) and free.get(g, -1) <= f["sched_arr"]
        free[g] = f["sched_dep"] + world.BUFFER
    a, b, n = W["conn"]
    assert (n > 0).all() and len(set(zip(a.tolist(), b.tolist()))) == len(a)


def test_a_storm_queues_arrivals_and_only_then():
    calm, stormy = world.day(W, 1), world.day(W, 1, world.real(S))
    extra = stormy["arr"] - calm["arr"]
    assert (extra >= 0).all() and extra[W["sched_arr"] < S["start"] - 60].max() == 0 and extra[(W["sched_arr"] > S["start"]) & (W["sched_arr"] < S["actual_end"])].mean() > 10
    assert np.array_equal(world.day(W, 1)["arr"], calm["arr"])                             # same seed, same day


def test_features_see_only_what_is_known():
    known = np.full(len(W["flights"]), np.nan)
    x = engine.features(W, known, None)
    assert x.shape == (len(W["flights"]), 8) and np.isnan(x[:, 0]).all() and (x[:, 3] == 0).all()
    x = engine.features(W, known, S)
    inside = (W["sched_arr"] - world.TAXI >= S["start"])
    assert x[inside, 3].max() > 10 and (x[~inside, 3] == 0).all()                           # the queue counts aircraft expected ahead in the storm


def test_model_beats_the_departures_board_and_is_roughly_calibrated():
    m = engine.model(17)
    k = m["metrics"]
    assert k["mae_min"] < k["mae_baseline_min"] and k["mae_in_storm_min"] < k["mae_in_storm_baseline_min"] and 0.65 < k["inside_10_90_band"] < 0.9
    assert engine.p_later_than(m, 10.0, -500) == 1.0 and engine.p_later_than(m, 10.0, 500) == 0.0 and 0.2 < engine.p_later_than(m, 10.0, 10.0) < 0.8


def test_gates_never_double_booked_and_sizes_respected():
    arr = world.day(W, 2, world.real(S))["arr"]
    for assign in (W["plan"], "first_free"):
        in_, off, gate, held = engine.operate(W, arr, NOW, assign, world.real(S), hold=15)
        assert (in_ >= arr).all() and (off >= in_ + W["turn"]).all() and (off >= W["sched_dep"]).all()
        for g in range(len(W["gates"])):
            mine = sorted((i for i in range(len(arr)) if gate[i] == g), key=lambda i: in_[i])
            for x, y in zip(mine, mine[1:]):
                assert in_[y] >= off[x] + world.BUFFER
            assert all(world.compatible(W["flights"][i], W["gates"][g]) for i in mine)
        assert (held <= 15).all() and (held[off <= NOW] == 0).all()                        # holds are decided at departure time, so only departures after now


def test_holding_saves_connections_and_metering_delays_departures():
    arr = world.day(W, 1, world.real(S))["arr"]
    plain, held = engine.score(W, arr, NOW, W["plan"], world.real(S)), engine.score(W, arr, NOW, W["plan"], world.real(S), hold=15)
    assert held["missed_connections"] < plain["missed_connections"] and held["departures_held"] > 0 and plain["departures_held"] == 0
    calm = engine.score(W, arr, NOW, W["plan"])
    assert calm["departure_delay_min"] < plain["departure_delay_min"] and calm["gate_wait_min"] < plain["gate_wait_min"]


def test_optimiser_keeps_locks_and_never_worsens_the_expected_cost():
    m = engine.model(17)
    d = world.day(W, 1, world.real(S))
    known = np.where(d["dep_origin"] <= NOW, d["origin_delay"], np.nan)
    pred, _, _ = engine.predict(m, W, known, S)
    sc = engine.futures(m, W, pred, np.where(d["arr"] <= NOW, d["arr"], 10 ** 6), NOW, S, 3)
    i = next(i for i in range(len(W["flights"])) if d["arr"][i] > NOW + 60 and W["flights"][i]["size"] == "S")
    k = next(k for k, g in enumerate(W["gates"]) if g["size"] == "L" and k != W["plan"][i])
    new, info = engine.optimise(W, W["plan"], sc, NOW, {i: k}, hold=15, tries=60)
    assert new[i] == k and info["expected_cost_after"] <= info["expected_cost_before"] and all(new[j] == W["plan"][j] for j in range(len(new)) if d["arr"][j] <= NOW)
    assert all(world.compatible(W["flights"][j], W["gates"][g]) for j, g in enumerate(new))
    assert len(sc) == 3 and all(np.array_equal(a[d["arr"] <= NOW], d["arr"][d["arr"] <= NOW]) for a, _ in sc) and all(s["end"] >= S["start"] for _, s in sc)


def test_missed_rule():
    in_, off, gate = np.zeros(len(W["flights"]), int), np.full(len(W["flights"]), 10 ** 6), np.array(W["plan"])
    assert not engine.missed(W, in_, off, gate).any()                                      # with departures far away nobody misses
    assert engine.missed(W, in_, np.zeros(len(W["flights"]), int), gate).all()             # with every door already shut everybody does
