"""Delay prediction, connection risk, gate operations (the simulator of what a gate plan leads to) and the gate optimiser."""
import functools
import heapq

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from . import world

VERSION = "delay-gbm-1"
CHANGE_MIN = 3            # a gate change is priced as this many minutes for every passenger on the aircraft
MISS_MIN = 90             # a missed connection is priced as this many minutes for that passenger
EDGES = [5, 20, 45]       # predicted-delay buckets for the error distribution


def features(w, known_delay, storm):
    """One row per flight from what is known before it lands: its delay leaving the origin (NaN if it has not left),
    the forecast storm window, and how many other aircraft are expected ahead of it in the storm queue."""
    e = w["sched_arr"] - world.TAXI + np.nan_to_num(known_delay)                       # expected landing
    order = np.sort(e)
    density = np.searchsorted(order, e + 15) - np.searchsorted(order, e - 15)
    if storm:
        into = np.clip(e - storm["start"], -240, 480)
        queue = np.where(e >= storm["start"], np.searchsorted(order, e) - np.searchsorted(order, storm["start"]), 0)
        length, rate = storm["forecast_end"] - storm["start"], storm["rate"]
    else:
        into, queue, length, rate = np.full(len(e), -240.0), np.zeros(len(e)), 0, world.RATE
    return np.column_stack([known_delay, w["sched_arr"] / 60, into, queue, np.full(len(e), length), np.full(len(e), rate), density, into - length])


def dataset(w, days, r):
    """Each past flight seen from a random 30 to 180 minutes before its scheduled landing."""
    xs, ys, base = [], [], []
    for storm, d in days:
        lead = r.uniform(30, 180, len(d["arr"]))
        known = np.where(d["dep_origin"] <= w["sched_arr"] - world.TAXI - lead, d["origin_delay"], np.nan)
        xs.append(features(w, known, storm))
        ys.append(d["arr"] - w["sched_arr"])
        base.append(np.nan_to_num(known))
    return np.vstack(xs), np.concatenate(ys), np.concatenate(base)


@functools.lru_cache(maxsize=4)
def model(seed, days=90):
    """Train on the first 80% of past days, test on the rest. The baseline is what a departures board does: the delay
    the flight left its origin with, or on time if it has not left."""
    w = world.build(seed)
    hist, r = world.history(w, seed, days), np.random.default_rng(seed)
    cut = int(len(hist) * 0.8)
    x, y, _ = dataset(w, hist[:cut], r)
    xt, yt, bt = dataset(w, hist[cut:], r)
    fit = lambda **kw: HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_depth=3, random_state=0, **kw).fit(x, y)      # noqa: E731
    m = {"mean": fit(loss="absolute_error"), "lo": fit(loss="quantile", quantile=0.1), "hi": fit(loss="quantile", quantile=0.9)}        # the point estimate is a median: delays have a long tail
    pred = m["mean"].predict(xt)
    stormy = xt[:, 3] > 0
    m["overrun"] = np.array([(st["actual_end"] - st["start"]) / (st["forecast_end"] - st["start"]) for st, _ in hist[:cut] if st])       # how long storms really lasted against their forecast
    m["resid"] = [np.sort((yt - pred)[np.digitize(pred, EDGES) == k]) for k in range(len(EDGES) + 1)]
    m["metrics"] = {"trained_on_flights": len(y), "tested_on_flights": len(yt), "test_days": len(hist) - cut, "mae_min": round(float(np.abs(yt - pred).mean()), 2), "mae_baseline_min": round(float(np.abs(yt - bt).mean()), 2),
                    "mae_in_storm_min": round(float(np.abs(yt - pred)[stormy].mean()), 2), "mae_in_storm_baseline_min": round(float(np.abs(yt - bt)[stormy].mean()), 2), "flights_in_storm_queue": int(stormy.sum()),
                    "inside_10_90_band": round(float(((yt >= m["lo"].predict(xt)) & (yt <= m["hi"].predict(xt))).mean()), 3)}
    return m


def predict(m, w, known_delay, storm):
    x = features(w, known_delay, storm)
    return m["mean"].predict(x), m["lo"].predict(x), m["hi"].predict(x)


def p_later_than(m, pred, threshold):
    """Chance the real delay exceeds `threshold`, from how wrong predictions of this size were on the test days."""
    res = m["resid"][int(np.digitize(pred, EDGES))]
    return float(1 - np.searchsorted(res, threshold - pred, side="right") / len(res))


def late_prob(m, w, pred, arr, now):
    """-> f(flight, t): chance the aircraft reaches the gate area later than minute t. Certain for those already in."""
    def f(i, t):
        return float(arr[i] > t) if arr[i] <= now else p_later_than(m, pred[i], t - w["sched_arr"][i])
    return f


def operate(w, arr, now, assign, storm=None, hold=0):
    """Run the gates for the day, event by event. Aircraft that reached the gate area by `now` used the published plan.
    After that `assign` is a plan (list: flight -> gate; an aircraft whose gate is taken waits for it) or "first_free"
    (take a free compatible gate on arrival, the nearest to the one planned; if none, wait for the first that opens).
    In a storm pushbacks are metered, so aircraft sit on their gates longer.

    `hold` > 0 turns on holding for connections: when a departure after `now` is ready, look at the connecting passengers
    whose inbound aircraft is already at its gate, and wait up to `hold` minutes if the passengers saved (MISS_MIN each)
    outweigh the delay to everyone on board. Passengers on aircraft not yet at a gate are not waited for.
    -> in_block, off_block, gate, held (minutes each departure was held)"""
    n, gates = len(arr), w["gates"]
    in_, off, gate, held = np.zeros(n, int), np.zeros(n, int), np.array(w["plan"]), np.zeros(n, int)
    inbound = w["inbound"]
    busy, queue, holding, last = [False] * len(gates), [[] for _ in gates], [], -1e9
    events = [(int(arr[i]), 1, i) for i in range(n)]              # kinds: 0 a gate opens, 1 an arrival, 2 ready to push back
    heapq.heapify(events)

    def dock(i, g, t):
        busy[g], in_[i], gate[i] = True, t, g
        heapq.heappush(events, (int(max(w["sched_dep"][i], t + w["turn"][i])), 2, i))

    while events:
        t, kind, i = heapq.heappop(events)
        if kind == 1:
            if assign == "first_free" and t > now:
                home = gates[w["plan"][i]]
                open_ = [k for k, x in enumerate(gates) if not busy[k] and not queue[k] and world.compatible(w["flights"][i], x)]
                if not open_:
                    holding.append(i)
                    continue
                g = min(open_, key=lambda k: (k != home["index"], world.walk(home, gates[k])))
            else:
                g = (w["plan"] if t <= now else assign)[i]
            queue[g].append(i) if busy[g] else dock(i, g, t)
        elif kind == 2:
            if hold and t > now and not held[i]:
                need = sorted((in_[a] + world.DEPLANE + w["walks"][gate[a], gate[i]] + world.DOOR - t, p) for a, p in inbound[i] if in_[a])        # minutes until each group could be on board
                options, saved = [(0, 0)], 0
                for wait, p in need:
                    saved += p * (wait > 0)
                    if 0 < wait <= hold:
                        options.append((MISS_MIN * saved - int(w["pax"][i]) * wait, int(wait)))
                gain, h = max(options)
                if h:
                    held[i] = h
                    heapq.heappush(events, (t + h, 2, i))
                    continue
            at = max(t, last)
            rate = storm["dep_rate"] if storm and storm["start"] <= at < storm["end"] else world.DEP_RATE
            off[i] = last = int(np.ceil(max(t, last + 60 / rate)))
            heapq.heappush(events, (off[i] + world.BUFFER, 0, gate[i]))
        else:
            busy[i] = False
            nxt = queue[i].pop(0) if queue[i] else next((j for j in holding if world.compatible(w["flights"][j], gates[i])), None)
            if nxt is not None:
                if nxt in holding:
                    holding.remove(nxt)
                dock(nxt, i, t)
    return in_, off, gate, held


def missed(w, in_, off, gate):
    """Per connecting group: True if it fails. Off the inbound aircraft, across to the other gate, and the door is shut."""
    a, b, _ = w["conn"]
    return in_[a] + world.DEPLANE + w["walks"][gate[a], gate[b]] > off[b] - world.DOOR


def score(w, arr, now, assign, storm=None, hold=0):
    """What a way of assigning gates leads to, counted over aircraft arriving after `now`, in passenger-minutes and in plain counts."""
    in_, off, gate, held = operate(w, arr, now, assign, storm, hold)
    later = arr > now
    a, b, pax = w["conn"]
    lost = int(pax[missed(w, in_, off, gate) & (later[a] | later[b])].sum())
    wait, changed = (in_ - arr) * later, (gate != np.array(w["plan"])) & later
    return {"cost_passenger_min": int((wait * w["pax"]).sum() + (np.maximum(0, off - w["sched_dep"]) * w["pax"])[later].sum() + CHANGE_MIN * w["pax"][changed].sum() + MISS_MIN * lost),
            "aircraft_held_for_a_gate": int((wait > 0).sum()), "gate_wait_min": int(wait.sum()), "passenger_min_waiting": int((wait * w["pax"]).sum()),
            "departure_delay_min": int(np.maximum(0, off - w["sched_dep"])[later].sum()), "missed_connections": lost, "gate_changes": int(changed.sum()), "departures_held": int((held > 0).sum()), "minutes_held": int(held.sum())}


def futures(m, w, pred, arr, now, storm, k, seed=0):
    """k ways the rest of the day could go: each aircraft's delay drawn from how wrong predictions of that size have been,
    and the storm's length from how past storms ran against their forecast. -> [(arrivals, storm)]"""
    r, out = np.random.default_rng(seed), []
    bucket = np.digitize(pred, EDGES)
    for _ in range(k):
        err = np.array([r.choice(m["resid"][b]) for b in bucket])
        st = storm and {**storm, "end": int(storm["start"] + (storm["forecast_end"] - storm["start"]) * r.choice(m["overrun"]))}
        out.append((np.where(arr <= now, arr, np.maximum(now + 1, w["sched_arr"] + pred + err)).round().astype(int), st))
    return out


def optimise(w, plan, scenarios, now, locks=None, hold=0, tries=1500, seed=0):
    """Reassign aircraft still to arrive so that the average cost over `scenarios` (see futures) is as low as it can be
    made: waiting for a gate, leaving late, gate changes at CHANGE_MIN a head, missed connections at MISS_MIN a head.
    Each candidate plan is played through the same gate simulator that scores the real day, so pushback queues and
    knock-on waits are counted, not approximated. Search: move one aircraft to another gate it fits, or swap two;
    keep the change if the average cost falls. `locks` (flight -> gate) are fixed. -> (new plan, info)
    # ponytail: plain hill climbing with a fixed number of tries. If it stalls on bigger airports, add restarts or a solver for the start point."""
    locks, r = locks or {}, np.random.default_rng(seed)
    cur = list(plan)
    for i, k in locks.items():
        cur[i] = k
    cost = lambda p: float(np.mean([score(w, arr, now, p, st, hold)["cost_passenger_min"] for arr, st in scenarios]))       # noqa: E731
    todo = [i for i in range(len(plan)) if all(arr[i] > now for arr, _ in scenarios) and i not in locks]
    fits = {i: [k for k, g in enumerate(w["gates"]) if world.compatible(w["flights"][i], g)] for i in todo}
    start = best = cost(cur)
    moved = 0
    for _ in range(tries if todo else 0):
        new, i = list(cur), todo[r.integers(len(todo))]
        if r.random() < 0.3:
            j = todo[r.integers(len(todo))]
            if new[j] not in fits[i] or new[i] not in fits[j] or new[i] == new[j]:
                continue
            new[i], new[j] = new[j], new[i]
        else:
            new[i] = fits[i][r.integers(len(fits[i]))]
            if new[i] == cur[i]:
                continue
        c = cost(new)
        if c < best:
            cur, best, moved = new, c, moved + 1
    return cur, {"aircraft_decided": len(todo), "futures_considered": len(scenarios), "plans_tried": tries, "improvements_kept": moved, "expected_cost_before": round(start), "expected_cost_after": round(best)}
