"""Measure the delay model, the connection-risk calibration and the plans on airports and days the demo never uses.
Writes docs/evaluation.md.

    python -m ap.evaluate
"""
import pathlib

import numpy as np

from . import engine, world

OUT = pathlib.Path(__file__).resolve().parents[1] / "docs" / "evaluation.md"
SEEDS = (101, 102, 103)
DAYS = 3


def table(head, rows):
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def pct(x):
    return f"{100 * x:.1f}%"


def main():
    model_rows, plan_rows, calib = [], [], []
    for seed in SEEDS:
        w, m = world.build(seed), engine.model(seed)
        k = m["metrics"]
        model_rows.append([seed, k["tested_on_flights"], k["mae_min"], k["mae_baseline_min"], k["mae_in_storm_min"], k["mae_in_storm_baseline_min"], pct(k["inside_10_90_band"])])
        r = np.random.default_rng(seed + 500)
        for d in range(DAYS):
            storm = world.random_storm(r)
            storm["start"] = int(r.uniform(840, 1080))                                # an afternoon or evening storm, so there is an evening left to manage
            storm["forecast_end"], storm["actual_end"] = storm["start"] + (storm["forecast_end"] - storm["start"]) % 150 + 45, 0
            storm["actual_end"] = int(storm["start"] + (storm["forecast_end"] - storm["start"]) * r.lognormal(0.08, 0.2))
            now, R = storm["start"], world.real(storm)
            truth = world.day(w, 7000 + d, R)
            known = np.where(truth["dep_origin"] <= now, truth["origin_delay"], np.nan)
            pred, _, _ = engine.predict(m, w, known, storm)
            arrk = np.where(truth["arr"] <= now, truth["arr"], 10 ** 6)
            sc = engine.futures(m, w, pred, arrk, now, storm, 6)
            plan, _ = engine.optimise(w, w["plan"], sc, now, hold=15, tries=800)
            perfect, _ = engine.optimise(w, w["plan"], [(truth["arr"], R)], now, hold=15, tries=800)
            s = lambda assign, hold=0: engine.score(w, truth["arr"], now, assign, R, hold)        # noqa: E731
            plan_rows.append((seed, d, s(w["plan"]), s("first_free"), s(w["plan"], 15), s(plan, 15), s(perfect, 15)))
            # calibration of the connection risk under the scheduled gates: predicted chance against what happened
            est = np.where(truth["arr"] <= now, truth["arr"], np.maximum(now + 1, w["sched_arr"] + pred)).round().astype(int)
            in_p, off_p, gate, _ = engine.operate(w, est, now, w["plan"], world.forecast(storm))
            in_t, off_t, gate_t, _ = engine.operate(w, truth["arr"], now, w["plan"], R)
            fail = engine.missed(w, in_t, off_t, gate_t)
            for j, (a, b, n) in enumerate(w["conns"]):
                if truth["arr"][a] > now:
                    spare = off_p[b] - world.DOOR - world.DEPLANE - w["walks"][gate[a], gate[b]] - in_p[a]
                    calib.append((engine.p_later_than(m, pred[a], spare + est[a] - w["sched_arr"][a]), bool(fail[j]), n))
    calib = np.array(calib, dtype=float)
    bins = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.01]
    crow = []
    for lo, hi in zip(bins, bins[1:]):
        sel = (calib[:, 0] >= lo) & (calib[:, 0] < hi)
        if sel.any():
            crow.append([f"{lo:.1f} to {min(hi, 1):.1f}", int(sel.sum()), pct(np.average(calib[sel, 0], weights=calib[sel, 2])), pct(np.average(calib[sel, 1], weights=calib[sel, 2]))])
    names = ["scheduled gates", "first free gate", "scheduled gates + holds", "new gates + holds", "perfect knowledge + holds"]
    agg = lambda i, key: sum(r[2 + i][key] for r in plan_rows)          # noqa: E731
    doc = f"""# Evaluation

Everything here is measured on synthetic airports generated with seeds the demo does not use (`{SEEDS}`), on {DAYS} storm
days each, with the storm starting between 14:00 and 18:00 and the clock set to its start. Reproduce with `python -m ap.evaluate`
(about 5 minutes). The numbers say how well the methods do against the simulator's hidden version of each day. They are not a
claim about a real airport: real delays have causes this generator does not have, and real gate rules are richer.

## 1. How late will each aircraft be?

Trained on 72 past days of the same timetable, tested on 18 later days, each flight seen from a random 30 to 180 minutes before
its scheduled landing. The baseline is what a departures board shows: the delay the aircraft left its origin with, or on time
if it has not left yet.

{table(["airport (seed)", "test flights", "error, model", "error, board", "in a storm queue, model", "in a storm queue, board", "inside the 10–90% band"], model_rows)}

Errors are mean absolute errors in minutes. On two airports the model beats the board, most clearly in a storm queue, where the
board knows nothing about the runway. On the third it is worse, in the storm queue most of all, and I could not find the
reason: its storms were forecast about as well as the others'. A model that can be worse than the board on some airports needs
a check against the board before it is trusted; this slice has none. 80% of outcomes inside the 10–90% band would be perfect calibration.

## 2. Is the chance of a missed connection honest?

Every connecting group on the {len(plan_rows)} evaluation evenings whose inbound aircraft was still to arrive, under the scheduled gates:
the chance the system gave against the share that really missed (passenger-weighted). Where the two columns differ, the system
is too sure of itself at the top end: a "90%+" group misses less often than that, because the departure it is running for is
often late too.

{table(["predicted chance", "groups", "mean predicted", "really missed"], crow)}

## 3. Do the plans help?

The rest of each day replayed as it really went, counting aircraft that arrived after the storm began. "First free gate" is
the obvious rule: take the nearest free compatible gate when you arrive. "Holds" wait up to 15 minutes for connecting passengers
whose aircraft is already at a gate. "New gates" is this system's plan, judged on six sampled futures. "Perfect knowledge" plans
with the real arrival times, which nothing could have.

{table(["plan", "gate waiting (aircraft-min)", "missed connections", "departures held (min)", "gate changes", "cost (passenger-min)"],
       [[n, agg(i, "gate_wait_min"), agg(i, "missed_connections"), f"{agg(i, 'departures_held')} ({agg(i, 'minutes_held')})", agg(i, "gate_changes"), f"{agg(i, 'cost_passenger_min'):,}"] for i, n in enumerate(names)])}

Totals over the {len(plan_rows)} evenings. By evening (missed connections, scheduled gates / first free / scheduled + holds / new gates + holds / perfect):

{table(["airport", "day", "scheduled", "first free", "+ holds", "new gates + holds", "perfect + holds"], [[r[0], r[1], *[x["missed_connections"] for x in r[2:]]] for r in plan_rows])}

What the table says. Holding departures cuts missed connections by {pct(1 - agg(2, "missed_connections") / agg(0, "missed_connections"))} at a price of
{agg(2, "minutes_held")} minutes of held departures. The new gates lower the passenger-minute cost by {pct(1 - agg(3, "cost_passenger_min") / agg(0, "cost_passenger_min"))} (perfect knowledge
of arrival times: {pct(1 - agg(4, "cost_passenger_min") / agg(0, "cost_passenger_min"))}), mostly by keeping full aircraft off the taxiway, but they do not cut missed connections
further: the optimiser minimises the cost it is given, in which a missed connection is worth 90 passenger-minutes, and it
trades some of them for waiting time. In a storm the gates are full, so any assignment mostly decides who waits. The cost
column is dominated by departure delay the storm itself causes, which no plan can remove.
"""
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(doc)
    print(doc)


if __name__ == "__main__":
    main()
