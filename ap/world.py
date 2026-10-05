"""An invented hub airport: 28 gates on two piers, a day of about 280 aircraft turns arriving in six waves, the passengers
connecting between them, and what really happens on a given day (late departures from origin, a runway queue, a storm).
Everything is a pure function of a seed. Times are minutes after midnight."""
import functools

import numpy as np

SIZES = {"S": {"turn": 30, "span": 29, "seats": 76}, "M": {"turn": 40, "span": 36, "seats": 174}, "L": {"turn": 70, "span": 65, "seats": 310}}
ORDER = "SML"
PIERS, PER_PIER = 2, 14
BUFFER = 8                       # minutes a gate stays blocked after an aircraft pushes back
TAXI = 8                         # runway to gate
DEPLANE, DOOR = 12, 8            # minutes to get off the inbound aircraft; the outbound door closes this long before departure
RATE, DEP_RATE = 46, 50          # landings and take-offs an hour with both runways
BANKS = list(range(390, 1300, 70))
ORIGINS = ["ATL", "BOS", "DEN", "DFW", "IAH", "JFK", "LAX", "MIA", "ORD", "PHX", "SEA", "SFO", "YYZ", "MEX", "LHR", "CDG", "FRA", "NRT"]
INTL = {"YYZ", "MEX", "LHR", "CDG", "FRA", "NRT"}
DEMO_STORM = {"start": 960, "forecast_end": 1050, "actual_end": 1075, "rate": 18, "dep_rate": 12}       # the forecast says 90 minutes; it lasts 115
NOW = 960


def make_gates():
    out = []
    for k in range(PIERS * PER_PIER):
        pier, slot = divmod(k, PER_PIER)
        size = "L" if slot < 4 else "M" if slot < 11 else "S"
        out.append({"index": k, "code": f"{'AB'[pier]}{slot + 1}", "pier": pier, "slot": slot, "size": size, "intl": pier == 1, "span": SIZES[size]["span"]})
    return out


def walk(a, b):
    """Minutes on foot between two gates."""
    return round(3 + 0.7 * abs(a["slot"] - b["slot"]) + (9 if a["pier"] != b["pier"] else 0))


def compatible(f, g):
    return ORDER.index(g["size"]) >= ORDER.index(f["size"]) and (g["intl"] or not f["intl"])


@functools.lru_cache(maxsize=8)
def build(seed=17, candidates=460):
    r = np.random.default_rng(seed)
    gates = make_gates()
    times = np.sort(np.where(r.random(candidates) < 0.8, r.choice(BANKS, candidates) + r.normal(0, 22, candidates), r.uniform(330, 1290, candidates))).round().astype(int)
    flights, plan, free = [], [], [0] * len(gates)
    for t in times:
        origin = ORIGINS[r.integers(len(ORIGINS))]
        intl = origin in INTL
        size = r.choice(list(ORDER), p=[0.05, 0.45, 0.5] if intl else [0.3, 0.62, 0.08])
        f = {"size": str(size), "intl": intl, "origin": origin, "sched_arr": int(t), "turn": SIZES[size]["turn"], "pax": int(SIZES[size]["seats"] * r.uniform(0.7, 0.96)),
             "block": int(r.uniform(420, 640) if intl else r.uniform(55, 290))}
        f["sched_dep"] = f["sched_arr"] + f["turn"] + int(r.uniform(10, 45))
        ok = [g for g in gates if compatible(f, g) and free[g["index"]] + 5 <= t]
        if not ok:
            continue                                       # the timetable only holds flights that have a gate
        g = min(ok, key=lambda g: (ORDER.index(g["size"]) - ORDER.index(f["size"]), g["intl"] and not intl, -free[g["index"]]))       # the gate freed most recently: planners pack the convenient gates
        free[g["index"]] = f["sched_dep"] + BUFFER
        f.update(index=len(flights), flight_no=f"RB{100 + len(flights)}", tail_no=f"N{300 + len(flights)}RB")
        flights.append(f)
        plan.append(g["index"])
    pairs = {}                                             # (inbound, outbound) -> passengers: scheduled with at least 5 minutes to spare
    for a in flights:
        left = int(a["pax"] * r.uniform(0.25, 0.5))
        options = [b for b in flights if b is not a and a["sched_arr"] + DEPLANE + walk(gates[plan[a["index"]]], gates[plan[b["index"]]]) + 5 <= b["sched_dep"] - DOOR <= a["sched_arr"] + 170]
        while left > 0 and options:
            n, b = min(left, int(r.integers(1, 10))), options[r.integers(len(options))]["index"]
            pairs[a["index"], b] = pairs.get((a["index"], b), 0) + n
            left -= n
    conns = [(a, b, n) for (a, b), n in pairs.items()]
    inbound = [[] for _ in flights]
    for a, b, k in conns:
        inbound[b].append((a, k))
    walks = np.array([[walk(a, b) for b in gates] for a in gates])
    return {"seed": seed, "walks": walks, "conn": np.array(conns).T, "inbound": inbound, "gates": gates, "flights": flights, "plan": plan, "conns": conns, "sched_arr": np.array([f["sched_arr"] for f in flights]), "sched_dep": np.array([f["sched_dep"] for f in flights]),
            "turn": np.array([f["turn"] for f in flights]), "pax": np.array([f["pax"] for f in flights]), "block": np.array([f["block"] for f in flights])}


def day(w, seed, storm=None):
    """What really happens. -> origin_delay (minutes late leaving the origin), dep_origin (when that became known) and arr
    (when the aircraft reaches the gate area), after queueing for the runway. `storm` as real(storm) gives it."""
    r = np.random.default_rng([w["seed"], seed])
    n = len(w["flights"])
    late = np.where(r.random(n) < 0.65, r.uniform(-5, 5, n), np.minimum(r.exponential(18, n), 120)).round()
    eta = w["sched_arr"] - TAXI + late + r.normal(0, 3, n)
    land, last = np.empty(n), -1e9
    for i in np.argsort(eta):
        t = max(eta[i], last)
        rate = storm["rate"] if storm and storm["start"] <= t < storm["end"] else RATE
        land[i] = last = max(eta[i], last + 60 / rate)
    return {"origin_delay": late, "dep_origin": w["sched_arr"] - TAXI - w["block"] + late, "arr": (land + TAXI).round().astype(int)}


def real(storm):
    """The storm as it turns out; forecast(storm) is the storm as announced."""
    return storm and {**storm, "end": storm["actual_end"]}


def forecast(storm):
    return storm and {**storm, "end": storm["forecast_end"]}


def random_storm(r):
    start, length = int(r.uniform(420, 1200)), r.uniform(45, 150)
    return {"start": start, "forecast_end": int(start + length), "actual_end": int(start + length * r.lognormal(0.08, 0.2)), "rate": int(r.uniform(14, 24)), "dep_rate": int(r.uniform(9, 16))}


def history(w, seed, days=90):
    """Past days of the same timetable, half of them with a storm. -> [(storm or None, day)]"""
    r = np.random.default_rng([w["seed"], seed, 7])
    out = []
    for d in range(days):
        storm = random_storm(r) if r.random() < 0.5 else None
        out.append((storm, day(w, seed * 1000 + d, real(storm))))
    return out
