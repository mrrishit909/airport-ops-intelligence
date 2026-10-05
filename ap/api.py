"""Public API (modular monolith). Blueprint services map to: ingestion-gateway (flight updates), flight-state-service,
prediction-service (arrival delay, connection risk), gate-optimizer (proposals a manager approves) and scenario-service
(replay against what really happened). Not built: turnaround-service, resource-optimizer, notification-service, the stream.

    uvicorn ap.api:app          python -m core.jobs ap.api      # the worker
"""
import datetime
import functools
import json
import uuid
from typing import Literal

import numpy as np
from fastapi import Depends, Header, Query
from fastapi.encoders import jsonable_encoder
from psycopg.types.json import Jsonb
from pydantic import BaseModel, Field

from core import audit, db, jobs
from core.app import Ctx, Problem, create_app, run

from . import engine, world

READ = {"ops:read", "jobs:read"}
CONTROL = READ | {"events:write", "disruptions:write", "gates:optimize", "scenarios:run"}
PERMISSIONS = {"viewer": READ, "controller": CONTROL, "manager": CONTROL | {"airport:load", "models:train", "plans:approve", "audit:read"}}
app = create_app("airport-ops-intelligence", PERMISSIONS)
auth = app.state.auth
IdemKey = Header(None, alias="Idempotency-Key")
DAY = datetime.datetime(2026, 10, 5, tzinfo=datetime.UTC)        # the simulated day; airport local time is written as UTC
DAY_SEED = 1


def ts(minute):
    return DAY + datetime.timedelta(minutes=int(minute))


def minute(t):
    return int((t - DAY).total_seconds() // 60)


def clock(m):
    return f"{int(m) // 60:02d}:{int(m) % 60:02d}"


def state(c):
    """Everything the models need, from storage: what is known about each flight, the published gate plan, the storm forecast."""
    ap = c.execute("SELECT * FROM airports").fetchone()
    if not ap:
        raise Problem(409, "no_airport", "load an airport first: POST /v1/airport:load")
    if not c.execute("SELECT 1 FROM delay_models LIMIT 1").fetchone():
        raise Problem(409, "no_model", "train the delay model first: POST /v1/models/delay:train")
    rows = c.execute("""SELECT f.ordinal, f.origin_delay_min, f.actual_arr_min, g.ordinal AS gate, a.locked
                          FROM flights f JOIN gate_assignments a ON a.flight_id = f.id JOIN gates g ON g.id = a.gate_id ORDER BY f.ordinal""").fetchall()
    storm = c.execute("SELECT payload FROM disruptions ORDER BY created_at DESC LIMIT 1").fetchone()
    hold = c.execute("SELECT hold_min FROM plans WHERE status = 'approved'").fetchone()
    st = {"airport": ap, "w": world.build(ap["model_seed"]), "now": ap["now_min"], "plan": [r["gate"] for r in rows], "locked": {r["ordinal"] for r in rows if r["locked"]},
          "known": np.array([np.nan if r["origin_delay_min"] is None else r["origin_delay_min"] for r in rows]), "arrived": np.array([r["actual_arr_min"] or 0 for r in rows]),
          "storm": storm and storm["payload"], "hold": hold["hold_min"] if hold else 0}
    st["model"] = engine.model(ap["model_seed"])            # ponytail: refitted from the generator's history on first use and cached per process (2 s). Store the artefact once training is slow.
    st["key"] = (ap["model_seed"], st["known"].tobytes(), st["arrived"].tobytes(), json.dumps(st["storm"], sort_keys=True), st["now"])
    st["pred"], st["lo"], st["hi"], st["est"] = _estimates(*st["key"])
    return st


@functools.lru_cache(maxsize=16)
def _estimates(seed, known, arrived, storm, now):
    """Estimates for one state of knowledge; the same state (nothing new has happened) is not recomputed on every read."""
    w, m = world.build(seed), engine.model(seed)
    known, arrived, storm = np.frombuffer(known), np.frombuffer(arrived, dtype=np.int64), json.loads(storm)
    pred, lo, hi = engine.predict(m, w, known, storm)
    return pred, lo, hi, np.where(arrived > 0, arrived, np.maximum(now + 1, w["sched_arr"] + pred)).round().astype(int)


def outlook(st, plan=None, hold=None):
    """The rest of the day if arrivals go as estimated: who waits for a gate, and each connection's chance of failing."""
    plan, hold = st["plan"] if plan is None else plan, st["hold"] if hold is None else hold
    return _outlook(*st["key"], tuple(plan), hold)


@functools.lru_cache(maxsize=16)
def _outlook(seed, known, arrived, storm, now, plan, hold):
    st = {"w": world.build(seed), "now": now, "model": engine.model(seed), "storm": json.loads(storm)}
    st["pred"], st["lo"], st["hi"], st["est"] = _estimates(seed, known, arrived, storm, now)
    w, est, m = st["w"], st["est"], st["model"]
    in_p, off_p, gate, held = engine.operate(w, est, now, list(plan), world.forecast(st["storm"]), hold)
    later = est > now
    risk = np.zeros(len(w["conns"]))
    for k, (a, b, _) in enumerate(w["conns"]):
        if later[a] or later[b]:
            spare = off_p[b] - world.DOOR - world.DEPLANE - w["walks"][gate[a], gate[b]] - in_p[a]
            risk[k] = engine.p_later_than(m, st["pred"][a], spare + est[a] - w["sched_arr"][a]) if later[a] else float(spare < 0)
    pax, delay = w["conn"][2], (est - w["sched_arr"])[later]
    return {"in": in_p, "off": off_p, "gate": gate, "held": held, "risk": risk,
            "summary": {"aircraft_still_to_arrive": int(later.sum()), "arriving_15_min_late_or_more": int((delay >= 15).sum()), "predicted_delay_min": int(delay.clip(0).sum()),
                        "aircraft_held_for_a_gate": int((in_p > est)[later].sum()), "gate_wait_min": int((in_p - est)[later].sum()),
                        "connecting_passengers_at_risk": int(pax[risk >= 0.5].sum()), "expected_missed_connections": round(float((pax * risk).sum()))}}


def refresh(c, st):
    """Write the current estimates back to the flights and keep the latest prediction for each."""
    t, w = st["airport"]["tenant_id"], st["w"]
    todo = [i for i in range(len(st["est"])) if not st["arrived"][i]]
    c.execute("CREATE TEMP TABLE _est (ordinal int, est_arr timestamptz, est_dep timestamptz) ON COMMIT DROP")
    db.copy(c, "_est", ["ordinal", "est_arr", "est_dep"], [(i, ts(st["est"][i]), ts(max(w["sched_dep"][i], st["est"][i] + w["turn"][i]))) for i in todo])
    c.execute("UPDATE flights f SET est_arr = e.est_arr, est_dep = e.est_dep FROM _est e WHERE f.ordinal = e.ordinal")
    c.execute("DROP TABLE _est")
    c.execute("DELETE FROM predictions WHERE target = 'arrival_delay_min'")
    band = st["model"]["metrics"]["inside_10_90_band"]
    db.load(c, "predictions", ["id", "tenant_id", "entity_type", "entity_id", "model_version", "target", "value", "confidence", "low", "high"],
            [(uuid.uuid7(), t, "flight", fid(t, i), engine.VERSION, "arrival_delay_min", round(float(st["pred"][i]), 1), band, round(float(st["lo"][i]), 1), round(float(st["hi"][i]), 1)) for i in todo])


def fid(t, i):
    return uuid.uuid5(t, f"flight-{i}")


def gid(t, k):
    return uuid.uuid5(t, f"gate-{k}")


def worker_ctx(job):
    return Ctx(job["tenant_id"], uuid.UUID(job["payload"]["actor_id"]), "worker", "system")


# ---------------------------------------------------------------- airport and schedule

class LoadIn(BaseModel):
    seed: int = Field(17, ge=0, le=10 ** 6, description="Which synthetic airport to generate")


@app.post("/v1/airport:load", status_code=202, tags=["airport"], summary="(+) Load a synthetic hub airport as it stands at 16:00: gates, the day's flights, the gate plan and connecting passengers")
def load_airport(body: LoadIn, ctx: Ctx = Depends(auth("airport:load")), idem: str | None = IdemKey):
    def work(c):
        if c.execute("SELECT 1 FROM airports LIMIT 1").fetchone():
            raise Problem(409, "airport_exists", "this tenant already has an airport")
        return 202, jobs.enqueue(c, ctx, "airport.load", body.model_dump())
    return run(ctx, idem, body, work)


@jobs.handler("airport.load")
def load_job(c, job):
    t, w, now = job["tenant_id"], world.build(job["payload"]["seed"]), world.NOW
    truth = world.day(w, DAY_SEED, world.real(world.DEMO_STORM))
    _, off0, _, _ = engine.operate(w, truth["arr"], now, w["plan"])
    aid = uuid.uuid5(t, "airport")
    c.execute("INSERT INTO airports (id, tenant_id, iata, icao, tz, name, model_seed, day, now_min) VALUES (%s,%s,'RBD','KRBD','UTC','Riverbend International',%s,%s,%s)", [aid, t, w["seed"], DAY.date(), now])
    db.load(c, "gates", ["id", "tenant_id", "airport_id", "ordinal", "code", "terminal", "max_wingspan_m", "intl_capable"], [(gid(t, g["index"]), t, aid, g["index"], g["code"], "AB"[g["pier"]], g["span"], g["intl"]) for g in w["gates"]])
    rows, status = [], {}
    for f in w["flights"]:
        i = f["index"]
        arrived, left = truth["arr"][i] <= now, truth["dep_origin"][i] <= now
        status[i] = ("departed" if off0[i] <= now else "arrived") if arrived else "airborne" if left else "scheduled"
        est = int(truth["arr"][i]) if arrived else f["sched_arr"]
        rows.append((fid(t, i), t, aid, i, f["flight_no"], f["tail_no"], f["origin"], "RBD", ts(f["sched_arr"]), ts(f["sched_dep"]), ts(est), ts(max(f["sched_dep"], est + f["turn"])), status[i],
                     world.SIZES[f["size"]]["span"], f["intl"], f["pax"], float(truth["origin_delay"][i]) if left else None, int(truth["arr"][i]) if arrived else None))
    db.load(c, "flights", ["id", "tenant_id", "airport_id", "ordinal", "flight_no", "tail_no", "origin", "destination", "sched_arr", "sched_dep", "est_arr", "est_dep", "status", "wingspan_m", "international", "passengers", "origin_delay_min", "actual_arr_min"], rows)
    db.load(c, "gate_assignments", ["id", "tenant_id", "flight_id", "gate_id", "start_ts", "end_ts", "source"], [(uuid.uuid7(), t, fid(t, f["index"]), gid(t, w["plan"][f["index"]]), ts(f["sched_arr"]), ts(f["sched_dep"]), "schedule") for f in w["flights"]])
    db.load(c, "connections", ["tenant_id", "inbound_id", "outbound_id", "passengers"], [(t, fid(t, a), fid(t, b), n) for a, b, n in w["conns"]])
    count = lambda s: sum(v == s for v in status.values())           # noqa: E731
    out = {"airport_id": aid, "airport": "Riverbend International (RBD)", "clock": clock(now), "gates": len(w["gates"]), "international_gates": sum(g["intl"] for g in w["gates"]), "flights": len(rows), "passengers": int(w["pax"].sum()),
           "connecting_passengers": int(w["conn"][2].sum()), "connections": len(w["conns"]), "departed": count("departed"), "at_a_gate": count("arrived"), "airborne": count("airborne"), "not_yet_left_origin": count("scheduled")}
    audit.record(c, worker_ctx(job), "airport.loaded", "airport", aid, {k: out[k] for k in ("gates", "flights", "connections")})
    return out


@app.post("/v1/models/delay:train", status_code=202, tags=["predictions"], summary="(+) Train the arrival-delay model on 72 past days, test it on 18 later ones, then estimate every flight still to arrive")
def train(ctx: Ctx = Depends(auth("models:train")), idem: str | None = IdemKey):
    def work(c):
        if not c.execute("SELECT 1 FROM airports LIMIT 1").fetchone():
            raise Problem(409, "no_airport", "load an airport first: POST /v1/airport:load")
        return 202, jobs.enqueue(c, ctx, "delay.train", {})
    return run(ctx, idem, {}, work)


@jobs.handler("delay.train")
def train_job(c, job):
    seed = c.execute("SELECT model_seed FROM airports").fetchone()["model_seed"]
    metrics, mid = engine.model(seed)["metrics"], uuid.uuid7()
    c.execute("INSERT INTO delay_models (id, tenant_id, version, metrics) VALUES (%s,%s,%s,%s)", [mid, job["tenant_id"], engine.VERSION, Jsonb(metrics)])
    st = state(c)
    refresh(c, st)
    audit.record(c, worker_ctx(job), "delay_model.trained", "delay_model", mid, {"mae_min": metrics["mae_min"], "mae_baseline_min": metrics["mae_baseline_min"]})
    return {"model_id": mid, "version": engine.VERSION, "method": "gradient-boosted trees on the delay leaving the origin, time of day, the storm forecast and the queue expected ahead of each aircraft", **metrics, "flights_estimated": int((st["arrived"] == 0).sum())}


@app.get("/v1/airports/{airport_id}/operations", tags=["airport"], summary="The airport now and for the next hours: each flight's estimate and gate, who is expected to wait for a gate, connections at risk")
def operations(airport_id: uuid.UUID, window: str = Query("6h", pattern=r"^([1-9]|1[0-2])h$"), ctx: Ctx = Depends(auth("ops:read"))):
    with db.tx(ctx.tenant_id) as c:
        st = state(c)
        if st["airport"]["id"] != airport_id:
            raise Problem(404, "airport_not_found")
        rows = c.execute("SELECT ordinal, flight_no, origin, status, passengers, international, sched_arr, sched_dep, est_arr, est_dep FROM flights ORDER BY ordinal").fetchall()
    w, now, o = st["w"], st["now"], outlook(st)
    end, flights = now + 60 * int(window[:-1]), []
    for r in rows:
        i = r["ordinal"]
        if o["off"][i] + world.BUFFER > now - 30 and o["in"][i] < end:
            flights.append({**r, "gate": w["gates"][o["gate"][i]]["code"], "gate_ordinal": int(o["gate"][i]), "size": w["flights"][i]["size"], "sched_in_min": int(w["sched_arr"][i]), "est_arr_min": int(st["est"][i]), "in_min": int(o["in"][i]), "off_min": int(o["off"][i]),
                            "wait_min": int(o["in"][i] - st["est"][i]) if st["est"][i] > now else 0, "held_min": int(o["held"][i]), "predicted_delay_min": None if st["arrived"][i] else round(float(st["pred"][i]), 1), "locked": i in st["locked"]})
    return jsonable_encoder({"airport_id": airport_id, "clock": clock(now), "now_min": now, "window_end_min": end, "storm": st["storm"], "hold_minutes": st["hold"], **o["summary"],
                             "gates": [{"ordinal": g["index"], "code": g["code"], "size": g["size"], "international": g["intl"]} for g in w["gates"]], "flights": flights})


# ---------------------------------------------------------------- ingestion-gateway

class FlightUpdate(BaseModel):
    flight_no: str = Field(pattern=r"^[A-Z]{2}\d{3,4}$")
    observed_at: datetime.datetime
    origin_delay_min: float = Field(ge=-30, le=600, description="Minutes late (negative: early) the aircraft left its origin, or the delay the origin has now posted")


@app.post("/v1/events/flight-update", tags=["events"], summary="A flight's delay at its origin becomes known: record it and re-estimate. An update older than the one already stored is ignored")
def flight_update(body: FlightUpdate, ctx: Ctx = Depends(auth("events:write")), idem: str | None = IdemKey):
    def work(c):
        if body.observed_at.tzinfo is None:
            raise Problem(422, "timezone_required", "observed_at must carry a UTC offset")
        st = state(c)
        f = c.execute("SELECT id, ordinal, status, observed_at, est_arr FROM flights WHERE flight_no = %s FOR UPDATE", [body.flight_no]).fetchone()
        if not f:
            raise Problem(404, "flight_not_found")
        if f["status"] in ("arrived", "departed"):
            raise Problem(409, "already_arrived", "this aircraft is already on the ground here")
        if f["observed_at"] and body.observed_at <= f["observed_at"]:
            return 200, {"applied": False, "flight_no": body.flight_no, "reason": "a later observation is already stored"}
        c.execute("UPDATE flights SET origin_delay_min = %s, observed_at = %s WHERE id = %s", [body.origin_delay_min, body.observed_at, f["id"]])
        new = state(c)
        refresh(c, new)
        i = f["ordinal"]
        audit.record(c, ctx, "flight.updated", "flight", f["id"], {"flight_no": body.flight_no, "origin_delay_min": body.origin_delay_min})
        return 200, {"applied": True, "flight_no": body.flight_no, "estimated_arrival_before": clock(st["est"][i]), "estimated_arrival": clock(new["est"][i]), "predicted_delay_min": round(float(new["pred"][i]), 1),
                     "low": round(float(new["lo"][i]), 1), "high": round(float(new["hi"][i]), 1)}
    return run(ctx, idem, body, work)


class DisruptionIn(BaseModel):
    type: Literal["thunderstorm"]
    severity: int = Field(ge=1, le=5)
    starts_at: datetime.datetime
    ends_at: datetime.datetime = Field(description="When the forecast says it ends")
    arrival_rate_per_hour: int = Field(ge=5, le=world.RATE, description="Landings an hour while it lasts")
    departure_rate_per_hour: int = Field(ge=5, le=world.DEP_RATE, description="Pushbacks an hour while it lasts")


@app.post("/v1/disruptions", status_code=201, tags=["events"], summary="(+) Announce a storm from its forecast. Every estimate is redone; returns the outlook before and after")
def disruption(body: DisruptionIn, ctx: Ctx = Depends(auth("disruptions:write")), idem: str | None = IdemKey):
    def work(c):
        if body.starts_at.tzinfo is None or body.ends_at.tzinfo is None:
            raise Problem(422, "timezone_required", "times must carry a UTC offset")
        a, b = minute(body.starts_at), minute(body.ends_at)
        if not 0 <= a < b <= 1440 or b - a > 360:
            raise Problem(422, "bad_window", "the storm must start and end on the operating day, in that order, and last at most 6 hours")
        st = state(c)
        before = outlook(st)["summary"]
        did, payload = uuid.uuid7(), {"start": a, "forecast_end": b, "rate": body.arrival_rate_per_hour, "dep_rate": body.departure_rate_per_hour}
        c.execute("INSERT INTO disruptions (id, tenant_id, airport_id, type, severity, starts_at, ends_at, payload) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)", [did, ctx.tenant_id, st["airport"]["id"], body.type, body.severity, body.starts_at, body.ends_at, Jsonb(payload)])
        new = state(c)
        refresh(c, new)
        o, w = outlook(new), new["w"]
        worst = sorted((i for i in range(len(new["est"])) if not new["arrived"][i]), key=lambda i: -new["pred"][i])[:6]
        audit.record(c, ctx, "disruption.announced", "disruption", did, {"type": body.type, "from": clock(a), "forecast_end": clock(b)})
        return 201, {"disruption_id": did, "type": body.type, "from": clock(a), "forecast_end": clock(b), "before": before, "after": o["summary"],
                     "most_delayed": [{"flight_id": fid(ctx.tenant_id, i), "flight_no": w["flights"][i]["flight_no"], "origin": w["flights"][i]["origin"], "scheduled": clock(w["sched_arr"][i]), "estimated": clock(new["est"][i]),
                                       "predicted_delay_min": round(float(new["pred"][i])), "low": round(float(new["lo"][i])), "high": round(float(new["hi"][i])), "passengers": int(w["pax"][i])} for i in worst]}
    return run(ctx, idem, body, work)


# ---------------------------------------------------------------- prediction-service

@app.get("/v1/flights/{flight_id}/connection-risk", tags=["predictions"], summary="For one inbound flight: each onward connection, the minutes to spare and the chance it is missed")
def connection_risk(flight_id: uuid.UUID, ctx: Ctx = Depends(auth("ops:read"))):
    with db.tx(ctx.tenant_id) as c:
        f = c.execute("SELECT ordinal, flight_no FROM flights WHERE id = %s", [flight_id]).fetchone()
        if not f:
            raise Problem(404, "flight_not_found")
        st = state(c)
    w, i, o = st["w"], f["ordinal"], outlook(st)
    groups = []
    for k, (a, b, n) in enumerate(w["conns"]):
        if a == i:
            walk = int(w["walks"][o["gate"][a], o["gate"][b]])
            groups.append({"to_flight": w["flights"][b]["flight_no"], "passengers": n, "from_gate": w["gates"][o["gate"][a]]["code"], "to_gate": w["gates"][o["gate"][b]]["code"], "walk_min": walk, "departs": clock(o["off"][b]),
                           "minutes_to_spare": int(o["off"][b] - world.DOOR - world.DEPLANE - walk - o["in"][a]), "chance_missed": round(float(o["risk"][k]), 3)})
    groups.sort(key=lambda g: -g["chance_missed"])
    return {"flight_id": flight_id, "flight_no": f["flight_no"], "scheduled": clock(w["sched_arr"][i]), "estimated": clock(st["est"][i]), "arrived": bool(st["arrived"][i]), "predicted_delay_min": round(float(st["pred"][i]), 1),
            "low": round(float(st["lo"][i]), 1), "high": round(float(st["hi"][i]), 1), "wait_for_gate_min": int(o["in"][i] - st["est"][i]), "connecting_passengers": sum(g["passengers"] for g in groups),
            "expected_missed": round(sum(g["passengers"] * g["chance_missed"] for g in groups), 1), "connections": groups, "model": engine.VERSION,
            "how": "chance that the real delay exceeds the minutes to spare, from how wrong predictions of this size were on the 18 test days"}


# ---------------------------------------------------------------- gate-optimizer

class Lock(BaseModel):
    flight_no: str = Field(pattern=r"^[A-Z]{2}\d{3,4}$")
    gate: str = Field(pattern=r"^[AB]\d{1,2}$")


class OptimiseIn(BaseModel):
    locks: list[Lock] = Field([], max_length=20, description="Flights that must use a given gate")
    hold_minutes: int = Field(15, ge=0, le=20, description="Longest a departure may be held for connecting passengers; 0 turns holding off")
    futures: int = Field(6, ge=2, le=12, description="How many sampled versions of the rest of the day the plan is judged on")
    tries: int = Field(800, ge=50, le=3000)


@app.post("/v1/optimizations/gates", status_code=202, tags=["gates"], summary="Propose a new gate plan (and hold rule) for aircraft still to arrive. Nothing changes until a manager approves it")
def optimise(body: OptimiseIn, ctx: Ctx = Depends(auth("gates:optimize")), idem: str | None = IdemKey):
    def work(c):
        st, locks = state(c), {}
        for lock in body.locks:
            f = c.execute("SELECT ordinal, status FROM flights WHERE flight_no = %s", [lock.flight_no]).fetchone()
            g = next((g for g in st["w"]["gates"] if g["code"] == lock.gate), None)
            if not f or not g:
                raise Problem(404, "flight_not_found" if not f else "gate_not_found", lock.flight_no if not f else lock.gate)
            if f["status"] in ("arrived", "departed"):
                raise Problem(409, "already_arrived", f"{lock.flight_no} is already on the ground")
            if not world.compatible(st["w"]["flights"][f["ordinal"]], g):
                raise Problem(422, "aircraft_does_not_fit", f"{lock.flight_no} cannot use {lock.gate}: the gate is too small or not international")
            locks[str(f["ordinal"])] = g["index"]
        return 202, jobs.enqueue(c, ctx, "gates.optimise", {"locks": locks, "hold": body.hold_minutes, "futures": body.futures, "tries": body.tries})
    return run(ctx, idem, body, work)


@jobs.handler("gates.optimise")
def optimise_job(c, job):
    p, st = job["payload"], state(c)
    w, now, plan, hold = st["w"], st["now"], st["plan"], p["hold"]
    locks = {**{i: plan[i] for i in st["locked"]}, **{int(i): k for i, k in p["locks"].items()}}
    known = np.where(st["arrived"] > 0, st["arrived"], 10 ** 6)
    sc = engine.futures(st["model"], w, st["pred"], known, now, st["storm"], p["futures"])
    new, info = engine.optimise(w, plan, sc, now, locks, hold, p["tries"])
    avg = lambda gates, h: {k: round(float(np.mean([engine.score(w, arr, now, gates, s, h)[k] for arr, s in sc])), 1) for k in ("cost_passenger_min", "aircraft_held_for_a_gate", "gate_wait_min", "missed_connections", "departures_held", "minutes_held")}      # noqa: E731
    base = np.array(w["plan"])
    changes = [{"flight_no": w["flights"][i]["flight_no"], "from": w["gates"][plan[i]]["code"], "to": w["gates"][new[i]]["code"], "passengers": int(w["pax"][i]), "estimated_arrival": clock(st["est"][i]), "locked": i in locks} for i in range(len(new)) if new[i] != plan[i]]
    summary = {"hold_minutes": hold, **info, "expected": {"published_plan": avg(plan, st["hold"]), "published_plan_with_holds": avg(plan, hold), "proposed": avg(new, hold)}, "gate_changes": len(changes), "changes": sorted(changes, key=lambda x: x["estimated_arrival"]),
               "changes_from_the_schedule": int((np.array(new) != base).sum()), "locks": [{"flight_no": w["flights"][i]["flight_no"], "gate": w["gates"][k]["code"]} for i, k in sorted(locks.items())],
               "how": f"each candidate plan is played through the gate simulator on {len(sc)} sampled versions of the rest of the day; a change is kept only if the average cost falls"}
    pid = uuid.uuid7()
    c.execute("INSERT INTO plans (id, tenant_id, airport_id, hold_min, gates, locks, summary, created_by) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
              [pid, job["tenant_id"], st["airport"]["id"], hold, Jsonb([int(g) for g in new]), Jsonb({str(i): int(k) for i, k in locks.items()}), Jsonb(summary), uuid.UUID(p["actor_id"])])
    audit.record(c, worker_ctx(job), "plan.proposed", "plan", pid, {"gate_changes": len(changes), "hold_minutes": hold, "locks": len(locks)})
    return {"plan_id": pid, "status": "proposed", **summary}


@app.get("/v1/plans/{plan_id}", tags=["gates"], summary="(+) One proposal and its status")
def plan(plan_id: uuid.UUID, ctx: Ctx = Depends(auth("ops:read"))):
    with db.tx(ctx.tenant_id) as c:
        p = c.execute("SELECT id AS plan_id, status, hold_min, summary, created_at, approved_at FROM plans WHERE id = %s", [plan_id]).fetchone()
    if not p:
        raise Problem(404, "plan_not_found")
    return jsonable_encoder(p)


@app.post("/v1/plans/{plan_id}/approve", tags=["gates"], summary="(+) A manager publishes a proposal: the gate changes and the hold rule take effect")
def approve(plan_id: uuid.UUID, ctx: Ctx = Depends(auth("plans:approve")), idem: str | None = IdemKey):
    def work(c):
        p = c.execute("SELECT * FROM plans WHERE id = %s FOR UPDATE", [plan_id]).fetchone()
        if not p:
            raise Problem(404, "plan_not_found")
        if p["status"] != "proposed":
            raise Problem(409, "not_proposed", f"this plan is already {p['status']}")
        if c.execute("SELECT 1 FROM plans WHERE approved_at > %s", [p["created_at"]]).fetchone():
            raise Problem(409, "stale_plan", "another plan was published after this one was computed; optimise again")
        st, t = state(c), ctx.tenant_id
        moved = [(gid(t, k), fid(t, i)) for i, k in enumerate(p["gates"]) if k != st["plan"][i]]
        db.many(c, "UPDATE gate_assignments SET gate_id = %s, source = 'optimizer' WHERE flight_id = %s", moved)
        db.many(c, "UPDATE gate_assignments SET locked = true WHERE flight_id = %s", [(fid(t, int(i)),) for i in p["locks"]])
        c.execute("UPDATE plans SET status = 'superseded' WHERE status = 'approved'")
        c.execute("UPDATE plans SET status = 'approved', approved_by = %s, approved_at = clock_timestamp() WHERE id = %s", [ctx.actor_id, plan_id])
        audit.record(c, ctx, "plan.approved", "plan", plan_id, {"gate_changes_published": len(moved), "hold_minutes": p["hold_min"], "proposed_by": str(p["created_by"])})
        return 200, {"plan_id": plan_id, "status": "approved", "gate_changes_published": len(moved), "hold_minutes": p["hold_min"], "locks": len(p["locks"])}
    return run(ctx, idem, {}, work)


# ---------------------------------------------------------------- scenario-service

class ScenarioIn(BaseModel):
    plan_ids: list[uuid.UUID] = Field([], max_length=5)


@app.post("/v1/scenarios", tags=["scenarios"], summary="Replay the rest of the day as it really went (the simulator knows) under the scheduled gates, the obvious rule and each given plan")
def scenarios(body: ScenarioIn, ctx: Ctx = Depends(auth("scenarios:run"))):
    with db.tx(ctx.tenant_id) as c:
        st = state(c)
        plans = [c.execute("SELECT id, status, hold_min, gates FROM plans WHERE id = %s", [i]).fetchone() for i in body.plan_ids]
    if None in plans:
        raise Problem(404, "plan_not_found")
    w, now, storm = st["w"], st["now"], world.real(world.DEMO_STORM)
    arr = world.day(w, DAY_SEED, storm)["arr"]
    run_ = lambda name, gates, hold=0, **kw: {"name": name, "hold_minutes": hold, **kw, **engine.score(w, arr, now, gates, storm, hold)}       # noqa: E731
    rows = [run_("Scheduled gates, no holds", w["plan"]), run_("First free gate on arrival", "first_free"), run_("Scheduled gates, hold for connections", w["plan"], 15)]
    rows += [run_(f"Plan {k + 1}: new gates" + (", hold for connections" if p["hold_min"] else ""), p["gates"], p["hold_min"], plan_id=p["id"], status=p["status"]) for k, p in enumerate(plans)]
    later = arr > now
    err = lambda x: round(float(np.abs(x - (arr - w["sched_arr"]))[later].mean()), 1)       # noqa: E731
    return jsonable_encoder({"what": "the simulator's hidden version of this day, which no model or plan was shown", "clock": clock(now), "aircraft_after_now": int(later.sum()),
                             "storm": {"from": clock(storm["start"]), "forecast_end": clock(storm["forecast_end"]), "actual_end": clock(storm["actual_end"])},
                             "delay_prediction": {"mean_error_min": err(st["pred"]), "departures_board_error_min": err(np.nan_to_num(st["known"])), "inside_10_90_band": round(float(((arr - w["sched_arr"] >= st["lo"]) & (arr - w["sched_arr"] <= st["hi"]))[later].mean()), 3)},
                             "rows": rows})
