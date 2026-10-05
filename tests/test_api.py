"""Integration, end-to-end and security tests against a real Postgres."""
import uuid

import pytest

from core import db, jobs, scenario
from ap import world

from .conftest import bearer

MANAGER, CONTROLLER, VIEWER, OTHER = bearer("hub", "manager"), bearer("hub", "controller"), bearer("hub", "viewer"), bearer("other", "manager")
ZERO = "00000000-0000-0000-0000-000000000000"
STORM = {"type": "thunderstorm", "severity": 4, "starts_at": "2026-10-05T16:00:00Z", "ends_at": "2026-10-05T17:30:00Z", "arrival_rate_per_hour": 18, "departure_rate_per_hour": 12}


def test_nothing_works_before_the_airport_and_model_exist(client):
    assert client.get(f"/v1/airports/{ZERO}/operations", headers=VIEWER).json()["code"] == "no_airport" and client.post("/v1/models/delay:train", headers=MANAGER).json()["code"] == "no_airport"
    assert client.post("/v1/disruptions", headers=CONTROLLER, json=STORM).json()["code"] == "no_airport" and client.post("/v1/scenarios", headers=CONTROLLER, json={}).json()["code"] == "no_airport"


@pytest.fixture(scope="module")
def demo(client):
    results, _ = scenario.run(client, drain=jobs.drain)
    return results


def one(sql, params=(), tenant=None):
    with db.tx(tenant) as c:
        return c.execute(sql, params).fetchone()


def test_demo_journey(demo):
    load = demo["load"]["result"]
    assert load["flights"] == len(world.build()["flights"]) == one("SELECT count(*) AS n FROM flights")["n"] and load["gates"] == 28 and load["departed"] + load["at_a_gate"] + load["airborne"] + load["not_yet_left_origin"] == load["flights"]
    train = demo["train"]["result"]
    assert train["mae_min"] < train["mae_baseline_min"] and train["flights_estimated"] == load["airborne"] + load["not_yet_left_origin"] == one("SELECT count(*) AS n FROM predictions")["n"]
    ops = demo["ops"]
    assert ops["storm"] is None and ops["aircraft_still_to_arrive"] == train["flights_estimated"] and len(ops["gates"]) == 28 and all(f["in_min"] >= f["est_arr_min"] for f in ops["flights"])
    assert demo["upd"]["applied"] and demo["upd"]["estimated_arrival"] > demo["upd"]["estimated_arrival_before"] and not demo["stale"]["applied"]
    storm = demo["storm"]
    assert storm["after"]["aircraft_held_for_a_gate"] > storm["before"]["aircraft_held_for_a_gate"] and storm["after"]["predicted_delay_min"] > storm["before"]["predicted_delay_min"] and storm["most_delayed"][0]["flight_no"] == "RB351"
    assert demo["ops2"]["storm"]["start"] == 960 and demo["ops2"]["gate_wait_min"] == storm["after"]["gate_wait_min"]
    risk = demo["risk"]
    assert risk["flight_no"] == "RB351" and risk["connections"] and all(0 <= g["chance_missed"] <= 1 for g in risk["connections"]) and risk["connections"][0]["chance_missed"] >= risk["connections"][-1]["chance_missed"]
    assert risk["expected_missed"] == pytest.approx(sum(g["passengers"] * g["chance_missed"] for g in risk["connections"]), abs=0.1)
    plan = demo["plan"]["result"]
    e = plan["expected"]
    assert plan["status"] == "proposed" and e["proposed"]["cost_passenger_min"] <= e["published_plan_with_holds"]["cost_passenger_min"] <= e["published_plan"]["cost_passenger_min"] and plan["gate_changes"] == len(plan["changes"])
    assert demo["deny"]["code"] == "forbidden" and demo["ok"]["status"] == "approved" and demo["ok"]["gate_changes_published"] == plan["gate_changes"]
    plan2 = demo["plan2"]["result"]
    assert plan2["locks"] == [{"flight_no": "RB340", "gate": "B1"}] and demo["ok2"]["locks"] == 1 and one("SELECT g.code FROM gate_assignments a JOIN gates g ON g.id = a.gate_id JOIN flights f ON f.id = a.flight_id WHERE f.flight_no = 'RB340'")["code"] == "B1"
    assert one("SELECT status FROM plans WHERE id = %s", [plan["plan_id"]])["status"] == "superseded" and demo["ops3"]["hold_minutes"] == 15
    rep = demo["replay"]
    names = [r["name"] for r in rep["rows"]]
    assert len(rep["rows"]) == 5 and names[0].startswith("Scheduled") and rep["rows"][2]["missed_connections"] < rep["rows"][0]["missed_connections"] and rep["rows"][3]["missed_connections"] < rep["rows"][0]["missed_connections"]
    assert rep["delay_prediction"]["mean_error_min"] < rep["delay_prediction"]["departures_board_error_min"] and rep["storm"]["actual_end"] == "17:55"
    assert demo["audit"]["chain_valid"] and {"airport.loaded", "delay_model.trained", "flight.updated", "disruption.announced", "plan.proposed", "plan.approved"} <= {x["action"] for x in demo["audit"]["events"]}


def test_flight_update_validation(client, demo):
    post = lambda **kw: client.post("/v1/events/flight-update", headers=CONTROLLER, json={"flight_no": "RB371", "observed_at": "2026-10-05T16:05:00Z", "origin_delay_min": 20, **kw})     # noqa: E731
    assert post(flight_no="RB999").json()["code"] == "flight_not_found" and post(observed_at="2026-10-05T16:05:00").json()["code"] == "timezone_required" and post(origin_delay_min=5000).status_code == 422
    landed = one("SELECT flight_no FROM flights WHERE status = 'departed' LIMIT 1")["flight_no"]
    assert post(flight_no=landed).json()["code"] == "already_arrived"
    assert post().json()["applied"] and post(observed_at="2026-10-05T16:04:00Z").json()["applied"] is False


def test_disruption_validation(client, demo):
    bad = lambda **kw: client.post("/v1/disruptions", headers=CONTROLLER, json={**STORM, **kw}).json()["code"]     # noqa: E731
    assert bad(ends_at="2026-10-05T15:00:00Z") == "bad_window" and bad(ends_at="2026-10-06T01:00:00Z") == "bad_window" and bad(starts_at="2026-10-05T16:00:00") == "timezone_required"
    assert client.post("/v1/disruptions", headers=CONTROLLER, json={**STORM, "arrival_rate_per_hour": 99}).status_code == 422 and client.post("/v1/disruptions", headers=CONTROLLER, json={**STORM, "type": "snow"}).status_code == 422


def test_locks_are_checked(client, demo):
    post = lambda locks: client.post("/v1/optimizations/gates", headers=CONTROLLER, json={"locks": locks, "tries": 50}).json()     # noqa: E731
    assert post([{"flight_no": "RB340", "gate": "A14"}])["code"] == "aircraft_does_not_fit"          # a wide-body from abroad on a small domestic gate
    assert post([{"flight_no": "RB999", "gate": "B1"}])["code"] == "flight_not_found" and post([{"flight_no": "RB340", "gate": "C9"}]) and client.post("/v1/optimizations/gates", headers=CONTROLLER, json={"locks": [{"flight_no": "RB340", "gate": "Z1"}]}).status_code == 422
    landed = one("SELECT flight_no FROM flights WHERE status = 'departed' LIMIT 1")["flight_no"]
    assert post([{"flight_no": landed, "gate": "A1"}])["code"] == "already_arrived"


def test_approval_rules(client, demo):
    pid = demo["plan"]["result"]["plan_id"]
    assert client.post(f"/v1/plans/{pid}/approve", headers=MANAGER).json()["code"] == "not_proposed" and client.post(f"/v1/plans/{ZERO}/approve", headers=MANAGER).status_code == 404
    r = client.post("/v1/optimizations/gates", headers=CONTROLLER, json={"tries": 50, "futures": 2})
    jobs.drain()
    stale = client.get(r.json()["status_url"], headers=CONTROLLER).json()["result"]["plan_id"]
    r = client.post("/v1/optimizations/gates", headers=CONTROLLER, json={"tries": 50, "futures": 2})
    jobs.drain()
    fresh = client.get(r.json()["status_url"], headers=CONTROLLER).json()["result"]["plan_id"]
    assert client.post(f"/v1/plans/{fresh}/approve", headers=MANAGER).status_code == 200
    assert client.post(f"/v1/plans/{stale}/approve", headers=MANAGER).json()["code"] == "stale_plan"          # computed against a plan that has since been replaced
    assert client.get(f"/v1/plans/{fresh}", headers=VIEWER).json()["status"] == "approved"
    assert client.post("/v1/scenarios", headers=CONTROLLER, json={"plan_ids": [ZERO]}).status_code == 404


def test_roles(client, demo):
    assert client.get(f"/v1/airports/{ZERO}/operations").status_code == 401 and client.get("/v1/plans/" + ZERO, headers={"Authorization": "Bearer nope"}).status_code == 401
    for path, payload in (("/v1/disruptions", STORM), ("/v1/optimizations/gates", {}), ("/v1/scenarios", {}), ("/v1/events/flight-update", {"flight_no": "RB371", "observed_at": "2026-10-05T16:05:00Z", "origin_delay_min": 1})):
        assert client.post(path, headers=VIEWER, json=payload).status_code == 403, path
    for path in ("/v1/airport:load", "/v1/models/delay:train", f"/v1/plans/{ZERO}/approve"):
        assert client.post(path, headers=CONTROLLER, json={}).status_code == 403, path
    assert client.get("/v1/audit", headers=CONTROLLER).status_code == 403 and client.get("/v1/audit", headers=MANAGER).status_code == 200


def test_another_airport_sees_nothing(client, demo, seeded):
    aid = demo["load"]["result"]["airport_id"]
    assert client.get(f"/v1/airports/{aid}/operations", headers=OTHER).json()["code"] == "no_airport" and client.get(f"/v1/plans/{demo['plan']['result']['plan_id']}", headers=OTHER).status_code == 404
    assert client.get(f"/v1/flights/{demo['storm']['most_delayed'][0]['flight_id']}/connection-risk", headers=OTHER).status_code == 404
    for table in ("airports", "gates", "flights", "gate_assignments", "connections", "disruptions", "predictions", "delay_models", "plans"):
        assert one(f"SELECT count(*) AS n FROM {table}", tenant=seeded["other"])["n"] == 0 < one(f"SELECT count(*) AS n FROM {table}", tenant=seeded["hub"])["n"], table
    with pytest.raises(Exception, match="row-level security"), db.tx(seeded["other"]) as c:
        c.execute("INSERT INTO airports (id, tenant_id, iata, icao, tz, name, model_seed, day, now_min) VALUES (%s,%s,'XXX','KXXX','UTC','x',1,'2026-10-05',0)", [uuid.uuid4(), seeded["hub"]])


def test_idempotent_writes(client, demo):
    again = client.post("/v1/disruptions", headers={**CONTROLLER, "Idempotency-Key": "demo-1:storm"}, json=STORM)
    assert again.status_code == 201 and again.headers["Idempotent-Replay"] == "true" and again.json()["disruption_id"] == demo["storm"]["disruption_id"]
    assert client.post("/v1/disruptions", headers={**CONTROLLER, "Idempotency-Key": "demo-1:storm"}, json={**STORM, "severity": 5}).json()["code"] == "idempotency_key_reused"
    assert client.post("/v1/airport:load", headers=MANAGER, json={"seed": 17}).json()["code"] == "airport_exists"
