-- Domain model. Names follow the blueprint's schema; additions are marked (+).
-- Not built: turnaround_tasks, resources (ground crews and equipment). Positions are pier and slot numbers, not PostGIS geometry.

CREATE TABLE airports (id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants UNIQUE, iata char(3) NOT NULL, icao char(4) NOT NULL, tz text NOT NULL, name text NOT NULL,
  model_seed int NOT NULL, day date NOT NULL, now_min int NOT NULL);                                    -- (+) the simulated day and its clock, minutes after midnight
SELECT enable_tenant_rls('airports');

CREATE TABLE gates (id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, airport_id uuid NOT NULL REFERENCES airports ON DELETE CASCADE, ordinal int NOT NULL, code text NOT NULL, terminal text NOT NULL,
  max_wingspan_m numeric NOT NULL, intl_capable boolean NOT NULL, tow_required boolean NOT NULL DEFAULT false, UNIQUE (airport_id, ordinal));
SELECT enable_tenant_rls('gates');

CREATE TABLE flights (id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, airport_id uuid NOT NULL REFERENCES airports ON DELETE CASCADE, ordinal int NOT NULL, flight_no text NOT NULL, tail_no text NOT NULL,
  origin char(3) NOT NULL, destination char(3) NOT NULL, sched_arr timestamptz NOT NULL, sched_dep timestamptz NOT NULL, est_arr timestamptz NOT NULL, est_dep timestamptz NOT NULL,
  status text NOT NULL CHECK (status IN ('scheduled','airborne','arrived','departed')), wingspan_m numeric NOT NULL, international boolean NOT NULL, passengers int NOT NULL,   -- (+)
  origin_delay_min real, actual_arr_min int, observed_at timestamptz,                                    -- (+) what is known so far: delay leaving the origin, arrival at the gate area, time of the last update
  UNIQUE (airport_id, ordinal), UNIQUE (airport_id, flight_no));
SELECT enable_tenant_rls('flights');

CREATE TABLE gate_assignments (id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, flight_id uuid NOT NULL UNIQUE REFERENCES flights ON DELETE CASCADE, gate_id uuid NOT NULL REFERENCES gates,
  start_ts timestamptz NOT NULL, end_ts timestamptz NOT NULL, source text NOT NULL, locked boolean NOT NULL DEFAULT false, UNIQUE (gate_id, start_ts, flight_id));
SELECT enable_tenant_rls('gate_assignments');

CREATE TABLE connections (tenant_id uuid NOT NULL REFERENCES tenants, inbound_id uuid NOT NULL REFERENCES flights ON DELETE CASCADE, outbound_id uuid NOT NULL REFERENCES flights ON DELETE CASCADE, passengers int NOT NULL CHECK (passengers > 0),   -- (+)
  PRIMARY KEY (inbound_id, outbound_id));
SELECT enable_tenant_rls('connections');

CREATE TABLE disruptions (id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, airport_id uuid NOT NULL REFERENCES airports ON DELETE CASCADE, type text NOT NULL, severity int NOT NULL CHECK (severity BETWEEN 1 AND 5),
  starts_at timestamptz NOT NULL, ends_at timestamptz NOT NULL, payload jsonb NOT NULL, created_at timestamptz NOT NULL DEFAULT now(), CHECK (ends_at > starts_at));
SELECT enable_tenant_rls('disruptions');

CREATE TABLE predictions (id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, entity_type text NOT NULL, entity_id uuid NOT NULL, model_version text NOT NULL, target text NOT NULL, value numeric NOT NULL,
  confidence numeric, low numeric, high numeric, created_at timestamptz NOT NULL DEFAULT now(), UNIQUE (entity_id, target));      -- (+) low/high: the 10% and 90% estimates. Latest prediction only.
SELECT enable_tenant_rls('predictions');

CREATE TABLE delay_models (id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, version text NOT NULL, metrics jsonb NOT NULL, trained_at timestamptz NOT NULL DEFAULT now());   -- (+)
SELECT enable_tenant_rls('delay_models');

CREATE TABLE plans (id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, airport_id uuid NOT NULL REFERENCES airports ON DELETE CASCADE, status text NOT NULL DEFAULT 'proposed' CHECK (status IN ('proposed','approved','superseded')),   -- (+) an optimiser's proposal; nothing changes until a manager approves it
  hold_min int NOT NULL, gates jsonb NOT NULL, locks jsonb NOT NULL, summary jsonb NOT NULL, created_by uuid NOT NULL, approved_by uuid, created_at timestamptz NOT NULL DEFAULT now(), approved_at timestamptz);
SELECT enable_tenant_rls('plans');
