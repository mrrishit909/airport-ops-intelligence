export DATABASE_URL ?= postgresql://app:app-dev-only@127.0.0.1:55211/app
PY = ./venv/bin/python
BASE = http://127.0.0.1:8211

bootstrap:      ## local virtualenv + database container
	python3 -m venv venv && ./venv/bin/pip install -q -r requirements-dev.txt
	docker compose up -d db
seed:           ## migrate and load the synthetic world (no-op if already seeded)
	$(PY) -m ap.seed --if-empty
dev:            ## API with reload on :8211, jobs run by a local worker
	($(PY) -m core.jobs ap.api &) && $(PY) -m uvicorn ap.api:app --reload --port 8211
test:
	$(PY) -m pytest --cov=ap --cov=core --cov-report=term-missing
demo:           ## the full stack in Docker, then the demo scenario against it
	docker compose up --build -d --wait && $(PY) -m core.scenario $(BASE)
load-test:      ## run `make demo` first
	$(PY) -m core.loadtest $(BASE) hub-viewer-demo 32 10 "GET /v1/airports/7133ee34-50ad-510a-b202-d6db8aeb871e/operations?window=6h" "GET /v1/airports/7133ee34-50ad-510a-b202-d6db8aeb871e/operations?window=2h" "GET /v1/flights/42082f3a-7c94-5317-8c11-fc88fa60c997/connection-risk"
reset:          ## drop all data (volume included)
	docker compose down -v
.PHONY: bootstrap seed dev test demo load-test reset
