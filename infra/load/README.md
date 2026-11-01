# Load test: concurrency sweep, primary only vs with replica

`locustfile.py` replays accepted golden questions (with their version and doc-type filters)
against `GET /search`; `sweep.py` runs it headless at 1/5/10/25/50 users and writes
`<out>/<label>.json` and `.md` (p50/p95/p99, RPS, error rate per level, saturation point).
Saturation = first level whose p95 is at least 2x the lowest level's p95, or whose error rate
exceeds 1%.

## Lab database pair

```bash
docker compose -f infra/compose.lab.yml --profile replica up -d --wait   # primary :54410, replica :54411
uv run python infra/postgres/schema.py                                   # database `sre`, from template1
```

Load the corpus into the primary (the replica replays it):
`uv run sre-ingest snapshot load --in data/snapshot/corpus.parquet --activate --dsn postgresql://postgres:postgres@127.0.0.1:54410/sre`.
Check replication any time with
`infra/postgres/lag.sql` (run it on both ports; the primary reports `replay_lag_s` and
`streaming`).

Tear down only these two services and their volumes (never `down -v`: that takes `testdb` too):

```bash
docker compose -f infra/compose.lab.yml --profile replica rm -sfv pg-primary pg-replica
docker volume rm sre-lab_pg-primary-data sre-lab_pg-replica-data
```

## Primary only

```bash
DATABASE_URL=postgresql://postgres:postgres@127.0.0.1:54410/sre \
  uv run python -c "from sre_core.api.main import run; run()"            # serves on 127.0.0.1:8000
uv run python infra/load/sweep.py --host http://127.0.0.1:8000 --label primary-only --duration 30
```

## With the replica

Same server, plus `DATABASE_REPLICA_URL`; retrieval reads go to the replica and fall back to the
primary if it is unreachable:

```bash
DATABASE_URL=postgresql://postgres:postgres@127.0.0.1:54410/sre \
DATABASE_REPLICA_URL=postgresql://postgres:postgres@127.0.0.1:54411/sre \
  uv run python -c "from sre_core.api.main import run; run()"
uv run python infra/load/sweep.py --host http://127.0.0.1:8000 --label with-replica --duration 30
```

Restart the server between runs, and give both runs the same corpus, embedder and hardware.
The two containers share one host, so the replica splits CPU with the primary rather than adding
capacity; read the result as read-routing overhead and isolation, not as scale-out.

## Options

Environment for `locustfile.py`: `LOAD_MODE` (hybrid default; `keyword` skips the embedder),
`LOAD_K`, `LOAD_THINK_S`, `LOAD_GOLDEN`, `LOAD_TOKEN`, and `LOAD_ANSWER_WEIGHT` (adds
`POST /answer` at that weight beside `/search`'s 10; it calls a model, so point the server at a
stub or local model). `sweep.py` takes `--levels`, `--duration`, `--spawn-rate`, `--out`.

Dry run (5 s, one user): `uv run python infra/load/sweep.py --levels 1 --duration 5 --label dry`.
Without a server, `uv run locust -f infra/load/locustfile.py --list` checks the file loads.
