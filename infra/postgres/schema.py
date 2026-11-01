"""Build the schema on the lab primary, as CI does: a fresh database from template1, then
BOOTSTRAP_SQL and every migration. The standby replays it.

    uv run python infra/postgres/schema.py [--admin-dsn DSN] [--dbname sre]

Defaults to the `pg-primary` service (127.0.0.1:54410). Idempotent: an existing database is kept
and the script exits without re-applying.
"""

from __future__ import annotations

import argparse

import psycopg
from psycopg import sql

from sre_shared.testing.pg import BOOTSTRAP_SQL, apply_sql_files, migration_files, with_dbname

DEFAULT_ADMIN_DSN = "postgresql://postgres:postgres@127.0.0.1:54410/postgres"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--admin-dsn", default=DEFAULT_ADMIN_DSN)
    ap.add_argument("--dbname", default="sre")
    args = ap.parse_args()

    with psycopg.connect(args.admin_dsn, autocommit=True) as admin:
        exists = admin.execute(
            "select 1 from pg_database where datname = %s", [args.dbname]
        ).fetchone()
        if exists:
            print(f"database {args.dbname} exists; nothing to do")
            return
        admin.execute(
            sql.SQL("create database {} template template1").format(sql.Identifier(args.dbname))
        )
    apply_sql_files(with_dbname(args.admin_dsn, args.dbname), [BOOTSTRAP_SQL, *migration_files()])
    print(f"database {args.dbname}: bootstrap + {len(migration_files())} migrations applied")


if __name__ == "__main__":
    main()
