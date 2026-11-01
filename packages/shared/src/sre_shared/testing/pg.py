"""Disposable Postgres databases for integration tests.

One Postgres server (the `supabase/postgres` image the local stack runs) serves every test
session. Each session gets its own database, cloned from a template that already has the
migrations applied, and drops it at the end. Parallel workers therefore share one container
but never share state.

Where the server comes from:
- `SRE_TEST_DATABASE_URL` set (lab shared test container, CI service container): use it as the
  admin connection.
- otherwise: start a container with testcontainers for this session.
"""

from __future__ import annotations

import hashlib
import os
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import psycopg
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo

from sre_shared.settings import REPO_ROOT

# Keep in sync with the image `npx supabase start` runs (docker ps), so tests and lab match.
SUPABASE_PG_IMAGE = os.environ.get(
    "SRE_TEST_PG_IMAGE", "public.ecr.aws/supabase/postgres:17.11.0.006"
)
TEMPLATE_DB = "sre_template"
MIGRATIONS_DIR = REPO_ROOT / "supabase" / "migrations"
BOOTSTRAP_SQL = Path(__file__).with_name("bootstrap.sql")
_LOCK_KEY = 7_311_024  # advisory lock serializing template builds and clones


def migration_files() -> list[Path]:
    return sorted(MIGRATIONS_DIR.glob("*.sql"))


def migrations_fingerprint() -> str:
    h = hashlib.sha256(BOOTSTRAP_SQL.read_bytes())
    for f in migration_files():
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


def with_dbname(dsn: str, dbname: str) -> str:
    params = conninfo_to_dict(dsn)
    params["dbname"] = dbname
    return make_conninfo(**params)  # type: ignore[arg-type]


def apply_sql_files(dsn: str, files: list[Path]) -> None:
    with psycopg.connect(dsn, autocommit=True) as conn:
        for f in files:
            conn.execute(f.read_text(encoding="utf-8"))  # type: ignore[call-overload,unused-ignore]


def _template_fingerprint(admin: psycopg.Connection) -> str | None:
    row = admin.execute(
        "select shobj_description(oid, 'pg_database') from pg_database where datname = %s",
        [TEMPLATE_DB],
    ).fetchone()
    return None if row is None else (row[0] or "")


def ensure_template(admin_dsn: str) -> None:
    """Build `sre_template` (bootstrap + migrations) unless it is current. Caller holds the lock."""
    fp = migrations_fingerprint()
    with psycopg.connect(admin_dsn, autocommit=True) as admin:
        current = _template_fingerprint(admin)
        if current == fp:
            return
        if current is not None:
            admin.execute(
                sql.SQL("alter database {} is_template false").format(sql.Identifier(TEMPLATE_DB))
            )
            admin.execute(
                sql.SQL("drop database {} with (force)").format(sql.Identifier(TEMPLATE_DB))
            )
        admin.execute(
            sql.SQL("create database {} template template1").format(sql.Identifier(TEMPLATE_DB))
        )
    template_dsn = with_dbname(admin_dsn, TEMPLATE_DB)
    apply_sql_files(template_dsn, [BOOTSTRAP_SQL, *migration_files()])
    with psycopg.connect(admin_dsn, autocommit=True) as admin:
        admin.execute(
            sql.SQL("comment on database {} is {}").format(
                sql.Identifier(TEMPLATE_DB), sql.Literal(fp)
            )
        )
        admin.execute(
            sql.SQL("alter database {} is_template true").format(sql.Identifier(TEMPLATE_DB))
        )


@contextmanager
def session_database(admin_dsn: str, prefix: str = "sre_test") -> Iterator[str]:
    """Yield the DSN of a fresh database with migrations applied; drop it afterwards."""
    name = f"{prefix}_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    with psycopg.connect(admin_dsn, autocommit=True) as admin:
        admin.execute("select pg_advisory_lock(%s)", [_LOCK_KEY])
        try:
            ensure_template(admin_dsn)
            for attempt in range(10):
                try:
                    admin.execute(
                        sql.SQL("create database {} template {}").format(
                            sql.Identifier(name), sql.Identifier(TEMPLATE_DB)
                        )
                    )
                    break
                except psycopg.errors.ObjectInUse:
                    time.sleep(0.5 * (attempt + 1))
            else:
                raise RuntimeError(f"could not clone {TEMPLATE_DB}: still in use")
        finally:
            admin.execute("select pg_advisory_unlock(%s)", [_LOCK_KEY])
    try:
        yield with_dbname(admin_dsn, name)
    finally:
        with psycopg.connect(admin_dsn, autocommit=True) as admin:
            admin.execute(
                sql.SQL("drop database if exists {} with (force)").format(sql.Identifier(name))
            )


@contextmanager
def admin_server() -> Iterator[str]:
    """Admin DSN of the Postgres server tests should use."""
    env = os.environ.get("SRE_TEST_DATABASE_URL")
    if env:
        yield env
        return
    from testcontainers.core.container import DockerContainer
    from testcontainers.core.waiting_utils import wait_for_logs

    password = "postgres"  # noqa: S105 - disposable local container
    container = (
        DockerContainer(SUPABASE_PG_IMAGE)
        .with_env("POSTGRES_PASSWORD", password)
        .with_exposed_ports(5432)
    )
    with container:
        wait_for_logs(container, "database system is ready to accept connections", timeout=120)
        host = container.get_container_host_ip()
        port = container.get_exposed_port(5432)
        dsn = make_conninfo(
            host=host, port=port, user="postgres", password=password, dbname="postgres"
        )
        _wait_until_connectable(dsn)
        yield dsn


def _wait_until_connectable(dsn: str, timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    while True:
        try:
            with psycopg.connect(dsn, connect_timeout=3) as conn:
                conn.execute("select 1")
                return
        except psycopg.OperationalError:
            if time.monotonic() > deadline:
                raise
            time.sleep(1.0)
