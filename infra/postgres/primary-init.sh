#!/usr/bin/env bash
# Runs once, on the primary's first start (mounted into /docker-entrypoint-initdb.d): the
# replication role and the physical slot the standby streams from. Lab credentials.
set -Eeuo pipefail
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname postgres --no-psqlrc <<SQL
create role replicator with login replication password '${REPLICATION_PASSWORD:-replicator}';
select pg_create_physical_replication_slot('replica1');
SQL
