#!/usr/bin/env bash
# Entrypoint of pg-replica. On the first start (empty data directory) seed it from the primary
# with pg_basebackup -R (writes standby.signal and primary_conninfo), then hand over to the
# image's own entrypoint, which starts Postgres as a hot standby. Later starts reuse the data.
set -Eeuo pipefail
PGDATA="${PGDATA:-/var/lib/postgresql/data}"
PRIMARY_HOST="${PRIMARY_HOST:-pg-primary}"

if [ ! -s "$PGDATA/PG_VERSION" ]; then
  echo "replica: waiting for $PRIMARY_HOST"
  until pg_isready -q -h "$PRIMARY_HOST" -p 5432 -U replicator; do sleep 1; done
  mkdir -p "$PGDATA"
  chown postgres:postgres "$PGDATA"
  chmod 0700 "$PGDATA"
  echo "replica: pg_basebackup from $PRIMARY_HOST"
  PGPASSWORD="${REPLICATION_PASSWORD:-replicator}" gosu postgres pg_basebackup \
    -h "$PRIMARY_HOST" -p 5432 -U replicator -D "$PGDATA" \
    -X stream -S replica1 -R -c fast -P
fi
exec docker-entrypoint.sh "$@"
