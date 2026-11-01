-- Replica lag report. Run on each server; the guard in each block makes the wrong one return no rows:
--   psql postgresql://postgres:postgres@127.0.0.1:54410/postgres -f infra/postgres/lag.sql   (primary)
--   psql postgresql://postgres:postgres@127.0.0.1:54411/postgres -f infra/postgres/lag.sql   (replica)

-- Primary: is a standby streaming, and how far behind is its replay (bytes and seconds)?
select application_name,
       state,
       sync_state,
       pg_wal_lsn_diff(pg_current_wal_lsn(), replay_lsn) as replay_lag_bytes,
       extract(epoch from replay_lag) as replay_lag_s,
       state = 'streaming' as streaming
from pg_stat_replication
where not pg_is_in_recovery();

-- Replica: seconds since the last replayed transaction, and whether the WAL receiver is up.
-- (replay_age_s keeps growing on an idle primary even with zero lag; trust the primary's replay_lag_s.)
select pg_is_in_recovery() as in_recovery,
       extract(epoch from now() - pg_last_xact_replay_timestamp()) as replay_age_s,
       (select status = 'streaming' from pg_stat_wal_receiver) as streaming
where pg_is_in_recovery();
