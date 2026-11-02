-- Phase 3: agent threads, approvals, operations (idempotency + audit), escalation, feedback,
-- review labels, the side-work outbox and the semantic cache. Design doc p8-12; plan in
-- .orchestra/20261102-phase3-agent-mcp/plan.md.
--
-- Access model:
--   * core writes as the table owner (its pool connection) after checking authorization in
--     code, and reads a user's own rows `as_user()` so RLS applies exactly as for the REST API;
--   * users (authenticated) may read and write only their own threads, messages and feedback;
--   * mcp-ops connects and then `set role mcp_ops`: a NOLOGIN role with grants on operations,
--     incidents and pages only (and read on approvals), so a bug there cannot touch threads;
--   * LangGraph checkpoints live in schema agent_ckpt, which the REST API does not expose;
--     its tables are created by the checkpointer's own `setup()` (sre-core agent-setup).

-- ---------------------------------------------------------------------------------------
-- Identity helpers (JWT claims set by as_user / Supabase).
-- ---------------------------------------------------------------------------------------
create or replace function sre.jwt_sub() returns text
    language sql stable
    set search_path = ''
as $$
    select coalesce(auth.jwt() ->> 'sub', '')
$$;

-- viewer (read), operator (read, write with approval), admin (approves high-risk writes).
create or replace function sre.user_scopes() returns text[]
    language sql stable
    set search_path = ''
as $$
    select coalesce(
        array(select jsonb_array_elements_text(
            coalesce(auth.jwt() -> 'app_metadata' -> 'scopes', '[]'::jsonb))),
        '{}'::text[])
$$;
grant execute on function sre.jwt_sub(), sre.user_scopes() to anon, authenticated, service_role;

-- ---------------------------------------------------------------------------------------
-- Threads and messages (multi-turn). `next_turn` is assigned by the API with
-- `update ... set next_turn = next_turn + 1 returning`, never inside the graph (nodes re-run).
-- ---------------------------------------------------------------------------------------
create table public.threads (
    id          uuid primary key default gen_random_uuid(),
    owner_sub   text not null,
    title       text,
    status      text not null default 'active'
                check (status in ('active', 'awaiting_approval', 'escalated', 'closed')),
    next_turn   integer not null default 1 check (next_turn >= 1),
    summary     text,                       -- rolling summary that replaces old turns
    created_at  timestamptz not null default now(),
    updated_at  timestamptz not null default now()
);
create index threads_owner on public.threads (owner_sub, updated_at desc);

create table public.messages (
    id            bigint generated always as identity primary key,
    thread_id     uuid not null references public.threads (id) on delete cascade,
    turn          integer not null check (turn >= 1),
    role          text not null check (role in ('user', 'assistant', 'human')),
    content       text not null,
    route         text check (route in ('simple', 'complex', 'action', 'escalate', 'cache')),
    citations     jsonb not null default '[]'::jsonb,
    trace_run_id  text,                     -- LangSmith run, for feedback and review
    created_at    timestamptz not null default now(),
    unique (thread_id, turn, role)
);

-- ---------------------------------------------------------------------------------------
-- Approvals (15 minutes; decided once: `update ... where status = 'pending' and
-- expires_at > now() returning`).
-- ---------------------------------------------------------------------------------------
create table public.approvals (
    id               uuid primary key default gen_random_uuid(),
    thread_id        uuid not null references public.threads (id) on delete cascade,
    turn             integer not null,
    tool             text not null,
    args             jsonb not null,
    args_hash        text not null,
    namespace        text not null,
    target           text not null,
    risk             text not null check (risk in ('low', 'medium', 'high')),
    required_scope   text not null check (required_scope in ('operator', 'admin')),
    expected_effect  text not null,
    status           text not null default 'pending'
                     check (status in ('pending', 'approved', 'rejected', 'expired', 'executed')),
    requested_by     text not null,
    decided_by       text,
    decided_at       timestamptz,
    expires_at       timestamptz not null,
    created_at       timestamptz not null default now()
);
create index approvals_thread on public.approvals (thread_id, status);
create index approvals_pending on public.approvals (expires_at) where status = 'pending';

-- ---------------------------------------------------------------------------------------
-- Operations: every mcp-ops call (the audit log). Writes and escalations carry an idempotency
-- key = hash(thread, turn, tool, normalized args), unique, so a replay returns the stored row.
-- ---------------------------------------------------------------------------------------
create table public.operations (
    id               uuid primary key default gen_random_uuid(),
    idempotency_key  text not null,
    kind             text not null check (kind in ('read', 'write', 'escalation')),
    tool             text not null,
    args             jsonb not null,
    namespace        text,
    target           text,
    subject          text not null,         -- the end user (token sub), never the service
    scopes           text[] not null,
    thread_id        uuid,
    turn             integer,
    approval_id      uuid references public.approvals (id),
    status           text not null
                     check (status in ('pending', 'approved', 'running', 'succeeded', 'failed',
                                       'unknown', 'refused')),
    result           jsonb,
    error            text,
    task_id          text,
    latency_ms       real,
    created_at       timestamptz not null default now(),
    started_at       timestamptz,
    finished_at      timestamptz
);
create unique index operations_idempotency on public.operations (idempotency_key)
    where kind <> 'read';
create index operations_thread on public.operations (thread_id, turn);
create index operations_open on public.operations (status) where status in ('approved', 'running');

-- Escalation: an incident and a page row (no external pager in the lab).
create table public.incidents (
    id            uuid primary key default gen_random_uuid(),
    thread_id     uuid references public.threads (id) on delete set null,
    operation_id  uuid references public.operations (id),
    opened_by     text not null,
    severity      text not null check (severity in ('sev1', 'sev2', 'sev3', 'sev4')),
    title         text not null,
    summary       text not null,
    status        text not null default 'open' check (status in ('open', 'acknowledged', 'resolved')),
    created_at    timestamptz not null default now()
);

create table public.pages (
    id            bigint generated always as identity primary key,
    incident_id   uuid not null references public.incidents (id) on delete cascade,
    operation_id  uuid references public.operations (id),
    target        text not null,            -- the on-call rotation paged
    status        text not null default 'sent' check (status in ('sent', 'acked')),
    created_at    timestamptz not null default now()
);

-- ---------------------------------------------------------------------------------------
-- Side work (audit fan-out, feedback sync, online scoring, alerts): drained by the next request
-- or the monitor. `dedupe_key` makes enqueueing idempotent.
-- ---------------------------------------------------------------------------------------
create table public.outbox (
    id            bigint generated always as identity primary key,
    kind          text not null,
    dedupe_key    text not null unique,
    payload       jsonb not null,
    status        text not null default 'pending' check (status in ('pending', 'done', 'failed')),
    attempts      integer not null default 0,
    last_error    text,
    created_at    timestamptz not null default now(),
    processed_at  timestamptz
);
create index outbox_pending on public.outbox (created_at) where status = 'pending';

-- ---------------------------------------------------------------------------------------
-- Feedback from the UI, and labels mirrored from the review queues.
-- ---------------------------------------------------------------------------------------
create table public.feedback (
    id          bigint generated always as identity primary key,
    thread_id   uuid not null references public.threads (id) on delete cascade,
    turn        integer not null,
    owner_sub   text not null,
    rating      smallint not null check (rating in (-1, 1)),
    reason      text,
    created_at  timestamptz not null default now(),
    unique (thread_id, turn, owner_sub)
);

create table public.feedback_sources (
    id          bigint generated always as identity primary key,
    thread_id   uuid not null references public.threads (id) on delete cascade,
    turn        integer not null,
    chunk_id    bigint not null,
    owner_sub   text not null,
    useful      boolean not null,
    created_at  timestamptz not null default now(),
    unique (thread_id, turn, chunk_id, owner_sub)
);

create table public.ui_events (
    id          bigint generated always as identity primary key,
    thread_id   uuid not null references public.threads (id) on delete cascade,
    turn        integer not null,
    owner_sub   text not null,
    kind        text not null check (kind in ('regenerate', 'copy')),
    created_at  timestamptz not null default now()
);

create table public.review_labels (
    id           bigint generated always as identity primary key,
    source       text not null default 'langsmith',
    external_id  text not null,             -- e.g. the annotation queue item / feedback id
    run_id       text,
    thread_id    uuid references public.threads (id) on delete set null,
    turn         integer,
    reviewer     text,
    label        jsonb not null,
    created_at   timestamptz not null default now(),
    unique (source, external_id)
);

-- ---------------------------------------------------------------------------------------
-- Semantic cache. Reads always filter on scope_hash (sha256 of the sorted access groups),
-- corpus version and prompt version in SQL; literal_tokens guards exact-string questions.
-- Action and tool-output turns are never stored.
-- ---------------------------------------------------------------------------------------
create table public.semantic_cache (
    id                 bigint generated always as identity primary key,
    scope_hash         text not null,
    corpus_version_id  bigint not null references public.corpus_versions (id) on delete cascade,
    prompt_version     text not null,
    route              text not null check (route in ('simple', 'complex')),
    question           text not null,
    literal_tokens     text[] not null default '{}',
    embedding          extensions.halfvec(384) not null,
    answer             jsonb not null,
    hits               integer not null default 0,
    created_at         timestamptz not null default now(),
    expires_at         timestamptz not null
);
create index semantic_cache_scope on public.semantic_cache (scope_hash, corpus_version_id, prompt_version);
create index semantic_cache_emb on public.semantic_cache
    using hnsw (embedding extensions.halfvec_ip_ops) with (m = 16, ef_construction = 64);

-- ---------------------------------------------------------------------------------------
-- Row-level security.
-- ---------------------------------------------------------------------------------------
alter table public.threads          enable row level security;
alter table public.messages         enable row level security;
alter table public.approvals        enable row level security;
alter table public.operations       enable row level security;
alter table public.incidents        enable row level security;
alter table public.pages            enable row level security;
alter table public.outbox           enable row level security;
alter table public.feedback         enable row level security;
alter table public.feedback_sources enable row level security;
alter table public.ui_events        enable row level security;
alter table public.review_labels    enable row level security;
alter table public.semantic_cache   enable row level security;

-- Users: their own threads and what hangs off them.
create policy threads_own on public.threads
    for all to authenticated
    using (owner_sub = (select sre.jwt_sub()))
    with check (owner_sub = (select sre.jwt_sub()));

create policy messages_own on public.messages
    for select to authenticated
    using (exists (select 1 from public.threads t
                   where t.id = thread_id and t.owner_sub = (select sre.jwt_sub())));

-- Approvals: the thread's owner sees them; admins see the ones that need an admin.
create policy approvals_read on public.approvals
    for select to authenticated
    using (exists (select 1 from public.threads t
                   where t.id = thread_id and t.owner_sub = (select sre.jwt_sub()))
           or (required_scope = 'admin' and 'admin' = any ((select sre.user_scopes())::text[])));

create policy feedback_own on public.feedback
    for all to authenticated
    using (owner_sub = (select sre.jwt_sub()))
    with check (owner_sub = (select sre.jwt_sub())
                and exists (select 1 from public.threads t
                            where t.id = thread_id and t.owner_sub = (select sre.jwt_sub())));

create policy feedback_sources_own on public.feedback_sources
    for all to authenticated
    using (owner_sub = (select sre.jwt_sub()))
    with check (owner_sub = (select sre.jwt_sub())
                and exists (select 1 from public.threads t
                            where t.id = thread_id and t.owner_sub = (select sre.jwt_sub())));

create policy ui_events_own on public.ui_events
    for all to authenticated
    using (owner_sub = (select sre.jwt_sub()))
    with check (owner_sub = (select sre.jwt_sub())
                and exists (select 1 from public.threads t
                            where t.id = thread_id and t.owner_sub = (select sre.jwt_sub())));

grant select, insert, update on public.threads to authenticated;
grant select on public.messages, public.approvals to authenticated;
grant select, insert, update on public.feedback, public.feedback_sources to authenticated;
grant select, insert on public.ui_events to authenticated;
revoke all on public.operations, public.incidents, public.pages, public.outbox,
    public.review_labels, public.semantic_cache from anon, authenticated;
revoke all on public.threads, public.messages, public.approvals, public.feedback,
    public.feedback_sources, public.ui_events from anon;
grant all on public.threads, public.messages, public.approvals, public.operations,
    public.incidents, public.pages, public.outbox, public.feedback, public.feedback_sources,
    public.ui_events, public.review_labels, public.semantic_cache to service_role;

-- ---------------------------------------------------------------------------------------
-- mcp-ops: a NOLOGIN role it switches into (`set role mcp_ops`), so it cannot touch threads,
-- messages, feedback or the cache even through a bug.
-- ---------------------------------------------------------------------------------------
do $$
begin
    if not exists (select 1 from pg_roles where rolname = 'mcp_ops') then
        create role mcp_ops nologin noinherit;
    end if;
    -- The role mcp-ops connects as (the migration runner here) may switch into it. On
    -- Postgres 16+ a role's creator holds ADMIN on it but not SET, so grant SET explicitly.
    if not pg_has_role(current_user, 'mcp_ops', 'SET') then
        execute format('grant mcp_ops to %I with set true, inherit false', current_user);
    end if;
end
$$;
grant usage on schema public, sre to mcp_ops;
grant select, insert, update on public.operations, public.incidents, public.pages to mcp_ops;
grant select on public.approvals to mcp_ops;
grant usage on all sequences in schema public to mcp_ops;

create policy operations_mcp on public.operations
    for all to mcp_ops using (true) with check (true);
create policy incidents_mcp on public.incidents
    for all to mcp_ops using (true) with check (true);
create policy pages_mcp on public.pages
    for all to mcp_ops using (true) with check (true);
create policy approvals_mcp on public.approvals
    for select to mcp_ops using (true);

-- ---------------------------------------------------------------------------------------
-- LangGraph checkpoints: their own schema, not exposed by the REST API (only `public` is).
-- ---------------------------------------------------------------------------------------
create schema if not exists agent_ckpt;
revoke all on schema agent_ckpt from public, anon, authenticated;
grant usage, create on schema agent_ckpt to service_role;
