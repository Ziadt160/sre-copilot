-- SRE Copilot core schema: corpus versions, documents, chunks (dense + full-text), embedding
-- cache, eval results. RLS restricts reads by access group; writes go through the table owner
-- (ingest, evals) only.

create extension if not exists vector with schema extensions;

create schema if not exists sre;
comment on schema sre is 'SRE Copilot helper functions';

-- ---------------------------------------------------------------------------------------
-- Corpus versions (blue/green): building -> ready -> active -> retired.
-- ---------------------------------------------------------------------------------------
create table public.corpus_versions (
    id               bigint generated always as identity primary key,
    label            text        not null,
    status           text        not null default 'building'
                     check (status in ('building', 'ready', 'active', 'retired')),
    sources          jsonb       not null default '[]'::jsonb,  -- [{k8s_version, branch, commit_sha, url_base}]
    chunker_version  text        not null,
    embedding_models text[]      not null default '{}',
    stats            jsonb       not null default '{}'::jsonb,
    created_at       timestamptz not null default now(),
    activated_at     timestamptz,
    retired_at       timestamptz
);

-- At most one active corpus version.
create unique index corpus_versions_one_active
    on public.corpus_versions ((true)) where status = 'active';

-- ---------------------------------------------------------------------------------------
-- Documents: one parsed page per Kubernetes version.
-- ---------------------------------------------------------------------------------------
create table public.documents (
    id                bigint generated always as identity primary key,
    corpus_version_id bigint      not null references public.corpus_versions (id) on delete cascade,
    project           text        not null,
    k8s_version       text        not null,
    source_path       text        not null,
    url               text        not null,
    title             text        not null,
    doc_type          text        not null
                      check (doc_type in ('concept', 'task', 'reference', 'tutorial', 'other')),
    language          text        not null default 'en',
    last_modified     timestamptz,
    access_group      text        not null default 'public',
    content_hash      text        not null,
    parsed_text       text        not null,
    unique (corpus_version_id, project, k8s_version, source_path)
);

-- ---------------------------------------------------------------------------------------
-- Chunks: one nullable vector column per embedding model, both full-text configs.
-- ---------------------------------------------------------------------------------------
create table public.chunks (
    id                bigint generated always as identity primary key,
    corpus_version_id bigint      not null references public.corpus_versions (id) on delete cascade,
    document_id       bigint      not null references public.documents (id) on delete cascade,
    chunk_key         text        not null,
    ordinal           integer     not null,
    project           text        not null,
    k8s_version       text        not null,
    doc_type          text        not null
                      check (doc_type in ('concept', 'task', 'reference', 'tutorial', 'other')),
    source_path       text        not null,
    url               text        not null,
    title             text        not null,
    section_path      text[]      not null default '{}',
    language          text        not null default 'en',
    access_group      text        not null default 'public',
    header            text        not null,
    body              text        not null,
    char_start        integer     not null,
    char_end          integer     not null,
    token_count       integer     not null,
    emb_gte_small     extensions.halfvec(384),
    emb_bge_large     extensions.halfvec(1024),
    emb_bge_m3        extensions.halfvec(1024),
    tsv_english       tsvector generated always as
                      (to_tsvector('english'::regconfig, header || ' ' || body)) stored,
    tsv_simple        tsvector generated always as
                      (to_tsvector('simple'::regconfig, header || ' ' || body)) stored,
    created_at        timestamptz not null default now(),
    unique (corpus_version_id, chunk_key)
);

-- Dense legs: HNSW on normalized vectors, inner product. m / ef_construction are experiment knobs.
create index chunks_emb_gte_small_hnsw on public.chunks
    using hnsw (emb_gte_small extensions.halfvec_ip_ops) with (m = 16, ef_construction = 64);
create index chunks_emb_bge_large_hnsw on public.chunks
    using hnsw (emb_bge_large extensions.halfvec_ip_ops) with (m = 16, ef_construction = 64);
create index chunks_emb_bge_m3_hnsw on public.chunks
    using hnsw (emb_bge_m3 extensions.halfvec_ip_ops) with (m = 16, ef_construction = 64);

-- Keyword legs.
create index chunks_tsv_english_gin on public.chunks using gin (tsv_english);
create index chunks_tsv_simple_gin  on public.chunks using gin (tsv_simple);

-- Filters applied inside the search SQL.
create index chunks_filter_idx on public.chunks (corpus_version_id, k8s_version, doc_type);
create index chunks_document_idx on public.chunks (document_id);
create index documents_lookup_idx on public.documents (corpus_version_id, k8s_version, source_path);

-- ---------------------------------------------------------------------------------------
-- Embedding cache: text hash -> vector, per model and revision. Re-ingests skip unchanged text.
-- ---------------------------------------------------------------------------------------
create table public.embedding_cache (
    model          text        not null,
    model_revision text        not null,
    text_hash      text        not null,
    embedding      extensions.halfvec not null,
    created_at     timestamptz not null default now(),
    primary key (model, model_revision, text_hash)
);

-- ---------------------------------------------------------------------------------------
-- Eval results. Every loop (local, ci, nightly, hourly) writes here with version stamps.
-- ---------------------------------------------------------------------------------------
create table public.eval_runs (
    id                bigint generated always as identity primary key,
    kind              text        not null,            -- retrieval | answer | agent | conversation
    loop              text        not null,            -- local | ci | nightly | hourly
    experiment        text        not null,            -- config name
    config            jsonb       not null,
    stamps            jsonb       not null,            -- sre_shared.stamps.VersionStamps
    golden_version    text        not null,
    corpus_version_id bigint      references public.corpus_versions (id) on delete set null,
    status            text        not null default 'running'
                      check (status in ('running', 'done', 'failed')),
    summary           jsonb       not null default '{}'::jsonb,  -- aggregated metrics
    started_at        timestamptz not null default now(),
    finished_at       timestamptz
);

create table public.eval_scores (
    run_id  bigint           not null references public.eval_runs (id) on delete cascade,
    item_id text             not null,
    metric  text             not null,
    k       integer          not null default 0,   -- 0 for metrics without a cutoff
    value   double precision not null,
    details jsonb,
    primary key (run_id, item_id, metric, k)
);

create index eval_runs_experiment_idx on public.eval_runs (kind, experiment, started_at desc);

-- ---------------------------------------------------------------------------------------
-- Helpers.
-- ---------------------------------------------------------------------------------------

-- Groups the caller may read: its JWT app_metadata.groups, plus 'public' for everyone.
create or replace function sre.user_groups() returns text[]
    language sql stable
    set search_path = ''
as $$
    select array_append(
        coalesce(
            array(select jsonb_array_elements_text(
                coalesce(auth.jwt() -> 'app_metadata' -> 'groups', '[]'::jsonb))),
            '{}'::text[]),
        'public')
$$;

create or replace function sre.active_corpus_version() returns bigint
    language sql stable
    set search_path = ''
as $$
    select id from public.corpus_versions where status = 'active'
$$;

grant usage on schema sre to anon, authenticated, service_role;
grant execute on function sre.user_groups() to anon, authenticated, service_role;
grant execute on function sre.active_corpus_version() to anon, authenticated, service_role;

-- ---------------------------------------------------------------------------------------
-- Row-level security. Reads: by access group. Writes: no policies, so only the table owner
-- (postgres: ingest and evals) and service_role can write.
-- ---------------------------------------------------------------------------------------
alter table public.corpus_versions enable row level security;
alter table public.documents       enable row level security;
alter table public.chunks          enable row level security;
alter table public.embedding_cache enable row level security;
alter table public.eval_runs       enable row level security;
alter table public.eval_scores     enable row level security;

create policy corpus_versions_read on public.corpus_versions
    for select to anon, authenticated using (true);

-- `(select ...)` makes Postgres evaluate the group list once per query, not once per row.
create policy documents_read on public.documents
    for select to anon, authenticated
    using (access_group = any ((select sre.user_groups())::text[]));

create policy chunks_read on public.chunks
    for select to anon, authenticated
    using (access_group = any ((select sre.user_groups())::text[]));

grant usage on schema public to anon, authenticated, service_role;
grant select on public.corpus_versions, public.documents, public.chunks to anon, authenticated;
grant all on all tables in schema public to service_role;

revoke insert, update, delete, truncate on
    public.corpus_versions, public.documents, public.chunks,
    public.embedding_cache, public.eval_runs, public.eval_scores
    from anon, authenticated;
revoke select on public.embedding_cache, public.eval_runs, public.eval_scores
    from anon, authenticated;
