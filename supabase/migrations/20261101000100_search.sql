-- Hybrid retrieval candidates: dense and keyword legs in one round trip, with per-leg ranks.
-- security invoker: the caller's RLS policies (chunks_read) apply to every row read here.
-- Fusion (weighted RRF) happens in Python; this function only retrieves and ranks per leg.
--
-- The vector column and the tsvector column are chosen from fixed whitelists and each branch is
-- a static query: nothing is concatenated into SQL text.

create or replace function public.search_candidates(
    p_column  text,                  -- emb_gte_small | emb_bge_large | emb_bge_m3
    p_qvec    extensions.halfvec,    -- query vector, same dims as the column; null if dense is off
    p_query   text,                  -- query text for the keyword leg
    p_ts      text,                  -- english | simple
    p_n       integer,               -- candidates per leg
    p_version bigint,                -- corpus version id
    p_k8s     text[] default null,   -- k8s versions filter; null = any
    p_doc     text[] default null,   -- doc types filter; null = any
    p_dense   boolean default true,
    p_keyword boolean default true,
    p_exact   boolean default false  -- dense leg as an exact scan: ordering expression the HNSW index cannot serve
) returns table (
    chunk_id         bigint,
    dense_rank       integer,
    keyword_rank     integer,
    dense_similarity double precision,
    keyword_score    double precision
)
    language plpgsql stable
    security invoker
    set search_path = public, extensions
as $$
begin
    if p_column not in ('emb_gte_small', 'emb_bge_large', 'emb_bge_m3') then
        raise exception 'unknown embedding column: %', p_column using errcode = '22023';
    end if;
    if p_ts not in ('english', 'simple') then
        raise exception 'unknown text search config: %', p_ts using errcode = '22023';
    end if;
    if p_dense and p_qvec is null then
        raise exception 'dense leg needs a query vector' using errcode = '22023';
    end if;

    return query
    with dense as materialized (
        -- Dense leg. Rows not embedded for this model (null column) are skipped.
        select d.id, d.dist
        from (
            select c.id, c.emb_gte_small <#> p_qvec as dist
            from public.chunks c
            where p_dense and not p_exact and p_column = 'emb_gte_small'
              and c.corpus_version_id = p_version
              and c.emb_gte_small is not null
              and (p_k8s is null or c.k8s_version = any (p_k8s))
              and (p_doc is null or c.doc_type = any (p_doc))
            order by c.emb_gte_small <#> p_qvec
            limit p_n
        ) d
        union all
        select d.id, d.dist
        from (
            select c.id, c.emb_gte_small <#> p_qvec as dist
            from public.chunks c
            where p_dense and p_exact and p_column = 'emb_gte_small'
              and c.corpus_version_id = p_version
              and c.emb_gte_small is not null
              and (p_k8s is null or c.k8s_version = any (p_k8s))
              and (p_doc is null or c.doc_type = any (p_doc))
            order by (c.emb_gte_small <#> p_qvec) + 0
            limit p_n
        ) d
        union all
        select d.id, d.dist
        from (
            select c.id, c.emb_bge_large <#> p_qvec as dist
            from public.chunks c
            where p_dense and not p_exact and p_column = 'emb_bge_large'
              and c.corpus_version_id = p_version
              and c.emb_bge_large is not null
              and (p_k8s is null or c.k8s_version = any (p_k8s))
              and (p_doc is null or c.doc_type = any (p_doc))
            order by c.emb_bge_large <#> p_qvec
            limit p_n
        ) d
        union all
        select d.id, d.dist
        from (
            select c.id, c.emb_bge_large <#> p_qvec as dist
            from public.chunks c
            where p_dense and p_exact and p_column = 'emb_bge_large'
              and c.corpus_version_id = p_version
              and c.emb_bge_large is not null
              and (p_k8s is null or c.k8s_version = any (p_k8s))
              and (p_doc is null or c.doc_type = any (p_doc))
            order by (c.emb_bge_large <#> p_qvec) + 0
            limit p_n
        ) d
        union all
        select d.id, d.dist
        from (
            select c.id, c.emb_bge_m3 <#> p_qvec as dist
            from public.chunks c
            where p_dense and not p_exact and p_column = 'emb_bge_m3'
              and c.corpus_version_id = p_version
              and c.emb_bge_m3 is not null
              and (p_k8s is null or c.k8s_version = any (p_k8s))
              and (p_doc is null or c.doc_type = any (p_doc))
            order by c.emb_bge_m3 <#> p_qvec
            limit p_n
        ) d
        union all
        select d.id, d.dist
        from (
            select c.id, c.emb_bge_m3 <#> p_qvec as dist
            from public.chunks c
            where p_dense and p_exact and p_column = 'emb_bge_m3'
              and c.corpus_version_id = p_version
              and c.emb_bge_m3 is not null
              and (p_k8s is null or c.k8s_version = any (p_k8s))
              and (p_doc is null or c.doc_type = any (p_doc))
            order by (c.emb_bge_m3 <#> p_qvec) + 0
            limit p_n
        ) d
    ),
    dense_ranked as (
        -- relaxed_order iterative scans return approximately ordered rows: re-sort exactly.
        select dense.id, dense.dist,
               (row_number() over (order by dense.dist, dense.id))::integer as rnk
        from dense
    ),
    kw as materialized (
        select k.id, k.score
        from (
            select c.id, ts_rank_cd(c.tsv_english, q.tsq)::double precision as score
            from public.chunks c, websearch_to_tsquery('english'::regconfig, p_query) q(tsq)
            where p_keyword and p_ts = 'english'
              and c.corpus_version_id = p_version
              and c.tsv_english @@ q.tsq
              and (p_k8s is null or c.k8s_version = any (p_k8s))
              and (p_doc is null or c.doc_type = any (p_doc))
            order by score desc, c.id
            limit p_n
        ) k
        union all
        select k.id, k.score
        from (
            select c.id, ts_rank_cd(c.tsv_simple, q.tsq)::double precision as score
            from public.chunks c, websearch_to_tsquery('simple'::regconfig, p_query) q(tsq)
            where p_keyword and p_ts = 'simple'
              and c.corpus_version_id = p_version
              and c.tsv_simple @@ q.tsq
              and (p_k8s is null or c.k8s_version = any (p_k8s))
              and (p_doc is null or c.doc_type = any (p_doc))
            order by score desc, c.id
            limit p_n
        ) k
    ),
    kw_ranked as (
        select kw.id, kw.score,
               (row_number() over (order by kw.score desc, kw.id))::integer as rnk
        from kw
    )
    select coalesce(dr.id, kr.id)::bigint,
           dr.rnk,
           kr.rnk,
           (-dr.dist)::double precision,
           kr.score
    from dense_ranked dr
    full outer join kw_ranked kr on kr.id = dr.id
    order by coalesce(dr.rnk, kr.rnk), coalesce(dr.id, kr.id);
end;
$$;

grant execute on function public.search_candidates(
    text, extensions.halfvec, text, text, integer, bigint, text[], text[], boolean, boolean, boolean
) to anon, authenticated, service_role;
