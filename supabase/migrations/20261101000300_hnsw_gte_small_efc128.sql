-- gte-small HNSW index: ef_construction 64 -> 128, m stays 16 (decision 30).
-- Measured on golden v1, five builds each (`sre-eval hnsw --repeat 5`): ANN recall@10 at
-- ef_search 100 went from 0.974-0.991 to 0.983-0.996, at the same index size (54.5 MB for
-- 47,151 chunks) and about 5 s more single-threaded build time; query latency unchanged.
--
-- Single-threaded build: a parallel build keeps the graph in dynamic shared memory, which the
-- local Docker stack's /dev/shm cannot hold ("could not resize shared memory segment").

set max_parallel_maintenance_workers = 0;
set maintenance_work_mem = '128MB';

drop index if exists public.chunks_emb_gte_small_hnsw;
create index chunks_emb_gte_small_hnsw on public.chunks
    using hnsw (emb_gte_small extensions.halfvec_ip_ops) with (m = 16, ef_construction = 128);

reset maintenance_work_mem;
reset max_parallel_maintenance_workers;
