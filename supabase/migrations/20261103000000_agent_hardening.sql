-- Phase 3 verification (V07): one execution per approval, and thread state written only by core.

-- V07-1: two resumes of one approved approval could both submit the write. The resume that
-- executes it claims it first (update ... where execution_claimed_at is null returning). mcp-ops
-- still requires status 'approved' at admission, so the claim is a column, not a status.
-- V07-9: the sweeper also expires approved rows never claimed by their deadline (a crash between
-- the decision and the resume), so their threads are released.
alter table public.approvals add column execution_claimed_at timestamptz;
create index approvals_unclaimed on public.approvals (expires_at)
    where status = 'approved' and execution_claimed_at is null;

-- V07-10: through the REST API an owner could rewrite status, next_turn and summary on their own
-- thread. Users may rename a thread; core writes the rest on its service connection.
revoke update on public.threads from authenticated;
grant update (title) on public.threads to authenticated;
