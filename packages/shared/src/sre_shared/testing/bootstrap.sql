-- Supabase-compatible scaffolding for test databases cloned from template1.
-- The stack's `postgres` database already has all of this; every statement is idempotent and
-- never replaces an object Supabase owns. The definition of auth.jwt() matches Supabase's.

create schema if not exists extensions;
create schema if not exists auth;

do $$
begin
    if not exists (select 1 from pg_roles where rolname = 'anon') then
        create role anon nologin noinherit;
    end if;
    if not exists (select 1 from pg_roles where rolname = 'authenticated') then
        create role authenticated nologin noinherit;
    end if;
    if not exists (select 1 from pg_roles where rolname = 'service_role') then
        create role service_role nologin noinherit bypassrls;
    end if;
    if not pg_has_role(current_user, 'authenticated', 'member') then
        execute format('grant anon, authenticated, service_role to %I', current_user);
    end if;
    if not exists (
        select 1 from pg_proc p join pg_namespace n on n.oid = p.pronamespace
        where n.nspname = 'auth' and p.proname = 'jwt'
    ) then
        create function auth.jwt() returns jsonb
            language sql stable
        as $fn$
            select coalesce(
                nullif(current_setting('request.jwt.claim', true), ''),
                nullif(current_setting('request.jwt.claims', true), '')
            )::jsonb
        $fn$;
    end if;
end
$$;

grant usage on schema auth, extensions, public to anon, authenticated, service_role;
grant execute on function auth.jwt() to anon, authenticated, service_role;
