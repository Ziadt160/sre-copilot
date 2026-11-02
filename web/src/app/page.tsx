"use client";

import type { Session } from "@supabase/supabase-js";
import { useEffect, useState } from "react";
import { Chat } from "@/components/Chat";
import { supabase } from "@/lib/supabase";

function Login() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    const { error: err } = await supabase().auth.signInWithPassword({ email, password });
    if (err) setError(err.message);
    setBusy(false);
  }

  return (
    <main className="login">
      <h1>SRE Copilot</h1>
      <form onSubmit={(e) => void submit(e)}>
        <label htmlFor="email">Email</label>
        <input id="email" type="email" autoComplete="username" required value={email} onChange={(e) => setEmail(e.target.value)} />
        <label htmlFor="password">Password</label>
        <input
          id="password"
          type="password"
          autoComplete="current-password"
          required
          value={password}
          onChange={(e) => setPassword(e.target.value)}
        />
        {error && <p role="alert">{error}</p>}
        <button type="submit" className="primary" disabled={busy}>
          Sign in
        </button>
      </form>
    </main>
  );
}

export default function Page() {
  const [session, setSession] = useState<Session | null | undefined>(undefined);

  useEffect(() => {
    const client = supabase();
    void client.auth.getSession().then(({ data }) => setSession(data.session));
    const { data } = client.auth.onAuthStateChange((_event, next) => setSession(next));
    return () => data.subscription.unsubscribe();
  }, []);

  if (session === undefined) return <p className="meta">Loading...</p>;
  if (!session) return <Login />;
  return <Chat email={session.user.email ?? "signed in"} />;
}
