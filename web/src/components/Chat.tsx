"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { AgentEvent, ApprovalDecision } from "@/contracts/agent.types";
import {
  ApiError,
  chat,
  decide,
  getThread,
  listThreads,
  live,
  pendingApprovals,
  type ThreadSummary,
} from "@/lib/api";
import { supabase } from "@/lib/supabase";
import { applyEvent, newTurn, type TurnView } from "@/lib/turn";
import { ApprovalDialog } from "./ApprovalDialog";
import { TurnCard } from "./TurnCard";

function decisionError(e: unknown): string {
  if (e instanceof ApiError) {
    if (e.status === 409) return "This approval was already decided or has expired.";
    if (e.status === 403) return "Your role is not allowed to decide this approval.";
    if (e.status === 404) return "Approval not found.";
  }
  return e instanceof Error ? e.message : "Could not send the decision";
}

export function Chat({ email }: { email: string }) {
  const [threads, setThreads] = useState<ThreadSummary[]>([]);
  const [threadId, setThreadId] = useState<string | null>(null);
  const [turns, setTurns] = useState<TurnView[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [dialogKey, setDialogKey] = useState<string | null>(null);
  const [dialogError, setDialogError] = useState<string | null>(null);
  const counter = useRef(0);
  const abort = useRef<AbortController | null>(null);

  const refreshThreads = useCallback(async () => {
    try {
      setThreads(await listThreads());
    } catch {
      // the list is a convenience; the chat works without it
    }
  }, []);

  useEffect(() => {
    void refreshThreads();
  }, [refreshThreads]);

  const update = useCallback((key: string, fn: (v: TurnView) => TurnView) => {
    setTurns((all) => all.map((t) => (t.key === key ? fn(t) : t)));
  }, []);

  /** Folds a stream into one turn; opens the dialog when an approval is requested. */
  const consume = useCallback(
    async (key: string, events: AsyncGenerator<AgentEvent>) => {
      for await (const ev of events) {
        update(key, (v) => applyEvent(v, ev));
        if (ev.event === "turn") setThreadId(ev.thread_id);
        if (ev.event === "approval_required") {
          setDialogError(null);
          setDialogKey(key);
        }
      }
    },
    [update],
  );

  async function send(message: string) {
    const text = message.trim();
    if (!text || busy) return;
    const key = `t${++counter.current}`;
    setTurns((all) => [...all, newTurn(key, text)]);
    setInput("");
    setBusy(true);
    abort.current = new AbortController();
    try {
      await consume(
        key,
        await chat({ thread_id: threadId, message: text, client_message_id: crypto.randomUUID() }, abort.current.signal),
      );
    } catch (e) {
      const detail =
        e instanceof ApiError && e.status === 429
          ? "Daily budget reached. Try again later."
          : e instanceof ApiError && e.status === 409
            ? "This thread is busy with another turn."
            : e instanceof Error
              ? e.message
              : "request failed";
      update(key, (v) => ({ ...v, error: detail }));
    } finally {
      setBusy(false);
      void refreshThreads();
    }
  }

  async function onDecide(key: string, decision: ApprovalDecision) {
    const view = turns.find((t) => t.key === key);
    if (!view?.approval || !view.threadId) return;
    setDialogError(null);
    update(key, (v) => ({ ...v, approvalState: "sending" }));
    try {
      await decide(view.approval.approval_id, decision);
    } catch (e) {
      setDialogError(decisionError(e));
      update(key, (v) => ({ ...v, approvalState: "pending" }));
      return;
    }
    update(key, (v) => ({ ...v, approvalState: "decided" }));
    setDialogKey(null);
    // The run resumes server-side; follow it on the thread's live stream.
    setBusy(true);
    try {
      await consume(key, await live(view.threadId));
    } catch (e) {
      update(key, (v) => ({ ...v, error: e instanceof Error ? e.message : "lost the live stream" }));
    } finally {
      setBusy(false);
    }
  }

  async function openThread(id: string) {
    abort.current?.abort();
    setThreadId(id);
    setDialogKey(null);
    try {
      const [{ messages }, pending] = await Promise.all([getThread(id), pendingApprovals().catch(() => [])]);
      const views: TurnView[] = [];
      for (const m of messages) {
        if (m.role === "user") {
          views.push({ ...newTurn(`s${id}-${m.turn}`, m.content), threadId: id, turn: m.turn });
        } else {
          const v = views.find((x) => x.turn === m.turn);
          if (v) {
            v.text = m.content;
            v.route = m.route ?? undefined;
            v.sources = m.citations ?? [];
            v.finished = true;
          }
        }
      }
      const open = pending.find((p) => p.thread_id === id);
      const last = views.at(-1);
      if (open && last) {
        last.approval = { ...open, event: "approval_required", approval_id: open.id };
        last.approvalState = "pending";
      }
      setTurns(views);
      if (open && last) setDialogKey(last.key);
    } catch (e) {
      setTurns([{ ...newTurn("err", ""), error: e instanceof Error ? e.message : "could not load the thread" }]);
    }
  }

  function newThread() {
    abort.current?.abort();
    setThreadId(null);
    setTurns([]);
    setDialogKey(null);
  }

  const dialogView = turns.find((t) => t.key === dialogKey);

  return (
    <div className="layout">
      <aside aria-label="Threads">
        <div className="row">
          <button type="button" onClick={newThread}>
            New thread
          </button>
        </div>
        <ul className="threads">
          {threads.map((t) => (
            <li key={t.id}>
              <button
                type="button"
                aria-current={t.id === threadId ? "true" : undefined}
                onClick={() => void openThread(t.id)}
              >
                {t.title ?? "Untitled thread"}
              </button>
            </li>
          ))}
        </ul>
        <p className="meta">
          {email}{" "}
          <button type="button" onClick={() => void supabase().auth.signOut()}>
            Sign out
          </button>
        </p>
      </aside>

      <main>
        <h1>SRE Copilot</h1>
        <div className="log" role="log" aria-live="polite" aria-label="Conversation">
          {turns.length === 0 && <p className="meta">Ask about Kubernetes, or ask for an action such as a rollout restart.</p>}
          {turns.map((t) => (
            <TurnCard
              key={t.key}
              view={t}
              onReview={() => {
                setDialogError(null);
                setDialogKey(t.key);
              }}
              onRegenerate={(q) => void send(q)}
            />
          ))}
        </div>
        <form
          className="composer"
          onSubmit={(e) => {
            e.preventDefault();
            void send(input);
          }}
        >
          <label htmlFor="message">Message</label>
          <input id="message" value={input} onChange={(e) => setInput(e.target.value)} maxLength={4000} autoComplete="off" />
          <button type="submit" className="primary" disabled={busy || !input.trim()}>
            Send
          </button>
        </form>
      </main>

      {dialogView?.approval && (
        <ApprovalDialog
          key={dialogView.approval.approval_id}
          approval={dialogView.approval}
          busy={dialogView.approvalState === "sending"}
          error={dialogError}
          onDecide={(d) => void onDecide(dialogView.key, d)}
          onClose={() => setDialogKey(null)}
        />
      )}
    </div>
  );
}
