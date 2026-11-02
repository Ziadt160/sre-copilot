"use client";

import { useState } from "react";
import { sendFeedback, sendUiEvent } from "@/lib/api";
import type { TurnView } from "@/lib/turn";

interface Props {
  view: TurnView;
  onReview: () => void;
  onRegenerate: (question: string) => void;
}

const ACTIVE = new Set(["pending", "approved", "running"]);

export function TurnCard({ view, onReview, onRegenerate }: Props) {
  const [rating, setRating] = useState<1 | -1 | null>(null);
  const [reason, setReason] = useState("");
  const [votes, setVotes] = useState<Record<number, boolean>>({});
  const [note, setNote] = useState<string | null>(null);

  const { threadId, turn } = view;
  const canRate = Boolean(threadId && turn && view.finished);

  async function post(fn: () => Promise<void>, ok: string) {
    try {
      await fn();
      setNote(ok);
    } catch (e) {
      setNote(e instanceof Error ? `Could not send: ${e.message}` : "Could not send");
    }
  }

  function rate(value: 1 | -1) {
    if (!threadId || !turn) return;
    setRating(value);
    if (value === 1) void post(() => sendFeedback({ thread_id: threadId, turn, rating: 1, sources: votes }), "Feedback sent");
  }

  function sendReason() {
    if (!threadId || !turn || rating === null) return;
    void post(
      () => sendFeedback({ thread_id: threadId, turn, rating, reason: reason || null, sources: votes }),
      "Feedback sent",
    );
  }

  function vote(chunkId: number, useful: boolean) {
    if (!threadId || !turn) return;
    const next = { ...votes, [chunkId]: useful };
    setVotes(next);
    void post(() => sendFeedback({ thread_id: threadId, turn, sources: next }), "Source feedback sent");
  }

  function copy() {
    if (!threadId || !turn) return;
    void navigator.clipboard?.writeText(view.text).catch(() => undefined);
    void post(() => sendUiEvent({ thread_id: threadId, turn, kind: "copy" }), "Copied");
  }

  function regenerate() {
    if (threadId && turn) void sendUiEvent({ thread_id: threadId, turn, kind: "regenerate" }).catch(() => undefined);
    onRegenerate(view.question);
  }

  const op = view.operation;
  return (
    <article className="turn" aria-label={`Answer to: ${view.question}`}>
      <p className="user">{view.question}</p>
      <div className="assistant">
        <p className="meta">
          {view.route && <span className="badge">route: {view.route}</span>}
          {view.cache && <span className="badge">cache: {view.cache}</span>}
          {view.standalone && view.standalone !== view.question && <span> asked as: {view.standalone}</span>}
        </p>

        {view.proposal && (
          <p className="proposal">
            Proposed {view.proposal.kind}: <code>{view.proposal.tool}</code> <code>{JSON.stringify(view.proposal.args)}</code>
          </p>
        )}

        {view.approval && view.approvalState === "pending" && (
          <p role="status">
            Waiting for your approval.{" "}
            <button type="button" onClick={onReview}>
              Review approval
            </button>
          </p>
        )}

        {op && (
          <p className={`operation op-${op.status}`} role="status" data-testid="operation">
            Operation {op.tool}: <strong>{op.status}</strong>
            {op.progress ? ` (${op.progress})` : ""}
            {ACTIVE.has(op.status) && <span className="spinner" aria-hidden="true" />}
          </p>
        )}

        {view.escalation && (
          <p className="escalated" role="status">
            Escalated to a human (incident {view.escalation.incident_id.slice(0, 8)}
            {view.escalation.paged ? ", on-call paged" : ""}): {view.escalation.reason}
          </p>
        )}

        {view.abstain && (
          <p className="abstain" role="status">
            No answer: {view.abstain}
          </p>
        )}

        {view.text && <div className="answer" data-testid="answer">{view.text}</div>}
        {view.error && <p role="alert">{view.error}</p>}

        {view.sources.length > 0 && (
          <section aria-label="Sources">
            <h3>Sources</h3>
            <ol className="sources">
              {view.sources.map((s) => (
                <li key={s.chunk_id}>
                  <a href={s.url} target="_blank" rel="noreferrer">
                    {s.title}
                  </a>{" "}
                  <span className="meta">
                    {s.source_path} (k8s {s.k8s_version})
                  </span>
                  {canRate && (
                    <span className="row">
                      <button
                        type="button"
                        aria-pressed={votes[s.chunk_id] === true}
                        aria-label={`Source ${s.title} was useful`}
                        onClick={() => vote(s.chunk_id, true)}
                      >
                        Useful
                      </button>
                      <button
                        type="button"
                        aria-pressed={votes[s.chunk_id] === false}
                        aria-label={`Source ${s.title} was not useful`}
                        onClick={() => vote(s.chunk_id, false)}
                      >
                        Not useful
                      </button>
                    </span>
                  )}
                </li>
              ))}
            </ol>
          </section>
        )}

        {canRate && (
          <div className="row" role="group" aria-label="Rate this answer">
            <button type="button" aria-pressed={rating === 1} aria-label="Good answer" onClick={() => rate(1)}>
              Thumbs up
            </button>
            <button type="button" aria-pressed={rating === -1} aria-label="Bad answer" onClick={() => rate(-1)}>
              Thumbs down
            </button>
            <button type="button" onClick={copy}>
              Copy
            </button>
            <button type="button" onClick={regenerate}>
              Regenerate
            </button>
          </div>
        )}
        {rating === -1 && (
          <div className="row">
            <label htmlFor={`reason-${view.key}`}>What was wrong?</label>
            <input id={`reason-${view.key}`} value={reason} onChange={(e) => setReason(e.target.value)} maxLength={1000} />
            <button type="button" onClick={sendReason}>
              Send feedback
            </button>
          </div>
        )}
        {note && (
          <p className="meta" role="status" data-testid="note">
            {note}
          </p>
        )}
      </div>
    </article>
  );
}
