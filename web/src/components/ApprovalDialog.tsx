"use client";

import { useEffect, useRef, useState } from "react";
import type { ApprovalDecision, ApprovalRequiredEvent } from "@/contracts/agent.types";

function secondsLeft(expiresAt: string): number {
  return Math.max(0, Math.floor((new Date(expiresAt).getTime() - Date.now()) / 1000));
}

function clock(total: number): string {
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${m}:${String(s).padStart(2, "0")}`;
}

interface Props {
  approval: ApprovalRequiredEvent;
  busy: boolean;
  error?: string | null;
  onDecide: (decision: ApprovalDecision) => void;
  onClose: () => void;
}

/**
 * Modal built on the native <dialog>: showModal() traps focus, restores it on close and maps
 * Escape to close. Closing without deciding leaves the approval pending (the chat keeps a
 * "Review" button), so Escape can never approve or reject by accident.
 */
export function ApprovalDialog({ approval, busy, error, onDecide, onClose }: Props) {
  const ref = useRef<HTMLDialogElement>(null);
  const [left, setLeft] = useState(() => secondsLeft(approval.expires_at));
  const [editing, setEditing] = useState(false);
  const [argsText, setArgsText] = useState(() => JSON.stringify(approval.args, null, 2));
  const [reason, setReason] = useState("");
  const [parseError, setParseError] = useState<string | null>(null);

  // No close() on cleanup: it fires the dialog's close event, which the chat reads as "decide
  // later", and React runs every effect's cleanup once on mount in development (StrictMode), so
  // the dialog closed itself the moment it opened. Unmounting removes the element anyway.
  useEffect(() => {
    const dialog = ref.current;
    if (dialog && !dialog.open) dialog.showModal();
  }, []);

  useEffect(() => {
    const id = setInterval(() => setLeft(secondsLeft(approval.expires_at)), 1000);
    return () => clearInterval(id);
  }, [approval.expires_at]);

  const expired = left <= 0;
  const disabled = busy || expired;

  function submitEdit() {
    try {
      const parsed: unknown = JSON.parse(argsText);
      if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) {
        throw new Error("arguments must be a JSON object");
      }
      setParseError(null);
      onDecide({ decision: "edit", args: parsed as Record<string, unknown>, reason: reason || null });
    } catch (e) {
      setParseError(e instanceof Error ? e.message : "invalid JSON");
    }
  }

  return (
    <dialog ref={ref} className="approval" aria-labelledby="approval-title" onClose={onClose}>
      <h2 id="approval-title">Approve action: {approval.tool}</h2>
      <p className={`risk risk-${approval.risk}`}>Risk: {approval.risk}</p>
      <dl>
        <dt>Target</dt>
        <dd>{approval.target}</dd>
        <dt>Namespace</dt>
        <dd>{approval.namespace}</dd>
        <dt>Expected effect</dt>
        <dd>{approval.expected_effect}</dd>
        <dt>Needs scope</dt>
        <dd>{approval.required_scope}</dd>
        <dt>Expires in</dt>
        <dd>
          <span role="timer" aria-live="off" data-testid="countdown">
            {expired ? "expired" : clock(left)}
          </span>
        </dd>
      </dl>

      <label htmlFor="approval-args">Arguments{editing ? " (editable JSON)" : ""}</label>
      <textarea
        id="approval-args"
        rows={6}
        readOnly={!editing}
        value={editing ? argsText : JSON.stringify(approval.args, null, 2)}
        onChange={(e) => setArgsText(e.target.value)}
      />
      {parseError && <p role="alert">{parseError}</p>}

      <label htmlFor="approval-reason">Reason (optional)</label>
      <input id="approval-reason" value={reason} onChange={(e) => setReason(e.target.value)} maxLength={500} />

      {expired && <p role="alert">This approval expired and can no longer be run.</p>}
      {error && <p role="alert">{error}</p>}

      <div className="row">
        <button
          type="button"
          className="primary"
          disabled={disabled}
          onClick={() => (editing ? submitEdit() : onDecide({ decision: "approve", reason: reason || null }))}
        >
          {editing ? "Approve with edited arguments" : "Approve"}
        </button>
        {!editing && (
          <button type="button" disabled={disabled} onClick={() => setEditing(true)}>
            Edit arguments
          </button>
        )}
        <button
          type="button"
          className="danger"
          disabled={disabled}
          onClick={() => onDecide({ decision: "reject", reason: reason || null })}
        >
          Reject
        </button>
        <button type="button" onClick={onClose}>
          Decide later
        </button>
      </div>
    </dialog>
  );
}
