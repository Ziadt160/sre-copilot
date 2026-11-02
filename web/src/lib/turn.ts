import type {
  AgentEvent,
  ApprovalRequiredEvent,
  Citation,
  EscalatedEvent,
  OperationEvent,
  ToolProposedEvent,
} from "@/contracts/agent.types";

/** What the page shows for one assistant turn, built by folding the turn's SSE events. */
export interface TurnView {
  key: string;
  question: string;
  threadId?: string;
  turn?: number;
  route?: string;
  standalone?: string;
  cache?: string;
  sources: Citation[];
  text: string;
  proposal?: ToolProposedEvent;
  approval?: ApprovalRequiredEvent;
  /** pending: waiting for the person; sending; decided: the server accepted the decision. */
  approvalState?: "pending" | "sending" | "decided";
  operation?: OperationEvent;
  escalation?: EscalatedEvent;
  abstain?: string;
  error?: string;
  /** a done event was received (the answer is final). */
  finished: boolean;
}

export function newTurn(key: string, question: string): TurnView {
  return { key, question, sources: [], text: "", finished: false };
}

/** Pure reducer: one event in, the next view out. */
export function applyEvent(view: TurnView, ev: AgentEvent): TurnView {
  switch (ev.event) {
    case "turn":
      return { ...view, threadId: ev.thread_id, turn: ev.turn };
    case "route":
      return { ...view, route: ev.route, standalone: ev.standalone_question };
    case "cache_hit":
      return { ...view, cache: ev.cache, route: "cache" };
    case "sources":
      return { ...view, sources: ev.sources };
    case "token":
      return { ...view, text: view.text + ev.text };
    case "tool_proposed":
      return { ...view, proposal: ev };
    case "approval_required":
      return { ...view, approval: ev, approvalState: "pending" };
    case "operation":
      return { ...view, operation: ev };
    case "escalated":
      return { ...view, escalation: ev };
    case "abstain":
      return { ...view, abstain: ev.reason, sources: ev.sources.length ? ev.sources : view.sources };
    case "done":
      return {
        ...view,
        // `done` carries the final text; keep streamed tokens only if it is empty.
        text: ev.answer || view.text,
        sources: ev.citations.length ? ev.citations : view.sources,
        finished: true,
      };
    case "error":
      return { ...view, error: `${ev.type}: ${ev.detail}` };
    default:
      return view; // an event this UI version does not know
  }
}
