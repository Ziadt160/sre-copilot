import type {
  AgentEvent,
  ApprovalDecision,
  ChatRequest,
  Citation,
  FeedbackRequest,
  UiEventRequest,
} from "@/contracts/agent.types";
import { readEvents } from "./sse";
import { supabase } from "./supabase";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
  }
}

// The shapes of the read routes (T13 owns them; these follow the database rows).
export interface ThreadSummary {
  id: string;
  title: string | null;
  status?: string;
  updated_at?: string;
}
export interface StoredMessage {
  turn: number;
  role: "user" | "assistant" | "human";
  content: string;
  route?: string | null;
  citations?: Citation[];
}
export interface PendingApproval {
  id: string;
  thread_id: string;
  turn: number;
  tool: string;
  args: Record<string, unknown>;
  namespace: string;
  target: string;
  risk: "low" | "medium" | "high";
  required_scope: "operator" | "admin";
  expected_effect: string;
  expires_at: string;
}

async function headers(): Promise<Record<string, string>> {
  const { data } = await supabase().auth.getSession();
  const token = data.session?.access_token;
  return {
    "Content-Type": "application/json",
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
  };
}

async function fail(res: Response): Promise<never> {
  let detail = res.statusText;
  try {
    const body = (await res.json()) as { detail?: unknown };
    if (typeof body.detail === "string") detail = body.detail;
  } catch {
    // keep statusText
  }
  throw new ApiError(res.status, detail || `HTTP ${res.status}`);
}

async function json<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API}${path}`, { ...init, headers: { ...(await headers()), ...init?.headers } });
  if (!res.ok) return fail(res);
  return (await res.json()) as T;
}

async function post(path: string, body: unknown): Promise<Response> {
  const res = await fetch(`${API}${path}`, {
    method: "POST",
    headers: await headers(),
    body: JSON.stringify(body),
  });
  if (!res.ok) return fail(res);
  return res;
}

async function stream(res: Response): Promise<AsyncGenerator<AgentEvent>> {
  if (!res.ok) return fail(res);
  if (!res.body) throw new ApiError(res.status, "empty response body");
  return readEvents(res.body);
}

export async function chat(req: ChatRequest, signal?: AbortSignal) {
  const res = await fetch(`${API}/chat`, {
    method: "POST",
    headers: { ...(await headers()), Accept: "text/event-stream" },
    body: JSON.stringify(req),
    signal,
  });
  return stream(res);
}

export async function live(threadId: string, signal?: AbortSignal) {
  const res = await fetch(`${API}/threads/${threadId}/live`, {
    headers: { ...(await headers()), Accept: "text/event-stream" },
    signal,
  });
  return stream(res);
}

export const listThreads = () => json<ThreadSummary[]>("/threads");

export async function getThread(id: string): Promise<{ messages: StoredMessage[] }> {
  const body = await json<{ messages?: StoredMessage[] }>(`/threads/${id}`);
  return { messages: body.messages ?? [] };
}

export const pendingApprovals = () => json<PendingApproval[]>("/approvals?status=pending");

export async function decide(id: string, decision: ApprovalDecision): Promise<void> {
  await post(`/approvals/${id}`, decision);
}

export async function sendFeedback(req: FeedbackRequest): Promise<void> {
  await post("/feedback", req);
}

export async function sendUiEvent(req: UiEventRequest): Promise<void> {
  await post("/events", req);
}
