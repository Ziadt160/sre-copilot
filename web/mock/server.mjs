// A tiny mock of the chat API (and of Supabase's password login) for offline dev and the
// Playwright smoke test. It follows the route list of the contract: POST /chat (SSE),
// GET /threads, GET /threads/{id}, GET /threads/{id}/live (SSE), POST /approvals/{id},
// GET /approvals?status=pending, POST /feedback, POST /events. GET /__state shows what the UI
// posted so tests can assert on it.
import { createServer } from "node:http";
import { randomUUID } from "node:crypto";

const PORT = Number(process.env.MOCK_PORT ?? 8099);
const TICK = Number(process.env.MOCK_TICK_MS ?? 40);

const state = { threads: new Map(), approvals: new Map(), feedback: [], events: [], decisions: [] };
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const SOURCES = [
  {
    chunk_id: 101,
    source_path: "concepts/workloads/pods/_index.md",
    k8s_version: "1.34",
    url: "https://kubernetes.io/docs/concepts/workloads/pods/",
    title: "Pods",
  },
  {
    chunk_id: 102,
    source_path: "concepts/workloads/controllers/deployment.md",
    k8s_version: "1.34",
    url: "https://kubernetes.io/docs/concepts/workloads/controllers/deployment/",
    title: "Deployments",
  },
];

function b64(obj) {
  return Buffer.from(JSON.stringify(obj)).toString("base64url");
}

function cors(req, res) {
  res.setHeader("Access-Control-Allow-Origin", "*");
  // echo what the browser asks for (supabase-js sends several x-* headers)
  res.setHeader("Access-Control-Allow-Headers", req.headers["access-control-request-headers"] ?? "authorization, content-type");
  res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
}

function send(res, status, body) {
  res.writeHead(status, { "Content-Type": "application/json" });
  res.end(body === undefined ? undefined : JSON.stringify(body));
}

async function readJson(req) {
  let raw = "";
  for await (const chunk of req) raw += chunk;
  return raw ? JSON.parse(raw) : {};
}

function frame(res, name, data) {
  res.write(`event: ${name}\ndata: ${JSON.stringify(data)}\n\n`);
}

function sse(res) {
  res.writeHead(200, { "Content-Type": "text/event-stream", "Cache-Control": "no-cache" });
}

function thread(id, firstMessage) {
  let t = state.threads.get(id);
  if (!t) {
    t = { id, title: firstMessage.slice(0, 60), status: "active", updated_at: new Date().toISOString(), messages: [], next: 1, decision: null };
    state.threads.set(id, t);
  }
  return t;
}

async function chat(req, res) {
  const body = await readJson(req);
  if (!body.message) return send(res, 422, { detail: "message required" });
  const t = thread(body.thread_id ?? randomUUID(), body.message);
  const turn = t.next++;
  t.decision = null;
  t.messages.push({ turn, role: "user", content: body.message });
  t.updated_at = new Date().toISOString();
  sse(res);
  frame(res, "turn", { thread_id: t.id, turn });
  await sleep(TICK);

  if (/restart|rollout/i.test(body.message)) {
    const args = { namespace: "payments", name: "checkout" };
    frame(res, "route", { route: "action", standalone_question: body.message });
    frame(res, "tool_proposed", { tool: "restart_deployment", args, kind: "write" });
    const approval = {
      id: randomUUID(),
      thread_id: t.id,
      turn,
      tool: "restart_deployment",
      args,
      namespace: "payments",
      target: "deployment/checkout",
      risk: "medium",
      required_scope: "operator",
      expected_effect: "Rolling restart of deployment/checkout in payments; pods are replaced one by one.",
      expires_at: new Date(Date.now() + 15 * 60_000).toISOString(),
      status: "pending",
    };
    state.approvals.set(approval.id, approval);
    t.status = "awaiting_approval";
    const { tool, namespace, target, risk, required_scope, expected_effect, expires_at } = approval;
    frame(res, "approval_required", { approval_id: approval.id, tool, args, namespace, target, risk, required_scope, expected_effect, expires_at });
    return res.end(); // the stream ends here; the decision arrives on POST /approvals/{id}
  }

  if (/escalate|page|outage/i.test(body.message)) {
    frame(res, "route", { route: "escalate", standalone_question: body.message });
    frame(res, "escalated", { incident_id: randomUUID(), paged: true, reason: "Possible production outage; a human must look." });
    frame(res, "done", { thread_id: t.id, turn, route: "escalate", answer: "I escalated this to the on-call engineer.", citations: [] });
    return res.end();
  }

  if (/unknown|nothing/i.test(body.message)) {
    frame(res, "route", { route: "simple", standalone_question: body.message });
    frame(res, "abstain", { reason: "The documentation does not cover this.", sources: [] });
    return res.end();
  }

  const answer = "A Pod is the smallest deployable unit in Kubernetes [1]. A Deployment manages replicas of Pods [2].";
  frame(res, "route", { route: "simple", standalone_question: body.message });
  frame(res, "sources", { sources: SOURCES });
  for (const word of answer.split(/(?<= )/)) {
    frame(res, "token", { text: word });
    await sleep(TICK / 4);
  }
  frame(res, "done", { thread_id: t.id, turn, route: "simple", answer, citations: SOURCES, prompt_tokens: 120, completion_tokens: 30 });
  t.messages.push({ turn, role: "assistant", content: answer, route: "simple", citations: SOURCES });
  res.end();
}

async function liveStream(res, t) {
  sse(res);
  // wait for the decision the UI is posting (a resumed run starts after it)
  for (let i = 0; i < 100 && !t.decision; i++) await sleep(50);
  const d = t.decision;
  if (!d) return res.end();
  const turn = t.next - 1;
  if (d.decision === "reject") {
    const answer = "Understood: the restart was rejected and nothing was changed.";
    frame(res, "done", { thread_id: t.id, turn, route: "action", answer, citations: [] });
    t.messages.push({ turn, role: "assistant", content: answer, route: "action", citations: [] });
    return res.end();
  }
  const opId = randomUUID();
  const base = { operation_id: opId, tool: "restart_deployment" };
  frame(res, "operation", { ...base, status: "approved" });
  await sleep(TICK);
  frame(res, "operation", { ...base, status: "running", progress: "2/3 pods ready" });
  await sleep(TICK * 4);
  frame(res, "operation", { ...base, status: "succeeded", result: { restarted: true, args: d.args } });
  const answer = "Restarted deployment/checkout in payments; all pods are ready.";
  frame(res, "done", { thread_id: t.id, turn, route: "action", answer, citations: [] });
  t.messages.push({ turn, role: "assistant", content: answer, route: "action", citations: [] });
  res.end();
}

async function handle(req, res) {
  cors(req, res);
  const url = new URL(req.url ?? "/", `http://${req.headers.host}`);
  const path = url.pathname;
  if (req.method === "OPTIONS") return send(res, 204);
  if (path === "/healthz") return send(res, 200, { ok: true });

  // Supabase password login
  if (path === "/auth/v1/token" && req.method === "POST") {
    const now = Math.floor(Date.now() / 1000);
    const user = { id: "00000000-0000-4000-8000-000000000001", aud: "authenticated", role: "authenticated", email: "operator@example.test", app_metadata: {}, user_metadata: {}, created_at: new Date().toISOString() };
    const jwt = `${b64({ alg: "none", typ: "JWT" })}.${b64({ sub: user.id, aud: "authenticated", exp: now + 3600 })}.mock`;
    return send(res, 200, { access_token: jwt, token_type: "bearer", expires_in: 3600, expires_at: now + 3600, refresh_token: "mock-refresh", user });
  }
  if (path === "/auth/v1/logout") return send(res, 204);
  if (path === "/auth/v1/user") return send(res, 200, { id: "00000000-0000-4000-8000-000000000001", email: "operator@example.test" });

  if (path === "/__state") return send(res, 200, { feedback: state.feedback, events: state.events, decisions: state.decisions });
  if (path === "/__reset") {
    for (const k of ["feedback", "events", "decisions"]) state[k].length = 0;
    state.threads.clear();
    state.approvals.clear();
    return send(res, 204);
  }

  if (!req.headers.authorization?.startsWith("Bearer ")) return send(res, 401, { detail: "missing bearer token" });

  if (path === "/chat" && req.method === "POST") return chat(req, res);
  if (path === "/threads" && req.method === "GET") {
    const rows = [...state.threads.values()].map(({ id, title, status, updated_at }) => ({ id, title, status, updated_at }));
    return send(res, 200, rows.reverse());
  }
  let m = path.match(/^\/threads\/([^/]+)(\/live)?$/);
  if (m) {
    const t = state.threads.get(m[1]);
    if (!t) return send(res, 404, { detail: "thread not found" });
    if (m[2]) return liveStream(res, t);
    return send(res, 200, { thread: { id: t.id, title: t.title }, messages: t.messages });
  }
  if (path === "/approvals" && req.method === "GET") {
    const pending = [...state.approvals.values()].filter((a) => a.status === "pending");
    return send(res, 200, pending);
  }
  m = path.match(/^\/approvals\/([^/]+)$/);
  if (m && req.method === "POST") {
    const a = state.approvals.get(m[1]);
    if (!a) return send(res, 404, { detail: "approval not found" });
    const body = await readJson(req);
    if (a.status !== "pending" || Date.parse(a.expires_at) < Date.now()) return send(res, 409, { detail: "approval already decided or expired" });
    a.status = body.decision === "reject" ? "rejected" : "approved";
    const t = state.threads.get(a.thread_id);
    const args = body.decision === "edit" ? body.args : a.args;
    t.decision = { decision: body.decision, args };
    t.status = "active";
    state.decisions.push({ id: a.id, ...body });
    return send(res, 200, { approval: { ...a, status: a.status, args } });
  }
  if (path === "/feedback" && req.method === "POST") {
    state.feedback.push(await readJson(req));
    return send(res, 204);
  }
  if (path === "/events" && req.method === "POST") {
    state.events.push(await readJson(req));
    return send(res, 204);
  }
  return send(res, 404, { detail: "not found" });
}

createServer((req, res) => {
  handle(req, res).catch((e) => {
    if (!res.headersSent) send(res, 500, { detail: String(e) });
    else res.end();
  });
}).listen(PORT, "127.0.0.1", () => console.log(`mock api on http://127.0.0.1:${PORT}`));
