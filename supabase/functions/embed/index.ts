// Query embeddings with the Edge Runtime's built-in gte-small (mean pooling + L2 normalization).
// POST {"texts": string[]} -> {"embeddings": number[][]}
// Requires a valid Supabase JWT (the anon key is enough): verify_jwt stays at its default.

// deno-lint-ignore no-explicit-any
declare const Supabase: any;

// The Edge Runtime cancels a request at its CPU-time limit; a handful of near-512-token
// texts already reaches it, so callers send one text per request by default
// (sre_core.retrieval.edge, batch_size=1); one ~500-token text takes ~0.45 s.
const MAX_TEXTS = 8;
const session = new Supabase.ai.Session("gte-small");

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });

Deno.serve(async (req: Request) => {
  if (req.method !== "POST") return json({ error: "POST only" }, 405);
  let texts: unknown;
  try {
    ({ texts } = await req.json());
  } catch {
    return json({ error: "invalid JSON" }, 400);
  }
  if (
    !Array.isArray(texts) || texts.length > MAX_TEXTS ||
    !texts.every((t) => typeof t === "string")
  ) {
    return json({ error: `texts must be an array of at most ${MAX_TEXTS} strings` }, 400);
  }
  const embeddings: number[][] = [];
  for (const text of texts as string[]) {
    const out = await session.run(text, { mean_pool: true, normalize: true });
    embeddings.push(Array.from(out as ArrayLike<number>));
  }
  return json({ embeddings });
});
