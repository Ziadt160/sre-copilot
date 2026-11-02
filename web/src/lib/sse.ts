import type { AgentEvent } from "@/contracts/agent.types";

/**
 * Reads a Server-Sent Events body (fetch + ReadableStream: EventSource cannot POST).
 * Frames are `event: <name>` / `data: <json>` separated by a blank line; the server removes
 * `event` from the data, so it is put back here to give the typed discriminated union.
 */
export async function* readEvents(body: ReadableStream<Uint8Array>): AsyncGenerator<AgentEvent> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");
      let end = buffer.indexOf("\n\n");
      while (end >= 0) {
        const ev = parseFrame(buffer.slice(0, end));
        buffer = buffer.slice(end + 2);
        if (ev) yield ev;
        end = buffer.indexOf("\n\n");
      }
    }
    const tail = parseFrame(buffer);
    if (tail) yield tail;
  } finally {
    reader.releaseLock();
  }
}

export function parseFrame(frame: string): AgentEvent | null {
  let name = "message";
  const data: string[] = [];
  for (const line of frame.split("\n")) {
    if (line.startsWith(":")) continue; // comment / keep-alive
    if (line.startsWith("event:")) name = line.slice(6).trim();
    else if (line.startsWith("data:")) data.push(line.slice(5).replace(/^ /, ""));
  }
  if (data.length === 0) return null;
  try {
    return { ...JSON.parse(data.join("\n")), event: name } as AgentEvent;
  } catch {
    return { event: "error", type: "bad_frame", detail: "unreadable event from the server" };
  }
}
