"""Lab GPU embedding server for the BGE challengers (bge-large-en-v1.5, bge-m3).

Runs under an existing Python that already has torch + sentence-transformers (the project's uv
environment deliberately has no torch). Standard library HTTP only, bound to 127.0.0.1.
The client (`sre_shared.embed.remote.RemoteSTEmbedder`) sends the model id, revision, query
prefix and max sequence length with each request, so `sre_shared.embed.SPECS` stays the single
source of truth.

    set HF_HUB_OFFLINE=1
    C:\\Python314\\python.exe tools\\gpu_embed_server.py --port 8091

Endpoints
    GET  /health -> {"device", "torch", "sentence_transformers", "loaded": [...]}
    POST /embed  {"hf_id", "revision", "texts", "kind": "document"|"query", "query_prefix",
                  "max_seq_length", "batch_size"}
              -> {"dims", "count", "dtype": "float32", "data": base64 little-endian row-major,
                  "truncated": int}
    POST /count  {"hf_id", "revision", "texts", "max_seq_length"} -> {"counts": [...]}
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

_models: dict[tuple[str, str], SentenceTransformer] = {}
# One model load or batch at a time. transformers flips torch's global default dtype while it
# loads, so a load overlapping another load or a batch left a model with mixed half/float
# weights ("mat1 and mat2 must have the same dtype"), found when 4 eval workers hit a cold server.
_gpu_lock = threading.Lock()


def _model(hf_id: str, revision: str, max_seq_length: int) -> SentenceTransformer:
    """Load once per (model, revision). Call with `_gpu_lock` held."""
    key = (hf_id, revision)
    model = _models.get(key)
    if model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        kwargs: dict[str, Any] = {"dtype": torch.float16} if device == "cuda" else {}
        t0 = time.perf_counter()
        model = SentenceTransformer(
            hf_id, revision=revision, device=device, local_files_only=True, model_kwargs=kwargs
        )
        print(
            f"loaded {hf_id}@{revision[:7]} on {device} in {time.perf_counter() - t0:.1f}s",
            flush=True,
        )
        _models[key] = model
    model.max_seq_length = max_seq_length
    return model


def _token_counts(model: SentenceTransformer, texts: list[str]) -> list[int]:
    enc = model.tokenizer(texts, add_special_tokens=False, truncation=False)
    return [len(ids) for ids in enc["input_ids"]]


def _embed(req: dict[str, Any]) -> dict[str, Any]:
    texts: list[str] = req["texts"]
    if req.get("kind") == "query":
        texts = [req.get("query_prefix", "") + t for t in texts]
    with _gpu_lock:
        model = _model(req["hf_id"], req["revision"], int(req["max_seq_length"]))
        counts = _token_counts(model, texts) if texts else []
        vecs = model.encode(
            texts,
            batch_size=int(req.get("batch_size", 32)),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
    arr = np.ascontiguousarray(np.asarray(vecs, dtype="<f4"))
    if arr.size:  # fp16 inference leaves norms a hair off 1.0
        arr /= np.clip(np.linalg.norm(arr, axis=1, keepdims=True), 1e-12, None)
    limit = int(req["max_seq_length"]) - 2  # [CLS]/<s> and [SEP]/</s>
    dims = int(arr.shape[1]) if arr.ndim == 2 else model.get_sentence_embedding_dimension()
    return {
        "dims": dims,
        "count": len(texts),
        "dtype": "float32",
        "data": base64.b64encode(arr.tobytes()).decode("ascii"),
        "truncated": sum(1 for c in counts if c > limit),
    }


def _count(req: dict[str, Any]) -> dict[str, Any]:
    with _gpu_lock:
        model = _model(req["hf_id"], req["revision"], int(req.get("max_seq_length", 512)))
        return {"counts": _token_counts(model, req["texts"])}


class Handler(BaseHTTPRequestHandler):
    server_version = "sre-gpu-embed/1"

    def _send(self, status: int, body: dict[str, Any]) -> None:
        payload = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        if self.path != "/health":
            return self._send(404, {"error": "not found"})
        self._send(
            200,
            {
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "torch": torch.__version__,
                "sentence_transformers": sys.modules["sentence_transformers"].__version__,
                "loaded": [f"{h}@{r}" for h, r in _models],
            },
        )

    def do_POST(self) -> None:
        routes = {"/embed": _embed, "/count": _count}
        handler = routes.get(self.path)
        if handler is None:
            return self._send(404, {"error": "not found"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            req = json.loads(self.rfile.read(length))
            self._send(200, handler(req))
        except (KeyError, ValueError, TypeError) as e:
            self._send(400, {"error": f"{type(e).__name__}: {e}"})
        except Exception as e:  # model load or CUDA errors: report, keep serving
            self._send(500, {"error": f"{type(e).__name__}: {e}"})

    def log_message(self, fmt: str, *args: Any) -> None:  # quieter than the default
        if not self.path.startswith("/health"):
            sys.stderr.write(f"{self.command} {self.path} {fmt % args}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8091)
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        parser.error("bind to localhost only; this server has no authentication")
    print(
        f"gpu embed server on http://{args.host}:{args.port} (cuda={torch.cuda.is_available()})",
        flush=True,
    )
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
