"""gte-small on ONNX Runtime (CPU): the production and CI embedding path, no torch.

Uses the `Supabase/gte-small` export, the model family Supabase's Edge Runtime serves, with the
same post-processing (mean pooling over the attention mask, then L2 normalization).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download, model_info
from tokenizers import Tokenizer

from sre_shared.embed.base import SPECS, EmbeddingSpec, Vectors, l2_normalize
from sre_shared.settings import get_settings


class OnnxGteSmall:
    def __init__(
        self,
        spec: EmbeddingSpec = SPECS["gte-small"],
        onnx_file: str = "onnx/model.onnx",
        local_files_only: bool | None = None,
        intra_op_threads: int = 0,
    ) -> None:
        self.spec = spec
        offline = get_settings().hf_offline if local_files_only is None else local_files_only
        model_path = hf_hub_download(
            spec.hf_id, onnx_file, revision=spec.revision, local_files_only=offline
        )
        tok_path = hf_hub_download(
            spec.hf_id, "tokenizer.json", revision=spec.revision, local_files_only=offline
        )
        self._revision = spec.revision or _resolve_revision(spec.hf_id, offline)
        self.onnx_file = onnx_file

        self._counter = Tokenizer.from_file(tok_path)
        self._counter.no_truncation()
        self._counter.no_padding()

        self._tok = Tokenizer.from_file(tok_path)
        self._tok.enable_truncation(max_length=spec.max_tokens)
        pad_token = "[PAD]"  # noqa: S105 - a vocabulary token, not a secret
        pad_id = self._tok.token_to_id(pad_token)
        self._tok.enable_padding(pad_id=0 if pad_id is None else pad_id, pad_token=pad_token)

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = intra_op_threads
        self._session = ort.InferenceSession(
            model_path, sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._inputs = {i.name for i in self._session.get_inputs()}

    @property
    def revision(self) -> str:
        return self._revision

    def count_tokens(self, text: str) -> int:
        return len(self._counter.encode(text, add_special_tokens=False).ids)

    def embed_documents(self, texts: Sequence[str], batch_size: int = 32) -> Vectors:
        return self._embed(list(texts), batch_size)

    def embed_queries(self, texts: Sequence[str], batch_size: int = 32) -> Vectors:
        return self._embed([self.spec.query_prefix + t for t in texts], batch_size)

    def _embed(self, texts: list[str], batch_size: int) -> Vectors:
        if not texts:
            return np.zeros((0, self.spec.dims), dtype=np.float32)
        # Batches pad to their longest text, so embed in length order and restore the input
        # order at the end; mixed lengths otherwise pay for padding.
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        ordered = [texts[i] for i in order]
        out: list[Vectors] = []
        for start in range(0, len(ordered), batch_size):
            batch = self._tok.encode_batch(ordered[start : start + batch_size])
            ids = np.array([e.ids for e in batch], dtype=np.int64)
            mask = np.array([e.attention_mask for e in batch], dtype=np.int64)
            feeds = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self._inputs:
                feeds["token_type_ids"] = np.array([e.type_ids for e in batch], dtype=np.int64)
            hidden = self._session.run(None, feeds)[0]  # (batch, seq, dims)
            m = mask[..., None].astype(np.float32)
            pooled = (hidden * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-9, None)
            out.append(l2_normalize(pooled))
        result = np.empty((len(texts), self.spec.dims), dtype=np.float32)
        result[order] = np.concatenate(out, axis=0)
        return result


def _resolve_revision(repo_id: str, offline: bool) -> str:
    if offline:
        return "unpinned-offline"
    sha = model_info(repo_id).sha
    return sha or "unknown"
