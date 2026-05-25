"""CLIP image embedder for trajectory frames.

Used by FCVR to:
  (a) Cluster failed trajectories (one mean-pooled vector per trajectory).
  (b) Seed MMR key-frame selection (per-frame vectors within a trajectory).

Model choice rationale:
  - openai/clip-vit-base-patch32 (~150MB) is small, fast, and ships in
    `transformers` without extra deps. For Visual-GEPA we don't need a frontier
    visual encoder here — the role is only similarity geometry over key frames,
    which CLIP captures well enough.
  - We L2-normalize embeddings so cosine == dot product downstream.

Caching:
  - In B0 we hold embeddings in process; B2 will add a per-task disk cache
    (target ≥ 60% hit rate per EXPERIMENT_PLAN.md §5).
"""

from __future__ import annotations

import logging
from typing import Iterable, Sequence

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"


class CLIPImageEmbedder:
    """Wraps HuggingFace CLIPModel for image embeddings.

    Lazy-loads on first call so test imports don't trigger a model download.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_CLIP_MODEL,
        device: str | None = None,
        torch_dtype: str | None = None,
    ) -> None:
        self.model_name = model_name
        self._device = device
        self._torch_dtype = torch_dtype
        self._model = None
        self._processor = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch  # local import keeps tests import-free of heavy deps
        from transformers import CLIPModel, CLIPProcessor

        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"

        dtype = None
        if self._torch_dtype:
            dtype = getattr(torch, self._torch_dtype, None)

        logger.info(
            "loading CLIP embedder model=%s device=%s dtype=%s",
            self.model_name,
            self._device,
            self._torch_dtype,
        )
        self._processor = CLIPProcessor.from_pretrained(self.model_name)
        self._model = CLIPModel.from_pretrained(self.model_name, torch_dtype=dtype)
        self._model = self._model.to(self._device).eval()

    @property
    def embed_dim(self) -> int:
        self._ensure_loaded()
        return int(self._model.config.projection_dim)

    def encode_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        """Embed PIL images → (N, D) L2-normalized float32 ndarray."""
        if not images:
            return np.zeros((0, 0), dtype=np.float32)
        self._ensure_loaded()
        import torch

        # Convert any non-RGB to RGB (CLIP expects 3-channel).
        rgb_images = [im.convert("RGB") if im.mode != "RGB" else im for im in images]

        with torch.no_grad():
            inputs = self._processor(images=rgb_images, return_tensors="pt").to(self._device)
            out = self._model.get_image_features(**inputs)
            # In transformers >= 4.46, get_image_features can return a
            # BaseModelOutput-like wrapper instead of a raw tensor. Coerce.
            if not isinstance(out, torch.Tensor):
                for attr in ("image_embeds", "pooler_output", "last_hidden_state"):
                    candidate = getattr(out, attr, None)
                    if candidate is not None:
                        feats = candidate
                        break
                else:  # no break → no known attr
                    raise RuntimeError(
                        f"Unexpected CLIP output type {type(out).__name__}; "
                        "no image_embeds / pooler_output / last_hidden_state."
                    )
                # last_hidden_state is (B, T, D); pool over tokens.
                if feats.dim() == 3:
                    feats = feats.mean(dim=1)
            else:
                feats = out
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        return feats.detach().to("cpu", dtype=torch.float32).numpy()

    def encode_trajectory_frames(self, trajectory) -> np.ndarray:
        """Convenience: extract step screenshots in order → (T, D) ndarray."""
        frames: list[Image.Image] = []
        for step in getattr(trajectory, "steps", []) or []:
            img = getattr(step, "screenshot", None)
            if img is None:
                continue
            frames.append(img)
        return self.encode_images(frames)

    def encode_trajectory_means(self, trajectories: Iterable) -> np.ndarray:
        """Encode each trajectory as one mean-pooled vector → (N_traj, D)."""
        rows: list[np.ndarray] = []
        for traj in trajectories:
            per_frame = self.encode_trajectory_frames(traj)
            if per_frame.size == 0:
                # Defensive: emit a zero vector so KMeans can still ingest.
                rows.append(np.zeros(self.embed_dim, dtype=np.float32))
            else:
                vec = per_frame.mean(axis=0)
                n = float(np.linalg.norm(vec))
                rows.append(vec / max(n, 1e-12))
        if not rows:
            return np.zeros((0, self.embed_dim), dtype=np.float32)
        return np.stack(rows, axis=0)
