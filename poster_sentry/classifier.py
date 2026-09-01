"""
PosterSentry — Multimodal Scientific Poster Classifier
=======================================================

Architecture:
    ┌──────────┐    ┌──────────────┐    ┌───────────────┐
    │ PDF text │    │ PDF → image  │    │ PDF structure  │
    └────┬─────┘    └──────┬───────┘    └───────┬───────┘
         │                 │                    │
    model2vec         15 visual            15 structural
    → 512-d emb       features             features
         │                 │                    │
         └────────┬────────┴────────────────────┘
                  │
          concat → 542-d input
                  │
          LogisticRegression
                  │
         poster / non_poster

Single linear classifier on the concatenated feature vector.
Same paradigm as PubGuard — lightweight, CPU-only, fast.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .features import (
    VisualFeatureExtractor,
    PDFStructuralExtractor,
    N_VISUAL_FEATURES,
    N_STRUCTURAL_FEATURES,
)

logger = logging.getLogger(__name__)


class PosterSentry:
    """
    Multimodal poster classifier.

    Combines:
        - model2vec text embedding (512-d)
        - 15 visual features (color, edge, FFT, whitespace)
        - 15 structural features (page geometry, fonts, text blocks)

    into a single 542-d feature vector for logistic regression.
    """

    def __init__(
        self,
        model_name: str = "minishlab/potion-base-32M",
        models_dir: Optional[Path] = None,
        backend: str = None,
    ):
        self.model_name = model_name
        self.models_dir = models_dir or self._default_models_dir()
        self.models_dir = Path(self.models_dir)

        # PDF backend: "pdfplumber" (default, matches the released model) or
        # "pymupdf" (faster, AGPL, optional dependency). See features.py.
        from .features import DEFAULT_BACKEND
        self.backend = backend or DEFAULT_BACKEND

        self.text_model = None
        self.W: Optional[np.ndarray] = None
        self.b: Optional[np.ndarray] = None
        self.scaler_mean: Optional[np.ndarray] = None
        self.scaler_scale: Optional[np.ndarray] = None
        self.labels = ["non_poster", "poster"]

        # Head architecture. "stacked" heads score the text embedding with a
        # stage-one logistic regression whose poster probability becomes a
        # single text-score feature for the final classifier over
        # [text_score + 15 visual + 15 structural]; "flat" heads classify the
        # raw 542-dimensional concatenation and remain loadable.
        self.arch = "flat"
        self.s1_W: Optional[np.ndarray] = None
        self.s1_b: Optional[np.ndarray] = None
        self.s1_scaler_mean: Optional[np.ndarray] = None
        self.s1_scaler_scale: Optional[np.ndarray] = None

        self.visual_extractor = VisualFeatureExtractor(backend=self.backend)
        self.structural_extractor = PDFStructuralExtractor(backend=self.backend)
        self._initialized = False

    @staticmethod
    def _default_models_dir() -> Path:
        import os
        if env := os.environ.get("POSTER_SENTRY_MODELS_DIR"):
            return Path(env)
        # Check for bundled models shipped with the package
        pkg_models = Path(__file__).resolve().parent.parent / "models"
        if (pkg_models / "poster_sentry_head.npz").exists():
            return pkg_models
        # Fall back to user-local cache
        home = Path.home() / ".poster_sentry" / "models"
        home.mkdir(parents=True, exist_ok=True)
        return home

    # ── Initialization ──────────────────────────────────────────

    def initialize(self) -> bool:
        if self._initialized:
            return True
        logger.info("Initializing PosterSentry...")
        t0 = time.time()
        self._load_text_model()
        self._load_head()
        self._initialized = True
        logger.info(f"PosterSentry initialized in {time.time()-t0:.1f}s")
        return True

    def _load_text_model(self):
        from model2vec import StaticModel
        cache = self.models_dir / "poster-sentry-embedding"
        if cache.exists():
            self.text_model = StaticModel.from_pretrained(str(cache))
        else:
            self.text_model = StaticModel.from_pretrained(self.model_name)
            cache.parent.mkdir(parents=True, exist_ok=True)
            self.text_model.save_pretrained(str(cache))

    def _load_head(self):
        path = self.models_dir / "poster_sentry_head.npz"
        if path.exists():
            data = np.load(path, allow_pickle=True)
            self.arch = str(data["arch"]) if "arch" in data.files else "flat"
            self.W = data["W"]
            self.b = data["b"]
            if "labels" in data:
                self.labels = list(data["labels"])
            if "scaler_mean" in data and "scaler_scale" in data:
                self.scaler_mean = data["scaler_mean"]
                self.scaler_scale = data["scaler_scale"]
            if self.arch == "stacked":
                self.s1_W = data["s1_W"]
                self.s1_b = data["s1_b"]
                self.s1_scaler_mean = data["s1_scaler_mean"]
                self.s1_scaler_scale = data["s1_scaler_scale"]
            logger.info(f"  Loaded classifier head ({self.arch}): {path}")
        else:
            logger.warning(f"  Head not found: {path} — run training first")

    def save_head(self, path: Optional[Path] = None):
        path = path or (self.models_dir / "poster_sentry_head.npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(W=self.W, b=self.b, labels=np.array(self.labels),
                       arch=np.array(self.arch))
        if self.scaler_mean is not None:
            payload.update(scaler_mean=self.scaler_mean, scaler_scale=self.scaler_scale)
        if self.arch == "stacked":
            payload.update(s1_W=self.s1_W, s1_b=self.s1_b,
                           s1_scaler_mean=self.s1_scaler_mean,
                           s1_scaler_scale=self.s1_scaler_scale)
        np.savez(path, **payload)

    # ── Feature extraction ──────────────────────────────────────

    def extract_text(self, pdf_path: str, max_chars: int = 4000) -> str:
        """Extract and clean first-page text with the selected backend."""
        from .features import extract_first_page_text
        return extract_first_page_text(pdf_path, backend=self.backend, max_chars=max_chars)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Encode texts with model2vec, L2-normalize."""
        embeddings = self.text_model.encode(texts)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return (embeddings / norms).astype("float32")

    def build_feature_vector(
        self,
        text_emb: np.ndarray,
        visual_feats: np.ndarray,
        structural_feats: np.ndarray,
    ) -> np.ndarray:
        """Concatenate all features: [512 text + 15 visual + 15 structural] = 542."""
        return np.concatenate([text_emb, visual_feats, structural_feats])

    # ── Inference ───────────────────────────────────────────────

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        e = np.exp(logits - logits.max(axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)

    def _poster_col(self) -> int:
        try:
            return self.labels.index("poster")
        except ValueError:
            return 1

    def _score_features(self, text_embs, visual_arr, struct_arr):
        """Class probabilities (n, 2) and, for stacked heads, the text score."""
        ip = self._poster_col()
        text_scores = None
        if self.arch == "stacked":
            T = (text_embs - self.s1_scaler_mean) / np.where(
                self.s1_scaler_scale == 0, 1, self.s1_scaler_scale)
            text_scores = self._softmax(T @ self.s1_W + self.s1_b)[:, ip]
            X = np.concatenate(
                [text_scores[:, None].astype("float32"), visual_arr, struct_arr], axis=1)
        else:
            X = np.concatenate([text_embs, visual_arr, struct_arr], axis=1)
        if self.scaler_mean is not None and self.scaler_scale is not None:
            X = (X - self.scaler_mean) / np.where(self.scaler_scale == 0, 1, self.scaler_scale)
        probs = self._softmax(X @ self.W + self.b)
        return probs, text_scores

    def classify(self, pdf_path: str) -> Dict[str, Any]:
        """Classify a single PDF as poster or non-poster."""
        if not self._initialized:
            self.initialize()
        return self.classify_batch([pdf_path])[0]

    def classify_batch(self, pdf_paths: List[str]) -> List[Dict[str, Any]]:
        """Classify a batch of PDFs."""
        if not self._initialized:
            self.initialize()

        texts = []
        visual_vecs = []
        structural_vecs = []

        for p in pdf_paths:
            texts.append(self.extract_text(p))

            img = self.visual_extractor.pdf_to_image(p)
            if img is not None:
                vf = self.visual_extractor.extract(img)
            else:
                vf = {n: 0.0 for n in self.visual_extractor.FEATURE_NAMES}
            visual_vecs.append(self.visual_extractor.to_vector(vf))

            sf = self.structural_extractor.extract(p)
            structural_vecs.append(self.structural_extractor.to_vector(sf))

        # Embed text
        text_embs = self.embed_texts(texts)
        visual_arr = np.array(visual_vecs, dtype="float32")
        struct_arr = np.array(structural_vecs, dtype="float32")

        if self.W is None:
            return [{"path": p, "is_poster": False, "confidence": 0.0,
                     "error": "Model not trained"} for p in pdf_paths]

        probs, text_scores = self._score_features(text_embs, visual_arr, struct_arr)
        ip = self._poster_col()

        results = []
        for i, p in enumerate(pdf_paths):
            poster_prob = float(probs[i, ip])
            results.append({
                "path": str(p),
                "is_poster": poster_prob > 0.5,
                "confidence": round(poster_prob, 4),
                "text_score": round(float(text_scores[i]) if text_scores is not None
                                    else poster_prob, 4),
            })
        return results

    # ── Text-only classification (for PubGuard integration) ─────

    def classify_text(self, text: str) -> Dict[str, Any]:
        """Classify from text alone (no PDF needed). Used by PubGuard."""
        return self.classify_texts([text])[0]

    def classify_texts(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Classify from text alone (batch)."""
        if not self._initialized:
            self.initialize()
        if self.W is None:
            return [{"is_poster": False, "confidence": 0.0}] * len(texts)

        text_embs = self.embed_texts(texts)
        # Zero-fill visual and structural features
        zeros_visual = np.zeros((len(texts), N_VISUAL_FEATURES), dtype="float32")
        zeros_struct = np.zeros((len(texts), N_STRUCTURAL_FEATURES), dtype="float32")
        probs, _ = self._score_features(text_embs, zeros_visual, zeros_struct)
        ip = self._poster_col()

        return [{"is_poster": float(probs[i, ip]) > 0.5,
                 "confidence": round(float(probs[i, ip]), 4)}
                for i in range(len(texts))]
