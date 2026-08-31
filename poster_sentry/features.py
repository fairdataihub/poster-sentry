"""
Feature extractors for PosterSentry.

Two feature channels:
    1. Visual features — image-level statistics (color, edges, FFT, whitespace)
    2. PDF structural features — page geometry, text blocks, font diversity

PDF access supports selectable backends:

    "pdfplumber" (default)
        pdfplumber for text and structure (MIT) and pypdfium2 for page
        rendering (BSD). This permissively licensed stack was used to train
        and evaluate the released model, so it is the default and the one the
        paper reports.

    "pymupdf"
        PyMuPDF (fitz) for text, structure, and rendering. Faster, but PyMuPDF
        is AGPL-licensed, and its extracted features differ slightly from the
        default, so the released model is matched to the "pdfplumber" backend.
        Install the optional dependency with `pip install poster-sentry[pymupdf]`.

Select the backend with the ``backend`` argument or the
``POSTER_SENTRY_BACKEND`` environment variable. The default is "pdfplumber".
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_BACKEND = os.environ.get("POSTER_SENTRY_BACKEND", "pdfplumber")
BACKENDS = ("pdfplumber", "pymupdf")


def extract_first_page_text(pdf_path: str, backend: str = None, max_chars: int = 4000) -> str:
    """Extract and clean first-page text with the selected backend."""
    import re
    backend = backend or DEFAULT_BACKEND
    try:
        if backend == "pymupdf":
            import fitz
            doc = fitz.open(pdf_path)
            text = doc[0].get_text() if len(doc) else ""
            doc.close()
        else:
            import pdfplumber
            with pdfplumber.open(pdf_path) as doc:
                text = (doc.pages[0].extract_text() or "") if doc.pages else ""
        return re.sub(r"\s+", " ", text).strip()[:max_chars]
    except Exception:
        return ""


# ── Visual Feature Extractor ────────────────────────────────────

VISUAL_FEATURE_NAMES = [
    "img_width",
    "img_height",
    "img_aspect_ratio",
    "mean_r", "mean_g", "mean_b",
    "std_r", "std_g", "std_b",
    "local_contrast",
    "color_diversity",
    "edge_density",
    "spatial_complexity",
    "white_space_ratio",
    "high_contrast_ratio",
]

N_VISUAL_FEATURES = len(VISUAL_FEATURE_NAMES)


class VisualFeatureExtractor:
    """Extract visual features from rendered PDF pages."""

    FEATURE_NAMES = VISUAL_FEATURE_NAMES

    def __init__(self, target_size: Tuple[int, int] = (256, 256), backend: str = None):
        self.target_size = target_size
        self.backend = backend or DEFAULT_BACKEND

    def pdf_to_image(self, pdf_path: str, dpi: int = 72) -> Optional[np.ndarray]:
        """Render first page of PDF to RGB numpy array with the selected backend."""
        try:
            if self.backend == "pymupdf":
                return self._render_pymupdf(pdf_path, dpi)
            return self._render_pypdfium2(pdf_path, dpi)
        except Exception as e:
            logger.debug(f"PDF to image failed ({self.backend}): {e}")
            return None

    @staticmethod
    def _render_pypdfium2(pdf_path: str, dpi: int) -> Optional[np.ndarray]:
        import pypdfium2 as pdfium
        pdf = pdfium.PdfDocument(pdf_path)
        if len(pdf) == 0:
            pdf.close()
            return None
        page = pdf[0]
        img = np.array(page.render(scale=dpi / 72.0).to_pil().convert("RGB"))
        page.close()
        pdf.close()
        return img

    @staticmethod
    def _render_pymupdf(pdf_path: str, dpi: int) -> Optional[np.ndarray]:
        import fitz
        doc = fitz.open(pdf_path)
        if len(doc) == 0:
            doc.close()
            return None
        page = doc[0]
        pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72))
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
            img = img[:, :, :3]
        elif pix.n == 1:
            img = np.stack([img[:, :, 0]] * 3, axis=-1)
        doc.close()
        return img

    def extract(self, image: np.ndarray) -> Dict[str, float]:
        """Extract 15 visual features from an RGB image."""
        feats = {n: 0.0 for n in self.FEATURE_NAMES}
        try:
            from PIL import Image as PILImage

            h, w = image.shape[:2]
            feats["img_width"] = float(w)
            feats["img_height"] = float(h)
            feats["img_aspect_ratio"] = w / h if h > 0 else 0.0

            pil = PILImage.fromarray(image).resize(self.target_size, PILImage.Resampling.BILINEAR)
            resized = np.array(pil)

            for i, ch in enumerate(["r", "g", "b"]):
                feats[f"mean_{ch}"] = float(np.mean(resized[:, :, i]))
                feats[f"std_{ch}"] = float(np.std(resized[:, :, i]))

            gray = np.mean(resized, axis=2)
            feats["local_contrast"] = float(np.std(gray))

            small = np.array(pil.resize((32, 32)))
            quantized = (small // 32).astype(np.uint8)
            unique_colors = len(np.unique(quantized.reshape(-1, 3), axis=0))
            feats["color_diversity"] = unique_colors / 512.0

            gy = np.abs(np.diff(gray, axis=0))
            gx = np.abs(np.diff(gray, axis=1))
            feats["edge_density"] = float(np.mean(gy) + np.mean(gx)) / 255.0

            fft = np.fft.fft2(gray)
            fft_shift = np.fft.fftshift(fft)
            mag = np.abs(fft_shift)
            ch, cw = mag.shape[0] // 2, mag.shape[1] // 2
            radius = min(mag.shape) // 4
            y, x = np.ogrid[:mag.shape[0], :mag.shape[1]]
            center_mask = ((y - ch) ** 2 + (x - cw) ** 2) <= radius ** 2
            total_e = np.sum(mag ** 2)
            low_e = np.sum(mag[center_mask] ** 2)
            feats["spatial_complexity"] = 1.0 - (low_e / total_e) if total_e > 0 else 0.0

            white_px = np.sum(np.all(resized > 240, axis=2))
            feats["white_space_ratio"] = white_px / (self.target_size[0] * self.target_size[1])

            feats["high_contrast_ratio"] = float(np.sum(gray < 50) + np.sum(gray > 240)) / gray.size

        except Exception as e:
            logger.debug(f"Visual feature extraction failed: {e}")
        return feats

    def to_vector(self, feats: Dict[str, float]) -> np.ndarray:
        return np.array([feats.get(n, 0.0) for n in self.FEATURE_NAMES], dtype="float32")


# ── PDF Structural Feature Extractor ────────────────────────────

STRUCTURAL_FEATURE_NAMES = [
    "page_count",
    "page_width_pt",
    "page_height_pt",
    "page_aspect_ratio",
    "page_area_sqin",
    "is_landscape",
    "text_block_count",
    "font_count",
    "avg_font_size",
    "font_size_variance",
    "title_score",
    "text_density",
    "line_count",
    "file_size_kb",
    "size_per_page_kb",
]

N_STRUCTURAL_FEATURES = len(STRUCTURAL_FEATURE_NAMES)


def _count_text_blocks(lines: List[dict]) -> int:
    """Group text lines into blocks by vertical spacing (pdfplumber backend)."""
    if not lines:
        return 0
    lines = sorted(lines, key=lambda ln: ln["top"])
    heights = [max(ln["bottom"] - ln["top"], 1.0) for ln in lines]
    median_h = float(np.median(heights))
    blocks = 1
    for prev, cur in zip(lines, lines[1:]):
        if cur["top"] - prev["bottom"] > median_h:
            blocks += 1
    return blocks


class PDFStructuralExtractor:
    """Extract structural features from PDF layout."""

    FEATURE_NAMES = STRUCTURAL_FEATURE_NAMES

    def __init__(self, backend: str = None):
        self.backend = backend or DEFAULT_BACKEND

    def extract(self, pdf_path: str) -> Dict[str, float]:
        """Extract 15 structural features from a PDF with the selected backend."""
        feats = {n: 0.0 for n in self.FEATURE_NAMES}
        try:
            if self.backend == "pymupdf":
                return self._extract_pymupdf(pdf_path, feats)
            return self._extract_pdfplumber(pdf_path, feats)
        except Exception as e:
            logger.debug(f"PDF structural extraction failed ({self.backend}): {e}")
        return feats

    def _extract_pdfplumber(self, pdf_path: str, feats: Dict[str, float]) -> Dict[str, float]:
        import pdfplumber
        path = Path(pdf_path)
        with pdfplumber.open(str(path)) as doc:
            n_pages = len(doc.pages)
            if n_pages == 0:
                return feats
            feats["page_count"] = float(n_pages)
            feats["file_size_kb"] = path.stat().st_size / 1024.0
            feats["size_per_page_kb"] = feats["file_size_kb"] / max(n_pages, 1)

            page = doc.pages[0]
            pw, ph = float(page.width), float(page.height)
            feats["page_width_pt"] = pw
            feats["page_height_pt"] = ph
            feats["page_aspect_ratio"] = pw / ph if ph > 0 else 0.0
            feats["page_area_sqin"] = (pw / 72.0) * (ph / 72.0)
            feats["is_landscape"] = float(pw > ph)

            chars = page.chars
            words = page.extract_words()
            lines = page.extract_text_lines()
            feats["text_block_count"] = float(_count_text_blocks(lines))
            feats["line_count"] = float(len(lines))
            if words:
                word_area = sum(
                    max(wd["x1"] - wd["x0"], 0.0) * max(wd["bottom"] - wd["top"], 0.0)
                    for wd in words
                )
                page_area = pw * ph
                feats["text_density"] = word_area / page_area if page_area > 0 else 0.0
            fonts = {c.get("fontname", "") for c in chars if c.get("fontname")}
            font_sizes = [float(c["size"]) for c in chars if c.get("size", 0)]
            feats["font_count"] = float(len(fonts))
            if font_sizes:
                feats["avg_font_size"] = float(np.mean(font_sizes))
                feats["font_size_variance"] = float(np.var(font_sizes)) if len(font_sizes) > 1 else 0.0
                feats["title_score"] = max(font_sizes) / (float(np.mean(font_sizes)) + 1.0)
        return feats

    def _extract_pymupdf(self, pdf_path: str, feats: Dict[str, float]) -> Dict[str, float]:
        import fitz
        path = Path(pdf_path)
        doc = fitz.open(str(path))
        if len(doc) == 0:
            doc.close()
            return feats
        feats["page_count"] = float(len(doc))
        feats["file_size_kb"] = path.stat().st_size / 1024.0
        feats["size_per_page_kb"] = feats["file_size_kb"] / max(len(doc), 1)

        page = doc[0]
        rect = page.rect
        feats["page_width_pt"] = rect.width
        feats["page_height_pt"] = rect.height
        feats["page_aspect_ratio"] = rect.width / rect.height if rect.height > 0 else 0.0
        feats["page_area_sqin"] = (rect.width / 72.0) * (rect.height / 72.0)
        feats["is_landscape"] = float(rect.width > rect.height)

        blocks = page.get_text("dict")["blocks"]
        text_blocks = [b for b in blocks if b.get("type") == 0]
        feats["text_block_count"] = float(len(text_blocks))
        if text_blocks:
            heights = [b["bbox"][3] - b["bbox"][1] for b in text_blocks]
            widths = [b["bbox"][2] - b["bbox"][0] for b in text_blocks]
            total_area = sum(h * w for h, w in zip(heights, widths))
            page_area = rect.width * rect.height
            feats["text_density"] = total_area / page_area if page_area > 0 else 0.0

        fonts = set()
        font_sizes = []
        line_count = 0
        for block in text_blocks:
            for line in block.get("lines", []):
                line_count += 1
                for span in line.get("spans", []):
                    fonts.add(span.get("font", ""))
                    sz = span.get("size", 0)
                    if sz > 0:
                        font_sizes.append(sz)
        feats["font_count"] = float(len(fonts))
        feats["line_count"] = float(line_count)
        if font_sizes:
            feats["avg_font_size"] = float(np.mean(font_sizes))
            feats["font_size_variance"] = float(np.var(font_sizes)) if len(font_sizes) > 1 else 0.0
            feats["title_score"] = max(font_sizes) / (np.mean(font_sizes) + 1.0)
        doc.close()
        return feats

    def to_vector(self, feats: Dict[str, float]) -> np.ndarray:
        return np.array([feats.get(n, 0.0) for n in self.FEATURE_NAMES], dtype="float32")
