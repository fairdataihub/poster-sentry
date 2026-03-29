"""Basic tests for PosterSentry feature extractors."""

import numpy as np

from poster_sentry.features import (
    VisualFeatureExtractor,
    PDFStructuralExtractor,
    N_VISUAL_FEATURES,
    N_STRUCTURAL_FEATURES,
)


def test_visual_feature_count():
    assert N_VISUAL_FEATURES == 15


def test_structural_feature_count():
    assert N_STRUCTURAL_FEATURES == 15


def test_visual_extractor_zeros():
    ext = VisualFeatureExtractor()
    feats = {n: 0.0 for n in ext.FEATURE_NAMES}
    vec = ext.to_vector(feats)
    assert vec.shape == (15,)
    assert vec.dtype == np.float32


def test_structural_extractor_zeros():
    ext = PDFStructuralExtractor()
    feats = {n: 0.0 for n in ext.FEATURE_NAMES}
    vec = ext.to_vector(feats)
    assert vec.shape == (15,)
    assert vec.dtype == np.float32


def test_visual_extract_synthetic_image():
    ext = VisualFeatureExtractor()
    img = np.random.randint(0, 255, (400, 600, 3), dtype=np.uint8)
    feats = ext.extract(img)
    assert len(feats) == N_VISUAL_FEATURES
    assert feats["img_width"] == 600.0
    assert feats["img_height"] == 400.0
    assert feats["img_aspect_ratio"] == 600.0 / 400.0
