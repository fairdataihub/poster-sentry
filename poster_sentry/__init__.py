"""
PosterSentry — Multimodal Scientific Poster Classifier
=======================================================

Classifies PDFs as scientific posters vs. non-posters using both
text embeddings (model2vec) and visual features (image analysis).

Trained on 30K+ real scientific posters from Zenodo and Figshare
via the posters.science initiative at FAIR Data Innovations Hub.

Usage:
    from poster_sentry import PosterSentry

    sentry = PosterSentry()
    sentry.initialize()
    result = sentry.classify("document.pdf")
    # {'is_poster': True, 'confidence': 0.97, 'text_score': 0.95, 'visual_score': 0.99}
"""

from .classifier import PosterSentry
from .features import VisualFeatureExtractor, PDFStructuralExtractor

__version__ = "0.1.0"
__all__ = ["PosterSentry", "VisualFeatureExtractor", "PDFStructuralExtractor"]
