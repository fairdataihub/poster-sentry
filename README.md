# PosterSentry

**Lightweight multimodal classifier for scientific poster quality control in open repositories.**

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-yellow)](https://huggingface.co/fairdataihub/poster-sentry)

<p align="center">
  <img src="PosterSentry.png" alt="PosterSentry" width="400">
</p>

Part of the quality control pipeline for [**posters.science**](https://posters.science), a platform for making scientific conference posters Findable, Accessible, Interoperable, and Reusable (FAIR).

Developed by the [**FAIR Data Innovations Hub**](https://fairdataihub.org/) at the California Medical Innovations Institute (CalMI2).

## The Problem

Open repositories like Zenodo and Figshare host tens of thousands of records labeled as scientific posters. However, approximately **20% of these records are mislabeled** — containing multi-page papers, conference proceedings, abstract booklets, slide decks, or other non-poster documents. This label noise is a significant barrier to automated poster processing at scale.

## Architecture

PosterSentry classifies PDFs using three complementary feature channels concatenated into a **542-dimensional** vector:

| Channel | Features | Dimensions | Signal |
|---------|----------|------------|--------|
| **Text** | model2vec (potion-base-32M) embedding | 512 | Semantic content |
| **Visual** | Color stats, edge density, FFT spatial complexity, whitespace | 15 | Visual layout |
| **Structural** | Page count, area, font diversity, text blocks, density | 15 | PDF geometry |

A StandardScaler normalizes all features (preventing the 512-d text embedding from drowning out structural/visual signal), then a LogisticRegression classifier produces the final prediction.

The classifier head is a single linear layer stored as a numpy `.npz` file (**10 KB**). Inference is pure numpy — no GPU or deep learning framework required.

## Performance

Validated on 3,606 real scientific documents (zero synthetic data):

| Metric | Value |
|--------|-------|
| **Accuracy** | **87.3%** |
| F1 (poster) | 87.1% |
| F1 (non-poster) | 87.4% |
| Precision (poster) | 88.2% |
| Recall (poster) | 85.9% |
| Inference speed | < 1 sec/PDF (CPU) |

Applied to 30,205 PDFs from Zenodo and Figshare, PosterSentry classified **80.2% as true posters** and 19.8% as non-posters, with mean confidence of 0.799.

### Top Discriminative Features

| Feature | Coefficient | Signal |
|---------|-------------|--------|
| `size_per_page_kb` | +7.65 | Posters are dense, high-res single pages |
| `page_count` | -5.49 | More pages = not a poster |
| `file_size_kb` | -5.44 | Multi-page docs are bigger overall |
| `is_landscape` | +0.98 | Some posters are landscape |
| `color_diversity` | +0.95 | Posters are visually rich |
| `edge_density` | +0.79 | More visual edges in posters |

## Quick Start

### Installation

```bash
pip install poster-sentry
```

### CLI Usage

```bash
# Classify a single PDF
poster-sentry classify document.pdf

# Classify multiple PDFs
poster-sentry classify *.pdf --output results.tsv

# Print model info
poster-sentry info
```

### Python API

```python
from poster_sentry import PosterSentry

sentry = PosterSentry()
sentry.initialize()

# Classify a PDF (uses text + visual + structural features)
result = sentry.classify("document.pdf")
print(f"Is poster: {result['is_poster']}, Confidence: {result['confidence']:.2f}")

# Batch classification
results = sentry.classify_batch(["poster1.pdf", "paper.pdf", "newsletter.pdf"])

# Text-only classification (no PDF needed)
result = sentry.classify_text("Title: My Poster\nAuthors: ...")
```

### Pipeline Position

PosterSentry sits at the front of the posters.science pipeline — it screens incoming PDFs before expensive LLM-based extraction:

```
PDF Input
   |
   v
PosterSentry          -->  poster2json                     -->  FAIR output
(classify: poster?)        (Llama 3.1 8B structured extraction)  (poster-json-schema)
```

## System Requirements

| Requirement | Value |
|-------------|-------|
| CPU | Any modern CPU (no GPU needed) |
| RAM | 4 GB+ |
| Python | 3.10+ |
| Model size | 10 KB head + ~60 MB embeddings (downloaded once) |

## Related Resources

| Resource | Description |
|----------|-------------|
| [poster-sentry (HuggingFace)](https://huggingface.co/fairdataihub/poster-sentry) | Model weights and config |
| [poster-sentry-training-data (HuggingFace)](https://huggingface.co/datasets/fairdataihub/poster-sentry-training-data) | Training dataset (3,606 samples) |
| [poster-sentry-training (GitHub)](https://github.com/fairdataihub/poster-sentry-training) | Training code and replication |
| [poster2json](https://github.com/fairdataihub/poster2json) | Poster to structured JSON extraction |
| [posters.science](https://posters.science) | Platform |

## Development

```bash
git clone https://github.com/fairdataihub/poster-sentry.git
cd poster-sentry
pip install -e ".[dev]"
pytest
```

## Citation

```bibtex
@software{poster_sentry_2026,
  title = {PosterSentry: Multimodal Scientific Poster Classifier},
  author = {O'Neill, James and Soundarajan, Sanjay and Portillo, Dorian and Patel, Bhavesh},
  year = {2026},
  url = {https://github.com/fairdataihub/poster-sentry},
  note = {Part of the posters.science initiative at FAIR Data Innovations Hub}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

- [FAIR Data Innovations Hub](https://fairdataihub.org/) at California Medical Innovations Institute (CalMI2)
- [posters.science](https://posters.science) platform
- [MinishLab](https://github.com/MinishLab) for the model2vec embedding backbone
- Funded by [The Navigation Fund](https://doi.org/10.71707/rk36-9x79) — "Poster Sharing and Discovery Made Easy"
