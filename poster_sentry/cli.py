"""
PosterSentry CLI — classify PDFs as scientific posters or non-posters.

Usage:
    poster-sentry classify document.pdf
    poster-sentry classify *.pdf --output results.tsv
    poster-sentry info
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import List


def classify_cmd(args):
    """Classify one or more PDFs."""
    from .classifier import PosterSentry

    sentry = PosterSentry(
        models_dir=Path(args.models_dir) if args.models_dir else None,
    )
    sentry.initialize()

    results = []
    for pdf_path in args.pdfs:
        p = Path(pdf_path)
        if not p.exists():
            print(f"  SKIP  {p.name}  (file not found)", file=sys.stderr)
            continue
        t0 = time.time()
        result = sentry.classify(str(p))
        elapsed = time.time() - t0
        result["elapsed_s"] = round(elapsed, 3)
        results.append(result)

        icon = "POSTER" if result["is_poster"] else "NON-POSTER"
        print(
            f"  {icon:11s}  conf={result['confidence']:.3f}  "
            f"({elapsed:.2f}s)  {p.name}"
        )

    if args.output:
        out = Path(args.output)
        with open(out, "w", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["path", "is_poster", "confidence"])
            for r in results:
                writer.writerow([r["path"], r["is_poster"], r["confidence"]])
        print(f"\nResults written to {out}")

    if args.json:
        print(json.dumps(results, indent=2))


def info_cmd(args):
    """Print model info."""
    from . import __version__

    print(f"PosterSentry v{__version__}")
    print(f"  Architecture:  model2vec (512-d) + visual (15-d) + structural (15-d)")
    print(f"  Total features: 542")
    print(f"  Classifier:    LogisticRegression + StandardScaler")
    print(f"  Embedding:     minishlab/potion-base-32M")
    print(f"  License:       MIT")
    print(f"  HuggingFace:   huggingface.co/fairdataihub/poster-sentry")


def main():
    parser = argparse.ArgumentParser(
        prog="poster-sentry",
        description="Classify PDFs as scientific posters or non-posters.",
    )
    subparsers = parser.add_subparsers(dest="command")

    # classify
    p_cls = subparsers.add_parser("classify", help="Classify PDF(s)")
    p_cls.add_argument("pdfs", nargs="+", help="PDF file(s) to classify")
    p_cls.add_argument("--output", "-o", help="Write TSV results to file")
    p_cls.add_argument("--json", action="store_true", help="Print JSON output")
    p_cls.add_argument("--models-dir", default=None, help="Path to models directory")
    p_cls.set_defaults(func=classify_cmd)

    # info
    p_info = subparsers.add_parser("info", help="Print model info")
    p_info.set_defaults(func=info_cmd)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
