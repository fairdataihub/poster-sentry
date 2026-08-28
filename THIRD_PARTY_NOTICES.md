# Third-Party Notices

PosterSentry is released under the MIT License (see `LICENSE`). It depends at
runtime on the following third-party packages, all under permissive licenses
that are compatible with the MIT License and impose no copyleft obligations.

| Package | License | Role |
|---------|---------|------|
| numpy | BSD-3-Clause | numerics |
| scikit-learn | BSD-3-Clause | logistic regression, feature scaling |
| model2vec | MIT | static text embeddings |
| Pillow | HPND (MIT-style) | image handling |
| pdfplumber | MIT | PDF text and structure extraction |
| pypdfium2 | Apache-2.0 OR BSD-3-Clause | PDF page rendering |

## pypdfium2 and PDFium

`pypdfium2` provides bindings to the PDFium rendering engine and is licensed
under `Apache-2.0 OR BSD-3-Clause`. Its wheels bundle a compiled build of
PDFium, which is licensed under the BSD-3-Clause license by the PDFium Authors
and incorporates further third-party components (including libpng, zlib, and
FreeType) under their own permissive licenses.

The full texts of these licenses are distributed with the `pypdfium2` package,
in its `dist-info` directory:

- `Apache-2.0.txt`
- `BSD-3-Clause.txt`
- `LicenseRef-PdfiumThirdParty.txt` (PDFium and its bundled third-party components)

None of these licenses require PosterSentry, or software that uses it, to be
released under any particular license.

## Note on the previous PDF backend

Earlier releases used PyMuPDF for PDF access. PyMuPDF is distributed under the
AGPL-3.0 license, which is incompatible with PosterSentry's MIT license, so the
PDF backend was moved to pdfplumber (text and structure) and pypdfium2 (page
rendering) in release 1.1.0.
