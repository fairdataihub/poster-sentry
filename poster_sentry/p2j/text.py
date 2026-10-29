"""First-page text extraction, vendored from poster2json 0.10.5.

This module is a verbatim copy of the pdfplumber text path of poster2json
(``poster2json.extract.extract_text_with_pdfplumber`` and the helpers it calls,
with its "alto" layout engine in ``layout_streams``), by the FAIR Data
Innovations Hub, MIT license: https://github.com/fairdataihub/poster2json
(commit dcb856f). It is vendored so that PosterSentry keeps its small,
permissively licensed dependency set; poster2json itself depends on torch,
transformers, and PyMuPDF. Changes from upstream: imports are relative,
logging goes to the module logger instead of stdout, environment overrides
(POSTER2JSON_LAYOUT, POSTER2JSON_CAPTIONS) are not read, caption detection
is not included, and ``max_pages`` limits extraction to the first pages.
"""

import logging
import time
import math
import re
import unicodedata
from typing import Optional, Tuple

import numpy as np
import pdfplumber

_logger = logging.getLogger(__name__)


_TRUTHY_ENV = {"1", "true", "yes", "on"}


CAPTION_ENGINE = False


CAPTION_RAW_MODE = "marker"


_SECTION_KEYWORDS = frozenset({
    "abstract", "introduction", "background", "objective", "objectives",
    "aim", "aims", "hypothesis", "methods", "methodology",
    "materials and methods", "results", "results and discussion",
    "discussion", "conclusion", "conclusions", "summary",
    "references", "reference", "bibliography",
    "acknowledgements", "acknowledgments", "acknowledgement",
    "funding", "limitations", "future work", "future directions",
    "contact", "contact information",
})


_INLINE_LABELS = (
    "REFERENCES", "REFERENCE", "BIBLIOGRAPHY", "ABBREVIATIONS",
    "DISCLOSURES", "DISCLOSURE", "ACKNOWLEDGEMENTS", "ACKNOWLEDGMENTS",
    "FUNDING", "CONFLICTS OF INTEREST", "CONFLICT OF INTEREST",
    "COMPETING INTERESTS", "CORRESPONDING AUTHOR", "CONTACT",
    "CONCLUSIONS", "CONCLUSION",
)


_INLINE_LABEL_RE = re.compile(
    r"(?:(?<=[\s.;)])|^)(" + "|".join(_INLINE_LABELS) + r")\s*:\s+"
)


_INLINE_DASH_HEAD_RE = re.compile(
    r"(?:(?<=[.;])\s+|^)([A-Z][A-Z]{2,})\s*[—–]\s+(?=[A-Z])"
)


def log(msg: str):
    """Debug logging (poster2json prints a timestamped line)."""
    _logger.debug(msg)


def _parse_font_style(fontname: str) -> dict:
    """Parse bold/italic from a pdfplumber fontname string.

    Font names follow the pattern 'SUBSET+FamilyName-Style' where style
    contains 'Bold', 'Italic', 'BoldItalic', etc.
    """
    lower = fontname.lower()
    suffix = lower.rsplit("-", 1)[-1] if "-" in lower else ""
    return {
        "bold": ("bold" in lower or "black" in lower or "heavy" in lower
                 or "bd" in suffix or "demi" in suffix),
        "italic": ("italic" in lower or "oblique" in lower or "it" in suffix),
    }


def _filter_phantom_spaces(chars: list) -> list:
    """Remove space characters fully contained within a non-space character.

    Some PDFs embed invisible space glyphs on top of real characters, causing
    pdfplumber's extract_words to split words at those phantom boundaries
    (e.g. "For" → "F or").  A phantom space is one whose entire horizontal
    extent [x0, x1] falls within a non-space character's [x0, x1] on the
    same line.  Legitimate word-separating spaces extend beyond the adjacent
    character's bounds and are preserved.
    """
    non_space = [(c["x0"], c["x1"], c["top"]) for c in chars if c["text"].strip()]
    if not non_space:
        return chars

    from bisect import bisect_right

    ns_by_row = {}
    for x0, x1, top in non_space:
        row = round(top)
        ns_by_row.setdefault(row, []).append((x0, x1))
    for row in ns_by_row:
        ns_by_row[row].sort()

    filtered = []
    for c in chars:
        if not c["text"].strip():
            row = round(c["top"])
            is_phantom = False
            for r in (row - 1, row, row + 1):
                intervals = ns_by_row.get(r)
                if not intervals:
                    continue
                idx = bisect_right(intervals, (c["x0"], float("inf"))) - 1
                if idx >= 0 and intervals[idx][0] <= c["x0"] + 1 and c["x1"] <= intervals[idx][1] + 1:
                    is_phantom = True
                    break
            if is_phantom:
                continue
        filtered.append(c)
    return filtered


def _annotate_words_with_fonts(words: list, chars: list) -> None:
    """Add fontname to each word from overlapping raw characters.

    pdfplumber's extract_words groups chars by extra_attrs, which means
    including fontname would break words at font-subset boundaries (e.g.
    ligature ﬀ in a different subset splitting "different" → "di|ff|erent").
    We extract with extra_attrs=["size"] only, then look up each word's
    dominant fontname from the raw chars.
    """
    from bisect import bisect_left, bisect_right

    non_space = [c for c in chars if c["text"].strip()]
    if not non_space:
        return
    non_space.sort(key=lambda c: (round(c["top"]), c["x0"]))
    tops = [round(c["top"]) for c in non_space]

    for w in words:
        w_top = round(w["top"])
        lo = bisect_left(tops, w_top - 2)
        hi = bisect_right(tops, w_top + 2)
        matching = [
            c for c in non_space[lo:hi]
            if c["x0"] >= w["x0"] - 1 and c["x1"] <= w["x1"] + 1
        ]
        if matching:
            fonts = [c["fontname"] for c in matching if c.get("fontname")]
            w["fontname"] = max(set(fonts), key=fonts.count) if fonts else ""
        else:
            w["fontname"] = ""


def _line_is_bold(line: list) -> bool:
    """Check if a line's dominant font is bold."""
    styles = [_parse_font_style(w.get("fontname", "")) for w in line]
    bold_count = sum(1 for s in styles if s["bold"])
    return bold_count > len(styles) / 2 if styles else False


def _line_dominant_size(line: list) -> float:
    """Get the most common font size in a line."""
    sizes = [w["size"] for w in line if w.get("size", 0) > 0]
    if not sizes:
        return 0
    return max(set(sizes), key=sizes.count)


def _lines_to_blocks(lines: list, line_height_mult: float = 1.5,
                     leaf_ids: Optional[list] = None,
                     hyphen_join: bool = False,
                     orders: Optional[list] = None) -> list:
    """Group word lines (in reading order) into blocks.

    A new block starts when:
      - The vertical gap exceeds line_height_mult * median line height, OR
      - The font style changes (bold to non-bold or vice versa), OR
      - The font size jumps significantly (> 1.3x ratio), OR
      - (when leaf_ids is given, one XY-cut column region id per line) the
        line belongs to a different column region than the previous line, or
        starts clearly above the previous line (an upward jump means a column
        or flow change, never a continuation)

    With hyphen_join, words hyphenated at line ends are repaired when the
    lines of a block are joined.
    """
    if not lines:
        return []

    line_heights = []
    for line in lines:
        h = max(w["bottom"] for w in line) - min(w["top"] for w in line)
        line_heights.append(max(h, 1.0))
    median_lh = sorted(line_heights)[len(line_heights) // 2]
    gap_threshold = median_lh * line_height_mult

    vocab = None
    if hyphen_join:
        from .layout_blocks import page_vocab
        vocab = page_vocab([" ".join(w["text"] for w in ln) for ln in lines])

    block_groups = [[lines[0]]]
    group_region = [leaf_ids[0] if leaf_ids is not None else 0]
    group_order = [orders[0] if orders is not None else 0]
    for i in range(1, len(lines)):
        prev_bottom = max(w["bottom"] for w in lines[i - 1])
        curr_top = min(w["top"] for w in lines[i])
        gap = curr_top - prev_bottom

        prev_bold = _line_is_bold(lines[i - 1])
        curr_bold = _line_is_bold(lines[i])
        prev_size = _line_dominant_size(lines[i - 1])
        curr_size = _line_dominant_size(lines[i])
        size_ratio = max(curr_size, prev_size) / max(min(curr_size, prev_size), 0.1)

        style_break = False
        if size_ratio > 1.3:
            style_break = True
        elif prev_bold != curr_bold and gap > 0:
            curr_text = " ".join(w["text"] for w in lines[i])
            prev_text = " ".join(w["text"] for w in lines[i - 1])
            short_line = len(curr_text) <= 120 or len(prev_text) <= 120
            if short_line:
                style_break = True

        leaf_break = False
        if leaf_ids is not None:
            prev_top = min(w["top"] for w in lines[i - 1])
            prev_h = max(prev_bottom - prev_top, 1.0)
            # Different column region: break, unless the line continues the
            # previous one on the same row (e.g. a bullet or number in its
            # own cell next to its text).
            same_row = (abs(curr_top - prev_top) < 0.5 * prev_h
                        and min(w["x0"] for w in lines[i])
                        >= max(w["x1"] for w in lines[i - 1]) - 1.0)
            leaf_break = ((leaf_ids[i] != leaf_ids[i - 1] and not same_row)
                          or curr_top < prev_top - 0.25 * prev_h)
            if leaf_break and not same_row:
                from .layout_blocks import continues_paragraph
                # a paragraph can run from the bottom of one column into the
                # top of the next; keep that as one block
                if continues_paragraph(" ".join(w["text"] for w in lines[i - 1]),
                                       " ".join(w["text"] for w in lines[i])):
                    leaf_break = False

        if gap > gap_threshold or style_break or leaf_break:
            block_groups.append([lines[i]])
            group_region.append(leaf_ids[i] if leaf_ids is not None else 0)
            group_order.append(orders[i] if orders is not None else i)
        else:
            block_groups[-1].append(lines[i])

    blocks = []
    for group, region, order in zip(block_groups, group_region, group_order):
        all_words = [w for line in group for w in line]
        x0 = min(w["x0"] for w in all_words)
        top = min(w["top"] for w in all_words)
        x1 = max(w["x1"] for w in all_words)
        bottom = max(w["bottom"] for w in all_words)
        bw = x1 - x0
        bh = bottom - top
        cx = x0 + bw / 2

        seen_texts = []
        deduped_lines = []
        for line in group:
            line_top = min(w["top"] for w in line)
            line_words = set(w["text"] for w in line)
            line_text = " ".join(w["text"] for w in line)

            is_dup = False
            for sv, sw, _ in seen_texts:
                if abs(sv - line_top) < 2:
                    if line_words and sw:
                        overlap = len(line_words & sw) / max(len(line_words), len(sw))
                        if overlap > 0.8:
                            is_dup = True
                            break
            if not is_dup:
                seen_texts.append((line_top, line_words, line_text))
                deduped_lines.append(line_text)

        if not deduped_lines:
            continue

        if hyphen_join:
            from .layout_blocks import join_lines
            block_text = join_lines(deduped_lines, vocab)
        else:
            block_text = " ".join(deduped_lines)

        fontsizes = [w["size"] for w in all_words if w.get("size", 0) > 0]
        fontnames = [w["fontname"] for w in all_words if w.get("fontname")]

        dominant_size = max(set(fontsizes), key=fontsizes.count) if fontsizes else 0
        styles = [_parse_font_style(fn) for fn in fontnames]
        bold_count = sum(1 for s in styles if s["bold"])
        is_bold = bold_count > len(styles) / 2 if styles else False

        blocks.append({
            "hpos": x0,
            "vpos": top,
            "width": bw,
            "height": bh,
            "cx": cx,
            "text": block_text,
            "fontsize": dominant_size,
            "bold": is_bold,
            "region": region,
            "order": order,
        })

    return blocks


def _chars_to_word_lines(char_lines, words, leaf_ids=None):
    """Map XY-cut char-level lines back to word-level lines.

    For each char-level line, find the words whose characters
    predominantly fall on that line (by spatial overlap).  Legacy path used
    when layout_blocks.FLAGS["decomp"] is off.  With leaf_ids (one per char
    line) the result is (word_lines, leaf id per word line); words that no
    line claimed get leaf id -1.
    """
    from bisect import bisect_left, bisect_right

    word_tops = sorted(set(w["top"] for w in words))
    word_by_pos = {}
    for w in words:
        key = (round(w["top"], 1), round(w["x0"], 1))
        word_by_pos[key] = w

    assigned = set()
    result = []
    result_leaf = []
    for li, char_line in enumerate(char_lines):
        if not char_line:
            continue
        line_y_min = min(c["top"] for c in char_line)
        line_y_max = max(c["bottom"] for c in char_line)
        line_x_min = min(c["x0"] for c in char_line)
        line_x_max = max(c["x1"] for c in char_line)

        line_words = []
        for w in words:
            if id(w) in assigned:
                continue
            w_cy = (w["top"] + w["bottom"]) / 2
            w_cx = (w["x0"] + w["x1"]) / 2
            if (line_y_min - 2 <= w_cy <= line_y_max + 2
                    and line_x_min - 5 <= w_cx <= line_x_max + 5):
                line_words.append(w)
                assigned.add(id(w))

        if line_words:
            line_words.sort(key=lambda w: w["x0"])
            result.append(line_words)
            result_leaf.append(leaf_ids[li] if leaf_ids is not None else 0)

    unassigned = [w for w in words if id(w) not in assigned]
    if unassigned:
        unassigned.sort(key=lambda w: (w["top"], w["x0"]))
        cur = [unassigned[0]]
        for w in unassigned[1:]:
            if abs(w["top"] - cur[0]["top"]) <= 3:
                cur.append(w)
            else:
                result.append(sorted(cur, key=lambda w: w["x0"]))
                result_leaf.append(-1)
                cur = [w]
        if cur:
            result.append(sorted(cur, key=lambda w: w["x0"]))
            result_leaf.append(-1)

    if leaf_ids is not None:
        return result, result_leaf
    return result


def _split_inline_sections(text: str):
    """Split a run-on block at inline ALL-CAPS section labels and em-dash
    sub-headers. Returns a list of (heading_or_None, body) segments, or None
    if the block contains no inline boundary (the common case)."""
    marks = []
    for m in _INLINE_LABEL_RE.finditer(text):
        marks.append((m.start(), m.end(), m.group(1).title()))
    for m in _INLINE_DASH_HEAD_RE.finditer(text):
        marks.append((m.start(), m.end(), m.group(1).title()))
    if not marks:
        return None
    marks.sort()
    segments = []
    pre = text[:marks[0][0]].strip()
    if pre:
        segments.append((None, pre))
    for i, (_s, end, heading) in enumerate(marks):
        nxt = marks[i + 1][0] if i + 1 < len(marks) else len(text)
        segments.append((heading, text[end:nxt].strip()))
    return segments


def _xy_cut_blocks(page, chars, text_chars, x_tol, page_w, page_h,
                   geo: Optional[dict] = None) -> Tuple[list, list]:
    """Blocks and figure-label blocks for one page from the XY-cut tree and
    leaf-aware line builder."""
    from pdfplumber.utils.text import extract_words as _pw_extract_words
    from . import layout_blocks as _lb
    from . import xy_cut as _xy

    leaf_ids = None
    if _lb.FLAGS["decomp"]:
        # One decomposition: lines and words come from the same chars
        # of each XY-cut leaf, so every word belongs to exactly one line.
        rot_chars = []
        main_chars = chars
        if _lb.FLAGS["rotated"]:
            main_chars, rot_chars = _lb.split_rotated(chars)
        # The XY-cut sees every char (the cut tree is the same with or
        # without rotated labels); rotated chars are pulled out of the
        # leaves afterwards.
        leaves, regions = _xy.chars_to_leaves(
            chars, page_width=page_w, page_height=page_h)
        if rot_chars:
            rot_ids = {id(c) for c in rot_chars}
            kept = [([c for c in lf if id(c) not in rot_ids], rg)
                    for lf, rg in zip(leaves, regions)]
            kept = [(lf, rg) for lf, rg in kept if lf]
            leaves = [lf for lf, _ in kept]
            regions = [rg for _, rg in kept]
        if not leaves and not rot_chars:
            return [], []
        word_lines, leaf_ids = _lb.build_word_lines(
            leaves, x_tol, _lb.body_font_size(text_chars), regions=regions,
            spaces=[c for c in main_chars if c.get("text") == " "])
        if rot_chars:
            rlines, rids = _lb.rotated_word_lines(rot_chars, x_tol)
            word_lines = word_lines + rlines
            leaf_ids = leaf_ids + rids
    else:
        words = _pw_extract_words(chars, x_tolerance=x_tol,
                                 extra_attrs=["size"],
                                 use_text_flow=True)
        _annotate_words_with_fonts(words, chars)
        if not words:
            return [], []
        reading_lines, line_leaf = _xy.chars_to_leaf_lines(
            chars, page_width=page_w, page_height=page_h)
        if not reading_lines:
            return [], []
        word_lines, leaf_ids = _chars_to_word_lines(
            reading_lines, words, leaf_ids=line_leaf)
    if not word_lines:
        return [], []

    # Figure/table label text (plot ticks, table cells) is split off
    # into its own blocks so it is not glued into body sentences.
    fig_blocks = []
    want_geo = geo is not None
    masks = None
    if want_geo:
        geo["words"] = [w for ln in word_lines for w in ln]
    if _lb.FLAGS["fig"] or want_geo:
        masks = _lb.build_figure_masks(
            page.images,
            list(page.rects) + list(page.curves) + list(page.lines),
            page_w, page_h, _lb.body_font_size(text_chars))
        if want_geo:
            geo["masks"] = masks
    if _lb.FLAGS["fig"]:
        (body_lines, body_ids, lab_lines, lab_ids,
         body_order, lab_order) = _lb.split_figure_words(
            word_lines, leaf_ids, masks, _lb.body_font_size(text_chars))
        if body_lines and lab_lines:
            word_lines, leaf_ids, line_order = body_lines, body_ids, body_order
            fig_blocks = _lines_to_blocks(
                lab_lines, leaf_ids=lab_ids if _lb.FLAGS["leaf"] else None,
                orders=lab_order)
            for fb in fig_blocks:
                fb["fig"] = True
        else:
            line_order = None
    else:
        line_order = None

    all_blocks = _lines_to_blocks(
        word_lines,
        leaf_ids=leaf_ids if _lb.FLAGS["leaf"] else None,
        hyphen_join=_lb.FLAGS["hyphen"], orders=line_order)
    return all_blocks, fig_blocks


LAYOUT_ENGINE = "alto"


def _layout_engine() -> str:
    eng = (LAYOUT_ENGINE or "alto").strip().lower()
    return eng if eng in ("blocks", "alto") else "alto"


def _alto_blocks(page, chars, page_w, page_h, geo: Optional[dict] = None) -> list:
    """Blocks for one page from layout_streams, in the dict form _lines_to_blocks makes."""
    from . import layout_blocks as _lb
    from . import layout_streams as _ls

    info = {}
    ablocks = _ls.ordered_blocks(page, chars=chars, info=info)
    log(f"   stream layout glyphs={info.get('glyphs')} lines={info.get('lines')} "
        f"interleaved={info.get('interleaved_regions', 0)} blocks={len(ablocks)}")
    if not ablocks:
        return []
    if geo is not None:
        geo["words"] = [
            {"x0": w.x0, "x1": w.x1, "top": w.y0, "bottom": w.y1,
             "text": w.text, "size": w.size}
            for b in ablocks for ln in b.lines for w in (ln.words or ())]
    vocab = _lb.page_vocab([ln.text for b in ablocks for ln in b.lines])
    blocks = []
    for order, ab in enumerate(ablocks):
        texts = [ln.text for ln in ab.lines if ln.text]
        if not texts:
            continue
        text = _lb.join_lines(texts, vocab)
        blocks.append({
            "hpos": ab.x0,
            "vpos": ab.y0,
            "width": ab.x1 - ab.x0,
            "height": ab.y1 - ab.y0,
            "cx": ab.x0 + (ab.x1 - ab.x0) / 2,
            "text": text,
            "fontsize": ab.size,
            "bold": ab.bold,
            "region": 0,
            "order": order,
            "rotated": bool(ab.direction),
        })
    return blocks


def _alto_masks(page, chars, page_w, page_h, geo: dict) -> None:
    """Figure and table regions of the page for the caption module (alto engine)."""
    from . import layout_blocks as _lb

    _tc, _xt, _ = _char_stats(chars, tolerant=True)
    try:
        geo["masks"] = _lb.build_figure_masks(
            page.images,
            list(page.rects) + list(page.curves) + list(page.lines),
            page_w, page_h, _lb.body_font_size(_tc))
    except Exception as e:  # noqa: BLE001 - regions are optional for captions
        log(f"   figure regions unavailable ({type(e).__name__}: {e})")
        geo["masks"] = []


def _char_stats(chars: list, tolerant: bool = False) -> Tuple[list, float, float]:
    """Text chars, x tolerance and median font size of a page's chars.

    ``tolerant`` (alto engine) ignores chars whose size or box is not a
    finite number instead of raising.
    """
    def _ok(c) -> bool:
        if not c.get("text", "").strip():
            return False
        if not tolerant:
            return True
        try:
            return all(math.isfinite(float(c[k])) for k in ("size",))
        except (KeyError, TypeError, ValueError):
            return False

    text_chars = [c for c in chars if _ok(c)]
    median_fs = 0.0
    if text_chars:
        median_fs = float(np.median([c["size"] for c in text_chars]))
        x_tol = round(max(1.5, min(median_fs * 0.25, 3.0)), 2)
        space_chars = []
        for c in chars:
            if c.get("text") != " ":
                continue
            try:
                if c["x1"] - c["x0"] > 0:
                    space_chars.append(c)
            except (KeyError, TypeError):
                if not tolerant:
                    raise
        if space_chars:
            space_w = float(np.median([c["x1"] - c["x0"] for c in space_chars]))
            x_tol = round(min(x_tol, space_w * 0.7), 2)
    else:
        x_tol = 3
    return text_chars, x_tol, median_fs


def _page_blocks(page, chars, text_chars, x_tol, page_w, page_h,
                 geo: Optional[dict] = None) -> Tuple[list, list, bool]:
    """Blocks, figure-label blocks and whether the alto engine made them.

    Any failure of the alto engine (including RecursionError) is logged and
    that page falls back to the default blocks engine.

    ``geo``, when given, receives ``words`` (page words as dicts with x0, x1,
    top, bottom, text, size) and ``masks`` (figure and table regions) for the
    caption module, whichever engine made the blocks.
    """
    if _layout_engine() == "alto":
        try:
            if geo is not None:
                geo.clear()
            blocks = _alto_blocks(page, chars, page_w, page_h, geo)
            if geo is not None:
                _alto_masks(page, chars, page_w, page_h, geo)
            return blocks, [], True
        except Exception as e:  # noqa: BLE001 - includes RecursionError
            log(f"   alto layout failed ({type(e).__name__}: {e}); using the blocks engine")
            from . import layout_blocks as _lb
            if _lb.FLAGS["dedupe"]:
                chars = _lb.dedupe_overprinted(chars)
                for _i, _c in enumerate(chars):
                    _c["_si"] = _i
            text_chars, x_tol, _ = _char_stats(chars, tolerant=True)
            if not text_chars:
                return [], [], False
    all_blocks, fig_blocks = _xy_cut_blocks(page, chars, text_chars, x_tol, page_w, page_h, geo)
    return all_blocks, fig_blocks, False


_CAPTION_RE = re.compile(
    r"^[^A-Za-z]*((?:fig(?:ure)?|tab(?:le)?)\.?\s*\d+)\s*[.:]\s*(\S.*)$",
    re.IGNORECASE | re.DOTALL,
)


_CAPTION_RE_NORMALIZED = re.compile(
    r"^[^A-Za-z]*(fig(?:ure)?|tab(?:le)?)\.?\s*(\d+)\s*[.:]\s*(\S.*)$",
    re.IGNORECASE | re.DOTALL,
)


def _split_caption_label(text: str, keep_label: bool = False) -> Optional[Tuple[str, str]]:
    """Split "Fig. 1. Spectrum of ..." into a label and the caption text.

    By default the label is normalized to "Figure 1" or "Table 1".  With
    ``keep_label`` (the alto engine) it stays as printed ("Fig. 1" stays
    "Fig. 1"), so the abbreviation token that the source has is not lost.
    Returns None when the text does not open with a figure or table label.
    """
    if keep_label:
        m = _CAPTION_RE.match(text)
        if not m:
            return None
        return re.sub(r"\s+", " ", m.group(1)).strip(), m.group(2).strip()
    m = _CAPTION_RE_NORMALIZED.match(text)
    if not m:
        return None
    kind = "Figure" if m.group(1)[:3].lower() == "fig" else "Table"
    return f"{kind} {m.group(2)}", m.group(3).strip()


_META_RE = re.compile(r'@\S+\.\S+|orcid\.org|ORCID:\s*\d|^Authors?\b')


def _meta_rows(blocks: list, tol: float) -> list:
    """Order contact blocks by row, then left to right inside a row.

    Block tops are floats that differ by fractions of a point even when two
    blocks share a printed row, so sorting on the raw top scrambles a row of
    side by side blocks.  Blocks whose tops lie within ``tol`` of the first
    block of a row belong to that row.
    """
    out: list = []
    row: list = []
    row_top = 0.0
    for b in sorted(blocks, key=lambda b: (b["vpos"], b["hpos"])):
        if row and b["vpos"] - row_top > tol:
            out.extend(sorted(row, key=lambda r: r["hpos"]))
            row = []
        if not row:
            row_top = b["vpos"]
        row.append(b)
    out.extend(sorted(row, key=lambda r: r["hpos"]))
    return out


def _partition_blocks_default(blocks: list, page_h: float, page_median_fs: float,
                              col_start: float, col_end: float, span_threshold: float) -> list:
    """Order a page's blocks as headline, contact info, body, footer (default engine)."""
    _meta_re = re.compile(
        r'@\S+\.\S+|orcid\.org|ORCID:\s*\d|^Authors?\b',
    )
    top_zone = page_h * 0.12
    header_blocks = []
    meta_blocks = []
    body_blocks = []
    footer_blocks = []
    for blk in blocks:
        is_wide = blk["width"] > span_threshold
        is_title_font = page_median_fs > 0 and blk["fontsize"] >= page_median_fs * 1.4
        if is_title_font and blk["vpos"] + blk["height"] <= col_start + page_h * 0.02:
            header_blocks.append(blk)
        elif is_wide and blk["vpos"] > col_end:
            footer_blocks.append(blk)
        elif (not is_wide
              and blk["vpos"] + blk["height"] <= top_zone
              and _meta_re.search(blk["text"])):
            meta_blocks.append(blk)
        else:
            body_blocks.append(blk)

    header_blocks.sort(key=lambda b: b["vpos"])
    meta_blocks.sort(key=lambda b: (b["vpos"], b["hpos"]))
    footer_blocks.sort(key=lambda b: b["vpos"])
    return header_blocks + meta_blocks + body_blocks + footer_blocks


def _partition_blocks(blocks: list, page_h: float, page_median_fs: float,
                      col_start: float, col_end: float, span_threshold: float,
                      alto: bool = False) -> list:
    """Order a page's blocks as headline, contact info, body, footer.

    The default engine keeps the original partition.  For the alto engine a
    contact block (e-mail, ORCID, "Authors") is lifted ahead of the body only
    when it stands apart.  When the block just before it in reading order is an
    ordinary body block of the top zone (an author name, an affiliation), the
    contact block belongs to that block, so it stays where the reading order
    put it and name and e-mail keep their pairing.
    """
    if not alto:
        return _partition_blocks_default(
            blocks, page_h, page_median_fs, col_start, col_end, span_threshold)
    top_zone = page_h * 0.12
    header_blocks, meta_blocks, body_blocks, footer_blocks = [], [], [], []
    prev_kind = None
    # Only the leading run of title-font blocks is the headline.  A title-font
    # block that follows other text in reading order (a section heading under
    # the author line, a badge further down) keeps its place.
    headline_open = True
    for blk in blocks:
        is_wide = blk["width"] > span_threshold
        is_title_font = page_median_fs > 0 and blk["fontsize"] >= page_median_fs * 1.4
        in_top = blk["vpos"] + blk["height"] <= top_zone
        if (is_title_font and headline_open
                and blk["vpos"] + blk["height"] <= col_start + page_h * 0.02):
            header_blocks.append(blk)
            kind = "header"
        elif is_wide and blk["vpos"] > col_end:
            footer_blocks.append(blk)
            kind = "footer"
        elif not is_wide and in_top and _META_RE.search(blk["text"]):
            if prev_kind == "top_body":
                body_blocks.append(blk)
                kind = "top_body"
            else:
                meta_blocks.append(blk)
                kind = "meta"
        else:
            body_blocks.append(blk)
            kind = "top_body" if in_top else "body"
        if kind != "header":
            headline_open = False
        prev_kind = kind
    header_blocks.sort(key=lambda b: b["vpos"])
    footer_blocks.sort(key=lambda b: b["vpos"])
    meta_blocks = _meta_rows(meta_blocks, max(2.0, 0.5 * page_median_fs))
    return header_blocks + meta_blocks + body_blocks + footer_blocks


def extract_text_with_pdfplumber(pdf_path: str,
                                 captions_out: Optional[list] = None,
                                 max_pages: Optional[int] = None) -> Optional[str]:
    """Extract text from PDF using pdfplumber (layout-aware).

    Uses recursive XY-cut tree (ported from xpdf's splitChars algorithm)
    to determine reading order, then groups words into blocks and
    prefixes headers with '## ' based on font size/style analysis.

    Pipeline:
      1. Extract chars and words with (x0, y0, x1, y1, fontname, size)
      2. Recursive XY-cut on chars to determine reading order
      3. Map char-level lines back to word-level lines
      4. Group word lines into blocks by vertical gaps
      5. Classify blocks as header/meta/body/footer
      6. Detect headers via font size/bold heuristics

    When ``captions_out`` is a list and the caption engine is on, figure and
    table captions found from page geometry (see poster2json.captions) are
    appended to it as ``{"kind", "label", "text", "source", "page"}`` dicts in
    reading order, and each caption is replaced in the returned text by a
    one-line marker (see CAPTION_RAW_MODE).
    """
    import pdfplumber

    log(f"Attempting text extraction with pdfplumber for: {pdf_path}")
    t0 = time.time()

    try:
        pdf = pdfplumber.open(pdf_path)
    except Exception as e:
        log(f"pdfplumber failed to open {pdf_path}: {e}")
        return None

    all_output_lines = []
    use_captions = captions_out is not None and CAPTION_ENGINE
    cap_counts = {"figure": 0, "table": 0}

    try:
        for page_no, page in enumerate(pdf.pages[:max_pages] if max_pages else pdf.pages):
            page_w = page.width
            page_h = page.height

            from . import layout_blocks as _lb
            tolerant = _layout_engine() == "alto"
            try:
                chars = _filter_phantom_spaces(page.chars)
            except Exception:
                if not tolerant:
                    raise
                chars = list(page.chars)
            if _lb.FLAGS["dedupe"] and not tolerant:
                chars = _lb.dedupe_overprinted(chars)
            for _i, _c in enumerate(chars):
                _c["_si"] = _i  # PDF stream order, used to order chars in a line

            text_chars, x_tol, median_fs = _char_stats(chars, tolerant=tolerant)
            log(f"   x_tolerance={x_tol} (median_font={median_fs if text_chars else '?'}pt)")
            if not text_chars:
                continue

            geo = {} if use_captions else None
            all_blocks, fig_blocks, used_alto = _page_blocks(
                page, chars, text_chars, x_tol, page_w, page_h, geo)
            for blk in all_blocks:
                blk["col"] = 0
                blk["flow_idx"] = 0

            if not all_blocks:
                continue

            all_fontsizes = [blk["fontsize"] for blk in all_blocks if blk["fontsize"] > 0]
            page_median_fs = sorted(all_fontsizes)[len(all_fontsizes) // 2] if all_fontsizes else 0

            text_left = min(blk["hpos"] for blk in all_blocks)
            text_right = max(blk["hpos"] + blk["width"] for blk in all_blocks)
            text_span = text_right - text_left
            span_threshold = text_span * 0.5

            min_col_block_len = 10
            col_tops = {}
            for blk in all_blocks:
                c = blk["col"]
                if blk["width"] <= span_threshold and len(blk["text"].strip()) >= min_col_block_len:
                    col_tops[c] = min(col_tops.get(c, float("inf")), blk["vpos"])
            if not col_tops:
                for blk in all_blocks:
                    c = blk["col"]
                    if blk["width"] <= span_threshold:
                        col_tops[c] = min(col_tops.get(c, float("inf")), blk["vpos"])
            col_start = min(col_tops.values()) if col_tops else 0
            col_end = max(blk["vpos"] + blk["height"] for blk in all_blocks
                         if blk["width"] <= span_threshold) if col_tops else page_h

            all_blocks = _partition_blocks(
                all_blocks, page_h, page_median_fs, col_start, col_end, span_threshold,
                alto=used_alto)

            # Emit figure labels as separate blocks, right after the body block
            # that preceded them in the original line order (so labels are not
            # glued into a paragraph, and nothing moves far from where it was
            # printed).
            for fb in fig_blocks:
                at = 0
                best = -1
                for k, bk in enumerate(all_blocks):
                    if not bk.get("fig") and best < bk["order"] < fb["order"]:
                        best, at = bk["order"], k + 1
                while (at < len(all_blocks) and all_blocks[at].get("fig")
                       and all_blocks[at]["order"] < fb["order"]):
                    at += 1
                all_blocks.insert(at, fb)

            median_fontsize = page_median_fs

            if use_captions:
                raise RuntimeError("caption detection is not included in PosterSentry")
                _body = _lb.body_font_size(text_chars)
                _found = _cap_mod.find_captions(
                    all_blocks,
                    _cap_mod.regions_from_masks(geo.get("masks") or [], _body),
                    geo.get("words") or [], _body, page_h)
                for _c in _found:
                    cap_counts[_c["kind"]] += 1
                    _k = cap_counts[_c["kind"]]
                    _name = "Figure" if _c["kind"] == "figure" else "Table"
                    captions_out.append({
                        "kind": _c["kind"], "label": _c["label"],
                        "text": _c["text"], "source": _c["source"],
                        "page": page_no,
                    })
                    first = all_blocks[min(_c["blocks"])]
                    first["cap_marker"] = (
                        first.get("cap_marker", "") + ("\n" if first.get("cap_marker") else "")
                        + f"[{_name} caption {_k} extracted]")
                    first["cap_text"] = (
                        first.get("cap_text", "") + (" " if first.get("cap_text") else "")
                        + _c["text"])
                    for _bi in _c["blocks"]:
                        all_blocks[_bi]["cap"] = True

            for blk in all_blocks:
                _t = blk["text"].strip()
                if not _t:
                    continue
                if blk.get("cap"):
                    if blk.get("cap_marker"):
                        all_output_lines.append(blk["cap_marker"])
                        if CAPTION_RAW_MODE == "keep":
                            all_output_lines.append(_add_bidi_markers(blk["cap_text"]))
                    continue
                if blk.get("fig"):
                    all_output_lines.append(_add_bidi_markers(_t))
                    continue

                # Figure/table caption: split the "Figure N"/"Table N" label
                # onto its own header line so the LLM isolates the caption as a
                # caption section instead of merging it into adjacent body text.
                # Captions are small-font, so the font-size header heuristic
                # below never catches them; a buried short caption tanks
                # section-level ROUGE (matches at ~0.2 instead of ~1.0).
                _cap = _split_caption_label(_t, keep_label=used_alto)
                if _cap:
                    all_output_lines.append(f"## {_cap[0]}")
                    all_output_lines.append(_add_bidi_markers(_cap[1]))
                    continue

                # Standalone section heading printed as a bare keyword line
                # (e.g. "References", "Acknowledgements"). Force a ## header so
                # the LLM splits it out instead of merging it into the body.
                if _t.rstrip(":.").strip().lower() in _SECTION_KEYWORDS:
                    all_output_lines.append(f"## {_add_bidi_markers(_t)}")
                    continue

                # Run-on block cramming multiple sections onto one line via
                # inline ALL-CAPS labels ("...REFERENCES: ... ABBREVIATIONS: ...")
                # or em-dash sub-headers ("265. TWINS — Our second sample ...").
                # Split into labeled sections so the LLM doesn't merge them.
                _segs = _split_inline_sections(_t)
                if _segs:
                    for _head, _body in _segs:
                        if _head:
                            all_output_lines.append(f"## {_head}")
                        if _body:
                            all_output_lines.append(_add_bidi_markers(_body))
                    continue

                _words = _t.split()
                _single_char = sum(1 for w in _words if len(w) <= 1)
                if len(_words) >= 2 and _single_char > len(_words) * 0.6:
                    continue
                if len(_t) <= 3 and not re.match(r'^[•●▪]+$', _t):
                    continue
                _alpha_words = re.findall(r'[A-Za-z]{2,}', _t)
                _starts_with_letter = bool(re.match(r'[A-Za-z]', _t))
                _is_contact = bool(
                    re.search(r'@\S+\.\S+', _t)
                    or re.search(r'\+\d[\d\s]{6,}', _t)
                    or re.search(r'www\.', _t, re.I)
                    or re.search(r'https?://', _t, re.I)
                )

                is_header = False
                # Rotated text (axis labels, side captions) is not promoted by
                # weight or size; keyword headings were handled above.
                _rotated = bool(blk.get("rotated"))
                _is_continuation = bool(re.match(
                    r'(?:to|and|or|the|a|an|in|of|for|with|by|from|that|which|when)\s', _t
                ))
                min_words = 1 if blk["bold"] else 2
                if (median_fontsize > 0 and len(_t) <= 120 and not _rotated
                        and _starts_with_letter and len(_alpha_words) >= min_words
                        and not _is_contact and not _is_continuation):
                    fs = blk["fontsize"]
                    if blk["bold"] and fs >= median_fontsize:
                        is_header = True
                    elif fs > median_fontsize * 1.3:
                        if len(_alpha_words) == 1:
                            is_header = _alpha_words[0].isupper()
                        else:
                            is_header = True

                text = _add_bidi_markers(blk["text"])
                if is_header:
                    text = f"## {text}"
                all_output_lines.append(text)

    finally:
        pdf.close()

    elapsed = time.time() - t0

    if not all_output_lines:
        log(f"pdfplumber produced no text for: {pdf_path}")
        return None

    result = "\n".join(all_output_lines)
    log(f"pdfplumber extracted {len(result)} characters in {elapsed:.2f} seconds")
    return result


def _add_bidi_markers(text: str) -> str:
    """Wrap RTL character runs in bidi embedding markers.

    pdfplumber does not emit bidi markers around Hebrew, Arabic, and other
    RTL words. Without them the LLM receives bare RTL codepoints in an
    unstructured layout and fails to extract sections.
    """
    if not text:
        return text
    has_rtl = False
    for ch in text:
        if unicodedata.bidirectional(ch) in ("R", "AL"):
            has_rtl = True
            break
    if not has_rtl:
        return text
    words = text.split(" ")
    result = []
    for word in words:
        if any(unicodedata.bidirectional(c) in ("R", "AL") for c in word):
            result.append(f"‫{word}‬")
        else:
            result.append(word)
    return " ".join(result)

