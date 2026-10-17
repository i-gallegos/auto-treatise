"""Clean fetched CourtListener opinions into the analysis database.

Reads the raw ``citing_cases.db`` written by ``fetch_courtlistener`` and writes
``citing_cases_clean.db``: the case metadata plus ``opinion_text_clean`` (page
headers removed, hard-wrapped lines rejoined) and ``opinion_paragraphs_clean_json``
(the text split into paragraphs of at least 200 characters). Rows whose cleaned
text is shorter than ``--min-chars`` are dropped.

    python -m courtlistener.clean_courtlistener --src-db citing_cases.db
"""

import argparse
import json
from pathlib import Path
import re
import sqlite3

from bs4 import BeautifulSoup

MIN_CHARS = 500
MIN_PARAGRAPH_LENGTH = 200

# A paragraph ending in one of these followed by a period is joined to the next
# (for example "Sandy, JJ." or "see U.S.").
ABBREVIATIONS = {
    line
    for line in (
        Path(__file__).with_name("legal_abbreviations.txt")
        .read_text()
        .splitlines()
    )
    if line and not line.startswith("#")
}

# Page headers and footers of federal circuit opinions.
PAGE_HEADER_RE = re.compile(
    r"^\s*(?:"
    # "USCA11 Case: 17-15611  Date Filed: 11/25/2020  Page: 7 of 16"
    r"USCA\d+\s+Case:.*?Page:\s*\d+\s+of\s+\d+"
    r"|"
    # "No. 19-5759   United States v. Richardson   Page 2"
    r"No\.\s+[\d\-]+\s+.{5,60}\s+Page\s+\d+"
    r"|"
    # "Case: 20-10243  Date Filed: ..."
    r"Case:\s+[\d\-]+\s+Date\s+Filed:.*"
    r"|"
    # "Case 1:18-cv-01234-ABC  Document 45  Filed 01/01/20  Page 1 of 10"
    r"Case\s+\d+:\d+.+?Page\s+\d+\s+of\s+\d+"
    r")\s*$",
    re.MULTILINE,
)
FORMFEED_RE = re.compile(r"\x0c")
# Lines of dashes, underscores or equals signs.
RULE_LINE_RE = re.compile(r"^\s*[_=\-─━]{10,}\s*$")
# "  1 The shooting occurred..." or "  * See also..."
FOOTNOTE_RE = re.compile(r"^\s{2,}(\d+|[*†‡§])\s+\S")
SECTION_HEADING_RE = re.compile(
    r"^\s*(?:"
    r"(?:I{1,3}|IV|VI{0,3}|IX|X{0,3})(?:\.|$)"  # Roman numerals
    r"|[A-Z]\."  # letter headings
    r"|\d+\."  # numbered sections
    r"|(?:BACKGROUND|DISCUSSION|ANALYSIS|CONCLUSION|INTRODUCTION|FACTS|STANDARD"
    r"|PROCEDURAL\s+HISTORY|FACTUAL\s+BACKGROUND|LEGAL\s+STANDARD"
    r"|ARGUMENT|OPINION|ORDER|DISSENT|CONCURRENCE)"
    r")\s*$",
    re.IGNORECASE,
)
ALL_CAPS_RE = re.compile(r"^[A-Z][A-Z\s\.\,\-\:]{4,}$")


# ── HTML to text ──────────────────────────────────────────────────────────────


def classify_html(html):
    """'opinion_p' (<opinion> + <p>), 'div_p' (<div> + <p>) or 'pre'."""
    if "<opinion" in html:
        return "opinion_p"
    if "<div" in html.lower() and "<p" in html.lower() and "<pre" not in html:
        return "div_p"
    return "pre"


def clean_p_based(html):
    """Paragraphs of <p>-based HTML, separated by blank lines."""
    soup = BeautifulSoup(html, "html.parser")
    # Page-label anchors insert page numbers inline.
    for a in soup.find_all("a", class_="page-label"):
        a.decompose()
    paragraphs = []
    author = soup.find("author")
    if author:
        paragraphs.append(author.get_text().strip())
    for p in soup.find_all("p"):
        text = p.get_text(separator=" ").strip()
        if text:
            paragraphs.append(re.sub(r"  +", " ", text))
    return "\n\n".join(paragraphs)


def clean_pre_based(html):
    """Reflowed text of <pre>-based HTML (``<pre>`` blocks interleaved with
    citation ``<span>`` elements)."""
    soup = BeautifulSoup(html, "html.parser")
    parts = []
    for element in soup.children:
        if element.name == "p":
            parts.append("\n\n" + element.get_text() + "\n\n")
        elif hasattr(element, "get_text"):
            parts.append(element.get_text())
        elif isinstance(element, str):
            parts.append(element)
    return reflow_pre_text("".join(parts))


# ── Reflowing hard-wrapped text ───────────────────────────────────────────────


def _is_double_spaced(lines):
    """True when more than half the content lines are followed by a blank."""
    content = [i for i, line in enumerate(lines) if line.strip()]
    if len(content) < 10:
        return False
    followed_by_blank = sum(
        1 for i in content if i + 1 < len(lines) and not lines[i + 1].strip()
    )
    return followed_by_blank / len(content) > 0.50


def _strip_double_spacing(lines):
    """Drop the blank line between every line of a double-spaced text. Runs of
    two or more blanks, and a blank before an indented line, are kept as one."""
    result = []
    i = 0
    while i < len(lines):
        if lines[i].strip():
            result.append(lines[i])
            i += 1
            continue
        j = i
        while j < len(lines) and not lines[j].strip():
            j += 1
        if j - i >= 2:
            result.append("")
        elif j < len(lines):
            next_line = lines[j]
            if len(next_line) - len(next_line.lstrip()) >= 5:
                result.append("")
        i = j
    return result


def _join_para(lines):
    """Join lines into one paragraph; a hyphen ending a line is dropped."""
    parts = list(lines)
    for i in range(len(parts) - 1):
        if parts[i].endswith("-") and not parts[i].endswith("--"):
            parts[i] = parts[i][:-1]
    return " ".join(parts)


def reflow_pre_text(text):
    """Rejoin hard-wrapped lines into paragraphs. A new paragraph starts at a
    blank line, a heading, a footnote, or an indentation change after a line
    that ends a sentence."""
    text = FORMFEED_RE.sub("", text)
    text = PAGE_HEADER_RE.sub("", text)
    lines = text.split("\n")
    if _is_double_spaced(lines):
        lines = _strip_double_spacing(lines)

    paragraphs = []
    current = []
    prev_indent = 0
    for line in lines:
        stripped = line.rstrip()
        content = stripped.lstrip()
        indent = len(stripped) - len(content)

        if RULE_LINE_RE.match(stripped) or not content:
            if current:
                paragraphs.append(_join_para(current))
                current = []
            continue

        is_heading = (
            SECTION_HEADING_RE.match(stripped)
            or (ALL_CAPS_RE.match(content) and len(content) < 80)
            or (len(content) < 60 and content.endswith(":") and indent > 2)
        )
        if is_heading:
            if current:
                paragraphs.append(_join_para(current))
                current = []
            paragraphs.append(content)
            continue

        if FOOTNOTE_RE.match(stripped) and indent >= 2:
            if current:
                paragraphs.append(_join_para(current))
            current = [content]
            continue

        if not current:
            current = [content]
            prev_indent = indent
            continue

        prev_ends_sentence = bool(re.search(r"[.!?:;\"'\)]\s*$", current[-1]))
        new_paragraph = prev_ends_sentence and (
            (indent >= 5 and prev_indent < 3)
            or abs(indent - prev_indent) >= 4
        )
        if new_paragraph:
            paragraphs.append(_join_para(current))
            current = [content]
        else:
            current.append(content)
        prev_indent = indent

    if current:
        paragraphs.append(_join_para(current))
    result = "\n\n".join(p for p in paragraphs if p.strip())
    return re.sub(r"  +", " ", result)


def clean_opinion(html, text):
    """Cleaned text of an opinion: from the HTML when there is any, else from
    the plain text."""
    if html:
        if classify_html(html) == "pre":
            return clean_pre_based(html)
        return clean_p_based(html)
    return reflow_pre_text(text)


# ── Paragraphs ────────────────────────────────────────────────────────────────


def _squash(text):
    return re.sub(r"\s+", " ", text).strip()


def _join_wrapped_lines(text):
    """One line per paragraph: join hard-wrapped lines, keeping tab-indented
    lines and lines after a blank as paragraph starts. Standalone page numbers
    become blanks."""
    text = FORMFEED_RE.sub(
        "", text.replace("\r\n", "\n").replace("\r", "\n").replace("\n\n", "\n")
    )
    lines = [
        "" if re.match(r"^\s*\d+\s*$", line) else line.rstrip()
        for line in text.split("\n")
    ]
    out = []
    i = 0
    while i < len(lines):
        if lines[i].strip() == "":
            out.append("")
            i += 1
            continue
        buffer = (
            f"\t{lines[i].strip()}" if lines[i].startswith("\t") else lines[i].strip()
        )
        j = i
        while j + 1 < len(lines):
            nxt = lines[j + 1].strip()
            if nxt == "" or lines[j + 1].startswith("\t"):
                break
            ends_sentence = bool(re.search(r"[.!?:;\"')\]]\s*$", buffer))
            starts_continuation = bool(re.match(r"^[a-z\)\],.;:]", nxt))
            if buffer.endswith("-") and not buffer.endswith("--"):
                buffer = buffer[:-1] + nxt
            elif not ends_sentence and (starts_continuation or len(buffer) < 90):
                buffer = f"{buffer} {nxt}"
            else:
                break
            j += 1
        out.append(buffer)
        i = j + 1
    return "\n".join(chunk for chunk in out if chunk != "")


def _is_heading_like(paragraph):
    return bool(
        SECTION_HEADING_RE.match(paragraph) or ALL_CAPS_RE.match(paragraph)
    )


def _ends_with_abbreviation(paragraph):
    match = re.search(r"([^\s]+?\.)[\"')\]]*\s*$", paragraph)
    if not match:
        return False
    token = match.group(1)
    for char in ')("”-’':
        token = token.replace(char, "")
    return token in ABBREVIATIONS


def chunk_into_paragraphs(text):
    """Paragraphs of ``text``: wrapped lines joined, a paragraph joined to the
    next when it does not end in a period or ends in an abbreviation (unless
    either is a heading), and short paragraphs merged until each has at least
    ``MIN_PARAGRAPH_LENGTH`` characters."""
    lines = (_squash(line) for line in _join_wrapped_lines(text).split("\n"))
    paragraphs = [line for line in lines if line]
    if not paragraphs:
        return []

    repaired = [paragraphs[0]]
    for paragraph in paragraphs[1:]:
        previous = repaired[-1]
        ends_sentence = previous.endswith(".") and not _ends_with_abbreviation(
            previous
        )
        if (
            _is_heading_like(previous)
            or _is_heading_like(paragraph)
            or ends_sentence
        ):
            repaired.append(paragraph)
        else:
            repaired[-1] = f"{previous} {paragraph}"

    merged = []
    buffer = []
    length = 0
    for paragraph in repaired:
        buffer.append(paragraph)
        length += len(paragraph)
        if length >= MIN_PARAGRAPH_LENGTH:
            merged.append(" ".join(buffer))
            buffer = []
            length = 0
    if buffer:
        merged.append(" ".join(buffer))
    return merged


# ── Database ──────────────────────────────────────────────────────────────────

COPIED_COLUMNS = [
    ("id", "INTEGER PRIMARY KEY"),
    ("courtlistener_opinion_id", "TEXT"),
    ("courtlistener_cluster_id", "TEXT"),
    ("case_name", "TEXT"),
    ("decision_date", "TEXT"),
    ("citation", "TEXT"),
    ("court_id", "TEXT"),
    ("court_name", "TEXT"),
    ("court_abbreviation", "TEXT"),
    ("court_level", "TEXT"),
    ("jurisdiction", "TEXT"),
    ("circuit", "TEXT"),
    ("courtlistener_docket_id", "INTEGER"),
    ("docket_number", "TEXT"),
    ("precedential_status", "TEXT"),
    ("citation_count", "INTEGER"),
]
CLEAN_COLUMNS = [
    ("opinion_text_clean", "TEXT"),
    ("opinion_paragraphs_clean_json", "TEXT"),
]


def clean_to_output_db(src_db, out_db, force=False, min_chars=MIN_CHARS):
    """Clean every fetched opinion of ``src_db`` into a new ``out_db``."""
    out_path = Path(out_db)
    if out_path.exists():
        if not force:
            raise FileExistsError(f"{out_db} exists; use --force to overwrite")
        out_path.unlink()

    src = sqlite3.connect(src_db)
    src.row_factory = sqlite3.Row
    rows = src.execute(
        "SELECT * FROM cases WHERE fetch_status = 'fetched' ORDER BY id"
    ).fetchall()
    src.close()

    columns = COPIED_COLUMNS + CLEAN_COLUMNS
    out = sqlite3.connect(out_db)
    out.execute(
        "CREATE TABLE cases ("
        + ", ".join(f"{name} {kind}" for name, kind in columns)
        + ")"
    )
    insert = f"""INSERT INTO cases ({', '.join(name for name, _ in columns)}) VALUES ({', '.join('?' for _ in columns)})"""
    kept = 0
    for row in rows:
        clean = clean_opinion(row["opinion_html"], row["opinion_text"])
        if len(clean) < min_chars:
            continue
        values = [row[name] for name, _ in COPIED_COLUMNS]
        paragraphs = chunk_into_paragraphs(clean)
        values += [clean, json.dumps(paragraphs, ensure_ascii=False)]
        out.execute(insert, values)
        kept += 1
    out.commit()
    out.close()
    print(f"Kept {kept} of {len(rows)} opinions in {out_db}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src-db", type=Path, required=True)
    parser.add_argument(
        "--out-db",
        type=Path,
        help="default: <src-db stem>_clean.db beside the source",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--min-chars", type=int, default=MIN_CHARS)
    args = parser.parse_args()
    out_db = args.out_db or args.src_db.with_name(f"{args.src_db.stem}_clean.db")
    clean_to_output_db(args.src_db, out_db, args.force, args.min_chars)


if __name__ == "__main__":
    main()
