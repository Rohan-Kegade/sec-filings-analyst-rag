"""Convert raw 10-K HTML into clean text split by Item sections."""

import re
import warnings

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

from sec_rag.models import Filing, FilingMetadata, Section

# Modern 10-Ks are inline XBRL: XHTML with an <?xml?> declaration. We only need the
# visible text, which the lenient HTML parser extracts more reliably than an XML parser.
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

ITEM_TITLES = {
    "1": "Business",
    "1A": "Risk Factors",
    "1B": "Unresolved Staff Comments",
    "1C": "Cybersecurity",
    "2": "Properties",
    "3": "Legal Proceedings",
    "4": "Mine Safety Disclosures",
    "5": "Market for Registrant's Common Equity",
    "6": "Reserved",
    "7": "Management's Discussion and Analysis",
    "7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "8": "Financial Statements and Supplementary Data",
    "9": "Changes in and Disagreements with Accountants",
    "9A": "Controls and Procedures",
    "9B": "Other Information",
    "9C": "Disclosure Regarding Foreign Jurisdictions that Prevent Inspections",
    "10": "Directors, Executive Officers and Corporate Governance",
    "11": "Executive Compensation",
    "12": "Security Ownership of Certain Beneficial Owners and Management",
    "13": "Certain Relationships and Related Transactions",
    "14": "Principal Accountant Fees and Services",
    "15": "Exhibits and Financial Statement Schedules",
    "16": "Form 10-K Summary",
}
ITEM_ORDER = {item: i for i, item in enumerate(ITEM_TITLES)}

# Headings closer together than this are TOC entries or page headers, not section starts.
MIN_HEADING_GAP = 200

# Matches "Item 1A." / "ITEM 7:" / "Item 7A -" at the start of a line. Two-digit items
# come first and \b is required, so "Item 10" is not read as "Item 1".
ITEM_PATTERN = re.compile(
    r"^\s*item\s+(1[0-6]|1[abc]?|[2-6]|7a?|8|9[abc]?)\b\s*[\.\:\-–—]?",
    re.IGNORECASE | re.MULTILINE,
)


# Elements that start a new line. Everything else (span, b, ix:nonfraction, ...) is inline:
# XBRL wraps every number in its own tag, and breaking lines there split sentences like
# "a lease term of \n10.3\n years".
BLOCK_TAGS = ["p", "div", "br", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6",
              "hr", "section", "article", "header", "footer", "blockquote", "pre", "dt", "dd"]  # fmt: skip
CELL_SEPARATOR = " | "


def _cell_text(cell) -> str:
    return re.sub(r"\s+", " ", cell.get_text(" ")).strip()


def _join_cells(cells: list[str]) -> str:
    """Join a row's non-empty cells, gluing back the pieces filings put in cells of their own:
    "$" | "11,534" -> "$11,534", "(" | "1,984" | ")" -> "(1,984)", "3.1" | "%" -> "3.1%"."""
    joined: list[str] = []
    glue_next = False
    for cell in cells:
        if not cell:
            continue
        if glue_next and joined:
            joined[-1] += cell
            glue_next = False
        elif cell in (")", "%", ")%", "%)") and joined:
            joined[-1] += cell
        else:
            joined.append(cell)
        if cell in ("$", "(", "($", "$("):
            glue_next = True
    text = CELL_SEPARATOR.join(joined)
    return re.sub(r"\(\s+", "(", re.sub(r"\s+\)", ")", text))


def _table_text(table) -> str:
    """A table as one line per row, cells separated by " | ", rows that are empty dropped.

    Flattened to plain text, every cell landed on its own line, far from its column header,
    so "$ / 11,534 / $ / 896 / $ / 12,430" lost which number was the total.
    """
    rows = []
    for tr in table.find_all("tr"):
        if tr.find_parent("table") is not table:
            continue  # a nested table's rows are handled with that table
        row = _join_cells([_cell_text(cell) for cell in tr.find_all(["td", "th"], recursive=False)])
        if row:
            rows.append(row)
    return "\n".join(rows)


def html_to_text(html: str | bytes) -> str:
    soup = BeautifulSoup(html, "lxml")
    # Inline XBRL filings carry a hidden header full of machine-readable facts.
    for tag in soup.find_all(["script", "style", "ix:header"]):
        tag.decompose()
    # Innermost tables first, so an outer table sees its nested ones already as text.
    for table in reversed(soup.find_all("table")):
        table.replace_with(f"\n\n{_table_text(table)}\n\n")
    for tag in soup.find_all(BLOCK_TAGS):
        tag.insert_before("\n")
        tag.insert_after("\n")
    text = soup.get_text()
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def _find_section_starts(text: str) -> list[tuple[int, str]]:
    """Return (position, item) for the real start of each Item section.

    Item headings also appear in the table of contents and, in some filings, as a
    running header on every page. Headings followed almost immediately by another
    heading (TOC entries, a page header right above the real title) are dropped.
    Of the rest, we take the longest run in 10-K order and use the first occurrence
    of each item in it, which skips repeated page headers and stray cross-references.
    """
    matches = [(m.start(), m.group(1).upper()) for m in ITEM_PATTERN.finditer(text)]
    candidates = [
        (pos, item)
        for idx, (pos, item) in enumerate(matches)
        if idx + 1 == len(matches) or matches[idx + 1][0] - pos >= MIN_HEADING_GAP
    ]
    if not candidates:
        return []

    # Longest increasing subsequence by item order (n is a few hundred at most).
    length = [1] * len(candidates)
    prev = [-1] * len(candidates)
    for i, (_, item_i) in enumerate(candidates):
        for j in range(i):
            if ITEM_ORDER[candidates[j][1]] < ITEM_ORDER[item_i] and length[j] + 1 > length[i]:
                length[i], prev[i] = length[j] + 1, j

    chain = []
    idx = max(range(len(candidates)), key=length.__getitem__)
    while idx != -1:
        chain.append(candidates[idx][1])
        idx = prev[idx]
    chain.reverse()

    starts = []
    for pos, item in candidates:
        if len(starts) < len(chain) and item == chain[len(starts)]:
            starts.append((pos, item))
    return starts


def split_into_items(text: str) -> list[Section]:
    """Split filing text into Item sections."""
    starts = _find_section_starts(text)
    if not starts:
        return [Section(item="FULL", title="Full Filing", text=text)]

    sections = []
    for idx, (pos, item) in enumerate(starts):
        end = starts[idx + 1][0] if idx + 1 < len(starts) else len(text)
        body = text[pos:end].strip()
        if len(body) > 200:  # drop stray headings and cross-references
            sections.append(Section(item=item, title=ITEM_TITLES.get(item, ""), text=body))
    return sections


def parse_filing(html: str | bytes, metadata: FilingMetadata) -> Filing:
    return Filing(metadata=metadata, sections=split_into_items(html_to_text(html)))
