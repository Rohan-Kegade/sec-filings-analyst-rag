"""Convert raw 10-K HTML into clean text split by Item sections."""

import re

from bs4 import BeautifulSoup

from sec_rag.models import Filing, FilingMetadata, Section

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

# Matches "Item 1A." / "ITEM 7:" / "Item 7A -" at the start of a line.
ITEM_PATTERN = re.compile(
    r"^\s*item\s+(1[abc]?|[2-6]|7a?|8|9[abc]?|1[0-6])\s*[\.\:\-–—]?",
    re.IGNORECASE | re.MULTILINE,
)


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    # Inline XBRL filings carry a hidden header full of machine-readable facts.
    for tag in soup.find_all(["script", "style", "ix:header"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def split_into_items(text: str) -> list[Section]:
    """Split filing text into Item sections.

    The table of contents repeats every Item heading, so for each item we keep
    its last heading occurrence, which is normally the start of the real section.
    """
    last_match: dict[str, int] = {}
    for match in ITEM_PATTERN.finditer(text):
        last_match[match.group(1).upper()] = match.start()

    if not last_match:
        return [Section(item="FULL", title="Full Filing", text=text)]

    starts = sorted((pos, item) for item, pos in last_match.items())
    sections = []
    for idx, (pos, item) in enumerate(starts):
        end = starts[idx + 1][0] if idx + 1 < len(starts) else len(text)
        body = text[pos:end].strip()
        if len(body) > 200:  # drop stray headings and cross-references
            sections.append(Section(item=item, title=ITEM_TITLES.get(item, ""), text=body))
    return sections


def parse_filing(html: str, metadata: FilingMetadata) -> Filing:
    return Filing(metadata=metadata, sections=split_into_items(html_to_text(html)))
