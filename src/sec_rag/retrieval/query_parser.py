"""Extract metadata filters (tickers, fiscal years) from a natural-language question."""

import re

YEAR_PATTERN = re.compile(r"\b(?:FY\s?)?(20\d{2})\b", re.IGNORECASE)
TICKER_PATTERN = re.compile(r"\b[A-Z]{1,5}\b")


def extract_filters(question: str, known_tickers: set[str]) -> dict:
    filters: dict = {}

    tickers = {t for t in TICKER_PATTERN.findall(question) if t in known_tickers}
    if tickers:
        filters["ticker"] = sorted(tickers) if len(tickers) > 1 else tickers.pop()

    years = sorted({int(y) for y in YEAR_PATTERN.findall(question)})
    if years:
        filters["fiscal_year"] = years if len(years) > 1 else years[0]

    return filters
