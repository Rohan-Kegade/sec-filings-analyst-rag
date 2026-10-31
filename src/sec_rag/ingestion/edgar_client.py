"""Download 10-K filings from SEC EDGAR.

Uses the public EDGAR endpoints:
  - https://www.sec.gov/files/company_tickers.json      (ticker -> CIK)
  - https://data.sec.gov/submissions/CIK##########.json  (filing index per company)
"""

import json
import time
from pathlib import Path

import requests

from sec_rag.config import settings
from sec_rag.models import FilingMetadata

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"


class EdgarClient:
    def __init__(self, user_agent: str = settings.edgar_user_agent):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})
        self._ticker_map: dict[str, dict] | None = None

    def _get(self, url: str) -> requests.Response:
        time.sleep(settings.edgar_request_delay)
        response = self.session.get(url, timeout=30)
        response.raise_for_status()
        return response

    def lookup_company(self, ticker: str) -> dict:
        if self._ticker_map is None:
            data = self._get(TICKERS_URL).json()
            self._ticker_map = {row["ticker"].upper(): row for row in data.values()}
        try:
            return self._ticker_map[ticker.upper()]
        except KeyError:
            raise ValueError(f"Unknown ticker: {ticker}") from None

    def list_10k_filings(self, ticker: str, limit: int = 5) -> list[FilingMetadata]:
        company = self.lookup_company(ticker)
        cik = str(company["cik_str"]).zfill(10)
        recent = self._get(SUBMISSIONS_URL.format(cik=cik)).json()["filings"]["recent"]

        filings = []
        for i, form in enumerate(recent["form"]):
            if form != "10-K":
                continue
            accession = recent["accessionNumber"][i]
            filings.append(
                FilingMetadata(
                    ticker=ticker.upper(),
                    cik=cik,
                    company_name=company["title"],
                    fiscal_year=int(recent["reportDate"][i][:4]),
                    filing_date=recent["filingDate"][i],
                    accession_number=accession,
                    source_url=ARCHIVE_URL.format(
                        cik=int(cik),
                        accession=accession.replace("-", ""),
                        document=recent["primaryDocument"][i],
                    ),
                )
            )
            if len(filings) >= limit:
                break
        return filings

    def download(self, metadata: FilingMetadata, out_dir: Path = settings.raw_dir) -> Path:
        """Save the filing HTML plus a metadata sidecar. Skips files already on disk."""
        target_dir = out_dir / metadata.ticker
        target_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{metadata.fiscal_year}_{metadata.accession_number}"
        html_path = target_dir / f"{stem}.html"
        meta_path = target_dir / f"{stem}.json"

        if not html_path.exists():
            html_path.write_bytes(self._get(metadata.source_url).content)
            meta_path.write_text(metadata.model_dump_json(indent=2))
        return html_path


def load_raw_filings(raw_dir: Path = settings.raw_dir) -> list[tuple[FilingMetadata, Path]]:
    """Return (metadata, html_path) for every downloaded filing."""
    results = []
    for meta_path in sorted(raw_dir.glob("*/*.json")):
        metadata = FilingMetadata(**json.loads(meta_path.read_text()))
        results.append((metadata, meta_path.with_suffix(".html")))
    return results
