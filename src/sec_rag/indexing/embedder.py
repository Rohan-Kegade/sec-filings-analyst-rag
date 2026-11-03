"""Voyage AI embeddings over its HTTP API (voyage-finance-2 by default).

Voyage embeds a text differently depending on whether it is a stored passage ("document")
or a search query ("query"), so the two methods below send different input types.
Vectors come back normalized to length 1.
"""

import logging
import time

import requests

from sec_rag.config import settings

logger = logging.getLogger(__name__)

# Voyage models are served both by Voyage itself and by MongoDB Atlas, with separate keys:
# Atlas keys start with "al-" and work only on the Atlas endpoint.
VOYAGE_API_URL = "https://api.voyageai.com/v1/embeddings"
ATLAS_API_URL = "https://ai.mongodb.com/v1/embeddings"
RETRY_STATUSES = {429, 500, 502, 503, 504}  # rate limited or a temporary server error
MAX_RETRY_DELAY = 60.0  # seconds


def api_url_for_key(api_key: str | None) -> str:
    return ATLAS_API_URL if api_key and api_key.startswith("al-") else VOYAGE_API_URL


class Embedder:
    def __init__(
        self,
        model_name: str = settings.embedding_model,
        api_key: str | None = settings.voyage_api_key,
        api_url: str | None = settings.embedding_api_url,
        batch_size: int = settings.embedding_batch_size,
        max_retries: int = settings.embedding_max_retries,
        session: requests.Session | None = None,
    ):
        self.model_name = model_name
        self._api_key = api_key
        self.api_url = api_url or api_url_for_key(api_key)
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.session = session or requests.Session()
        self.total_tokens = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for start in range(0, len(texts), self.batch_size):
            vectors.extend(self._embed(texts[start : start + self.batch_size], "document"))
            if len(texts) > self.batch_size:
                logger.info("Embedded %d/%d chunks", len(vectors), len(texts))
        return vectors

    def embed_query(self, query: str) -> list[float]:
        return self._embed([query], "query")[0]

    def _embed(self, texts: list[str], input_type: str) -> list[list[float]]:
        if not texts:
            return []
        if not self._api_key:
            raise RuntimeError(
                "VOYAGE_API_KEY is not set. Add a Voyage AI or MongoDB Atlas API key to .env."
            )
        payload = {"input": texts, "model": self.model_name, "input_type": input_type}
        headers = {"Authorization": f"Bearer {self._api_key}"}
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.post(self.api_url, json=payload, headers=headers, timeout=60)
            except (requests.ConnectionError, requests.Timeout) as error:
                if attempt == self.max_retries:
                    raise
                self._wait(attempt, None, f"connection error ({error})")
                continue
            if response.status_code in RETRY_STATUSES and attempt < self.max_retries:
                self._wait(attempt, response.headers.get("Retry-After"),
                           f"HTTP {response.status_code}")
                continue
            if not response.ok:
                raise RuntimeError(
                    f"Voyage embeddings request failed (HTTP {response.status_code}): "
                    f"{response.text[:500]}"
                )
            body = response.json()
            self.total_tokens += body.get("usage", {}).get("total_tokens", 0)
            return [item["embedding"] for item in sorted(body["data"], key=lambda d: d["index"])]
        raise AssertionError("unreachable")

    def _wait(self, attempt: int, retry_after: str | None, reason: str) -> None:
        try:
            delay = float(retry_after) if retry_after else 2.0 * 2**attempt
        except ValueError:
            delay = 2.0 * 2**attempt
        delay = min(delay, MAX_RETRY_DELAY)
        logger.warning("Voyage %s; retrying in %.0fs (attempt %d/%d)",
                       reason, delay, attempt + 1, self.max_retries)
        time.sleep(delay)
