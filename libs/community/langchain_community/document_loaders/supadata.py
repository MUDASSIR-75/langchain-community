from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Literal, Optional

from langchain_core.documents import Document
from langchain_core.utils import get_from_env
from langchain_community.document_loaders.base import BaseLoader

SupadataOperation = Literal["metadata", "transcript"]

# Module-level alias so tests can patch `Supadata` via
# langchain_community.document_loaders.supadata.Supadata
try:  # pragma: no cover - exercised via tests with mocking
    from supadata import Supadata  # type: ignore[import]
except Exception:  # pragma: no cover
    Supadata = None  # type: ignore[assignment]


@dataclass
class SupadataLoader(BaseLoader):
    """Load documents from the Supadata Web & Video Data API.

    This loader wraps the official :mod:`supadata` Python SDK to fetch either:

    * structured media metadata (``operation="metadata"``)
    * media transcripts (``operation="transcript"``)

    Parameters
    ----------
    urls:
        List of URLs to fetch from Supadata.
    api_key:
        Supadata API key. If omitted, the ``SUPADATA_API_KEY`` environment
        variable is used.
    operation:
        Which Supadata endpoint to call: ``"metadata"`` or ``"transcript"``.
    lang:
        Optional transcript language preference.
    text:
        When ``True``, request a plain-text transcript instead of timestamped
        chunks (see Supadata documentation).
    mode:
        Transcript mode, for example ``"native"``, ``"auto"`` or ``"generate"``.
    params:
        Extra keyword arguments forwarded to the underlying Supadata SDK call.
    """

    urls: List[str]
    api_key: Optional[str] = None
    operation: SupadataOperation = "transcript"

    lang: Optional[str] = None
    text: Optional[bool] = None
    mode: Optional[str] = None

    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Explicit api_key wins; otherwise read from env.
        if self.api_key is None:
            # get_from_env(key, env_var) -> str
            self.api_key = get_from_env("api_key", "SUPADATA_API_KEY")

    def _get_client(self) -> Any:
        if Supadata is None:
            raise ImportError(
                "Could not import 'supadata'. Install it with "
                "`pip install supadata` to use `SupadataLoader`."
            )

        if not self.api_key:
            raise ValueError(
                "Supadata API key is empty. "
                "Set the SUPADATA_API_KEY environment variable or "
                "pass `api_key` when constructing SupadataLoader."
            )

        return Supadata(api_key=self.api_key)

    def lazy_load(self) -> Iterable[Document]:
        client = self._get_client()

        for url in self.urls:
            if self.operation == "metadata":
                yield self._load_metadata(client, url)
            elif self.operation == "transcript":
                yield self._load_transcript(client, url)
            else:
                raise ValueError(
                    f"Unsupported operation: {self.operation!r}. "
                    "Expected 'metadata' or 'transcript'."
                )

    def _load_metadata(self, client: Any, url: str) -> Document:
        result = client.metadata(url=url, **self.params)

        page_content = json.dumps(result, ensure_ascii=False, indent=2)
        metadata = {
            "source": url,
            "supadata_operation": "metadata",
        }

        return Document(page_content=page_content, metadata=metadata)

    def _load_transcript(self, client: Any, url: str) -> Document:
        transcript_kwargs: Dict[str, Any] = dict(self.params)
        if self.lang is not None:
            transcript_kwargs["lang"] = self.lang
        if self.text is not None:
            transcript_kwargs["text"] = self.text
        if self.mode is not None:
            transcript_kwargs["mode"] = self.mode

        result = client.transcript(url=url, **transcript_kwargs)

        content = getattr(result, "content", None)
        if content is not None:
            # Immediate transcript result.
            metadata = {
                "source": url,
                "supadata_operation": "transcript",
                "lang": getattr(result, "lang", None),
                "mode": transcript_kwargs.get("mode"),
            }
            return Document(page_content=str(content), metadata=metadata)

        # Asynchronous job: result carries a job_id instead of content.
        job_id = getattr(result, "job_id", None)
        metadata = {
            "source": url,
            "supadata_operation": "transcript_job",
            "job_id": job_id,
            "lang": transcript_kwargs.get("lang"),
            "mode": transcript_kwargs.get("mode"),
        }
        return Document(page_content="", metadata=metadata)
