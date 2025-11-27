"""
Tests for SupadataLoader integration.

These tests:
- Do NOT hit the real Supadata API.
- Patch the Supadata SDK client and assert correct calls.
- Check that Documents are created with expected content/metadata.
"""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

# >>> IMPORTANT <<<
# If your loader lives somewhere else (e.g. data_loaders.supadata),
# change BOTH of these strings accordingly.
LOADER_IMPORT_PATH = "langchain_community.document_loaders.supadata"
MODULE_PATH = LOADER_IMPORT_PATH


@pytest.fixture(autouse=True)
def clear_supadata_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure SUPADATA_API_KEY is clean for each test."""
    monkeypatch.delenv("SUPADATA_API_KEY", raising=False)


def make_mock_client() -> MagicMock:
    """Create a mock Supadata client with metadata/transcript methods."""
    client = MagicMock()
    client.metadata = MagicMock()
    client.transcript = MagicMock()
    return client


@patch(f"{MODULE_PATH}.Supadata")
def test_metadata_operation_uses_explicit_api_key(
    mock_supadata_cls: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Loader with operation='metadata' should call Supadata.metadata correctly."""
    # No env var, we want to force usage of explicit api_key
    monkeypatch.delenv("SUPADATA_API_KEY", raising=False)

    from langchain_community.document_loaders.supadata import SupadataLoader

    mock_client = make_mock_client()
    mock_client.metadata.return_value = {"title": "Test Title"}
    mock_supadata_cls.return_value = mock_client

    url = "https://example.com/video"
    loader = SupadataLoader(
        urls=[url],
        api_key="EXPLICIT_KEY",
        operation="metadata",
        params={"foo": "bar"},
    )

    docs = list(loader.lazy_load())

    # Supadata client should be constructed with our explicit key
    mock_supadata_cls.assert_called_once_with(api_key="EXPLICIT_KEY")

    # metadata() should be called with url + params
    mock_client.metadata.assert_called_once_with(url=url, foo="bar")

    assert len(docs) == 1
    doc = docs[0]
    assert isinstance(doc, Document)
    assert '"title"' in doc.page_content
    assert doc.metadata["source"] == url
    assert doc.metadata["supadata_operation"] == "metadata"


@patch(f"{MODULE_PATH}.Supadata")
def test_metadata_operation_uses_env_api_key_when_not_provided(
    mock_supadata_cls: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If api_key is not passed, loader should use SUPADATA_API_KEY env var."""
    monkeypatch.setenv("SUPADATA_API_KEY", "ENV_KEY")

    from langchain_community.document_loaders.supadata import SupadataLoader

    mock_client = make_mock_client()
    mock_client.metadata.return_value = {"id": "123"}
    mock_supadata_cls.return_value = mock_client

    loader = SupadataLoader(
        urls=["https://example.com"],
        operation="metadata",
    )

    list(loader.lazy_load())

    mock_supadata_cls.assert_called_once_with(api_key="ENV_KEY")


@patch(f"{MODULE_PATH}.Supadata")
def test_transcript_operation_immediate_result(
    mock_supadata_cls: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    For smaller inputs Supadata.transcript returns a transcript object
    with a 'content' attribute. Loader should put that into page_content.
    """
    monkeypatch.setenv("SUPADATA_API_KEY", "TEST_KEY")

    from langchain_community.document_loaders.supadata import SupadataLoader

    mock_client = make_mock_client()

    class DummyTranscript:
        def __init__(self) -> None:
            self.content = "hello from transcript"
            self.lang = "en"

    mock_client.transcript.return_value = DummyTranscript()
    mock_supadata_cls.return_value = mock_client

    url = "https://example.com/video"
    loader = SupadataLoader(
        urls=[url],
        operation="transcript",
        lang="en",
        text=True,
        mode="auto",
    )

    docs = list(loader.lazy_load())
    assert len(docs) == 1

    doc = docs[0]
    assert isinstance(doc, Document)
    assert "hello from transcript" in doc.page_content
    assert doc.metadata["source"] == url
    assert doc.metadata["supadata_operation"] == "transcript"
    assert doc.metadata.get("lang") in ("en", None)  # depends on your impl


@patch(f"{MODULE_PATH}.Supadata")
def test_transcript_operation_job_result(
    mock_supadata_cls: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    For larger inputs Supadata.transcript may return a job object with 'job_id'.
    Loader should return an empty page_content and put job_id in metadata.
    """
    monkeypatch.setenv("SUPADATA_API_KEY", "TEST_KEY")

    from langchain_community.document_loaders.supadata import SupadataLoader

    mock_client = make_mock_client()

    class DummyJob:
        def __init__(self) -> None:
            self.job_id = "job-123"

    mock_client.transcript.return_value = DummyJob()
    mock_supadata_cls.return_value = mock_client

    url = "https://example.com/long-video"
    loader = SupadataLoader(
        urls=[url],
        operation="transcript",
        lang="en",
        mode="auto",
    )

    docs = list(loader.lazy_load())
    assert len(docs) == 1

    doc = docs[0]
    assert isinstance(doc, Document)
    assert doc.page_content == ""  # by design for job-based results
    assert doc.metadata["source"] == url
    assert doc.metadata["supadata_operation"] in ("transcript_job", "transcript")
    assert doc.metadata.get("job_id") == "job-123"
