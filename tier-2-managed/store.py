"""Tier 2 (Gemini File Search) — FileSearchStore lifecycle helpers.

This module wraps ``client.file_search_stores.*`` and the long-running
``upload_to_file_search_store`` operation. It is the only place in Tier 2
that knows about operation polling and 503 retry semantics — main.py and
query.py treat the store as a black-box handle.

Decisions referenced (see 129-RESEARCH.md):

- Pattern 1: store-id caching via ``.store_id`` sidecar so re-runs are idempotent.
- Pattern 2: long-running operation polling (``client.operations.get(op)``).
- Pitfall 2: 30s/60s/120s exponential backoff for 503 TPM saturation, with
  ``documents.list()`` used to skip already-uploaded PDFs on retry.

The 5 lifecycle helpers are intentionally small so the CLI in Plan 04 can
use them as black boxes without learning about operations or retries.
"""

from __future__ import annotations

import time
from pathlib import Path

from google import genai
from google.genai import types

# Cache file path: ``tier-2-managed/.store_id`` (gitignored from Plan 129-01).
# Resolved relative to *this* module so callers can run from any CWD.
STORE_ID_PATH = Path(__file__).resolve().parent / ".store_id"

# Display name used when creating the store. Stable across runs so the
# server-side store is recognizable in the Gemini console.
STORE_DISPLAY_NAME = "rag-arch-patterns-tier-2"

# Cookbook documents 1-3s polling intervals; 2s is the comfortable middle.
POLL_INTERVAL_S = 2.0


def get_or_create_store(client: genai.Client, reset: bool = False):
    """Resolve the cached FileSearchStore handle, or create a new one.

    Caches the server-assigned name in ``STORE_ID_PATH`` (gitignored) so
    repeated ``--ingest`` calls re-use the same store. With ``reset=True``,
    deletes both the cached handle and any server-side store with the
    cached name, then falls through to create a fresh one.

    Returns the SDK ``FileSearchStore`` object whose ``.name`` is the
    server-assigned identifier (``fileSearchStores/<id>``).
    """
    if reset and STORE_ID_PATH.exists():
        cached = STORE_ID_PATH.read_text().strip()
        try:
            client.file_search_stores.delete(name=cached, config={"force": True})
        except Exception:
            # Already gone or never existed; OK to silently proceed.
            pass
        STORE_ID_PATH.unlink()

    if STORE_ID_PATH.exists():
        cached = STORE_ID_PATH.read_text().strip()
        try:
            return client.file_search_stores.get(name=cached)
        except Exception:
            # Stale cache (store was deleted server-side); fall through to recreate.
            STORE_ID_PATH.unlink()

    store = client.file_search_stores.create(
        config=types.CreateFileSearchStoreConfig(
            display_name=STORE_DISPLAY_NAME,
        )
    )
    STORE_ID_PATH.parent.mkdir(parents=True, exist_ok=True)
    STORE_ID_PATH.write_text(store.name)
    return store


def upload_pdf(
    client: genai.Client,
    store_name: str,
    pdf_path: str,
    display_name: str,
):
    """Upload a single PDF and poll until the long-running operation is done.

    Returns the completed Operation object (``op.response`` holds the
    imported document handle). Raises ``RuntimeError`` if the operation
    reports an error in ``op.error``.
    """
    op = client.file_search_stores.upload_to_file_search_store(
        file=pdf_path,
        file_search_store_name=store_name,
        config=types.UploadToFileSearchStoreConfig(display_name=display_name),
    )
    # Poll until done. Operation.done flips True when the embedding/index
    # step finishes server-side. The walrus refresh of ``op`` is per cookbook.
    while not (op := client.operations.get(op)).done:
        time.sleep(POLL_INTERVAL_S)
    if getattr(op, "error", None):
        raise RuntimeError(
            f"upload_to_file_search_store failed for {pdf_path}: {op.error}"
        )
    return op


def upload_with_retry(
    client: genai.Client,
    store_name: str,
    pdf_path: str,
    display_name: str,
    max_attempts: int = 3,
    base_wait_s: float = 30.0,
):
    """Wrap :func:`upload_pdf` with 30s/60s/120s exponential backoff.

    Pitfall 2 (discuss.ai.google.dev/t/121691): the community-verified
    minimum pause that dodges 503 TPM saturation is ~30s. Faster retries
    don't recover. Sequential single-doc uploads (not concurrent) plus this
    backoff is the documented escape hatch.
    """
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return upload_pdf(client, store_name, pdf_path, display_name)
        except Exception as e:
            last_exc = e
            if attempt == max_attempts - 1:
                break
            wait = base_wait_s * (2**attempt)  # 30, 60, 120
            time.sleep(wait)
    assert last_exc is not None
    raise last_exc


def list_existing_documents(client: genai.Client, store_name: str) -> set[str]:
    """Return display_names of documents already in the store.

    Used to make ``--ingest`` idempotent (Pitfall 2 resilience): if a 503
    storm interrupts a previous run, we resume from where we left off
    instead of re-uploading already-indexed PDFs.

    The SDK's ``documents.list`` takes the FileSearchStore resource name via
    the ``parent`` kwarg (NOT ``file_search_store_name`` — that name belongs
    to ``upload_to_file_search_store`` / ``import_file``; ``Documents.list``
    uses the standard Google API ``parent`` convention). It returns a
    ``google.genai.pagers.Pager[types.Document]`` that auto-fetches the next
    page on iteration, so a single ``for`` loop covers stores larger than
    one page.
    """
    out: set[str] = set()
    pager = client.file_search_stores.documents.list(parent=store_name)
    for doc in pager:
        dn = getattr(doc, "display_name", None)
        if dn:
            out.add(dn)
    return out


def delete_store(client: genai.Client, store_name: str) -> None:
    """Force-delete the store and remove the cached ``.store_id`` sidecar.

    Errors during the server-side delete (already gone, network blip) are
    silently swallowed because the only useful local action — clearing the
    sidecar — runs unconditionally. The next ``get_or_create_store`` call
    will create a fresh store.
    """
    try:
        client.file_search_stores.delete(name=store_name, config={"force": True})
    except Exception:
        pass
    if STORE_ID_PATH.exists():
        STORE_ID_PATH.unlink()
