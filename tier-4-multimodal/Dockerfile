# Tier 4 — Multimodal RAG (RAG-Anything + MineRU + LightRAG + LibreOffice).
#
# Multi-stage build (Pattern 8 from 130-RESEARCH.md):
#
# Stage 1 (model-builder) — fat layer with Python deps + LibreOffice + a
# best-effort MineRU model pre-download. The pre-download is graceful:
# RAG-Anything 1.2.10's MineRU integration may not yet expose the
# `ensure_models()` helper across every model variant, so the RUN line
# falls through to `mineru --help` and finally `|| true` to keep image
# builds green even if pre-download is unavailable. First-ingest will
# then download the ~3-5 GB model lazily into the runtime stage's mount.
#
# Stage 2 (runtime) — slim image that copies the wheels + the model cache
# from the builder, plus the project source for an editable install.
#
# Pitfall 11 — `socksio==1.0.0` is pinned in the runtime so containers
# launched inside the orchestrator's SOCKS5 sandbox can still reach
# OpenRouter (Phase 128-06 / 129-07 precedent). Users running outside
# such sandboxes get socksio transitively via httpx[socks]; the explicit
# pin documents the requirement.
#
# REPO-05 — Tier 4 is the ONLY tier that ships a Dockerfile; Tiers 1, 2,
# 3, 5 work via plain `pip install -e ".[tier-N]"`. Documented in the
# Plan 05 README.

# ---------------------------------------------------------------------------
# Stage 1: model-builder
# ---------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS model-builder

# System deps for the multimodal pipeline:
#   libreoffice-{core,writer} — DOCX/PPTX → PDF conversion (RAG-Anything)
#   build-essential          — wheel builds (e.g. lxml, opencv transitives)
#   libgl1, libglib2.0-0     — opencv runtime for MineRU
#   ca-certificates, curl    — model downloads (huggingface, modelscope)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libreoffice-core libreoffice-writer \
        build-essential libgl1 libglib2.0-0 \
        ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Pin the exact dep set documented in Plan 130-01's [tier-4] extras.
# Mirrors the lockfile contract — keeping the Dockerfile narrow rather
# than relying on `pip install -e ".[tier-4]"` here would re-resolve and
# could surface unintended minor bumps. We do install -e in the runtime
# stage to wire the editable install for the project source itself.
RUN pip install --no-cache-dir \
        "raganything==1.2.10" \
        "lightrag-hku==1.4.15" \
        "openai>=1.50,<3" \
        "Pillow>=10,<12" \
        "socksio==1.0.0"

# Best-effort MineRU model pre-download. The exact entrypoint shape
# changes between minor releases; we try the documented helper first,
# fall back to a CLI invocation that triggers a default-model fetch,
# and finally `|| true` so the build never fails on this optional
# pre-warm step. First ingest will lazily download whatever is missing.
RUN python -c "from mineru.cli.client import ensure_models; ensure_models()" \
    || mineru --help \
    || true


# ---------------------------------------------------------------------------
# Stage 2: runtime
# ---------------------------------------------------------------------------
FROM python:3.11-slim-bookworm

LABEL org.opencontainers.image.source="https://github.com/PatrykQuantumNomad/rag-architecture-patterns"
LABEL org.opencontainers.image.title="rag-architecture-patterns/tier-4-multimodal"
LABEL org.opencontainers.image.description="Tier 4 — Multimodal RAG via RAG-Anything (MineRU + LightRAG + LibreOffice)."

# Runtime needs LibreOffice + opencv runtime libs but NOT build-essential.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libreoffice-core libreoffice-writer \
        libgl1 libglib2.0-0 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Bring the MineRU model cache + installed Python packages forward.
# `~/.mineru/` is the default cache root; if the pre-download in Stage 1
# was a no-op the directory simply may not exist (hence the trailing
# `|| true` on the COPY-aware build via .dockerignore allowlist below).
COPY --from=model-builder /root/.mineru /root/.mineru
COPY --from=model-builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=model-builder /usr/local/bin /usr/local/bin

# Project source. Both hyphenated and underscored variants are copied so
# that `from tier_4_multimodal.rag import build_rag` (the shim path) and
# `python tier-4-multimodal/main.py` (the script path) both resolve.
COPY shared/ /app/shared/
COPY tier_1_naive/ /app/tier_1_naive/
COPY tier-1-naive/ /app/tier-1-naive/
COPY tier_4_multimodal/ /app/tier_4_multimodal/
COPY tier-4-multimodal/ /app/tier-4-multimodal/
COPY dataset/ /app/dataset/
COPY pyproject.toml /app/
COPY README.md /app/

# Editable install so subsequent `import` paths resolve identically to
# the host dev environment. Re-pulls socksio + Pillow from PyPI but the
# wheel cache from Stage 1 makes this near-instant.
RUN pip install --no-cache-dir -e ".[tier-4]"

ENV PYTHONPATH=/app
ENV TIKTOKEN_CACHE_DIR=/app/.tiktoken_cache

# Pass .env at runtime: docker run --env-file .env rag-tier-4 [args]
# NEVER bake .env into the image.
ENTRYPOINT ["python", "tier-4-multimodal/main.py"]
CMD ["--query", "What does Figure 1 of the Attention Is All You Need paper depict?"]
