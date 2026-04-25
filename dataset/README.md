# Dataset

The corpus is a deliberate **citation cluster anchored on the RAG / retrieval-augmented LM literature** — meta-recursive by design: every architecture in this repo indexes the papers that taught us how to build it.

## Composition (target)

| Asset class | Count | Storage | Source |
|-------------|-------|---------|--------|
| arXiv papers (PDF) | ~100 | `dataset/papers/` (LFS) | arXiv via `arxiv` + `semanticscholar` (Plan 04) |
| Extracted figures (PNG/JPG) | ~150–300 | `dataset/images/` (LFS) | `pymupdf` extraction from papers (Plan 05) |
| Video clips (MP4/WEBM) | 1–2 | `dataset/videos/` (LFS) | CC-licensed talks/explainers (Plan 05) |
| Manifests (JSON) | 4 | `dataset/manifests/` | `papers.json`, `figures.json`, `videos.json`, `metadata.json` (Plans 04–06) |

Total disk footprint targets ~300–500 MB, all routed through git-lfs.

## Provenance

Curation seeds on a small set of canonical retrieval-augmented LM papers, then walks the citation graph (forward + backward) via Semantic Scholar to expand to ~100 entries. The full curation pipeline (deduplication, license check, manifest assembly) lives in `scripts/` and is implemented in Plan 03.

Each paper carries a `paper_id`, arXiv ID, title, authors, venue, and a snapshot of its abstract in `manifests/papers.json`.

## Licensing

| Asset class | License model | Notes |
|-------------|---------------|-------|
| PDFs (papers) | [arXiv non-exclusive distribution license](https://arxiv.org/licenses/) | Original copyright retained by authors; redistribution permitted for non-commercial research per arXiv ToU |
| Figures (images) | Inherits parent paper's terms | Extracted at curation time; manifest records the source `paper_id` so the parent license is always traceable |
| Videos | Per-clip Creative Commons | Acceptable: CC BY, CC BY-SA, CC BY-NC, CC BY-NC-SA. **Rejected:** CC BY-ND (no-derivatives blocks transcription/embedding). Per-clip license stored in `manifests/videos.json`. |

The corpus is intended for non-commercial research — specifically as the test bed for the blog post's architecture comparison. Anyone reusing the corpus is responsible for verifying their use case fits each asset's license.

## Storage: git-lfs

All binaries (`*.pdf`, `*.png`, `*.jpg`, `*.jpeg`, `*.mp4`, `*.webm`) are tracked by git-lfs — see `../.gitattributes`. The LFS filter must already be active before any binary is staged; cloning with `GIT_LFS_SKIP_SMUDGE=1` lets you check out code without pulling the corpus.

Before adding a binary, verify the LFS filter applies:

```bash
git check-attr filter dataset/papers/<file>.pdf
# Expected: dataset/papers/<file>.pdf: filter: lfs
```

If the output says `filter: unspecified`, **stop** — the binary will land in the regular pack-file and bloat the repo permanently.

## Manifests

| File | Schema fingerprint | Built by |
|------|--------------------|----------|
| `manifests/papers.json` | `[{ paper_id, arxiv_id, title, authors, year, venue, abstract, pdf_path, license }]` | Plan 04 |
| `manifests/figures.json` | `[{ figure_id, paper_id, image_path, caption, page }]` | Plan 05 |
| `manifests/videos.json` | `[{ video_id, title, source_url, license, video_path, transcript_path }]` | Plan 05 |
| `manifests/metadata.json` | Top-level index: counts per asset class, build timestamp, schema version | Plan 06 |

The manifests are the single source of truth: every tier loads them rather than scanning `dataset/` directly.
