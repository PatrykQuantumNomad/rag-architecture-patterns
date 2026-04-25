# Curation scripts

These scripts curate the corpus that powers every tier in this repo. They
operate on external services (arXiv, Semantic Scholar) and on local PDFs and
videos. Phase 127 Plans 04 and 05 RUN them; Plan 03 only AUTHORED them.

## Pipeline order

Run the scripts in this order. Each takes the prior step's output as input.

```
scripts/curate_corpus.py    → dataset/papers/*.pdf            (Plan 04)
scripts/extract_figures.py  → dataset/images/*.png            (Plan 05)
scripts/cut_video_clips.py  → dataset/videos/*.mp4            (Plan 05)
```

## Mantra: dry-run first

**Always run with `--dry-run` first.** Every script supports it. The dry-run
prints the planned actions — IDs, ffmpeg commands, figure counts — without
making any external calls or writing any files. Inspect the plan, then drop
`--dry-run` to commit.

```bash
python scripts/curate_corpus.py --dry-run --max-papers 5 --hops 1
python scripts/extract_figures.py --pdf dataset/papers/2005.11401_*.pdf --dry-run
python scripts/cut_video_clips.py --dry-run
```

## arXiv ToU compliance

**3 seconds between requests, single connection, no parallelization.** This is
the arXiv Terms of Use; violating it gets your IP silently rate-limited or
banned (per 127-RESEARCH.md Pitfall 2).

`scripts/curate_corpus.py` uses `arxiv.Client(delay_seconds=3.0,
num_retries=5, page_size=100)` — the library enforces compliance. **Never
bypass it.** Don't write a custom `requests` loop; don't run multiple
copies in parallel; don't lower `delay_seconds`.

## Semantic Scholar quota

Semantic Scholar runs on a shared 5,000-requests-per-5-minute pool when
unauthenticated — bursts get 429s.

If curation hits 429s, request a free API key at
<https://semanticscholar.org/product/api> and set it in `.env`:

```ini
S2_API_KEY=your_semantic_scholar_api_key_here
```

The script reads it via `shared.config.settings.s2_api_key` (with an
`SEMANTIC_SCHOLAR_API_KEY` env-var fallback) and includes exponential backoff
(5s × 2^attempt, up to 5 retries) on 429s per 127-RESEARCH.md Pitfall 3.

## Video clip license verification protocol

Before running `cut_video_clips.py`, manually verify each entry's CC license
at the source talk's page. The script REFUSES to cut clips whose license is
unverified or is "No Derivatives".

| License           | Status | Notes                                    |
| ----------------- | ------ | ---------------------------------------- |
| CC-BY             | ACCEPT | Attribution only                         |
| CC-BY-SA          | ACCEPT | ShareAlike — derivatives must match      |
| CC-BY-NC          | ACCEPT | Non-commercial — fine for research demo  |
| CC-BY-NC-SA       | ACCEPT | NC + SA combined                         |
| CC-BY-ND          | REJECT | No Derivatives — cutting is a derivative |
| CC-BY-NC-ND       | REJECT | NC + ND — same problem                   |
| TBD_VERIFY_MANUALLY | REJECT | Sentinel — script refuses until updated  |

**Verification steps:**

1. Visit the source URL (e.g., `https://slideslive.com/...`).
2. Find the license badge on the talk page (or in the conference proceedings).
3. Update the entry in `scripts/video_sources.json`:

   ```json
   {
     "license": "CC-BY-4.0",
     "license_verified_at": "2026-04-25",
     "license_verified_by": "manual_review"
   }
   ```

4. Set `source_file_local_path` to the locally-downloaded mp4/webm:

   ```json
   { "source_file_local_path": "/Users/you/Downloads/lewis_neurips2020.mp4" }
   ```

5. Run `python scripts/cut_video_clips.py --dry-run` — confirm the script
   accepts the entry and prints the ffmpeg command.

6. Drop `--dry-run` to cut the clip.

If the license cannot be confirmed, **drop the clip**. Do not guess.

## ffmpeg stream-copy

`cut_video_clips.py` uses `ffmpeg -ss S -t D -i IN -c copy OUT` — stream copy,
no re-encode. This is lossless and runs in milliseconds (per 127-RESEARCH.md
"Don't Hand-Roll"). Re-encoding (any `-c:v` other than `copy`) will introduce
compression artifacts and a 5-30x slowdown.

ffmpeg 8.1 is verified on the project dev environment. If your local
ffmpeg is older than 4.x, double-check that `-c copy` works for your
input container and codec.

## LFS reminder

These scripts produce binaries (`*.pdf`, `*.png`, `*.jpg`, `*.jpeg`, `*.mp4`,
`*.webm`) that are tracked by git-lfs via `.gitattributes`. Verify the LFS
filter is active after the first download:

```bash
git check-attr filter dataset/papers/<file>.pdf  # should show: filter: lfs
git lfs ls-files                                  # should list pushed binaries
```

Before bulk pushes (Plans 04 and 05), confirm your GitHub LFS quota at
<https://github.com/settings/billing>. The free tier is 1 GB storage + 1 GB
bandwidth per month. The full corpus (~100 PDFs + figures + 1-2 video clips)
should land under that, but the per-file 100 MB limit and the monthly
bandwidth ceiling are easy to hit on iteration.

## Script reference

### `curate_corpus.py`

```
--dry-run               Plan only; no PDF downloads (S2 lookups still happen)
--seeds PATH            Seed JSON (default: scripts/seed_papers.json)
--output PATH           PDF download dir (default: dataset/papers/)
--max-papers N          Cluster size cap (default: 100)
--hops N                Citation graph hops (1=seeds, 2=+refs/cits, 3=+1; default: 2)
--output-manifest PATH  Manifest JSON (default: dataset/manifests/papers.json)
```

### `extract_figures.py`

```
--pdf PATH              Single-PDF mode
--all DIR               Batch mode: process every *.pdf in DIR
--out DIR               Output dir for figures (default: dataset/images/)
--manifest PATH         Manifest JSON (default: dataset/manifests/figures.json)
--min-dim N             Minimum width AND height in pixels (default: 200)
--max-aspect-ratio N    Reject if max(w/h, h/w) > N (default: 5.0)
--dry-run               Print what WOULD be extracted; write nothing
```

### `cut_video_clips.py`

```
--sources PATH          Source-clip JSON (default: scripts/video_sources.json)
--out DIR               Cut-clip dir (default: dataset/videos/)
--manifest PATH         Manifest JSON (default: dataset/manifests/videos.json)
--dry-run               Print ffmpeg commands; don't run them
```
