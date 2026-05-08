---
phase: 9
slug: frozen-handoff-doc
status: complete
nyquist_compliant: true
wave_0_complete: true
created: 2026-05-08
---

# Phase 9 — Validation Strategy

> Phase 9 is documentation composition + freeze. Validation is deterministic content checks plus a freeze immutability check.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | shell + content checks |
| **Config file** | none |
| **Quick run command** | `python -m evaluation.harness.freeze --version 1.0` (expected to refuse-clobber if already frozen) |
| **Full suite command** | `pytest -m 'not live'` |
| **Estimated runtime** | ~30–120 seconds (pytest); freeze refusal is instantaneous |

---

## Sampling Rate

- **After every docs task commit:** Run the relevant content check command(s) for the updated file(s)
- **After every plan wave:** Run `pytest -m 'not live'` if any source code is touched (ideally Phase 9 touches no source)
- **Before `/gsd-verify-work`:** Ensure the frozen markdown + manifest exist and include the required sections
- **Max feedback latency:** 2 minutes

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 09-01-01 | 01 | 1 | DOC-01..DOC-05 | — | Avoid leaking secrets | content | `python -c \"from pathlib import Path; p=Path('evaluation/results/comparison.md'); assert p.exists(); s=p.read_text(encoding='utf-8'); assert 'Tier Rollup' in s; assert 'Per-Question-Class' in s\"` | ✅ | ✅ green |
| 09-01-02 | 01 | 1 | DOC-04 | — | Avoid leaking secrets | content | `python -c \"from pathlib import Path; p=Path('.planning/phases/08-multi-judge-spot-check/08-02-SUMMARY.md'); assert p.exists(); s=p.read_text(encoding='utf-8'); assert 'tier-1' in s and 'tier-4' in s and 'tier-5' in s\"` | ✅ | ✅ green |
| 09-02-01 | 02 | 2 | DOC-01..DOC-03 | — | Avoid leaking secrets | content | `python -c \"from pathlib import Path; p=Path('evaluation/results/frozen/eval-numbers-v1.0.md'); assert p.exists(); s=p.read_text(encoding='utf-8'); assert 'Tier Rollup' in s; assert 'Per-Question-Class' in s\"` | ✅ | ✅ green |
| 09-02-02 | 02 | 2 | DOC-03 | — | Avoid leaking secrets | content | `python -c \"from pathlib import Path; p=Path('evaluation/results/frozen/eval-numbers-v1.0.manifest.json'); assert p.exists()\"` | ✅ | ✅ green |
| 09-02-03 | 02 | 2 | DOC-03 | — | Avoid leaking secrets | content | `.venv/bin/python -m evaluation.harness.freeze --version 1.0` (must exit non-zero and print refuse-clobber message if already frozen) | ✅ | ✅ green |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

Existing infrastructure covers all phase requirements.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Frozen markdown does not include secrets (API keys, tokens, env dumps) | DOC-04 | Automated checks are brittle; false negatives are costly | Open `evaluation/results/frozen/eval-numbers-v1.0.md` and visually scan for obvious secret patterns (e.g. `OPENROUTER_API_KEY`, `sk-`, `Bearer `, `.env`). |

---

## Validation Sign-Off

- [x] All tasks have automated verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all missing references
- [x] No watch-mode flags
- [x] Feedback latency < 120s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** approved 2026-05-08
**Executed:** 2026-05-08

