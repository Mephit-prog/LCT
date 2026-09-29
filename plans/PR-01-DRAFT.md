# PR-01 · Clean offline install and mandatory CI

**Base:** `lora-jina-synthetic_2` at the reviewed baseline. **Scope:** dependency/lint/CI only; no acceptance threshold or new model. This is the prerequisite for PR-02.

## Description
- Align Pillow/OpenCV with the Python 3.12 constraints; add `requirements-dev.txt` and `.github/workflows/offline.yml` (`pip check`, `compileall`, `pyflakes`, `pytest -q -rs`). Heavy model/OCR smoke remains opt-in, with skips explained.
- Remove eight pyflakes findings and deprecated `Image.fromarray(..., mode=...)` usages.
- Keep fuzzywuzzy's `difflib.SequenceMatcher` backend stable, without `python-Levenshtein` (would change scoring).

## Verification
- In the local clean Python 3.12 `.venv`: `pip check`, `python -m compileall -q wineid tests`, `python -m pyflakes wineid tests`, `python -m pytest -q -rs`: **215 passed, 10 skipped, 2 external-library warnings** with the combined PR-01/PR-02/working-copy tests. Hosted GitHub Actions has not run; no weights or external OCR service were used.
- CI is required to re-run from a fresh checkout. The current working tree also contains unrelated/untracked OCR/fusion and PR-03 work: select commits carefully; do not assume those files are present in a fresh checkout.

## Review / rollout
- No change to API acceptance semantics in this PR. Check the selected dependency versions on Ubuntu/Python 3.12 via hosted CI before merging. Do not stage `artifacts/`, weights, or private manifests.
