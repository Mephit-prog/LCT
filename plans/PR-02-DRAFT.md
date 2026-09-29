# PR-02 · Shared bootstrap and versioned acceptance profile

**Base:** PR-01 (stacked, not an independent merge). **Scope:** configuration/startup, policy compatibility, request-profile refusal; no new models or quality claims.

## Description
- `wineid/runtime.py` shares preflight/bootstrap/lifecycle between FastAPI and stdlib HTTP: validate finite timeouts, concurrency, OCR mode, aliases, local paths, remote-code opt-in, vision settings and policy type before serving; close OCR on partial startup/shutdown and close vision when supported. Canary runs before loading vision weights. The default ROI remains `refuse`.
- `wineid/profile.py`, `pipeline.py`, `text_policy.py`, `fusion_policy.py`: text/fusion policy schema **v3** binds aliases, OCR identity/manifest/weights, model/index/adapter, ROI/TTA/device, blend budgets, conditional OCR, Python/library versions, source-code hash, request timeout and OCR concurrency. Changed runtime profile fails closed; manual bbox, ROI override, oracle text and a larger-than-calibrated request budget cannot produce an accepted slug. HTTP startup refuses obsolete/legacy policy and mismatched budget. Eval remains forced top-1, not acceptance.
- Regression tests in `tests/test_runtime_profile.py` and `tests/test_release_profile.py` cover config mutations, HTTP adapters, mismatched policy, canary and partial startup cleanup. Updated `API.md` documents schema and settings.

## Verification
- Local `.venv` (Python 3.12): `python -m pytest -q -rs` (**215 passed, 10 skipped, 2 third-party warnings**), `python -m pyflakes wineid tests`, `python -m compileall -q wineid tests`, `python -m pip check`, `git diff --check`. Both HTTP adapter factories initialized from the same offline `open_runtime({})` successfully with 2103 catalog rows; no real OCR/vision weights downloaded.

## Compatibility / residual risk
- v2 artifacts are rejected; regenerate/recalibrate with the exact v3 profile and independent validation manifest. Do **not** convert thresholds by renaming the version. No independent accuracy, Jina/Paddle canary on target hardware, worker-wide hard deadline or cross-process resource limit is claimed (PR-03 through PR-07).
- The working tree contains previous uncommitted OCR/fusion work and in-progress PR-03 changes. Prepare the stacked commit by reviewing selected paths/hunks; do not stage generated artifacts or silently include unrelated work. No branch pushed and no remote PR created.
