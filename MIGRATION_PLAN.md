# ParkingScore TXT v2 migration

Baseline: commit `d816ef8`; 31 tests passed; Ruff passed. Existing untracked
`.DS_Store` is unrelated and must remain untouched.

The initial alignment specification was superseded by the approved simpler
architecture: ParkingScore owns scoring, series and `best`; ParkingReview is a
passive TXT consumer and feedback/statistics store. No runtime scoring contract
is used.

Migration phases:

1. Add backward-compatible domain fields and grouped criteria parsing with
   automatic stable IDs/hash/version. Do not modify the production `criteria.txt`.
2. Apply idempotent additive SQLite migrations for current projections,
   append-only assessment history, criteria versions, and publication audit.
3. Rebuild series locally with configurable `SERIES_WINDOW_MINUTES` and
   `bounded-total-span`.
4. Enable prompt/response/TXT v2 while retaining the first two legacy lines and
   without a contract-version field.
   Publish TXT atomically and retain one assessment UUID across retries.
5. Add cursor NDJSON export and resumable `evaluate --no-publish`.
6. Run unit/integration tests before any production rollout.

Implementation status:

- phases 1–5 are implemented in the working tree;
- the ParkingReview changes required to consume contract-free TXT v2 are
  documented in `PARKING_REVIEW_CHANGES.md`; its source tree was not changed;
- production `criteria.txt`, `.env`, FTP and real API credentials were not read
  or changed during implementation;
- production rollout remains a separate operator decision after staging checks.

Verification recorded on 2026-08-26:

- baseline before migration: `31 passed`; Ruff clean;
- contract-free implementation: `49 passed`; Ruff clean; `git diff --check`
  clean;
- grouped example `criteria.txt` passes `validate-criteria`;
- no live FTP, ParkingReview mutation, AI request or real-secret test was run.

## Preflight and rollout

1. Stop the old worker so two scoring workers can never publish concurrently.
2. Record the previous image tag/commit and make a filesystem-consistent backup
   of `/data/parking_score.db` (including WAL/SHM when present), or snapshot the
   whole Docker volume while the worker is stopped.
3. Preserve the currently approved `criteria.txt` as a versioned, read-only
   rollback artifact.
4. Deploy the compatible ParkingReview importer described in
   `PARKING_REVIEW_CHANGES.md` before ParkingScore starts publishing TXT v2
   without `series_contract_version`.
5. Stage the new ParkingScore image with placeholder/test credentials and run
   unit, lint and `evaluate --no-publish` checks. Confirm there are no FTP upload
   calls in the evaluation audit.
6. Validate the approved grouped `criteria.txt`; confirm a canary TXT is v2,
   omits `series_contract_version`, and keeps the first two legacy lines.
7. Run one worker against a canary FTP subtree, inspect ParkingReview ingestion,
   SQLite history/export and operator display, then explicitly approve the
   production switch. Do not run old and new scoring workers together.

## Rollback procedure

1. Stop the new worker before changing mounts or state.
2. Save its SQLite/volume snapshot and logs for diagnosis; do not discard new
   append-only events.
3. Re-mount the preserved previous criteria directory and deploy the recorded
   previous image.
4. Prefer starting the previous image on the pre-upgrade SQLite snapshot. The
   schema migration is additive and the old worker ignores new columns, but a
   snapshot restore gives the fastest deterministic rollback.
5. Start exactly one worker and run one diagnostic cycle. Verify heartbeat,
   queue counters and ParkingReview TXT ingestion before restoring normal polling.
6. If any v2 TXT was already published, leave it in place: its first two lines
   are backward compatible. ParkingReview accepts v2 and legacy TXT, so no FTP
   deletion or bulk rewrite is needed for rollback. The updated ParkingReview
   importer should continue accepting both variants.

Rollback is by deploying the previous image and restoring a pre-upgrade SQLite
snapshot. Database migrations are additive; the previous worker ignores new
tables and columns. Keep the previous active `criteria.txt` unchanged and
available for remounting.
