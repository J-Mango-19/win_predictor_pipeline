# Pre-pipeline reference snapshots

Captured from the original `win_predictor` repo before the pipeline began
publishing these files itself. They are **not** used at build or run time —
`scripts/prepare-assets.mjs` downloads the live versions from S3.

They exist so the new ingestion-side exporters can be validated by diff:

- `card_to_token_id.snapshot.json` — the `card_ids` table as it stood, 176
  entries. Compare against what `task_export_frontend_assets` publishes; ids
  must be **identical**, not merely similar. Any shift means the Postgres dump
  was rebuilt from scratch and every previously trained model is invalid.
- `png_urls.snapshot.json` — card art URLs. Compare against the output of the
  `/v1/cards` extractor; expect ~121 base / 41 evo / 14 hero keys.
