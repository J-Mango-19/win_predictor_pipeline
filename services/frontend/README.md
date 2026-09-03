# Clash Royale deck strength predictor

Client-side web app that runs the pipeline's trained ONNX model in the browser
via onnxruntime-web. Deployed to GitHub Pages by
`.github/workflows/deploy-frontend.yml`.

## How the model gets here

Nothing in `public/model/` or the generated files in `public/data/` is committed
— they are all produced by the pipeline and downloaded from S3 at build time:

| artifact | S3 key | produced by |
|---|---|---|
| INT8 ONNX model | `model-weights/model_int8.onnx` | training stage |
| model metadata | `model-weights/model_int8.metadata.json` | training stage |
| card → token id map | `frontend-assets/card_to_token_id.json` | ingestion stage |
| card image URLs | `frontend-assets/png_urls.json` | ingestion stage |

`scripts/prepare-assets.mjs` downloads all four, copies them into `public/`
under content-addressed names (`model-3f9a1c2b.onnx`), and writes
`public/model-manifest.json`. `src/App.tsx` reads that manifest first and gets
every other path from it, so a redeploy always serves a fresh URL and the model
can never be paired with a stale card map.

## Local development

```bash
npm ci
npm run fetch-assets   # needs AWS credentials, see below
npm run dev
```

`npm run dev` works without the assets — the UI renders and the error banner
tells you to run `fetch-assets`. Only prediction is disabled, so layout and
styling work needs no AWS access.

Re-stage from the local cache without re-downloading:
`npm run prepare-assets:offline`.

## AWS credentials

`fetch-assets` shells out to the AWS CLI. Any profile that can read the four
keys above works. CI uses a dedicated IAM user whose only permission is
`s3:GetObject` on exactly those keys — `cr-games-bucket` is not Terraform-managed,
so that policy is created by hand:

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": "s3:GetObject",
    "Resource": [
      "arn:aws:s3:::cr-games-bucket/model-weights/model_int8.onnx",
      "arn:aws:s3:::cr-games-bucket/model-weights/model_int8.metadata.json",
      "arn:aws:s3:::cr-games-bucket/frontend-assets/card_to_token_id.json",
      "arn:aws:s3:::cr-games-bucket/frontend-assets/png_urls.json"
    ]
  }]
}
```

Deliberately no `s3:ListBucket` and no wildcards: the same bucket holds the
Postgres dump, which contains player tags.

## Deployment

Triggered three ways, all landing on the same workflow:

- `repository_dispatch` (`model-updated`) — POSTed by the Prefect flow's final
  stage after a training run publishes new weights.
- `push` to `main` touching `services/frontend/**`.
- `workflow_dispatch` — the manual button, and the escape hatch when the Prefect
  process (which runs on a laptop) never reaches its final stage.

## Gotchas

- `ort.env.wasm.numThreads = 1` in `src/main.tsx` **must stay**. GitHub Pages
  cannot set the COOP/COEP headers that `SharedArrayBuffer` requires, so
  multithreaded onnxruntime-web fails to initialize.
- Graph I/O names (`deck_a`, `deck_b`, `deck_a_lvls`, `deck_b_lvls` → `logit`)
  are defined in `services/training/src/training/quantize.py`. They are mirrored
  as constants at the top of `src/App.tsx`; changing one means changing both.
- Card levels are normalized by `MAX_CARD_LEVEL = 16`, matching
  `services/training/src/training/data.py`.
- `reference/` holds pre-pipeline snapshots used only to validate the ingestion
  exporters by diff. They are never loaded by the app.
