# training

Trains the deck-matchup classifier and publishes an INT8-quantized ONNX model to S3.

## What it does

`python -m training.pipeline` runs, end to end:

1. **Download** – reads every `*.parquet` file under the S3 prefix
   `training-dataset-prefix` (a Prefect Variable) into one Polars DataFrame.
2. **Split** – shuffles (configurable) and splits into train/validation sets
   (default 80/20), writing `train.parquet` / `val.parquet` to `data.local_dir`.
3. **Train** – trains `TransformerBinaryClassifier` for `training.n_epochs`
   epochs on CUDA, MPS, or CPU (auto-detected). Deck A is always the winner, so
   the model is trained antisymmetrically against an all-ones target.
4. **Export + quantize + upload** – exports the trained model to an FP32 ONNX
   graph (dynamic batch dim), rewrites every `nn.Linear` weight to per-output-channel
   symmetric INT8 (stored as an INT8 initializer + scale behind a
   `DequantizeLinear`), and uploads the result to
   `s3://<s3-bucket-name>/<model-weights-prefix>/<output.model_filename>`.
   Embedding tables stay FP32 (a rounding error of the file size, and quantizing
   them hurts accuracy).

## Configuration

- **Training-specific** settings live in `src/training/config.yaml` and are
  validated by `src/training/config.py` (`TrainingPipelineConfig`).
- **Infrastructure** settings (S3 bucket, dataset prefix, weights prefix, region)
  are Prefect Variables set in `infra/orchestration/src/orchestration/prefect_vars.py`
  and read through helpers in `libs/common/src/common/utils.py`.

## Run

```bash
uv sync
uv run python -m training.pipeline
```

## Model format

The uploaded `.onnx` file is a standalone graph. Inputs (all with a dynamic
batch dim `N`):

| name          | dtype   | shape         |
| ------------- | ------- | ------------- |
| `deck_a`      | int64   | `(N, set_size)` |
| `deck_b`      | int64   | `(N, set_size)` |
| `deck_a_lvls` | float32 | `(N, set_size)` — card levels scaled to `[0, 1]` |
| `deck_b_lvls` | float32 | `(N, set_size)` |

Output `logit` is `(N, 1)`: the pre-sigmoid logit that deck A beats deck B.

The graph carries `model_config` (JSON) and `vocab_size` in its
`metadata_props`. Run it with any ONNX runtime, e.g.:

```python
import onnxruntime as ort
sess = ort.InferenceSession("model_int8.onnx")
logit = sess.run(None, {"deck_a": a, "deck_b": b,
                        "deck_a_lvls": la, "deck_b_lvls": lb})[0]
```
