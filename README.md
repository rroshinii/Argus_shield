# ARGUS Shield

ARGUS Shield is a defensive boundary around an ARGUS image task. It follows a simple operating frame: **detect, recover, abstain safely**.

1. Detect suspicious image regions with cheap, interpretable anomaly heuristics.
2. Recover by masking or inpainting suspicious regions and comparing predictions across image views.
3. Abstain safely when confidence, consistency, or anomaly thresholds do not support a trustworthy answer.

The `attacks/` package is evaluation-only. It generates controlled stress cases for measuring the defense and is never called by the production API path.

## Install

This project targets Python 3.11.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
make install
```

`requirements-lock.txt` records the environment used when this checkout was frozen. For a clean reproduction, install Python 3.11 first and then use the project install or the lock file as appropriate.

## Run Evaluation

The mock adapter is the default when no adapter configuration selects HTTP ARGUS. Prepare a manifest and clean image root, then run the complete raw-versus-shield harness:

```powershell
make run-eval MANIFEST=data/manifests/eval.jsonl IMAGE_ROOT=data/clean OUTPUT_DIR=eval/results
```

This writes row-level JSONL results, a summary CSV, an optional Parquet summary, partition metadata, and anomaly heatmap overlays. Generate a dashboard report from the validation partition with:

```powershell
python -m eval.report --results eval/results/results.jsonl --image-root data/clean --partition validation --output dashboard/data/report.json
```

## Run API

Set an API key for prediction and config requests, then start the FastAPI boundary:

```powershell
$env:SHIELD_API_KEY = "use-a-local-secret"
make run-api
```

The service exposes `GET /shield/health`, `GET /shield/config`, and `POST /shield/detect` (`/shield/predict` remains a compatibility alias). Health is public for liveness; detection and config require the `X-Shield-Key` header. The service can use `HttpArgusAdapter` internally when its adapter configuration selects an external ARGUS endpoint.

## Contract Assumptions

**These are assumptions, not the confirmed ARGUS task contract. Replace them when the real contract is available.** The current contract is documented in [docs/task_contract.md](docs/task_contract.md):

- `latency_budget_ms`: 250 ms.
- `false_positive_rate_budget`: 5% on clean images.
- `task_type`, `input_format`, `output_schema`, `access_level`, `query_budget`, `abstention_allowed`, and `submission_format` are still unconfirmed placeholders.
- The current adapter treats images as NumPy arrays and normalizes predictions to label, score, boxes, raw output, latency, and error.
- The default policy uses thresholds from [configs/defense_thresholds.yaml](configs/defense_thresholds.yaml); it does not tune thresholds at inference time.

## Latest Evaluation Results

No `eval/results/` artifact is checked into this checkout, so there is no genuine latest `eval/report.py` result table to reproduce here. The expected report table is produced from the validation JSON report and contains these measurements:

| Measurement | Raw adapter | Shield | Current checkout |
|---|---:|---:|---|
| Clean accuracy | not run | not run | Generate with `make run-eval` |
| Attacked accuracy by family | not run | not run | Generate with `make run-eval` |
| Abstention rate | n/a | not run | Generate with `eval.report` |
| Clean false-positive rate | n/a | not run | Generate with `eval.report` |
| Latency p50 / p95 | not run | not run | Generate with `make run-eval` |

## Known Limitations

- No evaluation result artifact or real ARGUS model access is present in this repository yet.
- Pixel perturbation, synthetic patch, naturalistic patch, and geometric/photometric stress transforms have generators and harness support, but their metrics are not measured until a manifest and image set are supplied.
- Semantic manipulation is not implemented or measured. It remains future work because it requires task-specific content understanding and a confirmed ARGUS contract.
- The anomaly detector is heuristic and can miss low-frequency, highly natural, or task-semantic manipulations.
- Full ensemble mode reruns ARGUS across multiple preprocessed views and a recovered image. It improves consistency evidence at the cost of additional model calls and latency; the two-stage gate keeps that cost off ordinary low-anomaly requests.
- The HTTP adapter is intentionally black-box. It does not assume gradients, logits, or embeddings unless a future contract explicitly grants that access.
- Threshold tuning must use validation data only. The report layer refuses the test partition unless explicitly invoked for the final run.
