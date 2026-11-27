# Context-enriched RCA experiments

Standalone reproduction code for root-cause analysis (RCA) on **RCAEval RE2**,
comparing a local Random Forest (RF) baseline with remote decision models across
Online Boutique (`RE2-OB`), Sock Shop (`RE2-SS`), and Train Ticket (`RE2-TT`).
The main experiment gives remote models application descriptions, service roles,
and baseline/incident metric summaries, then ranks candidate faulty services
from a single Choice probability distribution.

Run all commands from this directory. Python module names and model identifiers
are case-sensitive.

## Navigation

- [Quick start](#quick-start): install, prepare data, and run locally.
- [Project map](#project-map): find the implementation and supporting documents.
- [Data and reproducibility](#data-and-reproducibility): understand artifacts and verify the split.
- [Remote models and configuration](#remote-models-and-configuration): endpoints and execution.
- [Results](#results): locate predictions, reports, and provenance.
- [Other workflows](#other-workflows): telemetry runner, server launcher, and H2O reruns.
- [Troubleshooting](#troubleshooting): common setup and execution issues.

For detailed reproduction instructions, see [REPRODUCE.md](REPRODUCE.md).
For prompt examples, ranking rules, and evaluation semantics, see the
[experiment guide](Context_enriched_experiment/README.md).
For raw-data schemas and feature definitions, see [dataset_readme.md](dataset_readme.md).

## Quick start

Use Python 3.11 or newer. The experiment client and RF baseline do not require a
local GPU. Dataset download requires internet access and sufficient disk space;
remote model servers and their GPU environments are installed separately.
Direct dependencies are pinned in [requirements.txt](requirements.txt).

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

# Download RE2, extract features, and save a shared train/test split.
python prepare_dataset.py

# Run the real local RF baseline without a model server.
python -m Context_enriched_experiment.run \
  --models rf --output-dir results/context_rf
```

Choose a **new output directory for every context-enriched run**; existing
output directories are refused. To check the remote evaluation pipeline without
deploying a server, use mock responses:

```sh
python -m Context_enriched_experiment.run \
  --models rf kev nimble --mock --output-dir results/context_mock
```

Mocks simulate remote probabilities; RF remains real. Mock accuracy is not
measured remote-model performance.

## Project map

| Path | Purpose |
| --- | --- |
| [Context_enriched_experiment/run.py](Context_enriched_experiment/run.py) | Main CLI; connects saved data, context/metric prompts, and probability ranking. |
| [Context_enriched_experiment/prompts.py](Context_enriched_experiment/prompts.py) | Builds application descriptions, service roles, and per-incident metric tables. |
| [Context_enriched_experiment/ranking.py](Context_enriched_experiment/ranking.py) | Remote transport, probability validation, and service ranking. |
| [Context_enriched_experiment/evaluation.py](Context_enriched_experiment/evaluation.py) | Model execution, checkpoints, evaluation, and report generation. |
| [Context_enriched_experiment/rerun_h2o.py](Context_enriched_experiment/rerun_h2o.py) | Reruns H2O using a previous run's dataset and prompt budgets. |
| [rca_experiments/config.py](rca_experiments/config.py) | Dataset/split defaults, RF settings, endpoints, timeouts, telemetry budgets, and application context. |
| [rca_experiments/dataset.py](rca_experiments/dataset.py) | Downloads/loads incidents and saves the feature matrix, metadata, candidates, and split. |
| [rca_experiments/features.py](rca_experiments/features.py) | Extracts metric features around the supplied injection time. |
| [rca_experiments/models/](rca_experiments/models/) | Local RF implementation and remote adapters; `__init__.py` registers model identifiers. |
| [rca_experiments/telemetry.py](rca_experiments/telemetry.py) | Log, metric, and trace summaries used by the separate telemetry workflow. |
| [rca_experiments/runner.py](rca_experiments/runner.py) | Separate telemetry-based comparison runner. |
| [prepare_dataset.py](prepare_dataset.py) | Prepares reusable data without running models. |
| [download_RCAEval.py](download_RCAEval.py) | Downloads RE2 only; does not prepare the feature artifact. |
| [compare_systemone_models_re2.py](compare_systemone_models_re2.py) | CLI entry point for the telemetry-based runner. |
| [model_sequence.py](model_sequence.py) | Interactive Linux/macOS launcher for externally installed model servers. |
| [comands_to_run_models.txt](comands_to_run_models.txt) | Historical deployment command examples (filename spelling retained). |
| [reference/re2_dataset.split.json](reference/re2_dataset.split.json) | Original split and preparation settings for verification. |
| [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json) | Release-copy provenance and original/copied source hashes. |

Generated directories are excluded from version control:

```text
data/                           Raw RE2 download, including cases.parquet
artifacts/
  re2_dataset.joblib             Prepared features, labels, metadata, and split
  re2_dataset.split.json         Preparation settings and train/test case IDs
results/<run-name>/              Evaluation outputs
Context_enriched_experiment/runs/  Alternative location for run outputs
```

To change model inputs, start with `prompts.py`; to change ranking or validation,
use `ranking.py`; to change reports, use `evaluation.py`. Remote adapter classes
and their registry live under `rca_experiments/models/`.

## Data and reproducibility

Preparation downloads RE2 from the Hugging Face dataset `phamquiluan/RCAEval`.
Defaults are a stratified 50/50 train/test split, random seed **42**, and
**150-second** metric windows before and after benchmark injection time.
Change preparation defaults in `rca_experiments/config.py` before regenerating data.

To reuse an existing raw download:

```sh
python prepare_dataset.py --data-dir /path/to/data --skip-download \
  --output artifacts/re2_dataset.joblib
```

The data directory must contain `cases.parquet` and its referenced incident
folders. Preparation prints skipped incidents; review these before comparing
results. Keep the `.joblib` artifact paired with its `.split.json` sidecar.
The context runner reads window provenance from the sidecar, not current config.
Artifacts contain absolute raw-data paths; retain raw data at those paths for
workflows that read logs/traces. Only load trusted joblib artifacts.

Verify a newly prepared split against the saved reference:

```sh
python - <<'CHECK'
import json
from pathlib import Path
expected = json.loads(Path('reference/re2_dataset.split.json').read_text())
actual = json.loads(Path('artifacts/re2_dataset.split.json').read_text())
assert actual == expected, 'Split or preparation settings differ from the reference'
print('Reference split and preparation settings match')
CHECK
```

The main experiment reuses saved features and split indices. RF trains on the
training rows with the full feature matrix. Remote models receive static context
and selected metric summaries for test incidents, excluding labels, case paths,
logs, traces, and explicit topology. Candidate inventories come from benchmark
labels across the dataset. Information access therefore differs between RF and
remote models; this is not a comparison of identical input representations.

Direct dependencies are pinned, but transitive dependencies, upstream data, and
server/weight revisions are not fully locked. Matching the split alone does not
guarantee exact numerical reproduction. See [REPRODUCE.md](REPRODUCE.md) for details.

## Remote models and configuration

Deploy model servers separately, then edit their URLs, model names, and timeouts
in [rca_experiments/config.py](rca_experiments/config.py). Server implementations,
weights, and GPU environments are external to this repository. Defaults point
to localhost:

| CLI model identifier | Default endpoint |
| --- | --- |
| `rf` | Local CPU; no endpoint |
| `kev` | `http://127.0.0.1:8009` |
| `laya` | `http://127.0.0.1:8010/v1/systemone` |
| `nimble` | `http://127.0.0.1:8011` |
| `semif` | `http://127.0.0.1:8012` |
| `rizzo_flow` | `http://127.0.0.1:8013` |
| `von` | `http://127.0.0.1:8014` |
| `nanojev` | `http://127.0.0.1:8015` |
| `LiquidAI_d1_3B` | `http://127.0.0.1:8016` |
| `decider` | `http://127.0.0.1:8018` |
| `h2o_lightning_4B` | `http://127.0.0.1:8019` |
| `neohorse` | `http://127.0.0.1:8020` |
| `d3_flash_server` | `http://127.0.0.1:8021` |

```sh
# Selected models, using the same prepared artifact.
python -m Context_enriched_experiment.run \
  --models rf kev laya --output-dir results/context_selected

# All registered models, with RF first.
python -m Context_enriched_experiment.run \
  --models all --output-dir results/context_all
```

Before each real remote model, the main runner displays the endpoint and waits
for Enter so you can deploy it. Use an interactive terminal. It does not deploy
servers automatically. EOF stops before connecting; Ctrl+C retains checkpoints.

Main runner options:

| Option | Meaning / default |
| --- | --- |
| `--dataset PATH` | Prepared artifact; defaults to `artifacts/re2_dataset.joblib`. |
| `--models NAME ...` | Selected identifiers, or `all` alone; defaults to `all`. |
| `--output-dir PATH` | Required new directory. |
| `--mock` | Simulate remote responses without contacting servers. |
| `--metrics-per-service N` | First N fixed metric types for every candidate, from 1 to 8; default 8. |
| `--max-input-chars N` | Canonical state plus question JSON character budget; default 20,000. |

Metric priority is `cpu`, `mem`, `latency-90`, `error`, `workload`, `diskio`,
`socket`, `latency-50`. Missing values remain explicit. The character budget is
not a token limit; exceeding it fails the inference request without silently
truncating candidates. Reduce the metric count explicitly for smaller contexts,
and use consistent settings when comparing models.

```sh
python -m Context_enriched_experiment.run --help
python prepare_dataset.py --help
```

## Results

For the main context-enriched runner, inspect `comparison.csv` or `summary.json`
first, then individual predictions for diagnostics:

| File in the output directory | Contents |
| --- | --- |
| `comparison.csv` | Flat overall and per-application comparison for completed models. |
| `summary.json` | Top-1 accuracy, Top-3 hit rate, macro F1, latency, and failure statistics. |
| `<model>_predictions.jsonl` | Per-incident predictions, probabilities, correctness, actual prompts, timing, and errors. |
| `results.json` | Checkpointed configuration, model status, predictions, and summaries. |
| `configuration.json` | Dataset/source hashes, exact split, candidates, prompt policy, settings, and sidecar provenance. |

Top-1 and Top-3 come from the same descending probability ranking, with ties
resolved in saved candidate order. Missing/malformed probabilities fail explicitly;
missing predictions count as incorrect. A valid but wrong prediction is not a
request failure. See the [experiment guide](Context_enriched_experiment/README.md)
for precise failure and timing definitions. Checkpoints preserve completed work;
the main CLI does not provide a resume option.

## Other workflows

### Telemetry-based comparison

`compare_systemone_models_re2.py` invokes `rca_experiments.runner`, which uses the
shared adapters and telemetry summaries. Its prompts and reporting differ from
the main context-enriched experiment; keep the outputs in separate directories.

```sh
python compare_systemone_models_re2.py --models rf \
  --dataset artifacts/re2_dataset.joblib --output-dir results/telemetry_rf
python compare_systemone_models_re2.py --help
```

If the artifact is missing, this runner prepares it automatically;
`--skip-download` then requires existing raw data. It supports
`--pause-between-models` / `--no-pause-between-models` and defaults to `rf kev`.
Outputs include per-model prediction CSVs, `comparison.csv`, `summary.json`, and
`rca_re2_random_forest.joblib` when RF is run.

### External server launcher

[model_sequence.py](model_sequence.py) starts external servers one at a time,
logs their output, and stops the current process group at the manual handoff.
It requires Linux/macOS, an interactive terminal, and each expected server
folder with its own `.venv`. It is not an installer or an evaluation runner.
Run experiments from another terminal after checking server readiness.

```sh
python model_sequence.py --base-dir /path/to/model-server-folders
python model_sequence.py --help
```

The launcher sequence does not include KEV; its deployment example is in
[comands_to_run_models.txt](comands_to_run_models.txt).

### H2O rerun from a previous experiment

The historical default source run is not bundled. Supply an existing run:

```sh
python -m Context_enriched_experiment.rerun_h2o \
  --source-run results/context_all --output-dir results/h2o_rerun
```

The helper verifies dataset/sidecar hashes, reuses saved prompt budgets, and
reruns the entire saved test split, including previously successful incidents.
It uses current adapter code and endpoint settings, records provenance, and does
not modify the source run. For a fresh H2O experiment, use the normal runner with
`--models h2o_lightning_4B`.

## Troubleshooting

- **Missing artifact:** run `python prepare_dataset.py` first, or pass `--dataset`
  pointing to an existing prepared artifact with its matching sidecar.
- **Output directory already exists:** choose a new name; retain previous runs
  for comparison and provenance.
- **Connection refused or timeout:** check the URL in `config.py`, deploy the
  corresponding external server, and verify its backend is ready before Enter.
- **Runner stops before connecting:** run in an interactive terminal; EOF at a
  deployment prompt stops execution. Use `--mock` for offline pipeline checks.
- **Prompt too large:** lower `--metrics-per-service` explicitly or adjust the
  character budget to the server's capacity. Record the settings used.
- **Invalid probabilities:** inspect the prediction JSONL error and server API
  response. Every candidate needs a finite probability in [0, 1], with the sum
  within 0.02 of one; the runner does not fabricate missing rankings.
- **Split mismatch or skipped incidents:** inspect preparation output, dataset
  contents, and config before claiming reproduction of the reference experiment.

Historical prediction dumps and model weights are not bundled. Source-copy
provenance is recorded in [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json); it is not
a live manifest of subsequent edits. See [REPRODUCE.md](REPRODUCE.md) for release
contents and licensing context.
