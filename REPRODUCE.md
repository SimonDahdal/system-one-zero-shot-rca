# Reproducing the RCA experiments

This directory contains the standalone context-enriched experiment, the RF
baseline, and all remote-model adapters.
Copy or publish this directory on its own, or use it inside the parent repository.
The original working files are unchanged.

## Install and prepare data

Run all commands from this directory (`cd autonomus` when using the parent repo).

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python prepare_dataset.py
```

Preparation downloads RCAEval RE2 from `phamquiluan/RCAEval` on Hugging Face,
extracts features, and writes `artifacts/re2_dataset.joblib` and its `.split.json`
sidecar. To reuse a local download, pass `--data-dir /path/to/data --skip-download`.
The standalone `download_RCAEval.py` also downloads RE2 (the original helper
selected RE1). Dataset download requires network access and sufficient disk space.
Raw benchmark data, environments, cached bytecode, generated outputs, and local
credentials are not bundled.

Defaults are seed 42, test fraction 0.5, and a 150-second metric window.
[reference/re2_dataset.split.json](reference/re2_dataset.split.json) contains the
saved original split. After preparation, verify it before comparing results:

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

Prepared artifacts store absolute raw-data paths. Prepare them on the machine
where experiments will run, and retain the raw data at those paths. Review any
skipped incidents during preparation. A changed upstream dataset or dependency
can change the split; do not present such a run as an exact reproduction.

## Run the experiments

Start with the local CPU baseline:

```sh
python -m Context_enriched_experiment.run --models rf --output-dir results/context_rf
```

For remote evaluation, deploy the corresponding servers separately and set their
URLs in `rca_experiments/config.py`. This copy defaults to `127.0.0.1`, preserving
original ports and API paths. The model-server implementations, weights, and GPU
environments are not present in the source repository and are not bundled here.
`model_sequence.py` and `comands_to_run_models.txt` are deployment helpers/examples
that require those external server directories; they are not installers.

```sh
python -m Context_enriched_experiment.run --models all --output-dir results/context_all
```

The runners ask for deployment confirmation before each remote model. Use fresh
output directories. All models reuse the same prepared dataset and split.
See `Context_enriched_experiment/README.md` for probability validation, budgets, failures and output
semantics. `--mock` exercises the pipeline with synthetic responses and must not
be reported as measured model performance.

The historical `Context_enriched_experiment.rerun_h2o` helper needs a previous
run supplied with `--source-run`; its default historical run is not bundled.
Use the normal context runner for a fresh H2O evaluation.

## Reproducibility limits

Pinned direct dependencies are in `requirements.txt`;
transitive dependencies, upstream dataset revision and server/weight revisions
are not fully locked. Exact numerical reproduction of the paper requires the
original server versions and run settings as well as the reference split.

## Release contents and provenance

`SOURCE_MANIFEST.json` records original and copied SHA-256 hashes, including which
files were adjusted for portability. Changes are private server addresses to
localhost, the RE2 download helper, deployment-example cleanup, and README entry
instructions. Shared evaluation and ranking modules now live inside the
context-enriched package. Experiment algorithms and current local edits were preserved.
Historical prediction dumps are not bundled.

No software license was present in the source repository. The author should
choose a license before distributing this as reusable open-source software;
third-party data and model weights retain their respective terms.

