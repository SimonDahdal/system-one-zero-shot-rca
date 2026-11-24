# Context-enriched Choice probability experiment

Models receive an application description, candidate service roles, and actual
baseline/incident metric summaries for each incident. Top-1 and ranked Top-3
come from a single Choice probability distribution. No native Score request,
triplet selection, or sequential-choice fallback is used.

## Run all models

From the repository root, with the existing dataset and environment:

```sh
conda activate RCAEval_venv
PYTHONDONTWRITEBYTECODE=1 python -m Context_enriched_experiment.run \
  --models all \
  --output-dir "Context_enriched_experiment/runs/live_$(date +%Y%m%d_%H%M%S)"
```

Directory and module names are case-sensitive. RF runs first locally. Before EACH
remote model, the runner names the configured endpoint, asks you to deploy the
model, and waits for Enter before connecting. Servers are not deployed
by this program. EOF stops before connecting; Ctrl+C retains checkpoints.
Server addresses, model names and timeouts come from `rca_experiments/config.py`.

For selected models and a smaller input:

```sh
PYTHONDONTWRITEBYTECODE=1 python -m Context_enriched_experiment.run \
  --dataset artifacts/re2_dataset.joblib --models kev laya nimble semif \
  --metrics-per-service 4 --max-input-chars 12000 \
  --output-dir "Context_enriched_experiment/runs/selected_$(date +%Y%m%d_%H%M%S)"
```

## Rerun H2O Lightning separately

From the repository root:

```sh
conda activate RCAEval_venv
PYTHONDONTWRITEBYTECODE=1 python -m Context_enriched_experiment.rerun_h2o
```

This reruns **all 135 test incidents**, not just the previously failed ones,
using the dataset and prompt budgets from `runs/live_20261109_113850`.
It verifies the dataset and preparation-sidecar hashes before evaluation.
Deploy H2O Lightning (including its vLLM backend), then press Enter when prompted.
The endpoint and model settings come from the current `rca_experiments/config.py`.
The connectivity check only checks the API port; ensure its vLLM backend is ready too.

Outputs go to a new `runs/h2o_rerun_<timestamp>` directory, with the standard
`summary.json`, `comparison.csv`, `results.json`, configuration and H2O prediction
JSONL, plus `rerun_provenance.json`. Original results and the report are not modified.
Use the new summary/CSV to update H2O's reported results separately.
Optional flags: `--output-dir PATH`, `--source-run PATH`, `--dataset PATH` (an
identical relocated artifact with its sidecar), and `--mock` for an offline check.
Current prompt/adapter code is used and its hashes are recorded in the new configuration.

## Exactly what the model sees

1. Instructions to identify an originating fault, compare baseline/incident
   behavior, and distinguish downstream symptoms.
2. Application description and roles from the existing `SYSTEM_CONTEXT` entries
   for Online Boutique, Sock Shop, or Train Ticket. The separate causal/topology
   paragraphs are excluded. Role descriptions can mention service responsibilities
   and dependencies in prose.
3. Every candidate service, in the saved inventory order.
4. A table with service, metric, baseline mean, incident mean, delta, and z-shift.
5. One structured Choice question with individual service names as options.

Unlike the prompt-only baseline, the metric table changes with the incident.
No labels, fault types, case paths/IDs, logs, traces, RF predictions, or explicit
topology are included. Application context is static and independent of labels.

## Example prompt sent to a model

This complete example uses Sock Shop and `--metrics-per-service 2` to keep
it short: CPU and memory are shown for every candidate. The default of eight
includes the other six metric types too. **All metric values below are synthetic
illustrations, not measured results.** No units or ground-truth fault are assigned.
The text and structured question are generated using the actual prompt builders.

The request's **state** is:

```text
Identify the service most likely responsible for this incident.
Compare baseline and incident metrics; distinguish a likely originating fault from propagated symptoms.
Choose only from the candidate services. Missing measurements are not evidence of normal behavior.

Application description:
System: Sock Shop, an e-commerce microservice application.

Candidate services and roles:
- carts: owns each user's shopping cart and cart items.
- catalogue: owns sock/product descriptions, inventory-facing catalogue data, search, and product lookup.
- orders: creates and retrieves orders and coordinates the order workflow.
- payment: authorizes payment requests made during order placement.
- user: owns customer accounts, login information, and addresses/cards.

Baseline/incident metric summaries:
Baseline: [-150, 0) seconds; incident: [0, 150) seconds relative to benchmark injection time (provided onset, not detected onset).
Values retain dataset units; units are not recorded in the saved features. Do not assume percent, milliseconds, or rates.
Baseline mean = saved incident mean minus saved delta. Z-shift uses baseline sample standard deviation.
Metric coverage: 2/8 fixed metric types per candidate; noncandidate services omitted.
service | metric | baseline_mean | incident_mean | delta | z_shift
carts | cpu | 20 | 22 | 2 | 1
carts | mem | 100 | 102 | 2 | 0.2
catalogue | cpu | 25 | 26 | 1 | 0.5
catalogue | mem | 120 | 121 | 1 | 0.1
orders | cpu | 30 | 34 | 4 | 2
orders | mem | 150 | 155 | 5 | 0.5
payment | cpu | 28 | 97 | 69 | 12
payment | mem | 110 | 145 | 35 | 3.5
user | cpu | 18 | 19 | 1 | 0.5
user | mem | 90 | 91 | 1 | 0.1

Assess every candidate using its role and the observed metric changes.
Large changes alone do not establish causality.
```

The accompanying **Choice question** is:

```json
{
  "root_cause": {
    "type": "choice",
    "instructions": "Select the single service most likely to be faulty.",
    "criteria": {
      "carts": "carts",
      "catalogue": "catalogue",
      "orders": "orders",
      "payment": "payment",
      "user": "user"
    }
  }
}
```

Adapters encode this state and question in each server's API format. The model's
returned per-candidate probabilities are sorted to obtain Top-1 and ranked Top-3;
there is no separate Top-3 question. Application descriptions and roles stay fixed,
while the metric table is populated from each incident's saved features. Actual
requests are saved in each prediction's `prompts` field in the run outputs.

## Metric source and limitations

The existing artifact already stores features calculated by
`rca_experiments.features.extract_metric_features`. We reuse those saved features:

- Incident mean: `<service>_<metric>_mean`.
- Delta: incident mean minus baseline mean, stored as `_delta`.
- Baseline mean: reconstructed as incident mean minus delta.
- Z-shift: the saved change divided by baseline sample standard deviation.
  It is missing when baseline variance is unavailable or too small.

This is a mean-based summary, not raw time series. Six significant digits are
shown. Saved metrics retain their original numeric units; the artifact does not
record units, so the prompt explicitly avoids assuming percentages or milliseconds.
`latency-90` and `latency-50` are names of saved series; their window means are
reported, not a newly calculated percentile across the window.

The supplied artifact's preparation sidecar records 150-second windows:
`[-150, 0)` before injection and `[0, 150)` at/after injection. Onset is supplied
by benchmark metadata, not detected by a model. The runner reads duration from
`<dataset>.split.json`, records its hash, and does not infer duration from current
configuration. Missing sidecar/duration is explicitly marked unknown. Keep the
sidecar paired with its artifact; the artifact itself does not embed window
provenance. A null saved window means all available pre/post samples were used,
with exact duration unavailable here. No raw data is reread or dataset regenerated.

Metric types have a fixed priority order:
`cpu`, `mem`, `latency-90`, `error`, `workload`, `diskio`, `socket`, `latency-50`.
The default is all eight per candidate. `--metrics-per-service` keeps the first
N for EVERY candidate (1–8), independent of ground truth or observed anomalies.
Only exact `<candidate>_<metric>` keys are read; database and other noncandidate
services are not silently merged with candidates. Nonfinite or absent values
are marked `missing`; they are never replaced with zero. Coverage is stated in
the prompt. This limited candidate-service view can omit useful external symptoms.

`--max-input-chars` limits the canonical state plus question JSON (default 20,000).
It is not a token limit or exact wire size. Exceeding it fails before connection
for that inference request; no partial lines or candidates are silently dropped.
There are no experiment-level retries or context reductions. Shared client/SDK
internal retries may be unobservable. Reduce metrics-per-service explicitly for
servers with small contexts, and use the same setting for comparisons.

## Ranking, evaluation, and outputs

The exact saved training/test split and candidate inventory are reused. RF trains
only on saved training rows and uses the original full metric feature matrix.
Remote models receive the selected summarized metrics and static context, so
information access still differs from RF. Candidates remain the original
benchmark-wide label inventory, not an independently discovered deployment list.

All candidates must have finite probabilities in [0,1], summing to approximately
one (tolerance 0.02). Sort descending; ties use saved candidate order. Top-1 is
first, Top-3 is the first three individual labels in that same ranking. Missing
or malformed probabilities fail explicitly, even if a selected choice is present.
No fallback or fabricated ranking is used.

The existing reporting pipeline writes:

- `configuration.json`: artifact hash, exact train/test indices, incident IDs,
  candidates, context and metric policy, sidecar provenance, settings/source hashes.
- `results.json`: complete checkpointed configuration, per-model status, per-incident
  predictions and diagnostics, and summaries.
- `<model>_predictions.jsonl`: correctness, latency, errors, probabilities, actual
  prompt content/size, ties, request counts, and available usage/retry measurements.
- `summary.json`: Top-1 accuracy, Top-3 hit rate, macro F1, mean/median latency,
  failure rates, method counts and fallback rates, overall and per application.
- `comparison.csv`: flat overall/per-application table for completed models.

Missing predictions count as incorrect. Valid wrong predictions are classification
errors, not request failures. Request failure rate follows the existing definition:
remote incidents without valid complete predictions, including setup/preflight and
response-validation failures, rather than HTTP failures alone. RF has no requests;
its errors count under execution failures. Unavailable measurements are explicit.
Latency includes prompt construction and inference, excludes model setup, manual
deployment and result writing. Existing output directories are refused.

## Reuse and verification

Evaluation and probability ranking are included in this package as `evaluation.py`
and `ranking.py`. Dataset/model adapters and RF come from `rca_experiments`.

```sh
python -m Context_enriched_experiment.run --models rf kev nimble semif --mock \
  --output-dir results/context_mock
```

Prepare the dataset before running. Mocks simulate remote
probabilities; RF remains real. Mock accuracy is not model quality evidence.
