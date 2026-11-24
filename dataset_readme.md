# Dataset processing guide

This document describes how this repository turns RCAEval RE2 telemetry into a
shared experiment dataset and model inputs. For environment setup, model servers,
and experiment commands, see [README.md](README.md).

There are two processing stages:

1. **Preparation:** load incidents, extract metric features, save labels and metadata,
   and create a reusable train/test split.
2. **Inference:** use cached metric features and read raw logs/traces to build the
   remote models' shared prompt. These text summaries are not stored in the dataset artifact.

## 1. Raw data structure

The preparation code downloads `phamquiluan/RCAEval` through
`huggingface_hub.snapshot_download`, selecting `cases.parquet` and `re2*/**`.
It uses the existing local files when `--skip-download` is supplied.

Typical local layout:

```text
data/
  cases.parquet
  re2ss_catalogue_delay_2/
    metrics.parquet
    logs.parquet
    traces.parquet
    inject_time.txt
  ...other incident directories...
```

Incident directories are resolved as `DATA_DIR / str(case["case"])`. The catalog,
not the directory name or `inject_time.txt`, supplies the application, root-cause
label, and injection time used by this pipeline.

### Incident catalog

Only these `cases.parquet` columns are required by preparation:

| Column | Use |
| --- | --- |
| `case` | Incident identifier and relative directory path |
| `dataset` | Filter and application identifier: `RE2-OB`, `RE2-SS`, or `RE2-TT` |
| `inject_time` | Numeric fault-injection boundary |
| `root_cause_service` | Ground-truth service, used to construct the label |

The local catalog also includes fault descriptions, repetition numbers, time
ranges, telemetry counts, and availability flags. Preparation does not use those
fields as features or filter cases using `has_logs`/`has_traces`.

### Telemetry schemas

| Input | Required fields after loading | Reader |
| --- | --- | --- |
| Metrics | Numeric-compatible `time`, plus metric columns | `RCAEval.utility.read_metrics` |
| Logs | `timestamp`, `container_name`, `message` | `RCAEval.utility.read_logs` |
| Traces | `traceID`, `spanID`, `parentSpanID`, `serviceName`, `duration`, and a timestamp column | Direct read of `traces.parquet` |

Trace timestamps use `startTimeMillis` when present, otherwise `startTime`.
`statusCode` is optional. Metric/log file-format handling is delegated to the
installed RCAEval readers; the trace summary specifically expects Parquet.

## 2. Prepare a shared dataset

Run from the project directory in the configured Python environment:

```sh
# Download if needed, then prepare.
python prepare_dataset.py

# Prepare from existing local data.
python prepare_dataset.py --skip-download

# Use another data directory and keep a separate artifact.
python prepare_dataset.py --data-dir /path/to/data --skip-download \
  --output artifacts/variant.joblib
```

Preparation follows this order:

1. Read the catalog and retain rows whose `dataset` is in `config.DATASETS`.
2. For each retained row, load metrics and convert `inject_time` to a float.
3. Extract pre/post metric features and add the application's indicator feature.
4. Construct a label such as `RE2-SS::catalogue` and retain incident metadata.
5. Combine the feature dictionaries into a single matrix and clean missing values.
6. Split incident positions using stratified `train_test_split`.
7. Build the per-application candidate service lists and save the artifact and manifest.

An exception while reading/extracting an incident prints `Skipping <path>: <reason>`
and omits that incident. Preparation fails if none remain. Skip reasons are printed,
not written to a dedicated audit file. Missing logs do not remove an incident here;
log availability is checked later during remote inference.

## 3. Metric feature extraction

Implementation: [features.py](rca_experiments/features.py).

Let `t0` be the injection time and `W = METRIC_WINDOW_SECONDS`. With the current
`W = 150`, the windows are:

| Window | Interval |
| --- | --- |
| Baseline | `t0 - 150 <= time < t0` |
| Post-injection | `t0 <= time < t0 + 150` |

Setting `W` to `None` uses all available rows on each side of the boundary.
Both windows must contain rows. Duplicate column names are removed, keeping the
first occurrence. Time and metric values are converted to numeric values;
unparseable values become missing. Metric timestamps are not automatically
rescaled, so their units must match the injection time.

For every metric column `m`, let `a` be its baseline samples and `b` its
post-injection samples. Five features are generated:

| Feature | Definition |
| --- | --- |
| `m_mean` | Mean of `b` |
| `m_std` | Sample standard deviation of `b` (`ddof=1`) |
| `m_max` | Maximum of `b` |
| `m_delta` | `mean(b) - mean(a)` |
| `m_zshift` | `(mean(b) - mean(a)) / std(a)` |

`m_zshift` is missing when the baseline standard deviation is missing or no
greater than `1e-8`. Pandas aggregations skip missing samples; all-missing columns
or too few samples can still produce missing features. There is no resampling,
smoothing, feature scaling, or interpolation in this extraction step.

Each incident also receives `system_<dataset> = 1`, such as `system_RE2-SS`.
After combining incidents:

- Columns are the union of all incident feature names.
- Positive/negative infinity in the matrix is replaced with `NaN`.
- Columns missing for every incident are dropped.
- Application indicator columns are filled with zero where absent.
- Other missing values remain in the saved matrix.

RF later applies median imputation fitted **only on training rows**, using
`SimpleImputer(strategy="median", keep_empty_features=True)`. Preparation itself
does not impute missing metrics.

## 4. Labels, candidates, and the shared split

Implementation: [dataset.py](rca_experiments/dataset.py).

Labels have the form `dataset::root_cause_service`. Current preparation settings
are `RANDOM_STATE = 42` and `TEST_SIZE = 0.50`; check
[config.py](rca_experiments/config.py) before generating a new artifact.

The split operates on integer row positions and is stratified by the full label.
Every model uses the persisted `train_idx` and `test_idx`, including the same test
order. This is an incident-level random split, not a chronological or grouped
split by fault type/repetition. Stratification can fail if classes have too few
incidents or the requested split is too small to represent them.

Candidate lists contain sorted unique services for each application, derived
from **all successfully prepared labels**, including test labels. They represent
the known benchmark label space. Strict zero-shot studies should supply an
independent service inventory instead.

The runner removes `label` and `root_cause_service` from incident dictionaries
passed to `predict`. Model `fit` receives the full dataset object, so supervised
adapters must explicitly use training indices. RF does so; remote adapters use
candidate lists and initialize their clients.

## 5. Resulting dataset files

Default outputs:

```text
artifacts/
  re2_dataset.joblib
  re2_dataset.split.json
```

### `re2_dataset.joblib`

Contains an `ExperimentDataset` instance:

| Field | Type / shape | Contents |
| --- | --- | --- |
| `X` | DataFrame, `N × F` | Cleaned metric features and system indicators |
| `y` | Series, length `N` | Ground-truth `system::service` labels |
| `systems` | Series, length `N` | Application identifiers |
| `incidents` | List of `N` dictionaries | Paths, metadata, and cached feature dictionaries |
| `train_idx` | NumPy integer array | Training row positions |
| `test_idx` | NumPy integer array | Test row positions in evaluation order |
| `candidates` | Dictionary of lists | Sorted service names for each application |

All incident-level fields share the same positional order. Each incident dictionary
contains `case_dir`, `case`, `system`, `injection_time`, `root_cause_service`,
`label`, and `metric_features`.

`case_dir` is recorded from an absolute data directory. `metric_features` is a
copy made **before matrix-wide cleanup**; it can contain nonfinite values or
features later dropped from `X`. Remote metric summaries validate these values
when reading them.

Raw metric/log/trace tables and rendered prompts are not embedded in the artifact.
RF can use the prepared matrix; remote models still need the raw telemetry paths.

### `re2_dataset.split.json`

Contains `random_state`, `test_size`, `datasets`, `metric_window_seconds`, and
ordered `train`/`test` lists of case identifiers. It is a readable split record,
not a replacement for the feature artifact or a complete configuration snapshot.
It does not record every prompt setting, a dataset revision, or raw-file hashes.

### Current local artifact snapshot

The artifact inspected while writing this document contains:

| Quantity | Value |
| --- | --- |
| Incidents | 270 |
| Feature columns | 2,658 |
| Classes | 15 |
| Incidents per application | 90 each |
| Training incidents | 135 |
| Test incidents | 135 |
| Saved seed / test fraction | 42 / 0.50 |
| Saved metric window | 150 seconds |

These values describe the existing local artifact, not fixed dataset guarantees.
They can change with filtering, skipped incidents, raw data, or configuration.

## 6. Log processing at inference time

Implementation: [telemetry.py](rca_experiments/telemetry.py), `prepare_logs`.

Logs are loaded, sorted, and filtered around `t0`. Invalid timestamps are dropped.
If the median timestamp is greater than `1e11` while injection time is below
`1e11`, timestamps are divided by 1,000. Missing messages become empty strings;
missing container names become `unknown`.

The pipeline flags messages using an error-pattern regular expression covering
errors, exceptions, failures, timeouts, warnings, connection problems, and similar
signals. Templates group **exact normalized messages within each container**.
Normalization lowercases text, collapses whitespace, and masks timestamps, UUIDs,
IP addresses, hexadecimal values, and numbers. IDs are local to the incident's
selected window, not a global template vocabulary.

For each template, aggregation retains its representative message, occurrence
count, error flag, first/last time, and source position. Post-injection templates
are compared with baseline counts to identify new or increased messages.

Selection prioritizes error evidence, new templates, positive count changes,
and frequency. It gives represented containers an initial slot within the overall
budget, then fills remaining slots subject to the per-container limit. Selected
messages are displayed chronologically, with times relative to injection.
Nearby preceding messages from the same container may be added if they are within
five seconds and belong to another template.

The resulting text includes baseline/after statistics, representative baseline
templates, and post-injection changes. Message text is capped by character count.
Line budgets count selected templates; headers, statistics, and preceding context
can add extra output lines. Counts are raw counts, not rates normalized for the
different baseline and post-injection window durations.

Missing required columns or no post-injection logs cause prompt preparation to
fail for that incident. An empty baseline is allowed.

## 7. Metric anomaly summary at inference time

`prepare_metric_summary` uses cached `metric_features`; it does not reread metrics
or include RF predictions. For each metric with usable post-mean and delta:

```text
before_mean = after_mean - delta
percent_change = 100 × delta / max(abs(before_mean), 1e-8)
score = max(abs(z_shift), min(abs(percent_change) / 25, 20))
```

A missing/nonfinite z-shift contributes zero to the score. Nonpositive or
nonfinite scores are omitted. Metrics are ranked by score and selected within
the overall anomaly budget and per-service limit.

Service attribution uses deterministic matching against normalized candidate
names and aliases; unmatched metrics are labeled `unmapped`. Output shows the
service, metric name, before/after means, percentage change, and z-shift.

## 8. Trace summary at inference time

`prepare_trace_summary` reads `traces.parquet` and links each span to its parent
using `traceID` and `parentSpanID`. The parent's service becomes the caller and
the child service the callee. Rows without a usable timestamp/caller/callee and
same-service edges are removed.

Timestamp normalization divides by 1,000 up to twice when the median indicates
millisecond/microsecond values relative to a seconds-based injection time.
Duration values are numeric-coerced but not rescaled; latency comparisons use ratios.

If all nonmissing `statusCode` values are at most 2, value 2 marks an error.
Otherwise values of 400 or greater mark errors. Without that column, explicit
error flags are false.

Before/after aggregation per caller–callee edge calculates span count, median
duration, error count, and earliest timestamp. Ranking favors increased errors,
post-injection errors, and increased duration ratio. Up to `MAX_TRACE_EDGES`
edges are shown with counts, latency ratio, and first post-injection time.

Missing files, unreadable Parquet, or missing required trace fields produce an
informational trace summary rather than dropping the prepared incident. Trace
filtering currently uses the **global log-window constants**, not per-system log
window overrides. Train Ticket trace summaries are disabled by default.

## 9. Current telemetry budgets

Values below reflect the current configuration; `None` for a time window means
all available data on that side of injection.

| Setting | RE2-OB / RE2-SS | RE2-TT |
| --- | --- | --- |
| Metric window per side | 150 s | 150 s |
| Log baseline window | 50 s | 15 s |
| Log post-injection window | 80 s | 45 s |
| Baseline template budget | 30 | 2 |
| Post-injection template budget | 70 | 6 |
| Templates per container | 3 | 2 |
| Characters per log message | 700 | 180 |
| Containers in each statistics table | 20 | 8 |
| Metric anomalies | 5 | 2 |
| Maximum metrics per service | 5 | 5 |
| Trace summary | Enabled; up to 10 edges | Disabled |

`INCLUDE_METRIC_SUMMARY` and `INCLUDE_TRACE_SUMMARY` control global defaults;
`SYSTEM_TELEMETRY_CONFIG` supplies application-specific overrides.

The shared prompt combines benchmark context, static application architecture,
candidate services, metric anomalies, trace evidence, and log changes. It asks
for one root cause and an unordered set of up to three services. The Top-3 options
are combinations of the candidate list, not a ranked list.

Adapters with context-limit retries shorten only telemetry text after a recognized
server rejection. This does not alter the saved dataset, split, question instructions,
or candidate inventory. It can alter available evidence and therefore predictions.
See [README.md](README.md#context-limit-retries) for model-specific behavior.

## 10. Inspect and reproduce an artifact

Run this from the project directory, loading only an artifact you trust:

```python
from rca_experiments.dataset import load_dataset

dataset = load_dataset("artifacts/re2_dataset.joblib")
print("Features:", dataset.X.shape)
print("Classes:", dataset.y.nunique())
print("Train/test:", len(dataset.train_idx), len(dataset.test_idx))
print(dataset.systems.value_counts())
print(dataset.X.head())

position = int(dataset.test_idx[0])
print("First test case:", dataset.incidents[position]["case"])
print("True label:", dataset.y.iloc[position])
print("Candidates:", dataset.candidates[dataset.systems.iloc[position]])
```

Rebuild after changing dataset selection, split settings, or metric extraction.
Prompt budgets can change without rebuilding because logs/traces and text summaries
are processed during inference. Restart the Python process after editing configuration.

An existing artifact is reused even when current settings differ. Preserve the
artifact, split manifest, source/configuration revision, and raw telemetry to make
a run reproducible. There is no automatic raw-data fingerprint or configuration
compatibility check. Matrix-wide column cleanup and candidate inventory creation
occur before splitting; imputation and RF fitting occur on training rows only.

Dataset artifacts live under `artifacts/`; predictions, evaluation metrics, and
timings live under `results/`. Both directories and raw `data/` are excluded by
`.gitignore`.
