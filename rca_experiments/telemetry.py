"""Reusable log, metric and trace summaries."""
from pathlib import Path
from collections import Counter
import re
import numpy as np
import pandas as pd
from RCAEval.utility import read_logs
from .config import (
    AFTER_LOG_WINDOW_SECONDS,
    BEFORE_LOG_WINDOW_SECONDS,
    MAX_METRICS_PER_SERVICE,
    MAX_TRACE_EDGES,
)

ERROR_PATTERN = re.compile(
    r"error|exception|fail(?:ed|ure)?|timeout|timed out|refused|"
    r"unavailable|critical|fatal|oom|denied|reset|broken|"
    r"warn(?:ing)?|deadline_exceeded|resource_exhausted|"
    r"no route to host|name resolution|circuit breaker|"
    r"retry exhausted|connection.*closed|\b5\d\d\b",
    re.IGNORECASE,
)

LOG_TIMESTAMP_PATTERN = re.compile(
    r"\b\d{4}-\d{2}-\d{2}[t ]\d{2}:\d{2}:\d{2}"
    r"(?:[.,]\d+)?(?:z|[+-]\d{2}:?\d{2})?\b",
    re.IGNORECASE,
)
LOG_UUID_PATTERN = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-"
    r"[89ab][0-9a-f]{3}-[0-9a-f]{12}\b",
    re.IGNORECASE,
)
LOG_IP_PATTERN = re.compile(
    r"\b(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?\b"
)
LOG_HEX_PATTERN = re.compile(r"\b0x[0-9a-f]+\b", re.IGNORECASE)
LOG_NUMBER_PATTERN = re.compile(r"\b\d+(?:\.\d+)?\b")


def normalize_log_message(message):
    """Remove volatile values before comparing two log messages."""
    normalized = str(message).lower().replace("\n", " ")
    normalized = LOG_TIMESTAMP_PATTERN.sub("<timestamp>", normalized)
    normalized = LOG_UUID_PATTERN.sub("<id>", normalized)
    normalized = LOG_IP_PATTERN.sub("<address>", normalized)
    normalized = LOG_HEX_PATTERN.sub("<hex>", normalized)
    normalized = LOG_NUMBER_PATTERN.sub("<number>", normalized)
    return " ".join(normalized.split())


def assign_log_templates(logs):
    """Group exact normalized messages within each container."""
    logs = logs.copy()
    normalized = logs["message"].map(normalize_log_message).astype(str)
    template_keys = logs["container_name"].astype(str) + "\0" + normalized
    logs["template_id"] = pd.factorize(template_keys, sort=False)[0]
    return logs



def aggregate_log_templates(logs, count_name):
    """Collapse log instances while retaining frequency and timing."""
    columns = [
        "template_id", "container_name", "message", "is_error",
        count_name, "first_time", "last_time", "first_index",
    ]
    if logs.empty:
        return pd.DataFrame(columns=columns)

    table = logs.groupby("template_id", as_index=False).agg(
        container_name=("container_name", "first"),
        message=("message", "first"),
        is_error=("is_error", "max"),
        **{
            count_name: ("message", "size"),
            "first_time": ("timestamp", "min"),
            "last_time": ("timestamp", "max"),
            "first_index": ("source_index", "first"),
        },
    )
    return table


def select_balanced_templates(
    table, limit, count_column, max_per_container
):
    """Select strong evidence without allowing one service to dominate."""
    if table.empty or limit <= 0:
        return table.head(0).copy()

    ranked = table.copy()
    ranked["is_error"] = ranked["is_error"].fillna(False).astype(bool)
    ranked[count_column] = pd.to_numeric(
        ranked[count_column], errors="coerce"
    ).fillna(0)
    is_new = (
        ranked["is_new"].astype(int)
        if "is_new" in ranked else 0
    )
    positive_delta = (
        ranked["positive_delta"].clip(lower=0)
        if "positive_delta" in ranked else 0
    )
    ranked["priority"] = (
        ranked["is_error"].astype(int) * 1_000_000
        + is_new * 100_000
        + positive_delta * 100
        + ranked[count_column]
    )
    ranked = ranked.sort_values(
        ["priority", "first_time"], ascending=[False, True]
    )

    chosen = []
    chosen_set = set()
    per_container = Counter()

    # Give every represented container one slot before filling by global
    # priority. This preserves weak but potentially causal upstream evidence.
    first_per_container = ranked.groupby(
        "container_name", sort=False
    ).head(1).sort_values(
        ["priority", "first_time"], ascending=[False, True]
    )
    for idx, row in first_per_container.iterrows():
        if len(chosen) >= limit:
            break
        chosen.append(idx)
        chosen_set.add(idx)
        per_container[row["container_name"]] += 1

    for idx, row in ranked.iterrows():
        if len(chosen) >= limit:
            break
        if idx in chosen_set:
            continue
        container = row["container_name"]
        if per_container[container] >= max_per_container:
            continue
        chosen.append(idx)
        chosen_set.add(idx)
        per_container[container] += 1
        if len(chosen) >= limit:
            break

    return ranked.loc[chosen].copy()


def append_log_statistics(summary, title, logs, max_containers):
    summary.append(title)
    if logs.empty:
        summary.append("No logs available.")
        return
    counts = logs.groupby("container_name").agg(
        total=("message", "size"),
        errors=("is_error", "sum"),
        distinct_templates=("template_id", "nunique"),
    )
    counts = counts.sort_values(
        ["errors", "total"], ascending=False
    ).head(max_containers)
    summary.append(counts.to_string())


def preceding_context(logs, source_index):
    """Return a nearby preceding message from the same container."""
    positions = logs.index[logs["source_index"] == source_index].tolist()
    if not positions:
        return None
    position = positions[0]
    current = logs.loc[position]
    earlier = logs.loc[:position - 1]
    earlier = earlier[
        earlier["container_name"] == current["container_name"]
    ]
    if earlier.empty:
        return None
    previous = earlier.iloc[-1]
    if current["timestamp"] - previous["timestamp"] > 5:
        return None
    if previous["template_id"] == current["template_id"]:
        return None
    return previous


def prepare_logs(case_dir, injection_time, config):
    """
    Load logs and select messages without using
    the incident's ground-truth root cause.
    """

    logs = read_logs(str(case_dir)).copy()

    required = {"timestamp", "container_name", "message"}
    missing = required - set(logs.columns)

    if missing:
        raise ValueError(f"Missing log columns: {missing}")

    logs["timestamp"] = pd.to_numeric(
        logs["timestamp"], errors="coerce"
    )

    logs = logs.dropna(subset=["timestamp"])
    logs = logs.sort_values("timestamp").reset_index(drop=True)

    # Check timestamp units before filtering.
    # Convert milliseconds to seconds when appropriate.
    ts_median = logs["timestamp"].median()

    if (
        pd.notna(ts_median)
        and ts_median > 1e11
        and injection_time < 1e11
    ):
        logs["timestamp"] /= 1000.0

    logs["message"] = logs["message"].fillna("").astype(str)
    logs["container_name"] = (
        logs["container_name"].fillna("unknown").astype(str)
    )

    logs["is_error"] = logs["message"].str.contains(
        ERROR_PATTERN, na=False
    )
    logs["source_index"] = logs.index

    before = logs[logs["timestamp"] < injection_time].copy()
    after = logs[logs["timestamp"] >= injection_time].copy()

    before_window = config["before_window_seconds"]
    after_window = config["after_window_seconds"]
    if before_window is not None:
        before = before[
            before["timestamp"] >=
            injection_time - before_window
        ]
    if after_window is not None:
        after = after[
            after["timestamp"] <
            injection_time + after_window
        ]

    if after.empty:
        raise ValueError("No post-injection logs")

    windowed = pd.concat([before, after]).sort_values(
        "timestamp"
    ).reset_index(drop=True)
    windowed = assign_log_templates(windowed)
    before = windowed[windowed["timestamp"] < injection_time].copy()
    after = windowed[windowed["timestamp"] >= injection_time].copy()

    before_templates = aggregate_log_templates(before, "before_count")
    after_templates = aggregate_log_templates(after, "after_count")
    before_counts = before_templates.set_index("template_id")[
        "before_count"
    ] if not before_templates.empty else pd.Series(dtype=float)
    after_templates["before_count"] = (
        after_templates["template_id"].map(before_counts).fillna(0).astype(int)
    )
    after_templates["after_count"] = pd.to_numeric(
        after_templates["after_count"], errors="coerce"
    ).fillna(0).astype(int)
    after_templates["count_delta"] = (
        after_templates["after_count"] - after_templates["before_count"]
    )
    after_templates["positive_delta"] = after_templates[
        "count_delta"
    ].clip(lower=0)
    after_templates["is_new"] = after_templates["before_count"] == 0

    selected_before = select_balanced_templates(
        before_templates,
        config["before_log_lines"],
        "before_count",
        config["max_logs_per_container"],
    )
    selected_after = select_balanced_templates(
        after_templates,
        config["after_log_lines"],
        "after_count",
        config["max_logs_per_container"],
    )

    summary = [
        "LOG CHANGE ANALYSIS",
        "Negative times are baseline behavior and positive times are incident "
        "behavior. Template counts compare near-identical messages before and "
        "after injection.",
        "",
    ]

    append_log_statistics(
        summary, "BEFORE INJECTION STATISTICS", before,
        config["max_log_stat_containers"],
    )
    append_log_statistics(
        summary, "AFTER INJECTION STATISTICS", after,
        config["max_log_stat_containers"],
    )
    summary.extend(["", "REPRESENTATIVE BASELINE TEMPLATES"])

    for _, row in selected_before.sort_values("first_time").iterrows():
        message = row["message"].replace("\n", " ")[
            :config["max_log_message_chars"]
        ]
        first_relative = row["first_time"] - injection_time
        summary.append(
            f"[{row['container_name']}] count={row['before_count']}; "
            f"first={first_relative:+.3f}s; {message}"
        )

    summary.extend(["", "POST-INJECTION TEMPLATE CHANGES"])
    for _, row in selected_after.sort_values("first_time").iterrows():
        message = row["message"].replace("\n", " ")[
            :config["max_log_message_chars"]
        ]
        first_relative = row["first_time"] - injection_time
        last_relative = row["last_time"] - injection_time
        before_count = int(row["before_count"])
        after_count = int(row["after_count"])
        count_delta = int(row["count_delta"])
        change = (
            "NEW" if before_count == 0
            else "INCREASED" if count_delta > 0
            else "PERSISTING"
        )
        summary.append(
            f"[{change}] [{row['container_name']}] "
            f"before={before_count}; after={after_count}; "
            f"delta={count_delta:+d}; "
            f"first={first_relative:+.3f}s; "
            f"last={last_relative:+.3f}s; {message}"
        )
        previous = preceding_context(windowed, row["first_index"])
        if previous is not None:
            previous_message = previous["message"].replace("\n", " ")[
                :config["max_log_message_chars"]
            ]
            previous_relative = previous["timestamp"] - injection_time
            summary.append(
                f"  preceding context at {previous_relative:+.3f}s: "
                f"{previous_message}"
            )

    return "\n".join(summary).rstrip()


def service_for_metric(metric_name, candidates):
    """Best-effort mapping from a metric column name to a candidate."""
    compact_metric = re.sub(r"[^a-z0-9]", "", metric_name.lower())
    for candidate in sorted(candidates, key=len, reverse=True):
        compact_candidate = re.sub(
            r"[^a-z0-9]", "", candidate.lower()
        )
        aliases = {
            compact_candidate,
            compact_candidate.removeprefix("ts"),
            compact_candidate.replace("service", ""),
        }
        if any(
            len(alias) >= 4 and alias in compact_metric
            for alias in aliases
        ):
            return candidate
    return "unmapped"


def prepare_metric_summary(metric_features, candidates, max_anomalies):
    """Summarize cached metric features without rereading metric files."""
    anomalies = []
    for feature_name, z_shift in metric_features.items():
        if not feature_name.endswith("_zshift"):
            continue
        metric_name = feature_name[:-len("_zshift")]
        after_mean = metric_features.get(f"{metric_name}_mean", np.nan)
        delta = metric_features.get(f"{metric_name}_delta", np.nan)
        try:
            after_mean = float(after_mean)
            delta = float(delta)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(after_mean) or not np.isfinite(delta):
            continue
        try:
            z_shift = float(z_shift)
        except (TypeError, ValueError):
            z_shift = np.nan
        before_mean = after_mean - delta
        denominator = max(abs(before_mean), 1e-8)
        percent_change = 100.0 * delta / denominator
        score = max(
            abs(z_shift) if np.isfinite(z_shift) else 0,
            min(abs(percent_change) / 25.0, 20),
        )
        if score <= 0 or not np.isfinite(score):
            continue
        anomalies.append({
            "metric": metric_name,
            "service": service_for_metric(metric_name, candidates),
            "before": before_mean,
            "after": after_mean,
            "percent": percent_change,
            "z_shift": z_shift,
            "score": score,
        })

    anomalies.sort(key=lambda item: item["score"], reverse=True)
    lines = [
        "METRIC ANOMALY SUMMARY",
        "Largest metric changes after injection; this is raw telemetry and "
        "does not include the Random Forest prediction.",
    ]
    if not anomalies:
        lines.append("No usable pre/post metric anomalies.")
        return "\n".join(lines)

    selected = []
    per_service = Counter()
    for item in anomalies:
        service = item["service"]
        if per_service[service] >= MAX_METRICS_PER_SERVICE:
            continue
        selected.append(item)
        per_service[service] += 1
        if len(selected) >= max_anomalies:
            break

    for item in selected:
        z_text = (
            f"{item['z_shift']:+.2f}"
            if np.isfinite(item["z_shift"]) else "n/a"
        )
        lines.append(
            f"- service={item['service']}; metric={item['metric']}; "
            f"before_mean={item['before']:.4g}; "
            f"after_mean={item['after']:.4g}; "
            f"change={item['percent']:+.1f}%; z_shift={z_text}"
        )
    return "\n".join(lines)


def normalize_trace_time(values, injection_time):
    """Convert common millisecond/microsecond trace timestamps to seconds."""
    values = pd.to_numeric(values, errors="coerce")
    median = values.median()
    for _ in range(2):
        if (
            pd.notna(median)
            and median > 1e11
            and injection_time < 1e11
        ):
            values = values / 1000.0
            median /= 1000.0
    return values


def prepare_trace_summary(case_dir, injection_time, *,
                          before_window_seconds=BEFORE_LOG_WINDOW_SECONDS,
                          after_window_seconds=AFTER_LOG_WINDOW_SECONDS,
                          max_trace_edges=MAX_TRACE_EDGES):
    """Summarize caller-to-callee changes from traces when available."""
    trace_path = Path(case_dir) / "traces.parquet"
    if not trace_path.exists():
        return "TRACE SUMMARY\nNo trace data available for this incident."

    try:
        traces = pd.read_parquet(trace_path)
    except Exception as exc:
        return f"TRACE SUMMARY\nTrace data could not be read ({type(exc).__name__})."

    required = {
        "traceID", "spanID", "parentSpanID", "serviceName", "duration"
    }
    if not required.issubset(traces.columns):
        missing = sorted(required - set(traces.columns))
        return f"TRACE SUMMARY\nMissing trace columns: {missing}"

    time_column = (
        "startTimeMillis"
        if "startTimeMillis" in traces.columns else "startTime"
        if "startTime" in traces.columns else None
    )
    if time_column is None:
        return "TRACE SUMMARY\nNo usable trace timestamp column."

    traces = traces.copy()
    traces["timestamp"] = normalize_trace_time(
        traces[time_column], injection_time
    )
    traces["duration"] = pd.to_numeric(
        traces["duration"], errors="coerce"
    )
    if "statusCode" in traces.columns:
        status_codes = pd.to_numeric(
            traces["statusCode"], errors="coerce"
        )
        # OpenTelemetry uses 0=UNSET, 1=OK, 2=ERROR. Some exporters instead
        # store HTTP status codes, where 4xx/5xx indicate failure.
        if status_codes.dropna().max() <= 2:
            traces["error_status"] = status_codes.eq(2)
        else:
            traces["error_status"] = status_codes.ge(400)
    else:
        traces["error_status"] = False

    parent_services = traces[[
        "traceID", "spanID", "serviceName"
    ]].rename(columns={
        "spanID": "parentSpanID",
        "serviceName": "caller",
    })
    spans = traces.merge(
        parent_services,
        on=["traceID", "parentSpanID"],
        how="left",
    ).rename(columns={"serviceName": "callee"})
    spans = spans.dropna(subset=["timestamp", "caller", "callee"])
    spans = spans[spans["caller"] != spans["callee"]]

    before = spans[spans["timestamp"] < injection_time]
    after = spans[spans["timestamp"] >= injection_time]
    if before_window_seconds is not None:
        before = before[
            before["timestamp"] >=
            injection_time - before_window_seconds
        ]
    if after_window_seconds is not None:
        after = after[
            after["timestamp"] <
            injection_time + after_window_seconds
        ]

    def aggregate_edges(frame, prefix):
        if frame.empty:
            return pd.DataFrame(columns=[
                "caller", "callee", f"{prefix}_spans",
                f"{prefix}_duration", f"{prefix}_errors",
                f"{prefix}_first",
            ])
        return frame.groupby(["caller", "callee"], as_index=False).agg(
            **{
                f"{prefix}_spans": ("spanID", "size"),
                f"{prefix}_duration": ("duration", "median"),
                f"{prefix}_errors": ("error_status", "sum"),
                f"{prefix}_first": ("timestamp", "min"),
            }
        )

    before_edges = aggregate_edges(before, "before")
    after_edges = aggregate_edges(after, "after")
    edges = before_edges.merge(
        after_edges, on=["caller", "callee"], how="outer"
    )
    if edges.empty:
        return "TRACE SUMMARY\nNo cross-service spans in the selected window."

    for column in [
        "before_spans", "after_spans", "before_errors", "after_errors"
    ]:
        edges[column] = edges[column].fillna(0)
    edges["status_delta"] = (
        edges["after_errors"] - edges["before_errors"]
    )
    edges["duration_ratio"] = (
        edges["after_duration"]
        / edges["before_duration"].replace(0, np.nan)
    )
    edges["priority"] = (
        edges["status_delta"].clip(lower=0) * 1000
        + edges["after_errors"] * 100
        + edges["duration_ratio"].replace(
            [np.inf, -np.inf], np.nan
        ).fillna(1).clip(upper=100)
    )
    edges = edges.sort_values("priority", ascending=False).head(
        max_trace_edges
    )

    lines = [
        "TRACE SUMMARY",
        "Caller -> callee changes, including explicit error-status spans and "
        "latency changes.",
    ]
    for _, edge in edges.iterrows():
        ratio = edge["duration_ratio"]
        ratio_text = f"{ratio:.2f}x" if pd.notna(ratio) else "new edge"
        first = edge.get("after_first", np.nan)
        first_text = (
            f"{first - injection_time:+.3f}s"
            if pd.notna(first) else "n/a"
        )
        lines.append(
            f"- {edge['caller']} -> {edge['callee']}: "
            f"spans {int(edge['before_spans'])}->{int(edge['after_spans'])}; "
            f"error_spans {int(edge['before_errors'])}->"
            f"{int(edge['after_errors'])}; median_duration={ratio_text}; "
            f"first_after={first_text}"
        )
    return "\n".join(lines)


