"""Identical RCA prompt and choice decoding for KEV and Laya."""
from itertools import combinations
from ..telemetry import prepare_logs, prepare_metric_summary, prepare_trace_summary
from ..config import INCLUDE_METRIC_SUMMARY, SYSTEM_CONTEXT, telemetry_config


def compact_telemetry(sections, character_budget):
    """Shorten each telemetry section proportionally, retaining its head and tail."""
    total = sum(map(len, sections))
    if total <= character_budget:
        return sections
    compacted = []
    marker = "\n[Telemetry shortened to fit model context]\n"
    for section in sections:
        allowance = int(character_budget * len(section) / max(total, 1))
        if allowance <= len(marker):
            compacted.append(section[:allowance])
            continue
        remaining = allowance - len(marker)
        head = (remaining + 1) // 2
        tail = remaining // 2
        compacted.append(section[:head] + marker + (section[-tail:] if tail else ""))
    return compacted


def build_decision_prompt(incident, candidate_services, telemetry_char_budget=None):
    system = incident["system"]
    case_dir = incident["case_dir"]
    injection_time = incident["injection_time"]
    config = telemetry_config(system)

    try:
        log_text = prepare_logs(case_dir, injection_time, config)
    except Exception as exc:
        raise RuntimeError(
            f"log preparation failed: {exc}"
        ) from exc

    candidates = candidate_services[system]
    system_context = SYSTEM_CONTEXT[system].strip()
    top3_size = min(3, len(candidates))
    top3_options = {
        " | ".join(group): group
        for group in combinations(candidates, top3_size)
    }

    try:
        if INCLUDE_METRIC_SUMMARY:
            metric_text = prepare_metric_summary(
                incident["metric_features"],
                candidates,
                config["max_metric_anomalies"],
            )
        else:
            metric_text = "METRIC ANOMALY SUMMARY\nDisabled."
    except Exception as exc:
        raise RuntimeError(
            f"metric summary failed: {exc}"
        ) from exc

    trace_text = (
        prepare_trace_summary(case_dir, injection_time)
        if config["include_trace_summary"]
        else "TRACE SUMMARY\nDisabled for faster inference."
    )

    if telemetry_char_budget is not None:
        metric_text, trace_text, log_text = compact_telemetry(
            [metric_text, trace_text, log_text], telemetry_char_budget
        )

    top1_instructions = (
        "Select the single service most likely to be the PRIMARY root cause. "
        "Evaluate every candidate using: "
        "(1) direct internal failure evidence, (2) whether its anomalies "
        "precede failures elsewhere, (3) new or sharply increased log "
        "templates, (4) metric shifts, and (5) trace caller-to-callee "
        "evidence. Prefer an originating failed callee over an orchestrator "
        "or caller that only reports a propagated error. Do not select a "
        "service merely because it emits the most logs."
    )
    top3_instructions = (
        f"Select the unordered set of {top3_size} services most likely to "
        "contain the primary root cause. Use the same causal evidence and "
        "prefer originating dependencies over downstream symptom reporters."
    )

    state = (
        "Dataset context:\n"
        "This incident comes from the RCAEval RE2 fault-injection benchmark. "
        "The logs below include baseline observations from before injection "
        "and incident observations from after injection. Focus on new or "
        "changed behavior after time zero. A fault in one service can cause "
        "errors, retries, or timeouts in its callers.\n\n"
        f"Application architecture:\n{system_context}\n\n"
        f"Candidate services: {', '.join(candidates)}\n\n"
        f"{metric_text}\n\n"
        f"{trace_text}\n\n"
        f"{log_text}"
    )

    questions = {
        "root_cause_top1": {
            "type": "choice", "instructions": top1_instructions,
            "criteria": {service: None for service in candidates},
        },
        "root_cause_top3": {
            "type": "choice", "instructions": top3_instructions,
            "criteria": {option: None for option in top3_options},
        },
    }
    return state, questions, candidates, top3_options


def decode_choices(system, candidates, top3_options, selected_service_raw,
                   selected_top3_raw, model_name):
    top3_size = min(3, len(candidates))
    # Do not compare pd.NA directly with strings: pandas raises
    # "boolean value of NA is ambiguous" for that operation.
    top1_is_valid = (
        isinstance(selected_service_raw, str)
        and selected_service_raw in candidates
    )
    top3_is_valid = (
        isinstance(selected_top3_raw, str)
        and selected_top3_raw in top3_options
    )

    if not top1_is_valid and not top3_is_valid:
        raise ValueError(
            f"{model_name} returned no valid choices: "
            f"Top-1={selected_service_raw!r}, "
            f"Top-3={selected_top3_raw!r}"
        )

    if top3_is_valid:
        top3_services = list(top3_options[selected_top3_raw])
    else:
        top3_services = [selected_service_raw] + [
            service for service in candidates
            if service != selected_service_raw
        ][:top3_size - 1]
        print(
            f"{model_name} warning: missing Top-3 choice; constructed fallback "
            "from the valid Top-1 choice."
        )

    if top1_is_valid:
        selected_service = selected_service_raw
    else:
        selected_service = top3_services[0]
        print(
            f"{model_name} warning: missing Top-1 choice; using the first service "
            "from the valid Top-3 response."
        )

    selected_top3 = [
        f"{system}::{service}" for service in top3_services
    ]

    return f"{system}::{selected_service}", selected_top3

