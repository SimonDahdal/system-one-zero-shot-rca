"""Rank RCA services using native scores or explicit alternatives plus a local RF baseline."""
import argparse
import csv
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import socket
import statistics
import time
from types import SimpleNamespace
from urllib.parse import urlparse

from sklearn.metrics import f1_score
from rca_experiments import config
from rca_experiments.dataset import load_dataset
from rca_experiments.models import MODEL_REGISTRY, create_model
from rca_experiments.models.base import Prediction
from .ranking import (ServiceRanker, SharedTransport, RankingTrace, SCORE_MODELS,
                      RUBRIC, METHOD_SOURCES, mock_transport)

ALL_MODELS = ['rf', *[name for name in MODEL_REGISTRY if name != 'rf']]
REMOTE_MODELS = [name for name in MODEL_REGISTRY if name != 'rf']


def summarize(rows):
    def group(items):
        truth = [r['true_label'] for r in items]
        return {'incidents': len(items),
                'top1_accuracy': statistics.mean(r['top1_correct'] for r in items),
                'top3_hit_rate': statistics.mean(r['top3_correct'] for r in items),
                'macro_f1': f1_score(truth, [r['top1'] or '__missing__' for r in items],
                                     labels=sorted(set(truth)), average='macro', zero_division=0),
                'fallback_rate': statistics.mean(bool(r.get('fallback')) for r in items),
                'ranking_method_counts': dict(Counter(r.get('ranking_method') or 'failed_before_ranking' for r in items)),
                'request_failure_rate': statistics.mean(r['request_failed'] for r in items),
                'execution_failure_rate': statistics.mean(r.get('execution_failed', r['request_failed']) for r in items),
                'mean_latency_seconds': statistics.mean(r['latency_seconds'] for r in items),
                'median_latency_seconds': statistics.median(r['latency_seconds'] for r in items)}
    return {'overall': group(rows), 'per_application': {
        system: group([r for r in rows if r['system'] == system])
        for system in sorted({r['system'] for r in rows})}}


def wait_for_deployment(name):
    prefixes = {'laya': 'LAYA', 
                'nimble': 'NIMBLE', 
                'semif': 'SEMIF',
                'rizzo_flow': 'RIZZO', 
                'von': 'VON', 
                'nanojev': 'NANOJEV',
                'LiquidAI_d1_3B': 'D1', 
                'decider': 'DECIDER',
                'h2o_lightning_4B': 'H2O_LIGHTNING', 
                'neohorse': 'NEOHORSE',
                'd3_flash_server': 'D3_FLASH', 
                'kev': 'KEV'}
    prefix = prefixes[name]
    url = getattr(config, prefix + '_BASE_URL', None) or getattr(config, prefix + '_URL')
    address = urlparse(url)
    while True:
        print(f"\nNext model: {name} — {url}")
        print('Unload the previous model if needed, then deploy this model server.')
        print('The experiment does not deploy or stop servers automatically.')
        try:
            input('Press Enter to authorize connecting when ready (Ctrl+C to stop): ')
        except EOFError:
            raise SystemExit('No interactive input; stopped before connecting. Use an interactive terminal.') from None
        try:
            with socket.create_connection(
                    (address.hostname, address.port or (443 if address.scheme == 'https' else 80)),
                    timeout=2):
                return
        except OSError as exc:
            print(f'{name} server unavailable: {exc}. Check deployment and try again.')


def save_report(output_dir, report):
    """Atomically checkpoint a self-contained JSON and a flat comparison table."""
    temporary = output_dir / 'results.json.tmp'
    temporary.write_text(json.dumps(report, indent=2, default=str, allow_nan=False))
    temporary.replace(output_dir / 'results.json')
    table = []
    for name, entry in report['models'].items():
        summary = entry.get('summary')
        if summary is None:
            continue
        for application, metrics in [('overall', summary['overall']),
                                     *summary['per_application'].items()]:
            table.append({'model': name, 'application': application,
                          'evaluation_mode': entry['evaluation_mode'],
                          'input_mode': entry['input_mode'],
                          'top3_protocol': entry['top3_protocol'],
                          'status': entry['status'],
                          'training_or_setup_seconds': entry['training_or_setup_seconds'],
                          **{k: json.dumps(v, sort_keys=True) if isinstance(v, dict) else v
                             for k, v in metrics.items()}})
    if table:
        temporary = output_dir / 'comparison.csv.tmp'
        with temporary.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
        temporary.replace(output_dir / 'comparison.csv')


def run(dataset_path, models, output_dir, metric_chars=2000, max_anomalies=5, mock=False,
        ranking_method='auto', score_batch_size=8, max_input_chars=100000,
        max_context_reductions=5, *, ranker_class=ServiceRanker,
        prompt_builder=None, experiment_metadata=None, extra_source_roots=(),
        evidence_prompt_builder=None):
    dataset_path, output_dir = Path(dataset_path), Path(output_dir)
    if metric_chars < 0 or max_anomalies < 1:
        raise ValueError('metric_chars must be nonnegative; max_anomalies must be positive')
    if not models or len(set(models)) != len(models) or any(m not in ALL_MODELS for m in models):
        raise ValueError('Select distinct registered models')
    if ranking_method not in {'auto', 'score', 'probabilities', 'sequential'}:
        raise ValueError('Unknown ranking method')
    if ranking_method == 'score' and any(name not in SCORE_MODELS | {'rf'} for name in models):
        raise ValueError('Native Score is not mapped for Nimble/SemIf; use auto or probabilities')
    if score_batch_size < 1 or max_input_chars < 1 or max_context_reductions < 0:
        raise ValueError('Invalid ranking budgets')
    dataset = load_dataset(dataset_path)  # Never prepare, resplit, or overwrite the dataset.
    indices = [int(i) for i in dataset.test_idx]
    if not indices:
        raise ValueError('Saved test split is empty')
    output_dir.mkdir(parents=True, exist_ok=False)
    settings = {key: value for key, value in vars(config).items() if key.isupper()}
    manifest = {'dataset': str(dataset_path.resolve()),
                'dataset_sha256': hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
                'ranking_method': ranking_method, 'score_rubric': RUBRIC,
                'method_sources': METHOD_SOURCES, 'sources_checked_on': '2026-11-02',
                'score_batch_size': score_batch_size, 'max_input_chars': max_input_chars,
                'max_context_reductions': max_context_reductions,
                'tie_break': 'saved_candidate_order',
                'models': models, 'prompt_strategy': 'instructions_and_candidates_only',
                'metric_input_to_remote_models': False,
                'ignored_legacy_budgets': {'metric_chars': metric_chars, 'max_anomalies': max_anomalies},
                'mock': mock, 'deployment_confirmation': not mock, 'python': platform.python_version(),
                'shared_config': settings, 'test_indices': indices,
                'train_indices': [int(i) for i in dataset.train_idx],
                'incidents': [{'index': i, 'case': dataset.incidents[i]['case'],
                               'system': dataset.incidents[i]['system']} for i in indices],
                'candidates': dataset.candidates,
                'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for root in ('rca_experiments', *extra_source_roots)
                                  for p in Path(root).rglob('*.py')}}
    manifest['experiment'] = 'Context_enriched_experiment'
    manifest['native_score_enabled'] = ranking_method in {'auto', 'score'}
    if ranking_method == 'probabilities':
        manifest['missing_probabilities_policy'] = 'fail_without_fallback'
        for key in ('score_rubric', 'score_batch_size', 'max_context_reductions',
                    'ignored_legacy_budgets'):
            manifest.pop(key, None)
    if experiment_metadata:
        manifest.update(experiment_metadata)
    (output_dir / 'configuration.json').write_text(json.dumps(manifest, indent=2, default=str))
    summaries = {}
    report = {'schema_version': 2, 'status': 'running', 'configuration': manifest,
              'models': {name: {'status': 'pending'} for name in models}}
    save_report(output_dir, report)
    for name in models:
        if not mock and name != 'rf':
            wait_for_deployment(name)
        rows = []
        setup_error = None
        setup_start = time.perf_counter()
        try:
            if name == 'rf':
                model = create_model(name)
                model.fit(dataset)
            else:
                if mock:
                    transport = mock_transport(name)
                else:
                    model = create_model(name)
                    model.fit(SimpleNamespace(candidates=dataset.candidates))
                    transport = SharedTransport(name, model)
                ranker = ranker_class(name, transport, ranking_method, score_batch_size,
                                       max_input_chars, max_context_reductions)
        except Exception as exc:
            setup_error = f'{type(exc).__name__}: {exc}'
        entry = {'status': 'running', 'evaluation_mode':
                 'local_baseline' if name == 'rf' else ('mock' if mock else 'live'),
                 'input_mode': 'metric_features' if name == 'rf' else
                    ('evidence_guided' if evidence_prompt_builder is not None else
                     'context_enriched' if prompt_builder is not None else 'prompt_only'),
                 'top3_protocol': 'ranked_labels',
                 'preferred_method': 'rf_predict_proba' if name == 'rf' else
                    (('native_score' if name in SCORE_MODELS else 'choice_probabilities')
                     if ranking_method == 'auto' else ranking_method),
                 'training_or_setup_seconds': time.perf_counter() - setup_start,
                 'setup_error': setup_error, 'predictions': rows, 'summary': None}
        report['models'][name] = entry
        save_report(output_dir, report)
        with (output_dir / f'{name}_predictions.jsonl').open('x') as output:
            for position, idx in enumerate(indices):
                incident = dataset.incidents[idx]
                # Explicit allowlist. Even case identifiers can encode the fault.
                safe = {'system': incident['system'], 'case': f'incident-{position}'}
                diagnostics = RankingTrace(ranking_method).as_dict()
                start = time.perf_counter()
                try:
                    if setup_error:
                        raise RuntimeError(setup_error)
                    if name == 'rf':
                        prediction, = model.predict(dataset.X.iloc[[idx]], [safe])
                        diagnostics.update(method='rf_predict_proba', ranking=prediction.top3,
                                           score_kind='class_probability', request_count=0,
                                           tie_break='sklearn_classes_order', ties=None)
                    else:
                        prompt_kwargs = {}
                        if prompt_builder is not None:
                            # Never expose labels, case paths or fault metadata to prompt construction.
                            prompt_kwargs['state'] = prompt_builder(
                                safe['system'], list(dataset.candidates[safe['system']]),
                                dict(incident['metric_features']))
                        if evidence_prompt_builder is not None:
                            # Only the loader receives the raw-data locator and onset, never labels.
                            prompt_kwargs['state'] = evidence_prompt_builder(
                                safe['system'], list(dataset.candidates[safe['system']]),
                                dict(incident['metric_features']), model=name,
                                case_dir=incident['case_dir'],
                                injection_time=incident['injection_time'])
                        prediction, diagnostics = ranker.predict(
                            safe['system'], dataset.candidates[safe['system']], safe['case'],
                            **prompt_kwargs)
                except Exception as exc:
                    prediction = Prediction(None, [], f'{type(exc).__name__}: {exc}')
                latency = time.perf_counter() - start
                truth = dataset.y.iloc[idx]
                valid = {f"{safe['system']}::{s}" for s in dataset.candidates[safe['system']]}
                top1 = prediction.top1 if prediction.top1 in valid else None
                top3 = prediction.top3 if (len(prediction.top3) == min(3, len(valid))
                         and len(set(prediction.top3)) == len(prediction.top3)
                         and set(prediction.top3) <= valid
                         and prediction.top3[0] == top1) else []
                row = {'index': idx, 'case': incident['case'], 'system': safe['system'],
                       'true_label': truth, 'top1': top1, 'top3': top3,
                       'top1_correct': top1 == truth, 'top3_correct': truth in top3,
                       'latency_seconds': latency, 'error': prediction.error,
                       'request_failed': bool(prediction.error or top1 is None or not top3),
                       'fallback': diagnostics['fallback_used'],
                       'ranking_method': diagnostics['method'],
                       'ranking_diagnostics': diagnostics,
                       'prompts': [{k: call[k] for k in ('state', 'questions', 'state_chars',
                                                       'state_and_questions_chars')}
                                   for call in diagnostics['calls']],
                       'context_reductions': diagnostics['context_reductions'],
                       'adapter_context_retries': diagnostics['adapter_context_retries'],
                       'transport_retries': diagnostics['transport_retries'],
                       'input_tokens': diagnostics['input_tokens'],
                       'request_count': diagnostics['request_count'],
                       'failed_request_attempts': diagnostics['failed_request_attempts'],
                       'unavailable': diagnostics['unavailable']}
                if name == 'rf':
                    row.update(fallback=False, context_reductions=None,
                               adapter_context_retries=None,
                               unavailable={key: 'Not applicable to local RF baseline'
                                            for key in ('context_reductions', 'adapter_context_retries',
                                                        'transport_retries', 'input_tokens')})
                    diagnostics['unavailable'] = {**row['unavailable'],
                        'scores': 'Shared RF adapter returns ranked labels, not its full probability array'}
                row['execution_failed'] = row['request_failed']
                if name == 'rf':
                    row['request_failed'] = False  # RF never issues requests.
                output.write(json.dumps(row) + '\n')
                output.flush()
                rows.append(row)
                save_report(output_dir, report)
        summaries[name] = summarize(rows)
        entry['summary'] = summaries[name]
        entry['status'] = 'completed'
        save_report(output_dir, report)
        (output_dir / 'summary.json').write_text(json.dumps(summaries, indent=2))
    report['status'] = 'completed'
    save_report(output_dir, report)
    return summaries

