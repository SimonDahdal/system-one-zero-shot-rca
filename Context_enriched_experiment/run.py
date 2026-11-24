"""Run context-enriched Choice probability evaluation on the exact saved test split."""
import argparse
from functools import partial
import hashlib
import json
from pathlib import Path

from rca_experiments import config
from Context_enriched_experiment.evaluation import ALL_MODELS, run as evaluate
from Context_enriched_experiment.ranking import ChoiceProbabilityRanker
from .prompts import METRICS, build_prompt


def preparation_metadata(dataset_path):
    path = Path(dataset_path).with_suffix('.split.json')
    if not path.exists():
        return None, {'path': str(path), 'available': False}
    raw = path.read_bytes()
    saved = json.loads(raw)
    window = saved.get('metric_window_seconds')
    if window is not None and (isinstance(window, bool) or not isinstance(window, (float, int)) or window <= 0):
        raise ValueError('Invalid metric window in preparation metadata')
    return window, {'path': str(path.resolve()), 'available': True,
                    'sha256': hashlib.sha256(raw).hexdigest(), 'metric_window_seconds': window,
                    'provenance': 'Dataset preparation sidecar; artifact itself does not store window duration'}


def run(dataset_path, models, output_dir, *, mock=False, metrics_per_service=8, max_input_chars=20000):
    if not 1 <= metrics_per_service <= len(METRICS):
        raise ValueError('metrics_per_service must be between 1 and 8')
    window, provenance = preparation_metadata(dataset_path)
    return evaluate(
        dataset_path, models, output_dir, mock=mock, ranking_method='probabilities',
        max_input_chars=max_input_chars, max_context_reductions=0,
        ranker_class=ChoiceProbabilityRanker,
        prompt_builder=partial(build_prompt, metrics_per_service=metrics_per_service, window_seconds=window),
        experiment_metadata={
            'experiment': 'Context_enriched_experiment',
            'prompt_strategy': 'application_description_service_roles_baseline_incident_metrics',
            'metric_input_to_remote_models': True,
            'metric_summary': {'source': 'saved_incident_metric_features',
                'baseline_reconstruction': 'incident_mean - delta',
                'metric_types': list(METRICS[:metrics_per_service]),
                'metrics_per_service': metrics_per_service,
                'selection': 'fixed_metric_order_for_every_candidate_exact_service_prefix',
                'missing_values': 'explicit_missing', 'numeric_significant_digits': 6,
                'units': 'unavailable_in_saved_features', 'preparation_metadata': provenance},
            'context_source': 'rca_experiments.config.SYSTEM_CONTEXT description and roles only',
            'excluded_evidence': ['ground_truth', 'case_paths', 'logs', 'traces', 'explicit_topology'],
        },
        extra_source_roots=('Context_enriched_experiment',))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default=str(config.PREPARED_DATA_PATH))
    parser.add_argument('--models', nargs='+', choices=['all', *ALL_MODELS], default=['all'])
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--mock', action='store_true', help='Mock remote responses; RF remains real')
    parser.add_argument('--metrics-per-service', type=int, choices=range(1, 9), default=8)
    parser.add_argument('--max-input-chars', type=int, default=20000)
    args = parser.parse_args()
    if 'all' in args.models:
        if args.models != ['all']:
            parser.error('Use all alone or list individual models')
        args.models = ALL_MODELS.copy()
    try:
        print(json.dumps(run(args.dataset, args.models, args.output_dir, mock=args.mock,
                             metrics_per_service=args.metrics_per_service,
                             max_input_chars=args.max_input_chars), indent=2))
    except KeyboardInterrupt:
        print('\nStopped. Saved results remain in', args.output_dir)
        raise SystemExit(130) from None


if __name__ == '__main__':
    main()
