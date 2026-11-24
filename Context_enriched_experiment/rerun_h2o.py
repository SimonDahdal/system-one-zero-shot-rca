"""Re-evaluate H2O Lightning on the original test set in a separate run."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

from .run import preparation_metadata, run


DEFAULT_SOURCE = Path(__file__).parent / 'runs' / 'live_20261109_113850'


def rerun(source_run, output_dir, *, dataset_path=None, mock=False):
    source_run, output_dir = Path(source_run), Path(output_dir)
    source = json.loads((source_run / 'configuration.json').read_text())
    if source.get('experiment') != 'Context_enriched_experiment':
        raise ValueError('Source must be a context-enriched experiment run')
    if source.get('ranking_method') != 'probabilities':
        raise ValueError('Source must use Choice probabilities')
    dataset_path = Path(dataset_path or source['dataset'])
    if hashlib.sha256(dataset_path.read_bytes()).hexdigest() != source['dataset_sha256']:
        raise ValueError('Dataset differs from the original run; refusing a different test set')
    _, provenance = preparation_metadata(dataset_path)
    original = source['metric_summary']['preparation_metadata']
    if (provenance.get('available'), provenance.get('sha256')) != (
            original.get('available'), original.get('sha256')):
        raise ValueError('Dataset preparation sidecar differs from the original run')
    print(f"Rerunning all {len(source['test_indices'])} test incidents for H2O Lightning.")
    print(f'New results: {output_dir}')
    result = run(
        dataset_path, ['h2o_lightning_4B'], output_dir, mock=mock,
        metrics_per_service=source['metric_summary']['metrics_per_service'],
        max_input_chars=source['max_input_chars'])
    (output_dir / 'rerun_provenance.json').write_text(json.dumps({
        'source_run': str(source_run.resolve()),
        'source_configuration_sha256': hashlib.sha256(
            (source_run / 'configuration.json').read_bytes()).hexdigest(),
        'scope': 'all_saved_test_incidents',
        'mock': mock,
    }, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--dataset', type=Path, help='Relocated copy of the identical dataset')
    parser.add_argument('--output-dir', type=Path, help='New directory; existing directories are refused')
    parser.add_argument('--mock', action='store_true', help='Test the workflow without contacting a server')
    args = parser.parse_args()
    output = args.output_dir or DEFAULT_SOURCE.parent / (
        ('mock_h2o_' if args.mock else 'h2o_rerun_') + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    try:
        result = rerun(args.source_run, output, dataset_path=args.dataset, mock=args.mock)
    except KeyboardInterrupt:
        print(f'\nStopped. Checkpoints remain in {output}')
        raise SystemExit(130) from None
    except (ValueError, OSError) as exc:
        parser.exit(1, f'Error: {exc}\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
