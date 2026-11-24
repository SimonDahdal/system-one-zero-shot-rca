"""Build context from static application knowledge and cached numeric metrics only."""
import math
from numbers import Real

from rca_experiments.config import SYSTEM_CONTEXT

# Fixed order, same coverage for every candidate; no label/anomaly-based selection.
METRICS = ('cpu', 'mem', 'latency-90', 'error', 'workload', 'diskio', 'socket', 'latency-50')


def application_context(system, candidates):
    text = SYSTEM_CONTEXT.get(system, '')
    description, separator, remainder = text.partition('Relevant service roles:')
    roles = {}
    if separator:
        current = None
        for line in remainder.split('Important causal relationships:', 1)[0].splitlines():
            line = line.strip()
            if line.startswith('- '):
                current, _, role = line[2:].partition(' ')
                roles[current] = role
            elif line and current:
                roles[current] += ' ' + line
    return (description.strip() or f'Application description unavailable for {system}.',
            {service: roles.get(service, 'Role unavailable.') for service in candidates})


def finite(value):
    return float(value) if isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value) else None


def show(value):
    return 'missing' if value is None else format(value, '.6g')


def build_prompt(system, candidates, metric_features, *, metrics_per_service=8, window_seconds=None):
    if not 1 <= metrics_per_service <= len(METRICS):
        raise ValueError('metrics_per_service must be between 1 and 8')
    description, roles = application_context(system, candidates)
    if window_seconds is None:
        windows = 'Baseline: samples before injection; incident: samples at/after injection. Exact window duration unavailable.'
    else:
        windows = (f'Baseline: [{-window_seconds:g}, 0) seconds; incident: [0, {window_seconds:g}) seconds '
                   'relative to benchmark injection time (provided onset, not detected onset).')
    lines = [
        'Identify the service most likely responsible for this incident.',
        'Compare baseline and incident metrics; distinguish a likely originating fault from propagated symptoms.',
        'Choose only from the candidate services. Missing measurements are not evidence of normal behavior.',
        '', 'Application description:', description, '', 'Candidate services and roles:',
        *[f'- {service}: {roles[service]}' for service in candidates],
        '', 'Baseline/incident metric summaries:', windows,
        'Values retain dataset units; units are not recorded in the saved features. Do not assume percent, milliseconds, or rates.',
        'Baseline mean = saved incident mean minus saved delta. Z-shift uses baseline sample standard deviation.',
        f'Metric coverage: {metrics_per_service}/{len(METRICS)} fixed metric types per candidate; noncandidate services omitted.',
        'service | metric | baseline_mean | incident_mean | delta | z_shift',
    ]
    for service in candidates:
        for metric in METRICS[:metrics_per_service]:
            key = f'{service}_{metric}'
            after = finite(metric_features.get(key + '_mean'))
            delta = finite(metric_features.get(key + '_delta'))
            before = finite(after - delta) if after is not None and delta is not None else None
            zshift = finite(metric_features.get(key + '_zshift'))
            lines.append(' | '.join([service, metric, show(before), show(after), show(delta), show(zshift)]))
    lines.extend(['', 'Assess every candidate using its role and the observed metric changes.',
                  'Large changes alone do not establish causality.'])
    return '\n'.join(lines)
