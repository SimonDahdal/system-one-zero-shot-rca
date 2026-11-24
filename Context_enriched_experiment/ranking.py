"""Rank services using shared transports, without modifying shared model adapters."""
from dataclasses import dataclass, field
from importlib import import_module
import json
import math
from numbers import Real
import re
import time

from rca_experiments.models import MODEL_REGISTRY
from rca_experiments.models.base import Prediction

SCORE_MODELS = set(MODEL_REGISTRY) - {'rf', 'nimble', 'semif'}
# Upstream capability references checked 2026-11-02; deployed versions may differ.
METHOD_SOURCES = {
    'kev': 'https://github.com/jaredpalmer/kev/blob/main/README.md',
    'laya': 'https://github.com/NandhaKishorM/laya',
    'nimble': 'https://github.com/bespokelabsai/nimble/blob/main/docs/PARALLEL_SCORING.md',
    'semif': 'https://github.com/TheoLeeCJ/SemIf-OpenJev/blob/master/src/semif_phase1/direct.py',
    'rizzo_flow': 'https://github.com/Rizzo-AI-Academy/rizzo-flow',
    'von': 'https://github.com/wfzyx/von',
    'nanojev': 'https://github.com/TianyuCodings/NanoJev',
    'LiquidAI_d1_3B': 'https://huggingface.co/LiquidAI/d1-3B',
    'decider': 'https://github.com/Mapika/decider',
    'h2o_lightning_4B': 'https://huggingface.co/h2oai/h2o-lightning-4b',
    'neohorse': 'https://huggingface.co/TokenRhythm/NeoHorse-Jev-4B',
    'd3_flash_server': 'https://huggingface.co/vllm-sr/d3-flash',
}

RUBRIC = [
    'Very unlikely to be the faulty service.',
    'Unlikely to be the faulty service.',
    'Neither likely nor unlikely to be the faulty service.',
    'Likely to be the faulty service.',
    'Very likely to be the faulty service.',
]
QUESTION = 'root_cause'


def state_for(candidates):
    return 'Identify the faulty service.\nCandidate services: ' + ', '.join(candidates)


def score_questions(candidates):
    return {f'service_{i}': {
        'type': 'score',
        'instructions': f'How likely is {service} to be the faulty service?',
        'criteria': RUBRIC.copy(),
    } for i, service in enumerate(candidates)}


def choice_questions(candidates):
    return {QUESTION: {'type': 'choice',
            'instructions': 'Select the single service most likely to be faulty.',
            'criteria': {service: service for service in candidates}}}


def native_json(value):
    if hasattr(value, 'model_dump'):
        return native_json(value.model_dump(mode='json'))
    if isinstance(value, dict):
        return {str(k): native_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [native_json(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        # Preserve malformed numeric responses in strict JSON diagnostics.
        return repr(value)
    return value


def number(value, low, high, description):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f'{description} must be a finite number')
    value = float(value)
    if not low <= value <= high:
        raise ValueError(f'{description} outside [{low}, {high}]')
    return value


def distribution(raw, labels):
    if not isinstance(raw, dict) or set(raw) != set(labels):
        raise ValueError('Probability map must contain every candidate exactly once')
    values = {key: number(raw[key], 0, 1, f'Probability for {key}') for key in labels}
    if not math.isclose(sum(values.values()), 1, abs_tol=0.02):
        raise ValueError('Probabilities must sum to approximately one (tolerance 0.02)')
    return values


def parse_scores(answers, candidates):
    values = {}
    for i, candidate in enumerate(candidates):
        answer = answers.get(f'service_{i}')
        if not isinstance(answer, dict):
            raise ValueError(f'Missing score answer for {candidate}')
        score = number(answer.get('score'), 0, len(RUBRIC) - 1, f'Score for {candidate}')
        if answer.get('probabilities') is not None:
            probabilities = distribution(answer['probabilities'], [str(i) for i in range(len(RUBRIC))])
            expected = sum(int(level) * prob for level, prob in probabilities.items())
            if not math.isclose(score, expected, abs_tol=0.05):
                raise ValueError(f'Score and level probabilities disagree for {candidate}')
        values[candidate] = score
    return values


def unsupported_score(exc):
    """Never treat timeouts, context limits, or arbitrary HTTP errors as capabilities."""
    message = str(exc).lower()
    return bool(re.search(r'(?:unsupported|unknown|not supported|not implemented|does not support)'
                          r'[^\n]{0,70}\bscore\b|\bscore\b[^\n]{0,70}'
                          r'(?:unsupported|not supported|not implemented)', message))


def context_error(exc):
    return type(exc).__name__.endswith('ContextLimitError')


@dataclass
class RankingTrace:
    requested_method: str
    method: str | None = None
    fallback_used: bool = False
    fallback_reasons: list = field(default_factory=list)
    scores: dict | None = None
    score_kind: str | None = None
    ranking: list = field(default_factory=list)
    ties: list = field(default_factory=list)
    calls: list = field(default_factory=list)
    context_reductions: int = 0
    capability_retries: int = 0

    def fallback(self, reason):
        self.fallback_used = True
        self.fallback_reasons.append(reason)

    def ranked(self, values, candidates, method, kind):
        self.method, self.scores, self.score_kind = method, values, kind
        self.ranking = sorted(candidates, key=lambda service: -values[service])
        self.ties = [[s for s in candidates if values[s] == value]
                     for value in sorted(set(values.values()), reverse=True)
                     if sum(values[s] == value for s in candidates) > 1]

    def as_dict(self):
        from dataclasses import asdict
        result = asdict(self)
        result['tie_break'] = 'saved_candidate_order'
        result['request_count'] = len(self.calls)
        result['failed_request_attempts'] = sum(call.get('error') is not None for call in self.calls)
        result['adapter_context_retries'] = self.context_reductions
        result['transport_retries'] = None
        observed_tokens = [call.get('input_tokens') for call in self.calls]
        result['input_tokens'] = (sum(observed_tokens)
                                  if observed_tokens and all(v is not None for v in observed_tokens) else None)
        result['unavailable'] = {'transport_retries': 'Shared SDK/clients do not expose internal retries'}
        if result['input_tokens'] is None:
            result['unavailable']['input_tokens'] = 'Not reported for every request, or no requests made'
        return result


class SharedTransport:
    """Translate questions through the existing clients, URLs, SDK and validators."""
    def __init__(self, name, model):
        self.name, self.model = name, model
        self.module = import_module(MODEL_REGISTRY[name][0])

    def __call__(self, state, questions, request_id):
        name = self.name
        if name == 'kev':
            from typesafe_sdk import Choice, Score
            typed = {key: (Score if q['type'] == 'score' else Choice)(
                instructions=q['instructions'], criteria=q['criteria'])
                for key, q in questions.items()}
            response = self.model.client.system_one(state=state, questions=typed)
            return native_json(response)
        if name == 'laya':
            return {'answers': self.module.request_laya(state, questions)}
        if name == 'nimble':
            schema = self.module.nimble_schema(questions)
            return self.model.client.predict(state, schema)
        if name == 'semif':
            question = questions[QUESTION]
            return self.model.client.predict(
                state, question['instructions'],
                [{'id': s, 'description': s} for s in question['criteria']],
                decision_id=request_id)
        if name == 'nanojev':
            response = self.model.client.predict({'body': state}, questions, state_id=request_id)
            # Preserve raw state/usage fields as well as normalized answers.
            return {**response, 'answers': self.module.answers_for_state(response, request_id)}
        return self.model.client.predict(state if name == 'neohorse' else {'body': state}, questions)


def choice_answer(name, payload):
    """Support the documented distribution shapes, never guess logits are probabilities."""
    if name == 'nimble':
        answer = {'choice': payload.get('output', {}).get(QUESTION)}
        fields = payload.get('fields', {})
        detail = fields.get(QUESTION, {}) if isinstance(fields, dict) else {}
        if 'scores' in detail:
            answer['probabilities'] = detail['scores']
        return answer
    if name == 'semif':
        answer = {'choice': payload.get('choice')}
        if 'probabilities' in payload:
            raw = payload['probabilities']
            if isinstance(raw, list):
                ids = payload.get('option_ids')
                if not isinstance(ids, list) or len(ids) != len(raw) or len(set(ids)) != len(ids):
                    raise ValueError('SemIf probabilities need matching unique option_ids')
                raw = dict(zip(ids, raw))
            answer['probabilities'] = raw
        return answer
    answer = payload.get('answers', {}).get(QUESTION)
    if not isinstance(answer, dict):
        raise ValueError('Missing root_cause choice answer')
    return answer


class ServiceRanker:
    def __init__(self, name, transport, method='auto', score_batch_size=8,
                 max_input_chars=100000, max_context_reductions=5):
        if method not in {'auto', 'score', 'probabilities', 'sequential'}:
            raise ValueError('Unknown ranking method')
        if method == 'score' and name not in SCORE_MODELS:
            raise ValueError(f'{name} has no native Score mapping on its configured endpoint')
        if score_batch_size < 1 or max_input_chars < 1 or max_context_reductions < 0:
            raise ValueError('Invalid ranking budgets')
        self.name, self.transport, self.method = name, transport, method
        self.batch_size, self.max_input_chars = score_batch_size, max_input_chars
        self.max_context_reductions = max_context_reductions
        self.score_disabled_reason = None
        self._incident_state = None

    def request(self, candidates, questions, incident_id, trace):
        state = self._incident_state if self._incident_state is not None else state_for(candidates)
        canonical_chars = len(state) + len(json.dumps(questions, ensure_ascii=False))
        if canonical_chars > self.max_input_chars:
            raise ValueError(f'Canonical input is {canonical_chars} characters; '
                             f'budget is {self.max_input_chars}; no candidates dropped')
        call = {'state': state, 'questions': questions, 'state_chars': len(state),
                'state_and_questions_chars': canonical_chars, 'error': None}
        trace.calls.append(call)
        start = time.perf_counter()
        try:
            response = native_json(self.transport(state, questions, f'{incident_id}:call-{len(trace.calls)}'))
            call['response'] = response
            usage = response.get('usage', {})
            tokens = usage.get('input_tokens') if isinstance(usage, dict) else None
            if tokens is None and self.name == 'semif':
                tokens = response.get('input_tokens')
            call['input_tokens'] = tokens if isinstance(tokens, int) and not isinstance(tokens, bool) and tokens >= 0 else None
            return response
        except Exception as exc:
            call['error'] = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            call['latency_seconds'] = time.perf_counter() - start

    def score_batch(self, candidates, questions, incident_id, trace):
        try:
            return self.request(candidates, questions, incident_id, trace)['answers']
        except Exception as exc:
            if not context_error(exc) or len(questions) < 2 or trace.context_reductions >= self.max_context_reductions:
                raise
            # Reduce questions per request, preserving every candidate and every rubric.
            trace.context_reductions += 1
            keys = list(questions)
            half = len(keys) // 2
            left = self.score_batch(candidates, {k: questions[k] for k in keys[:half]}, incident_id, trace)
            right = self.score_batch(candidates, {k: questions[k] for k in keys[half:]}, incident_id, trace)
            return {**left, **right}

    def predict(self, system, candidates, incident_id, *, state=None):
        self._incident_state = state
        trace = RankingTrace(self.method)
        try:
            if not candidates or len(set(candidates)) != len(candidates):
                raise ValueError('Candidate inventory must be nonempty and unique')
            if len(candidates) == 1:
                trace.method, trace.ranking = 'single_candidate', candidates.copy()
            else:
                use_score = self.method in {'auto', 'score'} and self.name in SCORE_MODELS
                if use_score and self.score_disabled_reason:
                    trace.fallback(self.score_disabled_reason)
                    use_score = False
                if use_score:
                    trace.method = 'native_score'
                    questions, answers = score_questions(candidates), {}
                    keys = list(questions)
                    try:
                        for start in range(0, len(keys), self.batch_size):
                            group = {k: questions[k] for k in keys[start:start + self.batch_size]}
                            answers.update(self.score_batch(candidates, group, incident_id, trace))
                        values = parse_scores(answers, candidates)
                        trace.ranked(values, candidates, 'native_score', 'expected_rubric_level_0_to_4')
                    except Exception as exc:
                        if self.method != 'auto' or not unsupported_score(exc):
                            raise
                        self.score_disabled_reason = f'Native Score explicitly rejected: {exc}'
                        trace.capability_retries += 1
                        trace.fallback(self.score_disabled_reason)
                if not trace.ranking:
                    self.choice_rank(candidates, incident_id, trace)
            top3 = [f'{system}::{service}' for service in trace.ranking[:min(3, len(candidates))]]
            return Prediction(top3[0], top3), trace.as_dict()
        except Exception as exc:
            # A later elimination call can fail after a valid Top-1 was obtained.
            top1 = f'{system}::{trace.ranking[0]}' if trace.ranking else None
            return Prediction(top1, [], f'{type(exc).__name__}: {exc}'), trace.as_dict()
        finally:
            self._incident_state = None

    def choice_rank(self, candidates, incident_id, trace):
        remaining = list(candidates)
        sequential = self.method == 'sequential'
        trace.method = 'sequential_choice' if sequential else 'choice_probabilities'
        while len(trace.ranking) < min(3, len(candidates)):
            if len(remaining) == 1:
                trace.ranking.append(remaining[0])
                break
            response = self.request(remaining, choice_questions(remaining), incident_id, trace)
            answer = choice_answer(self.name, response)
            if not sequential:
                if answer.get('probabilities') is not None:
                    values = distribution(answer['probabilities'], remaining)
                    trace.ranked(values, candidates, 'choice_probabilities', 'conditional_candidate_probability')
                    return
                if self.method == 'probabilities':
                    raise ValueError('Per-candidate probabilities were not returned')
                trace.fallback('Choice response has no per-candidate probabilities; using repeated selection without replacement')
                trace.method = 'sequential_choice'
                sequential = True
            selected = answer.get('choice')
            if not isinstance(selected, str) or selected not in remaining:
                raise ValueError(f'Invalid or repeated selected service: {selected!r}')
            trace.ranking.append(selected)
            remaining.remove(selected)



class ChoiceProbabilityRanker(ServiceRanker):
    """Enforce probability-only behavior even if a caller requests another mode."""
    def __init__(self, name, transport, method='probabilities', *args, **kwargs):
        if method != 'probabilities':
            raise ValueError('This experiment supports only Choice probabilities')
        super().__init__(name, transport, method, *args, **kwargs)


def mock_transport(name):
    """Offline protocol fixtures: input-dependent, deterministic, no labels or network."""
    def respond(state, questions, request_id):
        if next(iter(questions.values()))['type'] == 'score':
            return {'answers': {key: {'type': 'score', 'score': (int(key.split('_')[1]) % 5) * 0.7}
                                for key in questions}}
        candidates = list(questions[QUESTION]['criteria'])
        total = sum(range(1, len(candidates) + 1))
        probs = {s: (i + 1) / total for i, s in enumerate(candidates)}
        choice = candidates[-1]
        if name == 'nimble':
            return {'output': {QUESTION: choice}, 'fields': {QUESTION: {'scores': probs}}}
        if name == 'semif':
            return {'choice': choice, 'option_ids': candidates, 'probabilities': list(probs.values())}
        return {'answers': {QUESTION: {'choice': choice, 'probabilities': probs}}}
    return respond
