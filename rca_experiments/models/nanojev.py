"""NanoJev batch API adapter using the shared RCA prompt."""
import json
import re
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from .. import config
from .base import Prediction
from .decision_prompt import build_decision_prompt, decode_choices


class NanoJevContextLimitError(RuntimeError):
    def __init__(self, tokens, limit):
        self.tokens = tokens
        self.limit = limit
        super().__init__(f"NanoJev prompt has {tokens} tokens; limit is {limit}")


def check_context_error(message):
    """Recognize the server's explicit no-truncation error, not generic failures."""
    match = re.search(r"候选路径为\s*(\d+)\s*token[s]?\s*[，,]\s*超过\s*max_length\s*=\s*(\d+)", str(message))
    if match:
        raise NanoJevContextLimitError(int(match[1]), int(match[2]))


class NanoJevClient:
    def __init__(self, base_url, timeout=300):
        if timeout <= 0:
            raise ValueError("NanoJev timeout must be positive")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def predict(self, state, questions, state_id="request-1"):
        return self.predict_batch([{"id": state_id, "state": state, "questions": questions}])

    def predict_batch(self, states):
        request = Request(
            self.base_url + "/api/evaluate",
            data=json.dumps({"states": states}).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                payload = json.load(response)
        except HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")[:1000]
            if exc.code == 400:
                # Decode JSON first: servers may escape the Chinese message.
                try:
                    error_body = json.loads(details)
                except ValueError:
                    error_body = {}
                check_context_error(error_body.get("error", details) if isinstance(error_body, dict) else details)
            raise RuntimeError(f"NanoJev returned HTTP {exc.code}: {details}") from exc
        except URLError as exc:
            raise RuntimeError(f"Cannot reach NanoJev: {exc.reason}") from exc
        except TimeoutError as exc:
            raise RuntimeError("NanoJev request timed out; model loading may need more time.") from exc
        if isinstance(payload, dict) and "error" in payload:
            check_context_error(payload["error"])
            raise RuntimeError(f"NanoJev API error: {payload['error']}")
        if not isinstance(payload, dict) or not isinstance(payload.get("states"), list):
            raise ValueError("NanoJev response must contain a states list")
        return payload


def answers_for_state(payload, state_id):
    matches = [item for item in payload["states"]
               if isinstance(item, dict) and item.get("id") == state_id]
    if len(matches) != 1:
        raise ValueError(f"NanoJev expected exactly one response for state {state_id!r}")
    item = matches[0]
    if "error" in item:
        check_context_error(item["error"])
        raise RuntimeError(f"NanoJev state {state_id}: {item['error']}")
    if not isinstance(item.get("answers"), dict):
        raise ValueError(f"NanoJev state {state_id} must contain an answers object")
    return item["answers"]


class NanoJevModel:
    def fit(self, dataset):
        self.candidates = dataset.candidates
        self.client = NanoJevClient(config.NANOJEV_BASE_URL, config.NANOJEV_TIMEOUT_SECONDS)

    def predict(self, features, incidents):
        predictions = []
        for position, incident in enumerate(incidents):
            try:
                state, questions, candidates, top3_options = build_decision_prompt(
                    incident, self.candidates
                )
                # NanoJev rejects null/empty semantic descriptions.
                questions = {name: {**question, "criteria": {
                    choice: description or choice
                    for choice, description in question["criteria"].items()
                }} for name, question in questions.items()}
                answers, pending = {}, {}
                for name, question in questions.items():
                    choices = list(question["criteria"])
                    if not choices:
                        raise ValueError(f"NanoJev question {name} has no candidate choices")
                    if len(choices) == 1:
                        answers[name] = {"choice": choices[0]}
                    else:
                        pending[name] = question
                if pending:
                    state_id = str(incident.get("case", f"{incident['system']}:{position}"))
                    budget = len(state)
                    for attempt in range(config.NANOJEV_CONTEXT_RETRIES + 1):
                        if config.PRINT_NANOJEV_INPUT:
                            print("\n========== NANOJEV INPUT ==========\n", state)
                            print("NanoJev candidates:", candidates)
                        try:
                            payload = self.client.predict({"body": state}, pending, state_id=state_id)
                            remote = answers_for_state(payload, state_id)
                            break
                        except NanoJevContextLimitError as exc:
                            if attempt >= config.NANOJEV_CONTEXT_RETRIES:
                                raise RuntimeError(
                                    f"{exc}; exhausted {config.NANOJEV_CONTEXT_RETRIES} "
                                    "telemetry reduction retries"
                                ) from exc
                            # Characters guide reduction; server token counts
                            # include candidate-path overhead and remain authoritative.
                            budget = int(budget * min(0.5, 0.8 * exc.limit / max(exc.tokens, 1)))
                            print(
                                f"NanoJev {state_id}: {exc}. "
                                f"Retry {attempt + 1}/{config.NANOJEV_CONTEXT_RETRIES} "
                                f"with telemetry limited to {budget} characters."
                            )
                            state, _, _, _ = build_decision_prompt(
                                incident, self.candidates, telemetry_char_budget=budget
                            )
                    answers.update({name: remote.get(name, {}) for name in pending})
                top1_answer = answers.get("root_cause_top1", {})
                top3_answer = answers.get("root_cause_top3", {})
                top1, top3 = decode_choices(
                    incident["system"], candidates, top3_options,
                    top1_answer.get("choice") if isinstance(top1_answer, dict) else None,
                    top3_answer.get("choice") if isinstance(top3_answer, dict) else None,
                    "NanoJev",
                )
                predictions.append(Prediction(top1, top3))
            except Exception as exc:
                predictions.append(Prediction(None, [], str(exc)))
        return predictions
