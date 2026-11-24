"""Von /v1/systemone adapter using the shared RCA prompt."""
import json
import re
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from .. import config
from .base import Prediction
from .decision_prompt import build_decision_prompt, decode_choices


class VonContextLimitError(RuntimeError):
    """Explicit server rejection before inference due to context length."""

    def __init__(self, tokens, limit):
        self.tokens = tokens
        self.limit = limit
        super().__init__(f"Von prompt has {tokens} tokens; limit is {limit}")


class VonClient:
    def __init__(self, base_url, timeout=300, model="von-latest"):
        if timeout <= 0:
            raise ValueError("Von timeout must be positive")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.model = model

    def predict(self, state, questions):
        request = Request(
            self.base_url + "/v1/systemone",
            data=json.dumps({"model": self.model, "state": state,
                             "questions": questions}).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                payload = json.load(response)
        except HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")[:1000]
            length_error = re.search(
                r"(\d+) tokens exceeds the context limit (\d+)", details
            )
            if exc.code == 422 and length_error:
                raise VonContextLimitError(
                    int(length_error[1]), int(length_error[2])
                ) from exc
            raise RuntimeError(f"Von returned HTTP {exc.code}: {details}") from exc
        except URLError as exc:
            raise RuntimeError(f"Cannot reach Von: {exc.reason}") from exc
        except TimeoutError as exc:
            raise RuntimeError("Von request timed out; model loading may need more time.") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("answers"), dict):
            raise ValueError("Von response must contain an answers object")
        return payload


class VonModel:
    def fit(self, dataset):
        self.candidates = dataset.candidates
        self.client = VonClient(config.VON_BASE_URL, config.VON_TIMEOUT_SECONDS,
                                  config.VON_MODEL)

    def predict(self, features, incidents):
        predictions = []
        for incident in incidents:
            try:
                state, questions, candidates, top3_options = build_decision_prompt(
                    incident, self.candidates
                )
                answers = {}
                pending = {}
                for name, question in questions.items():
                    choices = list(question["criteria"])
                    if not choices:
                        raise ValueError(f"Von question {name} has no candidate choices")
                    if len(choices) == 1:
                        # A single legal choice
                        # is determined locally without adding fake candidates.
                        answers[name] = {"choice": choices[0]}
                    else:
                        pending[name] = question
                if pending:
                    budget = len(state)
                    for attempt in range(config.VON_CONTEXT_RETRIES + 1):
                        if config.PRINT_VON_INPUT:
                            print("\n========== VON INPUT ==========\n", state)
                            print("Von candidates:", candidates)
                        try:
                            remote = self.client.predict({"body": state}, pending)["answers"]
                            break
                        except VonContextLimitError as exc:
                            if attempt >= config.VON_CONTEXT_RETRIES:
                                raise RuntimeError(
                                    f"{exc}; exhausted {config.VON_CONTEXT_RETRIES} "
                                    "telemetry reduction retries"
                                ) from exc
                            # Characters guide reduction; only the server can
                            # measure tokens including question/choice overhead.
                            factor = min(0.5, 0.8 * exc.limit / max(exc.tokens, 1))
                            budget = int(budget * factor)
                            print(
                                f"Von {incident.get('case', '')}: {exc}. "
                                f"Retry {attempt + 1}/{config.VON_CONTEXT_RETRIES} "
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
                    "Von",
                )
                predictions.append(Prediction(top1, top3))
            except Exception as exc:
                predictions.append(Prediction(None, [], str(exc)))
        return predictions
