"""SemIf decisions with the shared RCA state, instructions and candidate sets."""
import json
import re
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from .. import config
from .base import Prediction
from .decision_prompt import build_decision_prompt, decode_choices


class SemIfContextLimitError(RuntimeError):
    def __init__(self, tokens, limit):
        self.tokens = tokens
        self.limit = limit
        super().__init__(f"SemIf prompt has {tokens} tokens; limit is {limit}")


class SemIfClient:
    def __init__(self, base_url, timeout=300):
        if timeout <= 0:
            raise ValueError("SemIf timeout must be positive")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def predict(self, state, question, options, decision_id="decision-1"):
        if not 2 <= len(options) <= 16:
            raise ValueError("SemIf requires 2–16 options per request")
        request = Request(
            self.base_url + "/predict",
            data=json.dumps({"id": decision_id, "state": state,
                             "question": question, "options": options}).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                payload = json.load(response)
        except HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")[:1000]
            length_error = re.search(
                r"(\d+) input tokens exceed limit (\d+)", details
            )
            if exc.code == 422 and length_error:
                raise SemIfContextLimitError(
                    int(length_error[1]), int(length_error[2])
                ) from exc
            raise RuntimeError(f"SemIf returned HTTP {exc.code}: {details}") from exc
        except URLError as exc:
            raise RuntimeError(f"Cannot reach SemIf: {exc.reason}") from exc
        except TimeoutError as exc:
            raise RuntimeError("SemIf request timed out; model loading may need more time.") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("choice"), str):
            raise ValueError("SemIf response must contain a string choice")
        if payload["choice"] not in {option["id"] for option in options}:
            raise ValueError(f"SemIf returned an unknown choice: {payload['choice']!r}")
        return payload


class SemIfModel:
    def fit(self, dataset):
        self.candidates = dataset.candidates
        self.client = SemIfClient(config.SEMIF_BASE_URL, config.SEMIF_TIMEOUT_SECONDS)

    def predict(self, features, incidents):
        predictions = []
        for incident in incidents:
            try:
                state, questions, candidates, top3_options = build_decision_prompt(
                    incident, self.candidates
                )
                # Validate both questions before sending either request. Do not
                # silently prune the candidate sets to meet the API limit.
                if any(not 1 <= len(q["criteria"]) <= 16 for q in questions.values()):
                    raise ValueError("SemIf supports at most 16 options per question; "
                                     "the shared candidate set cannot be sent unchanged")
                budget = len(state)
                for attempt in range(config.SEMIF_CONTEXT_RETRIES + 1):
                    if config.PRINT_SEMIF_INPUT:
                        print("\n========== SEMIF INPUT ==========\n", state)
                        print("SemIf candidates:", candidates)
                    try:
                        answers = {}
                        for name, question in questions.items():
                            options = [{"id": choice, "description": choice}
                                       for choice in question["criteria"]]
                            # With one legal choice (e.g. Top-3 for three services),
                            # the answer is determined without an API call.
                            if len(options) == 1:
                                answers[name] = options[0]["id"]
                            else:
                                answers[name] = self.client.predict(
                                    state, question["instructions"], options,
                                    decision_id=f"{incident.get('case', incident['system'])}:{name}",
                                )["choice"]
                        break
                    except SemIfContextLimitError as exc:
                        if attempt >= config.SEMIF_CONTEXT_RETRIES:
                            raise RuntimeError(
                                f"{exc}; exhausted {config.SEMIF_CONTEXT_RETRIES} "
                                "telemetry reduction retries"
                            ) from exc
                        # The server measures tokens including question/options.
                        # Characters only guide reductions; they are not token counts.
                        factor = min(0.5, 0.8 * exc.limit / max(exc.tokens, 1))
                        budget = int(budget * factor)
                        print(
                            f"SemIf {incident.get('case', '')}: {exc}. "
                            f"Retry {attempt + 1}/{config.SEMIF_CONTEXT_RETRIES} "
                            f"with telemetry limited to {budget} characters."
                        )
                        state, _, _, _ = build_decision_prompt(
                            incident, self.candidates, telemetry_char_budget=budget
                        )
                        # Restart both questions: never combine decisions made
                        # from different versions of the telemetry evidence.
                top1, top3 = decode_choices(
                    incident["system"], candidates, top3_options,
                    answers["root_cause_top1"], answers["root_cause_top3"], "SemIf",
                )
                predictions.append(Prediction(top1, top3))
            except Exception as exc:
                predictions.append(Prediction(None, [], str(exc)))
        return predictions
