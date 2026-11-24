"""Nimble /predict adapter using the shared KEV/Laya RCA prompt."""
import json
import re
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from .. import config
from .base import Prediction
from .decision_prompt import build_decision_prompt, decode_choices


class NimbleContextLimitError(RuntimeError):
    """Server rejected a prompt before inference because it exceeds its limit."""

    def __init__(self, tokens, limit):
        self.tokens = tokens
        self.limit = limit
        super().__init__(f"Nimble prompt has {tokens} tokens; limit is {limit}")


class NimbleClient:
    def __init__(self, base_url, timeout=300):
        if timeout <= 0:
            raise ValueError("Nimble timeout must be positive")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def predict(self, context, schema):
        request = Request(
            self.base_url + "/predict",
            data=json.dumps({"context": context, "schema": schema}).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                payload = json.load(response)
        except HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")[:1000]
            length_error = re.search(
                r"Longest prompt has (\d+) tokens; limit is (\d+)", details
            )
            if exc.code == 422 and length_error:
                raise NimbleContextLimitError(
                    int(length_error[1]), int(length_error[2])
                ) from exc
            raise RuntimeError(f"Nimble returned HTTP {exc.code}: {details}") from exc
        except URLError as exc:
            raise RuntimeError(f"Cannot reach Nimble: {exc.reason}") from exc
        except TimeoutError as exc:
            raise RuntimeError("Nimble request timed out; model loading may need more time.") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("output"), dict):
            raise ValueError("Nimble response must contain an output object")
        return payload


def nimble_schema(questions):
    """Keep the shared instructions and candidate ordering in Nimble's format."""
    return {
        name: {
            "type": "enum",
            "choices": list(question["criteria"]),
            "description": question["instructions"],
        }
        for name, question in questions.items()
    }


class NimbleModel:
    def fit(self, dataset):
        self.candidates = dataset.candidates
        self.client = NimbleClient(config.NIMBLE_BASE_URL, config.NIMBLE_TIMEOUT_SECONDS)

    def predict(self, features, incidents):
        predictions = []
        for incident in incidents:
            try:
                state, questions, candidates, top3_options = build_decision_prompt(
                    incident, self.candidates
                )
                schema = nimble_schema(questions)
                budget = len(state)
                for attempt in range(config.NIMBLE_CONTEXT_RETRIES + 1):
                    if config.PRINT_NIMBLE_INPUT:
                        print("\n========== NIMBLE INPUT ==========\n", state)
                        print("Nimble candidates:", candidates)
                    try:
                        output = self.client.predict(state, schema)["output"]
                        break
                    except NimbleContextLimitError as exc:
                        if attempt >= config.NIMBLE_CONTEXT_RETRIES:
                            raise RuntimeError(
                                f"{exc}; exhausted {config.NIMBLE_CONTEXT_RETRIES} "
                                "telemetry reduction retries"
                            ) from exc
                        # Character counts are only a reduction heuristic. The server's
                        # tokenizer remains authoritative, including schema overhead.
                        factor = min(0.5, 0.8 * exc.limit / max(exc.tokens, 1))
                        budget = int(budget * factor)
                        print(
                            f"Nimble {incident.get('case', '')}: {exc}. "
                            f"Retry {attempt + 1}/{config.NIMBLE_CONTEXT_RETRIES} "
                            f"with telemetry limited to {budget} characters."
                        )
                        state, _, _, _ = build_decision_prompt(
                            incident, self.candidates, telemetry_char_budget=budget
                        )
                top1, top3 = decode_choices(
                    incident["system"], candidates, top3_options,
                    output.get("root_cause_top1"), output.get("root_cause_top3"),
                    "Nimble",
                )
                predictions.append(Prediction(top1, top3))
            except Exception as exc:
                predictions.append(Prediction(None, [], str(exc)))
        return predictions
