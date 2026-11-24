"""Laya /v1/systemone adapter using the same RCA prompt as KEV."""
import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from .. import config
from .base import Prediction
from .decision_prompt import build_decision_prompt, decode_choices


def request_laya(state, questions):
    request = Request(
        config.LAYA_URL,
        data=json.dumps({"state": {"body": state}, "questions": questions}).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urlopen(request, timeout=config.LAYA_TIMEOUT_SECONDS) as response:
            payload = json.load(response)
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(f"Laya API HTTP {exc.code}: {detail}") from exc
    except (URLError, TimeoutError) as exc:
        raise RuntimeError(f"Laya API call failed: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("answers"), dict):
        raise ValueError("Laya response must contain an answers object")
    return payload["answers"]


class LayaModel:
    def fit(self, dataset):
        self.candidates = dataset.candidates

    def predict(self, features, incidents):
        predictions = []
        for incident in incidents:
            try:
                state, questions, candidates, top3_options = build_decision_prompt(
                    incident, self.candidates
                )
                if config.PRINT_LAYA_INPUT:
                    print("\n========== LAYA INPUT ==========\n", state)
                    print("Laya candidates:", candidates)
                answers = request_laya(state, questions)
                top1_answer = answers.get("root_cause_top1", {})
                top3_answer = answers.get("root_cause_top3", {})
                top1, top3 = decode_choices(
                    incident["system"], candidates, top3_options,
                    top1_answer.get("choice") if isinstance(top1_answer, dict) else None,
                    top3_answer.get("choice") if isinstance(top3_answer, dict) else None,
                    "Laya",
                )
                predictions.append(Prediction(top1, top3))
            except Exception as exc:
                predictions.append(Prediction(None, [], str(exc)))
        return predictions
