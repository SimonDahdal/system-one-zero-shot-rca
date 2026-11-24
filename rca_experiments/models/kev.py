"""KEV zero-shot model adapter."""
from typesafe_sdk import Choice, TypeSafeClient
from .base import Prediction
from .decision_prompt import build_decision_prompt, decode_choices
from ..config import KEV_URL, KEV_MODEL, PRINT_KEV_INPUT


def predict_with_kev(incident, client, candidate_services):
    state, questions, candidates, top3_options = build_decision_prompt(
        incident, candidate_services
    )
    if PRINT_KEV_INPUT:
        print("\n========== KEV INPUT ==========\n", state)
        print("KEV candidates:", candidates)
    try:
        response = client.system_one(
            state=state,
            questions={name: Choice(instructions=question["instructions"],
                                    criteria=question["criteria"])
                       for name, question in questions.items()},
        )
    except Exception as exc:
        raise RuntimeError(f"KEV API call failed: {exc}") from exc
    return decode_choices(
        incident["system"], candidates, top3_options,
        response.choices["root_cause_top1"].choice,
        response.choices["root_cause_top3"].choice, "KEV",
    )


class KEVModel:
    def fit(self, dataset):
        self.candidates = dataset.candidates
        self.client = TypeSafeClient(api_key="local", base_url=KEV_URL, model=KEV_MODEL)

    def predict(self, features, incidents):
        predictions = []
        for incident in incidents:
            try:
                top1, top3 = predict_with_kev(incident, self.client, self.candidates)
                predictions.append(Prediction(top1, top3))
            except Exception as exc:
                predictions.append(Prediction(None, [], str(exc)))
        return predictions
