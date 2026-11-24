"""Run registered models against one persisted split and write common reports."""
import argparse
import json
from pathlib import Path
import time
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report
from . import config
from .dataset import load_dataset, prepare_dataset, save_dataset
from .models import MODEL_REGISTRY, create_model


def wait_for_model_switch(previous, upcoming, output_dir):
    print(f"\n{previous} finished. Results saved in {output_dir}.")
    print(f"Unload the previous model from VRAM and deploy {upcoming} now.")
    print("The runner does not stop or deploy your model servers automatically.")
    try:
        input(f"Press Enter when {upcoming} is ready (Ctrl+C to stop): ")
    except EOFError:
        raise SystemExit(
            "No interactive input available; stopped before starting the next model. "
            "Use --no-pause-between-models for unattended execution."
        ) from None


def run_experiments(dataset, model_names, output_dir, pause_between_models=None):
    if pause_between_models is None:
        pause_between_models = config.PAUSE_BETWEEN_MODELS
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    indices = dataset.test_idx
    # Keep ground truth outside the prediction input.
    incidents = [{k: v for k, v in dataset.incidents[int(i)].items()
                  if k not in {"label", "root_cause_service"}} for i in indices]
    results = pd.DataFrame({
        "case": [incident["case"] for incident in incidents],
        "system": [incident["system"] for incident in incidents],
        "true_label": dataset.y.iloc[indices].tolist(),
    })
    summaries = {}
    previous = None
    for name in model_names:
        if previous is not None and pause_between_models:
            wait_for_model_switch(previous, name, output_dir)
        model = create_model(name)
        start = time.perf_counter()
        model.fit(dataset)
        training_seconds = time.perf_counter() - start
        rows = []
        # Checkpoint each incident, including failures, for every model.
        for position, idx in enumerate(indices):
            start = time.perf_counter()
            prediction = model.predict(dataset.X.iloc[[int(idx)]], [incidents[position]])
            if len(prediction) != 1:
                raise ValueError(f"{name} must return one prediction per incident")
            prediction = prediction[0]
            truth = results.iloc[position]["true_label"]
            rows.append({
                **results.iloc[position].to_dict(),
                "prediction": prediction.top1,
                "top3_predictions": json.dumps(prediction.top3),
                "correct": prediction.top1 == truth,
                "top3_correct": truth in prediction.top3,
                "seconds": time.perf_counter() - start,
                "error": prediction.error,
            })
            pd.DataFrame(rows).to_csv(output_dir / f"{name}_predictions.csv", index=False)
            print(f"{name} [{position + 1}/{len(indices)}] {incidents[position]['case']}: "
                  f"{prediction.top1 or prediction.error}")
        frame = pd.DataFrame(rows)
        for column in ["prediction", "top3_predictions", "correct", "top3_correct", "seconds", "error"]:
            results[f"{name}_{column}"] = frame[column]
        predictions = frame.prediction.fillna("__FAILED__")
        total_test_seconds = float(frame.seconds.sum())
        mean_test_seconds = float(frame.seconds.mean())
        summary = {
            "top1_accuracy": accuracy_score(frame.true_label, predictions),
            "top3_accuracy": float(frame.top3_correct.mean()),
            "macro_f1": f1_score(frame.true_label, predictions,
                                 labels=sorted(frame.true_label.unique()),
                                 average="macro", zero_division=0),
            "failed_predictions": int(frame.prediction.isna().sum()),
            "training_seconds": training_seconds,
            "inference_seconds": total_test_seconds,
            "total_test_seconds": total_test_seconds,
            "mean_test_seconds": mean_test_seconds,
            "total_seconds": training_seconds + total_test_seconds,
            "per_system": frame.groupby("system")[["correct", "top3_correct"]].mean().to_dict("index"),
            "classification_report": classification_report(
                frame.true_label, predictions, zero_division=0, output_dict=True),
        }
        summaries[name] = summary
        if name == "rf":
            joblib.dump({"model": model.model, "features": dataset.X.columns.tolist(),
                         "datasets": list(dataset.candidates)}, output_dir / "rca_re2_random_forest.joblib")
        results.to_csv(output_dir / "comparison.csv", index=False)
        (output_dir / "summary.json").write_text(json.dumps(summaries, indent=2))
        print(f"{name}: Top-1={summary['top1_accuracy']:.3f}, "
              f"Top-3={summary['top3_accuracy']:.3f}, Macro F1={summary['macro_f1']:.3f}")
        print(f"{name}: mean test={mean_test_seconds:.3f}s/incident, "
              f"total test={total_test_seconds:.3f}s, "
              f"training + test={summary['total_seconds']:.3f}s")
        del model
        previous = name
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODEL_REGISTRY, default=config.DEFAULT_MODELS)
    parser.add_argument("--dataset", type=Path, default=config.PREPARED_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=config.RESULTS_DIR)
    parser.add_argument("--skip-download", action="store_true",
                        help="Use local raw data when creating a missing prepared dataset")
    parser.add_argument("--pause-between-models", action=argparse.BooleanOptionalAction,
                        default=config.PAUSE_BETWEEN_MODELS,
                        help="Wait for Enter between models to allow manual VRAM swaps (default: enabled)")
    args = parser.parse_args()
    if not args.dataset.exists():
        dataset = prepare_dataset(download=not args.skip_download)
        save_dataset(dataset, args.dataset)
    else:
        dataset = load_dataset(args.dataset)
    try:
        run_experiments(dataset, args.models, args.output_dir, args.pause_between_models)
    except KeyboardInterrupt:
        print("\nStopped. Previously written results remain in", args.output_dir)
        raise SystemExit(130) from None
