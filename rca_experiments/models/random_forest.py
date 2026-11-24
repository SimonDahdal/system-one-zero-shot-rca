"""Random Forest baseline using the shared training split."""
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from .. import config
from .base import Prediction


class RandomForestModel:
    def fit(self, dataset):
        self.model = make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            RandomForestClassifier(
                n_estimators=config.RF_N_ESTIMATORS,
                max_features=config.RF_MAX_FEATURES,
                class_weight=config.RF_CLASS_WEIGHT,
                random_state=config.RANDOM_STATE,
                n_jobs=config.RF_N_JOBS,
            ),
        )
        self.model.fit(dataset.X.iloc[dataset.train_idx], dataset.y.iloc[dataset.train_idx])

    def predict(self, features, incidents):
        probabilities = self.model.predict_proba(features)
        classes = self.model.classes_
        predictions = []
        for row, incident in zip(probabilities, incidents):
            eligible = [i for i, label in enumerate(classes)
                        if str(label).startswith(f"{incident['system']}::")]
            ranked = sorted(eligible, key=lambda i: row[i], reverse=True)[:3]
            top3 = [str(classes[i]) for i in ranked]
            predictions.append(Prediction(top3[0] if top3 else None, top3,
                                          None if top3 else "No trained classes for system"))
        return predictions
