import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from ml.config import RF_HYPERPARAMS

class ModelTrainer:
    def __init__(self):
        self._model = RandomForestClassifier(
            **RF_HYPERPARAMS,
            class_weight="balanced",
        )
        self._is_trained = False

    def train(self, X_train, y_train):
        self._model.fit(X_train, y_train)
        self._is_trained = True
        return self

    def evaluate(self, X_test, y_test):
        if not self._is_trained:
            raise RuntimeError("Call train() before evaluate().")

        y_pred = self._model.predict(X_test)

        report = classification_report(y_test, y_pred, output_dict=True)
        matrix = confusion_matrix(y_test, y_pred, labels=self._model.classes_)
        importance = dict(zip(X_test.columns, self._model.feature_importances_))

        self._print_evaluation(report, matrix, importance)

        return {
            "report": report,
            "confusion_matrix": matrix,
            "feature_importance": importance,
        }

    def get_model(self):
        if not self._is_trained:
            raise RuntimeError("Call train() before get_model().")
        return self._model

    def _print_evaluation(self, report, matrix, importance):
        print("\n── Classification Report ───────────────────────────")
        print(pd.DataFrame(report).transpose().round(3).to_string())

        print("\n── Confusion Matrix ────────────────────────────────")
        labels = self._model.classes_
        print(pd.DataFrame(matrix, index=labels, columns=labels).to_string())

        print("\n── Feature Importance ──────────────────────────────")
        for feature, score in sorted(importance.items(), key=lambda x: -x[1]):
            bar = "█" * int(score * 50)
            print(f"  {feature:<35} {score:.4f}  {bar}")
        print()
