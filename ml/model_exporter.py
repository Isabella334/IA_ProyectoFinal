import json
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from ml.config import FEATURE_COLS, MODEL_OUTPUT_PATH

class ModelExporter:
    def __init__(self, model: RandomForestClassifier):
        self._model = model

    def save(self, path: str = MODEL_OUTPUT_PATH):
        payload = self._serialize()
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)

        with open(output, "w") as f:
            json.dump(payload, f, indent=2)

        print(f"\n── Model Exported ──────────────────────────────────")
        print(f"  Path       : {output.resolve()}")
        print(f"  Trees      : {len(self._model.estimators_)}")
        print(f"  Classes    : {list(self._model.classes_)}")
        print(f"  Features   : {FEATURE_COLS}\n")

    def validate(self, path: str = MODEL_OUTPUT_PATH):
        output = Path(path)
        if not output.exists():
            raise FileNotFoundError(f"No model found at {output.resolve()}")

        with open(output, "r") as f:
            data = json.load(f)

        required_keys = {"feature_names", "classes", "trees"}
        missing = required_keys - data.keys()
        if missing:
            raise ValueError(f"Exported model is missing keys: {missing}")

        print(f"  ✓ Model at {output.resolve()} is valid")

    def _serialize(self):
        return {
            "feature_names": FEATURE_COLS,
            "classes": list(self._model.classes_),
            "n_estimators": len(self._model.estimators_),
            "trees": [self._serialize_tree(est) for est in self._model.estimators_],
        }

    def _serialize_tree(self, estimator):
        tree = estimator.tree_
        return self._serialize_node(tree, node_id=0)

    def _serialize_node(self, tree, node_id):
        is_leaf = tree.children_left[node_id] == -1

        if is_leaf:
            values = tree.value[node_id][0]
            predicted_class_idx = int(np.argmax(values))
            return {
                "leaf": True,
                "class_idx": predicted_class_idx,
                "votes": [int(v) for v in values],
            }

        return {
            "leaf": False,
            "feature_index": int(tree.feature[node_id]),
            "threshold": float(tree.threshold[node_id]),
            "left": self._serialize_node(tree, tree.children_left[node_id]),
            "right": self._serialize_node(tree, tree.children_right[node_id]),
        }
