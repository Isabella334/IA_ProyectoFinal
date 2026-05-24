import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import ConfusionMatrixDisplay

class TrainingReport:
    def __init__(self, eval_results: dict, classes: list):
        self._report = eval_results["report"]
        self._matrix = eval_results["confusion_matrix"]
        self._importance = eval_results["feature_importance"]
        self._classes = classes

    def show(self):
        fig = plt.figure(figsize=(18, 12))
        fig.suptitle("NPC Random Forest — Training Report", fontsize=16, fontweight="bold")

        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

        self._plot_confusion_matrix(fig.add_subplot(gs[0, 0:2]))
        self._plot_feature_importance(fig.add_subplot(gs[0, 2]))
        self._plot_class_metrics(fig.add_subplot(gs[1, 0]))
        self._plot_class_support(fig.add_subplot(gs[1, 1]))
        self._plot_accuracy_summary(fig.add_subplot(gs[1, 2]))

        plt.show()

    def save(self, path: str = "models/training_report.png"):
        fig = plt.figure(figsize=(18, 12))
        fig.suptitle("NPC Random Forest — Training Report", fontsize=16, fontweight="bold")

        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

        self._plot_confusion_matrix(fig.add_subplot(gs[0, 0:2]))
        self._plot_feature_importance(fig.add_subplot(gs[0, 2]))
        self._plot_class_metrics(fig.add_subplot(gs[1, 0]))
        self._plot_class_support(fig.add_subplot(gs[1, 1]))
        self._plot_accuracy_summary(fig.add_subplot(gs[1, 2]))

        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Report saved to {path}")

    def _plot_confusion_matrix(self, ax):
        disp = ConfusionMatrixDisplay(
            confusion_matrix=self._matrix,
            display_labels=self._classes,
        )
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title("Confusion Matrix")

    def _plot_feature_importance(self, ax):
        features = list(self._importance.keys())
        scores   = list(self._importance.values())

        sorted_pairs = sorted(zip(scores, features))
        scores, features = zip(*sorted_pairs)

        bars = ax.barh(features, scores, color="steelblue")
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
        ax.set_title("Feature Importance")
        ax.set_xlabel("Importance")
        ax.set_xlim(0, max(scores) * 1.25)

    def _plot_class_metrics(self, ax):
        labels = self._classes
        precision = [self._report[c]["precision"] for c in labels]
        recall    = [self._report[c]["recall"]    for c in labels]
        f1        = [self._report[c]["f1-score"]  for c in labels]

        x   = np.arange(len(labels))
        w   = 0.25

        ax.bar(x - w, precision, w, label="Precision", color="steelblue")
        ax.bar(x,     recall,    w, label="Recall",    color="seagreen")
        ax.bar(x + w, f1,        w, label="F1",        color="coral")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15)
        ax.set_ylim(0, 1.15)
        ax.set_title("Per-class Metrics")
        ax.legend(fontsize=8)

    def _plot_class_support(self, ax):
        labels  = self._classes
        support = [int(self._report[c]["support"]) for c in labels]

        bars = ax.bar(labels, support, color="mediumpurple")
        ax.bar_label(bars, padding=3)
        ax.set_title("Test Set Support")
        ax.set_ylabel("Samples")
        ax.set_xticklabels(labels, rotation=15)

    def _plot_accuracy_summary(self, ax):
        metrics = {
            "Accuracy":       self._report["accuracy"],
            "Macro F1":       self._report["macro avg"]["f1-score"],
            "Weighted F1":    self._report["weighted avg"]["f1-score"],
            "Macro Recall":   self._report["macro avg"]["recall"],
            "Macro Precision":self._report["macro avg"]["precision"],
        }

        labels = list(metrics.keys())
        values = list(metrics.values())
        colors = ["gold" if v >= 0.95 else "coral" for v in values]

        bars = ax.barh(labels, values, color=colors)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
        ax.set_xlim(0, 1.2)
        ax.set_title("Overall Summary")
        ax.axvline(x=0.95, color="gray", linestyle="--", linewidth=0.8, label="0.95 threshold")
        ax.legend(fontsize=8)
