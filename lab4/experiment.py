"""Воспроизводимый CPU-эксперимент ЛР4; запуск: python experiment.py."""
from __future__ import annotations
import json
import hashlib
import platform
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mlp import (MLP, Standardizer, load_dataset, stratified_split,
                 classification_metrics, select_threshold, roc_pr_curves, gradient_check)

ROOT = Path(__file__).resolve().parent
CONFIG = dict(layer_sizes=[4, 32, 16, 1], activation="relu", l2=1e-4,
              model_seed=42, split_seed=42, shuffle_seed=43, epochs=150,
              batch_size=32, learning_rate=0.03, momentum=0.9,
              patience=20, min_delta=1e-5)
DATASET_URL = "https://archive.ics.uci.edu/dataset/267/banknote+authentication"


def make_plots(history, y_test, p_test, threshold, sweep, best_epoch, output, y_val, p_val):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.6), constrained_layout=True)
    for ax, columns, ylabel in zip(axes, [("loss", "val_loss"), ("accuracy", "val_accuracy")],
                                   ["BCE без L2", "Accuracy (порог 0.5)"]):
        ax.plot(history.epoch, history[columns[0]], label="train", color="#2563eb")
        ax.plot(history.epoch, history[columns[1]], label="val", color="#ea580c")
        ax.axvline(best_epoch, color="#64748b", linestyle="--", label=f"Лучшая эпоха: {best_epoch}")
        ax.set(xlabel="Эпоха", ylabel=ylabel)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.savefig(output / "learning_curves.png", dpi=180)
    plt.close(fig)

    curves = roc_pr_curves(y_test, p_test)
    metrics = classification_metrics(y_test, p_test, threshold)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.6), constrained_layout=True)
    axes[0].plot(curves["fpr"], curves["tpr"], color="#2563eb", label=f'ROC-AUC = {metrics["roc_auc"]:.4f}')
    axes[0].plot([0, 1], [0, 1], "--", color="#94a3b8")
    axes[0].set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC на test")
    axes[1].step(curves["recall"], curves["precision"], where="pre", color="#ea580c",
                 label=f'Average Precision = {metrics["average_precision"]:.4f}')
    axes[1].axhline(np.mean(y_test), linestyle="--", color="#94a3b8", label="Доля класса 1")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="PR на test")
    for ax in axes:
        ax.set(xlim=(-.02, 1.02), ylim=(-.02, 1.04))
        ax.grid(alpha=.2)
        ax.legend(loc="lower left", fontsize=8)
    fig.savefig(output / "roc_pr.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.6), constrained_layout=True)
    matrix = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    axes[0].imshow(matrix, cmap="Blues")
    for i in range(2):
        for j in range(2):
            axes[0].text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=18,
                         color="white" if matrix[i, j] > matrix.max() / 2 else "#172554")
    axes[0].set(xticks=[0, 1], yticks=[0, 1], xlabel="Предсказанный класс",
                ylabel="Истинный класс", title=f"Матрица ошибок test, τ = {threshold:.4f}")
    # Для графика используем реальные границы решений, а не только кандидаты поиска.
    threshold_x = np.unique(np.r_[np.linspace(1e-6, 1 - 1e-6, 501), p_val, np.nextafter(p_val, 1.)])
    threshold_x = threshold_x[(threshold_x > 0) & (threshold_x < 1)]
    val_f1 = [classification_metrics(y_val, p_val, float(t))["f1"] for t in threshold_x]
    axes[1].step(threshold_x, val_f1, where="post", color="#2563eb")
    axes[1].axvline(threshold, color="#ea580c", linestyle="--", label=f"τ = {threshold:.4f}")
    axes[1].set(xlabel="Порог τ", ylabel="F1 на val", title="Выбор порога только по val", xlim=(0, 1))
    axes[1].grid(alpha=.2)
    axes[1].legend()
    fig.savefig(output / "confusion_threshold.png", dpi=180)
    plt.close(fig)


def run_experiment(output: Path | None = None) -> dict:
    output = ROOT / "artifacts" if output is None else Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = ROOT / "data" / "data_banknote_authentication.txt"
    x, y, original_rows, audit = load_dataset(path)
    splits = stratified_split(y, CONFIG["split_seed"])
    scaler = Standardizer().fit(x[splits["train"]])
    scaled = {name: scaler.transform(x[indices]) for name, indices in splits.items()}
    targets = {name: y[indices] for name, indices in splits.items()}
    # Разбиение сохраняется для проверки отсутствия утечки и повторения эксперимента.
    pd.concat([pd.DataFrame({"source_row": original_rows[indices], "split": name,
                            "class": y[indices]}) for name, indices in splits.items()],
              ignore_index=True).to_csv(output / "split.csv", index=False)
    model = MLP(CONFIG["layer_sizes"], CONFIG["activation"], CONFIG["l2"], CONFIG["model_seed"])
    initial = {name: model.loss(scaled[name], targets[name]) for name in ("train", "val")}
    history = model.fit(scaled["train"], targets["train"], scaled["val"], targets["val"],
                        epochs=CONFIG["epochs"], batch_size=CONFIG["batch_size"],
                        learning_rate=CONFIG["learning_rate"], momentum=CONFIG["momentum"],
                        patience=CONFIG["patience"], min_delta=CONFIG["min_delta"], seed=CONFIG["shuffle_seed"])
    history.to_csv(output / "history.csv", index=False)
    p_val = model.predict_proba(scaled["val"])
    threshold, sweep = select_threshold(targets["val"], p_val)
    sweep.to_csv(output / "threshold_search.csv", index=False)
    # Только после фиксации модели и τ оцениваем test. Переобучения на train+val нет.
    p_test = model.predict_proba(scaled["test"])
    model.save(output / "model.npz", scaler, threshold)
    pd.DataFrame({"source_row": original_rows[splits["test"]], "y_true": targets["test"],
                  "probability": p_test, "y_pred": (p_test >= threshold).astype(int)}).to_csv(
                      output / "test_predictions.csv", index=False)
    checks = {}
    for activation in ("relu", "tanh"):
        tiny = MLP((4, 5, 3, 1), activation=activation, l2=0.07, seed=12)
        # Ненулевые смещения убирают недифференцируемые точки ReLU (z=0).
        rng = np.random.default_rng(123)
        tiny.biases = [rng.uniform(0.2, 0.5, b.shape) for b in tiny.biases]
        checks[activation] = gradient_check(tiny, scaled["train"][:7], targets["train"][:7])
    result = {
        "author": "Тоц Леонид Александрович", "group": "ИВТ-2", "date": "2026-10-06",
        "task": "binary_classification", "dataset": "Banknote Authentication",
        "source": DATASET_URL, "doi": "10.24432/C55P57", "license": "CC BY 4.0",
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "pandas": pd.__version__, "matplotlib": matplotlib.__version__, "device": "CPU"},
        "audit": audit, "config": CONFIG,
        "splits": {name: {"rows": len(indices), "class_counts": np.bincount(y[indices]).tolist()}
                   for name, indices in splits.items()},
        "scaler": {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()},
        "parameters": sum(w.size + b.size for w, b in zip(model.weights, model.biases)),
        "initial_loss": initial, "best_epoch": model.best_epoch_, "stop_epoch": model.stop_epoch_,
        "early_stopping_triggered": model.stop_epoch_ < CONFIG["epochs"],
        "threshold": threshold, "threshold_rule": "max val F1; ties: closest to 0.5, then smaller",
        "gradient_checks": checks,
        "metrics": {name: {"loss": model.loss(scaled[name], targets[name]),
                           **classification_metrics(targets[name], model.predict_proba(scaled[name]), threshold)}
                    for name in ("train", "val", "test")},
        "test_at_05": classification_metrics(targets["test"], p_test, 0.5),
    }
    (output / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    make_plots(history, targets["test"], p_test, threshold, sweep, model.best_epoch_, output, targets["val"], p_val)
    return result


if __name__ == "__main__":
    result = run_experiment()
    print(json.dumps({key: result[key] for key in ("splits", "best_epoch", "stop_epoch", "threshold", "metrics", "gradient_checks")},
                     indent=2, ensure_ascii=False))
