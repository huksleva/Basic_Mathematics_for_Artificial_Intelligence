"""Воспроизводимые эксперименты ЛР5: python experiment.py (без зависимостей)."""

import csv
import json
from pathlib import Path

from vector_binary_classification import (
    HiddenLayerBinaryClassifier,
    OneHiddenBinaryClassifier,
    VectorBinaryNeuron,
    assert_close,
    classification_accuracy,
    mean_binary_cross_entropy,
    numerical_gradient,
    predict_label,
)

INPUTS = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
TARGETS = {"OR": [0, 1, 1, 1], "AND": [0, 0, 0, 1], "XOR": [0, 1, 1, 0]}
CONFIG = {
    "OR": {"learning_rate": 0.5, "epochs": 300, "loss_limit": 0.1, "parameters": 3},
    "AND": {"learning_rate": 0.3, "epochs": 300, "loss_limit": 0.05, "parameters": 5},
    "XOR": {"learning_rate": 0.5, "epochs": 500, "loss_limit": 0.03, "parameters": 9},
}


def new_model(task):
    """Ровно те начальные веса, которые зафиксированы в методичке."""
    if task == "OR":
        return VectorBinaryNeuron(weights=[0.0, 0.1, -0.2])
    if task == "AND":
        return OneHiddenBinaryClassifier(
            hidden_weights=[0.0, 0.1, -0.2], output_weights=[0.0, 0.3]
        )
    if task == "XOR":
        return HiddenLayerBinaryClassifier(
            hidden_layer_weights=[[0.1, 0.4, -0.3], [-0.1, -0.2, 0.5]],
            output_weights=[0.0, 0.3, -0.4],
        )
    raise ValueError(f"Неизвестный эксперимент: {task}")


def gradient_check(model, inputs, targets):
    """Проверить каждый параметр центральной разностью, вернув полную таблицу."""
    loss = mean_binary_cross_entropy(model, inputs, targets)
    loss.backward()
    rows = []
    for index, parameter in enumerate(model.parameters()):
        numerical = numerical_gradient(model, parameter, inputs, targets)
        assert_close(parameter.grad, numerical, tolerance=1e-5)
        rows.append({
            "parameter": index,
            "value": parameter.data,
            "autograd": parameter.grad,
            "numerical": numerical,
            "absolute_error": abs(parameter.grad - numerical),
        })
    return rows


def train(model, inputs, targets, learning_rate, epochs):
    """Полный батч; loss истории измеряется перед обновлением данной эпохи."""
    losses = []
    for epoch in range(epochs):
        loss = mean_binary_cross_entropy(model, inputs, targets)
        loss.backward()
        losses.append(loss.data)
        for parameter in model.parameters():
            parameter.data -= learning_rate * parameter.grad
    return losses


def run_experiment(output_dir=None):
    results = {}
    history = []
    predictions = []
    for task, config in CONFIG.items():
        targets = TARGETS[task]
        checked_model = new_model(task)
        checks = gradient_check(checked_model, INPUTS, targets)
        assert len(checks) == config["parameters"]
        assert any(abs(p.grad) > 1e-8 for p in checked_model.parameters())
        if task == "AND":
            assert_close(checked_model.hidden_weights[1].grad, 0.0)

        model = new_model(task)
        initial_weights = [p.data for p in model.parameters()]
        losses = train(model, INPUTS, targets, config["learning_rate"], config["epochs"])
        probabilities = [model(features).data for features in INPUTS]
        labels = [predict_label(model(features)) for features in INPUTS]
        final_loss = mean_binary_cross_entropy(model, INPUTS, targets).data
        accuracy = classification_accuracy(model, INPUTS, targets)
        assert losses[-1] < config["loss_limit"]
        assert losses[-1] < losses[0]
        assert final_loss < config["loss_limit"]
        assert accuracy == 1.0
        if task == "AND":
            assert all(abs(p.grad) > 1e-8 for p in model.parameters())
        else:
            assert all(abs(p.data) > 0.1 for p in model.parameters())

        results[task] = {
            **config,
            "initial_weights": initial_weights,
            "initial_loss": losses[0],
            "last_epoch_loss_before_update": losses[-1],
            "final_loss_after_update": final_loss,
            "accuracy": accuracy,
            "probabilities": probabilities,
            "predictions": labels,
            "targets": targets,
            "final_weights": [p.data for p in model.parameters()],
            "last_epoch_gradients": [p.grad for p in model.parameters()],
            "gradient_check": checks,
            "max_gradient_error": max(row["absolute_error"] for row in checks),
        }
        history.extend({"task": task, "epoch": epoch, "loss_before_update": loss}
                       for epoch, loss in enumerate(losses, start=1))
        predictions.extend({"task": task, "x1": features[0], "x2": features[1],
                            "target": target, "probability": probability, "prediction": label}
                           for features, target, probability, label in
                           zip(INPUTS, targets, probabilities, labels))

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "results.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        for filename, rows in [("history.csv", history), ("predictions.csv", predictions)]:
            with (output_dir / filename).open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    return results


if __name__ == "__main__":
    results = run_experiment(Path(__file__).resolve().parent / "artifacts")
    for task, result in results.items():
        print(f"{task}: BCE {result['initial_loss']:.9f} -> "
              f"{result['final_loss_after_update']:.9f}; "
              f"accuracy={result['accuracy']:.1f}; "
              f"max |grad - numerical|={result['max_gradient_error']:.3e}")
