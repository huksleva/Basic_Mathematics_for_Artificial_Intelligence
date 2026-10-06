"""Самописный бинарный MLP: только NumPy для модели и градиентов.

Объекты расположены по строкам: A=(batch, n_in), W=(n_in, n_out).
Pandas используется только для истории обучения, без готовых ML-моделей.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Устойчивая сигмоида без overflow при больших |z|."""
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    positive = z >= 0
    out[positive] = 1 / (1 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1 + exp_z)
    return out


@dataclass
class Standardizer:
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "Standardizer":
        self.mean_ = np.mean(x, axis=0)
        std = np.std(x, axis=0, ddof=0)
        self.scale_ = np.where(std > 0, std, 1.0)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Сначала вызовите fit только на train")
        return (np.asarray(x, dtype=np.float64) - self.mean_) / self.scale_


def load_dataset(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    raw = np.loadtxt(path, delimiter=",", dtype=np.float64)
    if raw.shape != (1372, 5) or not np.isfinite(raw).all():
        raise ValueError("Ожидался Banknote: 1372 строк, 4 признака, метка; без пропусков")
    if set(np.unique(raw[:, -1])) != {0.0, 1.0}:
        raise ValueError("Метки должны быть 0 и 1")
    # Удаляем полные дубликаты ДО разбиения; сохраняем первый исходный индекс.
    _, first = np.unique(raw, axis=0, return_index=True)
    first = np.sort(first)
    unique = raw[first]
    if len(np.unique(unique[:, :-1], axis=0)) != len(unique):
        raise ValueError("Одинаковые признаки имеют разные метки")
    audit = {
        "raw_rows": len(raw), "unique_rows": len(unique),
        "removed_duplicates": len(raw) - len(unique), "features": 4,
        "raw_class_counts": np.bincount(raw[:, -1].astype(int)).tolist(),
        "class_counts": np.bincount(unique[:, -1].astype(int)).tolist(),
        "missing_values": 0,
    }
    return unique[:, :-1], unique[:, -1].astype(int), first, audit


def stratified_split(y: np.ndarray, seed: int = 42) -> dict[str, np.ndarray]:
    """Непересекающиеся ~60/20/20 с сохранением долей обоих классов."""
    rng = np.random.default_rng(seed)
    chunks = {name: [] for name in ("train", "val", "test")}
    for label in np.unique(y):
        indices = rng.permutation(np.flatnonzero(y == label))
        n_train, n_val = round(0.6 * len(indices)), round(0.2 * len(indices))
        chunks["train"].append(indices[:n_train])
        chunks["val"].append(indices[n_train:n_train + n_val])
        chunks["test"].append(indices[n_train + n_val:])
    return {name: rng.permutation(np.concatenate(parts)) for name, parts in chunks.items()}


def roc_pr_curves(y: np.ndarray, p: np.ndarray) -> dict[str, np.ndarray]:
    """ROC и PR с совместной обработкой равных оценок вероятности."""
    y, p = np.asarray(y).ravel(), np.asarray(p).ravel()
    if len(y) != len(p) or not np.isfinite(p).all():
        raise ValueError("Неверные вероятности")
    positives, negatives = np.sum(y == 1), np.sum(y == 0)
    if positives == 0 or negatives == 0:
        raise ValueError("Для ROC-AUC нужны оба класса")
    order = np.argsort(-p, kind="stable")
    scores, labels = p[order], y[order]
    ends = np.r_[np.flatnonzero(np.diff(scores)), len(scores) - 1]
    tp = np.r_[0, np.cumsum(labels == 1)[ends]]
    fp = np.r_[0, (ends + 1) - tp[1:]]
    recall = tp / positives
    precision = np.divide(tp, tp + fp, out=np.ones_like(tp, dtype=float), where=tp + fp > 0)
    return {"fpr": fp / negatives, "tpr": recall, "recall": recall,
            "precision": precision, "thresholds": np.r_[np.inf, scores[ends]]}


def classification_metrics(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict:
    y, p = np.asarray(y).ravel(), np.asarray(p).ravel()
    prediction = p >= threshold
    tp = int(np.sum((y == 1) & prediction))
    tn = int(np.sum((y == 0) & ~prediction))
    fp = int(np.sum((y == 0) & prediction))
    fn = int(np.sum((y == 1) & ~prediction))
    curves = roc_pr_curves(y, p)
    return {
        "accuracy": (tp + tn) / len(y),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "roc_auc": float(np.trapezoid(curves["tpr"], curves["fpr"])),
        "average_precision": float(np.sum(np.diff(curves["recall"]) * curves["precision"][1:])),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def select_threshold(y_val: np.ndarray, p_val: np.ndarray) -> tuple[float, pd.DataFrame]:
    """Максимум F1 на val; при равенстве ближайший к 0.5, затем меньший."""
    scores = np.unique(np.asarray(p_val).ravel())
    candidates = np.unique(np.r_[np.nextafter(0., 1.), 0.5,
                                  (scores[1:] + scores[:-1]) / 2,
                                  np.nextafter(1., 0.)])
    rows = []
    for threshold in candidates:
        row = classification_metrics(y_val, p_val, float(threshold))
        rows.append({"threshold": float(threshold), **row})
    sweep = pd.DataFrame(rows)
    best_f1 = sweep["f1"].max()
    tied = sweep[np.isclose(sweep["f1"], best_f1, rtol=0, atol=1e-12)].copy()
    tied["distance"] = abs(tied["threshold"] - 0.5)
    threshold = float(tied.sort_values(["distance", "threshold"]).iloc[0]["threshold"])
    return threshold, sweep


def log_epoch_metrics(history: list[dict], epoch: int, loss: float, val_loss: float,
                      metric: float, val_metric: float, metric_name: str = "accuracy", **extra) -> None:
    history.append({"epoch": epoch, "loss": loss, "val_loss": val_loss,
                    metric_name: metric, f"val_{metric_name}": val_metric, **extra})


class MLP:
    """Сигмоидный бинарный классификатор, ReLU/tanh, SGD с моментом и L2."""

    def __init__(self, layer_sizes=(4, 32, 16, 1), activation="relu", l2=1e-4, seed=42):
        if len(layer_sizes) < 2 or layer_sizes[-1] != 1 or any(n <= 0 for n in layer_sizes):
            raise ValueError("Нужны положительные размеры и один выходной нейрон")
        if activation not in {"relu", "tanh"} or l2 < 0:
            raise ValueError("Поддерживаются relu/tanh и l2 >= 0")
        self.layer_sizes = tuple(layer_sizes)
        self.activation, self.l2 = activation, float(l2)
        rng = np.random.default_rng(seed)
        self.weights, self.biases = [], []
        for layer, (n_in, n_out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            # He для скрытых ReLU; Xavier для tanh и линейных логитов выхода.
            variance = 2 / n_in if activation == "relu" and layer < len(layer_sizes) - 2 else 1 / n_in
            self.weights.append(rng.normal(0, np.sqrt(variance), (n_in, n_out)))
            self.biases.append(np.zeros((1, n_out), dtype=np.float64))
        self.best_epoch_ = 0
        self.stop_epoch_ = 0

    def forward(self, x: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        activations, preactivations = [np.asarray(x, dtype=np.float64)], []
        for layer, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = activations[-1] @ w + b
            preactivations.append(z)
            if layer == len(self.weights) - 1:
                a = sigmoid(z)
            else:
                a = np.maximum(0, z) if self.activation == "relu" else np.tanh(z)
            activations.append(a)
        return activations, preactivations

    def loss(self, x: np.ndarray, y: np.ndarray, regularized: bool = False) -> float:
        _, z = self.forward(x)
        target = np.asarray(y).reshape(-1, 1)
        # BCE на логитах: softplus(z) - y*z, без log(0) и clipping градиента.
        value = float(np.mean(np.logaddexp(0, z[-1]) - target * z[-1]))
        if regularized:
            value += self.l2 / 2 * sum(float(np.sum(w * w)) for w in self.weights)
        return value

    def loss_and_gradients(self, x: np.ndarray, y: np.ndarray) -> tuple[float, list, list]:
        a, z = self.forward(x)
        target = np.asarray(y).reshape(-1, 1)
        m = len(target)  # реальный размер, включая последний неполный mini-batch
        delta = (a[-1] - target) / m
        dw, db = [None] * len(self.weights), [None] * len(self.weights)
        for layer in reversed(range(len(self.weights))):
            dw[layer] = a[layer].T @ delta + self.l2 * self.weights[layer]
            db[layer] = delta.sum(axis=0, keepdims=True)
            if layer > 0:
                derivative = (z[layer - 1] > 0) if self.activation == "relu" else 1 - a[layer] ** 2
                delta = (delta @ self.weights[layer].T) * derivative
        objective = float(np.mean(np.logaddexp(0, z[-1]) - target * z[-1]))
        objective += self.l2 / 2 * sum(float(np.sum(w * w)) for w in self.weights)
        return objective, dw, db

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)[0][-1].ravel()

    def predict(self, x: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(x) >= threshold).astype(int)

    def fit(self, x_train, y_train, x_val, y_val, *, epochs=150, batch_size=32,
            learning_rate=0.03, momentum=0.9, patience=20, min_delta=1e-5,
            seed=43) -> pd.DataFrame:
        if epochs < 1 or batch_size < 1 or patience < 1 or learning_rate <= 0 or not 0 <= momentum < 1:
            raise ValueError("Некорректные гиперпараметры")
        rng = np.random.default_rng(seed)
        vw = [np.zeros_like(w) for w in self.weights]
        vb = [np.zeros_like(b) for b in self.biases]
        history, best, wait, significant_best = [], None, 0, float("inf")
        lowest_val = float("inf")
        for epoch in range(1, epochs + 1):
            indices = rng.permutation(len(y_train))
            for start in range(0, len(indices), batch_size):
                batch = indices[start:start + batch_size]
                _, dw, db = self.loss_and_gradients(x_train[batch], y_train[batch])
                # Все градиенты вычислены до изменения любого параметра.
                for layer in range(len(self.weights)):
                    vw[layer] = momentum * vw[layer] + dw[layer]
                    vb[layer] = momentum * vb[layer] + db[layer]
                    self.weights[layer] -= learning_rate * vw[layer]
                    self.biases[layer] -= learning_rate * vb[layer]
            loss, val_loss = self.loss(x_train, y_train), self.loss(x_val, y_val)
            if not np.isfinite([loss, val_loss]).all():
                raise FloatingPointError("Обучение разошлось")
            log_epoch_metrics(history, epoch, loss, val_loss,
                              float(np.mean(self.predict(x_train) == y_train)),
                              float(np.mean(self.predict(x_val) == y_val)),
                              objective=self.loss(x_train, y_train, regularized=True))
            # Чекпойнт хранит абсолютный минимум; min_delta регулирует только patience.
            if val_loss < lowest_val:
                lowest_val, self.best_epoch_ = val_loss, epoch
                best = ([w.copy() for w in self.weights], [b.copy() for b in self.biases])
            if val_loss < significant_best - min_delta:
                significant_best, wait = val_loss, 0
            else:
                wait += 1
            if wait >= patience:
                break
        self.stop_epoch_ = epoch
        self.weights, self.biases = best
        result = pd.DataFrame(history)
        result.attrs.update(best_epoch=self.best_epoch_, stop_epoch=self.stop_epoch_,
                            best_val_loss=lowest_val, restored_best=True)
        return result

    def save(self, path: Path, scaler: Standardizer, threshold: float) -> None:
        arrays = {f"W{i}": w for i, w in enumerate(self.weights)}
        arrays.update({f"b{i}": b for i, b in enumerate(self.biases)})
        np.savez_compressed(path, **arrays, layer_sizes=self.layer_sizes,
                            activation=self.activation, l2=self.l2, mean=scaler.mean_,
                            scale=scaler.scale_, threshold=threshold,
                            best_epoch=self.best_epoch_, stop_epoch=self.stop_epoch_)

    @classmethod
    def load(cls, path: Path) -> tuple["MLP", Standardizer, float]:
        with np.load(path, allow_pickle=False) as stored:
            model = cls(tuple(stored["layer_sizes"]), str(stored["activation"]), float(stored["l2"]))
            model.weights = [stored[f"W{i}"].copy() for i in range(len(model.weights))]
            model.biases = [stored[f"b{i}"].copy() for i in range(len(model.biases))]
            model.best_epoch_, model.stop_epoch_ = int(stored["best_epoch"]), int(stored["stop_epoch"])
            return model, Standardizer(stored["mean"].copy(), stored["scale"].copy()), float(stored["threshold"])


def gradient_check(model: MLP, x: np.ndarray, y: np.ndarray, epsilon=1e-6) -> dict:
    """Центральная разность для КАЖДОГО веса и смещения маленькой сети."""
    if model.activation == "relu":
        _, z = model.forward(x)
        if any(np.any(abs(hidden) < 100 * epsilon) for hidden in z[:-1]):
            raise ValueError("Для численной проверки ReLU выберите точку вдали от z=0")
    _, dw, db = model.loss_and_gradients(x, y)
    analytical, numerical = [], []
    for parameter, gradient in zip(model.weights + model.biases, dw + db):
        for index in np.ndindex(parameter.shape):
            original = parameter[index]
            try:
                parameter[index] = original + epsilon
                plus = model.loss(x, y, regularized=True)
                parameter[index] = original - epsilon
                minus = model.loss(x, y, regularized=True)
            finally:
                parameter[index] = original
            analytical.append(gradient[index])
            numerical.append((plus - minus) / (2 * epsilon))
    analytical, numerical = np.array(analytical), np.array(numerical)
    return {"parameters_checked": len(analytical),
            "max_absolute_error": float(np.max(abs(analytical - numerical))),
            "relative_l2_error": float(np.linalg.norm(analytical - numerical) /
                                       max(1e-12, np.linalg.norm(analytical) + np.linalg.norm(numerical)))}
