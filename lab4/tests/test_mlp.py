"""Независимые проверки математики, протокола и сохранённого результата."""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mlp import (MLP, Standardizer, sigmoid, gradient_check, classification_metrics,
                 select_threshold, load_dataset, stratified_split)


@pytest.mark.parametrize("activation", ["relu", "tanh"])
@pytest.mark.parametrize("l2", [0.0, 0.07])
def test_all_gradients_match_central_difference(activation, l2):
    rng = np.random.default_rng(123)
    model = MLP((4, 5, 3, 1), activation=activation, l2=l2, seed=12)
    model.biases = [rng.uniform(.2, .5, b.shape) for b in model.biases]
    x, y = rng.normal(size=(7, 4)), rng.integers(0, 2, 7)
    result = gradient_check(model, x, y)
    assert result["parameters_checked"] == 47
    assert result["max_absolute_error"] < 1e-7
    assert result["relative_l2_error"] < 1e-6


def test_batch_average_is_applied_once():
    rng = np.random.default_rng(1)
    x, y = rng.normal(size=(5, 4)), np.array([0, 1, 1, 0, 1])
    model = MLP((4, 6, 1), l2=.03)
    loss, dw, db = model.loss_and_gradients(x, y)
    repeated_loss, repeated_dw, repeated_db = model.loss_and_gradients(
        np.repeat(x, 3, axis=0), np.repeat(y, 3))
    assert loss == pytest.approx(repeated_loss)
    for expected, actual in zip(dw + db, repeated_dw + repeated_db):
        np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_sigmoid_and_bce_are_stable_for_extreme_logits():
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        output = sigmoid(np.array([-1000., 0., 1000.]))
        np.testing.assert_equal(output, [0., .5, 1.])
        model = MLP((1, 1), l2=0)
        model.weights[0][:] = 1000
        loss, dw, db = model.loss_and_gradients(np.array([[-1.], [1.]]), np.array([1, 0]))
        assert loss == pytest.approx(1000.)
        assert all(np.isfinite(g).all() for g in dw + db)


def test_duplicate_removal_and_stratified_partition():
    x, y, source_rows, audit = load_dataset(ROOT / "data/data_banknote_authentication.txt")
    assert audit["removed_duplicates"] == 24
    assert len(source_rows) == len(set(source_rows)) == 1348
    assert len(np.unique(x, axis=0)) == len(x)
    splits = stratified_split(y)
    all_indices = np.concatenate(list(splits.values()))
    np.testing.assert_equal(np.sort(all_indices), np.arange(len(y)))
    for name, indices in splits.items():
        assert set(y[indices]) == {0, 1}
        assert abs(np.mean(y[indices]) - np.mean(y)) < .005
        np.testing.assert_equal(indices, stratified_split(y)[name])


def test_standardizer_uses_train_and_handles_constant_feature():
    train = np.array([[1., 5.], [3., 5.], [5., 5.]])
    scaler = Standardizer().fit(train)
    transformed = scaler.transform(train)
    np.testing.assert_allclose(transformed.mean(axis=0), 0, atol=1e-12)
    assert transformed[:, 0].std() == pytest.approx(1.)
    np.testing.assert_equal(scaler.scale_[1], 1.)
    old_mean = scaler.mean_.copy()
    assert scaler.transform(np.array([[100., 5.]]))[0, 0] > 50
    np.testing.assert_equal(scaler.mean_, old_mean)


@pytest.mark.parametrize("p", [np.array([.1, .3, .2, .9, .9, .3]), np.full(6, .5)])
def test_metrics_match_independent_sklearn_implementation(p):
    y = np.array([0, 1, 0, 1, 0, 1])
    metrics = classification_metrics(y, p, .4)
    assert metrics["accuracy"] == pytest.approx(accuracy_score(y, p >= .4))
    assert metrics["f1"] == pytest.approx(f1_score(y, p >= .4, zero_division=0))
    assert metrics["roc_auc"] == pytest.approx(roc_auc_score(y, p))
    assert metrics["average_precision"] == pytest.approx(average_precision_score(y, p))


def test_threshold_maximizes_validation_f1_and_uses_tie_rule():
    y, p = np.array([0, 0, 1, 1]), np.array([.1, .2, .8, .9])
    threshold, sweep = select_threshold(y, p)
    assert threshold == .5
    assert classification_metrics(y, p, threshold)["f1"] == sweep.f1.max()
    threshold, _ = select_threshold(y, np.array([.01, .1, .3, .4]))
    assert threshold == pytest.approx(.2)


def test_early_stopping_restores_earlier_best_parameters():
    model = MLP((2, 1), l2=0)
    x = np.zeros((8, 2))
    # Обучение на классе 1 ухудшает val класса 0 после каждой эпохи.
    history = model.fit(x, np.ones(8), x, np.zeros(8), epochs=10, batch_size=3,
                        learning_rate=.5, momentum=0, patience=2, min_delta=0)
    assert model.best_epoch_ == 1
    assert model.stop_epoch_ == 3
    assert model.loss(x, np.zeros(8)) == pytest.approx(history.val_loss.min())
    assert model.loss(x, np.zeros(8)) < history.val_loss.iloc[-1]
    assert list(history.columns) == ["epoch", "loss", "val_loss", "accuracy", "val_accuracy", "objective"]


def test_saved_experiment_reloads_and_has_no_data_leakage():
    result = json.loads((ROOT / "artifacts/results.json").read_text(encoding="utf-8"))
    model, scaler, threshold = MLP.load(ROOT / "artifacts/model.npz")
    raw = np.loadtxt(ROOT / "data/data_banknote_authentication.txt", delimiter=",")
    split = pd.read_csv(ROOT / "artifacts/split.csv")
    assert not split.source_row.duplicated().any()
    train = split[split.split == "train"].source_row.to_numpy()
    np.testing.assert_allclose(scaler.mean_, raw[train, :-1].mean(axis=0), atol=1e-12)
    predictions = pd.read_csv(ROOT / "artifacts/test_predictions.csv")
    x = scaler.transform(raw[predictions.source_row, :-1])
    np.testing.assert_allclose(model.predict_proba(x), predictions.probability, atol=1e-14)
    np.testing.assert_equal(model.predict(x, threshold), predictions.y_pred)
    metrics = classification_metrics(predictions.y_true, predictions.probability, threshold)
    assert metrics["accuracy"] > .95
    assert metrics["accuracy"] == result["metrics"]["test"]["accuracy"]
    assert metrics["roc_auc"] == pytest.approx(roc_auc_score(predictions.y_true, predictions.probability))
    assert metrics["average_precision"] == pytest.approx(average_precision_score(predictions.y_true, predictions.probability))
    history = pd.read_csv(ROOT / "artifacts/history.csv")
    assert model.loss(x, predictions.y_true) == pytest.approx(result["metrics"]["test"]["loss"])
    assert result["metrics"]["val"]["loss"] == pytest.approx(history.val_loss.min())
    for check in result["gradient_checks"].values():
        assert check["relative_l2_error"] < 1e-6
