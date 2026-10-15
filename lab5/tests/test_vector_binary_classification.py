"""Математические проверки autograd, общих весов и фиксированных экспериментов."""

import math

import pytest

from experiment import CONFIG, INPUTS, TARGETS, new_model, train
from vector_binary_classification import (
    HiddenLayerBinaryClassifier,
    Value,
    affine,
    binary_cross_entropy,
    classification_accuracy,
    mean_binary_cross_entropy,
    numerical_gradient,
    predict_label,
    sigmoid,
)


def test_branching_graph_and_repeated_backward():
    a = Value(2.0)
    objective = a * a + a
    for _ in range(2):
        objective.backward()
        assert objective.data == 6.0
        assert a.grad == 5.0


@pytest.mark.parametrize("operation, expected", [
    (lambda x: x.sin(), math.cos(0.7)),
    (lambda x: x.exp(), math.exp(0.7)),
    (lambda x: x.log(), 1 / 0.7),
    (lambda x: x.tanh(), 1 - math.tanh(0.7) ** 2),
    (lambda x: 2 - x, -1),
    (lambda x: x ** -1, -0.7 ** -2),
])
def test_local_derivatives(operation, expected):
    x = Value(0.7)
    result = operation(x)
    result.backward()
    assert x.grad == pytest.approx(expected, abs=1e-12)


def test_affine_explicit_bias_path_and_feature_derivatives():
    weights = [Value(0.2), Value(0.3), Value(-0.4)]
    features = [Value(2), Value(3)]
    result = affine(weights, features)
    result.backward()
    assert result.data == pytest.approx(-0.4)
    assert [p.grad for p in weights] == [1, 2, 3]
    assert [x.grad for x in features] == [0.3, -0.4]
    bias_path = result.parents[0].parents[0]
    assert bias_path.op == "*"
    assert bias_path.parents[0] is weights[0]
    assert bias_path.parents[1].data == 1.0


@pytest.mark.parametrize("target", [0, 1])
def test_sigmoid_bce_chain_rule(target):
    logit = Value(0.7)
    p = sigmoid(logit)
    loss = binary_cross_entropy(p, target)
    loss.backward()
    expected_p = 1 / (1 + math.exp(-0.7))
    assert p.data == pytest.approx(expected_p)
    assert loss.data == pytest.approx(-math.log(expected_p if target else 1 - expected_p))
    assert logit.grad == pytest.approx(expected_p - target, abs=1e-12)


def test_shared_parameters_equal_analytic_dataset_gradient():
    model = new_model("OR")
    parameters = model.parameters()
    mean_binary_cross_entropy(model, INPUTS, TARGETS["OR"]).backward()
    for j, parameter in enumerate(parameters):
        expected = sum(
            (1 / (1 + math.exp(-(0.1 * x[0] - 0.2 * x[1]))) - y) * [1.0, *x][j]
            for x, y in zip(INPUTS, TARGETS["OR"])
        ) / 4
        assert parameter.grad == pytest.approx(expected, abs=1e-12)
        assert parameter is model.parameters()[j]
    assert all(not isinstance(x, Value) for row in INPUTS for x in row)


@pytest.mark.parametrize("task", CONFIG)
def test_every_parameter_against_central_difference(task):
    model = new_model(task)
    mean_binary_cross_entropy(model, INPUTS, TARGETS[task]).backward()
    assert len(model.parameters()) == CONFIG[task]["parameters"]
    for parameter in model.parameters():
        original = parameter.data
        numerical = numerical_gradient(model, parameter, INPUTS, TARGETS[task])
        assert parameter.grad == pytest.approx(numerical, abs=1e-7)
        assert parameter.data == original
    if task == "AND":
        assert model.hidden_weights[1].grad == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("task", CONFIG)
def test_fixed_training_protocol_reaches_required_quality(task):
    model = new_model(task)
    config = CONFIG[task]
    losses = train(model, INPUTS, TARGETS[task], config["learning_rate"], config["epochs"])
    assert len(losses) == config["epochs"]
    assert losses[-1] < config["loss_limit"]
    assert losses[-1] < losses[0]
    assert mean_binary_cross_entropy(model, INPUTS, TARGETS[task]).data < config["loss_limit"]
    assert classification_accuracy(model, INPUTS, TARGETS[task]) == 1.0
    if task == "AND":
        assert all(abs(p.grad) > 1e-8 for p in model.parameters())
    else:
        assert all(abs(p.data) > 0.1 for p in model.parameters())


def test_fully_symmetric_hidden_neurons_remain_symmetric():
    model = HiddenLayerBinaryClassifier([[0.1, 0.4, -0.3]] * 2, [0.0, 0.3, 0.3])
    train(model, INPUTS, TARGETS["XOR"], 0.5, 20)
    assert [p.data for p in model.hidden_weights[0]] == [p.data for p in model.hidden_weights[1]]
    assert model.output_weights[1].data == model.output_weights[2].data


def test_threshold_includes_half_in_positive_class():
    assert predict_label(Value(0.4999)) == 0
    assert predict_label(Value(0.5)) == 1


def test_shape_and_target_contracts():
    with pytest.raises(AssertionError):
        affine([Value(1), Value(2)], [1, 2])
    with pytest.raises(AssertionError):
        mean_binary_cross_entropy(new_model("OR"), [], [])
    with pytest.raises(AssertionError):
        mean_binary_cross_entropy(new_model("OR"), INPUTS, [0])
    with pytest.raises(AssertionError):
        binary_cross_entropy(Value(0.5), 2)
    with pytest.raises(AssertionError):
        HiddenLayerBinaryClassifier([[0, 1, 2], [0, 1]], [0, 1, 1])
    with pytest.raises(AssertionError):
        HiddenLayerBinaryClassifier([[0, 1, 2]], [0, 1, 1])
