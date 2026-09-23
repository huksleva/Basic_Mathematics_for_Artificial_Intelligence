import math

from autograd_lab import Value, TanhNeuron, mean_squared_loss, train, assert_close


def test_leaf_value():
    v = Value(2)
    assert v.data == 2.0
    assert v.grad == 0.0
    assert v.parents == ()
    assert v.op == ""


def test_addition_forward():
    left, right = Value(2.5), Value(-1.0)
    result = left + right
    assert_close(result.data, 1.5)
    assert result.parents == (left, right)
    assert result.op == "+"


def test_multiplication_forward():
    left, right = Value(4.0), Value(-0.5)
    result = left * right
    scaled = 3.0 * left
    assert_close(result.data, -2.0)
    assert_close(scaled.data, 12.0)


def test_neg_sub_pow_forward():
    source = Value(3.0)
    assert_close((-source).data, -3.0)
    assert_close((5.0 - source).data, 2.0)
    assert_close((source ** 3).data, 27.0)


def test_elementary_functions_forward():
    assert_close(Value(0.0).sin().data, 0.0)
    assert_close(Value(2.0).exp().log().data, 2.0)
    assert_close(Value(0.7).tanh().data, math.tanh(0.7))


def test_shared_node_gradient():
    a = Value(2.0)
    objective = a * a + a
    objective.backward()
    assert_close(objective.data, 6.0)
    assert_close(a.grad, 5.0)
    assert_close(objective.grad, 1.0)


def test_finite_difference_check():
    def plain_function(x):
        return math.sin(x) * math.exp(x) + x ** 3

    point, step = 0.37, 1e-6
    input_value = Value(point)
    objective = input_value.sin() * input_value.exp() + input_value ** 3
    objective.backward()

    numerical_gradient = (
        plain_function(point + step) - plain_function(point - step)
    ) / (2 * step)

    assert abs(input_value.grad - numerical_gradient) < 1e-5


def test_neuron_forward():
    neuron = TanhNeuron(weight=2.0, bias=-1.0)
    prediction = neuron(0.5)
    assert_close(prediction.data, 0.0)
    assert neuron.parameters() == [neuron.weight, neuron.bias]


def test_training_converges():
    inputs = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0]
    targets = [-1.0, -1.0, -1.0, -1.0, 1.0, 1.0]

    neuron = TanhNeuron(weight=0.0, bias=0.0)
    initial_loss = mean_squared_loss(neuron, inputs, targets)
    assert_close(initial_loss.data, 1.0)

    neuron = TanhNeuron(weight=0.0, bias=0.0)
    losses = train(neuron, inputs, targets, learning_rate=0.2, epochs=80)

    predictions = [neuron(x).data for x in inputs]
    labels = [1.0 if p >= 0.0 else -1.0 for p in predictions]
    accuracy = sum(p == t for p, t in zip(labels, targets)) / len(targets)

    assert losses[-1] < 0.03
    assert accuracy == 1.0
