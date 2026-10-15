"""ЛР5. Скалярный autograd и бинарные классификаторы по выданной методичке.

Вычисления используют только стандартную библиотеку. Полный Value повторяет
предыдущую лабораторную; каждый входной вектор раскрывается в скалярный DAG.
Источник учебной реализации: Dmitry Vlasov, 2026, lab_guide.pdf.
"""

import math


def assert_close(actual, expected, tolerance=1e-9):
    difference = abs(actual - expected)
    assert difference <= tolerance, (
        f"Ожидалось {expected}, получено {actual}, "
        f"абсолютная ошибка {difference}"
    )


class Value:
    def __init__(self, data, parents=(), op=""):
        self.data = float(data)
        self.grad = 0.0
        self.parents = parents
        self.op = op
        self._backward = lambda: None

    def __repr__(self):
        return (
            f"Value(data={self.data:.6g}, grad={self.grad:.6g}, "
            f"op={self.op!r})"
        )

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        result = Value(
            self.data + other.data,
            parents=(self, other),
            op="+",
        )

        def backward():
            self.grad += result.grad
            other.grad += result.grad

        result._backward = backward
        return result

    __radd__ = __add__

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        result = Value(
            self.data * other.data,
            parents=(self, other),
            op="*",
        )

        def backward():
            self.grad += result.grad * other.data
            other.grad += result.grad * self.data

        result._backward = backward
        return result

    __rmul__ = __mul__

    def __neg__(self):
        result = Value(-self.data, parents=(self,), op="neg")

        def backward():
            self.grad += result.grad * (-1)

        result._backward = backward
        return result

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __pow__(self, power):
        result = Value(
            self.data ** power,
            parents=(self,),
            op=f"pow({power})",
        )

        def backward():
            local_derivative = power * self.data ** (power - 1)
            self.grad += result.grad * local_derivative

        result._backward = backward
        return result

    def sin(self):
        result = Value(math.sin(self.data), parents=(self,), op="sin")

        def backward():
            self.grad += result.grad * math.cos(self.data)

        result._backward = backward
        return result

    def exp(self):
        result = Value(math.exp(self.data), parents=(self,), op="exp")

        def backward():
            self.grad += result.grad * result.data

        result._backward = backward
        return result

    def log(self):
        result = Value(math.log(self.data), parents=(self,), op="log")

        def backward():
            self.grad += result.grad * (1 / self.data)

        result._backward = backward
        return result

    def tanh(self):
        result = Value(math.tanh(self.data), parents=(self,), op="tanh")

        def backward():
            self.grad += result.grad * (1 - result.data ** 2)

        result._backward = backward
        return result

    def backward(self):
        topo = []
        visited = set()

        def build(node):
            if node not in visited:
                visited.add(node)
                for parent in node.parents:
                    build(parent)
                topo.append(node)

        build(self)

        for node in topo:
            node.grad = 0.0
        self.grad = 1.0

        for node in reversed(topo):
            node._backward()


def affine(weights, features):
    # Сумма по расширенным координатам: w_0 * 1 + w_1 * x_1 + ...
    augmented_features = [1.0, *features]
    assert len(weights) == len(augmented_features), (
        "Число весов должно включать вес постоянной координаты"
    )

    total = weights[0] * augmented_features[0]
    for weight, feature in zip(weights[1:], augmented_features[1:]):
        total = total + weight * feature
    return total


def sigmoid(logit):
    return (1.0 + (-logit).exp()) ** -1


def binary_cross_entropy(probability, target):
    target = float(target)
    assert target in (0.0, 1.0), "Метка бинарной классификации должна быть 0 или 1"
    return -(
        target * probability.log()
        + (1.0 - target) * (1.0 - probability).log()
    )


def mean_binary_cross_entropy(model, inputs, targets):
    assert len(inputs) == len(targets), "Число объектов и меток должно совпадать"
    assert len(inputs) > 0, "Нужен хотя бы один объект"

    total = Value(0.0)
    for features, target in zip(inputs, targets):
        probability = model(features)
        total = total + binary_cross_entropy(probability, target)
    return total * (1.0 / len(inputs))


def predict_label(probability):
    return 1 if probability.data >= 0.5 else 0


def classification_accuracy(model, inputs, targets):
    predictions = [predict_label(model(features)) for features in inputs]
    correct = sum(
        prediction == target
        for prediction, target in zip(predictions, targets)
    )
    return correct / len(targets)


def numerical_gradient(model, parameter, inputs, targets, step=1e-6):
    original_data = parameter.data
    parameter.data = original_data + step
    loss_plus = mean_binary_cross_entropy(model, inputs, targets).data
    parameter.data = original_data - step
    loss_minus = mean_binary_cross_entropy(model, inputs, targets).data
    parameter.data = original_data
    return (loss_plus - loss_minus) / (2.0 * step)


class VectorBinaryNeuron:
    def __init__(self, weights):
        assert len(weights) > 1, "Нужны вес смещения и хотя бы один входной вес"
        self.weights = [Value(weight) for weight in weights]

    def __call__(self, features):
        logit = affine(self.weights, features)
        return sigmoid(logit)

    def parameters(self):
        return self.weights


class OneHiddenBinaryClassifier:
    def __init__(self, hidden_weights, output_weights):
        assert len(hidden_weights) > 1, "Нужны вес смещения и хотя бы один входной вес"
        assert len(output_weights) == 2, "Выходу нужны вес смещения и вес скрытого состояния"
        self.hidden_weights = [Value(weight) for weight in hidden_weights]
        self.output_weights = [Value(weight) for weight in output_weights]

    def hidden_state(self, features):
        hidden_logit = affine(self.hidden_weights, features)
        return hidden_logit.tanh()

    def __call__(self, features):
        hidden = self.hidden_state(features)
        output_logit = affine(self.output_weights, [hidden])
        return sigmoid(output_logit)

    def parameters(self):
        return self.hidden_weights + self.output_weights


class HiddenLayerBinaryClassifier:
    def __init__(self, hidden_layer_weights, output_weights):
        assert len(hidden_layer_weights) > 0, "Нужен хотя бы один скрытый нейрон"
        feature_count = len(hidden_layer_weights[0])
        assert feature_count > 1, "У скрытого нейрона нужны смещение и хотя бы один вход"
        assert all(
            len(weights) == feature_count
            for weights in hidden_layer_weights
        ), "Все скрытые нейроны должны иметь одинаковое число входов"
        assert len(output_weights) == len(hidden_layer_weights) + 1, (
            "Выходу нужны вес смещения и веса всех скрытых состояний"
        )
        self.hidden_weights = [
            [Value(weight) for weight in weights]
            for weights in hidden_layer_weights
        ]
        self.output_weights = [Value(weight) for weight in output_weights]

    def hidden_states(self, features):
        return [
            affine(weights, features).tanh()
            for weights in self.hidden_weights
        ]

    def __call__(self, features):
        hidden = self.hidden_states(features)
        output_logit = affine(self.output_weights, hidden)
        return sigmoid(output_logit)

    def parameters(self):
        parameters = []
        for weights in self.hidden_weights:
            parameters.extend(weights)
        parameters.extend(self.output_weights)
        return parameters
