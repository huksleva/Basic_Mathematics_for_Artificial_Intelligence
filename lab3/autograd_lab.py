"""
Скалярный автоматический дифференциатор и однослойный tanh-нейрон.

Этот модуль — тот же код, что заполняется по шагам в student_autograd_lab.ipynb
согласно lab_guide.pdf. Вынесен в отдельный файл, чтобы его можно было
импортировать в pytest-тесты (см. tests/test_autograd_lab.py) и переиспользовать
без Jupyter.
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


class TanhNeuron:
    def __init__(self, weight, bias):
        self.weight = Value(weight)
        self.bias = Value(bias)

    def __call__(self, x):
        return (self.weight * x + self.bias).tanh()

    def parameters(self):
        return [self.weight, self.bias]


def mean_squared_loss(neuron, inputs, targets):
    total = Value(0.0)
    for input_number, target_number in zip(inputs, targets):
        prediction = neuron(input_number)
        error = prediction - target_number
        total = total + error ** 2
    return total * (1.0 / len(inputs))


def train(neuron, inputs, targets, learning_rate=0.2, epochs=80):
    """Возвращает список значений loss по эпохам (обучает neuron on-place)."""
    losses = []
    for _ in range(epochs):
        loss = mean_squared_loss(neuron, inputs, targets)
        loss.backward()
        losses.append(loss.data)
        for parameter in neuron.parameters():
            parameter.data -= learning_rate * parameter.grad
    return losses
