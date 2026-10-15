"""Заполнение и выполнение выданной заготовки с сохранением идентификаторов ячеек.

Обычная пересборка: python build_notebook.py. Для первоначальной сборки можно
указать --template path/to/student_vector_binary_classification_lab.ipynb.txt.
Полученный notebook содержит весь код и не импортирует модули проекта.
"""

import argparse
import inspect
import json
from pathlib import Path
import sys
from textwrap import dedent

import nbformat as nbf
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager

import vector_binary_classification as implementation
from experiment import CONFIG, run_experiment

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "student_vector_binary_classification_lab.ipynb"
MARKER = "# VECTOR_BINARY_CLASSIFICATION_ESSENTIALS_CELL: "


def source(*names):
    return "\n\n".join(inspect.getsource(getattr(implementation, name)).strip() for name in names)


def model_expression(task):
    return {
        "OR": "VectorBinaryNeuron(weights=[0.0, 0.1, -0.2])",
        "AND": """OneHiddenBinaryClassifier(
    hidden_weights=[0.0, 0.1, -0.2],
    output_weights=[0.0, 0.3],
)""",
        "XOR": """HiddenLayerBinaryClassifier(
    hidden_layer_weights=[[0.1, 0.4, -0.3], [-0.1, -0.2, 0.5]],
    output_weights=[0.0, 0.3, -0.4],
)""",
    }[task]


def gradient_cell(task):
    name = task.lower()
    prefix = ""
    if task != "OR":
        targets = {"AND": [0, 0, 0, 1], "XOR": [0, 1, 1, 0]}[task]
        prefix = f"{task}_INPUTS = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]\n"
        prefix += f"{task}_TARGETS = {targets!r}\n\n"
    body = f"gradient_{name} = {model_expression(task)}\n"
    body += dedent(f"""
        {name}_loss = mean_binary_cross_entropy(gradient_{name}, {task}_INPUTS, {task}_TARGETS)
        {name}_loss.backward()
        {name}_gradient_errors = []
        for parameter in gradient_{name}.parameters():
            expected_gradient = numerical_gradient(
                gradient_{name}, parameter, {task}_INPUTS, {task}_TARGETS,
            )
            assert_close(parameter.grad, expected_gradient, tolerance=1e-5)
            {name}_gradient_errors.append(abs(parameter.grad - expected_gradient))
        assert len(gradient_{name}.parameters()) == {CONFIG[task]['parameters']}
        assert any(abs(parameter.grad) > 1e-8 for parameter in gradient_{name}.parameters())
    """)
    if task == "AND":
        body += "assert_close(gradient_and.hidden_weights[1].grad, 0.0)\n"
    body += dedent(f"""
        print("BCE для {task} до обучения:", {name}_loss.data)
        print("Градиенты {task}:", [parameter.grad for parameter in gradient_{name}.parameters()])
        print("Максимальная ошибка центральной разности:", max({name}_gradient_errors))
    """)
    return prefix + body


def training_cell(task):
    name = task.lower()
    config = CONFIG[task]
    body = f"learning_rate = {config['learning_rate']}\nepochs = {config['epochs']}\n"
    body += f"trained_{name} = {model_expression(task)}\n"
    body += dedent(f"""
        {name}_losses = []
        for epoch in range(epochs):
            loss = mean_binary_cross_entropy(trained_{name}, {task}_INPUTS, {task}_TARGETS)
            loss.backward()
            {name}_losses.append(loss.data)
            for parameter in trained_{name}.parameters():
                parameter.data -= learning_rate * parameter.grad

        {name}_predictions = [trained_{name}(features).data for features in {task}_INPUTS]
        {name}_accuracy = classification_accuracy(trained_{name}, {task}_INPUTS, {task}_TARGETS)
        assert {name}_losses[-1] < {config['loss_limit']}, {name}_losses[-1]
        assert {name}_losses[-1] < {name}_losses[0]
        assert {name}_accuracy == 1.0, {name}_accuracy
    """)
    if task == "AND":
        body += "assert all(abs(parameter.grad) > 1e-8 for parameter in trained_and.parameters())\n"
    else:
        body += f"assert all(abs(parameter.data) > 0.1 for parameter in trained_{name}.parameters())\n"
    body += dedent(f"""
        print("{task}: loss в первой эпохе:", {name}_losses[0])
        print("{task}: loss в последней эпохе (до обновления):", {name}_losses[-1])
        print("{task}: вероятности после всех обновлений:", {name}_predictions)
        print("{task}: точность:", {name}_accuracy)
    """)
    return body


ANSWERS = r"""
## Ответы на вопросы для самопроверки

**1. Какие вершины создаёт `affine` для двух признаков?**

Даны три общих листа $w_0,w_1,w_2$ и константные листья $1,x_1,x_2$.
Возникают три узла умножения $q_0=w_0\cdot1$, $q_1=w_1x_1$, $q_2=w_2x_2$,
затем два сложения: $s=q_0+q_1$, $o=s+q_2$.
Путь смещения существует в DAG как настоящее умножение на единицу,
поэтому $\partial o/\partial w_0=1$ получается общим правилом.
Каждый вызов создаёт новые константы и промежуточные узлы, используя прежние веса.

**2. Почему одна ссылка на `weights[0]` даёт сумму вкладов по выборке?**

Все четыре прямых прохода ссылаются на один и тот же объект веса.
Средняя BCE объединяет их потери в один DAG. Каждый потомок добавляет свой вклад
через `+=`, и вес получает
$\partial L/\partial w_j=\frac14\sum_{\mu=1}^4(p_\mu-y_\mu)\tilde x_{\mu j}$.
Деление на 4 выполняется один раз при вычислении средней потери.
`features` и метки являются фиксированными данными; они не входят в `parameters()`.
Даже если внутренние константные листья получают `grad`, цикл обучения обновляет
только возвращённые моделью веса.

**3. Какой узел хранит вероятность класса 1?**

Выходной `sigmoid`: $p=\sigma(o)\in(0,1)$. Скрытый `tanh` хранит характеристику
$h\in(-1,1)$ и не является вероятностью. Метка равна 1 при $p\ge0.5$, иначе 0.

**4. Почему BCE вместе с sigmoid даёт градиент $p-y$?**

$$\frac{\partial\ell}{\partial p}
=-\frac y p+\frac{1-y}{1-p}=\frac{p-y}{p(1-p)},\qquad
\frac{\partial p}{\partial o}=p(1-p).$$

По цепному правилу множители сокращаются:
$\partial\ell/\partial o=p-y$. Для средней потери вклад одного объекта
равен $(p_i-y_i)/n$. В коде результат получается обратным обходом скалярных
операций, а не подстановкой этой формулы. Accuracy зависит от пороговых меток,
поэтому почти всюду локально постоянна и не даёт полезного градиента.

**5. Что нарушится при обновлении `parameter.data` до `loss.backward()`?**

Сохранённые результаты прямого прохода относятся к старым весам,
а замыкания некоторых операций читают текущее `data` родителей.
Возникнет смесь состояний, и градиент не будет производной построенной потери.
Правильный порядок: свежий DAG → `backward()` → обновление всех параметров.

**6. Почему одинаковые скрытые нейроны не приобретают разные роли?**

При полной симметрии, то есть одинаковых скрытых весах **и одинаковых весах
соединения с выходом**, оба нейрона дают одинаковые состояния и градиенты.
Детерминированный градиентный спуск сохраняет это равенство, поэтому два нейрона
работают как один. Важно учитывать выходные связи:
$g_{r_{ih}}=g_{o_i}v_h(1-h_{ih}^2)$; при разных $v_h$ одинаковые скрытые веса
сами по себе не гарантируют одинаковых градиентов. В фиксированном XOR-опыте
различаются и начальные скрытые веса, и выходные связи.

**7. Почему XOR является тестом выразительности?**

OR проверяет обучение одной линейной границы, AND со скрытым нейроном проверяет
цепное правило, а XOR требует нескольких нелинейных характеристик.
Для одного скрытого нейрона $p=\sigma(v_0+v_1\tanh(u_0+u_1x_1+u_2x_2))$.
Монотонность функций превращает порог $p\ge0.5$ в полуплоскость или постоянный
ответ. Ни одна полуплоскость не разделяет XOR. Действительно, для любого
$f(x)=c_0+c_1x_1+c_2x_2$:

$$f(0,1)+f(1,0)=f(0,0)+f(1,1).$$

Для положительных точек левая сумма должна быть неотрицательной, а для
отрицательных правая должна быть строго отрицательной, что невозможно.
Два скрытых `tanh` создают две разные характеристики, и выход комбинирует их.
Четыре логические точки позволяют проверить эту возможность; они не оценивают
качество прикладного классификатора на новых данных.

## Связь с MLP

Один скрытый слой можно записать компактно:
$r_i=U\tilde x_i$, $h_i=\tanh(r_i)$, $p_i=\sigma(v^T\tilde h_i)$.
В XOR $U$ имеет размер $2\times3$, а $v$ содержит три веса с учётом смещения.
В данном коде каждая такая формула раскрыта в скалярные операции. Для MLP
аффинное преобразование и нелинейность повторяются по слоям; обратный проход
по-прежнему использует цепное правило и суммирование вкладов в общих параметрах.
"""


def build(template):
    notebook = nbf.read(template, as_version=4)
    results = run_experiment(ROOT / "artifacts")
    cells = {
        "imports_and_helper": "import math\n\n" + source("assert_close"),
        "value": source("Value"),
        "scalar_graph_refresh": dedent("""
            a = Value(2.0)
            objective = a * a + a
            objective.backward()
            assert_close(objective.data, 6.0)
            assert_close(a.grad, 5.0)
            print("Скалярный Value готов: J =", objective.data, ", dJ/da =", a.grad)
        """),
        "vector_helpers": source("affine", "sigmoid", "binary_cross_entropy",
                                 "mean_binary_cross_entropy", "predict_label",
                                 "classification_accuracy", "numerical_gradient"),
        "vector_neuron": source("VectorBinaryNeuron"),
        "or_forward": dedent("""
            OR_INPUTS = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
            OR_TARGETS = [0, 1, 1, 1]
            or_neuron = VectorBinaryNeuron(weights=[0.0, 0.1, -0.2])
            or_probabilities = [or_neuron(features).data for features in OR_INPUTS]
            assert_close(or_probabilities[0], 0.5)
            assert_close(or_probabilities[1], 1.0 / (1.0 + math.exp(0.2)))
            assert len(or_neuron.parameters()) == 3
            assert all(isinstance(parameter, Value) for parameter in or_neuron.parameters())
            print("Начальные вероятности OR:", or_probabilities)
        """),
        "one_hidden_model": source("OneHiddenBinaryClassifier"),
        "xor_model": source("HiddenLayerBinaryClassifier"),
    }
    for task in CONFIG:
        cells[f"{task.lower()}_gradients"] = gradient_cell(task)
        cells[f"train_{task.lower()}"] = training_cell(task)

    markdown = {
        "markdown-00": """# Лабораторная работа №5: векторный вход и бинарная классификация

**Выполнил:** Тоц Леонид Александрович<br>
**Группа:** ИВТ-2<br>
**Дисциплина:** Базовая математика для искусственного интеллекта<br>
**Дата:** 15.10.2026

Цель: построить классификаторы OR, AND и XOR на скалярном `Value`, проверить
градиенты и объяснить роль скрытого слоя. Все вычисления используют только
стандартную библиотеку Python. Ноутбук самостоятельный: запустите все ячейки
сверху вниз в Python 3.10+ или Google Colab; другие файлы и загрузки не нужны.

Сохранены порядок и идентификаторы ячеек выданной student-заготовки.
Реализация следует методичке Dmitry Vlasov (2026); исходные `lab_guide.pdf`
и `theory.pdf` находятся рядом без изменений. Пояснения, численные проверки,
сводка и ответы на вопросы включены в эту выполненную работу.
""",
        "markdown-01": """## Подготовка: скалярный граф и вектор признаков

Полный `Value` сохранён из предыдущей лабораторной. Он хранит одно число,
его градиент, родителей и локальное правило обратного прохода.
Вектор признаков остаётся списком обычных чисел и раскрывается в скалярный DAG.
""",
        "markdown-02": """### Проверка скалярного механизма

Для $J=a^2+a$ при $a=2$ ожидаем $J=6$ и $dJ/da=2a+1=5$.
Повторное использование листа проверяет накопление независимых вкладов через `+=`.
""",
        "markdown-03": r"""### Векторные вспомогательные функции

`affine` вычисляет $w_0\cdot1+w_1x_1+w_2x_2$: нулевой вес является смещением.
Сырые признаки единицу не содержат; её добавляет сама функция.
В формуле $o_i=\sum_{\alpha=0}^d w_\alpha\tilde x_{i\alpha}$ индекс $i$
фиксирует объект, а $\alpha$ суммируется. В Python это цикл `zip` по координатам.
Сумма средней BCE по объектам использует отдельный немой индекс $\mu$.

`sigmoid` превращает логит в вероятность, BCE измеряет отрицательный логарифм
вероятности правильной метки, а accuracy служит диагностикой при пороге 0.5.
Производные проверяются центральной разностью с шагом $10^{-6}$.
""",
        "markdown-04": """## Этап 1. Один sigmoid-нейрон с векторным входом

Три долгоживущих `Value` задают два входных веса и смещение.
`parameters()` возвращает ссылки на эти объекты; признаки остаются данными.
""",
        "markdown-05": """### Прямой проход OR

Начальные веса: `[0.0, 0.1, -0.2]`. Проверяем размерность и вероятности до обучения.
Для нулевого входа логит равен нулю, а вероятность равна 0.5.
""",
        "markdown-06": r"""### Градиенты OR

Проверены все три веса. Для сравнения с autograd используется центральная разность;
аналитическая формула $\nabla_{w_j}L=\frac14\sum_{\mu=1}^4
(p_\mu-y_\mu)\tilde x_{\mu j}$ дополнительно проверяется в итоговой ячейке.
""",
        "markdown-07": """### Обучение OR

Полный батч из четырёх объектов, 300 эпох, шаг 0.5.
Каждая эпоха создаёт свежий граф, вычисляет градиенты и только затем обновляет веса.
Критерии: BCE < 0.1 и accuracy = 1.0.
""",
        "markdown-08": r"""## Этап 2. Один скрытый нейрон и один выходной нейрон

$h_i=\tanh(u_0+u_1x_{i1}+u_2x_{i2})$, $p_i=\sigma(v_0+v_1h_i)$.
Всего пять параметров. AND линейно разделима; скрытый нейрон здесь позволяет
проследить градиент через композицию двух аффинных преобразований и `tanh`.
""",
        "markdown-09": r"""### Градиенты AND

Проверены все пять параметров, включая скрытые веса.
При заданной инициализации градиент `hidden_weights[1]` равен нулю из-за
точной компенсации вкладов объектов, что подтверждает и центральная разность.
Это не означает обрыва графа. Путь к скрытому логиту задаётся формулой
$g_{r_i}=\frac{p_i-y_i}{n}v_1(1-h_i^2)$.
""",
        "markdown-10": """### Обучение AND

300 эпох, шаг 0.3, фиксированные начальные веса из методички.
Критерии: BCE < 0.05, accuracy = 1.0; в последней эпохе все пять градиентов ненулевые.
""",
        "markdown-11": """### Зачем в маршруте нужен XOR

Один скрытый `tanh` и монотонный sigmoid-выход задают одну линейную границу
либо постоянный прогноз. XOR такой границей не разделяется.
Доказательство приведено в ответе на вопрос 7.
""",
        "markdown-12": """## Этап 3. Скрытый слой из двух нейронов и XOR

Два скрытых нейрона создают разные нелинейные характеристики. Выходной
sigmoid-нейрон объединяет их. Всего девять весов: 2 × 3 скрытых и 3 выходных.
Класс допускает произвольное положительное число скрытых нейронов;
в обязательном эксперименте используются ровно два.
""",
        "markdown-13": """### Градиенты XOR

Центральной разностью проверены все девять параметров до обучения.
Два пути от скрытых состояний к выходу входят в единый скалярный DAG.
""",
        "markdown-14": """### Обучение XOR

500 эпох, шаг 0.5. Сохранена заданная асимметричная инициализация:
скрытые веса `[[0.1, 0.4, -0.3], [-0.1, -0.2, 0.5]]`, выходные `[0.0, 0.3, -0.4]`.
Критерии: BCE < 0.03, accuracy = 1.0.
""",
        "markdown-15": """## Диагностика и интерпретация

Все обязательные проверки пройдены. Порог 0.5 применяется после sigmoid.
Градиенты обнуляются внутри `backward()` при каждом новом обратном проходе.
Веса сохраняют идентичность между эпохами, промежуточные DAG создаются заново.
Результаты относятся к четырём точкам таблиц истинности: отдельного test здесь нет.
""",
        "markdown-16": """## Чек-лист перед сдачей

- [x] Все обязательные `assert` пройдены.
- [x] Центральная разность совпала с autograd для OR, AND и XOR (3 + 5 + 9 весов).
- [x] Объяснено, почему признаки не входят в `parameters()`.
- [x] Показано использование одного параметра всеми объектами выборки.
- [x] Получены требуемые BCE и accuracy для трёх моделей.
- [x] Объяснено ограничение одного скрытого нейрона на XOR.
- [x] Даны письменные ответы на все семь вопросов.
""" + "\n" + ANSWERS,
    }
    # Удаляем только добавленные сборщиком итоговые ячейки при пересборке.
    notebook.cells = [cell for cell in notebook.cells if not cell.id.startswith("lab5-extra-")]
    found = set()
    for cell in notebook.cells:
        if cell.cell_type == "code":
            key = cell.id.removeprefix("code-")
            cell.source = MARKER + key + "\n" + cells[key].strip() + "\n"
            cell.outputs = []
            cell.execution_count = None
            found.add(key)
        elif cell.id in markdown:
            cell.source = markdown[cell.id].strip()
    assert found == set(cells), (found, set(cells))

    verification = r"""
print("Проверка аналитического градиента OR и общих параметров:")
for j, parameter in enumerate(gradient_or.parameters()):
    expected = sum(
        (gradient_or(x).data - y) * [1.0, *x][j]
        for x, y in zip(OR_INPUTS, OR_TARGETS)
    ) / len(OR_INPUTS)
    assert_close(parameter.grad, expected)
    assert parameter is gradient_or.parameters()[j]
    print(f"w{j}: autograd={parameter.grad:.12f}, аналитически={expected:.12f}")

print("\nИтог после всех обновлений (свежий прямой проход):")
print(f"{'Задача':<8} {'BCE':>12} {'Accuracy':>10} {'Верные метки':>18}")
for task, model, inputs, targets, limit in [
    ('OR', trained_or, OR_INPUTS, OR_TARGETS, 0.1),
    ('AND', trained_and, AND_INPUTS, AND_TARGETS, 0.05),
    ('XOR', trained_xor, XOR_INPUTS, XOR_TARGETS, 0.03),
]:
    final_loss = mean_binary_cross_entropy(model, inputs, targets).data
    labels = [predict_label(model(x)) for x in inputs]
    accuracy = classification_accuracy(model, inputs, targets)
    assert final_loss < limit and accuracy == 1.0 and labels == targets
    print(f"{task:<8} {final_loss:>12.9f} {accuracy:>10.1f} {str(labels):>18}")
    print("  Веса:", [round(p.data, 9) for p in model.parameters()])
print("OK: все этапы и аналитическая проверка выполнены")
"""
    notebook.cells.append(nbf.v4.new_code_cell(verification.strip(), id="lab5-extra-verification"))
    summary = "## Итоговые результаты\n\n"
    summary += "| Задача | Параметров | Эпох | Шаг | Начальная BCE | Итоговая BCE | Accuracy |\n"
    summary += "|---|---:|---:|---:|---:|---:|---:|\n"
    for task, result in results.items():
        summary += (f"| {task} | {result['parameters']} | {result['epochs']} | "
                    f"{result['learning_rate']} | {result['initial_loss']:.9f} | "
                    f"{result['final_loss_after_update']:.9f} | {result['accuracy']:.1f} |\n")
    summary += """
Итоговая BCE вычислена после последнего обновления. Последний элемент истории
содержит BCE перед этим обновлением, как в выданном цикле обучения.

Один sigmoid-нейрон выучил OR. AND подтвердил прохождение градиента через
скрытый `tanh`. Два разных скрытых нейрона позволили решить XOR.
Механизм `Value` остался скалярным на всех этапах.
Все результаты воспроизводятся без случайной инициализации и внешних датасетов.
"""
    notebook.cells.append(nbf.v4.new_markdown_cell(summary.strip(), id="lab5-extra-summary"))
    notebook.metadata.update({
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "authors": [{"name": "Тоц Леонид Александрович"}],
    })

    kernel_root = ROOT / "tmp/kernels"
    spec = kernel_root / "lab5"
    spec.mkdir(parents=True, exist_ok=True)
    (spec / "kernel.json").write_text(json.dumps({
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "Lab5 local", "language": "python",
        "env": {"IPYTHONDIR": str(ROOT / "tmp/ipython"),
                "JUPYTER_RUNTIME_DIR": str(ROOT / "tmp/jupyter_runtime")},
    }), encoding="utf-8")
    manager = KernelManager(
        kernel_name="lab5",
        kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernel_root)]),
    )
    NotebookClient(notebook, timeout=180, kernel_name="lab5", km=manager,
                   resources={"metadata": {"path": str(ROOT)}}).execute()
    # Оставляем стандартное переносимое имя ядра в сохранённом notebook.
    notebook.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nbf.validate(notebook)
    nbf.write(notebook, OUTPUT)
    print(f"Выполнен и сохранён {OUTPUT.name}; code cells: "
          f"{sum(cell.cell_type == 'code' for cell in notebook.cells)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, default=OUTPUT)
    build(parser.parse_args().template)
