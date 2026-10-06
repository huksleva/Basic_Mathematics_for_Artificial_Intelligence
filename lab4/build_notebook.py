"""Сборка и выполнение учебного ноутбука в локальном окружении lab4."""
from pathlib import Path
import json
import nbformat as nbf
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager

ROOT = Path(__file__).resolve().parent
cells = []


def md(source):
    cells.append(nbf.v4.new_markdown_cell(source.strip()))


def code(source):
    cells.append(nbf.v4.new_code_cell(source.strip()))


md(r"""
# Лабораторная работа 4
## Полносвязная сеть (бинарный классификатор) своими руками на CPU
**Выполнил:** Тоц Леонид Александрович<br>
**Группа:** ИВТ-2<br>
**Дисциплина:** Базовая математика для искусственного интеллекта

Цель: реализовать MLP, прямой проход и backpropagation на NumPy, обучить сеть
mini-batches и оценить классификацию на независимом test. Готовые нейросетевые
модели и автоматическое дифференцирование не используются.

Выбран один из разрешённых датасетов: **Banknote Authentication**,
[UCI](https://archive.ics.uci.edu/dataset/267/banknote+authentication),
DOI: 10.24432/C55P57, Volker Lohweg, 2012, лицензия CC BY 4.0.
4 числовых признака: variance, skewness, curtosis, entropy; цель class ∈ {0, 1}.
Класс 1 считаем положительным без переименования исходных меток.

Открывайте ноутбук в папке lab4 вместе с mlp.py, experiment.py и data/.
В Google Colab первая ячейка автоматически загрузит код и данные из GitHub.
Все ячейки выполнены; для повторения выберите Run All.
""")
code("""
from pathlib import Path
import os
import sys
import subprocess

try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    repository = Path('/content/Basic_Mathematics_for_Artificial_Intelligence')
    if not repository.exists():
        subprocess.run(['git', 'clone', '--depth', '1', '--branch', 'main',
                        'https://github.com/huksleva/Basic_Mathematics_for_Artificial_Intelligence.git',
                        str(repository)], check=True)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r',
                    str(repository / 'lab4/requirements.txt')], check=True)
    os.chdir(repository / 'lab4')

ROOT = Path.cwd()
if not (ROOT / 'mlp.py').exists():
    ROOT = ROOT / 'lab4'
if not (ROOT / 'mlp.py').exists():
    raise FileNotFoundError('Откройте ноутбук в папке lab4 вместе с кодом и данными')
sys.path.insert(0, str(ROOT))

import json
import inspect
import numpy as np
import pandas as pd
from IPython.display import display, Image, Code
from mlp import (MLP, Standardizer, load_dataset, stratified_split,
                 gradient_check, select_threshold, classification_metrics)
from experiment import CONFIG, run_experiment

print('Автор: Тоц Леонид Александрович, ИВТ-2')
print('NumPy:', np.__version__, '| pandas:', pd.__version__, '| устройство: CPU')
""")
md(r"""
## 1. Данные, повторы и разделение
В исходном файле 1372 записи, пропусков нет. Полные дубликаты удаляем **до**
разбиения, иначе один и тот же объект может попасть в train и test.
Стратифицированное разбиение около 60/20/20 сохраняет доли классов.
Seed фиксирован; индексы исходных строк сохраняются в artifacts/split.csv.
""")
code("""
x, y, source_rows, audit = load_dataset(ROOT / 'data/data_banknote_authentication.txt')
split = stratified_split(y, seed=CONFIG['split_seed'])
display(pd.Series(audit, name='Аудит данных'))
display(pd.DataFrame([
    {'split': name, 'rows': len(idx), 'class_0': int((y[idx]==0).sum()),
     'class_1': int((y[idx]==1).sum()), 'fraction': len(idx)/len(y)}
    for name, idx in split.items()
]))
assert len(np.unique(np.concatenate(list(split.values())))) == len(y)
assert len(np.unique(x, axis=0)) == len(x)
print('OK: дубликаты удалены, части не пересекаются')
""")
md(r"""
## 2. Стандартизация без утечки
Для каждого признака $x'_j=(x_j-\mu_j)/s_j$.
Среднее и стандартное отклонение (ddof=0) вычисляем **только на train**.
На val/test применяем те же параметры; постоянный признак получает scale=1.
""")
code("""
scaler = Standardizer().fit(x[split['train']])
scaled = {name: scaler.transform(x[idx]) for name, idx in split.items()}
display(pd.DataFrame({'feature': ['variance','skewness','curtosis','entropy'],
                      'train_mean': scaler.mean_, 'train_std': scaler.scale_}))
np.testing.assert_allclose(scaled['train'].mean(axis=0), 0, atol=1e-12)
np.testing.assert_allclose(scaled['train'].std(axis=0), 1, atol=1e-12)
print('OK: стандартизация train проверена')
""")
md(r"""
## 3. Математика и реализация MLP
Объекты лежат **по строкам**. Для слоя $l$:

$$Z_l=A_{l-1}W_l+b_l,\quad A_l=\mathrm{ReLU}(Z_l),\quad
\hat p=\sigma(Z_L).$$

$$J=\frac1m\sum_i[\log(1+e^{z_i})-y_i z_i]
 +\frac{\lambda}{2}\sum_l\|W_l\|_F^2.$$

BCE вычисляется устойчиво через np.logaddexp. L2 применяется только к весам.
Выходная ошибка $D_L=(\hat p-y)/m$; скрытая ошибка
$D_l=(D_{l+1}W_{l+1}^T)\odot f'(Z_l)$.

$$\nabla W_l=A_{l-1}^TD_l+\lambda W_l,\quad
\nabla b_l=\sum_iD_{l,i}.$$

Усреднение по **фактическому** размеру mini-batch выполняется один раз.
Все градиенты вычисляются до изменения весов. Ниже приведён реальный код
обратного прохода из mlp.py; остальные методы доступны в том же модуле.
""")
code("""
display(Code(inspect.getsource(MLP.loss_and_gradients), language='python'))
model_preview = MLP(CONFIG['layer_sizes'], CONFIG['activation'], CONFIG['l2'], CONFIG['model_seed'])
print('Архитектура:', model_preview.layer_sizes)
print('Формы W:', [w.shape for w in model_preview.weights])
print('Параметров:', sum(w.size+b.size for w,b in zip(model_preview.weights, model_preview.biases)))
""")
md(r"""
## 4. Численная проверка backpropagation
Для каждого из 47 параметров маленькой сети 4→5→3→1 сравниваем ручной градиент
с центральной разностью $[J(\theta+\varepsilon)-J(\theta-\varepsilon)]/(2\varepsilon)$,
где $\varepsilon=10^{-6}$. Проверяем ReLU и tanh с L2=0.07.
Смещения ненулевые: проверка ReLU должна проводиться вдали от недифференцируемой точки $z=0$.
""")
code("""
checks = {}
for activation in ['relu', 'tanh']:
    tiny = MLP((4,5,3,1), activation=activation, l2=.07, seed=12)
    rng = np.random.default_rng(123)
    tiny.biases = [rng.uniform(.2,.5,b.shape) for b in tiny.biases]
    checks[activation] = gradient_check(tiny, scaled['train'][:7], y[split['train']][:7])
    assert checks[activation]['relative_l2_error'] < 1e-6
display(pd.DataFrame(checks).T)
print('OK: ручные градиенты совпадают с численными')
""")
md(r"""
## 5. Протокол эксперимента и обучение
Архитектура 4→32→16→1, ReLU в скрытых слоях и сигмоида на выходе.
Инициализация He для скрытых ReLU, Xavier для выходных логитов; bias=0.
SGD с моментом: $v\leftarrow\beta v+g$, $\theta\leftarrow\theta-\eta v$.
Максимум 150 эпох, batch=32, lr=0.03, momentum=0.9, L2=0.0001.
Ранняя остановка: patience=20, min_delta=0.00001 по val BCE;
восстанавливается абсолютный минимум val BCE, даже если обучение дошло до лимита.

После каждой эпохи история сохраняется в pandas.DataFrame:
epoch, loss, val_loss, accuracy, val_accuracy, objective.
loss/val_loss - чистая BCE; objective - train BCE + L2.
Accuracy в истории использует фиксированный порог 0.5.
test не используется ни для обучения, ни для выбора эпохи или порога.
""")
code("""
display(pd.Series(CONFIG, name='Гиперпараметры'))
result = run_experiment(ROOT / 'artifacts')
history = pd.read_csv(ROOT / 'artifacts/history.csv')
display(history.head())
display(history.tail())
print('Лучшая эпоха:', result['best_epoch'], '| завершение:', result['stop_epoch'])
print('Сработала ранняя остановка:', result['early_stopping_triggered'])
print('Восстановлены веса с минимальной val BCE')
""")
md(r"""
## 6. Кривые обучения
По графикам сравниваем train и val. Растущая ошибка val при падении train
указывала бы на переобучение. В данном запуске такого роста нет.
Лучшая эпоха отмечена вертикальной линией; значения истории относятся
к весам каждой эпохи, итоговые метрики - к восстановленному чекпойнту.
""")
code("""
display(Image(filename=str(ROOT / 'artifacts/learning_curves.png')))
display(history.iloc[[0,9,49,99,148,149]])
""")
md(r"""
## 7. Порог классификации
Решающее правило $\hat y=1[\hat p\ge\tau]$ используется при оценке,
а не при обучении. Выбираем максимум F1 **только на val**.
Проверяем середины между уникальными вероятностями и 0.5; при равном F1
берём ближайший к 0.5 порог, затем меньший. Здесь максимум F1=1 достижим
при 0.5, поэтому сохраняем $\tau=0.5$.
""")
code("""
trained, saved_scaler, tau = MLP.load(ROOT / 'artifacts/model.npz')
p_val = trained.predict_proba(saved_scaler.transform(x[split['val']]))
selected, sweep = select_threshold(y[split['val']], p_val)
assert selected == tau
display(sweep[sweep.f1 == sweep.f1.max()])
print('Выбранный порог:', tau)
print('Границы идеального разделения val:', p_val[y[split['val']]==0].max(),
      p_val[y[split['val']]==1].min())
""")
md(r"""
## 8. Независимая оценка test
Accuracy=(TP+TN)/N; Precision=TP/(TP+FP); Recall=TP/(TP+FN);
F1=2TP/(2TP+FP+FN). ROC-AUC не зависит от единственного порога.
Для PR используем **Average Precision** (ступенчатая сумма precision по
приращениям recall), а не трапециевидную площадь PR.
""")
code("""
summary = pd.DataFrame(result['metrics']).T
display(summary[['loss','accuracy','precision','recall','f1','roc_auc','average_precision']])
test = result['metrics']['test']
display(pd.DataFrame([[test['tn'],test['fp']],[test['fn'],test['tp']]],
                     index=['Истинный 0','Истинный 1'], columns=['Прогноз 0','Прогноз 1']))
display(Image(filename=str(ROOT / 'artifacts/roc_pr.png')))
display(Image(filename=str(ROOT / 'artifacts/confusion_threshold.png')))
""")
md(r"""
## 9. Сохранение модели и воспроизводимость
model.npz содержит W, b, архитектуру, L2, параметры стандартизации и порог.
test_predictions.csv содержит вероятности и прогнозы с исходными индексами.
history.csv, split.csv и results.json позволяют проверить эксперимент.
sklearn используется исключительно в независимых тестах метрик, а не в модели.
""")
code("""
predictions = pd.read_csv(ROOT / 'artifacts/test_predictions.csv')
raw = np.loadtxt(ROOT / 'data/data_banknote_authentication.txt', delimiter=',')
restored_p = trained.predict_proba(saved_scaler.transform(raw[predictions.source_row, :-1]))
np.testing.assert_allclose(restored_p, predictions.probability, atol=1e-14)
assert len(predictions) == 269
assert test['accuracy'] == 1.0
print('OK: сохранённая модель воспроизводит все прогнозы test')
print('SHA-256 исходного датасета:', result['sha256'])
display(pd.Series(result['environment'], name='Окружение'))
""")
md(r"""
## 10. Вывод
MLP и backpropagation реализованы вручную. После удаления 24 дубликатов данные
разделены на 809/270/269 объектов; стандартизация обучена только на train.
Обучение завершилось по лимиту 150 эпох; восстановлена лучшая эпоха 149.
При пороге 0.5 на test правильно классифицированы 269 из 269 объектов:
Accuracy=F1=ROC-AUC=Average Precision=1.0, BCE≈0.000446.

Это результат одного воспроизводимого разбиения небольшого датасета.
Он не гарантирует отсутствие ошибок на новых источниках данных. По кривым
не обнаружено ухудшения val, но по одному разбиению нельзя доказать
полное отсутствие переобучения. Для более устойчивой оценки можно повторить
эксперимент с другими seed или использовать кросс-валидацию.

Материалы: методичка «Лабораторная работа 4 - самописный MLP на CPU»,
«Протокол обучения MLP с сохранением истории (CPU)», «Логирование истории
обучения MLP», материалы по backpropagation и справочник метрик из задания.
Источник данных: [UCI Banknote Authentication](https://doi.org/10.24432/C55P57).
""")

notebook = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.13"},
    "authors": [{"name": "Тоц Леонид Александрович"}],
})
path = ROOT / "Тоц_ЛА_ИВТ-2_ЛР4_MLP.ipynb"
kernel_root = ROOT / "tmp" / "kernels"
spec = kernel_root / "lab4"
spec.mkdir(parents=True, exist_ok=True)
(spec / "kernel.json").write_text(json.dumps({
    "argv": [str(ROOT / ".venv/Scripts/python.exe"), "-m", "ipykernel_launcher", "-f", "{connection_file}"],
    "display_name": "Lab4 local", "language": "python",
    "env": {"IPYTHONDIR": str(ROOT / "tmp/ipython"),
            "JUPYTER_RUNTIME_DIR": str(ROOT / "tmp/jupyter_runtime")},
}), encoding="utf-8")
manager = KernelManager(kernel_name="lab4", kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernel_root)]))
nbf.write(notebook, path)
client = NotebookClient(notebook, timeout=180, kernel_name="lab4", km=manager,
                        resources={"metadata": {"path": str(ROOT)}})
client.execute()
nbf.validate(notebook)
nbf.write(notebook, path)
print("Выполнен ноутбук:", path.name)
print("Code cells:", sum(cell.cell_type == "code" for cell in notebook.cells))
