"""PDF-отчёт на основе реальных результатов эксперимента ЛР4."""
from pathlib import Path
import json
import html
import pandas as pd
import numpy as np
import matplotlib
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                               Image, PageBreak, KeepTogether)
from mlp import MLP, load_dataset, stratified_split

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'output/pdf/Тоц_ЛА_ИВТ-2_ЛР4_Отчёт.pdf'
RESULT = json.loads((ROOT / 'artifacts/results.json').read_text(encoding='utf-8'))
HISTORY = pd.read_csv(ROOT / 'artifacts/history.csv')
pdfmetrics.registerFont(TTFont('Arial', 'C:/Windows/Fonts/arial.ttf'))
pdfmetrics.registerFont(TTFont('ArialBold', 'C:/Windows/Fonts/arialbd.ttf'))
pdfmetrics.registerFont(TTFont('Consolas', 'C:/Windows/Fonts/consola.ttf'))
pdfmetrics.registerFont(TTFont('Math', str(Path(matplotlib.get_data_path()) / 'fonts/ttf/DejaVuSans.ttf')))
pdfmetrics.registerFontFamily('Arial', normal='Arial', bold='ArialBold', italic='Arial', boldItalic='ArialBold')
NAVY, BLUE, GRAY = colors.HexColor('#172554'), colors.HexColor('#2563eb'), colors.HexColor('#475569')
styles = getSampleStyleSheet()
styles.add(ParagraphStyle('BodyRu', fontName='Arial', fontSize=10.5, leading=15,
                         textColor=NAVY, spaceAfter=9))
styles.add(ParagraphStyle('TitleRu', fontName='ArialBold', fontSize=23, leading=28,
                         textColor=NAVY, spaceAfter=13))
styles.add(ParagraphStyle('H1Ru', fontName='ArialBold', fontSize=17, leading=22,
                         textColor=NAVY, spaceAfter=13))
styles.add(ParagraphStyle('H2Ru', fontName='ArialBold', fontSize=12, leading=17,
                         textColor=BLUE, spaceBefore=10, spaceAfter=7))
styles.add(ParagraphStyle('SmallRu', fontName='Arial', fontSize=8.4, leading=11.5,
                         textColor=GRAY, spaceAfter=7))
styles.add(ParagraphStyle('FormulaRu', fontName='Arial', fontSize=11.5, leading=18,
                         textColor=NAVY, leftIndent=12, spaceAfter=10))
styles.add(ParagraphStyle('CellRu', fontName='Arial', fontSize=9.3, leading=12.5, textColor=NAVY))
styles.add(ParagraphStyle('CodeRu', fontName='Consolas', fontSize=9, leading=12.5,
                         textColor=NAVY, leftIndent=10, spaceAfter=4))
story = []


def p(text, style='BodyRu'):
    for symbol in ('∈', '⊙', '∇'):
        text = text.replace(symbol, f'<font name="Math">{symbol}</font>')
    story.append(Paragraph(text, styles[style]))


def heading(text):
    p(text, 'H1Ru')


def sub(text):
    p(text, 'H2Ru')


def table(rows, widths):
    content = [[Paragraph(str(value), styles['CellRu']) for value in row] for row in rows]
    t = Table(content, colWidths=widths, repeatRows=1, hAlign='LEFT')
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#dbeafe')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8fafc')]),
        ('LINEBELOW', (0, 0), (-1, 0), .7, colors.HexColor('#93c5fd')),
        ('LINEBELOW', (0, 1), (-1, -1), .3, colors.HexColor('#e2e8f0')),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 8), ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 6), ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.extend([t, Spacer(1, 10)])


def figure(name, caption):
    story.append(Image(str(ROOT / 'artifacts' / name), width=495, height=171.35))
    p(caption, 'SmallRu')


def new_page():
    story.append(PageBreak())


# Страница 1: автор, постановка и данные.
p('БАЗОВАЯ МАТЕМАТИКА ДЛЯ ИСКУССТВЕННОГО ИНТЕЛЛЕКТА', 'SmallRu')
p('Лабораторная работа 4', 'TitleRu')
p('Полносвязная сеть своими руками на CPU<br/><b>Бинарный классификатор Banknote Authentication</b>')
table([['Выполнил', 'Группа', 'Дата'], ['Тоц Леонид Александрович', 'ИВТ-2', '06.10.2026']], [305, 80, 110])
sub('Цель и постановка задачи')
p('Реализовать многослойный перцептрон (MLP), прямой и обратный проходы вручную, '
  'обучить его mini-batches на CPU и оценить качество на независимой тестовой выборке. '
  'Выбран вариант бинарной классификации из разрешённого списка задания.')
sub('1. Датасет и подготовка')
p('Banknote Authentication: 1372 записи, 4 числовых признака, одна целевая переменная '
  'class ∈ {0, 1}. Признаки получены из изображений образцов банкнот. '
  'Положительным считается исходный класс 1; исходные метки не переименованы. Пропусков нет.')
table([['Признак', 'Содержание'],
       ['variance', 'Дисперсия вейвлет-преобразованного изображения'],
       ['skewness', 'Асимметрия вейвлет-преобразованного изображения'],
       ['curtosis', 'Эксцесс вейвлет-преобразованного изображения'],
       ['entropy', 'Энтропия изображения']], [105, 390])
p('До разбиения удалены 24 полных дубликата. Осталось 1348 уникальных записей: '
  '738 класса 0 и 610 класса 1. Одинаковые признаки с противоречащими метками не обнаружены. '
  'Разбиение стратифицированное, около 60/20/20, seed=42.')
table([['Часть', 'Записей', 'Класс 0', 'Класс 1'],
       ['train', 809, 443, 366], ['val', 270, 148, 122], ['test', 269, 147, 122]], [140, 115, 120, 120])
p('Источник: Volker Lohweg (2012), UCI Machine Learning Repository. DOI: '
  '<link href="https://doi.org/10.24432/C55P57" color="#2563eb">10.24432/C55P57</link>. '
  'Лицензия данных: CC BY 4.0. Файл сохранён локально без изменения значений.', 'SmallRu')

# Страница 2: математика.
new_page()
heading('2. Математическая модель')
sub('Стандартизация')
p('Средние μ и стандартные отклонения s (ddof=0) вычислены только по train. '
  'Для val и test применяются те же параметры: x′<sub>j</sub> = (x<sub>j</sub> - μ<sub>j</sub>) / s<sub>j</sub>. '
  'При s=0 используется масштаб 1. Это исключает утечку информации из val/test при предобработке.')
sub('Прямой проход и размерности')
p('Объекты расположены по строкам. Для батча из m объектов A<sub>0</sub> имеет размер m × 4. '
  'Сеть содержит два скрытых слоя: 4 → 32 → 16 → 1. Матрицы весов имеют размеры '
  '4 × 32, 32 × 16, 16 × 1; смещения - 1 × 32, 1 × 16, 1 × 1. Всего 705 параметров.')
p('Z<sub>l</sub> = A<sub>l-1</sub>W<sub>l</sub> + b<sub>l</sub><br/>'
  'A<sub>l</sub> = max(0, Z<sub>l</sub>) для скрытых слоёв<br/>'
  'p = σ(Z<sub>L</sub>) = 1 / (1 + exp(-Z<sub>L</sub>))', 'FormulaRu')
sub('Функция потерь и L2')
p('J = (1/m) ∑<sub>i</sub> [log(1 + exp(z<sub>i</sub>)) - y<sub>i</sub>z<sub>i</sub>] '
  '+ (λ/2) ∑<sub>l</sub> ||W<sub>l</sub>||<super>2</super><sub>F</sub>', 'FormulaRu')
p('Первая часть - бинарная кросс-энтропия (BCE). Она вычисляется через np.logaddexp(0, z), '
  'чтобы избежать log(0) и переполнения exp. Сигмоида вычисляется отдельно для z ≥ 0 и z &lt; 0. '
  'L2-регуляризация с λ=0.0001 применяется к весам, но не к смещениям.')
sub('Ручное обратное распространение ошибки')
p('D<sub>L</sub> = (p - y) / m<br/>'
  'D<sub>l</sub> = (D<sub>l+1</sub>W<sub>l+1</sub><super>T</super>) ⊙ f′(Z<sub>l</sub>)<br/>'
  '∇W<sub>l</sub> = A<sub>l-1</sub><super>T</super>D<sub>l</sub> + λW<sub>l</sub><br/>'
  '∇b<sub>l</sub> = ∑<sub>i</sub> D<sub>l,i</sub>', 'FormulaRu')
p('Для ReLU f′(z)=1 при z&gt;0 и 0 при z≤0. В коде также поддерживается tanh '
  'с производной 1 - tanh²(z). Деление на фактический размер батча выполняется один раз '
  'в D<sub>L</sub>; повторное усреднение в ∇W не требуется. Последний батч содержит 9 объектов. '
  'Все градиенты вычисляются до изменения любого веса.')

# Страница 3: протокол.
new_page()
heading('3. Протокол обучения и логирование')
table([['Параметр', 'Значение'],
       ['Архитектура / активации', '4 → 32 → 16 → 1; ReLU / sigmoid'],
       ['Инициализация', 'He для скрытых ReLU; Xavier для выхода; bias=0'],
       ['Оптимизатор', 'Самописный SGD с моментом: β=0.9, η=0.03'],
       ['Mini-batch / максимум эпох', '32 / 150'],
       ['L2 / ранняя остановка', 'λ=0.0001; patience=20; min_delta=0.00001'],
       ['Критерий лучшей модели', 'Минимальная BCE на val; восстановление весов'],
       ['Seed', 'Разбиение=42; инициализация=42; перемешивание=43'],
       ['Устройство / арифметика', 'CPU, NumPy, float64']], [200, 295])
p('Обновление параметров: v ← βv + g; θ ← θ - ηv. Для каждого веса и смещения '
  'хранится свой вектор момента. Перед каждой эпохой train перемешивается. '
  'Гиперпараметры зафиксированы до тестовой оценки; перебор по test не выполнялся.')
sub('Последовательность эксперимента')
for text in [
    '1. Проверить файл, удалить полные дубликаты, сформировать train/val/test.',
    '2. Обучить стандартизацию на train и применить её к трём частям.',
    '3. На каждом mini-batch выполнить forward → BCE → backward → SGD.',
    '4. После эпохи вычислить train/val BCE и Accuracy, записать историю.',
    '5. По val BCE сохранить лучший чекпойнт; при отсутствии значимого улучшения 20 эпох остановиться.',
    '6. Восстановить лучшую модель, выбрать порог по val F1 и один раз оценить test.',
]:
    p(text)
sub('История в pandas.DataFrame')
p('Столбцы: epoch, loss, val_loss, accuracy, val_accuracy, objective. '
  'loss и val_loss - BCE без L2; objective - train BCE с L2. Accuracy в истории '
  'вычисляется при фиксированном пороге 0.5. Метрики считаются на полных train/val '
  'после обновлений эпохи, а не как невзвешенное среднее ошибок батчей.')
p('История сохранена в artifacts/history.csv. model.npz содержит веса, смещения, '
  'архитектуру, параметры стандартизации и порог. Индексы разбиения и прогнозы '
  'test сохраняются отдельно. Готовые MLP, PyTorch, TensorFlow и autograd не используются.')

# Страница 4: сходимость.
new_page()
heading('4. Кривые обучения и сходимость')
figure('learning_curves.png', 'Рис. 1. BCE и Accuracy на train/val. Вертикальная линия - лучшая эпоха 149.')
rows = [['Эпоха', 'Train BCE', 'Val BCE', 'Train acc.', 'Val acc.']]
for index in [0, 9, 49, 99, 148, 149]:
    row = HISTORY.iloc[index]
    rows.append([int(row.epoch), f'{row.loss:.6f}', f'{row.val_loss:.6f}',
                 f'{row.accuracy:.4f}', f'{row.val_accuracy:.4f}'])
table(rows, [65, 110, 110, 105, 105])
p(f'До обучения BCE составляла {RESULT["initial_loss"]["train"]:.6f} на train и '
  f'{RESULT["initial_loss"]["val"]:.6f} на val. К 10-й эпохе Accuracy на обеих частях '
  'достигла 1.0; последующее уменьшение BCE отражает рост уверенности прогнозов.')
p('Обучение завершилось по лимиту 150 эпох. Ранняя остановка была включена, '
  'но её условие не сработало. Минимум val BCE получен на эпохе 149: 0.000224. '
  'Именно её веса восстановлены для итоговой оценки; график сохраняет все 150 эпох.')
p('На кривой val не наблюдается систематического роста ошибки при уменьшении train BCE. '
  'Явных признаков переобучения в этом запуске нет. Однако одно разбиение не доказывает '
  'полного отсутствия переобучения; L2 и выбор чекпойнта по val остаются средствами контроля.')
sub('Порог решения')
p('ŷ=1[p≥τ]. Порог не входит в функцию обучения. Он выбран по максимальному F1 на val '
  'среди середин между уникальными вероятностями, крайних допустимых значений и 0.5. '
  'При равном F1 выбирается ближайший к 0.5 порог, затем меньший. '
  'Максимум F1=1 достигается при τ=0.5, поэтому выбран этот порог.')

# Страница 5: оценка.
new_page()
heading('5. Итоговые метрики и графики')
rows = [['Часть', 'BCE', 'Accuracy', 'F1', 'ROC-AUC', 'AP']]
for name in ['train', 'val', 'test']:
    metric = RESULT['metrics'][name]
    rows.append([name, f'{metric["loss"]:.6f}', f'{metric["accuracy"]:.4f}',
                 f'{metric["f1"]:.4f}', f'{metric["roc_auc"]:.4f}', f'{metric["average_precision"]:.4f}'])
table(rows, [70, 105, 90, 70, 85, 75])
p('На test из 269 объектов: TN=147, TP=122, FP=0, FN=0. '
  'Precision=Recall=1.0. Использован чекпойнт эпохи 149 и τ=0.5, '
  'зафиксированные по val. Повторного обучения на train+val нет.')
figure('roc_pr.png', 'Рис. 2. ROC и PR на test. AP - Average Precision, а не трапециевидная площадь PR.')
figure('confusion_threshold.png', 'Рис. 3. Матрица ошибок test и выбор порога по F1 на val.')
p('Accuracy=(TP+TN)/N; F1=2TP/(2TP+FP+FN). ROC-AUC вычислена по всем порогам '
  'трапециевидным интегрированием ROC. AP вычислена как сумма Precision·ΔRecall '
  'по группам одинаковых оценок. ROC-AUC и AP получены из вероятностей, а не бинарных прогнозов.', 'SmallRu')

# Страница 6: проверка, вывод, источники.
new_page()
heading('6. Проверка реализации и вывод')
sub('Численные градиенты и независимые тесты')
p('Центральная разность с ε=10<super>-6</super> проверена для всех 47 параметров сети '
  '4 → 5 → 3 → 1, для ReLU и tanh, с L2=0.07. Для ReLU выбраны ненулевые смещения '
  'и точки вдали от z=0: в точке излома центральная разность не обязана совпадать '
  'с принятой в коде односторонней производной.')
table([['Активация', 'Параметров', 'Макс. |Δg|', 'Отн. L2-ошибка'],
       *[[a, check['parameters_checked'], f'{check["max_absolute_error"]:.3e}',
          f'{check["relative_l2_error"]:.3e}'] for a, check in RESULT['gradient_checks'].items()]],
      [110, 105, 140, 140])
p('13 тестов пройдены. Проверены градиенты с L2 и без неё, однократное усреднение '
  'по батчу, устойчивость при логитах ±1000, непересечение частей и удаление дубликатов, '
  'стандартизация, подбор порога, восстановление раннего лучшего чекпойнта и загрузка модели. '
  'Accuracy/F1/ROC-AUC/AP независимо сверены со scikit-learn, включая равные вероятности; '
  'scikit-learn используется только в тестах.')
sub('Вывод')
p('Цель выполнена: реализован собственный MLP с ручным backpropagation и обучением '
  'mini-batches на CPU. При τ=0.5 все 269 тестовых записей классифицированы верно; '
  'Accuracy=F1=ROC-AUC=AP=1.0, BCE≈0.000446. Результат воспроизводится из локального '
  'датасета с фиксированными seed и сохранённой моделью.')
p('Высокое качество относится к одному небольшому датасету и одному разбиению. '
  'Оно не гарантирует отсутствие ошибок на данных из других источников. '
  'Для более устойчивой оценки полезны повторные стратифицированные разбиения '
  'или кросс-валидация с подбором параметров внутри обучающих частей.')
sub('Воспроизводимость')
p('Запуск из lab4: python experiment.py. Проверки: python -m pytest -q tests. '
  'Ноутбук Тоц_ЛА_ИВТ-2_ЛР4_MLP.ipynb содержит выполненные ячейки, таблицы и графики. '
  'Полный протокол, версии библиотек и SHA-256 данных записаны в artifacts/results.json.')
sub('Источники')
p('1. Lohweg, V. (2012). Banknote Authentication. UCI. '
  '<link href="https://archive.ics.uci.edu/dataset/267/banknote+authentication" color="#2563eb">'
  'Страница датасета</link>; DOI: 10.24432/C55P57. Дата обращения: 06.10.2026.<br/>'
  '2. Материалы преподавателя (Dmitry Vlasov, 2025): «Лабораторная работа 4 - самописный '
  'MLP на CPU»; протокол и логирование истории; псевдокод, алгоритм и вывод формул '
  'backpropagation с mini-batches; справочник метрик классификации и регрессии.<br/>'
  '3. Everton Gomede. Understanding Backpropagation: The Engine Behind Neural Network Learning '
  '(30.09.2023), предоставленный учебный материал.', 'SmallRu')


def decorate(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor('#cbd5e1'))
    canvas.line(50, 42, 545, 42)
    canvas.setFont('Arial', 8)
    canvas.setFillColor(GRAY)
    canvas.drawString(50, 29, 'Тоц Л. А. | ИВТ-2 | Лабораторная работа 4')
    canvas.drawRightString(545, 29, str(doc.page))
    canvas.restoreState()


OUT.parent.mkdir(parents=True, exist_ok=True)
doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=50, rightMargin=50,
                        topMargin=42, bottomMargin=58, title='ЛР4. MLP на CPU',
                        author='Тоц Леонид Александрович', subject='Базовая математика для искусственного интеллекта')
doc.build(story, onFirstPage=decorate, onLaterPages=decorate)
print(OUT)
