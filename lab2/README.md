# ЛР № 01. Основы языковых моделей — выполненный вариант (Greenfield Town)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huksleva/Basic_Mathematics_for_Artificial_Intelligence/blob/main/lab2/lab_01_student.ipynb)

Готовый `student`-ноутбук: все функции с `TODO` реализованы, notebook выполнен от начала
до конца без ошибок, все контрольные `assert` и все 9 проверок обязательного
мини-портфолио прошли (`✓`). Markdown-ячейки с выводами, ручной проверкой softmax,
паспортом результата и оценками Лайкерта заполнены.

## Что внутри `lab_01_student.ipynb`

| §         | Реализовано                                                                                     | Ключевой результат                                                                  |
|-----------|-------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------|
| 1         | `tokenize`, `build_vocabulary`, `doc_to_bow`, `make_ngrams`                                     | словарь 51 токен, DTM `(12, 51)`, BoW переставленных фраз совпадает, биграммы — нет |
| 2         | `SimpleClassifier` (Linear→ReLU→Linear), обучение 800 шагов                                     | loss 1.0895 → 0.0056; оба новых документа классифицированы верно                    |
| 3         | `make_skipgram_pairs`, `SkipGramModel`, `nearest_words`                                         | 196 пар, словарь 33 токена; ближайший сосед `gardener` — `garden` (cos ≈ 0.573)     |
| 4         | `initialize_vocabulary`, `get_pair_counts`, `merge_pair`, `byte_pair_encoding`, `tokenize_word` | словарь BPE 50 токенов, 33 слияния; `garden7 → ['_garden', '<UNK>']`                |
| 5         | `CountLanguageModel`, `train`, `generate_text`, `compute_perplexity`                            | backoff работает, `P(unknown)` > 0, сумма вероятностей = 1, PPL = 5.63              |
| 6         | `ngram_counter`, `rouge_n_recall`, `lcs_length`, `rouge_l`, `expected_score`, `update_elo`      | ROUGE-1/2/L кандидата A = 1.0 / 0.6 / 0.831; сумма Elo-рейтингов сохранена (4500)   |
| Портфолио | базовый обязательный прогон + песочница `CUSTOM_CASE`                                           | все 9 инвариантов ✓; песочница не меняет базовый результат                          |

Ручная математика §2.1 (softmax и кросс-энтропия для логитов `(2.0, 1.0, 0.5)`),
паспорт результата (4 блока: BoW, skip-gram, BPE, Count LM) и оценки Лайкерта кандидата A
(связность / информативность / фактологическая точность) заполнены текстом с
конкретными числами из выполненного notebook, а не общими фразами.

## Как запустить самостоятельно

```bash
uv sync --locked
uv run jupyter lab
```

Notebook ожидает `data/greenfield_town.json` в текущем каталоге или в
`01-lab-language-model-basics/data/greenfield_town.json`. Данные и seed (`42`) не менялись.

## Что не входит в сдачу

- `lab_01_example.ipynb` (Harbor) и `lab_01_teacher.ipynb` — не сдаются, использовались
  только как образец логики и для сверки.
- `brown.txt.gz`, `embedding_corpus.txt` — архив прежнего профиля ЛР, не участвуют в
  активном маршруте Greenfield Town.

## Границы выводов (коротко)

- **BoW/классификатор** не видит порядок слов и не гарантирует обобщение — только
  снижение train loss на 12 обучающих документах.
- **Skip-gram** отражает совместную встречаемость слов в малом корпусе (12 строк,
  окно 2), а не смысловую синонимию.
- **BPE** `<UNK>` — неизвестный *символ* этой модели, а не неизвестное слово Count LM.
- **Count LM** хранит не более двух предыдущих токенов (триграмма) и не объединяет
  похожие слова (`garden` ≠ `gardener`).
- **PPL, ROUGE, Лайкерт, Elo** отвечают на разные вопросы и не заменяют друг друга:
  высокий ROUGE не доказывает фактологическую точность, PPL не измеряет смысл текста,
  Elo не даёт абсолютной оценки по одному матчу.
