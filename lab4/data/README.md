# Banknote Authentication

Автор: Volker Lohweg (2012). UCI Machine Learning Repository.
Страница: https://archive.ics.uci.edu/dataset/267/banknote+authentication
DOI: https://doi.org/10.24432/C55P57
Лицензия: Creative Commons Attribution 4.0 (CC BY 4.0).
Загрузка: 06.10.2026, официальный архив
https://archive.ics.uci.edu/static/public/267/banknote%2Bauthentication.zip

`data_banknote_authentication.txt` - исходный файл UCI без изменений,
1372 строки, 5 колонок без заголовка: variance, skewness, curtosis, entropy, class.
class - исходные метки 0/1. Остальные колонки числовые; пропусков нет.
SHA-256: d0539aaed2139ba7a587b3e34fb345ce503ff7d5d33dbf9912d8e195ce425cb9.

В `load_dataset` удаляются 24 полных дубликата до разбиения. Это преобразование
применяется в памяти и не меняет исходный файл. В split.csv сохраняются исходные
индексы оставленных строк, считая первую строку индексом 0.
