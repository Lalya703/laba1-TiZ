# Богданова Владислава МТ-201: частотный анализ TOP-20, сохранение результатов и отчёт

import argparse        # читает команды c cmd
import json
import os              # для работы с файлами

import pandas as pd    # библиотека для табличек

# подключаем файлы нашей команды:
from p1_load_clean import load_data, tidy
from p3_anonymize import MARK
from p4_tokens import split_words
from p5_lemmas import build_dataset

def top20(words, name):
    """функция для поиска 20 самых частых слов"""
    counts_dict = {}
    for w in words:
        if w in counts_dict:
            counts_dict[w] = counts_dict[w] + 1
        else:
            counts_dict[w] = 1
            
    # превращаем словарик в список и сортируем его по частоте (от больших к меньшим)
    sorted_words = sorted(counts_dict.items(), key=lambda item: item[1], reverse=True)

    best20 = sorted_words[:20]
    df_top = pd.DataFrame(best20, columns=[name, "frequency"])
    return df_top


def main():
    # настраиваем чтение имени файла из консоли
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("-o", "--out", default=".")
    a = ap.parse_args()
    
    folder_out = a.out  # папка для резов
    
    # создание папки через os
    if not os.path.exists(folder_out):
        os.makedirs(folder_out)

    # 1. загрузка исх файла из p1_...
    df = load_data(a.src)                                    

    # 2. cчитаем ТОП-20 слов ДО очистки
    before = []
    for m in df["message"]:                     # берем нашу табл,
        string_msg = str(m)                     # поочередно вытаскиваеи сообщение,
        clean_string = tidy(string_msg)         # проходит чистку у p1,
        words_list = split_words(clean_string)  # потом делит на кусочки у p4
        for w in words_list:                    # складываем в before
            before.append(w)

    # 3. запуск конвейера обработки и получаем итоговую таблицу (р2-р5)
    res = build_dataset(df)

    # 4. считаем ТОП-20 слов ПОСЛЕ очистки
    after = []
    for s in res["lemmas"]:                     # берем из готовой таблице кол с чистыми словами
        list_of_lemmas = json.loads(s)          # расшифровка из json
        for lemma in list_of_lemmas:
            after.append(lemma)                 # складываем в after

    # вызываем функция для 2х мешков слов
    t_before = top20(before, "word")
    t_after = top20(after, "lemma")

    # 5. сохр итоговые файлы + формируем пути к файлам с помощью склеивания строчек через os.path.join
    path_csv = os.path.join(folder_out, "clean_messages.csv")
    path_top_before = os.path.join(folder_out, "top20_before.csv")
    path_top_after = os.path.join(folder_out, "top20_after.csv")
    path_report = os.path.join(folder_out, "report.txt")

    # сохр табл в CSV-файлы
    res.to_csv(path_csv, index=False, encoding="utf-8-sig")
    t_before.to_csv(path_top_before, index=False, encoding="utf-8-sig")
    t_after.to_csv(path_top_after, index=False, encoding="utf-8-sig")

    # 6. подсчет найденных сущностей
    counts = {}
    for key in MARK:                            # пробегаемся по заглушкам
        total_sum = 0
        for cell in res[key]:                   # см каж ячейку в колонке
            if cell != "[]":
                total_sum = total_sum + 1
        counts[key] = total_sum                 # записываем итоговое число в словарик

    # 7. итоговый текстовый отчет 
    name_of_file = os.path.basename(a.src)  # вытаскиваем чистое имя файла без лишних путей
    
    txt_report = f"""Лабораторная работа №1
ОТЧЕТ
Файл данных: {name_of_file}

1. Первичный анализ
- Объём: {len(df)} сообщений.
- Столбцы: {", ".join(df.columns)}.
- Типы: message_id — {df['message_id'].dtype}; остальные — текст.
- Пропусков: {int(df.isna().sum().sum())}.

2. Конвейер
сообщение -> сущности (regex) -> нижний регистр, пробелы, повторы пунктуации,
мусорные символы -> обезличивание ([URL], [EMAIL], [PHONE], [DATE], [PRICE]) ->
токены -> стоп-слова -> лемматизация -> частотный анализ.

3. Найдено сообщений с сущностями
- URL: {counts['urls']}
- e-mail: {counts['emails']}
- телефоны: {counts['phones']}
- даты: {counts['dates']}
- цены: {counts['prices']}
Сущности лежат в полях urls, emails, phones, dates, prices; в anonymized_text
они заменены маркерами.

4. Поля clean_messages.csv
message_id, channel, created_at, original_text, urls, emails, phones, dates,
prices, anonymized_text, clean_text, tokens, lemmas.

5. TOP-20 до очистки
{t_before.to_string(index=False)}

6. TOP-20 после очистки и лемматизации
{t_after.to_string(index=False)}

7. Вывод
До очистки в топе преобладают служебные и технические элементы (предлоги,
части ссылок и адресов: ru, https, www, mail, market, example, rub, item).
После извлечения сущностей, обезличивания, удаления стоп-слов и лемматизации
в топе остаются смысловые слова обращений покупателей: заказ, корзина, товар,
покупка, наушник, ноутбук и др. Словоформы объединяются (покупку/покупка,
заказа/заказ), поэтому частоты точнее, а технический шум исчезает.
Опечатки намеренно не исправляются (цель — очистка и нормализация).

8. Воспроизводимость
python lab1_preprocess.py <файл.csv>. Зависимости: requirements.txt.
Лемматизация: pymorphy3; без него используется запасной словарь для варианта 1.
"""
    
    # запись текста в файл
    file_object = open(path_report, "w", encoding="utf-8")
    file_object.write(txt_report)
    file_object.close()
    
    # вывод таблиц на экран для проверки
    print(t_before.to_string(index=False))
    print(t_after.to_string(index=False))


# запуск нашей главной функции
if __name__ == "__main__":
    main()
