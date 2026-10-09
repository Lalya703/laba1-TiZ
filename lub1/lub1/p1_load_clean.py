# Фролов Павел МТ-202: загрузка и первичный анализ данных + базовая очистка текста
import re            # поиск и замена по шаблону

import pandas as pd  # библиотека для таблиц

# допустимые символы: буквы, цифры, пробел и обычная пунктуация
JUNK = re.compile(r"[^\w\s\[\].,;:!?№\-()\"«»/=|]")

def load_data(path):
    """Этап 1. Загрузка CSV и первичный анализ."""
    df = pd.read_csv(path, encoding="utf-8-sig")    
    print("Размер:", df.shape)
    print("Типы:\n", df.dtypes)
    print("Пропуски:\n", df.isna().sum())
    return df

def tidy(text):
    """Этап 2. Регистр, повторы пунктуации, мусорные символы, пробелы."""
    text = text.lower() # переводит в нижний регистр
    text = re.sub(r"([!?.,;:])\1+", r"\1", text) #удаляет кучку символов, заменяя на один 
    text = JUNK.sub(" ", text) # делает мусорные символы в пробелы 
    text = re.sub(r"\[[a-z]+\]", lambda m: m.group().upper(), text)  # обратно в верхний регистр
    return re.sub(r"\s+", " ", text).strip() # удаляет лишние пробелы