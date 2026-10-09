# Мендогралова Софья МТ-201: лемматизация + формирование итогового датасета
import json

import pandas as pd

from p1_load_clean import tidy
from p2_entities import find_all
from p3_anonymize import hide
from p4_tokens import STOP, split_words, drop_stop

# Запасной словарь: нужен только если не установлен pymorphy3.
# Опечатки (закааз, оплта и т.п.) намеренно не исправляются:
# цель работы - очистка и нормализация, а не орфокоррекция.
LEMMA_DICT = {
    "покупку": "покупка", "закааз": "заказ", "закааза": "заказ", "заказа": "заказ",
    "закаазывал": "заказывать", "заказывал": "заказывать", "пришел": "прийти",
    "получил": "получить", "мышку": "мышка", "ответьте": "ответить",
    "прошла": "пройти", "висит": "висеть", "смотрите": "смотреть",
    "отменили": "отменить", "объяснения": "объяснение", "причин": "причина",
    "корзине": "корзина", "корзину": "корзина", "применяется": "применяться",
    "подскажите": "подсказать", "моего": "мой", "жду": "ждать", "хочу": "хотеть",
    "целая": "целый", "подошёл": "подойти", "размеру": "размер",
    "изменилась": "измениться", "добавления": "добавление", "дешевле": "дешёвый",
    "купила": "купить", "наушники": "наушник", "нужен": "нужный",
    "пришлите": "прислать", "пожалуста": "пожалуйста", "дней": "день",
    "работает": "работать", "занимает": "занимать", "оплта": "оплата",
    "бухгалтерии": "бухгалтерия", "вернутся": "вернуться",
}


def make_lemmatizer():
    """Этап 6. pymorphy3, если установлен; иначе запасной словарь."""
    try:
        import pymorphy3
        morph = pymorphy3.MorphAnalyzer()  # создаём один раз
        return lambda words: [morph.parse(w)[0].normal_form for w in words]
    except ImportError:
        print("pymorphy3 не найден: используется запасной словарь")
        return lambda words: [LEMMA_DICT.get(w, w) for w in words]


def build_dataset(df):
    """Этап 7. Итоговый датасет: прогоняем каждое сообщение через весь конвейер."""
    lemmatize = make_lemmatizer()
    rows = []
    for _, r in df.iterrows():
        raw = str(r["message"])
        found = find_all(raw)
        anon = tidy(hide(raw))
        words = drop_stop(split_words(anon))
        lem = [w for w in lemmatize(words) if w not in STOP]  # стоп-слова и среди лемм
        rows.append({
            "message_id": r["message_id"], "channel": r["channel"],
            "created_at": r["created_at"], "original_text": raw,
            **{k: json.dumps(v, ensure_ascii=False) for k, v in found.items()},
            "anonymized_text": anon,
            "clean_text": " ".join(words),
            "tokens": json.dumps(words, ensure_ascii=False),
            "lemmas": json.dumps(lem, ensure_ascii=False),
        })
    return pd.DataFrame(rows)
