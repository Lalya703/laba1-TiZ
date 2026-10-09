# Арканова Ксения МТ - 201
#Этап 4. Замена сущностей маркерами [URL], [EMAIL], [PHONE], [DATE], [PRICE].
from p2_entities import PAT

MARK = {"urls": "[URL]", "emails": "[EMAIL]", "phones": "[PHONE]",
        "dates": "[DATE]", "prices": "[PRICE]"}


def hide(text):
    for k, p in PAT.items():
        text = p.sub(MARK[k], text)
    return text
