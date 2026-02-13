from srtools import cyrillic_to_latin
import re
from collections import Counter

_ONES = ["", "jedan", "dva", "tri", "četiri", "pet", "šest", "sedam", "osam", "devet"]
_TEENS = [
    "deset", "jedanaest", "dvanaest", "trinaest", "četrnaest",
    "petnaest", "šesnaest", "sedamnaest", "osamnaest", "devetnaest",
]
_TENS = [
    "", "", "dvadeset", "trideset", "četrdeset",
    "pedeset", "šezdeset", "sedamdeset", "osamdeset", "devedeset",
]
_HUNDREDS = [
    "", "sto", "dvesto", "tristo", "četiristo",
    "petsto", "šeststo", "sedamsto", "osamsto", "devetsto",
]


def _sr_num_to_words(n: int) -> str:
    """Конвертирует целое число (0 .. 999_999_999) в сербские слова."""
    if n == 0:
        return "nula"

    parts: list[str] = []

    # --- милиони ---
    if n >= 1_000_000:
        mil = n // 1_000_000
        last2 = mil % 100
        last1 = mil % 10
        if mil == 1:
            parts.append("milion")
        else:
            parts.append(_sr_num_to_words(mil))
            # 2-4 (но не 12-14) -> miliona; остальное тоже miliona
            if last1 in (2, 3, 4) and last2 not in (12, 13, 14):
                parts.append("miliona")
            else:
                parts.append("miliona")
        n %= 1_000_000

    # --- хиљаде ---
    if n >= 1000:
        th = n // 1000
        last2 = th % 100
        last1 = th % 10
        if th == 1:
            parts.append("hiljadu")
        elif last1 == 1 and last2 != 11:
            # feminine "jedna" + genitive singular "hiljada"
            words = _sr_num_to_words(th)
            words = re.sub(r"jedan$", "jedna", words)
            parts.append(words + " hiljada")
        elif last1 == 2 and last2 != 12:
            # feminine "dve" + genitive singular "hiljade"
            if th == 2:
                parts.append("dve hiljade")
            else:
                words = _sr_num_to_words(th)
                words = re.sub(r"dva$", "dve", words)
                parts.append(words + " hiljade")
        elif last1 in (3, 4) and last2 not in (13, 14):
            parts.append(_sr_num_to_words(th) + " hiljade")
        else:
            parts.append(_sr_num_to_words(th) + " hiljada")
        n %= 1000

    # --- стотине ---
    if n >= 100:
        parts.append(_HUNDREDS[n // 100])
        n %= 100

    # --- десетице и јединице ---
    if 10 <= n <= 19:
        parts.append(_TEENS[n - 10])
    else:
        if n >= 20:
            parts.append(_TENS[n // 10])
            n %= 10
        if n > 0:
            parts.append(_ONES[n])

    return " ".join(parts)


def normalize_sr_num(text: str) -> str:
    if not re.search(r"\d", text):
        return text

    def repl(m: re.Match) -> str:
        n = int(m.group(0))
        if n > 999_999_999:
            return " "
        return " " + _sr_num_to_words(n) + " "

    return re.sub(r"\d+", repl, text)

def normalize_sr_text(text):
    text = text.lower()
    text = cyrillic_to_latin(text)

    text = (text # сербская латиница - там нет W!
        .replace("w", "v")
        .replace("q", "k")
        .replace("y", "j")
        .replace("x", "ks")
    )

    text=normalize_sr_num(text)
    text = re.sub(
        r"[‐-‒–—−/\\_.,:;!?…()\[\]{}<>\"“”„«»‹›]+",
        " ",
        text,
    )
    text = re.sub(r"[^a-zčćđšž\s]", "", text)
    text = " ".join(text.split())
    return text



def is_repetition_loop(text, duration_sec=None):
    """
    Проверяет, является ли текст результатом зацикливания Whisper.
    Returns:
        True если текст похож на зацикливание, False если нормальный.
    """
    words = text.strip().split()
    if len(words) == 0:
        return True  # пустой текст тоже фильтруем

    if any(len(w) > 30 for w in words):
        return True
    
    # 1. Слишком много слов на секунду аудио
    if duration_sec:
        max_words_per_sec=8.0
        wps = len(words) / max(duration_sec, 0.1)
        if wps > max_words_per_sec:
            return True
    
    # 2. Слишком мало уникальных слов (повторы)
    min_unique_ratio = 0.4
    unique_ratio = len(set(words)) / len(words)
    if len(words) >= 5 and unique_ratio < min_unique_ratio:
        return True
    
    # 3. Самое частое слово повторяется слишком много раз
    if len(words) >= 4:
        most_common_count = Counter(words).most_common(1)[0][1]
        if most_common_count / len(words) > 0.7:
            return True

    # 4. Проверяем последовательные повторы в конце (локальное зацикливание)
    if len(words) >= 4:
        # Считаем, сколько раз подряд повторяется последнее слово
        last_word = words[-1]
        consecutive_count = 1
        for i in range(len(words) - 2, -1, -1):
            if words[i] == last_word:
                consecutive_count += 1
                # Выходим сразу, как только нашли 4 повтора - это зацикливание
                if consecutive_count >= 4:
                    return True
            else:
                break

    return False


def remove_repetition_loops(text, duration_sec=None):
    """
    Удаляет зацикливания из текста, выданного Whisper моделью.
    Ищет паттерны зацикливания с начала текста и оставляет первое вхождение.

    Args:
        text: исходный текст
        duration_sec: длительность аудио в секундах (опционально)

    Returns:
        Очищенный текст без зацикливаний
    """
    if not text:
        return ""

    words = text.strip().split()
    if len(words) == 0:
        return ""

    # 1. Обрезаем с позиции первого длинного слова (>30 символов)
    for i, word in enumerate(words):
        if len(word) > 30:
            if i == 0:
                return ""
            return " ".join(words[:i]).strip()

    # 2. Ищем зацикливание с начала текста
    # ВАЖНО: проверяем от длинных паттернов к коротким!

    # 2.1. Повтор 4-грамм 3 раза подряд - оставляем первую 4-грамм
    if len(words) >= 12:  # минимум 3*4 = 12 слов
        for i in range(len(words) - 11):
            fourgram = tuple(words[i:i+4])
            # Проверяем, повторяется ли эта 4-грамма еще 2 раза подряд (всего 3 раза)
            found = True
            for rep in range(1, 3):
                offset = i + rep * 4
                if offset + 4 > len(words) or tuple(words[offset:offset+4]) != fourgram:
                    found = False
                    break
            if found:
                # Нашли 3 повтора 4-граммы подряд
                # Оставляем все до зацикливания + первое вхождение 4-граммы
                return " ".join(words[:i+4]).strip()

    # 2.2. Повтор 3-грамм 3 раза подряд - оставляем первую 3-грамм
    if len(words) >= 9:  # минимум 3*3 = 9 слов
        for i in range(len(words) - 8):
            trigram = tuple(words[i:i+3])
            # Проверяем, повторяется ли эта 3-грамма еще 2 раза подряд (всего 3 раза)
            found = True
            for rep in range(1, 3):
                offset = i + rep * 3
                if offset + 3 > len(words) or tuple(words[offset:offset+3]) != trigram:
                    found = False
                    break
            if found:
                # Нашли 3 повтора 3-граммы подряд
                # Оставляем все до зацикливания + первое вхождение 3-граммы
                return " ".join(words[:i+3]).strip()

    # 2.3. Повтор 2-грамм 4 раза подряд - оставляем первую 2-грамм
    if len(words) >= 8:  # минимум 2*4 = 8 слов
        for i in range(len(words) - 7):
            bigram = tuple(words[i:i+2])
            # Проверяем, повторяется ли эта 2-грамма еще 3 раза подряд (всего 4 раза)
            found = True
            for rep in range(1, 4):
                offset = i + rep * 2
                if offset + 2 > len(words) or tuple(words[offset:offset+2]) != bigram:
                    found = False
                    break
            if found:
                # Нашли 4 повтора 2-граммы подряд
                # Оставляем все до зацикливания + первое вхождение 2-граммы
                return " ".join(words[:i+2]).strip()

    # 2.4. Повтор одного слова 6 раз подряд - оставляем текст до зацикливания + первое слово
    if len(words) >= 6:
        for i in range(len(words) - 5):
            if all(words[i] == words[i+j] for j in range(1, 6)):
                # Нашли 6 повторов одного слова подряд
                # Оставляем все до зацикливания + первое вхождение слова
                return " ".join(words[:i+1]).strip()


    # Зацикливаний не найдено, возвращаем весь текст
    return " ".join(words).strip()


# tets_exaples = [
#     "da je u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u te",
#     "kako je svača kako je svača kako je svača kako je svača kako je svača kako je svača kako je svača kako je svača kako je svača kako je svača kako je svača kako je svača kako je svača kako je svača kako je svača kako je svača kako je svača kako je svača kako je svača kako je svača kako je svača kako je svača kako je svača kako je svača kako je svača kako je svača kako je svača kako je svača kako je svača kako je svača kako je svača kako je svača kako je svača kako je svača kako je svača kako je svača kako je svača kako je svača kako je svača kako je svača kako je svača kako je svača kako je svača kako je svača kako je svača kako je svača kako je svača kako je svača kako je svača kako je svača kako je svača kako je svača kako je svača kako je svača kako je svača kako je s",
#     "u krakovu sada sam sada sada sada sada sam sada sada sada sam sada sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sam sada sam sada sada sam sada sam sada sam sada sam sada sam sada sam sada sam sada sam sada sam sada sam sada sam sada sam sada sam",
#     "kako su stradaće belorusi stvarno su belorusi najviše i brojčom odstradali oni su bili prvi na udaru nemaca ali karim dok se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko",
#     "u sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve",
#     "da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono",
#     "u timsacom u timsacseksijama u novo saksijama u novo saksijama u novo satskom u timsacima u novo saksijama u novo saksijama u novo saksijama u novo saksijama u novo saksijama u novo saksijama u novo saksijama u novo saksijama u novo saksijama u novo saksijama u novo saksijama u novo saksijama u novo saksijama u novo saksijama u novo saksijama u novo saksijama u novo saksijama u novo saksijama u novo saksijama u novo saksijama u novo saksijama u novo saksijama u novo saksijama u novo saksijama u novo saksijama u novo saksijama u novo saksijama u novo saksijama u novo saksijama u novo saksijama u novo saksijama u novo saksijama u novo saksijama u novo saksijama u novo saksijama u novo saksijama u novo saksijama u novo saksijama u novo saksijama u novo saksijama u novo s"
# ]

