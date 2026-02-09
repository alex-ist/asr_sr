from srtools import cyrillic_to_latin
import re

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

