from srtools import cyrillic_to_latin
import re

NUM_MAP = {
    "0": " nula ",
    "1": " jedan ",
    "2": " dva ",
    "3": " tri ",
    "4": " četiri ",
    "5": " pet ",
    "6": " šest ",
    "7": " sedam ",
    "8": " osam ",
    "9": " devet ",
    "10": " deset ",
    "11": " jedanaest ",
    "12": " dvanaest ",
    "13": " trinaest ",
    "14": " četrnaest ",
    "15": " petnaest ",
    "16": " šesnaest ",
    "17": " sedamnaest ",
    "18": " osamnaest ",
    "19": " devetnaest ",
    "20": " dvadeset ",
}

def normalize_sr_num(text):
    if not re.search(r"\d", text):
        return text

    def repl(m: re.Match) -> str:
        s = m.group(0)                # например "12" или "123"
        return NUM_MAP.get(s, " ")    # если нет в словаре — удаляем

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
        r"[‐-‒–—−/\\_.,:;!?…()\[\]{}<>\"'“”„«»‹›]+",
        " ",
        text,
    )
    text = re.sub(
        r"[ʼ]",
        "'",
        text,
    )
    
    text = re.sub(r"[^a-zčćđšž'\s]", "", text)
    text = " ".join(text.split())
    return text
