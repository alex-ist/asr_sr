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
    "21": " dvadeset jedan ",
    "22": " dvadeset dva ",
    "23": " dvadeset tri ",
    "24": " dvadeset četiri ",
    "25": " dvadeset pet ",
    "26": " dvadeset šest ",
    "27": " dvadeset sedam ",
    "28": " dvadeset osam ",
    "29": " dvadeset devet ",
    "30": " trideset ",
    "31": " trideset jedan ",
    "32": " trideset dva ",
    "33": " trideset tri ",
    "34": " trideset četiri ",
    "35": " trideset pet ",
    "36": " trideset šest ",
    "37": " trideset sedam ",
    "38": " trideset osam ",
    "39": " trideset devet ",
    "40": " četrdeset ",
    "41": " četrdeset jedan ",
    "42": " četrdeset dva ",
    "43": " četrdeset tri ",
    "44": " četrdeset četiri ",
    "45": " četrdeset pet ",
    "46": " četrdeset šest ",
    "47": " četrdeset sedam ",
    "48": " četrdeset osam ",
    "49": " četrdeset devet ",
    "50": " pedeset ",
    "51": " pedeset jedan ",
    "52": " pedeset dva ",
    "53": " pedeset tri ",
    "54": " pedeset četiri ",
    "55": " pedeset pet ",
    "56": " pedeset šest ",
    "57": " pedeset sedam ",
    "58": " pedeset osam ",
    "59": " pedeset devet ",
    "60": " šezdeset ",
    "61": " šezdeset jedan ",
    "62": " šezdeset dva ",
    "63": " šezdeset tri ",
    "64": " šezdeset četiri ",
    "65": " šezdeset pet ",
    "66": " šezdeset šest ",
    "67": " šezdeset sedam ",
    "68": " šezdeset osam ",
    "69": " šezdeset devet ",
    "70": " sedamdeset ",
    "71": " sedamdeset jedan ",
    "72": " sedamdeset dva ",
    "73": " sedamdeset tri ",
    "74": " sedamdeset četiri ",
    "75": " sedamdeset pet ",
    "76": " sedamdeset šest ",
    "77": " sedamdeset sedam ",
    "78": " sedamdeset osam ",
    "79": " sedamdeset devet ",
    "80": " osamdeset ",
    "81": " osamdeset jedan ",
    "82": " osamdeset dva ",
    "83": " osamdeset tri ",
    "84": " osamdeset četiri ",
    "85": " osamdeset pet ",
    "86": " osamdeset šest ",
    "87": " osamdeset sedam ",
    "88": " osamdeset osam ",
    "89": " osamdeset devet ",
    "90": " devedeset ",
    "91": " devedeset jedan ",
    "92": " devedeset dva ",
    "93": " devedeset tri ",
    "94": " devedeset četiri ",
    "95": " devedeset pet ",
    "96": " devedeset šest ",
    "97": " devedeset sedam ",
    "98": " devedeset osam ",
    "99": " devedeset devet ",
    "100": " sto ",
    "200": " dvesto ",
    "300": " tristo ",
    "400": " četiristo ",
    "500": " petsto ",
    "600": " šeststo ",
    "700": " sedamsto ",
    "800": " osamsto ",
    "900": " devetsto ",
    "1000": " hiljadu ",
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
