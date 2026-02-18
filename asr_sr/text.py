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
    """Convert integer (0 .. 999_999_999) to Serbian words."""
    if n == 0:
        return "nula"

    parts: list[str] = []

    # Millions
    if n >= 1_000_000:
        mil = n // 1_000_000
        last2 = mil % 100
        last1 = mil % 10
        if mil == 1:
            parts.append("milion")
        else:
            parts.append(_sr_num_to_words(mil))
            parts.append("miliona")
        n %= 1_000_000

    # Thousands
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

    # Hundreds
    if n >= 100:
        parts.append(_HUNDREDS[n // 100])
        n %= 100

    # Tens and ones
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

    # there is no "w", "q", "y", "x" in Serbian
    text = (text 
        .replace("w", "v")
        .replace("q", "k")
        .replace("y", "j")
        .replace("x", "ks")
    )

    text = normalize_sr_num(text)
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
    Check if text is a Whisper repetition loop.
    Returns True if text appears to be a repetition loop.
    """
    words = text.strip().split()
    if len(words) == 0:
        return True

    if any(len(w) > 30 for w in words):
        return True

    # Too many words per second
    if duration_sec:
        max_words_per_sec=8.0
        wps = len(words) / max(duration_sec, 0.1)
        if wps > max_words_per_sec:
            return True

    # Too few unique words
    min_unique_ratio = 0.4
    unique_ratio = len(set(words)) / len(words)
    if len(words) >= 5 and unique_ratio < min_unique_ratio:
        return True

    # Most common word repeats too often
    if len(words) >= 4:
        most_common_count = Counter(words).most_common(1)[0][1]
        if most_common_count / len(words) > 0.7:
            return True

    # Consecutive repetitions at the end
    if len(words) >= 4:
        last_word = words[-1]
        consecutive_count = 1
        for i in range(len(words) - 2, -1, -1):
            if words[i] == last_word:
                consecutive_count += 1
                if consecutive_count >= 4:
                    return True
            else:
                break

    return False


def remove_repetition_loops(text, duration_sec=None):
    """
    Remove repetition loops from Whisper output.
    Detects repetition patterns and keeps only the first occurrence.

    Args:
        text: input text
        duration_sec: audio duration in seconds (optional)

    Returns:
        Cleaned text without repetitions
    """
    if not text:
        return ""

    words = text.strip().split()
    if len(words) == 0:
        return ""

    # Truncate at first long word (>30 chars)
    for i, word in enumerate(words):
        if len(word) > 30:
            if i == 0:
                return ""
            return " ".join(words[:i]).strip()

    # Detect repetition patterns (check longer patterns first)

    # 4-gram repeated 3 times
    if len(words) >= 12:
        for i in range(len(words) - 11):
            fourgram = tuple(words[i:i+4])
            found = True
            for rep in range(1, 3):
                offset = i + rep * 4
                if offset + 4 > len(words) or tuple(words[offset:offset+4]) != fourgram:
                    found = False
                    break
            if found:
                return " ".join(words[:i+4]).strip()

    # 3-gram repeated 3 times
    if len(words) >= 9:
        for i in range(len(words) - 8):
            trigram = tuple(words[i:i+3])
            found = True
            for rep in range(1, 3):
                offset = i + rep * 3
                if offset + 3 > len(words) or tuple(words[offset:offset+3]) != trigram:
                    found = False
                    break
            if found:
                return " ".join(words[:i+3]).strip()

    # 2-gram repeated 4 times
    if len(words) >= 8:
        for i in range(len(words) - 7):
            bigram = tuple(words[i:i+2])
            found = True
            for rep in range(1, 4):
                offset = i + rep * 2
                if offset + 2 > len(words) or tuple(words[offset:offset+2]) != bigram:
                    found = False
                    break
            if found:
                return " ".join(words[:i+2]).strip()

    # Single word repeated 6 times
    if len(words) >= 6:
        for i in range(len(words) - 5):
            if all(words[i] == words[i+j] for j in range(1, 6)):
                return " ".join(words[:i+1]).strip()

    return " ".join(words).strip()


# test_examples = [
#     "da je u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u tega u te",
#     "kako je svača kako je svača kako je svača kako je svača kako je svača kako je svača kako je svača kako je svača kako je svača kako je svača kako je svača kako je svača kako je svača kako je svača kako je svača kako je svača kako je svača kako je svača kako je svača kako je svača kako je svača kako je svača kako je svača kako je svača kako je svača kako je svača kako je svača kako je svača kako je svača kako je svača kako je svača kako je svača kako je svača kako je svača kako je svača kako je svača kako je svača kako je svača kako je svača kako je svača kako je svača kako je svača kako je svača kako je svača kako je svača kako je svača kako je svača kako je svača kako je svača kako je svača kako je svača kako je svača kako je svača kako je svača kako je svača kako je s",
#     "u krakovu sada sam sada sada sada sada sam sada sada sada sam sada sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sada sam sada sam sada sam sada sada sam sada sam sada sam sada sam sada sam sada sam sada sam sada sam sada sam sada sam sada sam sada sam sada sam",
#     "kako su stradaće belorusi stvarno su belorusi najviše i brojčom odstradali oni su bili prvi na udaru nemaca ali karim dok se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko se uko",
#     "u sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve od sve",
#     "da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono da je ono",
#     "u timsacom u timsacseksijama u novo saksijama u novo saksijama u novo satskom u timsacima u novo saksijama u novo saksijama u novo saksijama u novo saksijama u novo saksijama u novo saksijama u novo saksijama u novo saksijama u novo saksijama u novo saksijama u novo saksijama u novo saksijama u novo saksijama u novo saksijama u novo saksijama u novo saksijama u novo saksijama u novo saksijama u novo saksijama u novo saksijama u novo saksijama u novo saksijama u novo saksijama u novo saksijama u novo saksijama u novo saksijama u novo saksijama u novo saksijama u novo saksijama u novo saksijama u novo saksijama u novo saksijama u novo saksijama u novo saksijama u novo saksijama u novo saksijama u novo saksijama u novo saksijama u novo saksijama u novo saksijama u novo s"
# ]

