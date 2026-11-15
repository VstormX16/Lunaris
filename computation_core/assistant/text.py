"""Text normalisation and hashed n-gram features. Typo- and diacritic-tolerant, Turkish and English alike."""
import re
import zlib

import numpy as np

N_BUCKETS = 1 << 15

_FOLD = str.maketrans({"ı": "i", "İ": "i", "ş": "s", "Ş": "s", "ğ": "g", "Ğ": "g", "ç": "c", "Ç": "c", "ö": "o", "Ö": "o",
                       "ü": "u", "Ü": "u", "â": "a", "î": "i", "û": "u"})


def normalize(text: str) -> str:
    """Lower-case, fold Turkish letters to ASCII (people type 'sımdı', 'puan nıcın'), keep digits and a few symbols."""
    t = text.replace("̇", "").translate(_FOLD).lower()  # Python lower-cases 'İ' to i + a combining dot: drop the dot
    t = re.sub(r"[^a-z0-9+.:/%\-\s]", " ", t)
    t = re.sub(r"([a-z])" + chr(92) + "1{2,}", chr(92) + "1", t)  # stretched letters: heyyyy, nasilll -> hey, nasil
    return re.sub(r"\s+", " ", t).strip()


def tokens(text: str) -> list[str]:
    return [w.strip(".:-/") for w in normalize(text).split() if w.strip(".:-/")]


def _h(s: str) -> int:
    return zlib.crc32(s.encode("utf-8")) % N_BUCKETS


def features(text: str) -> np.ndarray:
    """Bucket ids of word unigrams, word bigrams and character 3-5-grams (word boundaries marked).

    Character n-grams are why 'tarihi' / 'tarhi' / 'tarihını' land on mostly the same buckets."""
    ws = tokens(text)
    ids = []
    for i, w in enumerate(ws):
        w = re.sub(r"\d", "0", w) if re.fullmatch(r"[\d.:/\-]+", w) else w  # every number looks alike to the model
        ids.append(_h("w:" + w))
        if i:
            ids.append(_h("b:" + ws[i - 1] + "_" + w))
        p = f"<{w}>"
        for n in (3, 4, 5):
            ids.extend(_h("c:" + p[j:j + n]) for j in range(max(1, len(p) - n + 1)))
    return np.array(ids if ids else [_h("empty")], dtype=np.int64)


_TR_WORDS = set("""ve bir bu su icin ile mi mu ne neden nasil hangi daha en cok biraz olur olmaz yap yapar mi misin musun
 puan arac tarih konum gunes dunya pil gece eniyi iyi kotu dusuk yuksek lutfen tamam evet hayir ben sen biz bana sana
 acikla goster karsilastir degistir tarihi araci konumu nedir demek dayanir donar yeterli guvenilir kral hocam
 merhaba selam selamlar tesekkurler tesekkur sag saol gunaydin nasilsin gorusuruz sen kimsin ne yapabilirsin neler yapar yapabilir sorarsam ozetle ozet
 anlamadim anlayamadim kafam karisti yok bos ver istemiyorum istemem vazgec vazgectim berbat kotusun sacma sacmalama rezalet emin yalan uydurma nerden nereden tekrar devam detay ayrintili sade basit kisaca abi valla yani hic ise yaramiyor yaramaz aptal salak gerek kabul aynen tabi tabii oldu olsun peki sonra bi daha soyle anlat anlatir bakalim tamam olur hadi nerede nereye daha iyi nasil niye nicin bugun yarin hafta ocak subat mart nisan mayis haziran temmuz agustos eylul ekim kasim aralik""".split())
_EN_WORDS = set("""the a an and or is are was be what why how which can could should would please explain show compare change
 date vehicle site location sun earth battery night best better score low high thanks hello hi yes no it this that for with
 does do my me you your""".split())


def detect_language(text: str, default: str = "en") -> str:
    """'tr' or 'en', from Turkish letters and common words (a stopword vote; the NLU model itself is bilingual)."""
    if re.search(r"[ıİşŞğĞçÇöÖüÜ]", text):
        return "tr"
    ws = tokens(text)
    tr, en = sum(w in _TR_WORDS for w in ws), sum(w in _EN_WORDS for w in ws)
    if re.search(r"(lar|ler|dir|dur|miyor|muyor|iyor|acak|ecek|mis|mus|siz|siniz)\b", normalize(text)):
        tr += 1
    return "tr" if tr > en else "en" if en > tr else default
