"""Training corpus builder: nlu_data's base sentences + more wording, placeholders, synonym swaps, typos, entity masking.

Placeholders {D} date, {V} vehicle, {S} site, {M} multiplier and {N} days are filled with real mentions and then masked by
entities.extract, so the model learns the *shape* of a request ('change the date to <date>') whatever the actual date is.
"""
import numpy as np

from .entities import extract
from .nlu_data import CONTEXT_BITS, CONTEXTUAL, INTENTS as BASE, PREFIX, SUFFIX, _human, _noise

MORE = {
    "greet": {"en": ["hello how are you doing", "hi assistant how is it going", "good day", "hey buddy", "morning", "hello again"], "tr": ["selam nasilsin", "merhaba nasil gidiyor", "gunaydin nasilsin", "selam dostum", "gunaydin", "tekrar merhaba", "kral selam"]},
    "thanks": {"en": ["that was helpful thanks", "thanks for the help", "many thanks", "you are great thank you", "thx"], "tr": ["cok yardimci oldun sag ol", "yardimin icin tesekkurler", "valla sag ol", "tesekkurler kral", "sagolasin", "super oldu eyvallah"]},
    "bye": {"en": ["i have to go now", "ok that is enough for today", "bye bye", "later"], "tr": ["ben gidiyorum", "bugunluk bu kadar yeter", "hadi bb", "kendine iyi bak gorusuruz", "cikiyorum"]},
    "about": {"en": ["what are you capable of", "what kind of questions do you answer", "who made you", "are you chatgpt", "are you built from scratch", "what is your purpose"], "tr": ["neler yapabiliyorsun", "hangi sorulara cevap veriyorsun", "seni kim gelistirdi", "chatgpt misin", "sifirdan mi yapildin", "senin amacin ne", "bu asistan nasil calisiyor"]},
    "smalltalk": {"en": ["how is your day", "are you bored", "do you get tired", "what do you think about space", "can you tell me something fun", "you are cool"], "tr": ["gunun nasil geciyor", "sikildin mi", "yorulur musun", "uzay hakkinda ne dusunuyorsun", "eglenceli bir sey anlat", "baya havalisin", "keyfin nasil"]},
    "explain_result": {"en": ["what is this page showing", "make sense of these numbers for me", "tell me the story of this result", "give me the short version", "what should i take away from this", "i am lost can you explain"], "tr": ["bu sayfa ne gosteriyor", "su rakamlari benim icin yorumla", "bu sonucun hikayesini anlat", "kisaca ozetler misin", "bundan ne anlamaliyim", "kafam karisti anlatir misin", "sonuc nedir"]},
    "why_score": {"en": ["what is the reason for this score", "explain the low result", "where does the score come from", "which factor is hurting us", "what went wrong with the score", "why does it look so bad"], "tr": ["bu puanin sebebi ne", "dusuk sonucu acikla", "puan nereden geliyor", "hangi etken bizi asagi cekiyor", "puanda ne ters gitti", "niye bu kadar kotu gorunuyor", "puan niye yerlerde", "bunu dusuren faktor ne"]},
    "improve": {"en": ["what can be done to get a higher score", "any suggestions to make it better", "how do we turn this around", "what are the ways to raise it", "can we do something about it", "what would you change"], "tr": ["puani yukseltmek icin ne yapilabilir", "daha iyi olmasi icin onerin var mi", "bunu nasil tersine ceviririz", "yukseltmenin yollari neler", "bununla ilgili bir sey yapabilir miyiz", "sen ne degistirirdin", "daha iyi sonuc almanin yolu ne"]},
    "best_date": {"en": ["what is the optimal date", "pick the best day for me", "should we wait a few days", "look at the neighbouring dates", "which start date works best", "what is the best moment to start", "if we start earlier or later is it better"], "tr": ["en uygun gun hangisi", "bana en iyi gunu sec", "birkac gun beklemeli miyiz", "yakin tarihlere bak", "hangi baslangic tarihi en iyi calisir", "baslamak icin en iyi an ne zaman", "daha erken ya da gec baslasak daha mi iyi", "hangi tarihte baslarsak daha mantikli", "tarihi oynatsak ne olur", "tarih araligini tara"]},
    "compare_dates": {"en": ["what if we started on {D}", "how would {D} look", "compare today with {D}", "check {D} for me", "{D} instead of the current date", "what changes if the date is {D}", "run the analysis for {D}"], "tr": ["{D} tarihinde baslarsak ne olur", "{D} nasil gorunur", "bugunle {D} i karsilastir", "{D} icin bak", "su anki tarih yerine {D}", "tarih {D} olursa ne degisir", "analizi {D} icin calistir", "{D} olsa nasil olur", "{D} ile {D} karsilastir", "{D} yerine {D} olsa ne degisir"]},
    "best_vehicle": {"en": ["which lander does best at this spot", "could {V} do better here", "compare {V} with the others at this location", "what would be the ideal vehicle here", "try all the vehicles at this site"], "tr": ["bu noktada hangi arac en iyisi", "{V} burada daha iyi olur mu", "{V} i diger araclarla karsilastir", "burasi icin ideal arac hangisi", "tum araclari bu konumda dene", "hangi lander bu yere en uygun"]},
    "best_site": {"en": ["where is the best place to land this vehicle", "can we land somewhere else", "how about {S}", "compare the sites", "check around this point for a safer place", "is {S} better than here", "compare {S} and {S}"], "tr": ["bu araci nereye indirmek en iyi", "baska bir yere inebilir miyiz", "{S} nasil", "konumlari karsilastir", "bu noktanin cevresinde daha guvenli bir yer var mi", "{S} burdan iyi mi", "yakin cevrede daha iyi yer ara", "{S} ve {S} i karsilastir"]},
    "sun_info": {"en": ["how much daylight does it get", "what is the sunlit percentage", "how long until the sun sets", "tell me about the light conditions", "is it dark most of the time"], "tr": ["ne kadar gun isigi aliyor", "aydinlik yuzdesi kac", "gunes ne zaman batiyor", "isik kosullari nasil", "cogu zaman karanlik mi", "gunes acisi kac derece", "gunes kac derecede vuruyor"]},
    "comms_info": {"en": ["can we reach the ground station", "how good is the link with earth", "is there line of sight to earth", "how long without contact", "do we need a relay satellite"], "tr": ["yer istasyonuna ulasabiliyor muyuz", "dunya ile baglanti ne kadar iyi", "dunyaya gorus hatti var mi", "ne kadar sure temassiz kaliriz", "role uydusu lazim mi", "iletisim engeli var mi", "dunya yonunde tepe var mi"]},
    "slope_info": {"en": ["how steep is the landing zone", "will it tip over on this ground", "is the surface flat", "what is the incline here"], "tr": ["inis bolgesi ne kadar dik", "bu zeminde devrilir mi", "yuzey duz mu", "buradaki meyil ne", "zemin egimi nasil", "arazi tehlikeli mi"]},
    "night_thermal": {"en": ["will it be too cold in the dark", "can the lander keep warm", "how bad is the lunar night", "is there a thermal problem"], "tr": ["karanlikta cok soguk olur mu", "arac isinabilir mi", "ay gecesi ne kadar kotu", "isil bir sorun var mi", "gece uyuyup bir daha uyanir mi", "gece boyunca isinma nasil"]},
    "battery_info": {"en": ["is the battery large enough", "what if the battery were {M} times bigger", "does it have enough power for the whole window", "how long can it run on batteries"], "tr": ["batarya yeterince buyuk mu", "batarya {M} kat buyuk olsa", "butun pencere icin yeterli gucu var mi", "pillerle ne kadar calisir", "pil kapasitesi yeter mi", "pili {M} kat yaparsak"]},
    "data_quality": {"en": ["how accurate is all this", "which numbers are guesses", "what is verified and what is not", "can i rely on this analysis", "is this real nasa data"], "tr": ["bunlar ne kadar dogru", "hangi rakamlar tahmin", "neler dogrulanmis neler degil", "bu analize guvenebilir miyim", "bu gercek nasa verisi mi", "veriler ne kadar saglam"]},
    "glossary": {"en": ["what does dte stand for", "define the suitability index", "what is meant by a veto gate", "what is illumination", "explain the term slope limit", "what does skyline mean"], "tr": ["dte neyin kisaltmasi", "uygunluk indeksini tanimla", "veto kapisi ne demek", "aydinlanma nedir", "egim siniri terimini acikla", "ufuk cizgisi ne demek", "sunlit nedir", "termi acikla"]},
    "table_request": {"en": ["can i see that in a table", "make a table out of it", "organize it into a table", "show the numbers in a table"], "tr": ["bunu tabloda gorebilir miyim", "bundan tablo yap", "tabloya donustur", "rakamlari tabloda goster", "tablolastir"]},
    "apply": {"en": ["sounds good do it", "go for it", "yes use the suggestion", "alright set it", "ok apply this one", "sure"], "tr": ["kulaga hos geliyor yap", "hadi bakalim yap", "evet oneriyi kullan", "tamamdir ayarla", "olur bunu uygula", "tamam gec", "olur o tarihe gec", "peki o zaman uygula"]},
    "change_context": {"en": ["change the date to {D}", "set the vehicle to {V}", "switch the location to {S}", "use {V} instead", "start on {D}", "make the date {D}", "set the site to {S}", "switch to {V}", "move the date by {N} days"], "tr": ["tarihi {D} yap", "araci {V} olarak ayarla", "konumu {S} yap", "onun yerine {V} kullan", "baslangici {D} yap", "tarihi {D} olarak degistir", "konumu {S} olarak ayarla", "{V} a gec", "tarihi {N} gun kaydir"]},
    "unknown": {"en": ["what is the meaning of life", "how tall is mount everest", "help me with my homework", "tell me a story about dragons", "what is the best programming language", "how many people live in london"], "tr": ["hayatin anlami ne", "everest kac metre", "odevime yardim et", "bana ejderhalar hakkinda hikaye anlat", "en iyi programlama dili hangisi", "istanbulda kac kisi yasiyor", "bu aksam ne yesem", "futbol skoru kac"]},
}
_TERMS = {"en": ["dte", "the veto", "slope limit", "illumination", "the skyline", "a relay satellite", "the lunar night", "the 30 day window", "a heat source", "the index", "held down by", "the horizon", "sunlit share"],
          "tr": ["dte", "veto", "egim siniri", "aydinlanma orani", "ufuk cizgisi", "role uydusu", "ay gecesi", "30 gunluk pencere", "isi kaynagi", "uygunluk indeksi", "held down by", "horizon", "sunlit"]}
_GLOSS_FRAMES = {"en": ["what is {T}", "what does {T} mean", "explain {T}", "define {T}", "tell me what {T} is", "what do you mean by {T}"],
                 "tr": ["{T} nedir", "{T} ne demek", "{T} kavrami nedir", "{T} terimini acikla", "{T} ne anlama geliyor", "{T} neyin nesi"]}
MORE2 = {
    "glossary": {l: [f.replace("{T}", t) for f in _GLOSS_FRAMES[l] for t in _TERMS[l]] for l in ("en", "tr")},
    "greet": {"en": ["hello can you help me", "hi i have a question", "hello i need some help", "hey assistant are you there", "hi how are you", "good morning assistant"], "tr": ["selam bir sorum var", "merhaba yardim eder misin", "selam bi bakar misin", "naber hocam", "merhaba bir sey soracagim", "selam asistan orada misin", "gunaydin kral"]},
    "thanks": {"en": ["thanks buddy", "thank you very much", "ok cool thanks", "great job thanks"], "tr": ["eyv kral", "eyvallah hocam", "tesekkurler hocam", "sagol kral", "cok tesekkurler", "eyv"]},
    "about": {"en": ["how do you work", "how does this assistant work", "what is your name", "what are you made of", "how were you trained", "can you only talk about this analysis", "what are you good at"], "tr": ["nasil calisiyorsun", "bu asistan nasil calisiyor", "adin ne", "neyle yapildin", "nasil egitildin", "sadece bu analiz hakkinda mi konusabilirsin", "en iyi oldugun sey ne"]},
    "smalltalk": {"en": ["you seem nice", "you are a good assistant", "i like talking to you", "what is your favorite planet", "do you ever sleep", "how is the weather up there on the moon"], "tr": ["hava nasil orada", "baya iyi bir asistansin", "seninle konusmak guzel", "en sevdigin gezegen hangisi", "hic uyur musun", "ay'da hava nasil", "bugun keyfin nasil"]},
    "best_date": {"en": ["should we wait a bit before landing", "should we delay it by some days", "is it better to go earlier or later", "which part of the month is best", "would waiting a few days help", "pick another day that scores higher", "what day would you choose instead"], "tr": ["inisi biraz bekletsek mi", "birkac gun ertelesek iyi olur mu", "daha erken mi gec mi gitmek daha iyi", "ayin hangi donemi en iyi", "birkac gun beklemek ise yarar mi", "skoru daha yuksek baska bir gun sec", "sen hangi gunu secerdin", "baska bir gun secsek skor artar mi", "hangi tarihte baslamak en verimli"]},
    "compare_dates": {"en": ["how about starting on {D}", "what about {D}", "and for {D}", "look at {D} as well", "now try {D}", "same thing for {D}", "what would {D} give"], "tr": ["{D} olursa nasil olur", "ya {D} olsa", "{D} icin de bak", "bide {D} dene", "ayni seyi {D} icin yap", "{D} ne verir", "iki hafta sonrasina bakalim", "bir ay sonra nasil olur", "{D} e bakalim"]},
    "best_vehicle": {"en": ["which craft gives the best result in this region", "pick the strongest lander for this place", "compare all the spacecraft here", "who would do best in this spot"], "tr": ["hangi arac bu bolgede en iyi sonuc veriyor", "bu yer icin en guclu araci sec", "tum uzay araclarini burada karsilastir", "bu noktada kim en iyi yapar", "araclari karsilastirsana"]},
    "best_site": {"en": ["suggest a different landing point", "give me another landing spot", "is there another place that works better", "can you suggest a safer landing zone"], "tr": ["farkli bir inis noktasi onerir misin", "baska bir inis yeri ver", "daha iyi calisan baska bir yer var mi", "daha guvenli bir inis bolgesi onerir misin", "bu noktanin yakininda daha uygun yer"]},
    "explain_result": {"en": ["sum up the findings", "recap the analysis", "what are the main takeaways", "give me the bottom line", "tldr of this result"], "tr": ["bulgulari ozetle", "analizi toparla", "ana sonuclar neler", "bana ozetini ver", "kisaca sonuc ne"]},
    "night_thermal": {"en": ["is there a risk of freezing", "will the cold damage it", "could it freeze during the night", "how does it handle the cold"], "tr": ["donma riski var mi", "soguk ona zarar verir mi", "gece boyunca donabilir mi", "soguga nasil dayanir"]},
    "unknown": {"en": ["what is the highest mountain on earth", "write python code for me", "will it rain tomorrow", "who is the best football player", "how do i lose weight", "what is the speed of light", "tell me about the roman empire", "what should i cook tonight", "how many moons does mars have", "which phone should i buy"], "tr": ["dunyanin en yuksek dagi hangisi", "bana python kodu yaz", "yarin yagmur yagacak mi", "en iyi futbolcu kim", "nasil kilo veririm", "isigin hizi ne kadar", "roma imparatorlugunu anlat", "bu aksam ne pisirsem", "marsin kac uydusu var", "hangi telefonu almaliyim", "tarih dersi icin yardim et", "kediler neden mirildar"]},
}
for _k, _v in MORE2.items():
    for _l in ("en", "tr"):
        MORE[_k][_l] = MORE[_k].get(_l, []) + _v[_l]

INTENTS = {k: {l: BASE[k][l] + MORE.get(k, {}).get(l, []) for l in ("en", "tr")} for k in BASE}

# what a vehicle must have to cope at a place / date (answered from the real night length, Sun height, Earth gaps and slopes)
INTENTS["requirements"] = {
    "en": ["what features should the vehicle have", "what does a lander need to survive at {S}", "what are the requirements for a lander at {S}", "what kind of lander would work at {S} on {D}",
           "what specs should a rover have for this mission", "what should a vehicle be able to do to work there", "how big a battery does it need there", "does it need a heater at {S}",
           "what would i have to build to land at {S}", "design requirements for a south pole lander", "what must the spacecraft have to last the night", "which capabilities does the mission need",
           "list the minimum requirements for the vehicle", "what equipment is needed for {D}", "how should we design the lander for {S}", "what would a vehicle need to cope with these conditions",
           "what do i need on board to get through the darkness", "what kind of power system is needed there", "what should the solar panels look like at {S}", "what slope tolerance does it need there"],
    "tr": ["arac hangi ozelliklere sahip olmali", "{S} da hayatta kalmak icin arac neye ihtiyac duyar", "{S} icin lander gereksinimleri neler", "{D} tarihinde {S} icin nasil bir arac lazim",
           "bu gorev icin aracin teknik ozellikleri ne olmali", "orada calismasi icin aracin neyi olmali", "orada ne kadar buyuk bir batarya gerekir", "{S} da isitici gerekir mi",
           "{S} ya inmek icin nasil bir arac yapmaliyim", "guney kutbu araci icin tasarim gereksinimleri", "geceyi atlatmak icin aracta ne olmali", "gorevin ihtiyac duydugu yetenekler neler",
           "arac icin asgari gereksinimleri listele", "{D} icin hangi ekipman gerekli", "landeri {S} icin nasil tasarlamaliyiz", "bu kosullara dayanmak icin aracin nesi olmali",
           "karanlikta dayanmak icin aracin neye ihtiyaci var", "orada nasil bir guc sistemi gerekir", "{S} da gunes panelleri nasil olmali", "orada egim toleransi kac derece olmali",
           "nasil bir arac gondermeliyiz ki dayansin", "aracin sahip olmasi gereken ozellikler neler"],
}

# A plan said before the question ('on 16 Nov I want to go to the south pole, which vehicle suits it?'). The label is always the
# question's: the plan alone asks nothing. These carriers are prepended to task questions during training (generate).
CARRIER = {
    "en": ["i want to go to the lunar south pole on {D}", "we are planning to land at {S} on {D}", "we plan a mission to the south pole", "i am sending a lander to {S}",
           "say we land near the pole on {D}", "our team wants to reach {S} in {D}", "for a polar mission starting {D}", "imagine a landing at {S}"],
    "tr": ["{D} tarihinde ayin guney kutbuna gitmek istiyorum", "{D} de {S} bolgesine inis planliyoruz", "guney kutbuna bir gorev dusunuyoruz", "{S} ya bir arac gondermek istiyorum",
           "diyelim ki {D} de kutba indik", "ekibimiz {D} de {S} ya ulasmak istiyor", "{D} de baslayacak bir kutup gorevi icin", "kral {S} ya inis yapacagiz"],
}
CARRIED = {"best_vehicle", "best_date", "best_site", "requirements", "explain_result", "why_score", "improve", "sun_info", "comms_info", "slope_info", "night_thermal", "battery_info"}

SYN = {  # light synonym swaps so the model does not hang on one word ('puan' vs 'skor' vs 'sonuc')
    "tr": {"puan": ["skor", "sonuc", "not", "indeks"], "puani": ["skoru", "sonucu"], "neden": ["niye", "nicin", "ne yuzden"], "dusuk": ["kotu", "berbat", "az"], "kotu": ["berbat", "dusuk"],
           "tarih": ["gun", "zaman"], "tarihi": ["gunu", "zamani"], "arac": ["lander", "uzay araci"], "iyi": ["uygun", "mantikli", "yuksek"], "daha": ["biraz daha", "daha da"],
           "nasil": ["ne sekilde", "hangi yolla"], "acikla": ["anlat", "yorumla"], "goster": ["ver", "sun"], "konum": ["nokta", "yer", "bolge"], "konumu": ["noktayi", "yeri"],
           "gunes": ["isik"], "dunya": ["yer"], "iletisim": ["haberlesme", "baglanti"], "egim": ["meyil", "dikilik"], "pil": ["batarya"], "batarya": ["pil"], "yap": ["et"]},
    "en": {"score": ["result", "rating", "index"], "why": ["how come", "what makes"], "low": ["poor", "bad", "weak"], "date": ["day", "time"], "best": ["optimal", "ideal", "top"],
           "better": ["higher", "improved"], "vehicle": ["lander", "spacecraft", "craft"], "site": ["location", "spot", "place"], "explain": ["describe", "walk me through"],
           "show": ["give", "display"], "improve": ["raise", "boost", "fix"], "battery": ["power pack"], "communication": ["link", "contact"]},
}
_D = {"en": ["december 5", "dec 12", "january 3", "2026-12-05", "12.12.2026", "next week", "10 days later", "1 february", "tomorrow", "march 2 2027"],
      "tr": ["5 aralik", "12 aralikta", "3 ocak", "2026-12-05", "12.12.2026", "gelecek hafta", "10 gun sonra", "1 subat", "yarin", "2 mart 2027"]}
_V = ["apollo 15", "chandrayaan 3", "im-1", "luna 25", "chang'e 3", "blue ghost", "slim", "beresheet", "lunokhod 1", "surveyor 1", "athena", "peregrine"]
_S = ["shackleton", "haworth", "malapert", "faustini", "nobile", "-86.2 20.5", "-88 45"]
_M = ["2", "3", "4", "8", "iki", "two", "ten"]


def _pick(xs, rng):
    return xs[int(rng.integers(len(xs)))]


def _fill(core, lang, rng):
    return (core.replace("{D}", _pick(_D[lang], rng)).replace("{V}", _pick(_V, rng)).replace("{S}", _pick(_S, rng))
            .replace("{M}", _pick(_M, rng)).replace("{N}", _pick(["3", "7", "10", "14"], rng)))


def _syn(s, lang, rng, p=0.3):
    table = SYN[lang]
    return " ".join(_pick(table[w], rng) if w in table and rng.random() < p else w for w in s.split())


FEEDBACK_PATH = None  # set below (a data file of real questions that the model once got wrong, with their correct labels)


def _feedback():
    import json
    from pathlib import Path
    p = Path(__file__).resolve().parents[2] / "data" / "assistant_feedback.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else []


def _finish(s, lang, rng):
    """Politeness words, typos, a dropped word, then the entity mask: how a sentence reaches the model."""
    s = _pick(PREFIX[lang], rng) + s + _pick(SUFFIX[lang], rng)
    if rng.random() < 0.5:
        s = _human(s, lang, rng)
    if rng.random() < 0.45:
        s = _noise(s, rng)
    ws = s.split()
    if len(ws) > 4 and rng.random() < 0.2:  # word dropout
        ws.pop(int(rng.integers(len(ws))))
    return extract(" ".join(ws)).masked


def generate(n_per_intent: int = 900, seed: int = 0, n_per_topic: int = 420, feedback_copies: int = 40):
    """(sentences, intents, topics). Base sentences, plus grammar-composed request x topic sentences, plus real feedback."""
    from .nlu_grammar import compose, topic_for
    from .text import detect_language
    rng = np.random.default_rng(seed)
    X, y, T = [], [], []
    for intent, langs in INTENTS.items():
        for _ in range(n_per_intent):
            lang = "tr" if rng.random() < 0.5 else "en"
            core = _fill(_pick(langs[lang], rng), lang, rng)
            if rng.random() < 0.5:
                core = _syn(core, lang, rng)
            if rng.random() < 0.15:  # code-mixed sentences: people drop English words into Turkish ones
                olang = "en" if lang == "tr" else "tr"
                core += " " + _fill(_pick(langs[olang], rng), olang, rng).split()[-1]
            if intent in CONTEXTUAL and rng.random() < 0.3:
                core += " " + _pick(CONTEXT_BITS[lang], rng)
            if intent in CARRIED and rng.random() < 0.2:  # the mission plan first, then the question
                core = _fill(_pick(CARRIER[lang], rng), lang, rng) + _pick([", ", " ", ". ", " ve "] if lang == "tr" else [", ", " ", ". ", " and "], rng) + core
            X.append(_finish(core, lang, rng))
            y.append(intent)
            T.append(topic_for(intent, core))
    for sent, intent, topic in compose(rng, n_per_topic):
        lang = detect_language(sent, "en")
        X.append(_finish(_syn(sent, lang, rng, 0.15), lang, rng))
        y.append(intent)
        T.append(topic)
    for item in _feedback():  # real questions the assistant once failed on, with the right answer written by hand
        lang = detect_language(item["text"], "en")
        for _ in range(feedback_copies):
            X.append(_finish(item["text"], lang, rng) if rng.random() < 0.7 else extract(item["text"]).masked)
            y.append(item["intent"])
            T.append(item.get("topic", "none"))
    from .chips import all_chips  # the buttons the app offers: typed or clicked, they must always be understood
    from .nlu_grammar import TOPIC_OF_INTENT
    for text, intent in all_chips():
        lang = detect_language(text, "en")
        for _ in range(30):
            X.append(_finish(text, lang, rng) if rng.random() < 0.6 else extract(text).masked)
            y.append(intent)
            T.append(TOPIC_OF_INTENT.get(intent, "none"))
    return X, y, T
