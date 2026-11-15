"""A small grammar that composes (kind of request) x (topic) into many sentences, in Turkish and English.

Hand-writing one sentence per question does not scale: the user can ask 'how do I raise the sun angle', 'how do I shorten the
night', 'what lowers the slope' ... Here each topic has several ways of being named and each kind of request several frames, so
the model sees thousands of combinations and learns that the *topic words* and the *request words* are separate things.
"""
import numpy as np

TOPICS = ["none", "score", "sun_angle", "sun_light", "earth", "slope", "night", "battery", "date", "vehicle", "site"]

# phrases a person might use for each quantity (ASCII-folded, like the model's input)
LEX = {
    "sun_angle": {"en": ["the sun elevation", "the sun angle", "the elevation angle of the sun", "how high the sun is", "the height of the sun above the horizon", "the solar elevation", "the angle of the sunlight", "the sun's altitude"],
                  "tr": ["gunes ufuk acisi", "gunesin ufuk acisi", "gunes acisi", "gunesin yuksekligi", "gunesin ufuktan yuksekligi", "gunes elevasyonu", "isik gelis acisi", "gunesin gokyuzundeki yuksekligi", "gunes ufuk acim", "gunesin acisi", "ufuk acisi"]},
    "sun_light": {"en": ["the sunlight", "the daylight", "the time in the sun", "the illumination", "the sunlit share", "the hours of sun", "the amount of sun"],
                  "tr": ["gunes isigi", "aydinlik suresi", "gunes alma suresi", "aydinlik orani", "gunesli zaman", "gun isigi suresi", "gunes miktari"]},
    "earth": {"en": ["the earth link", "the earth visibility", "the communication time", "the contact with earth", "the time in view of earth", "the line of sight to earth", "the radio link"],
              "tr": ["dunya baglantisi", "dunya gorunurlugu", "dunya ile iletisim", "iletisim suresi", "dunyayi gorme suresi", "haberlesme", "telsiz baglantisi", "dunya gorus hatti"]},
    "slope": {"en": ["the slope", "the ground slope", "the steepness", "the terrain steepness", "the tilt of the ground", "the incline"],
              "tr": ["egim", "zemin egimi", "meyil", "dikilik", "arazi dikligi", "inis alani egimi"]},
    "night": {"en": ["the night length", "the darkness period", "the longest night", "the dark time", "the duration of the night", "the long darkness"],
              "tr": ["gece suresi", "karanlik suresi", "gece uzunlugu", "en uzun gece", "karanlik donem", "karanlik sure"]},
    "battery": {"en": ["the battery", "the power budget", "the battery life", "the energy", "the battery capacity", "the power situation"],
                "tr": ["pil", "batarya", "pil yeterliligi", "enerji", "guc durumu", "pil omru"]},
    "score": {"en": ["the score", "the overall score", "the result", "the suitability", "the rating", "the grade"],
              "tr": ["puan", "skor", "sonuc", "uygunluk puani", "genel sonuc", "not"]},
}
# True: a bigger number is better; False: smaller is better (changes the verb: raise / reduce)
HIGHER_IS_BETTER = {"sun_angle": True, "sun_light": True, "earth": True, "slope": False, "night": False, "battery": True, "score": True}

UP = {
    "en": ["what can i do to increase {T}", "how can i raise {T}", "how do i improve {T}", "what would improve {T}", "make {T} better", "i want higher {T}", "ways to boost {T}",
           "what should i change to get more of {T}", "how do we increase {T}", "what would raise {T}", "can i do anything to improve {T}", "how to get better {T}"],
    "tr": ["ne yaparsam {T} artar", "{T} nasil artar", "{T} artirmak icin ne yapmaliyim", "{T} nasil artirilir", "{T} yukseltmenin yolu ne", "{T} daha iyi olsun istiyorum",
           "{T} iyilestirmek icin ne yapabilirim", "{T} icin ne yapsam", "{T} yukselsin istiyorum ne yapayim", "{T} nasil yukselir", "ne yapsak {T} daha iyi olur", "{T} yukseltmek icin neyi degistirmeliyim"],
}
DOWN = {
    "en": ["how can i reduce {T}", "what would shorten {T}", "how do i lower {T}", "how to get less of {T}", "what can i do to decrease {T}", "i want lower {T}", "what would cut down {T}", "how do we reduce {T}"],
    "tr": ["{T} nasil azalir", "{T} azaltmak icin ne yapmali", "{T} nasil kisalir", "{T} nasil dusurulur", "ne yaparsam {T} azalir", "{T} kisalsin istiyorum ne yapayim", "{T} dusurmek icin neyi degistirmeliyim", "{T} nasil azaltirim"],
}
INFO = {  # a question about the current state: topic -> the intent that answers it
    "en": ["how is {T}", "tell me about {T}", "what is {T} here", "what about {T}", "how good is {T}", "i am curious about {T}"],
    "tr": ["{T} nasil", "{T} hakkinda bilgi ver", "{T} ne durumda", "{T} ne kadar", "{T} ile ilgili bilgi", "{T} nasil gorunuyor"],
}
WHY = {
    "en": ["why is {T} so low", "why is {T} poor", "what is wrong with {T}", "why is {T} bad here"],
    "tr": ["{T} neden dusuk", "{T} niye kotu", "{T} ile ilgili sorun ne", "{T} neden bu kadar az", "{T} neden kotu cikti"],
}
# Nouns that are NOT about the analysis. They go into the same request frames as the real topics ('how can i improve {T}'),
# labelled unknown, so the model cannot conclude that 'how can i ...' alone means an improvement question about the analysis.
OFF_TOPIC = {
    "en": ["my salary", "my english", "my cooking", "the taste of the soup", "my football skills", "my health", "my grades", "my sleep", "my car", "the traffic in my city", "my guitar playing", "my phone battery life at home", "the price of bitcoin", "my garden", "my social media followers"],
    "tr": ["maasimi", "ingilizcemi", "yemeklerimi", "corbanin tadini", "futbol becerimi", "sagligimi", "notlarimi", "uykumu", "arabamin hizini", "sehrimdeki trafigi", "gitar calmami", "bitcoin fiyatini", "bahcemi", "takipci sayimi", "kilomu"],
}
OFF_VP = {  # whole off-topic aims, for frames like 'how do i ...' / 'ne yaparsam ...'
    "en": ["learn french", "sleep better", "save more money", "lose weight", "run faster", "cook better", "write better essays", "make new friends", "pass my driving test", "grow tomatoes", "learn to draw", "play the piano well", "fix my posture", "get over a cold"],
    "tr": ["fransizca ogrenirim", "daha iyi uyurum", "daha cok para biriktiririm", "kilo veririm", "daha hizli kosarim", "daha iyi yemek yaparim", "yeni arkadaslar edinirim", "ehliyet sinavini gecerim", "domates yetistirebilirim", "resim yapmayi ogrenirim", "piyano calarim", "durusumu duzeltirim", "soguk algınligindan kurtulurum", "satranc oynarim"],
}
OFF_VP_FRAMES = {"en": ["how do i {V}", "what can i do to {V}", "how can i {V}", "what should i do to {V}", "any tips to {V}"], "tr": ["ne yaparsam {V}", "ne yapsam {V}", "nasil {V}", "ne yapmaliyim ki {V}"]}
OFF_TASKS = {  # whole requests that have nothing to do with the analysis
    "en": ["calculate {N} times {M}", "convert {N} miles to kilometres", "how do i get to the {P}", "how do i get to {P}", "what is {N} plus {M}", "solve this equation for me", "give me a recipe for {F}", "who won the {G} yesterday", "translate this to german"],
    "tr": ["{N} kere {M} kac eder", "{N} mili kilometreye cevir", "{P} nasil gidilir", "{N} arti {M} kac eder", "bu denklemi coz", "{F} tarifi ver", "dun {G} kim kazandi", "bunu almancaya cevir"],
}
_FILL = {"{N}": ["7", "12", "17", "23", "100"], "{M}": ["3", "8", "19", "23"], "{P}": ["airport", "station", "the museum", "havalimani", "gar", "muze"], "{F}": ["pasta", "cake", "pilav", "baklava", "soup"], "{G}": ["match", "game", "race", "mac", "yaris"]}

INFO_INTENT = {"sun_angle": "sun_info", "sun_light": "sun_info", "earth": "comms_info", "slope": "slope_info", "night": "night_thermal", "battery": "battery_info", "score": "explain_result"}
# which topic each of the older single-intent examples belongs to
TOPIC_OF_INTENT = {"sun_info": "sun_light", "comms_info": "earth", "slope_info": "slope", "night_thermal": "night", "battery_info": "battery",
                   "best_date": "date", "compare_dates": "date", "best_vehicle": "vehicle", "best_site": "site", "try_option": "none", "why_score": "score", "explain_result": "score", "improve": "score"}
ANGLE_WORDS = ("aci", "derece", "angle", "elevation", "degree", "height", "yuksek")


def topic_for(intent: str, core: str) -> str:
    t = TOPIC_OF_INTENT.get(intent, "none")
    if intent == "sun_info" and any(w in core for w in ANGLE_WORDS):
        t = "sun_angle"
    return t


def compose(rng, n_per_topic: int = 420):
    """(sentence, intent, topic) triples: improve / why / info requests about each topic."""
    pick = lambda xs: xs[int(rng.integers(len(xs)))]  # noqa: E731
    out = []
    for topic, lex in LEX.items():
        for _ in range(n_per_topic):
            lang = "tr" if rng.random() < 0.5 else "en"
            term = pick(lex[lang])
            r = rng.random()
            if r < 0.62:
                frame = pick(UP[lang] if HIGHER_IS_BETTER[topic] else DOWN[lang])
                if rng.random() < 0.12:  # people sometimes use the 'wrong' verb: 'increase the slope' for 'make it bigger/worse' is rare, so keep it small
                    frame = pick(UP[lang] + DOWN[lang])
                out.append((frame.replace("{T}", term), "improve", topic))
            elif r < 0.82:
                out.append((pick(INFO[lang]).replace("{T}", term), INFO_INTENT[topic], topic))
            else:
                out.append((pick(WHY[lang]).replace("{T}", term), "why_score", topic))
    for _ in range(int(n_per_topic * 1.4)):  # off-topic requests in the very same frames
        lang = "tr" if rng.random() < 0.5 else "en"
        r = rng.random()
        if r < 0.4:
            frame = pick(UP[lang] + DOWN[lang] + INFO[lang] + WHY[lang])
            out.append((frame.replace("{T}", pick(OFF_TOPIC[lang])), "unknown", "none"))
        elif r < 0.7:
            out.append((pick(OFF_VP_FRAMES[lang]).replace("{V}", pick(OFF_VP[lang])), "unknown", "none"))
        else:
            task = pick(OFF_TASKS[lang])
            for k, v in _FILL.items():
                task = task.replace(k, pick(v))
            out.append((task, "unknown", "none"))
    return out
