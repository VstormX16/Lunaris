"""Dialogue manager: understands the message (own NLU model + entity rules), reads the page context, calls the engine, and
writes the answer in plain language (Turkish or English) with tables when they help.

Everything the assistant states as a number was just computed by the engine; the language model only decides *what to
do* and the templates below decide *how to say it*.
"""
import random
import re
from datetime import date, datetime, timedelta

import numpy as np

from ..mission_risk import N_SAMPLES, mission_risk
from . import tools
from .chips import route_chip
from .reactions import detect as detect_reaction
from .entities import SITES, extract
from .glossary import TERMS
from .nlu_frames import clauses, concepts, frame
from .text import detect_language, normalize
from .tools import Scenario

CONF_MIN = 0.5
AGREE_MIN = 0.6  # share of ensemble members that must pick the same intent (a majority of the 5)
COMMAND = re.compile(r"\b(yap|yapin|ayarla|degistir|guncelle|olsun|sec|kullan|gec|uygula|set|switch|use|change|update|make|let|apply|swap|replace|put|move)\b")
REF_DATE = re.compile(r"\b(o tarih|onerilen tarih|en iyi tarih|that date|the suggested date|the best date)")
SLOT_AWARE = {"explain_result", "why_score", "improve", "sun_info", "comms_info", "slope_info", "night_thermal", "battery_info", "data_quality"}
DEFAULT_VEHICLE = "blue-ghost-mission-1"  # free chat, when the user has not named a vehicle and the question needs one
NO_SIMPLIFY = {"greet", "thanks", "bye", "smalltalk", "about", "unknown", "clarify", "affirm", "deny", "complaint", "repeat", "elaborate", "glossary", "confused", "doubt", "apply", "table_request", "change_context"}
KEEP_TABLE = {"compare_results", "best_vehicle", "best_site", "best_date"}
NOTICED = re.compile(r"noticed on my own|Kendiliğimden")
JARGON = [(re.compile(r"\s*\[[^\]]*(unverified|doğrulanmamış)[^\]]*\]"), ""), (re.compile(r"\bDTE\b"), "direct link to Earth"), (re.compile(r"\bRHU\b|radioisotope heater|radyoizotop ısıtıcı"), "nuclear heater"),
          (re.compile(r"\bveto\b|\bcap\b", re.I), "limit"), (re.compile(r"horizon mask|ufuk maskesi"), "skyline"), (re.compile(r"\bterminator\b", re.I), "day-night line")]


STAMP = re.compile(r"(?:,? (?:and )?(?:starts|starting|from|ve))? ?\d{4}-\d\d-\d\d \d\d:\d\d UTC(?:'?(?:de|da)?(?: başlıyor| başlar)?)?")
ASSUMPTION = re.compile(r"\s*\((?:assumption|varsayım)[^)]*\)")
HOURS = re.compile(r"\*{0,2}\b(\d{2,4}) ?(?:h|hours|saat)\b\*{0,2}")
INFO_ONE = {"sun_info", "comms_info", "slope_info", "night_thermal", "battery_info", "data_quality", "why_score"}  # one clear paragraph is enough


def plain_words(text: str, lang: str = "en") -> str:
    for rx, sub in JARGON:
        text = rx.sub(sub, text)
    text = ASSUMPTION.sub("", STAMP.sub("", text))
    day = "gün" if lang == "tr" else "days"
    text = re.sub(r"\s*\(\d+ separate dark periods?\)", "", text)
    out = HOURS.sub(lambda m: f"about {round(int(m[1]) / 24)} {day}".replace("about ", "yaklaşık " if lang == "tr" else "about ") if int(m[1]) >= 48 else m[0], text)
    return re.sub(r"\b(the|when the) (?:about|yaklaşık) (\d+) (?:days|gün) night", r"\1 \2-day night", out)


SCENARIO_INTENTS = SLOT_AWARE | {"best_date", "compare_dates", "best_vehicle", "best_site", "try_option", "confused", "doubt", "requirements"}
SECONDARY_OK = {"best_vehicle", "best_date", "best_site", "requirements"}  # questions that can share one message and both be answered
# reactions to the assistant itself: they follow what was just said, so they must not overwrite what the conversation is about
CUE = {"repeat": re.compile(r"tekrar|again|once more|bir daha|bi daha|repeat|ne dedin|ne demistin|last|son |yeniden|geri sar|goster|show"), "elaborate": re.compile(r"devam|continue|more|detay|ayrinti|daha|further|elaborate|expand|why|neden|then|sonra|else|go on|talk|anlat|ac")}
REACTION_INTENTS = {"affirm", "deny", "confused", "doubt", "complaint", "repeat", "elaborate"}
REACTIONS = {"affirm", "deny", "confused", "doubt", "complaint", "repeat", "elaborate", "thanks", "greet", "bye", "smalltalk", "clarify", "unknown", "apply"}
# 'and why?' / 'go on' continue the last topic with the natural next question
NEXT_STEP = {"explain_result": "why_score", "why_score": "improve", "improve": "best_date", "best_date": "table_request", "compare_results": "improve",
             "best_vehicle": "table_request", "best_site": "table_request", "compare_dates": "table_request", "try_option": "why_score", "sun_info": "improve", "comms_info": "improve",
             "slope_info": "improve", "night_thermal": "improve", "battery_info": "improve", "data_quality": "explain_result", "change_context": "explain_result",
             "requirements": "best_vehicle"}
# which of two scenarios on the page a message is about: "B", "second scenario", "scenario A", "first result"
B_REF = re.compile(r"(?:^| )(?:b|ikinci|second)(?: |$)|(?:scenario|senaryo|option|secenek|result|sonuc|operation|operasyon) (?:b|2)(?: |$)")
A_REF = re.compile(r"(?:^| )(?:birinci|first)(?: |$)|(?:scenario|senaryo|option|secenek|result|sonuc|operation|operasyon) (?:a|1)(?: |$)|(?:^| )a (?:senaryo|scenario|sonuc|result)")
HEATER = re.compile(r"\b(isitici|isi kaynagi|heater|rhu|radioisotope)")


def _fmt_pct(x):
    return f"{100 * x:.0f}%"


def _verdict(index, lang):
    bands = [(60, "iyi", "good"), (40, "orta", "moderate"), (20, "zayıf", "weak"), (0, "çok zayıf", "very weak")]
    for t, tr, en in bands:
        if index >= t:
            return tr if lang == "tr" else en
    return bands[-1][1 if lang == "tr" else 2]


def _date_tr(iso, lang):
    d = datetime.fromisoformat(iso)
    tr = ["Oca", "Şub", "Mar", "Nis", "May", "Haz", "Tem", "Ağu", "Eyl", "Eki", "Kas", "Ara"]
    en = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    return f"{d.day} {(tr if lang == 'tr' else en)[d.month - 1]} {d.year}"


def _range_text(a, b, lang):
    if a == b:
        return _date_tr(a, lang)
    return f"{_date_tr(a, lang)} – {_date_tr(b, lang)}"


def _place(lat, lon):
    lon = lon - 360 if lon > 180 else lon
    return f"{abs(lat):.2f}°{'S' if lat < 0 else 'N'}, {abs(lon):.2f}°{'W' if lon < 0 else 'E'}"


def _site_name(lat, lon):
    """The named site at these coordinates, or None."""
    for name, (la, lo) in SITES.items():
        if abs(la - lat) < 1e-3 and abs((lo - lon + 180) % 360 - 180) < 1e-3:
            return name.capitalize()
    return None


LIMIT_SHORT = {
    "slope": {"en": "ground slope", "tr": "eğim"}, "comms": {"en": "no Earth link", "tr": "Dünya bağlantısı yok"}, "power": {"en": "no sunlight", "tr": "güneş yok"},
    "thermal": {"en": "long cold night", "tr": "uzun soğuk gece"}, "energy": {"en": "flat battery", "tr": "pil bitiyor"},
}
LIMIT_TEXT = {
    "slope": {"en": "the ground is too steep for this vehicle", "tr": "zemin bu araç için fazla dik"},
    "comms": {"en": "Earth is never in line of sight", "tr": "Dünya hiç görüş hattında değil"},
    "power": {"en": "there is no sunlight to recharge", "tr": "şarj için hiç güneş yok"},
    "thermal": {"en": "a long night may freeze a vehicle with no known heat source", "tr": "ısı kaynağı olmayan (ya da bilinmeyen) bir aracı uzun bir gece dondurabilir"},
    "energy": {"en": "the battery would run flat early", "tr": "pil erken bitecek"},
}


WHY_TOPIC = {"sun_angle", "sun_light", "earth", "slope", "night"}


class Assistant:
    def __init__(self, dem, profiles: dict, nlu):
        self.dem, self.profiles, self.nlu = dem, profiles, nlu
        self.rng = random.Random(7)

    # ------------------------------------------------------------------ entry point
    def respond(self, message: str, context: dict | None = None, memory: dict | None = None, today: date | None = None) -> dict:
        memory = dict(memory or {})
        ctx = context or {}
        self.today = today or datetime.utcnow().date()
        self._notes = []  # what the message itself changed for this one answer (date, vehicle, place)
        ent = extract(message, today)
        lang = detect_language(message, memory.get("lang") or "en")
        norm = normalize(message)
        scs = ctx.get("scenarios") or []
        self.simple = ctx.get("level") == "simple"  # plain words for everyone, instead of the full technical answer
        self._assumed = []
        self._focus = self._which(scs, norm, memory)
        if ctx.get("free"):
            self._remember(ent, memory)  # what the user said (date, vehicle, place) is kept even when this message is not understood
        chip = route_chip(message)  # a ready-made question is understood by its text, not by the model's mood
        self._secondary = None
        if chip:
            intent, conf, top = chip, 1.0, [(chip, 1.0)]
        else:
            intent, conf, top = self.route(message, ent)
        has_slot = bool(ent.vehicles or ent.sites or ent.coords or ent.dates)
        if intent == "change_context" and has_slot and not chip and not COMMAND.search(norm):
            intent = "try_option"  # 'peki ya IM-1 ile' / 'and luna 25' is a question, not an instruction to change the page
        if ctx.get("free") and not scs and (intent in SCENARIO_INTENTS or intent == "compare_dates"):
            ctx = {**ctx, "scenarios": [self._free_scenario(ent, memory, lang)]}  # no page: the scenario is what the conversation has said so far
        if intent == "clarify":
            reply = self._clarify(top, lang)
        else:
            reply = getattr(self, f"_h_{intent}")(message, norm, ent, ctx, memory, lang)
        second = self._secondary
        if second and second != intent and intent in SECONDARY_OK:  # '... which vehicle, and what features should it have?': both get an answer
            if ctx.get("free") and not scs:  # free chat: the second answer continues where the first one ended (its winning place)
                ctx = {**ctx, "scenarios": [self._free_scenario(ent, memory, lang)]}
            more = getattr(self, f"_h_{second}")(message, norm, ent, ctx, memory, lang)
            reply["blocks"] = [b for b in reply["blocks"] if not (isinstance(b, dict) and b.get("type") == "chips")] + \
                [_t(lang, "**Sorunun ikinci kısmı:**", "**The second part of your question:**")] + more["blocks"]
        intent_out = intent
        reply["blocks"] = [{"type": "text", "text": b} if isinstance(b, str) else b for b in reply["blocks"]]
        if self.simple and intent not in NO_SIMPLIFY:
            reply["blocks"] = self._simplify(reply["blocks"], intent, lang)
        if self._notes and intent in SLOT_AWARE:
            reply["blocks"].insert(0, {"type": "text", "text": _t(lang, "Hesapladığım senaryo: ", "Scenario I used: ") + " · ".join(self._notes)})
        if self._assumed and intent in SCENARIO_INTENTS:
            reply["blocks"].insert(0, {"type": "text", "text": _t(lang, "Söylemediğin şeyler için şunu varsaydım: ", "For what you did not say I assumed: ") + " · ".join(self._assumed) + _t(lang, ". İstediğin an değiştirebilirsin.", ". Tell me if you want another.")})
        if len(scs) > 1 and intent in SCENARIO_INTENTS:
            sc0 = scs[self._focus]
            label = "AB"[self._focus]
            name = (self.profiles.get(sc0["vehicleId"]) or {}).get("name", sc0["vehicleId"])
            reply["blocks"].insert(0, {"type": "text", "text": _t(lang, f"**Senaryo {label}** hakkında konuşuyorum: ", f"I am talking about **scenario {label}**: ") + f"{name} · {_date_tr(sc0['date'], lang)} · {_place(float(sc0['lat']), float(sc0['lon']))}"})
        reply.setdefault("memory", {})
        memory.update(reply.pop("memory"))
        if len(scs) > 1:
            memory["focus"] = self._focus
            self._tag_actions(reply, memory, "AB"[self._focus].lower())
        memory["lang"] = lang
        memory["last_intent"] = intent_out
        if intent_out not in REACTIONS:
            memory["topic_intent"] = intent_out
            memory["last_blocks"] = [b for b in reply["blocks"] if not (isinstance(b, dict) and b.get("type") in ("chips", "actions"))][:8]
        return {"reply": reply, "intent": intent_out, "confidence": conf, "top": [(n, round(p, 3)) for n, p in top], "memory": memory, "language": lang}

    def _remember(self, ent, mem):
        """Free chat: the date, vehicle and place a message names become what the conversation is about."""
        f = dict(mem.get("free") or {})
        if ent.dates:
            first = ent.dates[0]
            f["date"] = first if isinstance(first, str) else (self.today + timedelta(days=first["shift_days"])).isoformat()
        for vid in ent.vehicles[:1]:
            if vid in self.profiles:
                f["vehicleId"] = vid
        if ent.sites:
            f["lat"], f["lon"], f["site"], f["site_given"] = *SITES[ent.sites[0]], ent.sites[0].capitalize(), True
        elif ent.coords:
            f["lat"], f["lon"] = ent.coords[0]
            f["site"], f["site_given"] = _place(*ent.coords[0]), True
        elif ent.region and not f.get("site_given"):
            f["region"] = ent.region
        mem["free"] = f
        return f

    def _free_scenario(self, ent, mem, lang):
        """Free chat: date, vehicle and place come from the message, then from what was said earlier; anything never said gets a stated default."""
        f = self._remember(ent, mem)
        if "date" not in f:
            f["date"] = self.today.isoformat()
            self._assumed.append(_t(lang, f"tarih {_date_tr(f['date'], lang)} (bugün)", f"date {_date_tr(f['date'], lang)} (today)"))
        if "lat" not in f:
            f["lat"], f["lon"], f["site"] = -85.987, 2.08, "Malapert"
            self._assumed.append(_t(lang, "konum Malapert (güney kutbunda çok çalışılan aday bölge)" if f.get("region") else "konum Malapert",
                                          "place Malapert (a much-studied south-pole candidate)" if f.get("region") else "place Malapert"))
        if "vehicleId" not in f:
            f["vehicleId"] = DEFAULT_VEHICLE if DEFAULT_VEHICLE in self.profiles else next(iter(self.profiles))
            self._assumed.append(_t(lang, f"araç {self.profiles[f['vehicleId']]['name']}", f"vehicle {self.profiles[f['vehicleId']]['name']}"))
        return {"date": f["date"], "vehicleId": f["vehicleId"], "lat": float(f["lat"]), "lon": float(f["lon"])}

    def _simplify(self, blocks, intent, lang):
        """The simple level: the same facts, without the technical clutter. Tables shrink to the top three rows (or go), the assistant's
        own observations and the long bullet lists are trimmed, vendor notes in brackets are dropped and the jargon is said in plain words."""
        out, skip_bullets, texts = [], False, 0
        for b in blocks:
            t = b.get("type")
            if t == "text":
                if NOTICED.search(b["text"]) or re.match(r"\s*(Why|Neden):", b["text"]):
                    skip_bullets = NOTICED.search(b["text"]) is not None
                    continue
                skip_bullets = False
                texts += 1
                if intent in INFO_ONE and texts > (2 if intent == "why_score" else 1):
                    continue
                out.append({**b, "text": plain_words(b["text"], lang)})
            elif t == "bullets":
                if skip_bullets:
                    skip_bullets = False
                    continue
                out.append({**b, "items": [plain_words(i, lang) for i in b["items"][:3]]})
            elif t == "table":
                if intent in KEEP_TABLE:
                    out.append({**b, "rows": b["rows"][:3]})
            else:
                out.append(b)
        return out

    def _which(self, scs, norm, mem):
        """0 for scenario A, 1 for B: what the message names, else what the last answer was about."""
        if len(scs) < 2:
            return 0
        b, a = bool(B_REF.search(norm)), bool(A_REF.search(norm))
        if b and not a:
            return 1
        if a and not b:
            return 0
        return 1 if mem.get("focus") == 1 else 0

    @staticmethod
    def _tag_actions(reply, memory, which):
        """With two scenarios on the page, an action must say which one it changes."""
        for blk in reply["blocks"]:
            if isinstance(blk, dict) and blk.get("type") == "actions":
                for btn in blk["buttons"]:
                    btn["action"].setdefault("scenario", which)
        for act in reply.get("actions") or []:
            act.setdefault("scenario", which)
        sug = memory.get("suggestion")
        if isinstance(sug, dict):
            sug.setdefault("scenario", which)

    def route(self, message: str, ent=None, rules: bool = True):
        """The decision the user experiences: a short human reaction recognised by its pattern, else the model's intent, or
        'clarify' when it is not sure enough to act on.

        'Not sure' = the averaged confidence is low, or the independently trained members disagree about the intent.
        `rules=False` leaves out the reaction patterns, to measure the model on its own."""
        self._secondary = None
        if rules:
            r = detect_reaction(message)
            if r:
                return r, 0.9, [(r, 0.9)]
        intent, conf, top = self._model_route(message, rules)
        if not rules:
            return intent, conf, top
        ent = ent or extract(message, getattr(self, "today", None))
        # a long message can hide its question behind the plan ('on 16 Nov I want to go to the pole, which vehicle...'):
        # each clause is also read on its own, and the clearest task-question wins
        # ('A or B?' is one comparison, never two questions)
        parts = clauses(message) if len(normalize(message).split()) >= 6 and "compare" not in concepts(message) else []
        if intent in ("clarify", "unknown"):
            for p in parts:
                i2, c2, t2 = self._model_route(p, rules)
                if i2 in SCENARIO_INTENTS | {"glossary", "data_quality", "about"} and c2 > conf and i2 != "clarify":
                    intent, conf, top = i2, c2, t2
        # the concepts in the message (nlu_frames): agreement makes the model sure; when the model is unsure they decide
        fr = frame(message, ent)
        if fr:
            f, strength = fr
            if f == intent:
                conf = max(conf, 0.85)
            elif (intent in ("clarify", "unknown", *REACTION_INTENTS) and conf < 0.95 or conf < 0.85
                  or strength >= 1.0 and intent in SCENARIO_INTENTS and f in SECONDARY_OK):  # an explicit 'which vehicle' / 'when' / 'where most' settles sibling task questions
                intent, conf = f, max(conf, 0.75 * strength)
                top = [(f, conf)] + [t for t in top if t[0] != f][:2]
        # two questions in the same message ('... which vehicle, or what features should it have?'): the first asked is the
        # main one, the other is answered after it
        asked = []
        for p in parts:
            f2 = frame(p, ent)
            if f2 and f2[0] in SECONDARY_OK and f2[0] not in asked:
                asked.append(f2[0])
        if len(asked) >= 2 and intent in SECONDARY_OK:
            intent, conf = asked[0], max(conf, 0.85)
            top = [(intent, conf)] + [t for t in top if t[0] != intent][:2]
            self._secondary = asked[1]
        elif asked and asked[0] != intent and intent in SECONDARY_OK:
            self._secondary = asked[0]
        return intent, conf, top

    def _model_route(self, message, rules=True):
        """The intent network alone (with its own 'not sure' and cue-word guards)."""
        intent, conf, top = self.nlu.predict(message)
        agree = self.nlu.agreement(message, intent) if hasattr(self.nlu, "agreement") else 1.0
        short_reaction = intent in REACTION_INTENTS and len(normalize(message).split()) <= 6 and conf >= 0.3 and agree >= 0.4  # short replies are never confident
        if intent != "unknown" and (conf < CONF_MIN or agree < AGREE_MIN) and not short_reaction:
            intent = "clarify"
        if rules and intent in ("repeat", "elaborate") and not CUE[intent].search(normalize(message)):
            intent = "unknown"  # 'write me a fairy tale' sounds like 'write that again' to a small model; without a cue word it is not a reaction
        return intent, conf, top

    # ------------------------------------------------------------------ helpers
    def _scenario(self, ctx, ent=None, mem=None, norm="", lang="en"):
        """The page's scenario, with whatever the message itself names put on top: '20 aralikta', 'o tarihte', 'apollo 15 ile', 'shackleton'."""
        scs = ctx.get("scenarios") or [None]
        sc = scs[min(getattr(self, "_focus", 0), len(scs) - 1)]
        if not sc:
            return None
        base = Scenario(sc["date"], sc["vehicleId"], float(sc["lat"]), float(sc["lon"]))
        if ent is None:
            return base
        date_, vehicle, lat, lon = base.date, base.vehicle_id, base.lat, base.lon
        notes = []
        if ent.dates:
            first = ent.dates[0]
            date_ = first if isinstance(first, str) else ((self.today if first.get("from_today") else datetime.fromisoformat(base.date).date()) + timedelta(days=first["shift_days"])).isoformat()
        elif REF_DATE.search(norm) and (mem or {}).get("suggestion", {}) and (mem["suggestion"] or {}).get("date"):
            date_ = mem["suggestion"]["date"]
        if date_ != base.date:
            notes.append(_date_tr(date_, lang))
        for vid in ent.vehicles[:1]:
            if vid in self.profiles and vid != vehicle:
                vehicle = vid
                notes.append(self.profiles[vid]["name"])
        if ent.sites:
            lat, lon = SITES[ent.sites[0]]
            notes.append(ent.sites[0].capitalize())
        elif ent.coords:
            lat, lon = ent.coords[0]
            notes.append(_place(lat, lon))
        self._notes = notes
        return Scenario(date_, vehicle, lat, lon)

    def _earliest(self, sc):
        """A plan for the future is not offered start dates in the past; an analysis of a past date scans both ways."""
        today = self.today.isoformat()
        return today if sc.date >= today else None

    def _profile(self, sc):
        return self.profiles.get(sc.vehicle_id)

    def _need_scenario(self, lang):
        return {"blocks": [_t(lang, "Önce bir analiz çalıştır (tarih, araç ve konum seç); sonra o sonuç hakkında konuşabiliriz.",
                                 "Run an analysis first (pick a date, a vehicle and a location) and then we can talk about that result.")]}

    def _clarify(self, top, lang):
        names = {"best_date": ("en iyi tarih", "best date"), "why_score": ("puan neden böyle", "why the score is what it is"), "explain_result": ("sonucu açıklama", "explaining the result"),
                 "improve": ("iyileştirme önerileri", "ways to improve"), "best_vehicle": ("en iyi araç", "best vehicle"), "best_site": ("daha iyi konum", "a better location"),
                 "compare_dates": ("iki tarihi karşılaştırma", "comparing dates"), "sun_info": ("güneş durumu", "sunlight"), "comms_info": ("Dünya ile iletişim", "communications with Earth"),
                 "slope_info": ("eğim", "slope"), "night_thermal": ("gece ve soğuk", "night and cold"), "battery_info": ("pil", "battery"), "data_quality": ("veri güvenilirliği", "data reliability"),
                 "glossary": ("bir terimin anlamı", "what a term means"), "change_context": ("tarih/araç/konum değiştirme", "changing the date, vehicle or place"),
                 "about": ("neler yapabildiğim", "what I can do"), "compare_results": ("iki sonucu karşılaştırma", "comparing the two results"), "table_request": ("tablo", "a table"), "apply": ("öneriyi uygulama", "applying a suggestion"),
                 "requirements": ("aracın sahip olması gerekenler", "what a vehicle would need there")}
        opts = [names[n][0 if lang == "tr" else 1] for n, _ in top if n in names][:2]
        # the chips are the readings it hesitated between, as ready questions, so one click answers what was meant
        ask = {"best_date": ("Tarihi nasıl oynatabiliriz?", "How can we shift the date?"), "why_score": ("Puan neden düşük?", "Why is the score low?"),
               "explain_result": ("Sonucu özetle", "Summarize the result"), "best_vehicle": ("Hangi araç burada daha iyi?", "Which vehicle is better here?"),
               "best_site": ("Daha iyi bir konum var mı?", "Is there a better location?"), "data_quality": ("Veriler ne kadar güvenilir?", "How reliable is this data?"),
               "about": ("Neler yapabilirsin?", "What can you do?"), "compare_results": ("İki sonucu karşılaştır", "Compare the two results"), "table_request": ("Bunu tabloda göster", "Show it as a table")}
        chips = [ask[n][0 if lang == "tr" else 1] for n, _ in top if n in ask][:2]
        chips += [c for c in {"tr": ["Puan neden düşük?", "Tarihi nasıl oynatabiliriz?", "Sonucu özetle"], "en": ["Why is the score low?", "How can we shift the date?", "Summarize the result"]}[lang] if c not in chips][:3 - len(chips)]
        txt = _t(lang, "Tam anlayamadım. Şunlardan birini mi kastettin: " + " / ".join(opts) + "? İstersen şöyle sorabilirsin:",
                       "I did not quite get that. Did you mean one of: " + " / ".join(opts) + "? You can also try:")
        return {"blocks": [txt, {"type": "chips", "items": chips}]}

    def _follow(self, lang, kind):
        sets = {"result": {"tr": ["Puan neden düşük?", "Tarihi nasıl oynatabiliriz?", "Hangi araç burada daha iyi?"], "en": ["Why is the score low?", "How can we shift the date?", "Which vehicle is better here?"]},
                "date": {"tr": ["Bunu tabloda göster", "Bu tarihi uygula", "Daha iyi bir konum var mı?"], "en": ["Show it as a table", "Apply that date", "Is there a better location?"]},
                "more": {"tr": ["Sonucu özetle", "Tarihi nasıl oynatabiliriz?", "Veriler ne kadar güvenilir?"], "en": ["Summarize the result", "How can we shift the date?", "How reliable is this data?"]}}
        return {"type": "chips", "items": sets[kind][lang]}

    # ------------------------------------------------------------------ small talk
    def _h_greet(self, msg, norm, ent, ctx, mem, lang):
        if ctx.get("free"):
            return {"blocks": [_t(lang, "Merhaba! Doğrudan sorabilirsin: örneğin \"12 Eylül'de hangi aracı göndermeliyiz?\" ya da \"Shackleton Mart'ta nasıl?\". Tarihi, aracı ve konumu sen söylersin, rakamları gerçek NASA verisiyle hesaplarım.",
                                    "Hello! Just ask: for example \"Which vehicle should we send on September 12?\" or \"How does Shackleton look in March?\". You tell me the date, vehicle and place, and I work the numbers out from real NASA data."),
                               {"type": "chips", "items": ["Which vehicle should we send on September 12?", "How does Shackleton look on March 3?", "What can you do?"]}]}
        sc = self._scenario(ctx)
        base = _t(lang, "Merhaba! Ben bu analize özel yapılmış asistanım. Seçtiğin tarih, araç ve konum hakkında soru sorabilirsin.",
                        "Hello! I am the assistant built for this analysis. Ask me about the date, vehicle and place you picked.")
        return {"blocks": [base, self._follow(lang, "result" if sc else "more")]}

    def _h_thanks(self, msg, norm, ent, ctx, mem, lang):
        return {"blocks": [self.rng.choice([_t(lang, "Rica ederim! Başka bir şey merak edersen sor.", "You are welcome! Ask if there is anything else."),
                                            _t(lang, "Ne demek, buradayım.", "Anytime, I am here.")])]}

    def _h_bye(self, msg, norm, ent, ctx, mem, lang):
        return {"blocks": [_t(lang, "Görüşürüz! Analizin burada duruyor, istediğin zaman dönebilirsin.", "See you! Your analysis stays right here whenever you come back.")]}

    def _h_smalltalk(self, msg, norm, ent, ctx, mem, lang):
        tr = ["İyiyim, teşekkürler! Ay'ın güney kutbunda gün ışığı hesaplamaktan sıkılmıyorum. Sen nasılsın?", "Gayet iyi. Kutuptaki gölgeler bana hep ilginç geliyor. Analizinle ilgili ne merak ediyorsun?",
              "Yorulmam, sadece hesap yaparım. İstersen sonucunu birlikte bir inceleyelim."]
        en = ["I am doing well, thanks! Computing sunlight at the lunar south pole never gets old. How about you?", "Pretty good. The shadows at the pole always fascinate me. What would you like to know about your analysis?",
              "I do not get tired, I just compute. Want to go through your result together?"]
        return {"blocks": [self.rng.choice(tr if lang == "tr" else en), self._follow(lang, "more")]}

    def _h_about(self, msg, norm, ent, ctx, mem, lang):
        tr = ("Ben bu projeye özel, sıfırdan yazılmış bir asistanım: soruyu anlayan küçük bir sinir ağım var (NumPy ile eğitildi, hazır bir model kullanmıyorum). Rakamları uydurmam; her sayıyı gerçek NASA verisiyle çalışan hesap motorundan alırım.\n\n"
              "Şunları yapabilirim:")
        en = ("I am an assistant written from scratch for this project: a small neural network of my own understands your question (trained in NumPy, no pre-made model). I never make numbers up; every figure comes from the calculation engine that runs on real NASA data.\n\n"
              "Here is what I can do:")
        items_tr = ["Sonucu sade bir dille açıklamak ve puanın neden öyle olduğunu söylemek", "Tarihi oynatınca ne olacağını görmek ve en iyi başlangıç tarihini bulmak", "Araçları ve yakın konumları karşılaştırmak",
                    "Güneş, Dünya ile iletişim, eğim, gece ve pil hakkında sorularını yanıtlamak", "Terimleri açıklamak, tablo hazırlamak, tarih/araç/konumu senin yerine değiştirmek"]
        items_en = ["Explain the result in plain words and say why the score is what it is", "Try other dates and find the best start date", "Compare vehicles and nearby places",
                    "Answer questions about sunlight, Earth contact, slope, the night and the battery", "Explain terms, build tables, and change the date, vehicle or place for you"]
        return {"blocks": [tr if lang == "tr" else en, {"type": "bullets", "items": items_tr if lang == "tr" else items_en}, self._follow(lang, "result")]}

    def _h_unknown(self, msg, norm, ent, ctx, mem, lang):
        return {"blocks": [_t(lang, "Bu konuda yardımcı olamam; ben yalnızca Ay'daki iniş analizi hakkında konuşabilirim (tarih, araç, konum, güneş, iletişim, eğim, gece, pil). Şunları deneyebilirsin:",
                                    "I cannot help with that; I can only talk about the lunar landing analysis (date, vehicle, place, sunlight, communications, slope, night, battery). You could try:"),
                           self._follow(lang, "result")]}

    # ------------------------------------------------------------------ explanations
    def _h_explain_result(self, msg, norm, ent, ctx, mem, lang):
        sc = self._scenario(ctx, ent, mem, norm, lang)
        if not sc:
            return self._need_scenario(lang)
        if self.simple:
            return self._plain_result(sc, lang, closing=False)
        p = self._profile(sc)
        m, a = tools.assess(self.dem, p, sc)
        idx = a["vehicle_index"]
        lines = [_t(lang, f"{p['name']}, {_place(sc.lat, sc.lon)} noktasında {_date_tr(sc.date, lang)} tarihinde başlayan 30 gün için **{idx:.0f}/100** aldı: sonuç **{_verdict(idx, lang)}**. (Bu bölgede puanlar nadiren 60'ı geçer.)",
                         f"{p['name']} at {_place(sc.lat, sc.lon)}, for the 30 days starting {_date_tr(sc.date, lang)}, scores **{idx:.0f}/100**: **{_verdict(idx, lang)}**. (Scores here rarely go above 60.)")]
        if a["limiting_factor"]:
            lines.append(_t(lang, f"Puanı asıl aşağı çeken şey: {LIMIT_TEXT[a['limiting_factor']]['tr']}. Bu olmasa puan {a['weighted_index']:.0f} olurdu.",
                                  f"What holds the score down: {LIMIT_TEXT[a['limiting_factor']]['en']}. Without it the score would be {a['weighted_index']:.0f}."))
        rows = [[_t(lang, "Güneşli zaman", "Time in sunlight"), _fmt_pct(m["illumination_fraction"])],
                [_t(lang, "Dünya görünür", "Earth in view"), _fmt_pct(m["dte_fraction"])],
                [_t(lang, "En uzun gece", "Longest night"), f"{m['longest_darkness_hours']:.0f} h"],
                [_t(lang, "En uzun Dünya kesintisi", "Longest Earth outage"), f"{m['longest_outage_hours']:.0f} h"],
                [_t(lang, "Ortalama eğim", "Average slope"), f"{m['mean_slope_deg']:.1f}°"]]
        table = {"type": "table", "title": _t(lang, "Temel sayılar", "Key numbers"), "columns": [_t(lang, "Ölçü", "Measure"), _t(lang, "Değer", "Value")], "rows": rows}
        q = a["data_quality"]
        lines.append(_t(lang, f"Güvenilirlik: araç girdilerinin {q['inputs_verified']} tanesi doğrulanmış, {q['inputs_unverified']} tanesi proje verisinden, {q['inputs_unknown']} tanesi bilinmiyor.",
                              f"Reliability: {q['inputs_verified']} vehicle inputs are verified, {q['inputs_unverified']} come from the project dataset, {q['inputs_unknown']} are unknown."))
        head = (lines[:2] if a["limiting_factor"] else lines[:1]) + [self._odds_line(a, lang)]
        return self._with_insights({"blocks": head + [table, lines[-1], self._follow(lang, "result")], "memory": {"table": table}}, sc, p, m, a, lang)

    VERDICT_TR = {"go": "önerilir", "conditional": "mümkün ama riskli", "weak": "bu haliyle önerilmez", "stop": "yapılmamalı"}
    MODE_TR = {"landing": ("dik bir yere inmesi", "touching down on ground too steep"), "cold": ("gücü kesilen gecede donması", "freezing in a night without power"),
               "power": ("pilin yeterli çalışma süresi verememesi", "too little power to keep working"), "comms": ("Dünya'ya yeterince veri gönderememesi", "too little contact with Earth")}

    def _odds_line(self, a, lang):
        """The call and the simulated success probability, exactly as the result page shows them."""
        r, v = a.get("risk"), a.get("verdict")
        if not r or not v:
            return ""
        p, od = r["p_success"], r["operating_days"]
        top = next((f for f in r["failure_modes"] if f["share"] >= 0.05), None)
        why = (_t(lang, f" En sık başarısızlık sebebi: {self.MODE_TR[top['mode']][0]} (%{100 * top['share']:.0f}).", f" The most common way it fails: {self.MODE_TR[top['mode']][1]} ({100 * top['share']:.0f}%).") if top else "")
        conf = {"high": ("yüksek", "high"), "medium": ("orta", "medium"), "low": ("düşük: bazı araç verileri yayımlanmamış", "low: some vehicle data is not published")}[r["confidence"]["level"]]
        return _t(lang, f"Karar: **{self.VERDICT_TR[v['kind']]}**. {r['samples']} benzetimli görevde başarı oranı **%{100 * p:.0f}** (güvenle iniyor, en az {r['goal_days']:.0f} gün çalışıyor ve verisini Dünya'ya ulaştırıyor); araç tipik olarak {od['p50']:.0f} gün çalışıyor (10 denemenin 8'inde {od['p10']:.0f}–{od['p90']:.0f} gün).{why} Güven: {conf[0]}.",
                        f"Call: **{v['title']}**. In **{100 * p:.0f}%** of {r['samples']} simulated missions it lands safely, works at least {r['goal_days']:.0f} days and gets its data home; it typically works {od['p50']:.0f} days ({od['p10']:.0f}–{od['p90']:.0f} in 8 of 10 runs).{why} Confidence: {conf[1]}.")

    def _insights(self, sc, p, m, a, lang):
        """Things worth saying that nobody asked: each one is computed by the engine, cheap ones only (< 0.3 s).
        Returns (bullets, buttons)."""
        out, buttons = [], []
        cur = a["vehicle_index"]
        # 1) a better start date nearby
        rows = tools.date_scan(self.dem, p, sc, 60, earliest=self._earliest(sc))
        _, _, bd = self._best_pick(rows, sc)
        if bd["index"] - cur >= 2:
            days = (datetime.fromisoformat(bd["date"]) - datetime.fromisoformat(sc.date)).days
            out.append(_t(lang, f"Başlangıcı {abs(days)} gün {'ileri' if days > 0 else 'geri'} alıp **{_date_tr(bd['date'], lang)}** yaparsan puan {cur:.0f} → {bd['index']:.0f} oluyor.",
                                f"Moving the start {abs(days)} days {'later' if days > 0 else 'earlier'} to **{_date_tr(bd['date'], lang)}** takes the score {cur:.0f} → {bd['index']:.0f}."))
            buttons.append({"label": _t(lang, f"{_date_tr(bd['date'], lang)} tarihini uygula", f"Apply {_date_tr(bd['date'], lang)}"), "action": {"type": "apply_scenario", "date": bd["date"]}})
        # 2) the thermal ceiling: what a heater would be worth
        if a["limiting_factor"] == "thermal":
            warm = {**p, "radioisotope_heat": {"value": True, "status": "unknown"}}
            _, a2 = tools.assess(self.dem, warm, sc)
            if a2["vehicle_index"] - cur >= 1:
                out.append(_t(lang, f"Puanı kısan şey gece. {'Bu aracın kendi ısı kaynağı yok' if p['radioisotope_heat']['value'] is False else 'Bu aracın ısıtıcı bilgisi bilinmiyor'}; ısıtıcısı olsaydı puan {cur:.0f} → {a2['vehicle_index']:.0f} olurdu.",
                                    f"The night is what caps the score. This vehicle's heater is {'absent' if p['radioisotope_heat']['value'] is False else 'unknown'}; with one the score would be {cur:.0f} → {a2['vehicle_index']:.0f}."))
        # 3) long radio silence
        if a["checks"]["comms"].get("mode") == "direct_to_earth" and m["longest_outage_hours"] >= 168:
            when = tools.hours_to_iso(sc.date, m["outage_start_h"])
            out.append(_t(lang, f"Dünya ile en uzun sessizlik {m['longest_outage_hours']:.0f} saat ({m['longest_outage_hours'] / 24:.1f} gün)" + (f", {when} UTC'de başlıyor" if when else "") + ". Bu sürede araç kendi başına kalır; görev planında bunu hesaba kat.",
                                f"The longest silence with Earth is {m['longest_outage_hours']:.0f} h ({m['longest_outage_hours'] / 24:.1f} days)" + (f", starting {when} UTC" if when else "") + ". The vehicle is on its own then; plan for it."))
        # 4) ground
        if m["safe_area_fraction"] < 0.9:
            out.append(_t(lang, f"İniş alanının yalnızca {_fmt_pct(m['safe_area_fraction'])} kadarı düz sayılır; en dik yer {m['max_slope_deg']:.0f}°. İniş noktası hassas seçilmeli.",
                                f"Only {_fmt_pct(m['safe_area_fraction'])} of the landing area counts as flat; the steepest spot is {m['max_slope_deg']:.0f}°. The touchdown point needs care."))
        # 5) what the score cannot see
        unknown = [k for k in ("battery_wh", "radioisotope_heat", "max_slope_deg", "survival_power_w") if (p.get(k) or {}).get("value") is None]
        if len(unknown) >= 2:
            out.append(_t(lang, f"Bu aracın {len(unknown)} önemli değeri bilinmiyor (pil, ısıtıcı, eğim sınırı, güç tüketimi gibi). Puan bu yüzden aracın kendisinden çok konumu yansıtıyor; bu değerler girilirse daha keskin bir değerlendirme yaparım.",
                                f"{len(unknown)} key values of this vehicle are unknown (battery, heater, slope limit, power use). So the score reflects the place more than the vehicle; with those values I can judge it more sharply."))
        return out, buttons

    def _with_insights(self, reply, sc, p, m, a, lang):
        """Adds 'what I noticed' under an answer, and its buttons; remembers the first as the suggestion."""
        notes, buttons = self._insights(sc, p, m, a, lang)
        if not notes:
            return reply
        blocks = reply["blocks"]
        at = len(blocks) - 1 if blocks and isinstance(blocks[-1], dict) and blocks[-1].get("type") == "chips" else len(blocks)
        extra = [_t(lang, "**Kendiliğimden fark ettiklerim:**", "**What I noticed on my own:**"), {"type": "bullets", "items": notes}]
        if buttons:
            extra.append({"type": "actions", "buttons": buttons})
            reply.setdefault("memory", {})["suggestion"] = buttons[0]["action"]
        reply["blocks"] = blocks[:at] + extra + blocks[at:]
        return reply

    def _h_why_score(self, msg, norm, ent, ctx, mem, lang):
        sc = self._scenario(ctx, ent, mem, norm, lang)
        if not sc:
            return self._need_scenario(lang)
        p = self._profile(sc)
        m, a = tools.assess(self.dem, p, sc)
        topic, tconf = self._topic(msg, norm)
        if topic in WHY_TOPIC and tconf >= 0.5:
            return self._why_topic(topic, sc, m, a, lang)
        w = a["index_weights"]
        limit = (a["checks"]["slope"].get("limit_deg") or a["checks"]["slope"].get("generic_limit_deg") or 15.0)
        win = 30 * 24
        vals = {"sun": (m["illumination_fraction"], _t(lang, "Güneşli zaman", "Time in sunlight")), "dte": (m["dte_fraction"], _t(lang, "Dünya görünür", "Earth in view")),
                "dark": (1 - m["longest_darkness_hours"] / win, _t(lang, "Uzun gece yok", "No long night")), "slope": (max(0, min(1, 1 - m["mean_slope_deg"] / limit)), _t(lang, "Düz zemin", "Flat ground"))}
        rows, losses = [], []
        for k, (v, label) in vals.items():
            if w.get(k, 0) > 0:
                lost = 100 * w[k] * (1 - v)
                losses.append((lost, label))
                rows.append([label, _fmt_pct(v), f"{100 * w[k]:.0f}%", f"{lost:.0f}"])
        table = {"type": "table", "title": _t(lang, "Puan nereden kayboluyor", "Where the points are lost"),
                 "columns": [_t(lang, "Etken", "Factor"), _t(lang, "Durum", "How good"), _t(lang, "Ağırlık", "Weight"), _t(lang, "Kaybedilen puan", "Points lost")], "rows": rows}
        losses.sort(reverse=True)
        blocks = []
        if a["limiting_factor"]:
            blocks.append(_t(lang, f"Puan **{a['vehicle_index']:.0f}** çünkü {LIMIT_TEXT[a['limiting_factor']]['tr']}: bu, diğer etkenler ne olursa olsun puanı bir tavana bağlıyor. Tavan olmasaydı puan {a['weighted_index']:.0f} olurdu.",
                              f"The score is **{a['vehicle_index']:.0f}** because {LIMIT_TEXT[a['limiting_factor']]['en']}: that puts a ceiling on the score no matter how good the rest is. Without the ceiling it would be {a['weighted_index']:.0f}."))
        else:
            blocks.append(_t(lang, f"Puan **{a['vehicle_index']:.0f}**. Hiçbir sert sınır devrede değil; puan şu etkenlerin birleşimi:",
                                   f"The score is **{a['vehicle_index']:.0f}**. No hard limit is binding; the score is the mix of these factors:"))
        blocks.append(table)
        if losses:
            blocks.append(_t(lang, f"En çok puan kaybettiren: **{losses[0][1]}** ({losses[0][0]:.0f} puan).", f"The biggest loss: **{losses[0][1]}** ({losses[0][0]:.0f} points)."))
        blocks.append(self._follow(lang, "date"))
        return self._with_insights({"blocks": blocks, "memory": {"table": table}}, sc, p, m, a, lang)

    def _why_topic(self, topic, sc, m, a, lang):
        """Why one quantity is what it is: the number, then the physical reason in plain words."""
        lat = abs(sc.lat)
        if topic == "sun_angle":
            txt = _t(lang, f"Bu noktada Güneş ışıklı olduğu saatlerde ortalama **{m['sun_el_mean_lit']:.1f}°**, en çok {m['sun_el_max']:.1f}° yükseliyor. Sebebi enlem: {lat:.1f}° güneydesin, kutba çok yakın. Güneş'in o enlemde ulaşabileceği en yüksek açı kabaca 90° − enlem + Ay'ın eksen eğikliği (1,5°) kadardır, yani yaklaşık {90 - lat + 1.5:.1f}°. Üstüne bir de yerel tepeler Güneş'i alçakken kesebilir.",
                              f"Here the Sun is on average **{m['sun_el_mean_lit']:.1f}°** high while it is up, at most {m['sun_el_max']:.1f}°. The reason is latitude: you are at {lat:.1f}°S, very near the pole. The highest the Sun can get at that latitude is roughly 90° − latitude + the Moon's axial tilt (1.5°), about {90 - lat + 1.5:.1f}°. Local hills can also cut the Sun off while it is low.")
        elif topic == "sun_light":
            txt = _t(lang, f"Güneş 30 günün **{_fmt_pct(m['illumination_fraction'])}** kadarında ufuk çizgisinin üstünde. Ay'da gün ve gece döngüsü yaklaşık 29,5 gündür; kutuplarda Güneş çok alçak olduğu için çevredeki tepeler gölge de düşürüyor. En uzun karanlık dönem {m['longest_darkness_hours']:.0f} saat.",
                              f"The Sun is above the skyline for **{_fmt_pct(m['illumination_fraction'])}** of the 30 days. The lunar day and night repeat about every 29.5 days, and because the Sun is so low at the poles, surrounding hills also cast shadows. The longest dark spell is {m['longest_darkness_hours']:.0f} h.")
        elif topic == "earth":
            txt = _t(lang, f"Dünya 30 günün **{_fmt_pct(m['dte_fraction'])}** kadarında görünüyor. Ay hep aynı yüzünü Dünya'ya döndürür; kutupta Dünya ufuk çizgisine çok yakın durur ve Ay'ın hafif sallanması (libration) onu bir görünüp bir kaybolduruyor, tepeler de araya girebiliyor. En uzun kesinti {m['longest_outage_hours']:.0f} saat.",
                              f"Earth is in view for **{_fmt_pct(m['dte_fraction'])}** of the 30 days. The Moon always shows the same face to Earth; at the pole Earth sits very close to the skyline, so the Moon's slight wobble (libration) makes it appear and vanish, and hills can get in the way. The longest outage is {m['longest_outage_hours']:.0f} h.")
        elif topic == "slope":
            txt = _t(lang, f"500 m'lik iniş alanında ortalama eğim **{m['mean_slope_deg']:.1f}°**, en dik yer {m['max_slope_deg']:.0f}°, yükseklik farkı {m['relief_m']:.0f} m. Kutup bölgesi eski kraterlerle dolu; kraterlerin kenarları ve duvarları dikleşiyor, düz zemin ise az.",
                              f"On the 500 m landing area the average slope is **{m['mean_slope_deg']:.1f}°**, the steepest spot {m['max_slope_deg']:.0f}° and the height difference {m['relief_m']:.0f} m. The polar region is full of old craters; their rims and walls are steep and flat ground is scarce.")
        else:  # night
            txt = _t(lang, f"En uzun gece **{m['longest_darkness_hours']:.0f} saat**. Kutuplarda Güneş ufka çok yakın dolaştığı için bir tepenin ya da kraterin arkasında kaldığında uzun süre karanlık kalır; ayrıca Ay'ın gün-gece döngüsü yaklaşık 29,5 gündür.",
                              f"The longest night is **{m['longest_darkness_hours']:.0f} hours**. Because the Sun hugs the horizon at the poles, once it drops behind a hill or crater rim it stays dark for a long time; and the lunar day-night cycle is about 29.5 days.")
        return {"blocks": [txt, self._follow(lang, "date")]}

    # ------------------------------------------------------------------ what-if scans
    def _plateau(self, rows):
        top = max(r["index"] for r in rows)
        near = [r for r in rows if r["index"] >= top - 0.5]
        runs, cur = [], [near[0]]
        for r in near[1:]:
            if (datetime.fromisoformat(r["date"]) - datetime.fromisoformat(cur[-1]["date"])).days <= 1:
                cur.append(r)
            else:
                runs.append(cur)
                cur = [r]
        runs.append(cur)
        return top, runs

    def _best_pick(self, rows, sc):
        """(top score, the plateau of equally good start dates nearest the chosen one, its middle date)."""
        top, runs = self._plateau(rows)
        best_run = min(runs, key=lambda r: abs((datetime.fromisoformat(r[0]["date"]) - datetime.fromisoformat(sc.date)).days))
        return top, best_run, best_run[len(best_run) // 2]

    def _h_compare_results(self, msg, norm, ent, ctx, mem, lang):
        """Two scenarios on the page (A and B): which is better, by how much, why, and what would lift the weaker one."""
        scs = ctx.get("scenarios") or []
        if len(scs) < 2:
            return {"blocks": [_t(lang, "Ekranda tek senaryo var. İki sonucu karşılaştırmak için Yeni İşlem'de “Comparison” ya da “Full comparison” seç; sonra ikisini birlikte değerlendiririm.",
                                        "There is only one scenario on screen. To compare two results, pick “Comparison” or “Full comparison” in New Operation and I will weigh them together."),
                               self._follow(lang, "result")]}
        S = [Scenario(s["date"], s["vehicleId"], float(s["lat"]), float(s["lon"])) for s in scs[:2]]
        P = [self._profile(s) for s in S]
        (ma, aa), (mb, ab) = [tools.assess(self.dem, p, s) for p, s in zip(P, S)]
        ia, ib = aa["vehicle_index"], ab["vehicle_index"]
        gap = ia - ib
        tie = abs(gap) < 2
        win = None if tie else (0 if gap > 0 else 1)
        weak = None if tie else 1 - win
        L = "AB"
        if tie:
            head = _t(lang, f"**A** ve **B** hemen hemen aynı puanı alıyor: **{ia:.0f}** ve **{ib:.0f}** (100 üzerinden).",
                            f"**A** and **B** score almost the same: **{ia:.0f}** and **{ib:.0f}** out of 100.")
        else:
            hi, lo = (ia, ib) if win == 0 else (ib, ia)
            head = _t(lang, f"**{L[win]}** daha iyi: **{hi:.0f}** puan, **{L[weak]}** ise **{lo:.0f}** puan aldı; fark **{abs(gap):.0f}** puan ({_verdict(hi, 'tr')} ve {_verdict(lo, 'tr')}).",
                            f"**{L[win]}** is better: **{hi:.0f}** points against **{lo:.0f}** for **{L[weak]}**, a gap of **{abs(gap):.0f}** points ({_verdict(hi, 'en')} versus {_verdict(lo, 'en')}).")
        diffs = []
        if S[0].date != S[1].date:
            diffs.append(_t(lang, f"tarih (A {_date_tr(S[0].date, lang)}, B {_date_tr(S[1].date, lang)})", f"date (A {_date_tr(S[0].date, lang)}, B {_date_tr(S[1].date, lang)})"))
        if S[0].vehicle_id != S[1].vehicle_id:
            diffs.append(_t(lang, f"araç (A {P[0]['name']}, B {P[1]['name']})", f"vehicle (A {P[0]['name']}, B {P[1]['name']})"))
        if (S[0].lat, S[0].lon) != (S[1].lat, S[1].lon):
            diffs.append(_t(lang, f"konum (A {_place(S[0].lat, S[0].lon)}, B {_place(S[1].lat, S[1].lon)})", f"place (A {_place(S[0].lat, S[0].lon)}, B {_place(S[1].lat, S[1].lon)})"))
        what = _t(lang, "İkisi şunlarda ayrılıyor: " + "; ".join(diffs) + ".", "They differ in: " + "; ".join(diffs) + ".") if diffs else None

        def edge(x, y, higher_better, tol):
            if abs(x - y) <= tol:
                return "="
            return "A" if (x > y) == higher_better else "B"
        rows = [
            [_t(lang, "Puan", "Score"), f"{ia:.0f}", f"{ib:.0f}", edge(ia, ib, True, 1.5)],
            [_t(lang, "Puanı kısan", "Held down by"), LIMIT_SHORT[aa["limiting_factor"]][lang] if aa["limiting_factor"] else "—", LIMIT_SHORT[ab["limiting_factor"]][lang] if ab["limiting_factor"] else "—", ""],
            [_t(lang, "Güneşli zaman", "Time in sunlight"), _fmt_pct(ma["illumination_fraction"]), _fmt_pct(mb["illumination_fraction"]), edge(ma["illumination_fraction"], mb["illumination_fraction"], True, 0.02)],
            [_t(lang, "En uzun gece", "Longest night"), f"{ma['longest_darkness_hours']:.0f} h", f"{mb['longest_darkness_hours']:.0f} h", edge(ma["longest_darkness_hours"], mb["longest_darkness_hours"], False, 6)],
            [_t(lang, "Dünya görünür", "Earth in view"), _fmt_pct(ma["dte_fraction"]), _fmt_pct(mb["dte_fraction"]), edge(ma["dte_fraction"], mb["dte_fraction"], True, 0.02)],
            [_t(lang, "En uzun Dünya kesintisi", "Longest Earth outage"), f"{ma['longest_outage_hours']:.0f} h", f"{mb['longest_outage_hours']:.0f} h", edge(ma["longest_outage_hours"], mb["longest_outage_hours"], False, 6)],
            [_t(lang, "Ortalama eğim", "Average slope"), f"{ma['mean_slope_deg']:.1f}°", f"{mb['mean_slope_deg']:.1f}°", edge(ma["mean_slope_deg"], mb["mean_slope_deg"], False, 0.5)],
        ]
        table = {"type": "table", "title": _t(lang, "A ve B yan yana", "A and B side by side"), "columns": [_t(lang, "Ölçü", "Measure"), "A", "B", _t(lang, "Daha iyi", "Better")], "rows": rows}

        reasons = []
        if aa["limiting_factor"] != ab["limiting_factor"]:
            for i, a_ in ((0, aa), (1, ab)):
                if a_["limiting_factor"]:
                    reasons.append(_t(lang, f"**{L[i]}** puanı şu yüzden kısılıyor: {LIMIT_TEXT[a_['limiting_factor']]['tr']}.", f"**{L[i]}** is held down because {LIMIT_TEXT[a_['limiting_factor']]['en']}."))
        d_sun = ma["illumination_fraction"] - mb["illumination_fraction"]
        if abs(d_sun) >= 0.05:
            w_ = 0 if d_sun > 0 else 1
            reasons.append(_t(lang, f"**{L[w_]}** zamanın %{100 * abs(d_sun):.0f} puan daha fazlasında güneş alıyor.", f"**{L[w_]}** is in sunlight {100 * abs(d_sun):.0f} percentage points more of the time."))
        d_night = ma["longest_darkness_hours"] - mb["longest_darkness_hours"]
        if abs(d_night) >= 24:
            w_ = 1 if d_night > 0 else 0
            reasons.append(_t(lang, f"**{L[1 - w_]}**'nın en uzun gecesi {abs(d_night):.0f} saat daha uzun.", f"**{L[1 - w_]}** has a longest night {abs(d_night):.0f} hours longer."))
        d_out = ma["longest_outage_hours"] - mb["longest_outage_hours"]
        if abs(d_out) >= 24:
            w_ = 1 if d_out > 0 else 0
            reasons.append(_t(lang, f"**{L[1 - w_]}**'nın Dünya ile en uzun kesintisi {abs(d_out):.0f} saat daha uzun.", f"**{L[1 - w_]}** loses contact with Earth for {abs(d_out):.0f} hours longer at worst."))
        d_slope = ma["mean_slope_deg"] - mb["mean_slope_deg"]
        if abs(d_slope) >= 2:
            w_ = 1 if d_slope > 0 else 0
            reasons.append(_t(lang, f"**{L[1 - w_]}** daha dik bir zeminde ({abs(d_slope):.1f}° fark).", f"**{L[1 - w_]}** sits on steeper ground ({abs(d_slope):.1f}° more)."))
        blocks = [head]
        if what:
            blocks.append(what)
        blocks.append(table)
        if reasons:
            blocks.append(_t(lang, "**Farkı yaratan şeyler:**", "**What makes the difference:**"))
            blocks.append({"type": "bullets", "items": reasons[:4]})
        elif not tie:
            blocks.append(_t(lang, "Sayılar birbirine yakın; fark birkaç küçük etkenin toplamından geliyor.", "The numbers are close; the gap comes from several small effects adding up."))
        memory = {"table": table}
        if weak is not None:  # what would lift the weaker one: the engine's own date scan on it
            rows_ = tools.date_scan(self.dem, P[weak], S[weak], 60, earliest=self._earliest(S[weak]))
            cur = next(r for r in rows_ if r["date"] == S[weak].date)
            _, _, bd = self._best_pick(rows_, S[weak])
            if bd["index"] - cur["index"] >= 2:
                blocks.append(_t(lang, f"**{L[weak]}** için başlangıcı **{_date_tr(bd['date'], lang)}** yaparsan puanı {cur['index']:.0f} → {bd['index']:.0f} olur.",
                                       f"For **{L[weak]}**, starting on **{_date_tr(bd['date'], lang)}** would take it {cur['index']:.0f} → {bd['index']:.0f}."))
                act = {"type": "apply_scenario", "date": bd["date"], "scenario": L[weak].lower()}
                blocks.append({"type": "actions", "buttons": [{"label": _t(lang, f"{L[weak]}: {_date_tr(bd['date'], lang)} tarihini uygula", f"Apply {_date_tr(bd['date'], lang)} to {L[weak]}"), "action": act}]})
                memory["suggestion"] = act
        x = L[weak] if weak is not None else "B"
        blocks.append({"type": "chips", "items": [_t(lang, f"{x} neden daha düşük?", f"Why is {x} lower?"), _t(lang, f"{x}'yı nasıl iyileştirebilirim?" if x == "A" else f"{x}'yi nasıl iyileştirebilirim?", f"How can I improve {x}?"),
                                                  _t(lang, "A senaryosunu özetle", "Summarize scenario A")]})
        return {"blocks": blocks, "memory": memory}

    # ------------------------------------------------------------------ reactions: people answering the assistant, not asking it
    def _h_affirm(self, msg, norm, ent, ctx, mem, lang):
        """'yes' / 'ok': take up what was just offered; with nothing on offer, ask what to do."""
        if mem.get("suggestion"):
            return self._h_apply(msg, norm, ent, ctx, mem, lang)
        return {"blocks": [_t(lang, "Süper! Peki neyle devam edelim?", "Great! What shall we look at next?"), self._follow(lang, "result" if ctx.get("scenarios") else "more")]}

    def _h_deny(self, msg, norm, ent, ctx, mem, lang):
        return {"blocks": [_t(lang, "Tamam, hiçbir şeyi değiştirmiyorum. Başka bir şey merak ediyor musun?", "Okay, I will not change anything. Is there something else you wonder about?"),
                           self._follow(lang, "result" if ctx.get("scenarios") else "more")], "memory": {"suggestion": None}}

    def _h_complaint(self, msg, norm, ent, ctx, mem, lang):
        """Frustration or an insult: no defensiveness, no lecture; say sorry once, and offer a way out."""
        return {"blocks": [_t(lang, "Üzgünüm, işine yaramadım. Neyin yanlış ya da eksik olduğunu bana bir cümleyle söylersen düzeltmeye çalışırım; ben sorularını tam anlayamamış olabilirim. İstersen şunlardan birini dene:",
                                    "Sorry, I was not useful there. Tell me in one sentence what was wrong or missing and I will try to fix it; I may simply have misunderstood your question. You could also try one of these:"),
                           self._follow(lang, "result" if ctx.get("scenarios") else "more")]}

    def _h_repeat(self, msg, norm, ent, ctx, mem, lang):
        blocks = mem.get("last_blocks")
        if not blocks:
            return {"blocks": [_t(lang, "Tekrar edecek bir cevabım yok henüz; bir şey sor, ben de anlatayım.", "I have nothing to repeat yet; ask me something and I will answer."), self._follow(lang, "more")]}
        return {"blocks": [_t(lang, "Tabii, tekrar söylüyorum:", "Sure, once more:")] + list(blocks) + [self._follow(lang, "more")]}

    def _h_elaborate(self, msg, norm, ent, ctx, mem, lang):
        """'go on' / 'and why?': the natural next question of the last topic, answered."""
        topic = mem.get("topic_intent")
        nxt = NEXT_STEP.get(topic)
        if not nxt:
            return {"blocks": [_t(lang, "Hangi konuda devam edeyim? Bir şey sorarsan genişletirim.", "What should I go deeper on? Ask something and I will expand it."), self._follow(lang, "result" if ctx.get("scenarios") else "more")]}
        if nxt == "table_request" and not mem.get("table"):
            nxt = "explain_result"
        reply = getattr(self, f"_h_{nxt}")(msg, norm, ent, ctx, mem, lang)
        # the conversation has moved on to what was just answered: 'tell me more' again continues from here, and 'say that again' repeats it
        reply.setdefault("memory", {}).update({"topic_intent": nxt, "last_blocks": [b for b in reply["blocks"] if not (isinstance(b, dict) and b.get("type") in ("chips", "actions"))][:8]})
        return reply

    def _h_confused(self, msg, norm, ent, ctx, mem, lang):
        """'I did not get it': the same result in a few plain sentences, with no technical words."""
        scs = ctx.get("scenarios") or []
        if len(scs) > 1 and mem.get("topic_intent") == "compare_results":
            return self._plain_compare(scs, lang)
        sc = self._scenario(ctx, None, mem, norm, lang)
        if not sc:
            return {"blocks": [_t(lang, "Hangi kısmı anlamadığını söylersen sade anlatırım. Önce bir analiz çalıştırman gerekebilir.", "Tell me which part was unclear and I will put it simply. You may need to run an analysis first."),
                               self._follow(lang, "more")]}
        return self._plain_result(sc, lang, closing=True)

    def _plain_result(self, sc, lang, closing):
        """The result in a few plain sentences: the score, the one big problem, how much Sun and Earth there is."""
        p = self._profile(sc)
        m, a = tools.assess(self.dem, p, sc)
        idx = a["vehicle_index"]
        plain = {"slope": ("zemin araç için fazla eğimli", "the ground is too sloped for the vehicle"), "comms": ("Dünya'yla hiç konuşamıyor", "it can never talk to Earth"),
                 "power": ("hiç güneş alamıyor", "it never gets sunlight"), "thermal": ("gece çok uzun sürüyor ve araç donabilir", "the night is very long and the vehicle could freeze"),
                 "energy": ("pili erken bitiyor", "its battery runs out too early")}
        lines = [_t(lang, f"Kısaca: **{p['name']}**, bu noktada ve bu tarihte 100 üzerinden **{idx:.0f}** alıyor. Bu **{_verdict(idx, lang)}** demek.",
                          f"In short: **{p['name']}** gets **{idx:.0f} out of 100** at this place and date. That is **{_verdict(idx, lang)}**.")]
        if a["limiting_factor"]:
            why = plain[a["limiting_factor"]]
            lines.append(_t(lang, f"Puanı düşüren tek büyük şey: {why[0]}.", f"The one big thing pulling it down: {why[1]}."))
        else:
            lines.append(_t(lang, "Puanı bozan büyük bir sorun yok.", "Nothing big is dragging the score down."))
        lines.append(_t(lang, f"Ay'da 30 günün yaklaşık **%{100 * m['illumination_fraction']:.0f}**'inde güneş var, **%{100 * m['dte_fraction']:.0f}**'inde Dünya görünüyor.",
                              f"Over 30 days the Sun is up about **{100 * m['illumination_fraction']:.0f}%** of the time, and Earth can be seen **{100 * m['dte_fraction']:.0f}%** of the time."))
        if a.get("risk"):
            r = a["risk"]
            lines.append(_t(lang, f"Görevi {r['samples']} kez farklı koşullarla denedim: **10 denemenin {round(10 * r['p_success'])} tanesinde** güvenle iniyor ve en az {r['goal_days']:.0f} gün çalışıyor.",
                                  f"I tried the mission {r['samples']} times with varied conditions: it lands safely and works at least {r['goal_days']:.0f} days in **{round(10 * r['p_success'])} out of 10** tries."))
        if closing:
            lines.append(_t(lang, "Hangi kısmı daha açık anlatayım?", "Which part should I explain more?"))
        return {"blocks": lines + [self._follow(lang, "result")]}

    def _plain_compare(self, scs, lang):
        S = [Scenario(s["date"], s["vehicleId"], float(s["lat"]), float(s["lon"])) for s in scs[:2]]
        idx = [tools.assess(self.dem, self._profile(s), s)[1]["vehicle_index"] for s in S]
        if abs(idx[0] - idx[1]) < 2:
            line = _t(lang, f"Kısaca: A ve B neredeyse aynı ({idx[0]:.0f} ve {idx[1]:.0f}); hangisini seçtiğin pek fark etmez.", f"In short: A and B are about the same ({idx[0]:.0f} and {idx[1]:.0f}); it hardly matters which you pick.")
        else:
            w = "A" if idx[0] > idx[1] else "B"
            line = _t(lang, f"Kısaca: **{w}** daha iyi ({max(idx):.0f} puana karşı {min(idx):.0f}). Daha yüksek puan, aracın o yerde ve o tarihte daha rahat çalışacağı anlamına gelir.",
                            f"In short: **{w}** is better ({max(idx):.0f} against {min(idx):.0f}). A higher score means the vehicle copes more easily at that place and date.")
        return {"blocks": [line, _t(lang, "İstersen farkın nedenini de anlatırım.", "I can also tell you what causes the difference."), {"type": "chips", "items": [_t(lang, "İki sonucu karşılaştır", "Compare the two results")]}]}

    def _h_doubt(self, msg, norm, ent, ctx, mem, lang):
        """'are you sure?': answer honestly where every number comes from, and show the inputs behind this result."""
        intro = _t(lang, "Haklı bir soru, kontrol etmen iyi. Hiçbir rakamı uydurmuyorum: Güneş ve Dünya'nın konumları NASA'nın SPICE verisinden, arazi NASA LRO LOLA yükseklik verisinden hesaplanıyor. "
                         "Bilmediğim araç değerlerini tahmin etmiyorum, “bilinmiyor” diye bırakıyorum. Sınırlarım da var: puan bir hesap modeli, bir görev garantisi değil.",
                         "Fair question, and good that you check. I do not make numbers up: the Sun and Earth positions come from NASA SPICE, the terrain from NASA LRO LOLA elevation data. "
                         "Vehicle values I do not know are left as “unknown”, never guessed. I also have limits: the score is a calculation model, not a mission guarantee.")
        if not ctx.get("scenarios"):
            return {"blocks": [intro, self._follow(lang, "more")]}
        dq = self._h_data_quality(msg, norm, ent, ctx, mem, lang)
        return {"blocks": [intro] + [b for b in dq["blocks"] if not (isinstance(b, dict) and b.get("type") == "chips")] + [_t(lang, "Kendin de doğrulayabilirsin: Moon Map'te aynı noktaya aynı tarihte tıklarsan Güneş ve Dünya açılarını görürsün.", "You can verify it yourself: click the same point on Moon Map for the same date and you will see the Sun and Earth angles."), self._follow(lang, "more")],
                "memory": dq.get("memory", {})}

    def _h_best_date(self, msg, norm, ent, ctx, mem, lang):
        sc = self._scenario(ctx, ent, mem, norm, lang)  # 'the best date in December at Haworth': the message's month and place count
        if not sc:
            return self._need_scenario(lang)
        p = self._profile(sc)
        rows = tools.date_scan(self.dem, p, sc, span_days=60, earliest=self._earliest(sc))
        month = ent.dates[0][:7] if ent.dates and ent.dates[0] in ent.months else None
        if month:  # 'the best date in December': only December's start dates compete
            rows = [r for r in rows if r["date"][:7] == month] or rows
        cur = next(r for r in rows if r["date"] == sc.date)
        top, best_run, pick = self._best_pick(rows, sc)
        gain = top - cur["index"]
        peaks = []  # the best separate windows: top scores that are at least 20 days apart
        for r in sorted(rows, key=lambda r: -r["index"]):
            if all(abs((datetime.fromisoformat(r["date"]) - datetime.fromisoformat(q["date"])).days) >= 20 for q in peaks):
                peaks.append(r)
            if len(peaks) == 5:
                break
        table_rows = [[_date_tr(r["date"], lang) + (_t(lang, " (şimdiki)", " (current)") if r["date"] == sc.date else ""), f"{r['index']:.0f}", _fmt_pct(r["sun"]), _fmt_pct(r["dte"]), f"{r['dark_h']:.0f} h"] for r in sorted(peaks, key=lambda r: r["date"])]
        span_tr, span_en = ("bu ay", "this month") if month else ("±60 gün", "±60 days")
        where = _site_name(sc.lat, sc.lon) or _place(sc.lat, sc.lon)
        table = {"type": "table", "title": _t(lang, f"{where} için en iyi başlangıç tarihleri ({span_tr}, aynı araç)", f"Best start dates at {where} ({span_en}, same vehicle)"),
                 "columns": [_t(lang, "Başlangıç", "Start"), _t(lang, "Puan", "Score"), _t(lang, "Güneş", "Sun"), _t(lang, "Dünya", "Earth"), _t(lang, "En uzun gece", "Longest night")], "rows": table_rows}
        blocks = []
        if gain < 1.0:
            blocks.append(_t(lang, f"Tarihi oynatmak burada pek işe yaramıyor: seçtiğin tarih ({cur['index']:.0f}) {span_tr} içindeki en iyi sonuçla ({top:.0f}) aşağı yukarı aynı.",
                                   f"Moving the date would not help much here: your date ({cur['index']:.0f}) is about as good as the best in {span_en} ({top:.0f})."))
            memory = {"table": table}
        else:
            days = (datetime.fromisoformat(pick["date"]) - datetime.fromisoformat(sc.date)).days
            direction = _t(lang, f"{abs(days)} gün {'ileri' if days > 0 else 'geri'}", f"{abs(days)} days {'later' if days > 0 else 'earlier'}")
            why = []
            if pick["sun"] - cur["sun"] > 0.03:
                why.append(_t(lang, f"güneşli zaman {_fmt_pct(cur['sun'])} → {_fmt_pct(pick['sun'])}", f"time in sunlight {_fmt_pct(cur['sun'])} → {_fmt_pct(pick['sun'])}"))
            if pick["dte"] - cur["dte"] > 0.03:
                why.append(_t(lang, f"Dünya görünürlüğü {_fmt_pct(cur['dte'])} → {_fmt_pct(pick['dte'])}", f"Earth in view {_fmt_pct(cur['dte'])} → {_fmt_pct(pick['dte'])}"))
            if cur["dark_h"] - pick["dark_h"] > 12:
                why.append(_t(lang, f"en uzun gece {cur['dark_h']:.0f} h → {pick['dark_h']:.0f} h", f"longest night {cur['dark_h']:.0f} h → {pick['dark_h']:.0f} h"))
            if cur["limit"] and not pick["limit"]:
                why.append(_t(lang, f"puanı kısan sınır ({LIMIT_TEXT[cur['limit']]['tr']}) kalkıyor", f"the ceiling ({LIMIT_TEXT[cur['limit']]['en']}) disappears"))
            blocks.append(_t(lang, f"Evet, tarihi oynatmak işe yarıyor. Başlangıcı **{_date_tr(pick['date'], lang)}** yaparsan ({direction}) puan **{cur['index']:.0f} → {pick['index']:.0f}** oluyor (+{gain:.0f}). Araç ve konum aynı kalıyor.",
                                   f"Yes, moving the date helps. Starting on **{_date_tr(pick['date'], lang)}** ({direction}) lifts the score **{cur['index']:.0f} → {pick['index']:.0f}** (+{gain:.0f}), with the same vehicle and place."))
            if len(best_run) > 2:
                blocks.append(_t(lang, f"Aynı puanı {_range_text(best_run[0]['date'], best_run[-1]['date'], lang)} arasındaki her başlangıç da veriyor; yani tam gün tutmak şart değil.",
                                       f"Any start between {_range_text(best_run[0]['date'], best_run[-1]['date'], lang)} gives the same score, so the exact day is not critical."))
            if why:
                blocks.append(_t(lang, "Nedeni: " + "; ".join(why) + ". (Ay'ın gün ve gece döngüsü yaklaşık 29,5 günde tekrar eder, o yüzden iyi tarihler dalgalar halinde gelir.)",
                                       "Why: " + "; ".join(why) + ". (The Moon's day and night repeat about every 29.5 days, so good dates come in waves.)"))
            memory = {"table": table, "suggestion": {"type": "apply_scenario", "date": pick["date"]}}
            blocks.append({"type": "actions", "buttons": [{"label": _t(lang, f"{_date_tr(pick['date'], lang)} tarihini uygula", f"Apply {_date_tr(pick['date'], lang)}"), "action": {"type": "apply_scenario", "date": pick["date"]}}]})
        blocks.insert(1, table)
        blocks.append(self._follow(lang, "date"))
        return {"blocks": blocks, "memory": memory}

    def _h_compare_dates(self, msg, norm, ent, ctx, mem, lang):
        sc = self._scenario(ctx)
        if not sc:
            return self._need_scenario(lang)
        base = date.fromisoformat(sc.date)
        dates = []
        for d in ent.dates:
            if isinstance(d, dict):
                n = d["shift_days"]
                dates.append(((date.today() if d.get("from_today") else base) + timedelta(days=n)).isoformat())
            else:
                dates.append(d)
        if not dates:
            return {"blocks": [_t(lang, "Hangi tarihi denememi istersin? Örneğin: \"5 Aralık'ta başlarsak ne olur?\"", "Which date should I try? For example: \"What if we start on December 5?\""),
                               {"type": "chips", "items": ["Bir hafta sonra başlarsak ne olur?", "1 Ocak'ta başlarsak?"] if lang == "tr" else ["What if we start a week later?", "What if we start on January 1?"]}]}
        dates = dates[:3]
        p = self._profile(sc)
        cands = ([sc.date] if sc.date not in dates else []) + dates
        out = []
        for d in cands:
            m, a = tools.assess(self.dem, p, Scenario(d, sc.vehicle_id, sc.lat, sc.lon))
            out.append((d, m, a))
        rows = [[_date_tr(d, lang) + (_t(lang, " (şimdiki)", " (current)") if d == sc.date else ""), f"{a['vehicle_index']:.0f}", _fmt_pct(m["illumination_fraction"]), _fmt_pct(m["dte_fraction"]), f"{m['longest_darkness_hours']:.0f} h"] for d, m, a in out]
        table = {"type": "table", "title": _t(lang, "Tarih karşılaştırması", "Date comparison"), "columns": [_t(lang, "Başlangıç", "Start"), _t(lang, "Puan", "Score"), _t(lang, "Güneş", "Sun"), _t(lang, "Dünya", "Earth"), _t(lang, "En uzun gece", "Longest night")], "rows": rows}
        best = max(out, key=lambda o: o[2]["vehicle_index"])
        cur = next(o for o in out if o[0] == sc.date)
        if best[0] == sc.date:
            txt = _t(lang, f"Şimdiki tarih ({cur[2]['vehicle_index']:.0f}) denediklerinin en iyisi.", f"Your current date ({cur[2]['vehicle_index']:.0f}) is the best of those.")
            memory = {"table": table}
            actions = []
        else:
            delta = best[2]["vehicle_index"] - cur[2]["vehicle_index"]
            txt = _t(lang, f"En iyisi **{_date_tr(best[0], lang)}**: {best[2]['vehicle_index']:.0f} (şimdikinden {delta:+.0f}).", f"The best is **{_date_tr(best[0], lang)}**: {best[2]['vehicle_index']:.0f} ({delta:+.0f} vs now).")
            memory = {"table": table, "suggestion": {"type": "apply_scenario", "date": best[0]}}
            actions = [{"type": "actions", "buttons": [{"label": _t(lang, f"{_date_tr(best[0], lang)} tarihini uygula", f"Apply {_date_tr(best[0], lang)}"), "action": {"type": "apply_scenario", "date": best[0]}}]}]
        return {"blocks": [table, txt] + actions + [self._follow(lang, "date")], "memory": memory}

    def _h_best_vehicle(self, msg, norm, ent, ctx, mem, lang):
        if ent.vehicles:
            return self._h_try_option(msg, norm, ent, ctx, mem, lang)
        if ctx.get("free") and not (mem.get("free") or {}).get("site_given"):
            return self._best_vehicle_everywhere(ent, ctx, mem, lang)
        sc = self._scenario(ctx)
        if not sc:
            return self._need_scenario(lang)
        rows = tools.vehicle_scan(self.dem, list(self.profiles.values()), sc)
        battery_only = [r for r in rows if self.profiles[r["id"]]["solar_w"]["value"] == 0]
        ranked = self._by_odds(sorted([r for r in rows if r not in battery_only], key=lambda r: (-r["index"], -r["known_inputs"])), sc)
        cur_rank = next((i + 1 for i, r in enumerate(ranked) if r["id"] == sc.vehicle_id), None)
        table = {"type": "table", "title": _t(lang, "Bu konum ve tarih için araçlar", "Vehicles for this place and date"),
                 "columns": [_t(lang, "Sıra", "Rank"), _t(lang, "Araç", "Vehicle"), _t(lang, "Başarı şansı", "Success chance"), _t(lang, "Puan", "Score"), _t(lang, "Bilinen girdi", "Known inputs")],
                 "rows": [[str(i + 1), r["name"] + (" ◀" if r["id"] == sc.vehicle_id else ""), self._pct_cell(r, lang), f"{r['index']:.0f}", f"{r['known_inputs']}/7"] for i, r in enumerate(ranked[:7])]}
        best = ranked[0]
        ties = sum(1 for r in ranked if abs(r["index"] - best["index"]) < 0.5)
        star = self._star_note(ranked[:7], lang)
        blocks = [_t(lang, f"Bu yer ve tarih için en iyi seçim **{best['name']}**: benzetimli başarı şansı **%{100 * best['p']:.0f}**, puan {best['index']:.0f}." + (f" Seçtiğin araç {cur_rank}. sırada." if cur_rank else ""),
                           f"The best choice for this place and date is **{best['name']}**: simulated success chance **{100 * best['p']:.0f}%**, score {best['index']:.0f}." + (f" Your vehicle is ranked {cur_rank}." if cur_rank else "")), table] + ([star] if star else [])
        if ties > 2:
            blocks.append(_t(lang, f"Dikkat: {ties} araç birbirine çok yakın puan alıyor, çünkü çoğunun eğim sınırı, güç tüketimi ya da ısıtıcı bilgisi bilinmiyor. Bilinmeyen değeri tahmin etmiyorum, bu yüzden sıralama büyük ölçüde konumun kendisini yansıtıyor.",
                              f"Note: {ties} vehicles score almost the same because their slope limit, power use or heater are unknown for most of them. I never guess unknown values, so this ranking mostly reflects the place itself."))
        if battery_only:
            blocks.append(_t(lang, f"Yalnızca pille çalışan {len(battery_only)} aracı ({', '.join(r['name'] for r in battery_only[:3])}…) sıralamaya almadım: bunlar günler için tasarlanmış, 30 günlük bir pencere onlar için anlamlı değil.",
                                   f"I left out the {len(battery_only)} battery-only vehicles ({', '.join(r['name'] for r in battery_only[:3])}…): they are built for days, so a 30-day window is meaningless for them."))
        blocks.append(self._follow(lang, "more"))
        return {"blocks": blocks, "memory": {"table": table}}

    def _by_odds(self, ranked, sc, k=5):
        """The k best by score, each run through the mission simulation, re-ranked by its success chance (the score breaks ties);
        the rest keep their order below them."""
        head = []
        for r in ranked[:k]:
            risk = mission_risk(self.profiles[r["id"]], self.dem, sc.lat, sc.lon, sc.start_utc(), tools.WINDOW_DAYS, 60)
            head.append({**r, "p": risk["p_success"], "conf": risk["confidence"]["level"]})
        head.sort(key=lambda r: (-round(r["p"], 2), -r["index"]))
        return head + ranked[k:]

    @staticmethod
    def _pct_cell(r, lang):
        """A success chance for a table, with * when part of the risk could not be checked (the footnote says why)."""
        if not r or r.get("p") is None:
            return "—"
        star = " *" if r.get("conf") == "low" else ""
        return (f"%{100 * r['p']:.0f}" if lang == "tr" else f"{100 * r['p']:.0f}%") + star

    @staticmethod
    def _star_note(rows, lang):
        if any(r and r.get("conf") == "low" for r in rows):
            return _t(lang, "* Bu aracın bazı verileri (çoğunlukla ısıtıcı bilgisi) yayımlanmamış; o risk hesaplanamadığı için şans iyimser olabilir.",
                            "* Some of this vehicle's data (usually whether it has a heater) is not published; that risk could not be checked, so the chance may be optimistic.")
        return None

    def _best_vehicle_everywhere(self, ent, ctx, mem, lang):
        """'What should we send on 12 September?': no place named, so every named candidate site is tried and the best vehicle at each is listed."""
        sc0 = self._free_scenario(ent, mem, lang)
        self._assumed = [a for a in self._assumed if not a.startswith(("konum", "place", "araç", "vehicle"))]  # the place and vehicle are what this answer finds out
        per_site, scans = [], {}
        for name, (la, lo) in SITES.items():
            sc = Scenario(sc0["date"], sc0["vehicleId"], la, lo)
            try:
                scan = tools.vehicle_scan(self.dem, list(self.profiles.values()), sc)
            except ValueError:  # a site outside the loaded elevation data
                continue
            scans[name] = (sc, scan)
            ranked = self._by_odds(sorted([r for r in scan if self.profiles[r["id"]]["solar_w"]["value"] != 0], key=lambda r: (-r["index"], -r["known_inputs"])), sc, k=3)
            if ranked:
                per_site.append((name, ranked[0], ranked[1] if len(ranked) > 1 else None))
        if not per_site:
            return {"blocks": [_t(lang, "Bu tarih için hesaplayabildiğim bir konum çıkmadı.", "I could not compute any site for that date.")]}
        per_site.sort(key=lambda t: (-round(t[1]["p"], 2), -t[1]["index"]))
        table = {"type": "table", "title": _t(lang, f"{_date_tr(sc0['date'], lang)} için her konumun en iyi aracı", f"Best vehicle at each site for {_date_tr(sc0['date'], lang)}"),
                 "columns": [_t(lang, "Konum", "Site"), _t(lang, "En iyi araç", "Best vehicle"), _t(lang, "Başarı şansı", "Success chance"), _t(lang, "Puan", "Score"), _t(lang, "Sonraki", "Next")],
                 "rows": [[n.capitalize(), b["name"], self._pct_cell(b, lang), f"{b['index']:.0f}", f"{s['name']} ({self._pct_cell(s, lang)})" if s else "—"] for n, b, s in per_site]}
        n0, b0, _ = per_site[0]
        ties = sum(1 for r in per_site if abs(r[1]["p"] - b0["p"]) < 0.01 and abs(r[1]["index"] - b0["index"]) < 0.5)
        region = _t(lang, "Ay'ın güney kutbu için ", "For the lunar south pole, ") if ent.region else ""
        blocks = [_t(lang, f"{region}**{_date_tr(sc0['date'], lang)}** başlangıcıyla en iyi seçim **{n0.capitalize()}** bölgesine **{b0['name']}**: {tools.WINDOW_DAYS} günlük pencerede benzetimli başarı şansı **%{100 * b0['p']:.0f}**, puan {b0['index']:.0f}/100.",
                           f"{region}starting **{_date_tr(sc0['date'], lang)}**, the best choice is **{b0['name']}** at **{n0.capitalize()}**: simulated success chance **{100 * b0['p']:.0f}%** over the {tools.WINDOW_DAYS}-day window, score {b0['index']:.0f}/100."), table,
                  _t(lang, f"{len(SITES)} aday bölgenin her birinde projedeki tüm araçları gerçek Güneş/Dünya konumları ve LOLA arazisiyle puanladım, her bölgenin en iyi üçünü de {N_SAMPLES} kez benzetimle (iniş noktası, birkaç günlük gecikme, doğrulanmamış araç değerleri) denedim. Bilinmeyen değerleri tahmin etmiyorum.",
                           f"I scored every vehicle at each of the {len(SITES)} candidate sites with the real Sun and Earth positions and LOLA terrain, and ran the best three at each through {N_SAMPLES} simulated missions (touchdown point, a few days of delay, unverified vehicle figures). Unknown values are never guessed.")]
        star = self._star_note([x for _, b, s in per_site for x in (b, s)], lang)
        if star:
            blocks.insert(2, star)
        if self.profiles[b0["id"]].get("kind") == "rover":  # a rover does not land on its own: name the best lander as well
            sc_w, scan_w = scans[n0]
            landers = [r for r in scan_w if self.profiles[r["id"]].get("kind") == "lander" and self.profiles[r["id"]]["solar_w"]["value"] != 0]
            if landers:
                bl = self._by_odds(sorted(landers, key=lambda r: (-r["index"], -r["known_inputs"])), sc_w, k=3)[0]
                blocks.insert(1, _t(lang, f"Not: {b0['name']} bir gezgin (rover), oraya bir iniş aracıyla taşınır. Kendi başına inen araçlar arasında {n0.capitalize()} için en iyisi **{bl['name']}**: başarı şansı %{100 * bl['p']:.0f}, puan {bl['index']:.0f}.",
                                          f"Note: {b0['name']} is a rover and needs a lander to get there. Among the vehicles that land on their own, the best at {n0.capitalize()} is **{bl['name']}**: {100 * bl['p']:.0f}% success chance, score {bl['index']:.0f}."))
        if ties > 1:
            blocks.append(_t(lang, f"{ties} konum aynı puanda; bir konum adı söylersen o konumu ayrıntılı bakarım.", f"{ties} sites tie at the top; name one and I will look at it in detail."))
        blocks.append({"type": "chips", "items": [_t(lang, f"{n0.capitalize()} için puan neden böyle?", f"Why is the score what it is at {n0.capitalize()}?"), _t(lang, "Tarihi nasıl oynatabiliriz?", "How can we shift the date?")]})
        mem["free"].update({"lat": SITES[n0][0], "lon": SITES[n0][1], "site": n0.capitalize(), "vehicleId": b0["id"]})  # a follow-up continues with the winner
        return {"blocks": blocks, "memory": {"table": table}}

    def _h_best_site(self, msg, norm, ent, ctx, mem, lang):
        if ent.sites or ent.coords:
            return self._h_try_option(msg, norm, ent, ctx, mem, lang)
        sc = self._scenario(ctx)
        if not sc:
            return self._need_scenario(lang)
        p = self._profile(sc)
        rows = tools.site_scan(self.dem, p, sc)
        here = next(r for r in rows if r["label"] == "here")
        rows = [r for r in rows if r is here or not (r["label"][0].isalpha() and _site_name(r["lat"], r["lon"]) == _site_name(here["lat"], here["lon"]) is not None)]  # the named site you stand on, once
        by_light = bool({"light", "most"} & concepts(msg)) and "slope" not in norm  # 'where does it get the most light': the sunlight ranks, the score breaks ties
        ranked = sorted(rows, key=lambda r: (-r["sun"], -r["index"]) if by_light else -r["index"])
        best = ranked[0]
        table = {"type": "table", "title": _t(lang, "Yakın noktalar ve aday bölgeler" + (" (en çok güneş alana göre)" if by_light else ""), "Nearby points and candidate sites" + (" (most sunlight first)" if by_light else "")),
                 "columns": [_t(lang, "Yer", "Place"), _t(lang, "Puan", "Score"), _t(lang, "Güneş", "Sun"), _t(lang, "Dünya", "Earth"), _t(lang, "Eğim", "Slope")],
                 "rows": [[("▶ " if r is here else "") + (_site_name(r["lat"], r["lon"]) or _t(lang, "şimdiki nokta", "current point") if r["label"] == "here" else r["label"].capitalize() if r["label"][0].isalpha() else r["label"]), f"{r['index']:.0f}", _fmt_pct(r["sun"]), _fmt_pct(r["dte"]), f"{r['slope']:.1f}°"] for r in ranked[:6]]}
        blocks = []
        if by_light:
            name = _site_name(best["lat"], best["lon"]) or (_t(lang, "şimdiki nokta", "the current point") if best["label"] == "here" else best["label"])
            blocks.append(_t(lang, f"**{_date_tr(sc.date, lang)}** başlangıçlı 30 günde en çok güneş alan yer **{name}** ({_place(best['lat'], best['lon'])}): zamanın **%{100 * best['sun']:.0f}** kadarı güneşli, puanı {best['index']:.0f}.",
                                   f"Over the 30 days from **{_date_tr(sc.date, lang)}** the sunniest place is **{name}** ({_place(best['lat'], best['lon'])}): in sunlight **{100 * best['sun']:.0f}%** of the time, score {best['index']:.0f}."))
            memory = {"table": table, "suggestion": {"type": "apply_scenario", "lat": round(best["lat"], 4), "lon": round(best["lon"] if best["lon"] <= 180 else best["lon"] - 360, 4)}}
        elif best["index"] - here["index"] < 1.0:
            blocks.append(_t(lang, f"Yakında ya da aday bölgeler arasında şimdiki noktadan belirgin biçimde daha iyi bir yer bulamadım ({here['index']:.0f}).",
                                   f"I did not find a place nearby or among the candidate sites that is clearly better than the current point ({here['index']:.0f})."))
            memory = {"table": table}
        else:
            blocks.append(_t(lang, f"Daha iyi bir yer var: **{best['label']}** ({_place(best['lat'], best['lon'])}), puan {here['index']:.0f} → **{best['index']:.0f}**.",
                                   f"A better place exists: **{best['label']}** ({_place(best['lat'], best['lon'])}), score {here['index']:.0f} → **{best['index']:.0f}**."))
            act = {"type": "apply_scenario", "lat": round(best["lat"], 4), "lon": round(best["lon"] if best["lon"] <= 180 else best["lon"] - 360, 4)}
            memory = {"table": table, "suggestion": act}
        blocks.insert(1, table)
        if memory.get("suggestion"):
            blocks.append({"type": "actions", "buttons": [{"label": _t(lang, "Bu konumu uygula", "Apply this place"), "action": memory["suggestion"]}]})
        blocks.append(_t(lang, "Not: noktalar 1,5 km aralıkla seçildi; küçük bir kayma bile ufuk çizgisini değiştirir.", "Note: the points are 1.5 km apart; even a small shift changes the skyline."))
        return {"blocks": blocks, "memory": memory}

    def _h_requirements(self, msg, norm, ent, ctx, mem, lang):
        """'What features should a vehicle have to go there then?': the needs read off the real Sun, Earth and terrain, and the
        project's vehicles that already meet them (by simulated success probability)."""
        sc = self._scenario(ctx, ent, mem, norm, lang)
        if not sc:
            return self._need_scenario(lang)
        self._assumed = [a for a in self._assumed if not a.startswith(("araç", "vehicle"))]  # the answer is about any vehicle
        q = tools.requirements(self.dem, list(self.profiles.values()), sc)
        where = _site_name(sc.lat, sc.lon) or _place(sc.lat, sc.lon)
        dark, out = q["dark_h"], q["outage_h"]
        need_kwh = 50 * dark / 1000
        items = []
        if dark >= 48:
            items.append(_t(lang, f"**Gece:** en uzun karanlık **{dark:.0f} saat** ({dark / 24:.1f} gün)" + (f", {q['dark_start']} UTC'de başlıyor" if q["dark_start"] else "") +
                             f". Isıtıcısız elektronik bunu atlatamaz (varsayımım: 48–120 saat): ya **radyoizotop ısı kaynağı** ya da geceyi ısıtıcıyla geçirecek kadar pil gerekir. Gece her 1 W'lık yük için {dark:.0f} Wh demek; 50 W'lık bir hayatta kalma yüküyle en az **{need_kwh:.1f} kWh** pil.",
                             f"**Night:** the longest darkness is **{dark:.0f} h** ({dark / 24:.1f} days)" + (f", starting {q['dark_start']} UTC" if q["dark_start"] else "") +
                             f". Unheated electronics do not get through that (my assumption: 48–120 h): it needs a **radioisotope heat source**, or a battery that runs heaters all night. Every watt drawn at night costs {dark:.0f} Wh; with a 50 W survival load that is at least **{need_kwh:.1f} kWh** of battery."))
        else:
            items.append(_t(lang, f"**Gece:** en uzun karanlık yalnızca {dark:.0f} saat: kendi ısı kaynağı şart değil; 50 W'lık bir yük için {need_kwh:.1f} kWh pil yeter.",
                                  f"**Night:** the longest darkness is only {dark:.0f} h: no heat source of its own is needed; {need_kwh:.1f} kWh of battery carries a 50 W load through it."))
        items.append(_t(lang, f"**Güneş paneli:** Güneş zamanın %{100 * q['sunlit']:.0f} kadarında görünüyor ama ufkun yalnızca ortalama **{q['sun_el_mean']:.1f}°** üstünde (en çok {q['sun_el_max']:.1f}°): paneller yatay değil **dikey ya da yana bakan** olmalı ve Güneş'i izlemeli.",
                              f"**Solar array:** the Sun is up {100 * q['sunlit']:.0f}% of the time but only **{q['sun_el_mean']:.1f}°** above the horizon on average (at most {q['sun_el_max']:.1f}°): the panels must be **vertical or side-facing**, not flat, and track the Sun."))
        if out >= 72:
            items.append(_t(lang, f"**İletişim:** Dünya en uzun **{out:.0f} saat** ({out / 24:.1f} gün) görünmüyor: ya bir **röle uydusu** ya da bu süre boyunca veriyi saklayacak bellek gerekir (Dünya zamanın %{100 * q['dte']:.0f} kadarında görünür).",
                                  f"**Communications:** Earth is out of view for up to **{out:.0f} h** ({out / 24:.1f} days): it needs a **relay satellite**, or storage for that long (Earth is in view {100 * q['dte']:.0f}% of the time)."))
        else:
            items.append(_t(lang, f"**İletişim:** Dünya zamanın %{100 * q['dte']:.0f} kadarında görünüyor, en uzun kesinti {out:.0f} saat: doğrudan Dünya bağlantısı yeterli.",
                                  f"**Communications:** Earth is in view {100 * q['dte']:.0f}% of the time, longest gap {out:.0f} h: a direct link to Earth is enough."))
        items.append(_t(lang, f"**Eğim:** iniş alanının %90'ı {q['slope_p90']:.0f} dereceden az eğimli (ortalama {q['slope_mean']:.1f}°): eğim toleransı en az **{np.ceil(q['slope_p90']):.0f}°** olmalı.",
                              f"**Slope:** 90% of the landing area is under {q['slope_p90']:.0f}° (average {q['slope_mean']:.1f}°): the slope tolerance should be at least **{np.ceil(q['slope_p90']):.0f}°**."))
        fits = q["fits"]
        good = [f for f in fits if f["p_success"] >= 0.8 and f["confidence"] != "low"]  # 'already meets it' only where every risk could be checked
        table = {"type": "table", "title": _t(lang, "Projedeki araçlar bu koşullarda", "The project's vehicles in these conditions"),
                 "columns": [_t(lang, "Araç", "Vehicle"), _t(lang, "Başarı şansı", "Success chance"), _t(lang, "Tipik çalışma", "Typical working time"), _t(lang, "Puan", "Score")],
                 "rows": [[f["name"], self._pct_cell({"p": f["p_success"], "conf": f["confidence"]}, lang), _t(lang, f"{f['days_p50']:.0f} gün", f"{f['days_p50']:.0f} days"), f"{f['index']:.0f}"] for f in fits[:6]]}
        head = _t(lang, f"**{_date_tr(q['date'], lang)}** başlangıçlı 30 günde **{where}** için bir aracın sahip olması gerekenler (hepsi gerçek NASA verisinden hesaplandı):",
                        f"What a vehicle needs at **{where}** for the 30 days from **{_date_tr(q['date'], lang)}** (all from real NASA data):")
        if good:
            tail = _t(lang, "Bu ihtiyaçları zaten karşılayan araçlar (%80'den yüksek benzetimli başarı şansı): " + ", ".join(f"**{f['name']}**" for f in good[:3]) + ".",
                            "Vehicles that already meet these needs (over 80% simulated success): " + ", ".join(f"**{f['name']}**" for f in good[:3]) + ".")
        elif any(f["p_success"] >= 0.8 for f in fits):
            maybe = [f for f in fits if f["p_success"] >= 0.8][:3]
            tail = _t(lang, "%80'in üstünde görünenler var (" + ", ".join(f"**{f['name']}**" for f in maybe) + ") ama bazı verileri yayımlanmadığı için tüm riskleri kontrol edilemedi (*).",
                            "Some look above 80% (" + ", ".join(f"**{f['name']}**" for f in maybe) + "), but some of their data is not published, so not every risk could be checked (*).")
        else:
            tail = _t(lang, f"Projedeki araçlardan hiçbiri bu koşullarda %80 başarı şansına ulaşmıyor; en yakını **{fits[0]['name']}** (%{100 * fits[0]['p_success']:.0f}).",
                            f"None of the project's vehicles reaches an 80% success chance here; the closest is **{fits[0]['name']}** ({100 * fits[0]['p_success']:.0f}%).")
        star = self._star_note([{"conf": f["confidence"]} for f in fits[:6]], lang)
        return {"blocks": [head, {"type": "bullets", "items": items}, tail, table] + ([star] if star else []) + [self._follow(lang, "more")], "memory": {"table": table}}

    # what each topic is measured by, how to say it, and what counts as a change worth mentioning
    TOPIC_METRIC = {
        "sun_angle": {"key": "sun_el", "sign": 1, "unit": "°", "fmt": "{:.1f}", "min_gain": 0.3, "name": ("Güneş'in ufuktan yüksekliği", "the Sun's height above the horizon")},
        "sun_light": {"key": "sun", "sign": 1, "unit": "%", "fmt": "{:.0f}", "scale": 100, "min_gain": 2.0, "name": ("güneşli zaman", "time in sunlight")},
        "earth": {"key": "dte", "sign": 1, "unit": "%", "fmt": "{:.0f}", "scale": 100, "min_gain": 2.0, "name": ("Dünya görünürlüğü", "Earth visibility")},
        "slope": {"key": "slope", "sign": -1, "unit": "°", "fmt": "{:.1f}", "min_gain": 0.5, "name": ("zemin eğimi", "ground slope")},
        "night": {"key": "dark_h", "sign": -1, "unit": " h", "fmt": "{:.0f}", "min_gain": 12.0, "name": ("en uzun gece", "the longest night")},
    }

    KEYWORDS = (  # (topic, words that name it): a backstop for the learned topic head, used only when the head is unsure
        ("sun_angle", ("ufuk aci", "gunes aci", "gunesin aci", "sun angle", "sun elevation", "elevation", "isik gelis")),
        ("battery", ("batar", "battery", "pil ")),
        ("night", ("gece", "night", "karanlik", "darkness")),
        ("slope", ("egim", "slope", "steep", "flatter", "duz bir", "dikilik")),
        ("earth", ("dunya", "earth", "iletisim", "communication", "baglanti")),
        ("sun_light", ("gunes isig", "sunlight", "daylight", "aydinlik", "gunes")),
    )

    def _topic_by_keywords(self, norm):
        text = norm + " "
        for topic, words in self.KEYWORDS:
            if any(w in text for w in words):
                return topic, 0.9
        return None

    def _h_improve(self, msg, norm, ent, ctx, mem, lang):
        topic, tconf = self._topic(msg, norm)
        if topic in self.TOPIC_METRIC and tconf >= 0.5:
            return self._improve_topic(topic, ctx, lang, ent, mem, norm)
        if topic == "battery" and tconf >= 0.5:
            return self._h_battery_info(msg, norm, ent, ctx, mem, lang)
        sc = self._scenario(ctx, ent, mem, norm, lang)
        if not sc:
            return self._need_scenario(lang)
        p = self._profile(sc)
        m0, a0 = tools.assess(self.dem, p, sc)
        cur = a0["vehicle_index"]
        dates = tools.date_scan(self.dem, p, sc, 60, earliest=self._earliest(sc))
        _, _, bd = self._best_pick(dates, sc)
        sites = tools.site_scan(self.dem, p, sc)
        bs = max(sites, key=lambda r: r["index"])
        veh = [r for r in tools.vehicle_scan(self.dem, list(self.profiles.values()), sc) if self.profiles[r["id"]]["solar_w"]["value"] != 0]
        bv = max(veh, key=lambda r: r["index"])
        rows = [[_t(lang, "Tarihi oynat", "Shift the date"), _date_tr(bd["date"], lang), f"{bd['index']:.0f}", f"{bd['index'] - cur:+.0f}"],
                [_t(lang, "Konumu kaydır", "Move the place"), bs["label"] if bs["label"] != "here" else _t(lang, "değişiklik yok", "no change"), f"{bs['index']:.0f}", f"{bs['index'] - cur:+.0f}"],
                [_t(lang, "Aracı değiştir", "Change the vehicle"), bv["name"], f"{bv['index']:.0f}", f"{bv['index'] - cur:+.0f}"]]
        table = {"type": "table", "title": _t(lang, "İyileştirme seçenekleri (her biri tek başına)", "Ways to improve (each on its own)"),
                 "columns": [_t(lang, "Ne yapılır", "What to change"), _t(lang, "Hangisi", "To what"), _t(lang, "Yeni puan", "New score"), _t(lang, "Fark", "Change")], "rows": rows}
        order = sorted(rows, key=lambda r: -float(r[3]))
        blocks = [_t(lang, f"Şimdiki puan **{cur:.0f}**. Tek başına en çok kazandıran: **{order[0][0].lower()}** ({order[0][3]}).", f"Your score is **{cur:.0f}**. The single change that helps most: **{order[0][0].lower()}** ({order[0][3]})."), table]
        unknown = [k for k in ("battery_wh", "solar_w", "radioisotope_heat", "max_slope_deg", "survival_power_w") if p.get(k, {}).get("value") is None]
        if unknown:
            blocks.append(_t(lang, f"Ayrıca bu aracın {len(unknown)} önemli değeri bilinmiyor (örneğin eğim sınırı, güç tüketimi). Bunlar girilirse puan araca özgü hale gelir; şimdilik tahmin etmiyorum.",
                                   f"Also {len(unknown)} important values of this vehicle are unknown (for example slope limit, power use). Once they are filled in, the score becomes specific to the vehicle; for now I do not guess."))
        buttons, mem_out = [], {"table": table}
        for gain, label, act in sorted([
            (bd["index"] - cur, _t(lang, f"{_date_tr(bd['date'], lang)} tarihini uygula", f"Apply {_date_tr(bd['date'], lang)}"), {"type": "apply_scenario", "date": bd["date"]}),
            (bs["index"] - cur, _t(lang, "Bu noktayı uygula", "Apply this point"), {"type": "apply_scenario", "lat": round(bs["lat"], 4), "lon": round(bs["lon"] if bs["lon"] <= 180 else bs["lon"] - 360, 4)}),
            (bv["index"] - cur, _t(lang, f"{bv['name']} aracını uygula", f"Apply {bv['name']}"), {"type": "apply_scenario", "vehicleId": bv["id"]}),
        ], key=lambda x: -x[0]):
            if gain >= 1.0 and not (act.get("lat") is not None and bs["label"] == "here"):
                buttons.append({"label": label, "action": act})
        if buttons:
            mem_out["suggestion"] = buttons[0]["action"]
            blocks.append({"type": "actions", "buttons": buttons[:3]})
        blocks.append(self._follow(lang, "date"))
        return {"blocks": blocks, "memory": mem_out}

    # ------------------------------------------------------------------ 'what about ...?' (try an option without changing the page)
    def _h_try_option(self, msg, norm, ent, ctx, mem, lang):
        sc = self._scenario(ctx)
        if not sc:
            return self._need_scenario(lang)
        p = self._profile(sc)
        if HEATER.search(norm):
            return self._whatif_heater(sc, p, lang)
        if ent.vehicles:
            return self._compare_vehicles(sc, ent.vehicles[:3], lang)
        if ent.sites or ent.coords:
            return self._compare_sites(sc, p, ent, lang)
        if ent.dates:
            return self._h_compare_dates(msg, norm, ent, ctx, mem, lang)
        return {"blocks": [_t(lang, "Neyi denememi istersin: başka bir araç, bir konum, bir tarih ya da ısıtıcı eklemek? Örneğin \"peki ya Apollo 15 ile?\"", "What should I try: another vehicle, a place, a date, or adding a heater? For example \"what about Apollo 15?\""),
                           {"type": "chips", "items": ["Peki ya IM-1 ile?", "Shackleton nasıl?", "Isıtıcı olsa ne değişir?"] if lang == "tr" else ["What about IM-1?", "How about Shackleton?", "What if it had a heater?"]}]}

    def _whatif_heater(self, sc, p, lang):
        base_m, base_a = tools.assess(self.dem, p, sc)
        warm = {**p, "radioisotope_heat": {"value": True, "status": "unknown", "note": "what-if"}}
        _, a2 = tools.assess(self.dem, warm, sc)
        cell = lambda a: [f"{a['vehicle_index']:.0f}", LIMIT_TEXT[a["limiting_factor"]][_L(lang)] if a["limiting_factor"] else _t(lang, "yok", "none")]  # noqa: E731
        table = {"type": "table", "title": _t(lang, "Isıtıcı olsa (varsayımsal)", "With a heater (what-if)"),
                 "columns": [_t(lang, "Durum", "Case"), _t(lang, "Puan", "Score"), _t(lang, "Puanı kısan", "Held down by")],
                 "rows": [[_t(lang, "Şimdiki", "Current")] + cell(base_a), [_t(lang, "Radyoizotop ısıtıcı olsa", "With a radioisotope heater")] + cell(a2)]}
        gain = a2["vehicle_index"] - base_a["vehicle_index"]
        known = p["radioisotope_heat"]["value"]
        txt = _t(lang, f"Varsayımsal bir hesap: bu aracın ısıtıcı durumu şu an {'var' if known is True else 'yok' if known is False else 'bilinmiyor'}. Radyoizotop ısıtıcısı olsaydı geceyi pil harcamadan atlatırdı ve puan **{base_a['vehicle_index']:.0f} → {a2['vehicle_index']:.0f}** ({gain:+.0f}) olurdu.",
                       f"A what-if: this vehicle's heater status is currently {'yes' if known is True else 'no' if known is False else 'unknown'}. With a radioisotope heater it would get through the night without using battery power and the score would go **{base_a['vehicle_index']:.0f} → {a2['vehicle_index']:.0f}** ({gain:+.0f}).")
        return {"blocks": [txt, table, self._follow(lang, "more")], "memory": {"table": table}}

    def _compare_vehicles(self, sc, ids, lang):
        usable = [i for i in ids if i in self.profiles]
        skipped = [i for i in ids if i not in self.profiles]
        if not usable:
            return {"blocks": [_t(lang, "Bu araç için henüz analiz profilim yok (yalnızca iniş araçları ve gezginler için var).", "I do not have an analysis profile for that craft yet (only landers and rovers have one).")]}
        (iso, m, series), = tools._windows(self.dem, sc, sc.date, 1)
        rows, results = [], []
        for vid in [sc.vehicle_id] + [u for u in usable if u != sc.vehicle_id]:
            a = tools.assess_vehicle(self.profiles[vid], m, self.dem, sc.lat, sc.lon, series=series)
            results.append((vid, a))
            rows.append([self.profiles[vid]["name"] + (_t(lang, " (şimdiki)", " (current)") if vid == sc.vehicle_id else ""), f"{a['vehicle_index']:.0f}",
                         LIMIT_TEXT[a["limiting_factor"]][_L(lang)] if a["limiting_factor"] else _t(lang, "yok", "none"), f"{a['data_quality']['inputs_verified'] + a['data_quality']['inputs_unverified']}/7"])
        table = {"type": "table", "title": _t(lang, "Araç karşılaştırması (aynı yer ve tarih)", "Vehicle comparison (same place and date)"),
                 "columns": [_t(lang, "Araç", "Vehicle"), _t(lang, "Puan", "Score"), _t(lang, "Puanı kısan", "Held down by"), _t(lang, "Bilinen girdi", "Known inputs")], "rows": rows}
        cur = results[0][1]["vehicle_index"]
        best_id, best_a = max(results[1:], key=lambda r: r[1]["vehicle_index"]) if len(results) > 1 else results[0]
        blocks = [table]
        mem = {"table": table}
        if best_id != sc.vehicle_id:
            diff = best_a["vehicle_index"] - cur
            name = self.profiles[best_id]["name"]
            if abs(diff) < 0.5:
                blocks.append(_t(lang, f"**{name}** burada aynı puanı alıyor ({cur:.0f}); fark, aracın bilinmeyen değerlerinde. Sayfayı değiştirmedim.", f"**{name}** scores the same here ({cur:.0f}); any difference lies in the vehicle's unknown values. I did not change the page."))
            else:
                blocks.append(_t(lang, f"**{name}** burada {abs(diff):.0f} puan {'daha iyi' if diff > 0 else 'daha düşük'} ({best_a['vehicle_index']:.0f} / {cur:.0f}). Sayfayı değiştirmedim; istersen uygulayayım.",
                                       f"**{name}** scores {abs(diff):.0f} points {'higher' if diff > 0 else 'lower'} here ({best_a['vehicle_index']:.0f} vs {cur:.0f}). I did not change the page; I can apply it if you like."))
            act = {"type": "apply_scenario", "vehicleId": best_id}
            blocks.append({"type": "actions", "buttons": [{"label": _t(lang, f"{self.profiles[best_id]['name']} aracını uygula", f"Apply {self.profiles[best_id]['name']}"), "action": act}]})
            mem["suggestion"] = act
        if skipped:
            blocks.append(_t(lang, "Profili olmayan araçlar atlandı: " + ", ".join(skipped), "Skipped craft without a profile: " + ", ".join(skipped)))
        blocks.append(self._follow(lang, "more"))
        return {"blocks": blocks, "memory": mem}

    def _compare_sites(self, sc, p, ent, lang):
        named = [(n.capitalize(), *SITES[n]) for n in ent.sites[:3]] + [(_place(la, lo), la, lo) for la, lo in ent.coords[:2]]
        rows, results = [], []
        for label, la, lo in [(_t(lang, "şimdiki nokta", "current point"), sc.lat, sc.lon)] + named:
            s2 = Scenario(sc.date, sc.vehicle_id, la, lo % 360.0 if la <= -85 else lo)
            if la <= -85 and np.isnan(float(self.dem.height_at_latlon(la, s2.lon))):
                rows.append([label, "—", _t(lang, "arazi verisi yok", "no terrain data"), "—", "—"])
                continue
            m, a = tools.assess(self.dem, p, s2)
            results.append((label, la, lo, a))
            rows.append([label, f"{a['vehicle_index']:.0f}", _fmt_pct(m["illumination_fraction"]), _fmt_pct(m["dte_fraction"]), f"{m['mean_slope_deg']:.1f}°"])
        table = {"type": "table", "title": _t(lang, "Konum karşılaştırması (aynı araç ve tarih)", "Place comparison (same vehicle and date)"),
                 "columns": [_t(lang, "Yer", "Place"), _t(lang, "Puan", "Score"), _t(lang, "Güneş", "Sun"), _t(lang, "Dünya", "Earth"), _t(lang, "Eğim", "Slope")], "rows": rows}
        blocks, mem = [table], {"table": table}
        others = results[1:]
        if others:
            label, la, lo, a = max(others, key=lambda r: r[3]["vehicle_index"])
            diff = a["vehicle_index"] - results[0][3]["vehicle_index"]
            if abs(diff) < 0.5:
                blocks.append(_t(lang, f"**{label}** şimdiki noktayla aynı puanı alıyor. Sayfayı değiştirmedim.", f"**{label}** scores the same as the current point. I did not change the page."))
            else:
                blocks.append(_t(lang, f"**{label}** şimdiki noktadan {abs(diff):.0f} puan {'daha iyi' if diff > 0 else 'daha düşük'}. Sayfayı değiştirmedim.", f"**{label}** scores {abs(diff):.0f} points {'higher' if diff > 0 else 'lower'} than the current point. I did not change the page."))
            if diff >= 1:
                act = {"type": "apply_scenario", "lat": round(la, 4), "lon": round(lo if lo <= 180 else lo - 360, 4)}
                blocks.append({"type": "actions", "buttons": [{"label": _t(lang, f"{label} noktasını uygula", f"Apply {label}"), "action": act}]})
                mem["suggestion"] = act
        blocks.append(self._follow(lang, "more"))
        return {"blocks": blocks, "memory": mem}

    def _topic(self, msg, norm):
        """(topic, confidence): the learned head first, the keyword list when the head is unsure."""
        topic, tconf = self.nlu.predict_topic(msg) if hasattr(self.nlu, "predict_topic") else ("none", 0.0)
        if topic not in self.TOPIC_METRIC and topic != "battery" or tconf < 0.6:
            topic, tconf = self._topic_by_keywords(norm) or (topic, tconf)
        return topic, tconf

    def _improve_topic(self, topic, ctx, lang, ent=None, mem=None, norm=""):
        """'What can I do to raise X?': try other dates and nearby places for that one quantity and report what really moves it."""
        sc = self._scenario(ctx, ent, mem, norm, lang)
        if not sc:
            return self._need_scenario(lang)
        spec = self.TOPIC_METRIC[topic]
        p = self._profile(sc)
        key, sign, scale = spec["key"], spec["sign"], spec.get("scale", 1)
        val = lambda r: r[key] * scale  # noqa: E731
        fmt = lambda v: spec["fmt"].format(v) + spec["unit"]  # noqa: E731
        name = spec["name"][0 if lang == "tr" else 1]
        dates = tools.date_scan(self.dem, p, sc, 60, earliest=self._earliest(sc))
        cur = next(r for r in dates if r["date"] == sc.date)
        # best date: the best value, and among equally good ones the one closest to the chosen date
        top = max(sign * val(r) for r in dates)
        near = [r for r in dates if sign * val(r) >= top - 1e-9 - 0.02 * abs(top)]
        bd = min(near, key=lambda r: abs((datetime.fromisoformat(r["date"]) - datetime.fromisoformat(sc.date)).days))
        sites = tools.site_scan(self.dem, p, sc)
        here = next(r for r in sites if r["label"] == "here")
        bs = max(sites, key=lambda r: sign * val(r))
        gain_d, gain_s = sign * (val(bd) - val(cur)), sign * (val(bs) - val(here))
        def change(v):
            small = 0.05 if spec["unit"] == "°" else 0.5
            return "0" if abs(v) < small else f"{v:+.{1 if spec['unit'] == '°' else 0}f}"

        rows = [[_t(lang, "Başlangıç tarihini oynat", "Move the start date"), _date_tr(bd["date"], lang), fmt(val(bd)), change(sign * (val(bd) - val(cur)))],
                [_t(lang, "Başka bir noktaya geç (yakın nokta ya da aday bölge)", "Move to another point (nearby or a candidate site)"),
                 _t(lang, "şimdiki nokta", "current point") if bs["label"] == "here" else bs["label"], fmt(val(bs)), change(sign * (val(bs) - val(here)))]]
        table = {"type": "table", "title": _t(lang, f"{name.capitalize()}: neyi değiştirirsen ne olur (şimdi {fmt(val(cur))})", f"{name.capitalize()}: what changing each thing gives (now {fmt(val(cur))})"),
                 "columns": [_t(lang, "Ne yapılır", "What to change"), _t(lang, "Hangisi", "To what"), _t(lang, "Yeni değer", "New value"), _t(lang, "Fark", "Change")], "rows": rows}
        blocks = []
        sign_word = _t(lang, "iyileşme", "improvement")
        if topic == "sun_angle":
            blocks.append(_t(lang, f"Şu an Güneş, ışıklı olduğu saatlerde ortalama **{cur['sun_el']:.1f}°** yüksekte (en çok {cur['sun_el_max']:.1f}°). Kutup bölgesinde Güneş her zaman alçak kalır: Ay'ın ekseni yalnızca 1,5° eğik, bu yüzden Güneş'in yüksekliği en çok birkaç dereceye çıkar.",
                              f"Right now the Sun is on average **{cur['sun_el']:.1f}°** above the horizon while it is up (highest {cur['sun_el_max']:.1f}°). Near the pole the Sun always stays low: the Moon's axis is tilted only 1.5°, so the Sun climbs just a few degrees."))
        else:
            blocks.append(_t(lang, f"Şu an {name}: **{fmt(val(cur))}**.", f"Right now {name} is **{fmt(val(cur))}**."))
        blocks.append(table)
        mem = {"table": table}
        buttons = []
        if gain_d >= spec["min_gain"]:
            buttons.append({"label": _t(lang, f"{_date_tr(bd['date'], lang)} tarihini uygula", f"Apply {_date_tr(bd['date'], lang)}"), "action": {"type": "apply_scenario", "date": bd["date"]}})
            mem["suggestion"] = buttons[0]["action"]
        if gain_s >= spec["min_gain"] and bs["label"] != "here":
            act = {"type": "apply_scenario", "lat": round(bs["lat"], 4), "lon": round(bs["lon"] if bs["lon"] <= 180 else bs["lon"] - 360, 4)}
            buttons.append({"label": _t(lang, "Bu noktayı uygula", "Apply this point"), "action": act})
            mem.setdefault("suggestion", act)
        if buttons:
            best = "date" if gain_d >= gain_s else "site"
            blocks.append(_t(lang, f"En çok işe yarayan: **{'tarihi kaydırmak' if best == 'date' else 'noktayı kaydırmak'}** (+{max(gain_d, gain_s):.1f}{spec['unit']} {sign_word}).",
                                   f"What helps most: **{'moving the date' if best == 'date' else 'moving the point'}** (+{max(gain_d, gain_s):.1f}{spec['unit']} {sign_word})."))
            blocks.append({"type": "actions", "buttons": buttons})
        else:
            blocks.append(_t(lang, f"Tarihi ya da yakın noktayı değiştirmek {name} için anlamlı bir fark yaratmıyor; bu değer büyük ölçüde konumun enlemi ve mevsim tarafından belirleniyor.",
                                   f"Changing the date or moving a few kilometres does not make a meaningful difference to {name}; it is mostly set by the place's latitude and the season."))
        if topic == "sun_angle":
            lat_rows = tools.latitude_sweep(sc)
            lt = {"type": "table", "title": _t(lang, "Asıl kaldıraç: enlem (düz zeminde, aynı boylam)", "The real lever: latitude (flat ground, same longitude)"),
                  "columns": [_t(lang, "Enlem", "Latitude"), _t(lang, "Ekvatora doğru", "Towards the equator"), _t(lang, "Güneş en çok", "Sun highest"), _t(lang, "Güneşli zaman", "Time the Sun is up")],
                  "rows": [[f"{abs(r['lat']):.1f}°{'G' if lang == 'tr' else 'S'}", f"+{r['offset_deg']:.0f}°" if r["offset_deg"] else "-", f"{r['el_max']:.1f}°", f"{100 * r['up_fraction']:.0f}%"] for r in lat_rows]}
            blocks.append(lt)
            blocks.append(_t(lang, "Her derece ekvatora yaklaşmak Güneş'in en yüksek açısını yaklaşık 1° artırır, ama o enlemlerde 40 m'lik ayrıntılı arazi verimiz yok: tepelerin Güneş'i kesip kesmediğini yalnızca yaklaşık (470 m'lik veriyle) söyleyebilirim ve iniş bölgesi kutuptan uzaklaştıkça Dünya görünürlüğü ve ısı koşulları da değişir.",
                              "Each degree nearer the equator raises the Sun's highest angle by about 1°, but we do not have the 40 m detailed terrain at those latitudes: I can only judge whether hills block the Sun roughly (470 m data), and moving away from the pole also changes Earth visibility and the thermal conditions."))
            mem["table"] = lt
        blocks.append(self._follow(lang, "date"))
        return {"blocks": blocks, "memory": mem}

    # ------------------------------------------------------------------ topic answers
    def _h_sun_info(self, msg, norm, ent, ctx, mem, lang):
        sc = self._scenario(ctx, ent, mem, norm, lang)
        if not sc:
            return self._need_scenario(lang)
        p = self._profile(sc)
        m, a = tools.assess(self.dem, p, sc)
        e = a["checks"]
        t = _t(lang, f"Bu noktada Güneş 30 günün **{_fmt_pct(m['illumination_fraction'])}** kadarında ufuk çizgisinin üstünde. En uzun kesintisiz gün ışığı {m['longest_sunlight_hours']:.0f} saat, en uzun gece {m['longest_darkness_hours']:.0f} saat ({m['n_dark_periods']} ayrı karanlık dönem).",
                      f"At this spot the Sun is above the skyline for **{_fmt_pct(m['illumination_fraction'])}** of the 30 days. The longest unbroken daylight is {m['longest_sunlight_hours']:.0f} h and the longest night {m['longest_darkness_hours']:.0f} h ({m['n_dark_periods']} separate dark periods).")
        from ..point_report import point_report
        r = point_report(self.dem, sc.lat, sc.lon if sc.lon <= 180 else sc.lon - 360, f"{sc.date}T12:00:00")
        s = r["sun"]
        ang = _t(lang, f"Başlangıç gününde öğlen Güneş ufkun {s['elevation_deg']:.1f}° üstünde; ışınlar yere dikeyden {s['incidence_flat_deg']:.0f}° açıyla geliyor (90° = ışık yüzeyi sıyırır, çok verimsiz).",
                       f"On the first day at noon the Sun is {s['elevation_deg']:.1f}° above the horizon; the rays reach the ground {s['incidence_flat_deg']:.0f}° from vertical (90° means light skimming the ground, which is weak).")
        return {"blocks": [t, ang, self._follow(lang, "more")]}

    def _h_comms_info(self, msg, norm, ent, ctx, mem, lang):
        sc = self._scenario(ctx, ent, mem, norm, lang)
        if not sc:
            return self._need_scenario(lang)
        p = self._profile(sc)
        m, a = tools.assess(self.dem, p, sc)
        mode = a["checks"]["comms"]
        if mode.get("mode") == "relay":
            t = _t(lang, "Bu araç Dünya ile bir röle uydusu üzerinden konuşuyor, bu yüzden Dünya'yı kendisinin görmesi gerekmiyor.", "This vehicle talks to Earth through a relay satellite, so it does not need to see Earth itself.")
        elif mode.get("assessed"):
            when = tools.hours_to_iso(sc.date, m["outage_start_h"])
            t = _t(lang, f"Dünya 30 günün **{_fmt_pct(m['dte_fraction'])}** kadarında ufuk çizgisinin üstünde (doğrudan telsiz bağlantısı mümkün). En uzun kesinti {m['longest_outage_hours']:.0f} saat" + (f" ve {when} UTC'de başlıyor" if when else "") + f"; en uzun kesintisiz bağlantı {m['longest_contact_hours']:.0f} saat.",
                          f"Earth is above the skyline for **{_fmt_pct(m['dte_fraction'])}** of the 30 days (direct radio link possible). The longest outage is {m['longest_outage_hours']:.0f} h" + (f" and starts {when} UTC" if when else "") + f"; the longest unbroken contact is {m['longest_contact_hours']:.0f} h.")
            if m["dte_fraction"] == 0:
                t += _t(lang, " Yani bu pencerede hiç doğrudan bağlantı yok; bir röle uydusu gerekir.", " So there is no direct link at all in this window; a relay satellite would be needed.")
        else:
            t = _t(lang, "Bu aracın iletişim biçimi bilinmiyor, bu yüzden bağlantıyı puana katamıyorum.", "This vehicle's communication mode is unknown, so I cannot score the link.")
        return {"blocks": [t, self._follow(lang, "more")]}

    def _h_slope_info(self, msg, norm, ent, ctx, mem, lang):
        sc = self._scenario(ctx, ent, mem, norm, lang)
        if not sc:
            return self._need_scenario(lang)
        p = self._profile(sc)
        m, a = tools.assess(self.dem, p, sc)
        s = a["checks"]["slope"]
        lim = s.get("limit_deg")
        t = _t(lang, f"500 m'lik iniş alanında ortalama eğim **{m['mean_slope_deg']:.1f}°**, en dik yer {m['max_slope_deg']:.0f}°, yükseklik farkı {m['relief_m']:.0f} m.",
                      f"On the 500 m landing area the average slope is **{m['mean_slope_deg']:.1f}°**, the steepest spot {m['max_slope_deg']:.0f}°, and the height difference {m['relief_m']:.0f} m.")
        if lim:
            share = s["area_within_limit_fraction"]
            t += _t(lang, f" Bu aracın eğim sınırı {lim:.0f}°; alanın {_fmt_pct(share)} kadarı bu sınırın içinde.", f" This vehicle's slope limit is {lim:.0f}°; {_fmt_pct(share)} of the area is within it.")
        else:
            t += _t(lang, " Bu aracın eğim sınırı bilinmiyor, bu yüzden genel 15° kullanıldı.", " This vehicle's slope limit is unknown, so a generic 15° was used.")
        return {"blocks": [t, self._follow(lang, "more")]}

    def _h_night_thermal(self, msg, norm, ent, ctx, mem, lang):
        sc = self._scenario(ctx, ent, mem, norm, lang)
        if not sc:
            return self._need_scenario(lang)
        p = self._profile(sc)
        m, a = tools.assess(self.dem, p, sc)
        th = a["checks"]["thermal"]
        heat = p["radioisotope_heat"]["value"]
        dark_when = tools.hours_to_iso(sc.date, m["dark_start_h"])
        t = _t(lang, f"En uzun gece **{m['longest_darkness_hours']:.0f} saat**" + (f" ve {dark_when} UTC'de başlıyor." if dark_when else "."), f"The longest night is **{m['longest_darkness_hours']:.0f} hours**" + (f" and starts {dark_when} UTC." if dark_when else "."))
        if heat is True:
            t += _t(lang, " Bu aracın radyoizotop ısıtıcısı var, yani geceyi pil harcamadan sıcak geçirebilir.", " This vehicle has a radioisotope heater, so it can stay warm through the night without using battery power.")
        elif th["first_long_night_start_h"] is not None:
            when = th["first_long_night_start_h"]
            t += _t(lang, f" Isı kaynağı {'olmadığı' if heat is False else 'bilinmediği'} için, {th['first_long_night_hours']:.0f} saatlik gece {when:.0f}. saatte başlayınca aracın donma riski var (varsayım: ısıtıcısız elektronik yaklaşık 72 saat dayanır). Pencerenin yaklaşık {_fmt_pct(th['alive_fraction'])} kadarında çalışabilir.",
                          f" Because the vehicle has {'no' if heat is False else 'no known'} heat source, when the {th['first_long_night_hours']:.0f} h night starts at hour {when:.0f} there is a risk of freezing (assumption: unheated electronics last about 72 h). It can work for about {_fmt_pct(th['alive_fraction'])} of the window.")
        else:
            t += _t(lang, " 72 saati aşan bir gece yok, donma riski düşük.", " There is no night longer than 72 h, so the freezing risk is low.")
        return {"blocks": [t, self._follow(lang, "more")]}

    def _h_battery_info(self, msg, norm, ent, ctx, mem, lang):
        sc = self._scenario(ctx, ent, mem, norm, lang)
        if not sc:
            return self._need_scenario(lang)
        p = self._profile(sc)
        sweep = tools.battery_sweep(self.dem, p, sc, (1, 2, 4, 8) if not ent.multiplier else tuple(sorted({1, ent.multiplier})))
        if sweep is None:
            return {"blocks": [_t(lang, "Bu aracın pil kapasitesi ya da güneş paneli gücü bilinmiyor, bu yüzden pili hesaplayamıyorum. Değer uydurmak yerine eksik olduğunu söylüyorum.",
                                       "This vehicle's battery capacity or solar array power is unknown, so I cannot work out the battery. I would rather say it is missing than invent a value.")]}
        table = {"type": "table", "title": _t(lang, "Pil büyüklüğü ne kazandırır", "What a bigger battery buys"),
                 "columns": [_t(lang, "Pil", "Battery"), _t(lang, "Taşıyabileceği sürekli güç", "Constant load it can carry"), _t(lang, "Puan", "Score")],
                 "rows": [[f"{r['mult']:g}× ({r['battery_wh']:.0f} Wh)", f"{r['sustainable_w']:.1f} W", f"{r['index']:.0f}"] for r in sweep]}
        known = p.get("survival_power_w", {}).get("value") is not None
        txt = _t(lang, "Pil büyüdükçe aracın gece boyunca taşıyabileceği sürekli güç artıyor (tablo).", "As the battery grows, the constant power the vehicle can carry through the night grows (table).")
        if not known:
            txt += _t(lang, " Ama puan değişmiyor, çünkü bu aracın kendi güç tüketimi bilinmiyor; tüketimi bilmeden pilin 'yeterli' olup olmadığını söyleyemem. Tüketim değeri girilirse puan da pile göre değişir.",
                            " But the score stays the same, because this vehicle's own power use is unknown; without it I cannot say whether the battery is 'enough'. Once the power use is filled in, the score will follow the battery.")
        return {"blocks": [txt, table, self._follow(lang, "more")], "memory": {"table": table}}

    def _h_data_quality(self, msg, norm, ent, ctx, mem, lang):
        sc = self._scenario(ctx, ent, mem, norm, lang)
        if not sc:
            return self._need_scenario(lang)
        p = self._profile(sc)
        names = {"battery_wh": _t(lang, "Pil kapasitesi", "Battery capacity"), "solar_w": _t(lang, "Güneş paneli gücü", "Solar array power"), "radioisotope_heat": _t(lang, "Radyoizotop ısıtıcı", "Radioisotope heater"),
                 "max_slope_deg": _t(lang, "Eğim sınırı", "Slope limit"), "comms": _t(lang, "İletişim biçimi", "Communication mode"), "survival_power_w": _t(lang, "Hayatta kalma gücü", "Survival power"), "payload_power_w": _t(lang, "Yük gücü", "Payload power")}
        status = {"verified": _t(lang, "doğrulanmış", "verified"), "unverified": _t(lang, "proje verisinden", "from the project dataset"), "unknown": _t(lang, "bilinmiyor", "unknown")}
        rows = []
        for k, label in names.items():
            f = p.get(k) or {"value": None, "status": "unknown"}
            rows.append([label, "—" if f["value"] is None else str(f["value"]), status[f["status"]]])
        table = {"type": "table", "title": _t(lang, f"{p['name']} girdileri", f"Inputs for {p['name']}"), "columns": [_t(lang, "Girdi", "Input"), _t(lang, "Değer", "Value"), _t(lang, "Durum", "Status")], "rows": rows}
        t = _t(lang, "Güneş ve Dünya konumları NASA'nın SPICE efemerisinden, arazi NASA LRO LOLA yükseklik verisinden hesaplanıyor; bunlar ölçüm. Aracın kendi değerleri ise tabloda: bilinmeyenleri tahmin etmiyorum.",
                     "Sun and Earth positions come from NASA's SPICE ephemeris and the terrain from NASA LRO LOLA elevation data; those are measurements. The vehicle's own values are in the table: I do not guess the unknown ones.")
        return {"blocks": [t, table, self._follow(lang, "more")], "memory": {"table": table}}

    def _h_glossary(self, msg, norm, ent, ctx, mem, lang):
        from .glossary import find_term
        name = find_term(norm)
        if name is None:
            titles = [TERMS[k]["title"][lang] for k in TERMS]
            return {"blocks": [_t(lang, "Hangi terimi açıklayayım? Bildiklerim:", "Which term should I explain? The ones I know:"), {"type": "bullets", "items": titles}]}
        t = TERMS[name]
        return {"blocks": [f"**{t['title'][lang]}**: {t[lang]}", self._follow(lang, "more")]}

    # ------------------------------------------------------------------ actions on the page
    def _h_table_request(self, msg, norm, ent, ctx, mem, lang):
        tb = mem.get("table")
        if not tb:
            return {"blocks": [_t(lang, "Henüz tabloya dökecek bir sonucum yok. Önce bir şey sor: örneğin \"tarihi nasıl oynatabiliriz?\"", "I have nothing to put in a table yet. Ask something first, for example \"how can we shift the date?\""), self._follow(lang, "result")]}
        return {"blocks": [_t(lang, "İşte tablo:", "Here is the table:"), tb]}

    def _h_apply(self, msg, norm, ent, ctx, mem, lang):
        s = mem.get("suggestion")
        if not s:
            return {"blocks": [_t(lang, "Uygulayacak bir önerim yok. Önce \"tarihi nasıl oynatabiliriz?\" ya da \"daha iyi bir konum var mı?\" diye sor.", "I have no suggestion to apply yet. Ask \"how can we shift the date?\" or \"is there a better place?\" first.")]}
        if ctx.get("free"):  # no analysis on screen: the chat itself takes the value and uses it from now on
            return {"blocks": [_t(lang, "Tamam, bundan sonra bunu kullanıyorum. Şimdi bir şey sorabilirsin.", "Done: I will use that from now on. Ask me something.")], "actions": [s], "memory": {"suggestion": None}}
        return {"blocks": [_t(lang, "Tamam, uyguluyorum ve analizi yeniden çalıştırıyorum.", "Done: applying it and re-running the analysis.")], "actions": [s], "memory": {"suggestion": None}}

    def _h_change_context(self, msg, norm, ent, ctx, mem, lang):
        sc = self._scenario(ctx)
        act = {"type": "apply_scenario"}
        said = []
        base = date.fromisoformat(sc.date) if sc else date.today()
        for d in ent.dates[:1]:
            iso = d if isinstance(d, str) else ((date.today() if d.get("from_today") else base) + timedelta(days=d["shift_days"])).isoformat()
            act["date"] = iso
            said.append(_t(lang, f"tarih {_date_tr(iso, lang)}", f"date {_date_tr(iso, lang)}"))
        for vid in ent.vehicles[:1]:
            if vid in self.profiles:
                act["vehicleId"] = vid
                said.append(_t(lang, f"araç {self.profiles[vid]['name']}", f"vehicle {self.profiles[vid]['name']}"))
            else:
                return {"blocks": [_t(lang, "Bu araç için henüz analiz profilim yok (yalnızca iniş araçları ve gezginler için var).", "I do not have an analysis profile for that craft yet (only landers and rovers have one).")]}
        for name in ent.sites[:1]:
            la, lo = SITES[name]
            act.update(lat=la, lon=lo)
            said.append(_t(lang, f"konum {name.capitalize()}", f"place {name.capitalize()}"))
        for la, lo in ent.coords[:1]:
            act.update(lat=la, lon=lo)
            said.append(_t(lang, f"konum {_place(la, lo)}", f"place {_place(la, lo)}"))
        if len(act) == 1:
            return {"blocks": [_t(lang, "Neyi değiştireyim? Tarih, araç ya da konum söyle; örneğin \"tarihi 5 Aralık yap\" ya da \"aracı Apollo 15 yap\".", "What should I change? Give a date, vehicle or place, for example \"set the date to December 5\" or \"switch to Apollo 15\"."), {"type": "chips", "items": ["Tarihi 5 Aralık yap", "Konumu Shackleton yap"] if lang == "tr" else ["Set the date to December 5", "Set the place to Shackleton"]}]}
        return {"blocks": [_t(lang, "Tamam: " + ", ".join(said) + ". Analizi yeniden çalıştırıyorum.", "Done: " + ", ".join(said) + ". Re-running the analysis.")], "actions": [act]}


def _L(lang):
    return "tr" if lang == "tr" else "en"


def _t(lang, tr, en):
    return tr if lang == "tr" else en
