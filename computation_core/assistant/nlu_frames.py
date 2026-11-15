"""A small semantic parser, written by hand: which *concepts* a message contains, and which request they add up to.

The intent network is good at the shape of a sentence, but a whole mission plan in one message ("on 16 November I want to go
to the lunar south pole, which vehicle suits it best?") buries the question under the plan, and the network's confidence
drops below the point where it would act. People, though, recognise such a request by a few concepts that must appear
together: a *vehicle* and a *which / best*; a *when* and a *go / land*; a *where* and *the most light*. This module finds
those concepts (word stems, so Turkish suffixes do not matter) and combines them with explicit rules. Its answer is evidence
next to the network's (see Assistant.route): when both agree the assistant acts; when the network is unsure and the concepts
are clear, the concepts decide; a confident network is never overruled by a weak rule.

It also splits a message into clauses, so 'X, Y veya Z' can be read part by part, and a second question inside the same
message ("... and what features should it have?") is not lost.
"""
import re

from .text import normalize

_C = {
    "vehicle": r"\b(arac|lander|rover|vehicle|spacecraft|craft|uzay ara|modul|gezgin|robot)",
    "which": r"\b(hangi|hangisi|which|what (lander|vehicle|spacecraft|rover|craft)|ne (secs|gonder|yollas)|kim(i)? gonder)",
    "best": r"\b(en iyi|en uygun|uygun|ideal|mantikli|best|suit|fits?\b|should (we|i) (send|pick|use|choose|go)|secmeli|secmem|gondermeli|tercih|optimal|recommend|oner|daha iyi|better|pick|choose|rank|sirala)",
    "go": r"\b(gonder|yolla|gitmek|gitsin|gidecek|gitmeli|gits|git\b|inis|inmek|inme|inecek|inil|indir|land\b|landing\b|send|go\b|going|launch|firlat|ucur)",
    "apply": r"\b(apply|uygula)",
    # an event in the sky or the link ('when does the Sun set', 'where is the Sun now'): a question about the scenario, not a search
    "event": r"\b(batiyor|batar\b|batacak|batis|doguyor|dogar\b|dogacak|dogus|sunset|sunrise|sets\b|rises\b|gece|night|karanlik|dark|contact|lose|loss|kaybed|kopar|gorunur|visible|su an|right now|now\b|next|sonraki|basliyor|baslar\b|starts?\b)",
    "when": r"\b(ne zaman|when\b|hangi (tarih|gun|hafta|ay)|which (date|day|week|month)|what (date|day|time)|best (time|date|day|week|moment)|en (iyi|uygun) (tarih|gun|zaman|donem)|uygun (tarih|gun)|mantikli (gun|tarih))",
    "where": r"\b(nere|nerde|where\b|hangi (yer|konum|bolge|nokta|koordinat|enlem|krater)|which (place|site|spot|location|area|crater|coordinate)|what (place|site|spot|location))",
    "light": r"\b(isik|gunes|sun\b|sunlight|sunny|light|aydinl|illumin|daylight)",
    "most": r"\b(en cok|en fazla|most\b|daha cok|daha fazla|more\b|maximum|max\b|en uzun|longest)",
    "require": r"\b(ozellik|gereksinim|gereklilik|spec|specs|specification|requirement|must have|nasil bir (arac|lander|rover)|ne tur (bir )?arac|what kind of (vehicle|lander|rover|spacecraft)|what (sort|type) of (vehicle|lander|rover)|tasarla|design|sahip olmali|"
               r"(batarya|pil|battery|heater|isitici)\w* ne (kadar|buyuk) olmali|how (big|large|much) (a |of a )?(battery|heater|solar)\w* (does|do|should|would|must)|need(s)? to (survive|have|carry)|ne gerek|what does it need|what would it need|neye ihtiyac)",
    "about": r"(ne ise yararsin|ne isine yararsin|ne ise yarar bu (asistan|bot)|(ne|neler) yapabil(irsin|iyorsun|irsiniz)\b|what (can|do) you do|what are you for|who are you|sen kimsin|kimsin sen)",
    "howis": r"(\bnasil( gorunuyor| gorunur| olur| olurdu| cikar| durumda)?\s*$|\bhow (is|does|would|will) .* (look|be|do|turn out)\b|\bhow is \w+|\bnasil gorunuyor)",
    "improve": r"\b(artir|arttir|yukselt|iyilestir|improve|increase|raise|boost)",
    "compare": r"\b(karsilastir|compare|versus|vs\b)|\bmi yoksa (?!fark|degil|hayir|olmaz)|\bor\b(?! not\b)",
}
_RX = {k: re.compile(v) for k, v in _C.items()}
SPLIT = re.compile(r"[.,;!?]+|\s(?:ve|veya|ya da|yoksa|ama|fakat|ayrica|hem de|and|or|but|also|plus)\s")


def concepts(text: str) -> set[str]:
    n = normalize(text)
    return {k for k, rx in _RX.items() if rx.search(n)}


def frame(text: str, ent=None) -> tuple[str, float] | None:
    """(intent, strength) the concepts add up to, or None. Strength 1.0 = the concepts alone leave no doubt."""
    c = concepts(text)
    place = bool(ent and (ent.sites or ent.coords or ent.region))
    when = bool(ent and ent.dates)
    if "about" in c:
        return "about", 0.9
    if "apply" in c:
        return None  # 'apply the best date' is an answer to a suggestion, not a new question
    if "require" in c and ("vehicle" in c or "olmali" in normalize(text) or "need" in normalize(text) or "gerek" in normalize(text)):
        return "requirements", 1.0
    if "improve" in c:
        return None  # 'how do I raise X' is the network's (it knows the topics); none of the rules below fits it
    if "vehicle" in c and ("which" in c or "best" in c) and "compare" not in c:
        return "best_vehicle", 1.0 if "which" in c else 0.8
    event = "event" in c and "best" not in c
    if "when" in c and ("go" in c or place or "best" in c) and not event:
        return "best_date", 1.0
    if "where" in c and ("light" in c or "most" in c or "best" in c or "go" in c) and not (event and "most" not in c):
        return "best_site", 1.0 if ("light" in c or "best" in c) else 0.8
    if "howis" in c and (place or when) and not ({"best", "where", "which", "compare", "light"} & c):
        return "explain_result", 0.8
    return None


def clauses(text: str) -> list[str]:
    """The parts of a long message, each one long enough to be a question of its own."""
    parts = [p.strip() for p in SPLIT.split(" " + text + " ") if p and p.strip()]
    return [p for p in parts if len(p.split()) >= 2]
