"""Short human reactions to the assistant itself ("ok", "no", "I don't get it", "are you sure?", "go on", "you're useless").

These messages are only a few words long, so the language model is rarely confident about them, and a "which of these did you
mean?" answer to "tamam" is exactly the kind of thing that makes an assistant feel dead. The patterns below are narrow on purpose:
they only fire on short messages and on whole-message replies, never on a real question that happens to contain the word "no".
They run before the model; everything they do not catch is left to the model as before.
"""
import re

from .text import normalize

YES = set("""evet evvet tamam tamamdir olur oldu olsun tabi tabii ki aynen oyle kabul okey ok okay yes yeah yep yup sure alright fine go ahead do it please
 lutfen peki o zaman hadi yap bakalim sounds good absolutely of course right that one works for me ya abi kral hocam heh he haydi katiliyorum evt""".split())
NO = set("""hayir yok no nope nah olmaz istemem istemiyorum vazgec vazgectim dont do not that it cancel stop yapma never mind forget leave as is bos ver gec gerek
 simdilik bu sefer olmasin sagol tesekkurler thanks thank you ya kalsin degistirme changed my i iptal et now really""".split())
FILL = {"ya", "abi", "kral", "hocam", "heh", "lutfen", "please", "peki", "o", "zaman", "bakalim", "hadi", "haydi", "he", "ki"}  # words that can only accompany a yes
NO_STARTS = {"hayir", "yok", "no", "nope", "nah", "olmaz", "istemem", "istemiyorum", "vazgec", "vazgectim", "dont", "cancel", "iptal", "yapma", "stop"}

_R = re.compile
COMPLAINT = _R(r"berbat|rezalet|kotusun|cok kotu|bu cok kotu|ise yaramiyor|ise yaramaz|yardimci olmuyor|sacmalama|sacmaladin|sacmasin|sacma bir|aptal|salak|bikdim|zaman kaybi|useless|garbage|terrible|awful|"
               r"stupid|dumb|you suck|worthless|waste of (my )?time|not (very )?(helpful|helping)|so annoying|frustrating|i hate|disappointed|hayal kirikligi|ugrastirma|olmuyor bu|olmadi bu|wrong again|yine yanlis|what a joke|siniri|trash|pathetic|unusable|lame|hic faydan|ne bicim|rezil")
DOUBT = _R(r"are you positive|hard to believe|emin misin|emin degilim|bundan emin|are you (sure|certain|lying|making)|you sure|not (convinced|sure)|dont (believe|buy)|do not (believe|buy)|inanmiyorum|inanamiyorum|yalan|uydur|sacma geldi|i doubt|"
           r"prove|kanitla|ispat|kaynagin|how do you know|how can you be sure|where did .*(come|get)|where does .* come from|nerden|nereden (cikardin|biliyorsun|buldun)|dogru mu|is that (true|right|correct)|"
           r"gercekten mi|supheli|suspicious|making (this|it) up|cant be right|sounds wrong|that is wrong|thats wrong|kim dedi|who told you|ters geldi|yok artik")
CONFUSED_ANY = _R(r"what do you (even )?mean|totally lost|so lost|kafa karist|sade ol|normal words|simple words|hic anlamad|anlamad|anlayamad|anlamiyorum|kafam (karisti|allak|durdu)|(didnt|did not|dont|do not|cant|cannot) (get|understand|follow)|not understand|simplify|plain (english|words)|jargon|over my head|im lost|too technical|"
                  r"break it down|come again|like im (five|5)|simpler|in simple|put that more simply|sadelestir|basitlestir|cocuga anlatir")
CONFUSED_WITH_VERB = _R(r"\b(basit|sade|kisa|kisaca|oz)\b")
CONFUSED_VERB = _R(r"\b(anlat|anlatsana|anlatir|soyle|acikla|yaz|ifade)")
CONFUSED_WHOLE = {"ne demek", "bu ne demek", "bu ne demek simdi", "ne demek bu", "ne demek istiyorsun", "yani ne demek istiyorsun", "what do you mean", "what does that mean", "what does that even mean",
                  "what is that supposed to mean", "huh", "ne", "hm"}
REPEAT = _R(r"^tekrar\b|^(bir|bi) daha (soyle|yaz|goster)|yeniden soyle|tekrar (et|eder|goster|yaz|soyle)|say (that |it )?again|\brepeat\b|once more|one more time|what did you (just )?say|ne dedin|ne demistin|az once ne|show (that )?again|geri sar")
ELABORATE = _R(r"^(please go on|daha ayrintili( olsun)?|what happens (after that|next)|bir de nedenini soyle|devam( et)?|keep talking|devam etsene|biraz daha anlat|anlatmaya devam|more( please)?|go on|continue|keep going|carry on|and then|then what|what else|baska ne var|peki sonra|ve sonra|peki neden|neden peki|and why|why is that|say more|more info|more details?( please)?|"
               r"tell me more|elaborate|expand( on that)?|daha fazla( anlat| bilgi ver)?|ayrintili anlat|detaylandir|detay ver|biraz daha (detay|ac)( ver)?|anlatmaya devam et|explain further|gerisini de anlat)$")
TOPIC = set("""puan skor arac tarih konum gunes dunya pil batarya gece egim iletisim sinyal isik score date vehicle site location sun earth battery night slope signal light result sonuc""".split())
MAX_TOKENS = 9


def detect(message: str) -> str | None:
    """The reaction a short message is, or None (the model decides)."""
    s = normalize(message)
    toks = s.split()
    n = len(toks)
    if n == 0 or n > MAX_TOKENS:
        return None
    if COMPLAINT.search(s):
        return "complaint"
    if DOUBT.search(s):
        return "doubt"
    if (CONFUSED_ANY.search(s) and not any(t in TOPIC for t in toks)) or s in CONFUSED_WHOLE or (CONFUSED_WITH_VERB.search(s) and CONFUSED_VERB.search(s) and n <= 7):
        return "confused"
    if REPEAT.search(s):
        return "repeat"
    if ELABORATE.match(s):
        return "elaborate"
    if n <= 5 and all(t in NO for t in toks) and (toks[0] in NO_STARTS or s in ("bos ver", "gerek yok", "never mind", "nevermind", "forget it", "not now", "no thanks", "not really", "i changed my mind", "leave it as is", "bu sefer olmasin")):
        return "deny"
    if n <= 3 and toks[0] in NO_STARTS and all(t in NO or t in YES for t in toks):
        return "deny"
    if n <= 4 and all(t in YES for t in toks) and any(t not in FILL for t in toks):
        return "affirm"
    if re.fullmatch(r"(sure |yeah |ok |okay )?why not|(alright |ok |okay )?lets do it", s):
        return "affirm"
    return None
