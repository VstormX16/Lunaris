"""Synthetic training sentences for the assistant's intent model (Turkish and English).

The model never sees the evaluation sentences in eval_set.py; those are written separately and in other words, so the
accuracy reported by train_nlu.py measures understanding, not memory of these templates.
"""
import numpy as np

# intent -> {"en": [...], "tr": [...]}  (Turkish written without diacritics on purpose; text.normalize folds them anyway)
INTENTS = {
    "greet": {
        "en": ["hi", "hello", "hey", "hey there", "good morning", "good evening", "hello assistant", "yo", "hi there", "greetings", "howdy"],
        "tr": ["merhaba", "selam", "selamlar", "gunaydin", "iyi aksamlar", "merhaba asistan", "mrb", "slm", "hey", "naber kral", "selam hocam", "iyi gunler"],
    },
    "thanks": {
        "en": ["thanks", "thank you", "thanks a lot", "great thanks", "perfect thank you", "cheers", "much appreciated", "awesome thanks", "ok thanks that helps", "nice thank you"],
        "tr": ["tesekkurler", "sag ol", "cok sag ol", "eyvallah", "tesekkur ederim", "super sag ol", "harika tesekkurler", "saol", "eline saglik", "yardimin icin sag ol", "tamamdir sagol"],
    },
    "bye": {
        "en": ["bye", "goodbye", "see you", "see you later", "that is all for now", "i am done", "talk later", "good night", "catch you later"],
        "tr": ["gorusuruz", "hoscakal", "bay bay", "simdilik bu kadar", "bitti gorusuruz", "iyi geceler", "sonra konusuruz", "kendine iyi bak", "hadi eyvallah gorusuruz"],
    },
    "about": {
        "en": ["who are you", "what can you do", "what are you", "how can you help me", "what do you know", "help", "how do i use you", "what questions can i ask", "tell me about yourself", "what is this assistant", "are you a real ai", "did you build yourself"],
        "tr": ["sen kimsin", "ne yapabilirsin", "bana nasil yardim edebilirsin", "sen nesin", "neler soruyabilirim", "yardim", "seni nasil kullanirim", "ne biliyorsun", "kendini tanit", "bu asistan ne ise yariyor", "gercek bir yapay zeka misin", "seni kim yapti", "hazir model mi kullaniyorsun"],
    },
    "smalltalk": {
        "en": ["how are you", "how is it going", "what is up", "are you ok", "how do you do", "are you there", "do you like the moon", "tell me something interesting", "are you tired", "you are smart", "you are funny", "tell me a joke"],
        "tr": ["nasilsin", "naber", "iyi misin", "ne haber", "orada misin", "ay'i sever misin", "ilginc bir sey soyle", "yoruldun mu", "cok akillisin", "komiksin", "bir saka yap", "nasil gidiyor", "keyifler nasil"],
    },
    "explain_result": {
        "en": ["explain the result", "what does this result mean", "explain this score", "summarize the analysis", "give me a summary", "what is this result telling me", "break it down for me", "explain it simply", "walk me through the result", "what am i looking at", "tell me what this means"],
        "tr": ["sonucu acikla", "bu sonuc ne demek", "bu puani acikla", "analizi ozetle", "bana ozet gec", "bu sonuc bana ne soyluyor", "sade bir dille anlat", "sonucu anlat", "ne gordugumu anlat", "bu ekran ne anlatiyor", "anlamadim aciklar misin", "ozetle bakalim", "bunu bana basitce anlatir misin"],
    },
    "why_score": {
        "en": ["why is the score low", "why is my score so low", "why did it get this score", "what is holding the score down", "why only 15", "what is wrong with this site", "why is this result bad", "what drags the score down", "why such a poor result", "what is the problem here", "why did the score drop", "what is limiting this"],
        "tr": ["puan neden dusuk", "bu puan neden bu kadar dusuk", "neden bu puani aldi", "puani ne asagi cekiyor", "neden sadece 15", "bu konumun nesi kotu", "sonuc neden kotu cikti", "puan niye dusuk cikti", "neden bu kadar kotu", "sorun ne burada", "puani dusuren sey ne", "bunu sinirlayan sey ne", "niye boyle cikti"],
    },
    "improve": {
        "en": ["how can we improve this", "how do we get a better score", "what can i change to improve it", "how to make this better", "what would improve the result", "give me ideas to raise the score", "what are my options to improve", "how do i fix this", "what should i change", "can this be made better"],
        "tr": ["bunu nasil iyilestirebiliriz", "daha iyi puan icin ne yapmaliyiz", "iyilestirmek icin neyi degistirebilirim", "nasil daha iyi hale getiririz", "puani artirmak icin ne yapabiliriz", "bana iyilestirme onerileri ver", "secenekler neler", "bunu nasil duzeltiriz", "neyi degistirmeliyim", "bu daha iyi olabilir mi", "ne yapsak puan artar"],
    },
    "best_date": {
        "en": ["how can we shift the date to get a better result", "which date is best", "is there a better date", "when should we land instead", "scan other dates", "find the best date for this", "can a different date improve it", "what is the best landing window", "should i move the date", "try other dates and tell me the best", "which day gives the highest score", "best time to land here", "how should i change the date"],
        "tr": ["tarihi nasil oynatabiliriz", "hangi tarih daha iyi", "daha iyi bir tarih var mi", "ne zaman inmeliyiz", "diger tarihleri tara", "bunun icin en iyi tarihi bul", "baska bir tarih puani arttirir mi", "en iyi inis zamani ne zaman", "tarihi degistirmeli miyim", "baska tarihleri dene en iyisini soyle", "hangi gun en yuksek puani veriyor", "burasi icin en uygun tarih", "tarihi nasil kaydirirsak daha iyi olur", "ayni kosullarda hangi tarih daha iyi", "tarihi biraz oynatsak ne olur"],
    },
    "compare_dates": {
        "en": ["compare these two dates", "what if we land on another date", "how would it look on a different date", "compare with the date", "what happens if we start later", "what if the date was a week later", "how does the result change on that date", "check that date instead", "evaluate this date for me", "run it again for another day"],
        "tr": ["su iki tarihi karsilastir", "baska bir tarihte inersek ne olur", "farkli bir tarihte nasil gorunur", "su tarihle karsilastir", "bir hafta sonra baslarsak ne olur", "o tarihte sonuc nasil degisir", "onun yerine su tarihe bak", "bu tarihi benim icin degerlendir", "baska bir gun icin tekrar calistir", "tarih degisirse sonuc ne olur", "o gun icin hesapla"],
    },
    "best_vehicle": {
        "en": ["which vehicle is best here", "what is the best spacecraft for this site", "would another lander do better here", "compare the vehicles for this location", "which lander suits this place", "is there a better vehicle for this site", "rank the vehicles for me", "which craft would survive here", "what vehicle should i choose"],
        "tr": ["bu konum icin en uygun arac hangisi", "burada hangi arac daha iyi", "baska bir arac burada daha iyi olur mu", "araclari bu konum icin karsilastir", "hangi arac bu yere uyar", "bu yer icin daha iyi arac var mi", "araclari siralar misin", "hangi arac burada hayatta kalir", "hangi araci secmeliyim", "arac degistirsem daha mi iyi olur"],
    },
    "best_site": {
        "en": ["is there a better location", "which site is better for this vehicle", "compare the candidate sites", "where else could it land", "find a better spot nearby", "should we move the landing point", "what about other landing sites", "is there a nearby place with a higher score", "which location is best for this lander"],
        "tr": ["daha iyi bir konum var mi", "bu arac icin hangi konum daha iyi", "aday konumlari karsilastir", "baska nereye inebilir", "yakinlarda daha iyi bir nokta bul", "inis noktasini tasimali miyiz", "diger inis bolgeleri nasil", "yakinda daha yuksek puanli bir yer var mi", "bu arac icin en iyi konum neresi", "konumu degistirsek daha iyi olur mu"],
    },
    "sun_info": {
        "en": ["how much sunlight is there", "how long is the sun up", "what angle does the sun hit at", "how many hours of daylight", "is there enough sun", "tell me about the sunlight", "how sunny is this site", "what about the illumination", "how long are the nights here"],
        "tr": ["ne kadar gunes var", "gunes ne kadar sure acik", "gunes hangi acidan vuruyor", "kac saat gunes isigi var", "yeterli gunes var mi", "gunes isigi hakkinda bilgi ver", "burasi ne kadar gunesli", "aydinlik orani nasil", "geceler ne kadar uzun", "gunes ne zaman dogar"],
    },
    "comms_info": {
        "en": ["can it talk to earth", "will we have a line of sight to earth", "what about communications", "how long are the communication gaps", "can it contact earth directly", "is earth visible", "how often is the earth link blocked", "do we lose contact", "does it need a relay"],
        "tr": ["dunya ile iletisim kurabilir mi", "dunyayi gorebilecek miyiz", "iletisim durumu nasil", "iletisim kesintileri ne kadar", "dunyaya dogrudan ulasabilir mi", "dunya gorunuyor mu", "dunya baglantisi ne siklikla kesiliyor", "baglanti kopuyor mu", "role uydusu gerekir mi", "haberlesme nasil olur"],
    },
    "slope_info": {
        "en": ["is the ground too steep", "what about the slope", "is it safe to land here", "how steep is the terrain", "can it land on this slope", "tell me about the terrain", "is the landing area flat enough", "what is the slope here"],
        "tr": ["zemin cok dik mi", "egim nasil", "buraya inmek guvenli mi", "arazi ne kadar dik", "bu egime inebilir mi", "arazi hakkinda bilgi ver", "inis alani yeterince duz mu", "buradaki egim ne", "yer cok mu egimli"],
    },
    "night_thermal": {
        "en": ["can it survive the night", "will it freeze", "what about the lunar night", "does it have a heater", "how cold does it get", "will the electronics survive the dark", "can it make it through the long night", "what about the thermal side"],
        "tr": ["geceye dayanir mi", "donar mi", "ay gecesi ne olacak", "isitici var mi", "ne kadar soguk olur", "elektronikler karanliga dayanir mi", "uzun geceyi atlatabilir mi", "isil durum nasil", "gece hayatta kalir mi", "soguga dayanir mi"],
    },
    "battery_info": {
        "en": ["is the battery enough", "what if we use a bigger battery", "how long does the battery last", "what happens if the battery is doubled", "tell me about the power budget", "will the battery run flat", "how much power can it draw", "what about energy"],
        "tr": ["pil yeterli mi", "daha buyuk batarya kullanirsak ne olur", "batarya ne kadar dayanir", "batarya iki katina cikarsa ne olur", "guc butcesi nasil", "pil bitecek mi", "ne kadar guc cekebilir", "enerji durumu ne", "bataryayi buyutsek", "pil kac saat gider"],
    },
    "data_quality": {
        "en": ["how reliable is this", "can i trust these numbers", "is the data real", "which values are verified", "how sure are you", "are these estimates or measurements", "where does the data come from", "which inputs are unknown"],
        "tr": ["bu ne kadar guvenilir", "bu rakamlara guvenebilir miyim", "veriler gercek mi", "hangi degerler dogrulanmis", "ne kadar eminsin", "bunlar tahmin mi olcum mu", "veriler nereden geliyor", "hangi girdiler bilinmiyor", "kaynak nedir", "hangi veri eksik"],
    },
    "glossary": {
        "en": ["what does that term mean", "what is a veto", "what does illumination fraction mean", "what is dte", "define slope", "what is direct to earth", "explain what a lunar night is", "what is the suitability index", "what does held down by mean", "what is a skyline"],
        "tr": ["bu terim ne demek", "veto nedir", "aydinlik orani ne demek", "dte nedir", "egim nedir", "dogrudan dunya iletisimi ne demek", "ay gecesi nedir", "uygunluk indeksi ne", "held down by ne demek", "ufuk cizgisi nedir", "sunlit ne demek"],
    },
    "table_request": {
        "en": ["show it as a table", "put that in a table", "can you make a table", "give me a table of that", "list them in a table", "show a comparison table", "table please"],
        "tr": ["tablo olarak goster", "bunu tabloya koy", "tablo yapar misin", "bunun tablosunu ver", "tabloda listele", "karsilastirma tablosu goster", "tablo lutfen", "tablo halinde ver"],
    },
    "apply": {
        "en": ["apply it", "use that date", "yes do it", "ok use it", "go with that", "apply the suggestion", "set it", "yes please apply", "do it", "use the best one", "ok let us try it"],
        "tr": ["uygula", "bu tarihi kullan", "evet yap", "tamam kullan", "bununla devam et", "oneriyi uygula", "ayarla", "evet lutfen uygula", "hadi yap", "en iyisini kullan", "tamam bunu deneyelim"],
    },
    "change_context": {
        "en": ["change the date to december 5", "set the vehicle to apollo 15", "switch the location to shackleton", "use chandrayaan 3 instead", "move the date forward a week", "make it start on the first of january", "change the site", "pick another vehicle", "set the coordinates to minus 86 and 20"],
        "tr": ["tarihi 5 aralik yap", "araci apollo 15 olarak ayarla", "konumu shackleton yap", "onun yerine chandrayaan 3 kullan", "tarihi bir hafta ileri al", "baslangici 1 ocak yap", "konumu degistir", "baska bir arac sec", "koordinatlari eksi 86 ve 20 yap", "tarihi degistir"],
    },
    "unknown": {
        "en": ["what is the weather today", "write me a poem", "who won the football match", "what is the price of bitcoin", "tell me a recipe for pasta", "translate this sentence to french", "what is two plus two", "who is the president", "book me a flight", "play some music", "what is the capital of france", "how do i fix my car", "what time is it in tokyo", "recommend me a movie"],
        "tr": ["hava bugun nasil", "bana bir siir yaz", "mac kac kac bitti", "bitcoin fiyati ne", "makarna tarifi ver", "bu cumleyi fransizcaya cevir", "iki arti iki kac", "cumhurbaskani kim", "ucak bileti al", "muzik cal", "fransanin baskenti neresi", "arabam bozuldu ne yapayim", "tokyoda saat kac", "bana film oner", "dolar kac lira"],
    },
}

PREFIX = {"en": ["", "", "", "so ", "ok ", "hey ", "quick question ", "just wondering ", "please ", "and ", "also ", "one more thing "],
          "tr": ["", "", "", "peki ", "kral ", "hocam ", "ya ", "bi sey soracagim ", "sunu merak ettim ", "lutfen ", "bide ", "ayrica ", "simdi "]}
SUFFIX = {"en": ["", "", "", "?", " ?", " please", " pls", " for me", " here"],
          "tr": ["", "", "", "?", " ?", " lutfen", " acaba", " bence", " bunun icin", " burada"]}
CONTEXT_BITS = {"en": ["for this vehicle", "at this site", "with this date", "for apollo 15", "on december 5", "for shackleton", "for chandrayaan 3", "in this scenario", "here"],
                "tr": ["bu arac icin", "bu konumda", "bu tarihte", "apollo 15 icin", "5 aralikta", "shackleton icin", "chandrayaan 3 icin", "bu senaryoda", "burada"]}
# intents that are fine with a context phrase tacked on (greetings and thanks are not)
CONTEXTUAL = {"compare_results", "explain_result", "why_score", "improve", "best_date", "compare_dates", "best_vehicle", "best_site", "sun_info",
              "comms_info", "slope_info", "night_thermal", "battery_info", "data_quality"}


def _noise(s: str, rng) -> str:
    """Typing mistakes of the kind people really make: dropped, doubled and swapped letters, dropped words."""
    ws = s.split()
    if len(ws) > 3 and rng.random() < 0.12:
        ws.pop(int(rng.integers(len(ws))))
    out = []
    for w in ws:
        if len(w) > 3 and rng.random() < 0.18:
            i = int(rng.integers(1, len(w) - 1))
            kind = rng.integers(3)
            w = w[:i] + w[i + 1:] if kind == 0 else w[:i] + w[i] + w[i:] if kind == 1 else w[:i - 1] + w[i] + w[i - 1] + w[i + 1:]
        out.append(w)
    return " ".join(out)


INTENTS["try_option"] = {
    "en": ["what about {V}", "and {V}", "how about {V}", "what if we used {V}", "try {V}", "how would {V} do here", "and with {V}", "what about {S}", "how about {S}", "compare with {S}",
           "what if it had a heater", "what if it had a radioisotope heater", "what happens with a heater", "and {S}", "{S} instead", "how does {V} compare", "what would {S} look like"],
    "tr": ["peki ya {V} ile", "{V} olsa", "{V} ile nasil olur", "ya {V}", "{V} kullansak", "peki {S} nasil", "ya {S}", "{S} ile karsilastir", "isitici olsa ne degisir", "isitici eklesek ne olur",
           "{V} burada nasil calisir", "{S} olsa", "{V} ile karsilastirsak", "peki {S} icin", "isitici olsaydi"],
}


# two scenarios on screen (a comparison): which is better, how they differ, why one is lower
INTENTS["compare_results"] = {
    "en": ["compare the two results", "which scenario is better", "which one is better a or b", "compare a and b", "how do the two scenarios differ", "what is the difference between a and b",
           "which result should i choose", "why is b lower than a", "why is a better than b", "which option wins", "compare both operations", "which of the two is safer",
           "tell me the difference between the two analyses", "evaluate both results", "rank the two scenarios", "is a or b the better choice", "which one would you pick",
           "how do scenario a and scenario b compare", "give me a side by side summary of both", "which plan is better", "what separates the two results", "judge the two scenarios",
           "put the two scenarios against each other", "why is a lower", "why is b lower", "which of my two scenarios is stronger", "so which should i go with", "how does b compare to a",
           "who wins a or b", "explain the comparison", "interpret the comparison for me", "what do the two results tell me"],
    "tr": ["iki sonucu karsilastir", "hangi senaryo daha iyi", "a mi b mi daha iyi", "a ile b yi karsilastir", "iki senaryo arasindaki fark ne", "hangisini secmeliyim",
           "b neden a dan dusuk", "a neden b den iyi", "hangisi kazaniyor", "ikisini de degerlendir", "iki sonuc arasindaki farklari soyle", "hangisi daha guvenli",
           "iki senaryoyu yan yana ozetle", "karsilastirmayi yorumla", "hangi secenek daha mantikli", "ikisini kiyasla", "birinci mi ikinci mi iyi", "iki analizi degerlendir",
           "b senaryosu a ya gore nasil", "karsilastirma sonucunu aciklar misin", "a neden daha dusuk", "b neden daha dusuk", "benim iki senaryomdan hangisi daha guclu",
           "hangisiyle gitmeliyim", "a ve b arasinda ne fark var", "iki sonuc ne anlatiyor", "karsilastirmayi anlat", "hangisi daha iyi sonuc vermis"],
}


# ---------------------------------------------------------------------------------------------------------------------
# People, not templates: reactions to the assistant itself, and the way real messages are written.
# ---------------------------------------------------------------------------------------------------------------------
INTENTS["affirm"] = {
    "en": ["yes", "yeah", "yep", "sure", "ok", "okay", "alright", "sounds good", "do it", "go for it", "yes do that", "fine by me", "that works", "absolutely", "of course", "please do", "right do that", "yup that one"],
    "tr": ["evet olur", "tamamdir", "peki tamam o zaman", "kesinlikle", "he tamam", "evet istiyorum", "okey", "tabi ki olur", "oldu yap", "evet lutfen", "yap bakalim", "olsun", "aynen katiliyorum", "hadi yap", "mantikli bence", "tamam kabul", "evvet", "heh olur"],
}
INTENTS["deny"] = {
    "en": ["no", "nope", "no way", "not really", "i would rather not", "cancel that", "never mind", "forget it", "dont", "stop", "leave it as is", "no need", "no keep it", "nah", "not that one", "skip it", "i changed my mind"],
    "tr": ["hayir hayir istemem", "yok", "olmaz", "istemem", "bos ver gec", "simdilik gerek yok", "yapma", "iptal et", "hayir tesekkurler", "yok sagol", "vazgec", "bence olmaz", "hayir degistirme", "bu sefer olmasin", "yok kalsin", "bunu istemedim"],
}
INTENTS["confused"] = {
    "en": ["i dont understand", "i do not get it", "what do you mean", "can you simplify that", "make it simpler", "in plain english please", "explain that in simple words", "that went over my head", "huh", "im lost", "too technical for me",
           "can you put that more simply", "come again", "what is that supposed to mean", "break it down for me", "use simpler words", "i have no clue what that says", "less jargon please", "too long short version please"],
    "tr": ["anlayamadim", "bunu anlamadim", "daha basit anlat", "sade bir dille anlatir misin", "yani ne demek istiyorsun", "ne demek bu", "kafam allak bullak oldu", "bir turlu anlamadim", "teknik terimleri birak", "basitlestir",
           "cocuga anlatir gibi anlat", "hic bir sey anlamadim", "acik konus", "kisa ve oz soyle", "kafam durdu", "bu cok karisik geldi", "ne dedin sen simdi", "sadelestir biraz"],
}
INTENTS["doubt"] = {
    "en": ["really", "are you certain", "is that actually true", "how can you be sure", "where does that come from", "prove it", "i dont believe you", "that sounds wrong", "are you making this up", "source", "i doubt that",
           "can i trust that", "show me the proof", "that cant be right", "you sure about it", "that does not sound right to me", "thats suspicious", "who told you that", "are you lying to me", "come on thats not true"],
    "tr": ["gercekten mi", "bundan emin misin", "bunu nereden biliyorsun", "kaynagin ne", "kanitla", "buna inanmiyorum", "yanlis olabilir mi bu", "dogru soyledigine emin degilim", "bana yalan mi soyluyorsun", "bu rakamlar tutuyor mu",
           "sallamiyorsun degil mi", "uydurma bu", "ispat et", "supheliyim", "gercek veri mi bu", "bu hesap bana ters geldi", "kim dedi bunu", "yok artik dogru olamaz", "yalan bu"],
}
INTENTS["complaint"] = {
    "en": ["this is bad", "you are not helping", "that was a terrible answer", "worthless", "this does not work", "so annoying", "you keep getting it wrong", "what a waste of time", "dumb bot", "this is frustrating", "awful", "you suck",
           "totally pointless", "wrong again", "i am disappointed", "you are stupid", "what a joke", "terrible assistant", "i hate this", "stop being useless"],
    "tr": ["bu cok kotu", "hic yardimci olmuyorsun", "bir ise yaramiyorsun", "sacma bir cevap bu", "yine yanlis", "salak asistan", "berbat cevap", "olmadi bu", "sinirimi bozuyorsun", "hep ayni seyi soyluyorsun", "cok yetersizsin",
           "hayal kirikligi", "ise yaramaz", "rezalet", "ugrastirma beni", "aptal bot", "bikdim senden", "cok sacmasin", "zaman kaybi"],
}
INTENTS["repeat"] = {
    "en": ["repeat", "once more", "say it again", "what did you just say", "repeat the last answer", "show that again", "i missed that again please", "can you say that one more time", "run that by me again", "again", "one more time"],
    "tr": ["tekrar", "bir daha yaz", "yeniden soyle", "az once ne dedin", "son cevabi tekrarla", "tekrar goster", "ayni seyi bir daha", "kacirdim bir daha soyle", "cevabi tekrar yazar misin", "geri sar", "bir kere daha"],
}
INTENTS["elaborate"] = {
    "en": ["continue", "keep going", "more details please", "elaborate", "expand on that", "why is that", "what else", "and why", "can you go deeper", "say more", "more info", "carry on", "then what", "what comes next", "explain further"],
    "tr": ["devam", "anlatmaya devam et", "daha fazla bilgi ver", "biraz daha ac", "detaylandir", "neden peki", "nedenini de soyle", "ne olur sonra", "ve sonra", "daha derine in", "bunu genislet", "baska ne var", "peki sonra", "gerisini de anlat"],
}
# praise is a kind of thanks
INTENTS["thanks"]["en"] += ["great job", "nice work", "that was helpful", "brilliant", "love it", "exactly what i needed", "you are the best", "well done", "this is awesome", "good answer"]
INTENTS["thanks"]["tr"] += ["harika is cikardin", "mukemmel cevap", "bravo", "cok yararli oldu", "tam istedigim gibi", "sen bir harikasin", "eline saglik tam olmus", "cok guzel anlatmissin", "helal olsun", "ise yaradi"]

# --- how real messages are written ---
FILLERS = {"en": ["like", "basically", "honestly", "i mean", "you know", "kinda", "just", "um", "so", "actually"],
           "tr": ["yani", "sey", "valla", "bence", "hani", "iste", "ya", "abi", "bi", "falan", "aslinda"]}
RAMBLE = {"en": ["so i was looking at this and", "honestly i dont really get it but", "ok so", "sorry if this is a dumb question but", "i have been staring at this for a while and", "my friend asked me and i could not answer so",
                 "hmm ok", "alright so basically", "wait a second", "i am new to this but", "before i forget"],
          "tr": ["ben aslinda sunu dusunuyordum", "kusura bakma ama", "bi bakiyorum da", "su an ekrana bakiyorum ve", "arkadasim sordu cevaplayamadim", "hmm tamam", "bak simdi", "ya sunu soyle", "bilmiyorum ama",
                 "yani ben boyle bi sey gordum", "bu isleri pek bilmem ama", "unutmadan"]}
TAILS = {"en": ["mate", "bro", "dude", "man", "thanks", "lol", "haha"], "tr": ["abi", "kral", "hocam", "ya", "valla", "kardes", "ahaha", "lol"]}


def _human(s: str, lang: str, rng) -> str:
    """Rambling lead-ins, filler words, stretched letters, words run together, slang tails: messages as people type them."""
    ws = s.split()
    if len(ws) >= 3 and rng.random() < 0.16:
        ws = _pick_one(RAMBLE[lang], rng).split() + ws
    if len(ws) >= 3 and rng.random() < 0.22:
        ws.insert(int(rng.integers(1, len(ws))), _pick_one(FILLERS[lang], rng))
    if rng.random() < 0.12:
        i = int(rng.integers(len(ws)))
        w = ws[i]
        ws[i] = w + w[-1] * int(rng.integers(2, 5))                  # nasilll, heyyyy
    if len(ws) >= 3 and rng.random() < 0.08:
        i = int(rng.integers(len(ws) - 1))
        ws[i:i + 2] = [ws[i] + ws[i + 1]]                              # two words run together
    if rng.random() < 0.15:
        ws.append(_pick_one(TAILS[lang], rng))
    if rng.random() < 0.10:
        ws[-1] += "?" * int(rng.integers(2, 4))
    return " ".join(ws)


def _pick_one(items, rng):
    return items[int(rng.integers(len(items)))]


# ---------------------------------------------------------------------------------------------------------------------
# More ways people ask the same things: everyday, loose wording (added after measuring on the unseen human sets).
# ---------------------------------------------------------------------------------------------------------------------
_BANK = {
    "explain_result": (["whats the bottom line", "can you sum this up for me", "run me through the result quickly", "so how did it turn out", "tl dr of the analysis please",
                        "what does the analysis say overall", "give me the short version of the result", "how does it look overall"],
                       ["sonuc ne olmus kisaca", "ozetini cikarir misin bunun", "genel olarak nasil gorunuyor", "kisaca nedir durum", "bana sonucu bi anlat", "ne cikmis ortaya", "toparlar misin sonucu", "analiz ne diyor"]),
    "why_score": (["how come the score came out so low", "what is hurting my score", "why did it only get that number", "whats behind this low number", "explain the low score to me", "why so bad"],
                  ["puan niye boyle cikti", "bu sayi neden bu kadar kucuk", "skoru ne dusuruyor", "neden bu kadar az almis", "puanin dusuk olmasinin sebebi ne", "niye kotu cikti sonuc"]),
    "improve": (["how do i push the score up", "what would make this score better", "got any ideas to raise it", "how can i get a higher number", "what should i adjust to do better"],
                ["puani yukari cekmek icin ne yapayim", "skoru nasil daha iyi hale getiririm", "ne degistirirsem daha iyi olur", "daha yuksek puan icin fikrin var mi", "bunu duzeltmenin yolu var mi"]),
    "best_date": (["what is the ideal start date", "when would be the smartest time to begin", "is another week better to start", "pick me the best launch day", "should i start earlier or later"],
                  ["en uygun baslangic tarihi ne", "ne zaman baslamak daha akilli olur", "baska bir hafta baslarsak iyi mi olur", "bana en iyi gunu bul", "daha erken mi daha gec mi baslayayim"]),
    "best_vehicle": (["what vehicle fits this place the best", "which of the craft would handle this site", "recommend me a lander for here", "what should we send there"],
                     ["buraya hangi arac uygun", "bu nokta icin hangi araci onerirsin", "en cok hangi arac yakisir buraya", "ne gondermeliyiz oraya", "arac secimi icin tavsiyen ne"]),
    "best_site": (["is there somewhere nicer to touch down", "any better place to land around here", "find me a safer spot", "what other location would score higher"],
                  ["daha uygun bir inis yeri var mi", "etrafta daha guvenli bir nokta bul", "baska hangi konum daha iyi puan alir", "daha mantikli bir yer onerir misin"]),
    "sun_info": (["how bright is it going to be there", "does the sun shine there much", "is it sunny up there", "how many hours of light do we get", "will the panels get light"],
                 ["orada ne kadar isik olur", "gunes sik parliyor mu orada", "panellere yeterli isik gelir mi", "kac saat gunes goruruz", "orasi ne kadar aydinlik"]),
    "comms_info": (["can mission control reach it", "would we have contact with the lander", "is earth visible from that site", "will the signal get through"],
                   ["yer istasyonu ulasabilir mi", "iniscisiyle iletisim olur mu", "dunya o noktadan gorunur mu", "sinyal gider mi oradan"]),
    "slope_info": (["is the terrain rough there", "how tilted is the ground", "would it tip over on that terrain", "is the landing zone flat enough"],
                   ["arazi engebeli mi", "zemin ne kadar egimli", "arac devrilir mi o arazide", "inis alani yeterince duz mu"]),
    "night_thermal": (["how bad are the nights", "would it get too cold to function", "does it need a heater for the dark period", "can the electronics take that cold"],
                      ["geceler ne kadar kotu", "cok mu soguk olur arac calisir mi", "karanlik donemde isitici lazim mi", "elektronikler o soguga dayanir mi"]),
    "battery_info": (["is the power storage enough", "will we run out of juice in the dark", "does the battery hold through the night", "how long can the battery last"],
                     ["enerji depolama yeter mi", "karanlikta sarj biter mi", "batarya geceyi cikarir mi", "pil ne kadar dayanir", "gece boyunca enerji yeter mi"]),
    "data_quality": (["how accurate are all these numbers", "can i rely on this", "whats the quality of the data", "are these figures real or estimates", "how reliable are the inputs"],
                     ["rakamlar ne kadar dogru", "buna guvenebilir miyim", "veri kalitesi nasil", "bu sayilar gercek mi tahmin mi", "girdiler ne kadar saglikli"]),
    "glossary": (["whats that word mean", "define libration", "can you explain what skyline means here", "what is meant by horizon mask", "what is a terminator line"],
                 ["libration nedir", "ne demek terminator", "dte yi acikla", "ufuk maskesi ne anlama geliyor", "terminator cizgisi nedir"]),
    "compare_results": (["so which one is the winner", "a or b which should i go with", "how do the two runs stack up", "which result looks better to you"],
                        ["peki kazanan hangisi", "a ile b den hangisini secmeliyim", "iki calisma nasil kiyaslaniyor", "sence hangi sonuc daha iyi"]),
    "bye": (["ok cool see you tomorrow", "got what i needed bye", "thats enough for today", "catch you later"], ["tamam yarin gorusuruz", "ihtiyacim olan buydu gorusuruz", "bugunluk yeter", "sonra gorusuruz"]),
    "unknown": (["write me a song", "tell me a story about dragons", "write a short essay on football", "who is the president of france", "what should i cook tonight", "give me a workout plan",
                 "recommend a movie", "help me with my homework", "translate hello to spanish", "write python code for sorting", "what is the weather like in london", "how do i lose weight"],
                ["bana bir hikaye anlat", "sarki sozu yaz", "futbol hakkinda kompozisyon yaz", "bu aksam ne yemek yapayim", "bir film oner", "odevime yardim et", "ispanyolcaya cevir",
                 "python kodu yaz", "borsa ne olacak", "fransa baskani kim", "kilo nasil verilir", "bana espri yap ve bir siir yaz"]),
    "complaint": (["man this is lame", "this bot is trash", "pathetic", "you never get anything right", "what a useless thing"], ["sen ne bicim asistansin", "hic faydan yok", "cok rezil bir sey bu", "hep yanlis anliyorsun", "ne ise yarar bu"]),
    "elaborate": (["keep talking", "go ahead and continue", "more", "i want to hear more about it"], ["anlatmaya devam", "biraz daha anlat", "dahasi var mi", "devam etsene"]),
    "affirm": (["mhm", "yep that is fine", "cool do that one", "yes go ahead"], ["hadi bakalim", "aynen kardes", "olur yapalim", "tamam yap gitsin"]),
}
for _intent, (_en, _tr) in _BANK.items():
    INTENTS[_intent]["en"] += _en
    INTENTS[_intent]["tr"] += _tr


# free chat: a date or a place and "what should we send", with no analysis on screen
_FREE_VEHICLE = {
    "en": ["what should we send on {D}", "which vehicle should we send on {D}", "what is the best vehicle for {D}", "which lander would you launch on {D}", "if we start on {D} which spacecraft is best",
           "what do we send there on {D}", "which craft should we pick for {D}", "best spacecraft for a start on {D}", "what should we land on {D}", "which mission suits {D} best", "what should we send"],
    "tr": ["{D} de ne gonderelim", "{D} de hangi araci gondermeliyiz", "{D} icin en iyi arac hangisi", "{D} de hangi inis aracini firlatalim", "{D} baslarsak hangi arac iyi", "{D} icin ne gondermeliyiz",
           "ne gonderelim", "hangi araci gondermeliyiz", "{D} tarihinde hangi araci secelim", "{D} de hangi araci yollayalim", "oraya ne gonderelim", "hangi araci yollasak iyi olur"],
}
_DATES = ["12 september", "march 3", "1 january", "december 5", "july 20", "12 eylul", "3 mart", "1 ocak", "5 aralik", "20 temmuz", "15 haziran", "october 9"]
for _lang, _items in _FREE_VEHICLE.items():
    for _t_ in _items:
        INTENTS["best_vehicle"][_lang] += [_t_.replace("{D}", _d) for _d in _DATES] if "{D}" in _t_ else [_t_]
