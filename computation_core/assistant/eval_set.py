"""Hand-written test sentences, deliberately worded differently from nlu_data.py's templates (informal, typo-prone,
Turkish / English / mixed). The intent model is never trained on these; its accuracy here is the honest number."""

EVAL = [
    # greet
    ("selam kral nasil gidiyor isler", "greet"), ("hello there assistant", "greet"), ("gunaydin hocam", "greet"), ("hey!", "greet"), ("merhabalar", "greet"),
    # thanks
    ("tamam super cok tesekkur", "thanks"), ("thanks mate that was useful", "thanks"), ("sagolasin kral", "thanks"), ("appreciate it", "thanks"),
    # bye
    ("ben cikiyorum gorusuruz", "bye"), ("ok bye for now", "bye"), ("iyi calismalar hadi bb", "bye"),
    # about
    ("bu asistan ne is yapiyor", "about"), ("what exactly are you able to do", "about"), ("sen hazir bir model misin yoksa kendin mi yaptik", "about"), ("what can i ask here", "about"),
    # smalltalk
    ("bugun nasilsin bakalim", "smalltalk"), ("how is your day going", "smalltalk"), ("sikildin mi benden", "smalltalk"), ("do you ever get bored", "smalltalk"),
    # explain_result
    ("su ekrandaki sonucu bir anlatsana", "explain_result"), ("can you give me the gist of this analysis", "explain_result"), ("bu sayilar ne anlama geliyor ya", "explain_result"),
    ("explain what this page is saying in plain words", "explain_result"), ("ozet gec bana", "explain_result"), ("i do not understand this result help me", "explain_result"),
    # why_score
    ("ya bu puan niye bu kadar dusuk cikti", "why_score"), ("why is it only 15 out of 100", "why_score"), ("puani yerlere ceken ne", "why_score"),
    ("what exactly killed the score here", "why_score"), ("neden bu kadar berbat bir sonuc verdi", "why_score"), ("explain why the result is poor", "why_score"),
    ("what is pulling this score down", "why_score"),
    # improve
    ("bunu duzeltmek icin elimizde ne var", "improve"), ("how could this scenario get better", "improve"), ("puani yukari cekmek icin hangi oyunlari yapabiliriz", "improve"),
    ("what levers do i have to improve this", "improve"), ("daha iyi sonuc almanin yolu var mi", "improve"),
    # best_date
    ("tarihi nasil oynatirsak daha iyi bir berim alabiliriz", "best_date"), ("which date would give a better result under the same conditions", "best_date"),
    ("tarihi kaydirsak puan duzelir mi", "best_date"), ("if i shift the start date by a few days will it get better", "best_date"),
    ("en uygun baslangic gunu hangisi", "best_date"), ("scan the calendar and tell me when it is best", "best_date"), ("hangi tarihte inersek daha mantikli olur", "best_date"),
    ("is there a better day to launch this", "best_date"), ("ayni sartlarda baska tarih daha iyi mi", "best_date"),
    # compare_dates
    ("12 aralik yerine 3 ocak olsa ne degisir", "compare_dates"), ("how would the score look if we started on january 3", "compare_dates"),
    ("iki tarihi yan yana koy", "compare_dates"), ("what if it were next month instead", "compare_dates"), ("su tarihte nasil olurdu", "compare_dates"),
    # best_vehicle
    ("buraya hangi arac daha cok yakisir", "best_vehicle"), ("would a different lander handle this spot better", "best_vehicle"), ("tum araclari bu noktada yaristir", "best_vehicle"),
    ("which of the vehicles scores highest at this location", "best_vehicle"), ("baska bir arac secsek mi", "best_vehicle"),
    # best_site
    ("yakinda daha mantikli bir nokta var mi", "best_site"), ("is there any better landing spot around here", "best_site"), ("bu aracla baska nerelere inebiliriz", "best_site"),
    ("compare shackleton haworth and malapert for me", "best_site"), ("konumu azicik kaydirsak olur mu", "best_site"),
    # sun_info
    ("bu bolgede gunes isigi nasil", "sun_info"), ("how many days of sun do we get", "sun_info"), ("gunes ne kadar sure bizi aydinlatiyor", "sun_info"),
    ("how bright is it up there", "sun_info"), ("karanlik donemler ne kadar uzun surer", "sun_info"),
    # comms_info
    ("yer istasyonuyla konusabilecek miyiz", "comms_info"), ("is there a communication blackout", "comms_info"), ("dunya ufukta gorunuyor mu", "comms_info"),
    ("will it lose its radio link to earth", "comms_info"), ("iletisim ne zaman kopuyor", "comms_info"),
    # slope_info
    ("inis alani cok mu egimli", "slope_info"), ("is the terrain flat enough to land on", "slope_info"), ("zeminin dikligi tehlikeli mi", "slope_info"), ("how risky is the slope here", "slope_info"),
    # night_thermal
    ("gece soguguna dayanabilir mi bu arac", "night_thermal"), ("will the lander freeze during the dark period", "night_thermal"), ("karanlikta donup olur mu", "night_thermal"),
    ("does it carry any heat source for the night", "night_thermal"),
    # battery_info
    ("pili buyutursek hayatta kalir mi", "battery_info"), ("what if i add more battery capacity", "battery_info"), ("bataryasi gece boyu yeter mi", "battery_info"),
    ("will the power run out in the dark", "battery_info"),
    # data_quality
    ("bu degerlere ne kadar guvenmeliyim", "data_quality"), ("are these numbers measured or guessed", "data_quality"), ("hangi bilgiler dogrulanmamis", "data_quality"),
    ("how trustworthy is this analysis", "data_quality"),
    # glossary
    ("held down by ne anlama geliyor", "glossary"), ("what is a landing slope limit", "glossary"), ("dte kisaltmasi neyin nesi", "glossary"), ("what does skyline mean in this context", "glossary"),
    ("veto mekanizmasi nedir", "glossary"),
    # table_request
    ("bunu bir tabloyla goster", "table_request"), ("can you lay that out in a table", "table_request"), ("tabloya dok bakalim", "table_request"),
    # apply
    ("tamam o tarihe gec", "apply"), ("yes go ahead and use that", "apply"), ("olur uygula bunu", "apply"), ("great apply the best date", "apply"),
    # change_context
    ("araci chang'e 3 yap", "change_context"), ("make the date 20 december", "change_context"), ("konumu malapert olarak degistir", "change_context"),
    ("switch the vehicle to im-1", "change_context"), ("tarihi 15 subat olarak ayarla", "change_context"),
    # unknown
    ("bugun hava kac derece", "unknown"), ("recommend a good restaurant nearby", "unknown"), ("bana bir hikaye anlat", "unknown"), ("what is the stock price of tesla", "unknown"),
    ("ankarada trafik var mi", "unknown"), ("how do i bake bread", "unknown"),
]


# A second, untouched test set (written after the model was tuned on EVAL above): the number to quote.
FRESH = [
    ("selam, bir sorum olacak", "greet"), ("hello, can you help me", "greet"), ("naber hocam", "greet"), ("good evening", "greet"),
    ("super oldu tesekkurler", "thanks"), ("ok great thank you so much", "thanks"), ("eyv kral", "thanks"),
    ("simdilik bu kadar iyi geceler", "bye"), ("alright see you tomorrow", "bye"),
    ("sen tam olarak neyi biliyorsun", "about"), ("how do you work", "about"), ("neye cevap verebilirsin", "about"), ("are you using chatgpt", "about"),
    ("hava nasil bakalim orada", "smalltalk"), ("you seem like a nice assistant", "smalltalk"), ("bugun keyfin yerinde mi", "smalltalk"),
    ("elimdeki sonucu kisaca anlatir misin", "explain_result"), ("walk me through what i am seeing on this page", "explain_result"), ("bu analiz ne soyluyor", "explain_result"), ("sum up the findings", "explain_result"),
    ("puan niye 15 cikti", "why_score"), ("what is the reason it scored so badly", "why_score"), ("puani asagi ceken faktor hangisi", "why_score"), ("why is this result so weak", "why_score"),
    ("daha yuksek puan almak icin ne tavsiye edersin", "improve"), ("how can i make this scenario work better", "improve"), ("bunu duzeltmenin bir yolu var mi", "improve"),
    ("baska bir gun secsek skor artar mi", "best_date"), ("what is the ideal day to start", "best_date"), ("tarih olarak en verimli donem hangisi", "best_date"), ("should we delay the landing by some days", "best_date"),
    ("20 aralik olursa nasil olur", "compare_dates"), ("how about starting on february 1", "compare_dates"), ("iki hafta sonrasina bakalim", "compare_dates"), ("evaluate the same thing for 3 january", "compare_dates"),
    ("hangi arac bu bolgede en iyi sonuc veriyor", "best_vehicle"), ("pick the strongest lander for this place", "best_vehicle"), ("araclari karsilastirsana", "best_vehicle"),
    ("bu yere yakin daha uygun bir bolge var mi", "best_site"), ("any safer spot close to this one", "best_site"), ("farkli bir inis noktasi onerir misin", "best_site"),
    ("ne kadar gun isigi aliyoruz", "sun_info"), ("how much of the time is the sun up", "sun_info"), ("gunes isigi yeterli mi bu bolgede", "sun_info"),
    ("dunya ile aramizda gorus hatti var mi", "comms_info"), ("how often can we talk to mission control", "comms_info"), ("telsiz baglantisi kesiliyor mu", "comms_info"),
    ("buradaki zemin ne kadar egimli", "slope_info"), ("would the lander be stable on this terrain", "slope_info"), ("inis alani yeterince duz mu acaba", "slope_info"),
    ("ay gecesinde arac isinabilecek mi", "night_thermal"), ("is there a risk of freezing", "night_thermal"), ("karanlik donemde soguktan bozulur mu", "night_thermal"),
    ("pil kapasitesi yeterli mi", "battery_info"), ("what happens with a larger battery", "battery_info"), ("batarya kac saat idare eder", "battery_info"),
    ("verilere guvenilir mi diye merak ediyorum", "data_quality"), ("which inputs are only assumptions", "data_quality"), ("bu degerler nereden geliyor", "data_quality"),
    ("lunar night ne demek", "glossary"), ("explain what the suitability index is", "glossary"), ("horizon kavrami nedir", "glossary"), ("what is a relay satellite", "glossary"),
    ("sonuclari tablo seklinde sun", "table_request"), ("give me that as a table please", "table_request"),
    ("olur o oneriyi kullan", "apply"), ("sure go with it", "apply"), ("peki uygula bakalim", "apply"),
    ("tarihi 3 mart yap", "change_context"), ("use blue ghost instead", "change_context"), ("konumu haworth olarak ayarla", "change_context"), ("move the start date forward by ten days", "change_context"),
    ("dunyanin en yuksek dagi hangisi", "unknown"), ("write python code for me", "unknown"), ("yarin yagmur yagacak mi", "unknown"), ("who is the best football player", "unknown"),
]


# Third set, written once after all tuning and never used to change the data or the model: THE number to quote.
# tests/test_assistant.py checks that none of these sentences occurs in the training corpus.
FINAL = [
    ("mrb nasilsiniz", "greet"), ("hey, anybody home", "greet"), ("iyi gunler dilerim", "greet"), ("hello hello", "greet"),
    ("cok yardimci oldunuz minnettarim", "thanks"), ("ok perfect that clears it up thanks", "thanks"), ("tesekkur ederim kral", "thanks"),
    ("bu kadar yeter cikiyorum", "bye"), ("later alligator", "bye"), ("hadi kolay gelsin gorusmek uzere", "bye"),
    ("seni ne icin gelistirdik", "about"), ("what can you tell me about the moon landing analysis tool", "about"), ("hangi konularda uzmansin", "about"),
    ("how are things on your side", "smalltalk"), ("canin sikildi mi", "smalltalk"), ("what is your favourite colour", "smalltalk"),
    ("su an ekranda ne var bir bakalim", "explain_result"), ("break down this outcome for a beginner", "explain_result"), ("ozetin ozeti nedir", "explain_result"), ("what did the engine conclude", "explain_result"),
    ("neden bu kadar az aldik", "why_score"), ("what's dragging the grade down so much", "why_score"), ("puandaki en buyuk kayip nerede", "why_score"), ("why did we get such a modest score", "why_score"),
    ("skoru yukselten bir degisiklik var mi", "improve"), ("any quick wins to raise the number", "improve"), ("bu senaryoyu guclendirmek icin neler onerirsin", "improve"),
    ("hangi tarihte baslarsak sonuc yukselir", "best_date"), ("is there a sweet spot in the calendar", "best_date"), ("takvimde en iyi gun hangisi", "best_date"), ("which week should we aim for", "best_date"),
    ("15 ocakta baslasak nasil sonuc verir", "compare_dates"), ("let us see how april 2 compares", "compare_dates"), ("bir hafta ilerisi icin hesapla", "compare_dates"), ("rerun it for 12/01/2027", "compare_dates"),
    ("burada hangi lander daha basarili", "best_vehicle"), ("rank the landers for this spot", "best_vehicle"), ("baska bir uzay araciyla daha iyi olur mu", "best_vehicle"),
    ("daha dusuk riskli bir nokta var mi civarda", "best_site"), ("could we touch down somewhere nearby with better numbers", "best_site"), ("inis noktasini baska yere almak mantikli mi", "best_site"),
    ("gunduz suresi ne kadar", "sun_info"), ("what is the share of sunlight in this period", "sun_info"), ("ne kadarlik zaman karanlikta kaliyoruz", "sun_info"),
    ("yer ile telsiz temasi saglanabilir mi", "comms_info"), ("will mission control hear from the lander", "comms_info"), ("dunyayi ne kadar gorebiliyoruz", "comms_info"),
    ("meyil arac icin sorun olusturur mu", "slope_info"), ("is the touchdown area level enough to be safe", "slope_info"), ("zemin kaymak kadar dik mi", "slope_info"),
    ("geceleri sogukta nasil kalacak", "night_thermal"), ("does it need a heater to last through the dark", "night_thermal"), ("gece donma tehlikesi var mi", "night_thermal"),
    ("pil gece icin yeter mi", "battery_info"), ("how much battery would it take", "battery_info"), ("bataryayi 3 kat yapsak ne degisir", "battery_info"),
    ("bu rakamlar ne kadar saglam", "data_quality"), ("what part of the vehicle data is missing", "data_quality"), ("sonuca guvenebilir miyim", "data_quality"),
    ("direct to earth kavrami ne", "glossary"), ("what does a veto mean here", "glossary"), ("illumination derken neyi kastediyorsun", "glossary"),
    ("bunu tablo olarak ver", "table_request"), ("put these results into a table", "table_request"),
    ("evet bunu uygula", "apply"), ("fine apply that", "apply"), ("tamam o zaman onu sec", "apply"),
    ("tarihi 9 nisan olarak degistir", "change_context"), ("swap the lander for luna 25", "change_context"), ("konumu faustini yap", "change_context"), ("set the start day to the first of march", "change_context"),
    ("bana bir sarki soyle", "unknown"), ("how do i get to the airport", "unknown"), ("bu hafta sonu maci kim kazandi", "unknown"), ("calculate 17 times 23", "unknown"),
]


# Questions of the form "what can I do to increase / reduce <something>?" (intent 'improve', with the topic the user means).
# Written BEFORE the topic head and its grammar were built, so they measure generalisation; never used for training.
TOPIC_TEST = [
    ("güneşin ufuktan yüksekliğini nasıl yükseltirim", "sun_angle"), ("how can I get the sun higher in the sky", "sun_angle"),
    ("what would raise the sun elevation here", "sun_angle"), ("güneş daha dik gelsin istiyorum ne yapmalıyım", "sun_angle"), ("isinlarin gelis acisini iyilestirmenin yolu var mi", "sun_angle"),
    ("güneş açısı çok düşük bunu nasıl artırırız", "sun_angle"),
    ("daha fazla güneş ışığı almak için ne yapmalıyım", "sun_light"), ("how do I get more sunlight on this lander", "sun_light"), ("aydınlık süresi nasıl uzar", "sun_light"),
    ("what would increase the share of time in the sun", "sun_light"), ("güneşli zamanı arttırmak için neyi değiştireyim", "sun_light"),
    ("dünya ile bağlantıyı artırmak için ne yapabilirim", "earth"), ("how can I improve the earth link", "earth"), ("iletişim kesintilerini nasıl azaltırım", "earth"),
    ("what would give us more time in view of earth", "earth"),
    ("eğimi azaltmak için ne yapmalı", "slope"), ("how can I find flatter ground", "slope"), ("daha düz bir yer nasıl bulurum", "slope"), ("what do I change to get a gentler slope", "slope"),
    ("geceyi nasıl kısaltırım", "night"), ("how do I reduce the darkness period", "night"), ("what can shorten the longest night", "night"),
    ("pili nasıl yeterli hale getiririm", "battery"), ("how do we make the battery last through the dark", "battery"), ("batarya sorununu çözmek için ne yapılabilir", "battery"),
    ("puanı artırmak için ne yapayım", "score"), ("what should I do to raise the overall score", "score"), ("sonucu iyileştirmek için neler yapabilirim", "score"),
]


# Fourth set (FINAL2), written before the off-topic negatives and the second retraining; used once for the quoted number.
FINAL2 = [
    ("gunes isigi bu kadar azken arac calisir mi", "sun_info"), ("what is the lowest the sun gets here", "sun_info"), ("isik acisi yeterli mi", "sun_info"),
    ("dunyayi gorebildigimiz saatler yeterli mi", "comms_info"), ("is there a long stretch without radio contact", "comms_info"),
    ("alan ne kadar yatik", "slope_info"), ("could the lander slide down this hill", "slope_info"),
    ("ay gecesi bu arac icin cok mu uzun", "night_thermal"), ("how cold will it be during the long dark", "night_thermal"),
    ("pil gece boyunca idare eder mi", "battery_info"), ("is the power enough for the whole period", "battery_info"),
    ("ne yapsam daha iyi sonuc cikar", "improve"), ("what could i tweak to get a better outcome", "improve"), ("hangi degisiklik sonucu yukseltir", "improve"),
    ("ne yaparsam geceler kisalir", "improve"), ("what can i do to get a flatter landing zone", "improve"), ("gunes ufuk acisini yukseltmenin yolu nedir", "improve"),
    ("how do i raise the share of daylight", "improve"), ("dunya ile temas suresini nasil uzatirim", "improve"),
    ("baska hangi tarihleri denemeliyim", "best_date"), ("which days look the most promising", "best_date"),
    ("15 martta nasil olur", "compare_dates"), ("how would it look a month from now", "compare_dates"),
    ("hangi arac burada one cikiyor", "best_vehicle"), ("any better candidate site than this one", "best_site"),
    ("bu sayilari bana anlat", "explain_result"), ("tell me what the page says in simple words", "explain_result"),
    ("skor niye 40 civarinda", "why_score"), ("what is the main reason behind this number", "why_score"),
    ("kisaca nedir bu veto", "glossary"), ("what does skyline mean on this page", "glossary"),
    ("bunu tabloda gostersene", "table_request"), ("sure apply it", "apply"), ("tamam kabul uygula", "apply"),
    ("tarihi 4 mayis yap", "change_context"), ("use slim instead", "change_context"),
    ("selam nasilsin", "greet"), ("hi there", "greet"), ("tesekkurler yardimin icin", "thanks"), ("see you", "bye"),
    ("sen ne is yaparsin", "about"), ("how do you know all this", "about"),
    ("how do i get a better paying job", "unknown"), ("ne yaparsam kilo veririm", "unknown"), ("how can i improve my english", "unknown"),
    ("what is the fastest way to learn guitar", "unknown"), ("ingilizcemi nasil gelistirebilirim", "unknown"), ("bu cumleyi turkceye cevir", "unknown"),
    ("best pizza recipe", "unknown"), ("arabamin lastigi patladi", "unknown"), ("what is the tallest building in the world", "unknown"),
]


# Fifth set (FINAL3): written before the 'try_option' intent and the slot-aware answers were trained; the number to quote now.
FINAL3 = [
    ("peki ya apollo 17 olsaydi", "try_option"), ("what about using blue ghost instead", "try_option"), ("ya haworth cevresine inseydik", "try_option"), ("how would slim fare at this spot", "try_option"),
    ("isitici koysak sonuc degisir mi", "try_option"), ("and if we tried malapert", "try_option"), ("chandrayaan 3 yerine luna 25 ile bakalim", "try_option"),
    ("araci chandrayaan 3 yap", "change_context"), ("konumu shackleton olarak guncelle", "change_context"), ("start date should be january 20", "change_context"),
    ("gunes bu noktada ne kadar yuksege cikiyor", "sun_info"), ("bu bolgede kac saat gunduz var", "sun_info"),
    ("temas kopuklugu ne kadar surer", "comms_info"), ("do we lose the earth link for long", "comms_info"),
    ("yer cok mu egimli buranin", "slope_info"), ("gece boyunca arac donar mi", "night_thermal"),
    ("batarya buyurse ne olur", "battery_info"), ("bu analizde hangi veriler tahmin", "data_quality"),
    ("ne yapsam sonuc daha iyi olur", "improve"), ("how could we make this scenario score higher", "improve"), ("ne yaparsam dunyayi daha cok gorurum", "improve"),
    ("baska bir gunde baslasak puan artar mi", "best_date"), ("what start date would you recommend", "best_date"),
    ("1 subatta baslarsak ne olur", "compare_dates"), ("what happens if the date is march 3", "compare_dates"),
    ("burada hangi arac daha dayanikli", "best_vehicle"), ("near here, is there a better spot", "best_site"),
    ("bunu ozetler misin", "explain_result"), ("explain this to me simply", "explain_result"), ("puan neden bu seviyede", "why_score"), ("why is this score what it is", "why_score"),
    ("dte ne demek", "glossary"), ("what is a heat source in this context", "glossary"), ("tablo istiyorum", "table_request"), ("ok go for it", "apply"),
    ("merhaba kral", "greet"), ("cok sagol", "bye"), ("sen nasil bir asistansin", "about"), ("keyfin yerinde mi", "smalltalk"), ("thank you very much", "thanks"),
    ("bana bir film oner", "unknown"), ("what is the weather in paris", "unknown"), ("ne yaparsam daha iyi basketbol oynarim", "unknown"), ("how do i get stronger", "unknown"),
]


# Sixth set (FINAL4): written before the off-topic verb phrases, the agreement guard and the extra command words were added.
FINAL4 = [
    ("how do i become a better swimmer", "unknown"), ("ne yaparsam almanca ogrenirim", "unknown"), ("what can i do to sleep deeper", "unknown"), ("daha cok para kazanmak icin ne yapmaliyim", "unknown"),
    ("how can i win at chess", "unknown"), ("ne yapsam ayakkabim kokmasin", "unknown"), ("which laptop should i get", "unknown"), ("sinav icin nasil calisayim", "unknown"),
    ("tarihi 8 subata guncelle", "change_context"), ("update the vehicle to chang'e 4", "change_context"), ("baslangic tarihi 2 mart olsun", "change_context"), ("let the site be faustini", "change_context"),
    ("peki ya beresheet olsaydi", "try_option"), ("what if we tried im-2", "try_option"), ("isitici olursa puan artar mi", "try_option"), ("haworth icin de bakalim", "try_option"),
    ("gunduz ne kadar surer burada", "sun_info"), ("kac gun karanlik kaliyoruz", "sun_info"), ("will we be in contact with earth most of the time", "comms_info"),
    ("tehlikeli bir egim var mi", "slope_info"), ("is the cold a danger for the lander", "night_thermal"), ("batarya yeterli mi acaba", "battery_info"),
    ("puani yukseltmek icin hangi adimlar", "improve"), ("what would you suggest to get a higher number", "improve"), ("dunya gorunurlugu icin ne yapsak", "improve"),
    ("hangi baslangic gunu daha uygun", "best_date"), ("is a different day better", "best_date"), ("22 ocakta olsa nasil", "compare_dates"),
    ("hangi uzay araci buraya uygun", "best_vehicle"), ("anything better close by", "best_site"),
    ("sonuc ne anlama geliyor", "explain_result"), ("why did we get this grade", "why_score"), ("veto nedir tam olarak", "glossary"),
    ("tamam olur uygula", "apply"), ("sonucu tabloya dok", "table_request"), ("merhabalar nasilsiniz", "greet"), ("iyi geceler", "bye"), ("cok tesekkur ederim", "thanks"),
]


# Seventh set (FINAL5): written before the data-derived domain-vocabulary guard was designed; the number to quote.
FINAL5 = [
    ("how do i get better at skiing", "unknown"), ("ne yaparsam daha cok kitap okurum", "unknown"), ("which camera should i buy for travelling", "unknown"), ("sinavdan once nasil rahatlarim", "unknown"),
    ("can you recommend a good podcast", "unknown"), ("ne yapsam sacim uzar", "unknown"), ("what is the best way to cook rice", "unknown"), ("bu hafta hava nasil olacak", "unknown"), ("who wrote hamlet", "unknown"),
    ("how do i raise a puppy", "unknown"), ("kedimi nasil egitirim", "unknown"),
    ("bu senaryo icin puan kac", "explain_result"), ("break the score down for me", "why_score"), ("neden bu kadar dusuk cikmis", "why_score"),
    ("hangi gunlerde daha iyi", "best_date"), ("could we pick a smarter start date", "best_date"), ("2 nisan icin hesapla", "compare_dates"), ("what would the result be on february 8", "compare_dates"),
    ("baska bir arac secsem daha mi iyi olur", "best_vehicle"), ("is there a safer point around here", "best_site"),
    ("gunes bu bolgede kac saat gorunuyor", "sun_info"), ("how strong is the sunlight here", "sun_info"), ("dunya ile haberlesme yeterli mi", "comms_info"), ("is the lander at risk of tipping", "slope_info"),
    ("elektronikler gece soguga dayanir mi", "night_thermal"), ("how long will the battery last", "battery_info"), ("hangi veriler eksik", "data_quality"),
    ("ne yaparsam gece kisalir", "improve"), ("what can i do to get a higher score", "improve"), ("slope nedir", "glossary"),
    ("peki ya luna 25 ile", "try_option"), ("what about shackleton", "try_option"), ("tarihi 3 mart yap", "change_context"), ("use apollo 14 instead", "change_context"),
    ("tamam uygula bakalim", "apply"), ("goster tablo olarak", "table_request"), ("selam", "greet"), ("hoscakal", "bye"), ("eyvallah kral", "thanks"), ("bu nasil calisiyor", "about"),
]


# Comparison set: written after the 'compare_results' intent existed, in words that are not in its training sentences.
COMPARE_TEST = [
    ("hangisi daha avantajli", "compare_results"), ("which of these two would you recommend", "compare_results"), ("a nin b den farki ne", "compare_results"),
    ("is the second scenario any better", "compare_results"), ("iki secenegi kiyaslayip hangisi iyi soyle", "compare_results"), ("give me a verdict on the two options", "compare_results"),
    ("how does b stack up against a", "compare_results"), ("ikisinden hangisini onerirsin", "compare_results"), ("tell me which result is stronger", "compare_results"),
    ("hangi senaryo daha basarili", "compare_results"), ("compare my two operations", "compare_results"), ("iki operasyonu karsilastirir misin", "compare_results"),
    ("so is a worth more than b", "compare_results"), ("sonuclardan hangisi daha saglam", "compare_results"), ("between the two which one wins", "compare_results"),
    ("a ve b nin farklarini anlat", "compare_results"), ("what is better, the first or the second", "compare_results"), ("birinci senaryo mu ikinci mi", "compare_results"),
    # things that must NOT be read as a comparison of the two results
    ("why is the score low", "why_score"), ("which vehicle works best here", "best_vehicle"), ("how can we shift the date", "best_date"), ("what about im-1", "try_option"),
    ("tarihleri karsilastir 1 ocak 1 subat", "compare_dates"), ("summarize the result", "explain_result"), ("sonucu ozetle", "explain_result"), ("veriler guvenilir mi", "data_quality"),
]


# How people really write: slang, typos, fillers, long rambling, and reactions to the assistant itself (doubt, confusion, anger, "ok", "no").
# Written by hand after the fact, in words that are not in the training sentences; test_assistant checks that none leaked.
HUMAN_TEST = [
    # -- the analysis, said the way people say it
    ("abi sonuc nasil cikmis ozetler misin", "explain_result"), ("su analizin ozeti ne ya", "explain_result"), ("sonuclar ne diyo kisaca", "explain_result"),
    ("bi ozet gecsene", "explain_result"), ("what did we end up with here", "explain_result"), ("give me the gist of this", "explain_result"), ("so where do we stand", "explain_result"),
    ("puan niye bu kadar kotu ya", "why_score"), ("neden 30 cikti anlamadim", "why_score"), ("why on earth is it so low", "why_score"), ("whats dragging the score down", "why_score"),
    ("bu skor niye dusuk hocam", "why_score"), ("hmm why only that score", "why_score"),
    ("nasil daha iyi yapabilirim bunu", "improve"), ("puani yukseltmek icin napmam lazim", "improve"), ("any way to bump the score up", "improve"), ("what can i tweak to make it better", "improve"),
    ("iyilestirmek icin ne onerirsin kral", "improve"),
    ("baska bi tarihte baslasak daha mi iyi olur", "best_date"), ("hangi gun baslamak en mantikli", "best_date"), ("when should we launch really", "best_date"), ("which day is the sweet spot", "best_date"),
    ("tarihi biraz kaydirsam duzelir mi", "best_date"),
    ("bu yere hangi arac yakisir", "best_vehicle"), ("en uygun arac hangisi bence", "best_vehicle"), ("whats the best rover for this spot", "best_vehicle"), ("which craft suits this place best", "best_vehicle"),
    ("daha iyi bi yer var mi acaba", "best_site"), ("baska nereye insek iyi olur", "best_site"), ("is there a nicer landing spot nearby", "best_site"), ("where else would be better to land", "best_site"),
    ("orada gunes var mi ya", "sun_info"), ("gunes ne kadar duruyo", "sun_info"), ("how much sun does it get", "sun_info"), ("is the sun up there most of the time", "sun_info"),
    ("dunya gorunuyor mu oradan", "comms_info"), ("iletisim kurabilir miyiz", "comms_info"), ("can we even talk to earth from there", "comms_info"), ("will we lose radio contact", "comms_info"),
    ("arazi cok mu dik", "slope_info"), ("egim sorun olur mu", "slope_info"), ("is the ground too steep", "slope_info"), ("can it even land on that slope", "slope_info"),
    ("gece donar mi arac", "night_thermal"), ("soguk yuzunden bozulur mu", "night_thermal"), ("will it freeze at night", "night_thermal"), ("how cold does it get", "night_thermal"),
    ("pil yeter mi", "battery_info"), ("batarya biter mi gece", "battery_info"), ("will the battery last", "battery_info"), ("is the battery big enough", "battery_info"),
    ("bu verilere guvenebilir miyim", "data_quality"), ("rakamlar ne kadar saglam", "data_quality"), ("how much can i trust these numbers", "data_quality"), ("is this data legit", "data_quality"),
    ("dte ne demek", "glossary"), ("what does libration mean", "glossary"), ("terminator nedir ya", "glossary"), ("ne anlama geliyor skyline", "glossary"),
    ("tarihi 20 aralik yap", "change_context"), ("araci apollo 15 yapsana", "change_context"), ("switch the place to shackleton", "change_context"), ("can you set the date to march 3", "change_context"),
    ("bunu tablo yapsana", "table_request"), ("put that in a table", "table_request"), ("tablo halinde goster", "table_request"),
    ("ya apollo 17 olsaydi", "try_option"), ("what if it were chandrayaan", "try_option"),
    ("1 ocak mi 1 subat mi daha iyi", "compare_dates"), ("is march better than february", "compare_dates"),
    ("a mi b mi abi hangisi", "compare_results"), ("which one wins between my two", "compare_results"),
    # -- small talk and the edges
    ("selamlar kral", "greet"), ("heyyy", "greet"), ("naber hocam", "greet"), ("yo whats up assistant", "greet"),
    ("super oldu eyvallah", "thanks"), ("ahaha tesekkurler cok iyi", "thanks"), ("perfect thx mate", "thanks"), ("valla cok iyi oldu sag ol", "thanks"),
    ("ben kaciyorum gorusuruz", "bye"), ("alright im out", "bye"), ("sonra bakarim simdilik hosca kal", "bye"),
    ("sen tam olarak ne is yapiyorsun", "about"), ("who made you and how do you work", "about"),
    ("canin sikildi mi", "smalltalk"), ("do you ever get bored", "smalltalk"),
    ("bana bir pizza tarifi ver", "unknown"), ("what is the capital of france", "unknown"), ("yarin hava nasil olacak", "unknown"), ("write me a poem about autumn", "unknown"),
    # -- reactions to the assistant
    ("evet", "affirm"), ("tamam", "affirm"), ("olur", "affirm"), ("tabii ki", "affirm"), ("aynen oyle", "affirm"), ("okay go ahead", "affirm"), ("yeah sure", "affirm"), ("yes please", "affirm"),
    ("hayir", "deny"), ("yok bos ver", "deny"), ("istemiyorum", "deny"), ("no thanks", "deny"), ("nope not now", "deny"), ("vazgectim", "deny"), ("dont do that", "deny"), ("gerek yok", "deny"),
    ("anlamadim ya", "confused"), ("bi daha anlat daha basit", "confused"), ("what does that even mean", "confused"), ("can you explain it like im five", "confused"), ("kafam karisti", "confused"),
    ("bu ne demek simdi", "confused"), ("didnt get that", "confused"), ("daha sade anlatsana", "confused"),
    ("emin misin bundan", "doubt"), ("bu sonuc dogru mu yoksa uyduruyor musun", "doubt"), ("are you sure about that", "doubt"), ("hmm i dont buy it", "doubt"), ("nerden cikardin bu rakami", "doubt"),
    ("sacma geldi bu", "doubt"), ("how do you know that", "doubt"), ("yalan soyluyorsun", "doubt"),
    ("olmuyor bu ya", "complaint"), ("cok kotusun", "complaint"), ("you are useless", "complaint"), ("this is garbage", "complaint"), ("hic ise yaramiyorsun", "complaint"),
    ("ugh not helpful at all", "complaint"), ("yine sacmaladin", "complaint"), ("berbat bir asistan", "complaint"),
    ("bir daha soyler misin", "repeat"), ("say that again", "repeat"), ("ne demistin", "repeat"), ("tekrar eder misin", "repeat"), ("can you repeat that", "repeat"), ("bi daha goster", "repeat"),
    ("devam et", "elaborate"), ("biraz daha detay ver", "elaborate"), ("tell me more", "elaborate"), ("go on", "elaborate"), ("peki neden", "elaborate"), ("ayrintili anlat", "elaborate"),
    ("and then", "elaborate"), ("daha fazla anlat", "elaborate"),
    # -- long, rambling, the way people actually type it
    ("abi ben aslinda sunu merak ediyordum acaba baska bi tarihte baslasak sonuc daha iyi cikar mi yoksa fark etmez mi", "best_date"),
    ("so i was looking at this result and honestly i have no idea why the score is so low can you walk me through it", "why_score"),
    ("hocam kusura bakma ama bu puan bana biraz dusuk geldi neden boyle oldugunu soyler misin", "why_score"),
    ("i mean the numbers look okay at first glance but is the battery actually going to survive the night or not", "battery_info"),
    ("yani su gece meselesi var ya aracin donma ihtimali var mi onu bilmek istiyorum", "night_thermal"),
    ("okay thanks that was really useful i think i get it now", "thanks"),
    ("ya bu kadar uzun anlatma bi kac cumlede soyle", "confused"),
    ("honestly im not convinced by that answer where did that number even come from", "doubt"),
]


# Second human set (HUMAN_TEST2): written AFTER the reaction rules existed and never used to tune anything. Includes traps for the rules
# (a "no" or "tamam" that begins a real question) and rambling messages.
HUMAN_TEST2 = [
    # reactions, new wording
    ("yep thats fine", "affirm"), ("cool do that one", "affirm"), ("hadi bakalim yap", "affirm"), ("aynen kardes", "affirm"), ("mhm go ahead", "affirm"),
    ("nah leave it", "deny"), ("yok kalsin simdilik", "deny"), ("hayir olmasin", "deny"), ("no dont change anything", "deny"), ("istemem bunu", "deny"),
    ("hmm i dont follow", "confused"), ("bu ne biçim anlatim cok karisik", "confused"), ("kisaca soylesene", "confused"), ("too much jargon man", "confused"), ("ben bunu hic anlayamadim", "confused"),
    ("is that really true", "doubt"), ("hmm sounds fishy to me", "doubt"), ("buna nasil emin olabiliyorsun", "doubt"), ("show me where that comes from", "doubt"), ("inanasim gelmiyor", "doubt"),
    ("ya sen ne biçim asistansin", "complaint"), ("man this bot is trash", "complaint"), ("yine ayni hata cok sinir bozucu", "complaint"), ("pathetic answer honestly", "complaint"), ("hic faydan yok", "complaint"),
    ("one more time please", "repeat"), ("az once ne soylediydin", "repeat"), ("can you show me that again", "repeat"), ("yeniden yazar misin", "repeat"),
    ("keep talking", "elaborate"), ("biraz daha ac bakalim", "elaborate"), ("and why is that so", "elaborate"), ("nedenini de anlatsana", "elaborate"), ("more please", "elaborate"),
    # ordinary questions, messy
    ("hocam su puan neden bu kadar dusuk cikti anlamadim", "why_score"), ("ok but why is it low tho", "why_score"), ("puani nasil artirabilirim ya", "improve"), ("any tips to get it higher", "improve"),
    ("hangi tarihte baslarsam sonuc daha iyi olur", "best_date"), ("is there a better week to go", "best_date"), ("arac olarak ne secsem daha iyi", "best_vehicle"), ("which rover should i pick here", "best_vehicle"),
    ("yakinlarda daha mantikli bi nokta var mi", "best_site"), ("could another spot work better", "best_site"),
    ("gunes yeterince geliyor mu orada", "sun_info"), ("will there be enough light", "sun_info"), ("dunyayla baglanti kopar mi", "comms_info"), ("is there a radio blackout", "comms_info"),
    ("zemin cok mu egimli", "slope_info"), ("how steep is the landing area", "slope_info"), ("gece soguktan bozulur mu bu arac", "night_thermal"), ("can it survive the cold dark nights", "night_thermal"),
    ("pil gece boyunca idare eder mi", "battery_info"), ("is the battery enough for the whole night", "battery_info"), ("verilere ne kadar guvenmeliyim", "data_quality"), ("how solid are these figures", "data_quality"),
    ("hocam ozetle su sonucu", "explain_result"), ("so what is the verdict here", "explain_result"), ("a mi b mi daha iyi sence", "compare_results"), ("whats the difference between my two runs", "compare_results"),
    ("libration ne ya", "glossary"), ("whats a terminator in this context", "glossary"), ("tarihi mart 3 olarak degistir", "change_context"), ("use apollo 15 instead please", "change_context"),
    # traps: words the rules look for, inside real questions
    ("no sun at night right", "sun_info"), ("tamam ama puan neden dusuk", "why_score"), ("yes but how can we shift the date", "best_date"), ("dte ne demek", "glossary"),
    ("what does terminator mean", "glossary"), ("hayir degil mi gece donmaz mi", "night_thermal"), ("evet ama tarihi 5 aralik yap", "change_context"), ("ok so which vehicle is best", "best_vehicle"),
    ("is it true that the pole gets sunlight all year", "sun_info"), ("source of the sun data", "data_quality"),
    # long, rambling
    ("ya ben bu ekrana bakiyorum da bir turlu anlam veremedim acaba baska bi tarihte baslarsak puan yukselir mi", "best_date"),
    ("so my friend told me the pole is always dark and honestly i am confused is there any sunlight at all there", "sun_info"),
    ("tesekkurler ya cok yardimci oldun bi sure sonra yine bakacagim", "thanks"), ("ok great that explains it see you tomorrow", "bye"),
    ("bana bir sarki sozu yazar misin", "unknown"), ("whats the best pizza in rome", "unknown"),
]


# Third human set (HUMAN_TEST3): the number to quote. Written after HUMAN_TEST2's failures had been fixed, in new words, and not used for anything.
HUMAN_TEST3 = [
    ("sure why not", "affirm"), ("alright lets do it", "affirm"), ("tamam o zaman uygula", "apply"), ("evet bunu istiyorum", "affirm"), ("yap yap", "affirm"),
    ("no i dont want that", "deny"), ("hayir bunu sec degil", "deny"), ("nope scrap that", "deny"), ("bu olmasin yok", "deny"), ("vazgectim ben", "deny"),
    ("what do you even mean by that", "confused"), ("ya bunu cocuk gibi anlat bana", "confused"), ("i am totally lost here", "confused"), ("kafa karistirdin biraz sade ol", "confused"), ("use normal words please", "confused"),
    ("are you positive about this", "doubt"), ("i find that hard to believe", "doubt"), ("nasil bu kadar emin konusuyorsun", "doubt"), ("whats your source for that", "doubt"), ("bana yalan mi soyledin", "doubt"),
    ("this is the worst assistant ever", "complaint"), ("hic ise yaramiyor bu sey", "complaint"), ("ugh you keep failing", "complaint"), ("cok sacma bir bot bu", "complaint"), ("totally unusable", "complaint"),
    ("can you say it again slower", "repeat"), ("bunu yeniden yazar misin", "repeat"), ("what was that last thing", "repeat"), ("bir kez daha goster", "repeat"),
    ("please go on", "elaborate"), ("daha ayrintili olsun", "elaborate"), ("what happens after that", "elaborate"), ("bir de nedenini soyle", "elaborate"),
    ("ozetle bana su sonucu abi", "explain_result"), ("how did we do overall", "explain_result"), ("neden bu puani almis acaba", "why_score"), ("what pulled the number down so much", "why_score"),
    ("daha iyi bir sonuc icin neyi degistirmeliyim", "improve"), ("how would i make this look better", "improve"), ("hangi gunden baslamak mantikli", "best_date"), ("is it smarter to start in january", "best_date"),
    ("hangi araci gondersek iyi olur", "best_vehicle"), ("which spacecraft would suit this location", "best_vehicle"), ("bunun yerine baska bir nokta secsek", "best_site"), ("any safer place close by", "best_site"),
    ("orda gunes isigi bol mu", "sun_info"), ("how often is the sun above the horizon there", "sun_info"), ("dunya ile baglanti sorunu olur mu", "comms_info"), ("can it send data back to earth from there", "comms_info"),
    ("yamac cok dik mi", "slope_info"), ("could it roll over on those slopes", "slope_info"), ("karanlikta soguktan zarar gorur mu", "night_thermal"), ("does the lunar night kill it", "night_thermal"),
    ("batarya yeterli mi gece icin", "battery_info"), ("how much power does it have for the dark", "battery_info"), ("bu bilgiler ne kadar dogru", "data_quality"), ("are the inputs trustworthy", "data_quality"),
    ("a mi b mi hangisi kazandi", "compare_results"), ("which of my two options is the stronger one", "compare_results"), ("sabitlenme nedir", "glossary"), ("what does elevation mask mean", "glossary"),
    ("tarihi 4 subat olsun", "change_context"), ("could you switch to the shackleton site", "change_context"), ("peki ya lunokhod 1 olsaydi", "try_option"), ("what if we had a heater onboard", "try_option"),
    # traps
    ("no way the pole has that much sun is it true", "sun_info"), ("tamam peki puan neden bu kadar dusuk", "why_score"), ("yes and how do i make the score better", "improve"), ("ne demek libration", "glossary"),
    ("sure but is the battery enough", "battery_info"), ("anlamadim neden dusuk puan", "why_score"),
    # long and messy, and the edges
    ("abi ben bunu arkadasima gostercem ama bi turlu aciklayamadim neden bu tarih daha kotu cikiyor bana anlatir misin", "why_score"),
    ("honestly i just want to know if the rover can live through a night there or not that is all i care about", "night_thermal"),
    ("tamam super tesekkurler cok iyiydi", "thanks"), ("alright thats all from me bye", "bye"), ("merhabaaa naber", "greet"), ("heyyy whats up", "greet"),
    ("bana bir masal yazar misin", "unknown"), ("who will win the match tonight", "unknown"), ("sen kimsin ne ise yarasin", "about"), ("are you a robot or what", "about"),
]


# Fourth human set (HUMAN_TEST4, 2026-11-15): the questions real users typed into the app that the assistant failed on (from
# data/assistant_unrecognized.jsonl), plus new ones in the same style: a whole mission plan in one sentence, Turkish suffixes on
# place names, two questions at once, a suggestion button retyped in other words. Written BEFORE the fixes and never trained on.
HUMAN_TEST4 = [
    # typed by real users (the log)
    ("16 kasım 2026 tarihinde ayın güney kutbuna gitmek istiyorum, buna en uygun araç hangisi olur", "best_vehicle"),
    ("28 kasım 2026 tarihinde ayın güney kutbuna araç yollamak istiyorum. hangi araç uygundur", "best_vehicle"),
    ("28 kasım 2026 tarihinde ayın güney kutbuna araç yollamak istiyorum. hangi araç uygundur veya hangi özelliklere sahip olmalıdır,", "best_vehicle"),
    ("Shackleton kraterine ne zaman gitmeliyim", "best_date"),
    ("Yakın tarihlerde en çok ışık alabileceğim ay enlem boylamı neresi olur", "best_site"),
    ("Ne yaparsam güneş ufuk açım artar", "improve"),
    ("yalan söylüyorsun", "doubt"),
    # a mission plan, then the question
    ("aralık başında güney kutbuna inmek istesek hangi aracı seçmeliyiz", "best_vehicle"),
    ("kral 5 ocakta shackletona hangi arac gitsin", "best_vehicle"),
    ("mart 2027de kutba araç göndericez hangisi mantıklı olur", "best_vehicle"),
    ("i want to land near the south pole on 10 december, which spacecraft fits best", "best_vehicle"),
    ("we are planning a polar mission for february what lander should we pick", "best_vehicle"),
    ("20 şubatta malaperte inecek en iyi araç ne", "best_vehicle"),
    ("ayın güney kutbu için en ideal uzay aracı hangisi abi", "best_vehicle"),
    ("malaperte en iyi ne zaman inilir", "best_date"),
    ("haworth icin en uygun tarih hangisi", "best_date"),
    ("when is the best time to land at shackleton", "best_date"),
    ("önümüzdeki aylarda güney kutbuna ne zaman gitsek iyi olur", "best_date"),
    ("faustiniye gitmek icin en mantikli gun ne zaman", "best_date"),
    ("güney kutbunda en çok güneş alan yer neresi", "best_site"),
    ("where near the south pole gets the most sunlight next month", "best_site"),
    ("isik en fazla nerde olur kutupta", "best_site"),
    ("hangi koordinat daha çok ışık alır", "best_site"),
    ("3 martta shackleton nasil", "explain_result"),
    ("how is shackleton on march 3", "explain_result"),
    ("malapert 20 aralıkta nasıl görünüyor", "explain_result"),
    # what the vehicle must have (a new question type)
    ("kutba gidecek bir aracın hangi özellikleri olmalı", "requirements"),
    ("what specs does a lander need to survive at shackleton in january", "requirements"),
    ("bu tarihte oraya gidecek aracın bataryası ne kadar olmalı", "requirements"),
    ("15 aralıkta malapert icin nasıl bir araç tasarlamalıyız", "requirements"),
    ("what kind of vehicle would survive there", "requirements"),
    ("bu görev için aracın gereksinimleri neler olmalı", "requirements"),
    # a suggestion button, typed again in other words
    ("tarihi nasıl kaydırsak daha iyi olur", "best_date"),
    ("sonucu kısaca özetler misin", "explain_result"),
    ("puanı niye bu kadar düşük çıktı ya", "why_score"),
    ("neden puan düşük", "why_score"),
    ("bunu tabloda gösterir misin", "table_request"),
    ("hangi senaryo daha iyi sence", "compare_results"),
    ("burada hangi araç daha iyi olurdu", "best_vehicle"),
    ("daha iyi bir yer var mı yakınlarda", "best_site"),
    ("bu veriler ne kadar güvenilir ki", "data_quality"),
    ("ne işe yararsın", "about"),
    ("sen neler yapabiliyorsun ki", "about"),
    ("güneşin ufuk açısını nasıl yükseltebilirim", "improve"),
]


# Fifth human set (HUMAN_TEST5, 2026-11-15): written after nlu_frames.py was finished, in words the rules had not been shown.
# Its one clean reading is 78.4% (frames + clauses on the old network). After that its failures were seen while the
# 'requirements' training sentences and the strong-frame rule were written, so later readings of it are development numbers.
HUMAN_TEST5 = [
    ("selam, ocak ortasinda guney kutbunda bir inis planliyoruz, sence hangi lander daha mantikli", "best_vehicle"),
    ("hocam ekimde malapertte calisacak bir arac lazim, hangisini onerirsin", "best_vehicle"),
    ("next spring we want to put something on the lunar south pole. what vehicle would you go with", "best_vehicle"),
    ("which of your landers handles the polar night best", "best_vehicle"),
    ("nobileye ne zaman inmek daha avantajli olur", "best_date"),
    ("what week next year is ideal for landing at faustini", "best_date"),
    ("kutupta en uygun inis donemi hangisi", "best_date"),
    ("ilk ne zaman gidebiliriz ki en iyi sonuc olsun", "best_date"),
    ("kutup bolgesinde gunesi en bol alan nokta hangisi", "best_site"),
    ("which spot around here keeps the sun the longest", "best_site"),
    ("nereye inersek en guvenli olur", "best_site"),
    ("where would you land if you had to choose", "best_site"),
    ("haworth subatta nasil olur", "explain_result"),
    ("how would nobile look in april", "explain_result"),
    ("orada gece hayatta kalmak icin aracin neyi olmali", "requirements"),
    ("what equipment would a rover need at haworth next month", "requirements"),
    ("malapert icin ideal bir aracin teknik ozellikleri ne olmali", "requirements"),
    ("how much battery and what heater would a lander need there", "requirements"),
    ("puan neden boyle dusuk geldi bana", "why_score"),
    ("bu sonucu kisaca anlatir misin", "explain_result"),
    ("bunu daha iyi yapmanin yolu var mi", "improve"),
    ("baska bi gun deneyelim mi daha iyi olur mu", "best_date"),
    ("is there a better vehicle for this", "best_vehicle"),
    ("gunes ne kadar sure gorunuyor orada", "sun_info"),
    ("dunyayla ne kadar konusabiliyoruz", "comms_info"),
    ("pil gece boyunca dayanir mi sence", "battery_info"),
    ("bu rakamlar nereden geliyor", "data_quality"),
    ("dte nedir kisaca", "glossary"),
    ("tarihi 1 mart yapalim", "change_context"),
    ("bunu tablo olarak goster", "table_request"),
    ("bu sohbetin amaci ne senin isin ne", "about"),
    ("bana fikra anlat", "unknown"),
    ("istanbul ankara arasi kac km", "unknown"),
    ("merhaba nasilsin", "greet"),
    ("tesekkur ederim cok yardimci oldun", "thanks"),
    ("tamam", "affirm"),
    ("hayir istemiyorum", "deny"),
]
