"""Plain-language definitions of the terms the analysis page uses, in Turkish and English."""

TERMS = {
    "index": {"keys": ["index", "indeks", "suitability", "uygunluk", "score", "puan", "skor"],
              "title": {"en": "Suitability index", "tr": "Uygunluk puanı"},
              "en": "A 0 to 100 grade for how well this vehicle can work at this place during the 30 days. It mixes four things: how much sunlight, how often Earth can be seen, how long the longest night is, and how flat the ground is. Then it is capped by hard limits (see 'held down by'). Around the south pole almost nothing scores above 60, so 60 is already very good.",
              "tr": "Bu aracın bu yerde 30 gün boyunca ne kadar iyi çalışabileceğinin 0-100 arası notu. Dört şeyi birleştirir: ne kadar güneş var, Dünya ne sıklıkla görünüyor, en uzun gece kaç saat, zemin ne kadar düz. Sonra sert sınırlarla kısılır ('held down by'). Güney kutbu çevresinde neredeyse hiçbir yer 60'ı geçmez, yani 60 zaten çok iyidir."},
    "illumination": {"keys": ["illumination", "sunlit", "daylight", "aydinlik", "aydinlanma", "illuminated"],
                     "title": {"en": "Illumination (sunlit share)", "tr": "Aydınlanma oranı"},
                     "en": "The share of the 30 days when the Sun is above the local skyline and shines on the spot. On the Moon a 'day' lasts about two weeks, and hills can hide the Sun even in daytime.",
                     "tr": "30 günün ne kadarında Güneş'in bölgenin ufuk çizgisinin üstünde olup noktayı aydınlattığı. Ay'da bir 'gün' yaklaşık iki hafta sürer ve tepeler gündüzün bile Güneş'i gizleyebilir."},
    "dte": {"keys": ["dte", "direct to earth", "direct-to-earth", "dunya baglantisi", "dogrudan dunya"],
            "title": {"en": "Direct-to-Earth (DTE) link", "tr": "Doğrudan Dünya bağlantısı (DTE)"},
            "en": "The share of time Earth is above the local skyline, so the vehicle can talk to Earth by radio without a relay satellite. If a ridge hides Earth, the link is cut.",
            "tr": "Dünya'nın bölgenin ufuk çizgisinin üstünde olduğu, yani aracın röle uydusu olmadan Dünya ile doğrudan telsizle konuşabildiği sürenin oranı. Bir sırt Dünya'yı gizlerse bağlantı kesilir."},
    "slope": {"keys": ["slope", "egim", "dikilik", "meyil", "tilt"],
              "title": {"en": "Slope", "tr": "Eğim"},
              "en": "How steep the ground is, in degrees from flat. Every lander has a limit; beyond it the vehicle can tip over. The page measures it on the 500 m landing area around the point.",
              "tr": "Zeminin düzlükten kaç derece dik olduğu. Her aracın bir sınırı vardır; geçilirse araç devrilebilir. Sayfa bunu noktanın etrafındaki 500 m'lik iniş alanında ölçer."},
    "veto": {"keys": ["veto", "held down", "capped", "cap", "kisitla", "tavan", "kisildi"],
             "title": {"en": "Held down by (veto gate)", "tr": "Held down by (veto kapısı)"},
             "en": "A hard limit on the score. If the ground is too steep, Earth is never visible, there is no sunlight, or a long night would freeze the vehicle, the score cannot go above a ceiling, however good the other numbers are. 'Held down by' names the limit that is binding.",
             "tr": "Puana konan sert bir tavan. Zemin çok dikse, Dünya hiç görünmüyorsa, güneş hiç yoksa ya da uzun bir gece aracı donduracaksa, diğer sayılar ne kadar iyi olursa olsun puan bu tavanı geçemez. 'Held down by' hangi sınırın devrede olduğunu söyler."},
    "skyline": {"keys": ["skyline", "horizon", "ufuk", "silüet", "siluet"],
                "title": {"en": "Skyline", "tr": "Ufuk çizgisi"},
                "en": "The outline of hills and crater rims around the spot, as seen from a 2 m mast. The Sun or Earth only counts as visible when it is higher than this line.",
                "tr": "Noktayı çevreleyen tepelerin ve krater kenarlarının, 2 m yüksekten bakıldığında oluşan çizgisi. Güneş ya da Dünya ancak bu çizginin üstündeyse görünür sayılır."},
    "night": {"keys": ["night", "gece", "darkness", "karanlik", "lunar night", "ay gecesi"],
              "title": {"en": "Lunar night", "tr": "Ay gecesi"},
              "en": "A stretch with no sunlight. At the poles it can last for weeks. Without a heat source or enough battery, a vehicle can get too cold and fail.",
              "tr": "Hiç güneş ışığı olmayan süre. Kutuplarda haftalarca sürebilir. Isı kaynağı ya da yeterli pil yoksa araç aşırı soğuyup bozulabilir."},
    "relay": {"keys": ["relay", "role"],
              "title": {"en": "Relay satellite", "tr": "Röle uydusu"},
              "en": "A satellite that passes the vehicle's messages to Earth. A vehicle that uses one does not need to see Earth itself.",
              "tr": "Aracın mesajlarını Dünya'ya ileten uydu. Röle kullanan araç Dünya'yı kendisi görmek zorunda değildir."},
    "thermal": {"keys": ["thermal", "heater", "rhu", "isil", "isitici", "radioisotope"],
                "title": {"en": "Heat source (RHU)", "tr": "Isı kaynağı (RHU)"},
                "en": "A small radioisotope heater that keeps the electronics warm through the night without using battery power. Vehicles without one are assumed to survive only about 72 hours of darkness.",
                "tr": "Geceyi pil harcamadan geçirmek için elektroniği sıcak tutan küçük radyoizotop ısıtıcı. Isıtıcısı olmayan araçların yalnızca yaklaşık 72 saatlik karanlığa dayandığı varsayılır."},
    "window": {"keys": ["window", "pencere", "30 day", "30 gun"],
               "title": {"en": "30-day window", "tr": "30 günlük pencere"},
               "en": "The analysis looks at the 30 days that start on the date you picked. Moving the start date moves the whole window.",
               "tr": "Analiz, seçtiğin tarihte başlayan 30 güne bakar. Başlangıç tarihini oynatmak tüm pencereyi kaydırır."},
}


def find_term(norm_text: str):
    """The glossary entry whose key appears in the (normalized) text, longest key first, or None."""
    import re
    best = None
    for name, t in TERMS.items():
        for k in t["keys"]:  # a key must start a word (Turkish suffixes may follow), so 'dte' never matches inside 'route'
            if re.search(r"(?<![a-z0-9])" + re.escape(k), norm_text) and (best is None or len(k) > best[1]):
                best = (name, len(k))
    return best[0] if best else None
