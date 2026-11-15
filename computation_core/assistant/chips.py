"""Every ready-made question the assistant or the page offers as a button, and what each one means.

A button's text is sent as a normal message, so it must be understood *every time*, however the model feels about it today.
Matching is on the normalised text (case, punctuation, Turkish letters and extra spaces do not matter) and happens before the
model is asked. The same list is also fed to the model as training data, so typing a variation of a button works too.
"""
from .text import normalize

CHIP_INTENTS: dict[str, list[str]] = {
    "about": ["What can you do?", "Neler yapabilirsin?"],
    "why_score": ["Why is the score low?", "Puan neden düşük?", "Why is the score what it is?", "Puan neden böyle?"],
    "best_date": ["How can we shift the date?", "How can we shift the date to get a better result?", "Tarihi nasıl oynatabiliriz?",
                  "Tarihi nasıl kaydırırsak daha iyi sonuç alırız?"],
    "best_vehicle": ["Which vehicle works best here?", "Which vehicle is better here?", "Hangi araç burada daha iyi?", "Hangi araç burada en iyi çalışır?",
                     "Which vehicle should we send on September 12?", "12 Eylül'de hangi aracı göndermeliyiz?"],
    "best_site": ["Is there a better location?", "Daha iyi bir konum var mı?"],
    "explain_result": ["Summarize the result", "How does Shackleton look on March 3?", "Sonucu özetle", "Summarize scenario A", "Summarize scenario B", "A senaryosunu özetle", "B senaryosunu özetle"],
    "data_quality": ["How reliable is this data?", "Veriler ne kadar güvenilir?"],
    "table_request": ["Show it as a table", "Bunu tabloda göster"],
    "apply": ["Apply that date", "Bu tarihi uygula"],
    "change_context": ["Set the date to December 5", "Set the place to Shackleton", "Tarihi 5 Aralık yap", "Konumu Shackleton yap"],
    "compare_dates": ["What if we start a week later?", "What if we start on January 1?", "Bir hafta sonra başlarsak ne olur?", "1 Ocak'ta başlarsak?"],
    "try_option": ["What about IM-1?", "How about Shackleton?", "What if it had a heater?", "Peki ya IM-1 ile?", "Shackleton nasıl?", "Isıtıcı olsa ne değişir?"],
    "compare_results": ["Compare the two results", "Which scenario is better?", "İki sonucu karşılaştır", "Hangi senaryo daha iyi?",
                        "Why is A lower?", "Why is B lower?", "A neden daha düşük?", "B neden daha düşük?"],
    "improve": ["How can I improve A?", "How can I improve B?", "A'yı nasıl iyileştirebilirim?", "B'yi nasıl iyileştirebilirim?"],
}

_BY_TEXT = {normalize(t): intent for intent, texts in CHIP_INTENTS.items() for t in texts}


def route_chip(message: str) -> str | None:
    """The intent of a ready-made question, or None when the message is something else."""
    return _BY_TEXT.get(normalize(message))


def all_chips() -> list[tuple[str, str]]:
    return [(t, intent) for intent, texts in CHIP_INTENTS.items() for t in texts]
