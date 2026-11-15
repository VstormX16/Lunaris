"""The assistant: its own language model, entity rules, and the answers built on the real engine."""
from datetime import date

import pytest

from computation_core.assistant import tools
from computation_core.assistant.dialog import Assistant
from computation_core.assistant.entities import extract
from computation_core.assistant.eval_set import EVAL, FINAL
from computation_core.assistant.nlu_corpus import INTENTS
from computation_core.assistant.nlu_model import IntentEnsemble
from computation_core.assistant.text import detect_language, normalize
from computation_core.assistant.tools import Scenario

TODAY = date(2026, 11, 14)


def test_text_folds_turkish_and_detects_language():
    assert normalize("Sımdı ÇOK şey İğne") == "simdi cok sey igne"
    assert detect_language("bu puan neden düşük çıktı") == "tr" and detect_language("why is the score low") == "en"
    assert detect_language("merhaba") == "tr"


def test_entities_dates_vehicles_sites_coordinates_and_mask():
    e = extract("12 aralik yerine 3 ocak olsa ne degisir", TODAY)
    assert e.dates == ["2026-12-12", "2027-01-03"] and e.masked == "xdatex yerine xdatex olsa ne degisir"
    assert extract("change the date to december 5 2027", TODAY).dates == ["2027-12-05"]
    assert extract("12.12.2026", TODAY).dates == ["2026-12-12"] and extract("2026-12-05", TODAY).dates == ["2026-12-05"]
    assert extract("tarihi bir hafta ileri al", TODAY).dates == [{"shift_days": 7}]
    assert extract("araci chang'e 3 yap", TODAY).vehicles == ["chang-e-3-yutu"]
    assert extract("compare Apollo 15 with IM-1", TODAY).vehicles == ["apollo-15", "im-1-odysseus"]  # 'apollo 1' must not steal 'apollo 15'
    assert extract("konumu malapert yap", TODAY).sites == ["malapert"]
    assert extract("set the coordinates to -86.2 and 20.5", TODAY).coords == [(-86.2, 20.5)]
    assert extract("koordinatlari eksi 86 ve 20 yap", TODAY).coords == [(-86.0, 20.0)]
    assert extract("batarya iki kat olursa", TODAY).multiplier == 2.0 and extract("bataryayi 8 kat buyut", TODAY).multiplier == 8.0
    assert extract("puan neden 15", TODAY).dates == [] and extract("puan neden 15", TODAY).coords == []  # a bare number is not a date


@pytest.fixture(scope="module")
def nlu():
    if not IntentEnsemble.paths():
        pytest.skip("assistant model not trained (python -m computation_core.assistant.train_nlu)")
    return IntentEnsemble.load()


def test_nlu_accuracy_on_hand_written_sentences_never_seen_in_training(nlu):
    bot = Assistant(None, {}, nlu)
    norm_train = {normalize(c) for v in INTENTS.values() for lang in v.values() for c in lang}
    for t, _ in FINAL:  # EVAL was the development set (one trivial overlap: 'hey'); FINAL must be untouched
        assert normalize(t) not in norm_train, f"test sentence leaked into the training corpus: {t}"
    for name, S, floor in (("EVAL", EVAL, 0.80), ("FINAL", FINAL, 0.70)):
        right = sum(bot.route(t)[0] == y for t, y in S)
        wrong = sum(bot.route(t)[0] not in (y, "clarify") for t, y in S)
        assert right / len(S) >= floor, (name, right / len(S))
        assert wrong / len(S) <= 0.15, (name, "confidently wrong", wrong / len(S))  # asking back is fine, answering the wrong thing is not


@pytest.fixture(scope="module")
def bot(nlu):
    from computation_core.vehicle_assessment import load_profiles
    from data_engine import spice_loader
    from data_engine.lola_downloader import load_dem
    from computation_core.global_dem import load_global_dem
    from computation_core.site_dem import SiteDEM
    from data_engine import config
    spice_loader.load_kernels()
    return Assistant(SiteDEM(load_dem(), load_global_dem(config.GLOBAL_DEM_PATH)), load_profiles(), nlu)  # as the server builds it: the whole Moon


CTX = {"scenarios": [{"date": "2026-12-01", "vehicleId": "chandrayaan-3-vikram-pragyan", "lat": -85.987, "lon": 2.08}]}


def _say(bot, q, mem=None):
    return bot.respond(q, CTX, mem or {}, TODAY)


def _blocks(r, kind):
    return [b for b in r["reply"]["blocks"] if b["type"] == kind]


def test_scan_rows_equal_the_plain_engine_and_never_offer_past_dates(bot):
    sc = Scenario("2026-12-01", "chandrayaan-3-vikram-pragyan", -85.987, 2.08)
    p = bot.profiles[sc.vehicle_id]
    rows = tools.date_scan(bot.dem, p, sc, 60, earliest=TODAY.isoformat())
    _, a = tools.assess(bot.dem, p, sc)
    assert abs(next(r for r in rows if r["date"] == sc.date)["index"] - a["vehicle_index"]) < 1e-9  # the shortcut is exact
    assert min(r["date"] for r in rows) >= TODAY.isoformat()


def test_assistant_answers_in_the_users_language_with_numbers_from_the_engine(bot):
    tr, en = _say(bot, "bu puan neden bu kadar düşük çıktı"), _say(bot, "why is the score so low")
    assert tr["intent"] == en["intent"] == "why_score" and tr["language"] == "tr" and en["language"] == "en"
    _, a = tools.assess(bot.dem, bot.profiles["chandrayaan-3-vikram-pragyan"], Scenario("2026-12-01", "chandrayaan-3-vikram-pragyan", -85.987, 2.08))
    assert f"**{a['vehicle_index']:.0f}**" in tr["reply"]["blocks"][0]["text"] and f"**{a['vehicle_index']:.0f}**" in en["reply"]["blocks"][0]["text"]
    assert _blocks(en, "table")[0]["rows"]  # a table of where the points are lost


def test_date_question_runs_a_scan_offers_a_button_and_remembers_it(bot):
    r = _say(bot, "Tarihi nasıl oynatabiliriz daha iyi bir sonuç almak için")
    assert r["intent"] == "best_date" and _blocks(r, "table") and r["memory"]["table"]
    sug = r["memory"].get("suggestion")
    assert sug and sug["type"] == "apply_scenario" and sug["date"] >= TODAY.isoformat()
    assert _blocks(r, "actions")[0]["buttons"][0]["action"] == sug
    r2 = _say(bot, "tamam uygula", r["memory"])  # "yes, apply it" in the next turn carries the action to the page
    assert r2["intent"] == "apply" and r2["reply"]["actions"] == [sug]
    r3 = _say(bot, "bunu tabloda göster", r["memory"])  # and the last table can be shown again
    assert r3["intent"] == "table_request" and _blocks(r3, "table")[0] == r["memory"]["table"]


def test_change_context_and_compare_dates_use_the_entities(bot):
    r = _say(bot, "tarihi 20 aralık yap")
    assert r["intent"] == "change_context" and r["reply"]["actions"] == [{"type": "apply_scenario", "date": "2026-12-20"}]
    r = _say(bot, "what if we start on January 3?")
    assert r["intent"] == "compare_dates" and len(_blocks(r, "table")[0]["rows"]) == 2
    r = _say(bot, "konumu shackleton yap")
    assert r["reply"]["actions"][0]["lat"] == -89.54


def test_battery_unknown_is_said_not_guessed_and_known_battery_gives_a_sweep(bot):
    # every profile has a battery figure since the 2026-11-15 dataset; take one away to check the "unknown" path
    vid = CTX["scenarios"][0]["vehicleId"]
    saved = bot.profiles[vid]
    bot.profiles[vid] = {**saved, "battery_wh": {"value": None, "status": "unknown"}}
    try:
        r = _say(bot, "what if the battery were 4 times bigger")
    finally:
        bot.profiles[vid] = saved
    assert r["intent"] == "battery_info" and not _blocks(r, "table") and "unknown" in r["reply"]["blocks"][0]["text"]
    ctx = {"scenarios": [{"date": "2026-12-01", "vehicleId": "im-1-odysseus", "lat": -85.987, "lon": 2.08}]}
    r = bot.respond("pil iki kat olsa ne olur", ctx, {}, TODAY)
    rows = _blocks(r, "table")[0]["rows"]
    assert float(rows[1][1].split()[0]) > float(rows[0][1].split()[0])  # more battery, more sustainable load


def test_vehicle_ranking_leaves_out_battery_only_craft_and_says_so(bot):
    r = _say(bot, "which vehicle works best here")
    assert r["intent"] == "best_vehicle"
    names = [row[1] for row in _blocks(r, "table")[0]["rows"]]
    assert not any("Apollo" in n or "Luna 9" in n for n in names)
    assert any("battery-only" in b["text"] for b in _blocks(r, "text"))


def test_out_of_scope_and_unsure_questions_are_handled_politely(bot):
    r = _say(bot, "bana bir şiir yaz")
    assert r["intent"] == "unknown" and "yalnızca" in r["reply"]["blocks"][0]["text"]
    r = bot.respond("why is the score low", {}, {}, TODAY)  # no analysis on the page yet
    assert "analysis" in r["reply"]["blocks"][0]["text"]
    assert _say(bot, "hello")["intent"] == "greet" and _say(bot, "thanks a lot")["intent"] == "thanks"


def test_every_intent_has_a_working_handler(bot):
    fill = {"{D}": "5 aralik", "{V}": "apollo 15", "{S}": "shackleton", "{M}": "2", "{N}": "3"}
    for intent, langs in INTENTS.items():
        for sents in langs.values():
            q = sents[0]
            for k, v in fill.items():
                q = q.replace(k, v)
            r = _say(bot, q)
            assert r["reply"]["blocks"] and all(b["type"] in ("text", "table", "bullets", "chips", "actions") for b in r["reply"]["blocks"]), (intent, q)


def test_topic_questions_unseen_in_training_are_understood(nlu):
    """'What can I do to raise <quantity>?' in words the grammar never produced; the topic head separates the quantity."""
    from computation_core.assistant.eval_set import FINAL2, TOPIC_TEST
    from computation_core.assistant.nlu_grammar import DOWN, INFO, LEX, UP, WHY
    produced = {normalize(f.replace("{T}", t)) for lang in ("en", "tr") for f in UP[lang] + DOWN[lang] + INFO[lang] + WHY[lang] for terms in LEX.values() for t in terms[lang]}
    assert not [t for t, _ in TOPIC_TEST + [(s, 0) for s, _ in FINAL2] if normalize(t) in produced], "a test sentence is one the grammar produces"
    bot = Assistant(None, {}, nlu)
    routed = sum(bot.route(t)[0] == "improve" for t, _ in TOPIC_TEST)
    topic_ok = sum(nlu.predict_topic(t)[0] == topic for t, topic in TOPIC_TEST)
    assert routed / len(TOPIC_TEST) >= 0.80 and topic_ok / len(TOPIC_TEST) >= 0.70, (routed, topic_ok, len(TOPIC_TEST))
    # off-topic requests that use the same 'how do i ...' wording must not be mistaken for improvement questions
    for q in ("how can i improve my english", "ne yaparsam kilo veririm"):  # (known weak spot: "how do i get a better paying job" still looks like an improvement question)
        assert bot.route(q)[0] in ("unknown", "clarify"), q


def test_sun_angle_question_gets_a_real_answer_with_the_latitude_lever(bot):
    r = _say(bot, "Ne yaparsam güneş ufuk açım artar")
    assert r["intent"] == "improve"
    tables = _blocks(r, "table")
    assert len(tables) == 2 and "Enlem" in tables[1]["title"] + tables[1]["columns"][0]
    highest = [float(row[2].rstrip("°")) for row in tables[1]["rows"]]
    assert highest == sorted(highest) and highest[-1] - highest[0] > 3  # nearer the equator the Sun climbs higher (about 1 degree per degree)
    assert any("1,5°" in b["text"] for b in _blocks(r, "text"))  # says why the Sun stays low at the pole


def test_window_sun_elevation_stats_match_the_time_series(bot):
    from computation_core.site_analysis import site_timeseries
    sc = Scenario("2026-12-01", "chandrayaan-3-vikram-pragyan", -85.987, 2.08)
    (iso, m, series), = tools._windows(bot.dem, sc, sc.date, 1)
    ts = site_timeseries(bot.dem, sc.lat, sc.lon, sc.start_utc(), 30, 60)
    lit = ts["sun_lit_fraction"] >= 0.5
    assert abs(m["sun_el_mean_lit"] - ts["sun_el"][lit].mean()) < 1e-9 and abs(m["sun_el_max"] - ts["sun_el"].max()) < 1e-9
    rows = tools.latitude_sweep(sc)
    assert [r["offset_deg"] for r in rows] == [0.0, 1.0, 2.0, 3.0, 5.0] and rows[-1]["el_max"] > rows[0]["el_max"] + 3


def test_other_topics_answer_with_the_matching_quantity(bot):
    r = _say(bot, "eğimi nasıl azaltırım")
    assert r["intent"] == "improve" and "eğim" in _blocks(r, "table")[0]["title"].lower()
    r = _say(bot, "how do I make the night shorter")
    assert r["intent"] == "improve" and "night" in _blocks(r, "table")[0]["title"].lower()


def test_message_slots_override_the_scenario_and_the_answer_says_so(bot):
    r = _say(bot, "20 aralikta gece kac saat var")
    assert r["intent"] == "night_thermal" and r["reply"]["blocks"][0]["text"].startswith("Hesapladığım senaryo: 20 Ara 2026")
    _, a = tools.assess(bot.dem, bot.profiles["chandrayaan-3-vikram-pragyan"], Scenario("2026-12-20", "chandrayaan-3-vikram-pragyan", -85.987, 2.08))
    m, _ = tools.assess(bot.dem, bot.profiles["chandrayaan-3-vikram-pragyan"], Scenario("2026-12-20", "chandrayaan-3-vikram-pragyan", -85.987, 2.08))
    assert f"**{m['longest_darkness_hours']:.0f} saat**" in r["reply"]["blocks"][1]["text"]  # the number is for THAT date, not the page's
    r2 = _say(bot, "peki en iyi tarih hangisi")
    r3 = _say(bot, "o tarihte gece kac saat", r2["memory"])  # 'that date' = the date the assistant just suggested
    assert r3["reply"]["blocks"][0]["text"].startswith("Hesapladığım senaryo")


def test_what_about_a_vehicle_or_place_compares_and_never_changes_the_page(bot):
    r = _say(bot, "what about apollo 15")
    assert r["intent"] == "try_option" and not r["reply"].get("actions")
    rows = _blocks(r, "table")[0]["rows"]
    assert len(rows) == 2 and "current" in rows[0][0] and "Apollo 15" in rows[1][0]
    r = _say(bot, "shackleton nasil")
    assert not r["reply"].get("actions") and any("Shackleton" in row[0] for row in _blocks(r, "table")[0]["rows"])
    r = _say(bot, "araci apollo 15 yap")  # an explicit command still changes the page
    assert r["intent"] == "change_context" and r["reply"]["actions"] == [{"type": "apply_scenario", "vehicleId": "apollo-15"}]


def test_heater_what_if_is_labelled_and_lifts_the_thermal_cap(bot):
    r = _say(bot, "isitici olsa ne degisir")
    rows = _blocks(r, "table")[0]["rows"]
    assert "varsayımsal" in r["reply"]["blocks"][0]["text"].lower() and float(rows[1][1]) >= float(rows[0][1])


def test_why_for_a_topic_explains_that_quantity_not_the_overall_score(bot):
    r = _say(bot, "gunes acisi neden bu kadar dusuk")
    assert r["intent"] == "why_score" and not _blocks(r, "table") and "enlem" in r["reply"]["blocks"][0]["text"]


def test_when_questions_give_the_utc_start_of_the_outage(bot):
    r = _say(bot, "dunya ile iletisim ne zaman kesiliyor")
    text = " ".join(b["text"] for b in _blocks(r, "text"))
    assert r["intent"] == "comms_info" and "UTC'de başlıyor" in text


def test_window_metrics_report_when_the_long_night_starts():
    assert tools.hours_to_iso("2026-12-01", 61) == "2026-12-03 13:00" and tools.hours_to_iso("2026-12-01", None) is None
    assert tools._longest_false_run_start([True, False, True, False, False, False, True]) == 3
    assert tools._longest_false_run_start([True, True]) is None


def test_assistant_volunteers_what_it_noticed_from_the_engine(bot):
    r = _say(bot, "sonucu özetle")
    texts = [b["text"] for b in _blocks(r, "text")]
    assert any("fark ettiklerim" in t for t in texts)
    notes = _blocks(r, "bullets")[0]["items"]
    rows = tools.date_scan(bot.dem, bot.profiles["chandrayaan-3-vikram-pragyan"], Scenario("2026-12-01", "chandrayaan-3-vikram-pragyan", -85.987, 2.08), 60, earliest=TODAY.isoformat())
    best = max(r_["index"] for r_ in rows)
    # the date tip is one of the engine's own best dates (it prefers the nearest of the near-best ones)
    near_best = {f"→ {r_['index']:.0f}" for r_ in rows if r_["index"] >= best - 2}
    assert any(t in n for n in notes for t in near_best)
    assert r["memory"]["suggestion"]["type"] == "apply_scenario"  # so 'tamam' applies it


# ---------------------------------------------------------------- ready-made questions and two-scenario comparisons ----
import re as _re
import unicodedata as _ud
from pathlib import Path as _Path

from computation_core.assistant.chips import CHIP_INTENTS, all_chips, route_chip

CTX_AB = {"scenarios": [{"date": "2026-12-01", "vehicleId": "chandrayaan-3-vikram-pragyan", "lat": -85.987, "lon": 2.08},
                        {"date": "2026-12-01", "vehicleId": "im-1-odysseus", "lat": -85.987, "lon": 2.08}]}


def _typed_variants(t):
    ascii_ = t.translate(str.maketrans("ığşüöçİĞŞÜÖÇ", "igsuocIGSUOC"))  # how people type Turkish without its special letters
    return [t, t.lower(), t.upper(), _re.sub(r"[?!.]", "", t), "  " + t + "  ", ascii_, ascii_.lower().rstrip("?") + "!!"]


def test_every_ready_made_question_is_understood_however_it_is_typed(bot):
    """A button's text typed by hand must give the same answer as clicking it: case, punctuation, Turkish letters and spacing do not matter."""
    for text, intent in all_chips():
        for v in _typed_variants(text):
            assert route_chip(v) == intent, (v, intent)
    for text, intent in all_chips():  # and through the whole assistant, with and without earlier conversation
        for mem in ({}, {"last_intent": "explain_result", "lang": "en", "focus": 1}):
            ctx = CTX_AB if intent in ("compare_results", "improve") else CTX
            assert bot.respond(text.lower(), ctx, dict(mem))["intent"] == intent, (text, mem)


def test_every_button_the_app_can_show_is_in_the_registry(bot):
    """Buttons come from the assistant's replies and from the panel's welcome message: none may be missing from the registry."""
    known = {normalize(t) for t, _ in all_chips()}
    seen = set()
    for msg in ["hello", "merhaba", "summarize the result", "why is the score low", "how can we shift the date", "what about IM-1", "how are you", "kendini tanit", "set the date",
                "bilinmeyen bir sey xyz", "tarihleri karsilastir", "how reliable is this data", "which vehicle works best here", "is there a better location"]:
        for ctx in (CTX, CTX_AB):
            for b in bot.respond(msg, ctx, {})["reply"]["blocks"]:
                if isinstance(b, dict) and b.get("type") == "chips":
                    seen.update(b["items"])
    for b in bot.respond("compare the two results", CTX_AB, {})["reply"]["blocks"]:
        if isinstance(b, dict) and b.get("type") == "chips":
            seen.update(b["items"])
    panel = (_Path(__file__).resolve().parents[1] / "frontend/src/components/AssistantPanel.tsx").read_text(encoding="utf8")
    for chunk in _re.findall(r"type: 'chips', items: (.*?) \}", panel):  # the quoted strings of the welcome chips
        seen.update(_re.findall(r"'([^']+)'", chunk))
    missing = {t for t in seen if normalize(t) not in known}
    assert not missing, f"buttons that are not in assistant/chips.py: {missing}"


def test_comparing_two_results_uses_the_engines_own_numbers(bot):
    out = bot.respond("compare the two results", CTX_AB, {})
    assert out["intent"] == "compare_results"
    blocks = out["reply"]["blocks"]
    table = next(b for b in blocks if isinstance(b, dict) and b.get("type") == "table")
    assert table["columns"][1:3] == ["A", "B"]
    sa, sb = (Scenario(s["date"], s["vehicleId"], s["lat"], s["lon"]) for s in CTX_AB["scenarios"])
    ia = tools.assess(bot.dem, bot.profiles[sa.vehicle_id], sa)[1]["vehicle_index"]
    ib = tools.assess(bot.dem, bot.profiles[sb.vehicle_id], sb)[1]["vehicle_index"]
    assert table["rows"][0][1:3] == [f"{ia:.0f}", f"{ib:.0f}"]
    head = blocks[0]
    if abs(ia - ib) >= 2:
        assert f"**{'A' if ia > ib else 'B'}** is better" in head["text"]
    # what it suggests for the weaker one is the engine's own best date, aimed at that scenario
    weak = "b" if ia > ib else "a"
    acts = [b for b in blocks if isinstance(b, dict) and b.get("type") == "actions"]
    if acts:
        assert acts[0]["buttons"][0]["action"]["scenario"] == weak


def test_with_one_scenario_the_assistant_says_there_is_nothing_to_compare(bot):
    out = bot.respond("compare the two results", CTX, {})
    assert out["intent"] == "compare_results" and "only one scenario" in out["reply"]["blocks"][0]["text"].lower()


def test_questions_can_be_about_a_or_b_and_the_assistant_stays_on_it(bot):
    first = bot.respond("why is the score low for scenario B", CTX_AB, {})
    assert first["memory"]["focus"] == 1
    assert "scenario B" in first["reply"]["blocks"][0]["text"]
    sb = Scenario("2026-12-01", "im-1-odysseus", -85.987, 2.08)
    idx = tools.assess(bot.dem, bot.profiles[sb.vehicle_id], sb)[1]["vehicle_index"]
    assert f"{idx:.0f}" in str(first["reply"]["blocks"])
    follow = bot.respond("summarize the result", CTX_AB, first["memory"])   # no letter given: still B
    assert follow["memory"]["focus"] == 1 and "scenario B" in follow["reply"]["blocks"][0]["text"]
    back = bot.respond("and for scenario A", CTX_AB, follow["memory"])
    assert back["memory"]["focus"] == 0
    # an action suggested while talking about B changes B
    acts = [b for b in follow["reply"]["blocks"] if isinstance(b, dict) and b.get("type") == "actions"]
    for a in acts:
        assert all(btn["action"]["scenario"] == "b" for btn in a["buttons"])


def test_comparison_questions_are_understood_by_the_model(nlu):
    from computation_core.assistant.eval_set import COMPARE_TEST
    bot = Assistant(None, {}, nlu)
    right = sum(bot.route(t)[0] == y for t, y in COMPARE_TEST)
    wrong = sum(bot.route(t)[0] not in (y, "clarify") for t, y in COMPARE_TEST)
    assert right / len(COMPARE_TEST) >= 0.70, right / len(COMPARE_TEST)
    assert wrong / len(COMPARE_TEST) <= 0.15, wrong / len(COMPARE_TEST)
    norm_train = {normalize(c) for v in INTENTS.values() for lang in v.values() for c in lang}
    assert not [t for t, y in COMPARE_TEST if y == "compare_results" and normalize(t) in norm_train], "a comparison test sentence leaked into training"
    pos = [(t, y) for t, y in COMPARE_TEST if y == "compare_results"]
    assert sum(bot.route(t)[0] == y for t, y in pos) / len(pos) >= 0.70   # the comparison sentences alone, without the control sentences


# ---------------------------------------------------------------- people: how they write, and how they react to the assistant ----
def test_stretched_letters_and_noise_do_not_change_the_meaning():
    assert normalize("Heyyyy nasıllll ??? 5555") == "hey nasil 5555"          # letters stretched out, numbers left alone
    assert normalize("EVEEET") == "evet" or normalize("EVEEET") == "evet"
    assert normalize("  Tarihi   nasıl  oynatabiliriz?!?! ") == "tarihi nasil oynatabiliriz"


def test_human_sentences_are_understood_and_none_leaked_into_training(nlu):
    from computation_core.assistant.eval_set import HUMAN_TEST, HUMAN_TEST2, HUMAN_TEST3
    bot = Assistant(None, {}, nlu)
    train = {normalize(c) for v in INTENTS.values() for lang in v.values() for c in lang}
    for name, S in (("HUMAN_TEST", HUMAN_TEST), ("HUMAN_TEST2", HUMAN_TEST2), ("HUMAN_TEST3", HUMAN_TEST3)):
        leaked = [t for t, _ in S if normalize(t) in train]
        # HUMAN_TEST and HUMAN_TEST2 are development sets (their failures were read and fixed, so some phrases are now in training); HUMAN_TEST3 is strict
        assert len(leaked) <= {"HUMAN_TEST": 10, "HUMAN_TEST2": 8, "HUMAN_TEST3": 2}[name], (name, leaked)      # a few one-word replies ('evet') and glossary terms are unavoidable; the rest must be unseen
        routed = [(bot.route(t)[0], y) for t, y in S]
        assert sum(r == y for r, y in routed) / len(routed) >= 0.80, name
        assert sum(r not in (y, "clarify") for r, y in routed) / len(routed) <= 0.10, name
    reactions = {"affirm", "deny", "confused", "doubt", "complaint", "repeat", "elaborate"}
    rs = [(bot.route(t)[0], y) for t, y in HUMAN_TEST + HUMAN_TEST2 + HUMAN_TEST3 if y in reactions]
    assert sum(r == y for r, y in rs) / len(rs) >= 0.85    # the reactions to the assistant itself


def test_reaction_patterns_do_not_swallow_real_questions(nlu):
    """'no', 'ok' or 'I do not understand' at the start of a real question must not turn it into a reaction."""
    bot = Assistant(None, {}, nlu)
    for text, want in [("no sun at night right", "sun_info"), ("tamam ama puan neden dusuk", "why_score"), ("dte ne demek", "glossary"), ("hocam su puan neden bu kadar dusuk cikti anlamadim", "why_score"),
                       ("yes but how can we shift the date", "best_date"), ("evet ama tarihi 5 aralik yap", "change_context"), ("bana bir masal yazar misin", "unknown")]:
        assert bot.route(text)[0] in (want, "clarify"), (text, bot.route(text)[0])
    assert bot.route("tamam")[0] == "affirm" and bot.route("hayir")[0] == "deny" and bot.route("say that again")[0] == "repeat"


def test_yes_takes_up_the_offer_and_no_drops_it(bot):
    offer = bot.respond("how can we shift the date", CTX, {})
    sug = offer["memory"].get("suggestion")
    assert sug and sug["type"] == "apply_scenario"
    yes = bot.respond("evet olur", CTX, offer["memory"])
    assert yes["intent"] == "affirm" and yes["reply"]["actions"][0]["date"] == sug["date"]
    no = bot.respond("no thanks", CTX, offer["memory"])
    assert no["intent"] == "deny" and no["memory"]["suggestion"] is None and not no["reply"].get("actions")
    after_no = bot.respond("yes", CTX, no["memory"])                     # nothing on offer any more: it asks, it does not invent
    assert after_no["intent"] == "affirm" and not after_no["reply"].get("actions")


def test_confusion_gets_the_same_result_in_plain_words(bot):
    first = bot.respond("summarize the result", CTX, {})
    out = bot.respond("anlamadim ya", CTX, first["memory"])
    assert out["intent"] == "confused"
    sc = Scenario(CTX["scenarios"][0]["date"], CTX["scenarios"][0]["vehicleId"], CTX["scenarios"][0]["lat"], CTX["scenarios"][0]["lon"])
    idx = tools.assess(bot.dem, bot.profiles[sc.vehicle_id], sc)[1]["vehicle_index"]
    text = " ".join(b["text"] for b in out["reply"]["blocks"] if isinstance(b, dict) and b.get("type") == "text")
    assert f"{idx:.0f}" in text and "Kısaca" in text and "DTE" not in text and "LOLA" not in text
    two = bot.respond("compare the two results", CTX_AB, {})
    simple = bot.respond("explain it simpler please", CTX_AB, two["memory"])
    assert simple["intent"] == "confused" and "A" in simple["reply"]["blocks"][0]["text"]


def test_doubt_is_answered_with_the_sources_not_with_defensiveness(bot):
    out = bot.respond("are you sure about that", CTX, {})
    assert out["intent"] == "doubt"
    blocks = out["reply"]["blocks"]
    assert any(isinstance(b, dict) and b.get("type") == "table" for b in blocks)       # the inputs behind this very result
    assert "SPICE" in blocks[0]["text"] and "guess" in blocks[0]["text"]
    bare = bot.respond("are you sure about that", {}, {})                              # nothing on screen: still an honest answer
    assert bare["intent"] == "doubt" and "SPICE" in bare["reply"]["blocks"][0]["text"]


def test_anger_gets_an_apology_and_a_way_forward(bot):
    for msg, ctx in (("you are useless", CTX), ("cok kotusun", {}), ("this is garbage", CTX_AB)):
        out = bot.respond(msg, ctx, {})
        assert out["intent"] == "complaint"
        text = out["reply"]["blocks"][0]["text"].lower()
        assert ("sorry" in text or "üzgünüm" in text) and any(isinstance(b, dict) and b.get("type") == "chips" for b in out["reply"]["blocks"])


def test_repeat_and_go_on_follow_the_conversation(bot):
    first = bot.respond("summarize the result", CTX, {})
    again = bot.respond("say that again", CTX, first["memory"])
    assert again["intent"] == "repeat"
    table = next(b for b in first["reply"]["blocks"] if isinstance(b, dict) and b.get("type") == "table")
    assert table in again["reply"]["blocks"]
    nothing = bot.respond("say that again", CTX, {})
    assert nothing["intent"] == "repeat" and "nothing to repeat" in nothing["reply"]["blocks"][0]["text"]
    why = bot.respond("why is the score low", CTX, {})
    more = bot.respond("tell me more", CTX, why["memory"])
    assert more["intent"] == "elaborate" and more["reply"]["blocks"] and more["memory"]["topic_intent"] == "improve"


def test_reactions_do_not_overwrite_what_the_conversation_is_about(bot):
    first = bot.respond("why is the score low", CTX, {})
    thanks = bot.respond("thanks a lot", CTX, first["memory"])
    assert thanks["memory"]["topic_intent"] == "why_score" and thanks["memory"]["last_blocks"] == first["memory"]["last_blocks"]


# ---------------------------------------------------------------- free chat: no analysis on screen ----
FREE = {"free": True}


def test_free_chat_answers_what_to_send_on_a_date_for_every_site(bot):
    for msg in ("Which vehicle should we send on September 12?", "12 eylülde ne gönderelim", "what should we send on 12 september"):
        out = bot.respond(msg, FREE, {}, today=TODAY)
        assert out["intent"] == "best_vehicle", msg
        table = next(b for b in out["reply"]["blocks"] if isinstance(b, dict) and b.get("type") == "table")
        assert len(table["rows"]) >= 3 and "2027" in table["columns"][0] + str(out["reply"]["blocks"][0])
        head = out["reply"]["blocks"][0]
        assert (head["text"] if isinstance(head, dict) else head).count("**") >= 4
    best = table["rows"][0]
    sc = Scenario("2027-09-12", next(i for i, p in bot.profiles.items() if p["name"] == best[1]), *tools_site(best[0]))
    _, a = tools.assess(bot.dem, bot.profiles[sc.vehicle_id], sc)
    assert f"{a['vehicle_index']:.0f}" == best[3]                               # the table's numbers are the engine's: the score
    assert best[2].rstrip(" *") == f"{100 * a['risk']['p_success']:.0f}%"       # and the simulated success chance


def tools_site(name):
    from computation_core.assistant.entities import SITES
    return SITES[name.lower()]


def test_free_chat_remembers_what_was_said_and_states_its_assumptions(bot):
    first = bot.respond("how does shackleton look on march 3", FREE, {}, today=TODAY)
    text = " ".join(b["text"] for b in first["reply"]["blocks"] if isinstance(b, dict) and b.get("type") == "text")
    assert "Shackleton" in text and "3 Mar 2027" in text and "assumed" in text      # no vehicle was named: it says which one it took
    memory = first["memory"]["free"]
    assert memory["site"] == "Shackleton" and memory["date"] == "2027-03-03"
    follow = bot.respond("why is the score low", FREE, first["memory"], today=TODAY)          # still Shackleton, still March 3
    assert follow["memory"]["free"]["date"] == "2027-03-03" and follow["intent"] == "why_score"


def test_free_chat_never_demands_an_analysis_and_greets_properly(bot):
    out = bot.respond("hello", FREE, {}, today=TODAY)
    assert out["intent"] == "greet" and "September 12" in str(out["reply"]["blocks"])
    assert "Run an analysis first" not in str(bot.respond("which vehicle is best", FREE, {}, today=TODAY)["reply"])


# ---------------------------------------------------------------- the simple level ----
SIMPLE = {**CTX, "level": "simple"}


def _text(out):
    return " ".join(b["text"] for b in out["reply"]["blocks"] if isinstance(b, dict) and b.get("type") == "text")


def test_simple_level_gives_the_plain_summary_and_advanced_is_untouched(bot):
    adv = bot.respond("summarize the result", CTX, {})
    simple = bot.respond("summarize the result", SIMPLE, {})
    assert any(isinstance(b, dict) and b.get("type") == "table" for b in adv["reply"]["blocks"])          # advanced: the table is still there
    assert not any(isinstance(b, dict) and b.get("type") == "table" for b in simple["reply"]["blocks"])    # simple: no table to decode
    sc = Scenario(CTX["scenarios"][0]["date"], CTX["scenarios"][0]["vehicleId"], CTX["scenarios"][0]["lat"], CTX["scenarios"][0]["lon"])
    idx = tools.assess(bot.dem, bot.profiles[sc.vehicle_id], sc)[1]["vehicle_index"]
    assert f"{idx:.0f} out of 100" in _text(simple) and "Time in sunlight" not in _text(simple)           # the same number, in words


def test_simple_level_trims_technical_clutter(bot):
    out = bot.respond("will it freeze at night", SIMPLE, {})
    t = _text(out)
    assert "UTC" not in t and "assumption" not in t and not _re.search(r"\b\d{3} h\b", t)                   # hours become days, no timestamps
    why = bot.respond("why is the score low", SIMPLE, {})
    assert len([b for b in why["reply"]["blocks"] if isinstance(b, dict) and b.get("type") == "text"]) <= 2
    nodes = bot.respond("how can we shift the date", SIMPLE, {})
    assert not any("noticed on my own" in str(b) for b in nodes["reply"]["blocks"])


def test_simple_level_leaves_conversation_and_lists_alone(bot):
    assert bot.respond("hello", SIMPLE, {})["intent"] == "greet"
    two = bot.respond("compare the two results", {**CTX_AB, "level": "simple"}, {})
    assert any(isinstance(b, dict) and b.get("type") == "table" for b in two["reply"]["blocks"])        # a comparison keeps its (short) table


def test_plain_words_replaces_jargon_and_converts_hours():
    from computation_core.assistant.dialog import plain_words
    assert plain_words("DTE is 66% and the night is 268 h long [input unverified: user dataset]") == "direct link to Earth is 66% and the night is about 11 days long"
    assert plain_words("a 30 h outage") == "a 30 h outage"                                              # short spans stay in hours
    assert plain_words("268 saat", "tr") == "yaklaşık 11 gün"


# ---------------------------------------------------------------- the whole Moon ----
def test_assistant_answers_anywhere_on_the_moon(bot):
    apollo11 = {"scenarios": [{"date": "2026-12-01", "vehicleId": "chandrayaan-3-vikram-pragyan", "lat": 0.674, "lon": 23.473}]}
    for msg in ("summarize the result", "why is the score low", "how can we shift the date", "which vehicle works best here", "is there a better location", "how steep is the ground", "what about shackleton"):
        out = bot.respond(msg, apollo11, {}, today=TODAY)
        assert out["intent"] != "unknown" and out["reply"]["blocks"], msg


# ---------------------------------------------------------------- whole mission plans, two questions at once ----
FREE = {"free": True, "scenarios": []}


def test_month_names_suffixed_places_and_the_region_are_read():
    e = extract("aralikta malapert icin en iyi tarih", TODAY)
    assert e.dates == ["2026-12-01"] and e.months == ["2026-12-01"] and e.sites == ["malapert"]
    assert extract("kral 5 ocakta shackletona hangi arac gitsin", TODAY).sites == ["shackleton"]
    assert extract("faustiniye gitmek", TODAY).sites == ["faustini"]
    assert extract("16 kasım 2026 tarihinde ayın güney kutbuna", TODAY).region == "south_pole"
    assert extract("may i ask something", TODAY).dates == []  # 'may' is a month only with a day or a year


def test_the_questions_users_really_typed_are_understood(nlu):
    from computation_core.assistant.eval_set import HUMAN_TEST4, HUMAN_TEST5
    from computation_core.assistant.chips import route_chip
    bot = Assistant(None, {}, nlu)
    for name, S, floor in (("HUMAN_TEST4", HUMAN_TEST4, 0.9), ("HUMAN_TEST5", HUMAN_TEST5, 0.75)):
        got = [route_chip(t) or bot.route(t)[0] for t, _ in S]
        right = sum(g == y for g, (_, y) in zip(got, S))
        assert right / len(S) >= floor, (name, right / len(S), [(t, g) for g, (t, y) in zip(got, S) if g != y])


def test_a_whole_plan_in_one_message_gets_the_vehicle_and_the_requirements(bot):
    out = bot.respond("28 kasım 2026 tarihinde ayın güney kutbuna araç yollamak istiyorum. hangi araç uygundur veya hangi özelliklere sahip olmalıdır,", FREE, {}, TODAY)
    assert out["intent"] == "best_vehicle"
    t = _text(out) + " ".join(i for b in _blocks(out, "bullets") for i in b["items"])
    assert "28 Kas 2026" in t and "başarı şansı" in t.lower()           # the date it was given, and the simulated odds
    assert "Sorunun ikinci kısmı" in t and "Gece:" in t and "Eğim:" in t  # the second question is answered too
    assert out["memory"]["free"]["date"] == "2026-11-28"


def test_requirements_are_numbers_from_the_engine(bot):
    out = bot.respond("what does a lander need to survive at shackleton in january", FREE, {}, TODAY)
    assert out["intent"] == "requirements"
    t = _text(out) + " ".join(i for b in _blocks(out, "bullets") for i in b["items"])
    assert "Shackleton" in t and _re.search(r"longest darkness is \*\*\d+ h\*\*", t) and _re.search(r"at least \*\*\d+°\*\*", t)
    table = _blocks(out, "table")[0]
    assert table["rows"] and all(r[1].rstrip(" *").endswith("%") for r in table["rows"])


def test_when_should_i_go_and_where_is_the_most_light(bot):
    out = bot.respond("Shackleton kraterine ne zaman gitmeliyim", FREE, {}, TODAY)
    assert out["intent"] == "best_date" and "Shackleton" in str(out["reply"]["blocks"])  # it says which place it scanned
    month = bot.respond("aralıkta malapert için en iyi tarih", FREE, {}, TODAY)
    table = _blocks(month, "table")[0]
    assert all("Ara" in r[0] for r in table["rows"])  # 'in December' keeps the suggestions in December
    light = bot.respond("güney kutbunda en çok güneş alan yer neresi", FREE, {}, TODAY)
    rows = _blocks(light, "table")[0]["rows"]
    suns = [int(r[2].rstrip("%")) for r in rows]
    assert light["intent"] == "best_site" and suns == sorted(suns, reverse=True)


def test_what_the_user_said_survives_a_message_it_did_not_understand(bot):
    mem = bot.respond("16 aralikta haworth icin hmm", FREE, {}, TODAY)["memory"]
    assert mem["free"]["date"] == "2026-12-16" and mem["free"]["site"] == "Haworth"


def test_the_assistant_quotes_the_same_call_and_odds_as_the_page(bot):
    out = _say(bot, "summarize the result")
    t = _text(out)
    assert _re.search(r"Call: \*\*(Go: recommended|Possible, with risks|Not recommended as it is|Do not proceed)\*\*", t)
    assert _re.search(r"In \*\*\d+%\*\* of 400 simulated missions", t)
