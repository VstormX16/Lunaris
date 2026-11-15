"""Build frontend/src/data/spacecrafts.json, the garage.

Two sources, laid one over the other:
1. `spacecrafts_dataset_2026-11-14.json`: the earlier dataset (43 columns per craft, as given, before our corrections).
2. "Uzay Araç Güncel Bilgileri.txt" (2026-11-15): one block per craft with 16 "- Label: value" lines. Where it has
   a field, it replaces the earlier one; fields it does not have (wheels, legs, thrust per engine, ...) stay.
Then every correction checked against NASA / Wikipedia (`spacecraft_checks.json`) is applied on top: a verified figure
is never replaced by the dataset. Values keep the source's language and wording; nothing is translated or rounded.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "Uzay Araç Güncel Bilgileri.txt"
BASE = ROOT / "data_engine" / "spacecrafts_dataset_2026-11-14.json"
OUT = ROOT / "frontend" / "src" / "data" / "spacecrafts.json"
CHECKS = ROOT / "data_engine" / "spacecraft_checks.json"  # what was compared with NASA / Wikipedia, and every correction made

LABELS = {
    "Araç Tipi": "type",
    "Başarı Durumu": "outcome",
    "Mass (Kütle)": "mass",
    "Kütle Tanımı": "massNote",
    "Boyutlar": "dimensions",
    "Battery (Batarya)": "battery",
    "Solar Array (Güç Üretimi)": "solarW",
    "Survival Load (Bekleme / Güvenlik Yükü)": "survivalW",
    "Payload Load (Bilimsel Yük Gücü)": "payloadW",
    "Slope Limit (Eğim Sınırı)": "slopeLimit",
    "Radioisotope Heat (Termal Isı Gücü)": "heatSource",
    "İtki Sistemi": "propulsion",
    "Bilgisayar / Yazılım": "computer",
    "Bilimsel Cihazlar": "instruments",
    "Haberleşme": "comms",
    "Kaynak URL": "datasetUrl",
}


def parse(text: str):
    """The blocks of the 2026-11-15 file: '<no>. GÖREV / ARAÇ: <name> (<year>, <country>)' then '- Label: value' lines."""
    rows = []
    for block in re.split(r"\n(?=\d+\. GÖREV / ARAÇ: )", text):
        head = re.match(r"(\d+)\. GÖREV / ARAÇ: (.+) \((\d{4}), (.+?)\)\s*$", block.splitlines()[0])
        if not head:
            continue  # the title banner
        r = {"no": int(head[1]), "name": head[2], "year": int(head[3]), "country": head[4]}
        for line in block.splitlines()[1:]:
            m = re.match(r"- (.+?): (.*)$", line)
            if m:
                assert m[1] in LABELS, f"craft {r['no']}: unknown field {m[1]!r}"
                r[LABELS[m[1]]] = m[2].strip()
        missing = set(LABELS.values()) - set(r)
        assert not missing, f"craft {r['no']}: missing {sorted(missing)}"
        rows.append(r)
    return rows


def _number(text: str) -> str:
    """'1538.0 kg' -> '1538', '385.6 kg' -> '385.6' (the earlier dataset's style)."""
    v = float(re.match(r"\s*(\d+(?:\.\d+)?)", text)[1])
    return str(int(v)) if v.is_integer() else str(v)


def _lead(text: str):
    m = re.match(r"\s*(\d+(?:\.\d+)?)", text)
    return float(m[1]) if m else None


def overlay(base: dict, new: dict) -> dict:
    """The new file's fields on top of the earlier row."""
    r = dict(base)
    r["type"], r["outcome"], r["massNote"] = new["type"], new["outcome"], new["massNote"]
    r["massKg"] = _number(new["mass"])
    r["dimensions"] = new["dimensions"]
    # battery: 'type · N adet · V · capacity', the last part may carry extras ('+ ALSEP SNAP-27 RTG')
    parts = [p.strip() for p in new["battery"].split("·")]
    r["batteryType"] = parts[0]
    for p in parts[1:]:
        if re.fullmatch(r"\d+ adet", p):
            r["batteryCount"] = p.split()[0]
        elif re.fullmatch(r"\d+(?:\.\d+)? V", p):
            r["voltage"] = p.split()[0]
        else:
            r["batteryCapacity"] = p
    for k in ("solarW", "survivalW", "payloadW", "slopeLimit", "heatSource", "propulsion", "computer", "instruments", "datasetUrl"):
        r[k] = new[k]
    m = re.match(r"(.+?)\s*\((.+)\)\s*$", new["comms"])
    r["band"], r["dataRate"] = (m[1], m[2]) if m else (new["comms"], r.get("dataRate", ""))
    return r


def build_rows():
    """The garage rows: earlier dataset + the new file + every recorded correction (nothing is written)."""
    base = {b["no"]: b for b in json.loads(BASE.read_text(encoding="utf-8"))}
    new = {n["no"]: n for n in parse(SOURCE.read_text(encoding="utf-8-sig"))}
    assert set(base) == set(new), "the new file must list the same craft as before"
    checks = json.loads(CHECKS.read_text(encoding="utf-8"))
    old_ids = {o["no"]: o["id"] for o in json.loads(OUT.read_text(encoding="utf-8"))}  # keep ids: the photos are named by them
    rows = []
    for no in sorted(base):
        b, n = base[no], new[no]
        assert b["name"].split(" (")[0].split(" &")[0][:6] == n["name"].split(" (")[0].split(" &")[0][:6], (b["name"], n["name"])
        r = overlay(b, n)
        r["id"] = old_ids[no]
        chk = checks[r["id"]]
        for c in chk["changes"]:
            # a checked figure wins; the dataset may still carry the value we corrected, or already the right one
            now = r[c["field"]]
            same = lambda a, b: a == b or (_lead(a) is not None and _lead(a) == _lead(b))  # noqa: E731  '650 W (3 panels)' is the corrected '650 (450 W nominal)'
            assert same(now, c["from"]) or same(now, c["to"]), f'{r["id"]}.{c["field"]}: dataset now has {now!r}; check it against {c["to"]!r}'
            r[c["field"]] = c["to"]
        r["sourceUrl"] = chk["sourceUrl"]  # the link that was opened and checked (the dataset's own link is kept as datasetUrl)
        r["checkLevel"] = chk["level"]  # nasa | wikipedia | unchecked
        r["checkNote"] = chk["note"]
        r["corrections"] = chk["changes"]
        rows.append(r)
    return rows


def main():
    rows = build_rows()
    OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(rows)} spacecraft written to {OUT}")


if __name__ == "__main__":
    main()
