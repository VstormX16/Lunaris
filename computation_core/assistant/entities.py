"""Pulls dates, vehicles, sites, coordinates and multipliers out of a message (Turkish or English) with plain rules.

Why rules and not a network: these are closed vocabularies (month names, the project's 27 vehicles, the known sites), where a
lookup is exact and a learned tagger would only add mistakes. The intent model sees the message with these replaced by
placeholder words (mask), so '12 aralik yerine 3 ocak' and 'change the date to december 5' do not look alike to it.
"""
import json
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

from .text import normalize

ROOT = Path(__file__).resolve().parents[2]
MONTHS = {"ocak": 1, "subat": 2, "mart": 3, "nisan": 4, "mayis": 5, "haziran": 6, "temmuz": 7, "agustos": 8, "eylul": 9, "ekim": 10,
          "kasim": 11, "aralik": 12, "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6, "july": 7, "august": 8,
          "september": 9, "october": 10, "november": 11, "december": 12, "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7,
          "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12}
_MONTH_RE = "|".join(sorted(MONTHS, key=len, reverse=True))
# Turkish suffixes on a month or number ("aralikta", "5'inde") are folded away by matching a trailing word-ish tail
_TAIL = r"[a-z]{0,6}"
SITES = {"shackleton": (-89.54, 0.0), "haworth": (-87.45, -4.5), "malapert": (-85.987, 2.08), "faustini": (-87.3, 84.0), "nobile": (-85.2, 53.5)}
NUMBER_WORDS = {"bir": 1, "iki": 2, "uc": 3, "dort": 4, "bes": 5, "alti": 6, "yedi": 7, "sekiz": 8, "dokuz": 9, "on": 10, "one": 1, "two": 2,
                "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "twice": 2, "double": 2, "triple": 3}


@dataclass
class Entities:
    dates: list = field(default_factory=list)       # ISO strings, or {"shift_days": n} for "a week later"
    vehicles: list = field(default_factory=list)    # profile / spacecraft ids
    sites: list = field(default_factory=list)       # names from SITES
    coords: list = field(default_factory=list)      # (lat, lon)
    multiplier: float | None = None                 # "twice the battery"
    region: str | None = None                       # "south_pole": the user named the region, not one place in it
    months: list = field(default_factory=list)      # the dates that stand for a whole month ("aralikta" -> its first day)
    masked: str = ""


def _aliases():
    """(regex, craft id) for every spacecraft; the short name before '&' or '(' is also an alias."""
    crafts = json.loads((ROOT / "frontend" / "src" / "data" / "spacecrafts.json").read_text(encoding="utf-8"))
    out = []
    for c in crafts:
        names = {c["name"]}
        short = re.split(r"\s*[&(]", c["name"])[0].strip()
        names.add(short)
        names.add(c["id"].replace("-", " "))
        if c["id"].startswith("im-"):
            names.add(c["id"].split("-odysseus")[0].split("-athena")[0].replace("-", " "))
        for n in names:
            w = normalize(n.replace("'", " ")).split()
            if not w:
                continue
            out.append((re.compile(r"(?<![a-z0-9])" + r"[\s\-]*".join(map(re.escape, w)) + r"(?![a-z0-9])"), c["id"], len(" ".join(w))))
    out.sort(key=lambda a: -a[2])  # longest alias first: 'apollo 15' before 'apollo 1'
    return out


_ALIASES = None


def _norm_keep_digits(text: str) -> str:
    return normalize(text.replace("'", " "))


def _parse_dates(t: str, today: date):
    """Dates found in normalized text t -> (list, spans, months). Year defaults to the year of `today`; a bare month/day never
    lands in the past. `months` are the dates that stand for a whole month ('aralikta')."""
    found, spans, months = [], [], []

    def add(m, value):
        found.append((m.start(), value))
        spans.append((m.start(), m.end()))

    for m in re.finditer(r"(?<!\d)(20\d\d)-(\d{1,2})-(\d{1,2})(?!\d)", t):
        try:
            add(m, date(int(m[1]), int(m[2]), int(m[3])).isoformat())
        except ValueError:
            pass
    for m in re.finditer(r"(?<![\d.])(\d{1,2})[./](\d{1,2})[./](20\d\d)(?!\d)", t):
        try:
            add(m, date(int(m[3]), int(m[2]), int(m[1])).isoformat())
        except ValueError:
            pass
    pat_dm = rf"(?<![\d.])(\d{{1,2}})(?:st|nd|rd|th)?\s*(?:of\s+)?({_MONTH_RE}){_TAIL}\b(?:\s+(20\d\d))?"
    pat_md = rf"\b({_MONTH_RE}){_TAIL}\s+(\d{{1,2}})(?:st|nd|rd|th)?\b(?:\s+(20\d\d))?"
    for pat, day_first in ((pat_dm, True), (pat_md, False)):
        for m in re.finditer(pat, t):
            if any(a <= m.start() < b for a, b in spans):
                continue
            d, mon = (int(m[1]), MONTHS[m[2]]) if day_first else (int(m[2]), MONTHS[m[1]])
            y = int(m[3]) if m[3] else today.year
            try:
                v = date(y, mon, d)
            except ValueError:
                continue
            if not m[3] and v < today - timedelta(days=1):
                v = date(y + 1, mon, d)
            add(m, v.isoformat())
    # a month alone ("aralikta", "in december", "mart 2027de"): its first day, or today when it is this month; full names only,
    # so 'may i ask' or 'mar' are never dates
    full = "|".join(sorted((k for k in MONTHS if len(k) > 3), key=len, reverse=True))
    for m in re.finditer(rf"\b({full})(?:{_TAIL})\b(?:\s+(20\d\d)(?:[a-z]{{0,4}})?)?", t):
        if any(a <= m.start() < b for a, b in spans):
            continue
        mon = MONTHS[m[1]]
        y = int(m[2]) if m[2] else today.year
        v = date(y, mon, 1)
        if not m[2] and (y, mon) < (today.year, today.month):
            v = date(y + 1, mon, 1)
        add(m, max(v, today).isoformat() if (v.year, v.month) == (today.year, today.month) else v.isoformat())
        months.append(found[-1][1])
    rel = [(r"\b(?:bugun|today)\b", 0), (r"\b(?:yarin|tomorrow)\b", 1), (r"\b(?:gelecek hafta|next week)\b", 7), (r"\b(?:gelecek ay|next month)\b", 30),
           (r"\b(?:gecen hafta|last week)\b", -7), (r"\b(?:gecen ay|last month)\b", -30)]
    for pat, n in rel:
        for m in re.finditer(pat, t):
            add(m, {"shift_days": n, "from_today": True})
    unit = {"gun": 1, "day": 1, "days": 1, "hafta": 7, "week": 7, "weeks": 7, "ay": 30, "month": 30, "months": 30}
    for m in re.finditer(r"\b(\d{1,3}|" + "|".join(NUMBER_WORDS) + r")\s*(gun|days?|hafta|weeks?|ay|months?)\w*\s*(sonra|ileri|later|forward|ahead|after|geri|earlier|back|once|before)?", t):
        if any(a <= m.start() < b for a, b in spans):
            continue
        n = int(m[1]) if m[1].isdigit() else NUMBER_WORDS[m[1]]
        sign = -1 if m[3] in ("geri", "earlier", "back", "once", "before") else 1
        if m[3] or re.search(r"\b(kaydir|ilerle|ertele|shift|move|push|delay|oynat)", t):
            add(m, {"shift_days": sign * n * unit[m[2].rstrip("s") if m[2].rstrip("s") in unit else m[2]]})
    found.sort()
    return [v for _, v in found], spans, months


def extract(text: str, today: date | None = None) -> Entities:
    global _ALIASES
    today = today or datetime.utcnow().date()
    t = _norm_keep_digits(text)
    ent = Entities()
    masked = t
    spans = []
    ent.dates, dspans, ent.months = _parse_dates(t, today)
    spans += [(a, b, "xdatex") for a, b in dspans]
    if _ALIASES is None:
        _ALIASES = _aliases()
    taken = []
    for rx, cid, _ in _ALIASES:
        for m in rx.finditer(t):
            if any(a < m.end() and m.start() < b for a, b, _k in taken):
                continue
            taken.append((m.start(), m.end(), cid))
            if cid not in ent.vehicles:
                ent.vehicles.append(cid)
    spans += [(a, b, "xvehx") for a, b, _ in taken if not any(sa < b and a < sb for sa, sb, _k in spans)]
    for name in SITES:
        for m in re.finditer(rf"\b{name}(?:[a-z]{{0,6}})\b", t):  # Turkish suffixes: shackletona, malaperte, haworthun
            if name not in ent.sites:
                ent.sites.append(name)
            if not any(sa < m.end() and m.start() < sb for sa, sb, _k in spans):
                spans.append((m.start(), m.end(), "xsitex"))
    for m in re.finditer(r"(?<![\d.])((?:-|eksi |minus )?\d{1,2}(?:[.,]\d+)?)\s*(?:,|;|/|\s|ve|and)\s*((?:-|eksi |minus )?\d{1,3}(?:[.,]\d+)?)(?![\d.:/])", t):
        def num(s):
            neg = s.startswith(("-", "eksi", "minus"))
            return (-1 if neg else 1) * float(re.sub(r"[^\d.]", "", s.replace(",", ".")) or "nan")
        la, lo = num(m[1]), num(m[2])
        if -90 <= la <= 90 and -180 <= lo <= 360 and (m[1].startswith(("-", "eksi", "minus")) or "." in m[1] or "," in m[1] or abs(la) >= 60) \
                and not any(sa < m.end() and m.start() < sb for sa, sb, _k in spans):
            ent.coords.append((la, lo))
            spans.append((m.start(), m.end(), "xsitex"))
    mm = re.search(r"\b(\d+(?:[.,]\d+)?|" + "|".join(NUMBER_WORDS) + r")\s*(?:kat|x|times|fold)\b|\bx\s*(\d+)\b|\b(twice|double|triple)\b", t)
    if mm:
        v = mm[1] or mm[2] or mm[3]
        ent.multiplier = float(v.replace(",", ".")) if re.fullmatch(r"\d+(?:[.,]\d+)?", v) else float(NUMBER_WORDS[v])
        if not any(sa < mm.end() and mm.start() < sb for sa, sb, _k in spans):
            spans.append((mm.start(), mm.end(), "xnumx"))
    if re.search(r"\b(guney kutb|south(ern)? pol|kutup|kutb|kutupta|polar|the pole)", t):
        ent.region = "south_pole"  # the project's data covers the south pole; a plain 'kutup' / 'the pole' means it too
    for a, b, k in sorted(spans, reverse=True):
        masked = masked[:a] + f" {k} " + masked[b:]
    ent.masked = re.sub(r"\s+", " ", masked).strip()
    return ent
