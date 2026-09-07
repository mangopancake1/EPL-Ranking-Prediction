"""
Real EFL Cup / UCL / UEL / UECL match dates, from jadwal_eflcup_ucl_uel_uecl.md
(user-sourced) -- replaces the generic, same-week-for-every-competition guess
config.EUROPEAN_WEEKS used before.

    python -m src.parse_european_schedule

Writes data/european_fixtures_2026_27.json: {club: [ISO dates]}, real dates
only for the clubs that actually appear in the source (our 9 European
entrants for UCL/UEL/UECL, whichever of the 20 reached EFL Cup round 2/3).
Everyone else falls back to config.EUROPEAN_WEEKS in features.py, same as
before this file existed.
"""

from __future__ import annotations

import re
from collections import defaultdict

import pandas as pd

import config

_ALIASES = {
    "Coventry": "Coventry City", "Hull": "Hull City", "Leeds": "Leeds United",
    "Man City": "Manchester City", "Man Utd": "Manchester Utd",
    "Newcastle": "Newcastle Utd", "Nottm Forest": "Nott'ham Forest",
    "Spurs": "Tottenham", "Palace": "Crystal Palace",
}
_OUR_20 = set(config.TEAMS)

_MONTH_ABBR = {m: i for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], start=1)}
_MONTH_FULL = {m: i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July", "August",
     "September", "October", "November", "December"], start=1)}
_MONTHS = {**_MONTH_ABBR, "Sept": 9, **_MONTH_FULL}

_ROW = re.compile(r"^\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|$", re.M)


def _parse_date(s: str) -> pd.Timestamp:
    """Handles all four date formats this file mixes across its sections."""
    m = re.match(r"^\w+,\s*(\d+)\s+(\w+)$", s)  # 'Wed, 26 Aug'
    if m:
        day, mon = int(m.group(1)), _MONTHS[m.group(2)]
        return pd.Timestamp(year=2026 if mon >= 7 else 2027, month=mon, day=day)
    m = re.match(r"^(\d+)\s+(\w+)\s+(\d{2})$", s)  # '11 Dec 26'
    if m:
        day, mon, yy = int(m.group(1)), _MONTHS[m.group(2)], int(m.group(3))
        return pd.Timestamp(year=2000 + yy, month=mon, day=day)
    m = re.match(r"^\w+\s+(\d+)\s+(\w+)\s+(\d{4})$", s)  # 'Wednesday 16 September 2026'
    if m:
        day, mon, yyyy = int(m.group(1)), _MONTHS[m.group(2)], int(m.group(3))
        return pd.Timestamp(year=yyyy, month=mon, day=day)
    m = re.match(r"^(\d+)\s+(\w+)$", s)  # '13 Oct'
    day, mon = int(m.group(1)), _MONTHS[m.group(2)]
    return pd.Timestamp(year=2026 if mon >= 7 else 2027, month=mon, day=day)


def parse(path=None) -> dict[str, list[str]]:
    path = path or (config.SCRAPED / "jadwal_eflcup_ucl_uel_uecl.md")
    text = path.read_text(encoding="utf-8")

    dates: dict[str, set] = defaultdict(set)
    for date_str, home, away in _ROW.findall(text):
        if date_str in ("Tanggal",) or set(date_str) <= {"-"}:
            continue
        try:
            d = _parse_date(date_str)
        except (AttributeError, KeyError):
            continue
        for raw in (home, away):
            club = _ALIASES.get(raw, raw)
            if club in _OUR_20:
                dates[club].add(d)

    return {club: sorted(d.isoformat() for d in ds) for club, ds in dates.items()}


def demo() -> None:
    out = parse()
    assert out, "expected at least one of our 20 clubs to appear in the schedule"
    # Every European entrant + everyone who plays a September EFL Cup round
    # must be covered -- a club present in the source file but missing here
    # means an alias or date format silently failed.
    for club in config.EUROPEAN_COMPETITION:
        assert club in out, f"{club} plays continental football and should be here"
    print(f"{len(out)} clubs with real fixture dates")
    for club in sorted(out, key=lambda c: -len(out[c]))[:5]:
        print(f"  {club:18s} {len(out[club])} tanggal, contoh {out[club][:2]}")


if __name__ == "__main__":
    import json

    result = parse()
    out_path = config.DATA / "european_fixtures_2026_27.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    demo()
    print(f"written to {out_path}")
