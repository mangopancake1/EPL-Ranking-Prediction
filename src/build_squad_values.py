"""
Build data/squad_values.json from full 2026/27 squad lists joined with FPL's
own appearance history.

    python -m src.build_squad_values

Source: epl_squads_2026_27.md -- complete squads for all 20 clubs, pasted
manually from Transfermarkt (captcha-walled otherwise). This replaced an
earlier top-500-by-value list that only covered each established club's most
expensive names and gave every promoted club's full squad an unfair
advantage in any value comparison -- see project memory for the bugs that
caused (Hull priced at 11 points, Sunderland relegated in simulation).

FPL bootstrap-static + element-summary supplies epl_matches: real appearance
history back to 2021/22 for anyone who has played in the league. Guessing it
from a value list would contradict the project's "None over a guess" rule.
fpl_matched flags whether a name resolved to an FPL id at all, so a genuine
"never played here" (matched, epl_matches=0) stays distinguishable from an
unresolved name (unmatched, epl_matches=0 by default but unknown in truth).
"""

from __future__ import annotations

import re
import time
import unicodedata

import pandas as pd

import config
from src.fetch import get_json

SQUADS_FILE = config.SCRAPED / "epl_squads_2026_27.md"

# The markdown's club headers don't match config.TEAM_ALIASES verbatim
# (ampersands, "FC"/"AFC" suffixes in different places) -- a fixed 20-entry
# map is simpler and safer than making the alias matcher fuzzier.
_HEADER_TO_TEAM = {
    "Arsenal FC": "Arsenal",
    "Chelsea FC": "Chelsea",
    "Liverpool FC": "Liverpool",
    "Manchester City": "Manchester City",
    "Manchester United": "Manchester Utd",
    "Tottenham Hotspur": "Tottenham",
    "Newcastle United": "Newcastle Utd",
    "Aston Villa": "Aston Villa",
    "Brighton & Hove Albion": "Brighton",
    "Nottingham Forest": "Nott'ham Forest",
    "Brentford FC": "Brentford",
    "AFC Bournemouth": "Bournemouth",
    "Crystal Palace": "Crystal Palace",
    "Everton FC": "Everton",
    "Fulham FC": "Fulham",
    "Leeds United": "Leeds United",
    "Sunderland AFC": "Sunderland",
    "Coventry City": "Coventry City",
    "Ipswich Town": "Ipswich Town",
    "Hull City": "Hull City",
}

# Club headings have arrived in two shapes across pastes: "## 1\. Arsenal FC"
# (numbered, dot sometimes markdown-escaped) and "## Arsenal FC (26/27)"
# (season-suffixed). Tolerating both beats asking for a re-paste.
_HEADER_RE = re.compile(r"^##\s*(?:\d+\\?\.\s*)?(.+?)(?:\s*\(\d\d/\d\d\))?\s*$")

# Transfermarkt writes positions two ways across pastes: abbreviated ("CB") and
# spelled out ("Centre-Back"). Everything downstream -- config.POSITION_LINES,
# squad depth, the fatigue model -- speaks the abbreviated vocabulary, so the
# long form is normalised here rather than teaching each consumer both.
_POSITION_ALIASES = {
    "Goalkeeper": "GK",
    "Centre-Back": "CB", "Left-Back": "LB", "Right-Back": "RB",
    "Defensive Midfield": "DM", "Central Midfield": "CM",
    "Attacking Midfield": "AM", "Left Midfield": "LM", "Right Midfield": "RM",
    "Left Winger": "LW", "Right Winger": "RW",
    "Centre-Forward": "CF", "Second Striker": "SS",
}
_KNOWN_POSITIONS = set(_POSITION_ALIASES) | set(_POSITION_ALIASES.values())

_SEPARATOR_RE = re.compile(r"^[|\s:-]+$")
_VALUE_RE = re.compile(r"€\s*([\d.]+)\s*(m|k)?", re.IGNORECASE)


def _parse_value(cell: str) -> float | None:
    """"€100m" -> 100.0, "€500k" -> 0.5, "-" -> None (no fabricated value)."""
    m = _VALUE_RE.search(cell)
    if not m:
        return None
    num = float(m.group(1))
    return num / 1000.0 if (m.group(2) or "").lower() == "k" else num


_DOB_RE = re.compile(r"^(\d{2})/(\d{2})/(\d{4})(?:\s*\((\d+)\))?$")

# Columns worth failing the build over. A paste that renames or drops one of
# these should stop here, where the cause is readable, rather than surface as an
# empty chart three modules downstream.
_REQUIRED_COVERAGE = {"age": 0.95, "date_of_birth": 0.95, "market_value_m": 1.0}


def _clean(cell: str | None) -> str | None:
    """Transfermarkt writes an unknown as '-'. That is absence, not a value."""
    if cell is None:
        return None
    cell = cell.strip()
    return cell or None if set(cell) != {"-"} else None


def _parse_date(cell: str | None) -> str | None:
    """dd/mm/yyyy -> ISO. None when the cell is '-' (26 contracts have no date)."""
    cell = _clean(cell)
    if not cell:
        return None
    m = _DOB_RE.match(cell)
    return f"{m.group(3)}-{m.group(2)}-{m.group(1)}" if m else None


def _parse_dob(cell: str) -> tuple[str | None, int | None]:
    """'25/02/1999 (27)' -> ('1999-02-25', 27). Age is Transfermarkt's own."""
    m = _DOB_RE.match(_clean(cell) or "")
    if not m:
        return None, None
    age = int(m.group(4)) if m.group(4) else None
    return f"{m.group(3)}-{m.group(2)}-{m.group(1)}", age


def load_squads() -> pd.DataFrame:
    """Parse epl_squads_2026_27.md into one row per player."""
    rows = []
    club = None

    for line in SQUADS_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        header = _HEADER_RE.match(line)
        if header:
            # Markdown escaping again: "Brighton \& Hove Albion".
            club = _HEADER_TO_TEAM.get(header.group(1).replace("\\", ""))
            continue
        if club is None or not line.startswith("|"):
            continue
        # Column header and the ---|--- separator carry no players. Detected by
        # content rather than exact spelling: the paste writes them as
        # "|#|Pemain|..." and "|-|-|-|", not the spaced form markdown usually has.
        if _SEPARATOR_RE.match(line) or "Pemain" in line or "| Player |" in line:
            continue

        cells = [c.strip() for c in line.strip("|").split("|")]
        # Two table shapes share the same leading fields (#, player, position,
        # dob, nationality) and both put the market value last; the full
        # Transfermarkt paste just adds height/foot/joined/signed-from/contract
        # in between. Anything else is not a squad row.
        if len(cells) not in (6, 11):
            continue
        _, name, position, dob_cell, nat = cells[:5]
        value_cell = cells[-1]
        # The full Transfermarkt paste carries four more columns worth keeping.
        # The short shape does not, so they resolve to None rather than a guess.
        foot = joined = signed_from = contract_cell = None
        if len(cells) == 11:
            foot, joined, signed_from, contract_cell = cells[6], cells[7], cells[8], cells[9]
        # An unrecognised position must stop the build, not pass through. When
        # the paste switched to spelled-out positions every player silently
        # fell outside POSITION_LINES, every line came out worth 0m, and the
        # fatigue model produced NaN that only surfaced as "lam value too
        # large" deep inside the simulation. Fail where the cause is visible.
        if position not in _KNOWN_POSITIONS:
            raise ValueError(
                f"unknown position {position!r} for {name} ({club}) -- add it to "
                "_POSITION_ALIASES so squad depth keeps working"
            )
        position = _POSITION_ALIASES.get(position, position)
        value = _parse_value(value_cell)
        if not name or value is None:
            continue  # unpriced squad filler carries no signal either way

        dob, age = _parse_dob(dob_cell)
        rows.append(
            {
                "player": name,
                "club": club,
                "position": position,
                "nationality": nat,
                "market_value_m": value,
                "age": age,
                "date_of_birth": dob,
                "contract_until": _parse_date(contract_cell),
                "signed_from": _clean(signed_from),
                "foot": _clean(foot),
                "joined": _parse_date(joined),
            }
        )

    df = pd.DataFrame(rows)
    unmapped_clubs = set(_HEADER_TO_TEAM.values()) - set(df["club"])
    if unmapped_clubs:
        raise ValueError(f"no players parsed for: {unmapped_clubs}")

    for column, floor in _REQUIRED_COVERAGE.items():
        share = df[column].notna().mean()
        if share < floor:
            raise ValueError(
                f"{column} parsed for only {share:.0%} of {len(df)} rows "
                f"(need {floor:.0%}) -- the paste's column layout has changed"
            )
    return df


def _fpl_headers():
    return {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        )
    }


def fetch_fpl_bootstrap() -> dict:
    return get_json("https://fantasy.premierleague.com/api/bootstrap-static/", headers=_fpl_headers())


def fetch_fpl_history(player_id: int) -> dict:
    return get_json(
        f"https://fantasy.premierleague.com/api/element-summary/{player_id}/", headers=_fpl_headers()
    )


def resolve_epl_matches(player_ids: dict[str, int]) -> dict[str, tuple[int, int]]:
    """
    For matched players only, pull real history_past: (matches, minutes).

    Minutes used to be collapsed straight into a match count and discarded,
    leaving the pipeline to reconstruct them as `matches * 60`. The real figure
    was already in hand; it is now kept, because 90 minutes a week and 20
    minutes a week are not the same evidence of belonging in this league.
    """
    out, failed = {}, []
    for name, pid in player_ids.items():
        try:
            hist = fetch_fpl_history(pid)
        except Exception as exc:
            # A swallowed failure here is not harmless: the player keeps
            # epl_matches=0, which reads downstream as "never played in the
            # league" -- a genuine veteran (Rodrigo Muniz, ~1000 minutes last
            # season) turned into a false newcomer by a timed-out request.
            failed.append(f"{name} ({type(exc).__name__})")
            continue
        minutes = sum(season.get("minutes", 0) for season in hist.get("history_past", []))
        out[name] = (round(minutes / 90.0), int(minutes))
        time.sleep(config.REQUEST_DELAY["fpl"] / 3)

    if failed:
        print(f"  ! {len(failed)} history fetches failed: {', '.join(failed[:8])}"
              + (" ..." if len(failed) > 8 else ""))
        if len(failed) > 0.05 * len(player_ids):
            raise RuntimeError(
                f"{len(failed)} of {len(player_ids)} FPL history fetches failed "
                "-- too many to trust the newcomer/minutes columns; rerun with a warm cache"
            )
    return out


# FPL's own club names, where they differ from config.TEAMS.
_FPL_TEAM_TO_TEAM = {
    "Leeds": "Leeds United",
    "Man City": "Manchester City",
    "Man Utd": "Manchester Utd",
    "Newcastle": "Newcastle Utd",
    "Nott'm Forest": "Nott'ham Forest",
    "Spurs": "Tottenham",
}


def _norm(s: str) -> str:
    """Accent-free lowercase words, so 'Martín' and 'Martin' are the same key."""
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^a-z]+", " ", s.lower()).split())


def match_fpl_ids(squads: pd.DataFrame, elements: list[dict], teams: list[dict]) -> dict[str, dict]:
    """
    Resolve each squad name to an FPL player id.

    An exact `first_name + second_name` match -- what this used to do alone --
    fails for most of the league, because the two sources name people
    differently and neither is wrong:

      Transfermarkt      FPL first + second           why exact match fails
      David Raya         David Raya Martin            extra legal surname
      Martin Zubimendi   Martin Zubimendi Ibanez      extra legal surname
      Moises Caicedo     Moises Caicedo Corozo        extra legal surname
      Ben White          Benjamin White               legal vs common first name
      Gabriel            Gabriel dos Santos Magalhaes mononym vs full legal name

    That left 234 of 571 players unresolved and, worse, indistinguishable from
    genuine newcomers: every one of them was written out with epl_matches=0,
    so David Raya and Ben White were priced as if they had never played in the
    league.

    Matching runs club by club, which is what makes the looser rules safe --
    "White" is ambiguous league-wide but unique inside Arsenal. Each rule is
    tried in order and only accepted when it lands on exactly one player; an
    ambiguous hit is left unmatched rather than guessed, so a wrong id can
    never quietly become a wrong appearance count.
    """
    by_id = {t["id"]: _FPL_TEAM_TO_TEAM.get(t["name"], t["name"]) for t in teams}

    pool = []
    for e in elements:
        first, second, web = e["first_name"], e["second_name"], e["web_name"]
        pool.append({
            "id": e["id"],
            "club": by_id.get(e["team"]),
            "full": _norm(f"{first} {second}"),
            "web": _norm(web),
            "short": _norm(f"{first} {second.split()[0]}") if second.split() else _norm(first),
            "tokens": set(_norm(f"{first} {second}").split()),
            "code": e.get("code"),   # the player-photo key, not the element id
        })

    def candidates(name: str, club: str) -> list[dict]:
        n = _norm(name)
        tokens = set(n.split())
        last = n.split()[-1] if n else ""
        same_club = [p for p in pool if p["club"] == club]

        for rule in (
            lambda p: p["full"] == n,                       # exact, accent-insensitive
            lambda p: p["short"] == n,                      # David Raya <- David Raya Martin
            lambda p: p["full"].startswith(n + " "),        # Martin Zubimendi <- ... Ibanez
            lambda p: p["web"] == n,                        # Gabriel, Estevao
            lambda p: p["web"] == last and len(tokens) <= 3,  # Ben White <- Benjamin White
            lambda p: tokens and tokens <= p["tokens"],     # every word present somewhere
        ):
            hits = [p for p in same_club if rule(p)]
            if len(hits) == 1:
                return hits
            # Ambiguous, so this rule can't decide -- fall through to the next
            # one, which may still be unique. Arsenal's "Gabriel" is the case:
            # three squad members' names start with it, but only one has that
            # as their FPL web_name.
        # Last resort: an exact full-name hit anywhere, for a player whose club
        # moved between the squad paste and FPL's own registration.
        hits = [p for p in pool if p["full"] == n]
        return hits if len(hits) == 1 else []

    out = {}
    for row in squads.itertuples(index=False):
        hits = candidates(row.player, row.club)
        if hits:
            out[row.player] = hits[0]
    return out


def build(limit_lookup: int | None = None) -> dict:
    squads = load_squads()
    bootstrap = fetch_fpl_bootstrap()

    matched = match_fpl_ids(squads, bootstrap["elements"], bootstrap["teams"])
    matched_ids = {name: m["id"] for name, m in matched.items()}
    codes = {name: m["code"] for name, m in matched.items() if m.get("code")}
    print(f"  {len(squads)} players parsed across {squads['club'].nunique()} clubs")
    print(f"  {len(matched_ids)} of {len(squads)} matched to an FPL player id")

    if limit_lookup:
        matched_ids = dict(list(matched_ids.items())[:limit_lookup])

    history = resolve_epl_matches(matched_ids)
    squads["fpl_matched"] = squads["player"].isin(matched_ids)
    squads["fpl_code"] = squads["player"].map(codes)
    squads["epl_matches"] = squads["player"].map(
        {n: m for n, (m, _) in history.items()}).fillna(0).astype(int)
    squads["source_league"] = squads["epl_matches"].apply(
        lambda m: "Premier League" if m > 0 else "Unknown"
    )
    # Real minutes from FPL's own history, not matches * 60. A player who came
    # off the bench thirty times and one who started thirty are the same number
    # of appearances and very different evidence.
    squads["minutes_last_season"] = squads["player"].map(
        {n: mins for n, (_, mins) in history.items()}).fillna(0).astype(int)

    return {
        "meta": {
            "generated": config.TODAY,
            "source": "epl_squads_2026_27.md (manual Transfermarkt paste, full squads) + FPL API appearance history",
            "coverage": "All 20 clubs, full squads -- not filtered to a global value cutoff.",
            "n_players": len(squads),
        },
        "players": squads[
            ["player", "club", "position", "nationality", "market_value_m", "age",
             "date_of_birth", "contract_until", "signed_from", "foot", "joined",
             "epl_matches", "source_league", "minutes_last_season", "fpl_matched",
             "fpl_code"]
        ].to_dict(orient="records"),
    }


if __name__ == "__main__":
    import json

    result = build()
    out_path = config.SQUAD_VALUES_FILE
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    assert result["players"], "no players parsed"
    clubs_covered = {p["club"] for p in result["players"]}
    assert clubs_covered == set(config.TEAMS), f"missing clubs: {set(config.TEAMS) - clubs_covered}"
    for team in config.TEAMS:
        n = sum(1 for p in result["players"] if p["club"] == team)
        assert n >= 15, f"{team} has only {n} players -- suspiciously thin for a full squad"

    have = lambda k: sum(1 for p in result["players"] if p.get(k) is not None)
    print(f"  age {have('age')}  contract {have('contract_until')}  "
          f"fpl_code {have('fpl_code')}  of {len(result['players'])}")
    print(f"OK  {len(result['players'])} players across {len(clubs_covered)} clubs -> {out_path}")
