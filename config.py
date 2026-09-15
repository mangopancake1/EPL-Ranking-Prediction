"""
Central configuration for the EPL 2026/27 rank prediction model.

Every tunable lives here. Values marked ESTIMATE are placeholders that
backtest.py calibrates from data -- do not treat them as ground truth.
"""

from pathlib import Path

# ---------------------------------------------------------------- paths

ROOT = Path(__file__).parent
DATA = ROOT / "data"
# Hand-pasted source documents (Transfermarkt, FotMob, FootyStats). Kept apart
# from data/, which holds only files this pipeline generates.
SCRAPED = ROOT / "data_scraping"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
CACHE = DATA / "cache"
OUTPUT = ROOT / "output"

for _d in (DATA, RAW, PROCESSED, CACHE, OUTPUT):
    _d.mkdir(parents=True, exist_ok=True)

LEAGUE_WEIGHTS_FILE = DATA / "league_weights.json"
SQUAD_VALUES_FILE = DATA / "squad_values.json"
MANAGER_PRIORS_FILE = DATA / "manager_priors.json"
MATCHES_FILE = RAW / "matches_football_data.csv"
FIXTURES_FILE = RAW / "fixtures_2026_27.csv"

# ---------------------------------------------------------------- season

SEASON = "2026-2027"
TODAY = "2026-09-02"

# Seasons used to fit the match model (most recent last).
# football-data.org's free tier only serves the trailing ~3 seasons
# (2020-2022 return 403 on this token) -- 5 seasons was the original plan,
# 3 is what's actually available.
TRAIN_SEASONS = ["2023-2024", "2024-2025", "2025-2026"]

# Seasons held out for backtesting. For each, the model trains only on
# seasons strictly before it and predicts the final table. With only 3
# seasons of match data, 2023-2024 has no prior season to train from and
# is dropped -- backtesting starts where there's at least one season behind it.
BACKTEST_SEASONS = ["2024-2025", "2025-2026"]

FOOTBALL_DATA_SEASONS = {
    "2023-2024": 2023,
    "2024-2025": 2024,
    "2025-2026": 2025,
    "2026-2027": 2026,
}

# ---------------------------------------------------------------- teams

# 20 clubs, 2026/27. Reconciled against the manager table: the source doc's
# "existing clubs" list omitted Sunderland and Leeds.
PROMOTED = ["Coventry City", "Ipswich Town", "Hull City"]
RELEGATED_2025_26 = ["West Ham", "Wolves", "Burnley"]

TEAMS = [
    "Arsenal",
    "Aston Villa",
    "Bournemouth",
    "Brentford",
    "Brighton",
    "Chelsea",
    "Coventry City",
    "Crystal Palace",
    "Everton",
    "Fulham",
    "Hull City",
    "Ipswich Town",
    "Leeds United",
    "Liverpool",
    "Manchester City",
    "Manchester Utd",
    "Newcastle Utd",
    "Nott'ham Forest",
    "Sunderland",
    "Tottenham",
]

# FBref / Understat / Transfermarkt spell club names differently.
# Canonical name -> aliases seen in the wild.
TEAM_ALIASES = {
    "Arsenal": ["Arsenal", "Arsenal FC"],
    "Aston Villa": ["Aston Villa", "Aston Villa FC"],
    "Bournemouth": ["Bournemouth", "AFC Bournemouth"],
    "Brentford": ["Brentford", "Brentford FC"],
    "Brighton": ["Brighton", "Brighton & Hove Albion", "Brighton and Hove Albion"],
    "Chelsea": ["Chelsea", "Chelsea FC"],
    "Coventry City": ["Coventry City", "Coventry"],
    "Crystal Palace": ["Crystal Palace"],
    "Everton": ["Everton", "Everton FC"],
    "Fulham": ["Fulham", "Fulham FC"],
    "Hull City": ["Hull City", "Hull"],
    "Ipswich Town": ["Ipswich Town", "Ipswich"],
    "Leeds United": ["Leeds United", "Leeds"],
    "Liverpool": ["Liverpool", "Liverpool FC"],
    "Manchester City": ["Manchester City", "Man City"],
    "Manchester Utd": ["Manchester Utd", "Manchester United", "Man United", "Man Utd"],
    "Newcastle Utd": ["Newcastle Utd", "Newcastle United", "Newcastle"],
    "Nott'ham Forest": ["Nott'ham Forest", "Nottingham Forest", "Nottm Forest"],
    "Sunderland": ["Sunderland", "Sunderland AFC"],
    "Tottenham": ["Tottenham", "Tottenham Hotspur", "Spurs"],
}

# Clubs that appear in training-window match history but are not in the EPL
# for 2026/27 (relegated in a prior season, or top-flight only historically).
# Their matches still matter -- they are legitimate opponents in the training
# data -- so they get clean names without being added to TEAMS.
NON_2026_27_TEAM_ALIASES = {
    "West Ham": ["West Ham", "West Ham United"],
    "Wolves": ["Wolves", "Wolverhampton Wanderers"],
    "Burnley": ["Burnley"],
    "Southampton": ["Southampton"],
    "Leicester City": ["Leicester City", "Leicester"],
    "Luton Town": ["Luton Town", "Luton"],
    "Sheffield United": ["Sheffield United", "Sheffield Utd"],
}

# ---------------------------------------------------------------- managers

# prior_stints drive the manager prior. The scraper resolves each stint to
# ppg / xG / xGA at that club, then league_weight scales it to EPL terms.
MANAGERS = {
    "Arsenal": {
        "name": "Mikel Arteta",
        "status": "established",
        "since": "2019-12-20",
        "prior_stints": [],  # already has enough Arsenal data
    },
    "Aston Villa": {
        "name": "Unai Emery",
        "status": "established",
        "since": "2022-11-01",
        "prior_stints": [],
    },
    "Brighton": {
        "name": "Fabian Hurzeler",
        "status": "established",
        "since": "2024-07-01",
        "prior_stints": [
            {"club": "St. Pauli", "league": "2. Bundesliga", "start": 2022, "end": 2024},
        ],
    },
    "Brentford": {
        "name": "Keith Andrews",
        "status": "new",
        "since": "2025-07-01",
        "prior_stints": [
            # International management: no club league to scale against.
            {"club": "Republic of Ireland", "league": "International", "start": 2024, "end": 2025},
        ],
    },
    "Chelsea": {
        "name": "Xabi Alonso",
        "status": "new",
        "since": "2026-07-01",
        # Deliberately excludes Real Madrid (small sample, atypical context).
        "prior_stints": [
            {"club": "Bayer Leverkusen", "league": "Bundesliga", "start": 2022, "end": 2025},
        ],
    },
    "Crystal Palace": {
        "name": "Pierre Sage",
        "status": "new",
        "since": "2026-07-01",
        "prior_stints": [
            {"club": "Lyon", "league": "Ligue 1", "start": 2023, "end": 2025},
        ],
    },
    "Everton": {
        "name": "David Moyes",
        "status": "established",
        "since": "2025-01-01",
        "prior_stints": [
            {"club": "West Ham", "league": "Premier League", "start": 2019, "end": 2024},
        ],
    },
    "Fulham": {
        "name": "Alvaro Arbeloa",
        "status": "new",
        "since": "2026-07-01",
        "prior_stints": [
            {"club": "Real Madrid Castilla", "league": "Youth/B Team", "start": 2023, "end": 2026},
        ],
    },
    "Liverpool": {
        "name": "Andoni Iraola",
        "status": "new",
        "since": "2026-07-01",
        "prior_stints": [
            {"club": "Bournemouth", "league": "Premier League", "start": 2023, "end": 2026},
        ],
    },
    "Manchester City": {
        "name": "Enzo Maresca",
        "status": "new",
        "since": "2026-07-01",
        "prior_stints": [
            {"club": "Chelsea", "league": "Premier League", "start": 2024, "end": 2026},
            {"club": "Leicester City", "league": "Championship", "start": 2023, "end": 2024},
        ],
    },
    "Manchester Utd": {
        "name": "Michael Carrick",
        "status": "interim_to_full",
        "since": "2026-01-01",
        "prior_stints": [
            {"club": "Middlesbrough", "league": "Championship", "start": 2022, "end": 2025},
        ],
    },
    "Newcastle Utd": {
        "name": "Matthias Jaissle",
        "status": "new",
        "since": "2026-08-01",  # confirmed 1 Aug 2026, two days before model run
        "prior_stints": [
            {"club": "Red Bull Salzburg", "league": "Austrian Bundesliga", "start": 2021, "end": 2023},
            {"club": "Al-Ahli", "league": "Saudi Pro League", "start": 2023, "end": 2026},
        ],
    },
    "Tottenham": {
        "name": "Roberto De Zerbi",
        "status": "new",
        "since": "2026-07-01",
        "prior_stints": [
            {"club": "Marseille", "league": "Ligue 1", "start": 2024, "end": 2026},
            {"club": "Brighton", "league": "Premier League", "start": 2022, "end": 2024},
        ],
    },
    "Bournemouth": {
        "name": "Marco Rose",
        "status": "new",
        "since": "2026-07-01",
        "prior_stints": [
            {"club": "RB Leipzig", "league": "Bundesliga", "start": 2022, "end": 2025},
        ],
    },
    "Nott'ham Forest": {
        "name": "Oliver Glasner",
        "status": "new",
        "since": "2026-07-01",
        "prior_stints": [
            {"club": "Crystal Palace", "league": "Premier League", "start": 2024, "end": 2026},
        ],
    },
    "Sunderland": {
        "name": "Regis Le Bris",
        "status": "established",
        "since": "2024-07-01",
        "prior_stints": [
            {"club": "Lorient", "league": "Ligue 1", "start": 2022, "end": 2024},
        ],
    },
    "Leeds United": {
        "name": "Daniel Farke",
        "status": "established",
        "since": "2023-07-01",
        "prior_stints": [],
    },
    "Coventry City": {
        "name": "Frank Lampard",
        "status": "established",
        "since": "2023-11-01",
        "prior_stints": [
            {"club": "Everton", "league": "Premier League", "start": 2022, "end": 2023},
            {"club": "Chelsea", "league": "Premier League", "start": 2019, "end": 2021},
        ],
    },
    "Ipswich Town": {
        "name": "Gary O'Neil",
        "status": "new",
        "since": "2026-07-01",
        # Career order is Bournemouth -> Wolves -> Strasbourg (Jan-Jun 2026,
        # Ligue 1, finished 8th) -> Ipswich. Strasbourg is the most recent and
        # therefore most relevant stint, but no clean final W-D-L for it turned
        # up (only mid-season fragments: 5-1-2 through 8 games, 8-5-2 through
        # 15 -- they don't reconcile into a final tally). Left out rather than
        # estimated; add it if a final record surfaces.
        "prior_stints": [
            {"club": "Wolves", "league": "Premier League", "start": 2023, "end": 2024},
            {"club": "Bournemouth", "league": "Premier League", "start": 2022, "end": 2023},
        ],
    },
    "Hull City": {
        "name": "Sergej Jakirovic",
        "status": "new",
        "since": "2026-07-01",
        "prior_stints": [
            {"club": "Hull City", "league": "Championship", "start": 2025, "end": 2026},
        ],
    },
}

# ---------------------------------------------------------------- model params

# --- Dixon-Coles ---------------------------------------------------------

# Exponential time decay applied to each historical match.
#   weight = exp(-XI * days_ago)
# 0.003/day -> half-life ~231 days (~7.6 months). Backtest-set 2026-08 against
# 2024-25 and 2025-26 (n=2 folds -- only 3 seasons of match data exist at all,
# so this is a direction, not a precise fit): every value in XI_GRID scored
# BETTER on RPS than the literature-default 0.0065, longest memory (0.002)
# scored best. That is consistent with having only 1-2 training seasons on
# hand in every backtest fold -- there is no old data yet for a fast decay to
# usefully discard. Revisit once TRAIN_SEASONS covers more seasons.
XI = 0.003
XI_GRID = [0.002, 0.003, 0.004, 0.005, 0.0065, 0.008, 0.010, 0.012]

# Low-score dependence parameter. Fitted jointly with attack/defence -- this
# is only the optimiser's starting point.
RHO_INIT = -0.05

# Goals carry the scoreline; xG carries the underlying performance and is
# more predictive. Two separate fits, blended at the parameter level.
XG_BLEND = 0.65  # weight on the xG-derived strengths

# Single global home advantage: every side knows its own ground.
# Fitted from data, one parameter for the whole league.

# --- Manager prior -------------------------------------------------------

MANAGER_PRIOR_HALFLIFE_YEARS = 4.0

# Anchors for translating per-match rates between leagues. Points per game is
# fixed by the draw rate alone -- avg = 1.5 - draws/2, since a decisive match
# puts 3 points into the league and a draw only 2. The training window's own
# draw rate is 24.5%, which gives 1.378; the old 1.36 here implied 28% draws
# and quietly made every promoted club and every above-average manager look a
# little better than the data says. xG per team per match sits near 1.40
# across the major leagues.
LEAGUE_AVG_PPG = 1.378
LEAGUE_AVG_XG = 1.40

# Converts an xG-shape margin (goals/match units) into the same units as a
# ppg margin, so the two real manager statistics -- results and underlying
# performance -- can be blended into one budget instead of xG only reshaping
# a ppg-sized total. MEASURED from every stored prior stint: the ratio of
# how much ppg margins actually spread (std) to how much xG margins spread,
# across data/manager_priors.json (n=21 ppg, n=20 xG). Re-run whenever that
# file grows meaningfully -- see src/run_pipeline.manager_delta's docstring.
MANAGER_XG_TO_PPG = 0.809

# Prior confidence scales with sample size measured in matches, not seasons.
# 76 = two Premier League seasons. Using matches keeps 34-game and 38-game
# leagues on the same footing and handles part-seasons (Carrick's 17 games)
# without a special case.
#
# Raised from 76 (two seasons) to 150 (about four) on the project's stated
# principle that experience outweighs a short hot streak: at 76 almost every
# manager maxed out, so a 23-match caretaker and a 261-match veteran carried
# the same confidence. At 150 only genuinely experienced managers reach full
# weight, and a thin CV is discounted rather than trusted.
MANAGER_FULL_CONFIDENCE_MATCHES = 150

# Once a manager has this many matches at the current club, their own record
# fully replaces the prior. Below it, the two blend linearly.
MANAGER_BLEND_MATCHES = 15

# A manager's record at a previous club is mostly that club's squad, not the
# manager. Only this share of their points-per-game margin over average is
# attributed to the manager and carried to the new job. Squad and manager
# cannot be separated cleanly without a manager-fixed-effects model on far
# more data than three seasons, so this is deliberately conservative.
#
# Raised 0.25 -> 0.50 deliberately, as a modelling stance rather than a
# measurement: who is in the dugout is treated as a first-order driver of how
# a team performs, not a footnote to last season's table. The conservative
# 0.25 assumed the squad explains almost everything and the manager almost
# nothing; 0.50 says the two share the credit. Still an ESTIMATE -- no data
# here can separate manager from squad cleanly -- but now an explicit one.
MANAGER_EFFECT_SHARE = 0.50  # ESTIMATE
MANAGER_EFFECT_SHARE_GRID = [0.0, 0.15, 0.25, 0.35, 0.50, 0.65]

# Hard ceiling on the strength shift a manager prior may apply, in log-lambda
# terms. 0.15 is roughly a 16% swing in goals -- large for a single input.
MANAGER_EFFECT_CAP = 0.25

# The fitted attack/defence for a club reflects whoever actually managed it
# for most of the training window -- not necessarily the incoming manager.
# Without this, a club that just lost a good manager keeps 100% of the
# strength that manager built AND gets the new manager's own prior stacked
# fully on top (Bournemouth/Iraola-to-Rose is the visible case). Shrinking the
# fitted baseline toward the league average approximates removing the
# departed manager's share, without claiming to know exactly how much of the
# record was them versus the squad.
# ESTIMATE, and NOT backtestable with current data for the same reason
# MANAGER_EFFECT_SHARE isn't: it would need to isolate a manager's specific
# contribution from a club's results, which no data source here provides.
# "new" gets the full shrink; "interim_to_full" gets half, since part of the
# training window already happened under them.
MANAGER_CHANGEOVER_SHRINK = 0.25  # ESTIMATE
MANAGER_CHANGEOVER_SHRINK_GRID = [0.0, 0.15, 0.25, 0.35, 0.50]

# --- Player adaptation ---------------------------------------------------

# A player's market value is discounted until proven in the Premier League.
# Anything below this many EPL appearances counts as unproven.
EPL_PROVEN_MATCHES = 38

# Squad strength sums the top N players by adjusted value: the XI plus core
# rotation, so academy fringe players cannot pad the total.
SQUAD_DEPTH_COUNT = 18

# Depth is measured per line, not across the squad as a whole -- a club with
# three excellent centre-backs and one senior goalkeeper is not "deep", and a
# flat top-18 count cannot tell the difference. Transfermarkt's position
# vocabulary, grouped into the four lines a manager actually rotates.
POSITION_LINES = {
    "GK": ["GK"],
    "DEF": ["CB", "LB", "RB"],
    "MID": ["DM", "CM", "AM", "RM", "LM"],
    "ATT": ["LW", "RW", "CF", "SS"],
}

# First-choice players per line: a generic 4-3-3. The exact shape a club plays
# matters less here than having a consistent yardstick across all 20.
SQUAD_LINES = {"GK": 1, "DEF": 4, "MID": 3, "ATT": 3}

# Adaptation recovers over a season. The model runs full-season with no
# mid-season refit, so each player contributes the season average of their
# adaptation curve: (initial + 1.0) / 2.
ADAPTATION_RECOVERY = True

# Players with little senior football adapt from a lower base.
LOW_MINUTES_PENALTY = 0.50
LOW_MINUTES_THRESHOLD = 900  # league minutes in the prior season

# Real pre-EPL performance rating (from karier_debutan_epl.md, user-sourced
# per-season senior-career rating history), when available for a specific
# player, replaces part of the league_weight-only guess in adaptation_factor
# -- two players priced the same from the same league are not necessarily the
# same quality, and league_weight alone cannot tell them apart.
#
# Same half-life philosophy as MANAGER_PRIOR_HALFLIFE_YEARS (a kept-separate
# constant since a player's career and a manager's are different questions,
# even though nothing here argues for a different decay rate yet).
PLAYER_RATING_HALFLIFE_YEARS = 4.0  # ESTIMATE
# A season's rating counts fully once a player reaches this many appearances;
# below that its weight shrinks proportionally, so a two-game cameo rating
# cannot outweigh a proper season.
PLAYER_RATING_FULL_MATCHES = 25  # ESTIMATE
# Reward consistency over a single standout season: the weighted score is the
# recency/sample-weighted mean rating minus this many weighted standard
# deviations. A player who is reliably good every season should outscore one
# great season among mediocre ones, even at the same average.
PLAYER_CONSISTENCY_PENALTY = 0.5  # ESTIMATE
# Rating floor/ceiling this scale is stretched across, chosen from the real
# range seen in fotmob_ratings.md's top-N-per-season tables (roughly
# 6.1-8.1). Below the floor maps to 0, at/above the ceiling maps to 1.
PLAYER_RATING_FLOOR = 6.3  # ESTIMATE
PLAYER_RATING_CEILING = 7.8  # ESTIMATE
# MEASURED (not chosen): mean of every rating in fotmob_ratings.md, n=2,786.
# The anchor a season's rating is rescaled around by its own league's
# strength -- same role LEAGUE_AVG_PPG plays for managers -- so a rating
# above/below THIS is what counts as an edge, and only the edge is
# discounted by league quality. Re-measure if fotmob_ratings.md grows.
PLAYER_RATING_AVG = 6.869

# --- Squad rating as a second strength estimate -------------------------
#
# A fitted club's attack/defence comes only from its own past results, so it
# cannot know about a squad that has since changed: sign an EPL-proven 7.0
# player and nothing moves, because his rating was earned in someone else's
# colours. The current squad's real ratings give a SECOND estimate of the
# same quantity, blended with the fitted one rather than added to it (adding
# would count a player who never left twice over).
#
# Weight on the fitted estimate. Kept high on purpose: FotMob ratings are
# themselves partly a product of match results, so the two estimates are not
# independent and letting the rating side dominate would re-import the fit's
# own information as if it were new. 0.75 leaves the fit in charge and lets
# the squad rating correct a quarter of the way.
SQUAD_RATING_BLEND = 0.65  # ESTIMATE
# Below this many rated players in a club's top-18, its squad rating is too
# thin to trust and that club keeps the fitted estimate untouched. FotMob's
# table is a top-N per season, so weak squads legitimately go uncovered.
SQUAD_RATING_MIN_PLAYERS = 8  # ESTIMATE

# Converts squad value into attack/defence strength for clubs with no usable
# Premier League record. Log-linear: doubling squad value adds
# ELASTICITY * ln(2) to strength, not double the strength.
SQUAD_VALUE_ELASTICITY = 0.36  # ESTIMATE
SQUAD_VALUE_ELASTICITY_GRID = [0.20, 0.28, 0.36, 0.44, 0.52]

# Penalty on a fitted club whose squad contains more players new to the league
# than the league average, scaled by that excess.
#
# Retired 2026-09. It had two lives and neither worked. v1: the value data
# covered only the ~500 most expensive players, so newcomer_share measured who
# cleared a global cutoff, not squad turnover (12 of 17 clubs read 0.00).
# Re-enabled 2026-08 on full squads -- but by then debutant_career had grown to
# ~120 players and every newcomer with a rating is excluded here (already
# priced into the squad_rating blend), leaving 14 of 17 clubs back at exactly
# 0.00. What signal remained was tiny and swung on a single FPL glitch
# (Rodrigo Muniz, a Fulham regular, misfetched as a newcomer, inflated their
# penalty by ~1 point). unproven_share does the squad-inexperience job with a
# threshold instead of an exact-zero test, so it doesn't collapse the same way.
# Kept at 0.0 rather than deleted: the plumbing (newcomer_share, the audit
# column, the chart handling) is harmless and a future data shift could revive
# the idea.
ADAPTATION_DRAG = 0.0  # retired -- see above; unproven_share carries this now
ADAPTATION_DRAG_GRID = [0.0, 0.15, 0.30, 0.45, 0.60]

# --- Fatigue / congestion ------------------------------------------------

# Thin squads suffer more from fixture pile-up: the same XI plays every game,
# tires, and picks up injuries.
FATIGUE_ENABLED = True
FATIGUE_REST_BASELINE = 6      # days between matches with no penalty
FATIGUE_MAX_PENALTY = 0.08     # cap on the lambda reduction
CONGESTION_WINDOW_DAYS = 28

# Clubs in continental competition play midweek. Confirm against the final
# 2025/26 table before the season starts.
# Real 2026/27 European qualification, confirmed by the user (England's 5th
# UCL slot via coefficient; Crystal Palace's UEL slot via their Conference
# League win, not league position). Replaces an earlier list written before
# the 2025-26 final table was available -- that one had Man City and Man Utd
# missing entirely despite finishing 2nd and 3rd.
EUROPEAN_COMPETITION = {
    "Arsenal": "UCL",            # champions
    "Manchester City": "UCL",
    "Manchester Utd": "UCL",
    "Aston Villa": "UCL",
    "Liverpool": "UCL",
    "Bournemouth": "UEL",
    "Sunderland": "UEL",
    "Crystal Palace": "UEL",     # Conference League winners
    "Brighton": "UECL",
}

# Midweek European nights, placed on the calendar rather than averaged across
# the season. UEFA fixes these weeks years in advance, so the dates are known
# even though the draws are not -- and for fatigue only the week matters, not
# the opponent. Wednesdays, where most of these ties sit.
#
# UCL/UEL league phase: 8 matchdays, September to late January. Knockout play-
# off and last 16 in February and March; a club reaching them plays more, but
# assuming every entrant goes deep would overstate the load, so the schedule
# below stops at the group stage plus a play-off round. UECL runs a 6-match
# league phase on the same weeks.
_UCL_UEL_LEAGUE_PHASE = [
    "2026-09-16", "2026-09-30", "2026-10-21", "2026-11-04",
    "2026-11-25", "2026-12-09", "2027-01-20", "2027-01-28",
]
_KNOCKOUT_PLAYOFF = ["2027-02-17", "2027-02-24"]

EUROPEAN_WEEKS = {
    "UCL": _UCL_UEL_LEAGUE_PHASE + _KNOCKOUT_PLAYOFF,
    "UEL": _UCL_UEL_LEAGUE_PHASE + _KNOCKOUT_PLAYOFF,
    "UECL": _UCL_UEL_LEAGUE_PHASE[:6] + _KNOCKOUT_PLAYOFF,
}

# --- Promotion penalty ---------------------------------------------------

# run_pipeline measures the promoted-club level directly: every club that came
# up during the training window, and the points per game it went on to take.
# That measurement already contains the promotion effect, so the default here
# is 1.0 -- trust it. The knob exists so the backtest can ask whether the
# measured level transfers to a new season or needs shrinking toward average.
#
# This also settles the double-count worry: the old design charged a guessed
# penalty on top of a squad-value estimate that already had adaptation baked
# into every player. Now the level is measured once and nothing is charged twice.
# --- weekly-updating series (src/weekly.py) -------------------------------
# The pre-season forecast ignores results already played, by design. The
# weekly series folds them into the Dixon-Coles fit and is compared against
# the pre-season one in May -- the test being whether not updating cost
# anything.
#
# For a promoted club, the fit gives a rating off very few games (3 vs 0, not
# 3 vs 1,140). That mini-fit is blended with the promoted-club baseline,
# weight w = LIVE_PRIOR_MATCH_WEIGHT / (LIVE_PRIOR_MATCH_WEIGHT + games_played)
# on the baseline. So the baseline's pull is stated in units of matches: it is
# "worth" this many games of real evidence before the two are equal.
LIVE_PRIOR_MATCH_WEIGHT = 10.0  # ESTIMATE -- a promoted club's prior in match-units
LIVE_PRIOR_MATCH_WEIGHT_GRID = [4.0, 7.0, 10.0, 14.0, 19.0]

# Championship-to-PL offset for the promoted clubs' level (src/championship).
# Fitted on every Premier League-era promoted club (engsoccerdata + openfootball
# for the 2022-23 gap, n=95, 32 windows): pl_ppg = a + b * champ_gd_per_game,
# slope +0.30, r = +0.28 in sample, positive in every era since 1992.
#
# Out of sample it is nothing (src/rolling_offset, 25 folds 2000-2024, 75
# promoted clubs, each fold fitted only on windows before it): the pooled GD
# line beats the population mean on squared ppg error by under one percent
# (0.0906 vs 0.0912) and wins 7 of 25 seasons; on relegation Brier it is worse
# than the population mean and both are worse than simply quoting the training
# relegation rate (0.2595 / 0.2571 / 0.2527). Splitting the line by parachute
# status is worse again (0.0978 MSE) -- the interaction seen in sample is two
# smaller fits, not a signal. The two-fold backtest agrees (rps 0.1439 off ->
# 0.1466 on, manager layer off).
#
# Switched OFF on that evidence (2026-09-15). The fit, the rolling test and
# the flag all stay so the question can be re-asked when more seasons exist.
CHAMPIONSHIP_OFFSET = False

PROMOTION_PENALTY = 1.00
PROMOTION_PENALTY_GRID = [0.80, 0.90, 1.00, 1.10, 1.20]

# --- Monte Carlo ---------------------------------------------------------

N_SIMULATIONS = 10_000
RANDOM_SEED = 20262027

# Per-simulation uncertainty in each team's fitted strength (log scale).
# Wider for clubs the model knows least about.
SIGMA_BASE = 0.08
SIGMA_NEW_MANAGER = 0.14
SIGMA_PROMOTED = 0.22
SIGMA_NON_UEFA_PRIOR = 0.18  # e.g. Newcastle: Saudi Pro League prior

# What counts as "already proven at this level" for the band above. A manager
# who has completed a full season in one of Europe's established first
# divisions is a known quantity; one who has not is a step into the unknown
# however good their record looks elsewhere.
#
# The old test was simply `not uefa` in league_weights.json, which flagged the
# English Championship -- the second tier this model measures promoted clubs
# from directly, and understands better than almost any league on earth -- as
# exotic. That handed Manchester City a widened band because Maresca once won
# it, and Manchester Utd one because Carrick managed in it.
#
# 0.50 cuts at the seven divisions that regularly produce European contenders:
# Premier League, Serie A, La Liga, Bundesliga, Ligue 1, Eredivisie, Primeira
# Liga. Austria's top flight (0.242) sits below it, so Salzburg does not
# qualify a manager on its own -- the user's explicit call.
PROVEN_LEAGUE_MIN_WEIGHT = 0.50

# A stint has to be long enough to count as a full season's work. Most of
# Europe plays 34-38 league games; 30 clears a genuine full campaign while
# excluding a half-season caretaker spell.
PROVEN_LEAGUE_MIN_MATCHES = 30

# A thin squad is fragile, not merely worse: lose a key player and there is no
# equivalent behind him, so the season can collapse -- or the same squad can
# stay fit and finish where its quality says. Widening the band is the honest
# way to say that, and it is separate from the fatigue multiplier, which
# lowers the AVERAGE during congested weeks. Injuries do not wait for a busy
# fixture list. Scaled continuously by how thin a squad is against the league,
# so an averagely-stocked club adds nothing at all.
# ESTIMATE: not backtestable without historical squad data, same limitation as
# the other squad-derived parameters.
SIGMA_THIN_SQUAD = 0.10
SIGMA_THIN_SQUAD_GRID = [0.0, 0.05, 0.10, 0.15, 0.20]

# Clubs enter the training window at different times: Leeds and Sunderland
# came up for 2025-26 and have one Premier League season on file, against
# three for the sides that never left. Without this the fit is equally
# confident about all of them, which is plainly wrong -- a single season is a
# single sample, however tidy it looks. Scaled by how much time-weighted
# evidence a club has relative to an ever-present one.
# ESTIMATE, and unbacktestable for the same reason as the rest of the sigma
# family: it prices what the model does not know.
SIGMA_THIN_EVIDENCE = 0.14

# Above this share of an ever-present club's time-weighted evidence, a club
# counts as fully known. Clubs that never left the division come out a few
# tenths of a percent apart purely from which weekend they played on, and
# widening bands over that would be reading noise as information.
EVIDENCE_FULL_THRESHOLD = 0.95

# Form drifts through the season instead of resetting every match. Without
# this, results are independent, extreme runs never appear, and the final
# rank distribution comes out unrealistically narrow.
FORM_DRIFT_SIGMA = 0.02  # per matchweek, log scale

POINTS_WIN, POINTS_DRAW, POINTS_LOSS = 3, 1, 0
MAX_GOALS = 10  # truncation for the scoreline probability grid

# ---------------------------------------------------------------- sources

FBREF_BASE = "https://fbref.com"
UNDERSTAT_BASE = "https://understat.com"
TRANSFERMARKT_BASE = "https://www.transfermarkt.co.uk"
FOOTBALL_DATA_BASE = "https://api.football-data.org/v4"

# football-data.org: free tier, 10 requests/minute. Register for a key and
# put it in the environment as FOOTBALL_DATA_TOKEN.
FOOTBALL_DATA_TOKEN_ENV = "FOOTBALL_DATA_TOKEN"

# Politeness delays, seconds. FBref returns 429 aggressively above ~1 req/3s;
# Transfermarkt is stricter still.
REQUEST_DELAY = {
    "fbref": 4.0,
    "understat": 2.0,
    "transfermarkt": 3.0,
    "football-data": 6.5,
    # FPL's public API is far more permissive than football-data.org's free
    # tier, whose budget this used to borrow for want of its own entry --
    # 6.5s/3 per player turned one squad rebuild into a 20-minute job and got
    # it killed by a timeout mid-run. Still deliberately unhurried.
    "fpl": 1.5,
}

CACHE_TTL_DAYS = 7  # re-fetch anything older
