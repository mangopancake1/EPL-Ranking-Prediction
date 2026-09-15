"""
Club -> league name (league_weights.json vocabulary), for scoring a debutant's
pre-EPL career season by season instead of one flat career-wide league.

Deliberately incomplete: only clubs whose current top-division league is
known with confidence are listed. A club not here (reserve/youth sides,
lower-tier clubs with no real EPL-adjacent relevance, anything ambiguous)
resolves to None and is treated as neutral in debutant_career.py -- skipped
rather than guessed, per this project's standing rule against fabricated
precision. Covers the clubs actually appearing in karier_debutan_epl.md.
"""

from __future__ import annotations

CLUB_LEAGUE: dict[str, str] = {
    # England - Championship (second tier; distinct from PL itself)
    "Sheff Wed": "Championship", "Bristol City": "Championship",
    "Middlesbrough": "Championship", "Norwich": "Championship",
    "Coventry": "Championship", "West Brom": "Championship",
    "Swansea": "Championship", "Stoke": "Championship", "Leicester": "Championship",
    "Southampton": "Championship", "Wigan": "Championship", "Hull": "Championship",
    "Ipswich": "Championship",
    # England - League One / Two (much weaker; not separately weighted, treated
    # as League One for all of them -- close enough at this end of the scale)
    "Barnsley": "League One", "Wycombe": "League One", "Northampton": "League One",
    "Forest Green": "League One", "Harrogate Town": "League One",
    "Barnet": "League One", "Barrow": "League One", "Carlisle": "League One",
    "Accrington": "League One", "Rotherham": "League One", "Peterborough": "League One",
    "Fleetwood": "League One", "Lincoln": "League One", "Oxford City": "League One",
    "Scunthorpe": "League One", "Salford": "League One", "Walsall": "League One",
    "Wimbledon": "League One", "Bristol Rovers": "League One",

    # Spain
    "Real Madrid": "La Liga", "Barcelona": "La Liga", "Atlético Madrid": "La Liga",
    "Sevilla": "La Liga", "Real Sociedad": "La Liga", "Real Betis": "La Liga",
    "Villarreal": "La Liga", "Girona": "La Liga", "Osasuna": "La Liga",
    "Getafe": "La Liga", "Las Palmas": "La Liga", "Rayo Vallecano": "La Liga",
    "Celta Vigo": "La Liga", "Elche": "La Liga", "Real Valladolid": "La Liga",
    "Real Zaragoza": "La Liga 2", "Sevilla Atletico": "La Liga 2",
    "Real Madrid Castilla": "La Liga 2", "Barça Atlètic": "La Liga 2",

    # Italy
    "Juventus": "Serie A", "Atalanta": "Serie A", "Cagliari": "Serie A",
    "Sassuolo": "Serie A", "Sampdoria": "Serie B", "Parma": "Serie A",
    "Salernitana": "Serie B", "Monza": "Serie A", "Novara": "Serie B",
    "Padova": "Serie B", "Pordenone Calcio": "Serie B", "Renate": "Serie B",

    # Germany
    "Dortmund": "Bundesliga", "Borussia Dortmund II": "2. Bundesliga",
    "Frankfurt": "Bundesliga", "Leverkusen": "Bundesliga", "RB Leipzig": "Bundesliga",
    "Wolfsburg": "Bundesliga", "Wolfsburg II": "2. Bundesliga",
    "Hoffenheim": "Bundesliga", "Freiburg": "Bundesliga", "Freiburg II": "2. Bundesliga",
    "Hertha BSC": "2. Bundesliga", "Hamburger SV": "Bundesliga",
    "Schalke 04": "2. Bundesliga", "Sandhausen": "2. Bundesliga",
    "M'gladbach": "Bundesliga", "Borussia Mönchengladbach II": "2. Bundesliga",

    # France
    "PSG": "Ligue 1", "Lyon": "Ligue 1", "Marseille": "Ligue 1", "Lille": "Ligue 1",
    "Lille B": "Ligue 2", "Monaco": "Ligue 1", "Rennes": "Ligue 1",
    "Rennes B": "Ligue 2", "Marseille B": "Ligue 2", "Sochaux B": "Ligue 2",
    "Nice": "Ligue 1", "Nantes": "Ligue 1",
    "Strasbourg": "Ligue 1", "Strasbourg B": "Ligue 2", "Toulouse": "Ligue 1",
    "Toulouse B": "Ligue 2", "Reims": "Ligue 1", "Reims B": "Ligue 2",
    "Angers": "Ligue 1", "Angers B": "Ligue 2", "Auxerre": "Ligue 1",
    "Le Havre": "Ligue 1", "Lens": "Ligue 1", "Saint-Étienne": "Ligue 2",
    "Saint-Étienne B": "Ligue 2", "Metz": "Ligue 2", "Metz B": "Ligue 2",
    "Clermont Foot": "Ligue 2", "Clermont Foot B": "Ligue 2",
    "Paris FC": "Ligue 2", "Paris FC B": "Ligue 2", "Dunkerque": "Ligue 2",
    "Rodez": "Ligue 2", "Guingamp B": "Ligue 2",

    # Netherlands
    "Ajax": "Eredivisie", "Jong Ajax": "Eredivisie", "PSV Eindhoven": "Eredivisie",
    "Jong PSV": "Eredivisie", "Feyenoord": "Eredivisie", "FC Twente": "Eredivisie",
    "FC Groningen": "Eredivisie", "Vitesse": "Eredivisie", "SC Heerenveen": "Eredivisie",
    "Sparta Rotterdam": "Eredivisie", "Jong Sparta Rotterdam": "Eredivisie",
    "De Graafschap": "Eredivisie", "FC Emmen": "Eredivisie", "ADO Den Haag": "Eredivisie",
    "Cambuur": "Eredivisie", "Excelsior Maassluis": "Eredivisie",
    "FC Utrecht": "Eredivisie",

    # Portugal
    "Sporting CP": "Primeira Liga", "Sporting CP B": "Primeira Liga",
    "Benfica": "Primeira Liga", "Benfica B": "Primeira Liga", "Braga": "Primeira Liga",
    "Braga B": "Primeira Liga", "Famalicão": "Primeira Liga", "Farense": "Primeira Liga",
    "Marítimo": "Primeira Liga", "Rio Ave": "Primeira Liga", "Rio Ave B": "Primeira Liga",
    "Santa Clara": "Primeira Liga", "Chaves": "Primeira Liga",

    # Belgium
    "Club Brugge": "Belgian Pro League", "Genk": "Belgian Pro League",
    "Genk U23": "Belgian Pro League", "Royal Antwerp": "Belgian Pro League",
    "Union St.Gilloise": "Belgian Pro League", "Westerlo": "Belgian Pro League",
    "Oostende": "Belgian Pro League", "Zulte Waregem": "Belgian Pro League",
    "Kortrijk": "Belgian Pro League",

    # Turkey / Greece / Switzerland / Austria / Scotland / Scandinavia / eastern Europe
    "Fenerbahçe": "Super Lig", "Trabzonspor": "Super Lig", "Göztepe": "Super Lig",
    "Antalyaspor": "Super Lig", "Olympiacos": "Super League Greece",
    "Olympiakos CFP II": "Super League Greece", "PAOK Thessaloniki": "Super League Greece",
    "Basel": "Swiss Super League", "FC Zürich": "Swiss Super League",
    "St.Truiden": "Belgian Pro League",
    "Salzburg": "Austrian Bundesliga", "FC Liefering": "Austrian Bundesliga",
    "Rapid Wien": "Austrian Bundesliga", "Sturm Graz": "Austrian Bundesliga",
    "Sturm Graz II": "Austrian Bundesliga", "Grazer AK": "Austrian Bundesliga",
    "Wolfsberger AC": "Austrian Bundesliga", "Wolfsberger AC II": "Austrian Bundesliga",
    "Hartberg": "Austrian Bundesliga",
    "Rangers": "Scottish Premiership", "Celtic": "Scottish Premiership",
    "Kilmarnock": "Scottish Premiership",
    "Slavia Prague": "Czech First League",
    "Tromsø": "Norwegian Eliteserien", "SC Dnipro-1": "Ukrainian Premier League",
    "Nordsjælland": "Danish Superliga", "Sønderjyske": "Danish Superliga",
    "Lyngby": "Danish Superliga", "Mjällby": "Swedish Allsvenskan",
    "IFK Göteborg": "Swedish Allsvenskan", "Young Boys": "Swiss Super League",
    "Hajduk Split": "Croatian HNL",

    # South America / rest of world
    "River Plate": "Argentine Primera", "Boca Juniors": "Argentine Primera",
    "Corinthians": "Brazilian Serie A", "Flamengo": "Brazilian Serie A",
    "Fluminense": "Brazilian Serie A", "Palmeiras": "Brazilian Serie A",
    "Atlético-MG": "Brazilian Serie A",
    "Urawa Red Diamonds": "J1 League",

    # Premier League itself -- a prior loan/appearance-list entry at a PL club
    "Chelsea": "Premier League", "Man City": "Premier League",
    "Liverpool": "Premier League", "Leeds": "Premier League",
    "Brentford": "Premier League", "Brighton": "Premier League",
    "Burnley": "Premier League", "Crystal Palace": "Premier League",
    "Sheff Utd": "Premier League", "West Ham": "Premier League",

    # A few more identifiable second-tier/top-flight clubs
    "Roma": "Serie A", "Düsseldorf": "2. Bundesliga", "VVV-Venlo": "Eredivisie",
    "FC Midtjylland": "Danish Superliga",
    "Preston": "Championship", "Huddersfield": "Championship",
    "Birmingham": "League One", "MK Dons": "League One", "Charlton": "League One",

    # Added with the deadline-day batch (2026-09-01)
    "Gent": "Belgian Pro League", "Gent U23": "Belgian Pro League",
    "Genoa": "Serie A", "Atalanta Bergamasca Calcio Under 23": "Serie B",
    "VfB Stuttgart": "Bundesliga", "Union Berlin": "Bundesliga",
    "Bayern München": "Bundesliga", "Bayern München II": "2. Bundesliga",
    "Ingolstadt": "2. Bundesliga", "Ingolstadt U19": "2. Bundesliga",
    "Paderborn": "2. Bundesliga", "SC Paderborn 07 II": "2. Bundesliga",
    "Millwall": "Championship", "Blackburn": "Championship",
    "Port Vale": "League One", "Luton": "Championship",
    "Salt Lake": "MLS", "Real Monarchs SLC": "MLS",
    "AIK": "Swedish Allsvenskan",
}
