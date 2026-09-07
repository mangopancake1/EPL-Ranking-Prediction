"""
Club crests for the dashboard.

Source files live under club_logo/<brandlogos.net folder>/<name>.png. Folder
names are inconsistent (sometimes the full club name, sometimes a random
slug), so they're mapped once here to config.TEAMS' canonical names -- if a
folder ever changes, this is the only place that needs touching.

Two forms are used:
  - the original file path, for st.image() in headers/tables (full
    resolution, free for Streamlit since it reads straight off disk);
  - a small base64 data URI, for ECharts symbols/markers. ECharts ships the
    whole chart option as JSON to the browser, so pasting 20 full-resolution
    crests into EVERY chart would bloat the page many times over -- hence a
    48x48 thumbnail for that use.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

from PIL import Image

import config

_ROOT = config.ROOT / "club_logo"

# folder name -> original PNG filename
_FILES = {
    "Arsenal": "Arsenal_FC-brandlogos.net-OPPAMI/arsenal_fc-logo_brandlogos.net_kae1j.png",
    "Aston Villa": "Aston_Villa_FC-brandlogos.net-OdXusZ/aston_villa_fc-logo.png",
    "Bournemouth": "AFC_Bournemouth-brandlogos.net-OewBWf/afc_bournemouth-logo_brandlogos.net_wifjg.png",
    "Brentford": "brentford-fc-logo-45458/brentford-fc-logo.png",
    "Brighton": "brighton-hove-albion-logo-60MGK/brighton-hove-albion-logo.png",
    "Chelsea": "Chelsea_FC-brandlogos.net-O0ZcrA/chelsea_fc-logo_brandlogos.net_jrklu.png",
    "Coventry City": "Coventry_City_FC-brandlogos.net-Oi1hZT/coventry-city-fc-logo.png",
    "Crystal Palace": "Crystal_Palace_FC-brandlogos.net-Oz9Cfh/crystal_palace_fc-logo_brandlogos.net_asddi.png",
    "Everton": "Everton_FC-brandlogos.net-OT4ELA/everton_fc-logo_brandlogos.net_wuxl3.png",
    "Fulham": "fulham-fc-logo-2F4D4/fulham-fc-logo.png",
    "Hull City": "Hull_City-brandlogos.net-Uhc5D/hull_city-logo_brandlogos.net_ttmss.png",
    "Ipswich Town": "Ipswich_Town_FC-brandlogos.net-OqV01X/ipswich_town_fc-logo_brandlogos.net_n1uch.png",
    "Leeds United": "leeds-united-fc-logo-59E02/leeds-united-fc-logo.png",
    "Liverpool": "Liverpool_FC-brandlogos.net-OH4lAk/liverpool_fc-brandlogo.net.png",
    "Manchester City": "manchester-city-fc-logo-TitlM/manchester-city-fc-logo.png",
    "Manchester Utd": "Manchester_United_F.C.-brandlogos.net-hK2Up/manchester_united_f.c.-logo_brandlogos.net_6znjs.png",
    "Newcastle Utd": "Newcastle_United_FC-brandlogos.net-H734d/newcastle_united_fc-logo_brandlogos.net_ypslm.png",
    "Nott'ham Forest": "Nottingham_Forest-brandlogos.net-OQPRBF/nottingham_forest-logo-brandlogo.net.png",
    "Sunderland": "Sunderland_AFC-brandlogos.net-Ozy4sL/sunderland_afc-logo_brandlogos.net_ddmyr.png",
    "Tottenham": "Tottenham_Hotspur_F.C.-brandlogos.net-OV6h3a/tottenham-hotspur-f.c.-logo.png",
}

_THUMB_SIZE = 48


def logo_path(team: str) -> Path | None:
    """Path to the original PNG, for st.image()."""
    rel = _FILES.get(team)
    if not rel:
        return None
    p = _ROOT / rel
    return p if p.exists() else None


def _thumbnail_b64(path: Path) -> str:
    img = Image.open(path).convert("RGBA")
    img.thumbnail((_THUMB_SIZE, _THUMB_SIZE), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


_thumb_cache: dict[str, str] = {}


def logo_data_uri(team: str) -> str | None:
    """48x48 thumbnail data URI, for ECharts symbols/markers."""
    if team in _thumb_cache:
        return _thumb_cache[team]
    p = logo_path(team)
    if p is None:
        return None
    uri = _thumbnail_b64(p)
    _thumb_cache[team] = uri
    return uri


if __name__ == "__main__":
    missing = [t for t in config.TEAMS if logo_path(t) is None]
    assert not missing, f"clubs without a crest: {missing}"

    sizes = {t: len(logo_data_uri(t) or "") for t in config.TEAMS}
    assert all(v > 0 for v in sizes.values()), "an empty thumbnail was produced"
    avg_kb = sum(sizes.values()) / len(sizes) / 1024 * 0.75  # base64 -> approx raw bytes
    print(f"OK  20 crests mapped, thumbnails average ~{avg_kb:.1f} KB, "
          f"all 20 thumbnails together ~{sum(sizes.values())/1024*0.75:.0f} KB")
