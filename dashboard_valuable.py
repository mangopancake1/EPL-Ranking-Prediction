"""
The "Most valuable players" section: a league-wide board and a per-club one.

Built from native Streamlit primitives -- bordered containers, columns,
metrics, pills, images -- in the manner of streamlit/demo-stockpeers, with the
ranked chart done in ECharts as the rest of this dashboard does. There is no
custom CSS here on purpose: an earlier version of this section leaned on
gradients, glows and medal-coloured numerals, and looked like decoration
wrapped around the data rather than a reading of it.

Two things worth knowing about the numbers:

* Market value is what a player is *worth*, not what he cost. Fees live in the
  transfer file; values are what the forecast actually uses, and the two
  disagree by a lot (Morgan Rogers: 110m value against a 138m fee).
* Every player here comes from the transfer-corrected roster, so anyone sold or
  loaned out in the closing days of the window is already gone.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
from streamlit_echarts import JsCode

import config
import dashboard_data as dd
import dashboard_logos as logos
from dashboard_data import st_echarts

# Premier League's own player photography, keyed by the FPL "code" the squad
# build stores. 531 of 549 players have one; the rest fall back to their club
# crest, which is honest about what is missing instead of inventing a face.
_PHOTO = "https://resources.premierleague.com/premierleague/photos/players/250x250/p{code}.png"


def _ratings() -> dict[str, tuple[float, str]]:
    """Best rating per player. A real Premier League one beats an estimate."""
    from src import debutant_career

    out = {name: (value, "career") for name, value in debutant_career.rating_lookup().items()}
    for r in dd.player_ratings_table().itertuples(index=False):
        out[r.player] = (float(r.rating), "epl")
    return out


def _face(row: pd.Series, club: str):
    code = row.get("fpl_code")
    if pd.notna(code):
        return _PHOTO.format(code=int(code))
    return logos.logo_path(club)


def _player_cell(cell, row: pd.Series, rating: tuple[float, str] | None) -> None:
    """One player, in a bordered cell: face, name, the value, the rating."""
    with cell.container(border=True):
        face = _face(row, row.club)
        if face is not None:
            st.image(face, width="stretch")

        st.markdown(f"**{row.player}**")

        meta = [str(row.position)]
        if pd.notna(row.get("age")):
            meta.append(f"{int(row['age'])}")
        meta.append(row.club)
        if pd.notna(row.get("contract_until")):
            meta.append(f"contract to {str(row['contract_until'])[:4]}")
        st.caption(" · ".join(meta))

        # delta_color="off" throughout this dashboard: these are labels, not
        # movements, and a green arrow would claim something that isn't there.
        if rating:
            value, source = rating
            st.metric(
                "Market value", f"€{row.market_value_m:,.0f}m",
                f"{value:.2f} rating" + ("" if source == "epl" else " (est.)"),
                delta_color="off",
                help=("Average Premier League match rating" if source == "epl"
                      else "No Premier League record yet — estimated from his career "
                           "abroad, discounted for the strength of that league"),
            )
        else:
            st.metric("Market value", f"€{row.market_value_m:,.0f}m",
                      "no rating on file", delta_color="off")


def _ranked_chart(players: pd.DataFrame, key: str) -> None:
    """The whole selection as one horizontal bar, club crest per row."""
    rows = players.iloc[::-1]                    # ECharts fills a value axis bottom-up
    names = rows.player.tolist()
    clubs = rows.club.tolist()
    values = [round(float(v), 1) for v in rows.market_value_m]

    # One rich-text slot per row so each label carries its own club's crest.
    slots, rich = {}, {}
    for name, club in zip(names, clubs):
        uri = logos.logo_data_uri(club)
        if uri:
            slots[name] = f"c{len(rich)}"
            rich[slots[name]] = {"height": 16, "width": 16,
                                 "backgroundColor": {"image": uri}, "align": "center"}
    rich["t"] = {"fontSize": 12, "padding": [0, 0, 0, 6]}

    st_echarts(
        {
            "tooltip": {"trigger": "item", "formatter": "{b}: €{c}m"},
            "grid": {"left": 8, "right": 70, "top": 6, "bottom": 6, "containLabel": True},
            "xAxis": {"type": "value", "show": False, "max": max(values) * 1.15},
            "yAxis": {
                "type": "category", "data": names,
                "axisLine": {"show": False}, "axisTick": {"show": False},
                "axisLabel": {
                    "margin": 10, "rich": rich,
                    "formatter": JsCode(
                        "function (name) { var m = "
                        + repr(slots).replace("'", '"')
                        + "; return (m[name] ? '{' + m[name] + '|}' : '') + '{t|' + name + '}'; }"
                    ),
                },
            },
            "series": [{
                "type": "bar",
                "barWidth": 13,
                "data": [
                    {"value": v, "itemStyle": {"color": dd.club_colour(c),
                                               "borderRadius": [0, 6, 6, 0]}}
                    for v, c in zip(values, clubs)
                ],
                "label": {"show": True, "position": "right", "fontSize": 11.5,
                          "formatter": JsCode("function (p) { return '€' + p.value + 'm'; }")},
            }],
        },
        height=f"{max(200, 26 * len(names) + 20)}px",
        key=key,
    )


def render(squads: dict[str, pd.DataFrame], default_club: str | None = None) -> None:
    st.subheader(":material/paid: Most valuable players")
    st.caption(
        "Market value, not transfer fee — what each player is worth today, which is "
        "the number the forecast actually uses. Anyone sold or loaned out in the "
        "closing days of the window has already been taken out, so this is the squad "
        "that will play rather than the one that was pasted in."
    )

    ratings = _ratings()
    everyone = pd.concat(squads.values(), ignore_index=True)

    controls, chart = st.columns([1, 2.2])

    with controls.container(border=True, height="stretch"):
        scope = st.pills("Show", ["Whole league", "One club"],
                         default="Whole league", key="mvp_scope")
        if scope == "One club":
            options = list(config.TEAMS)
            club = st.selectbox(
                "Club", options,
                index=options.index(default_club) if default_club in options else 0,
                key="mvp_club",
            )
            pool, top_n = squads[club], 10
        else:
            pool, top_n = everyone, 15

        ranked = pool.nlargest(top_n, "market_value_m").reset_index(drop=True)
        best = ranked.iloc[0]

        st.metric("Most valuable", best.player, f"€{best.market_value_m:,.0f}m",
                  delta_color="off")
        if scope == "Whole league":
            owner = ranked.club.value_counts()
            st.metric(f"Owns most of the top {top_n}", owner.index[0],
                      f"{owner.iloc[0]} players", delta_color="off")
        else:
            gap = best.market_value_m / max(ranked.market_value_m.iloc[-1], 0.1)
            st.metric("Top-heavy by", f"{gap:.1f}×",
                      f"best vs {len(ranked)}th most valuable", delta_color="off")

    with chart.container(border=True, height="stretch"):
        _ranked_chart(ranked, key="mvp_ranked")

    ""

    # Small multiples, the stockpeers idiom: the same cell repeated, one per
    # player, so the eye compares like with like instead of reading a ranking.
    n_cols = 5
    cells = st.columns(n_cols)
    for i, (_, row) in enumerate(ranked.head(n_cols).iterrows()):
        _player_cell(cells[i], row, ratings.get(row.player))

    with st.expander("Raw data preview", icon=":material/table_view:"):
        cols = ["player", "club", "position", "age", "market_value_m",
                "contract_until", "signed_from"]
        st.dataframe(
            pool[[c for c in cols if c in pool.columns]]
            .sort_values("market_value_m", ascending=False),
            hide_index=True, width="stretch",
            column_config={
                "player": "Player", "club": "Club", "position": "Pos",
                "age": st.column_config.NumberColumn("Age", format="%d"),
                "market_value_m": st.column_config.NumberColumn("Value", format="€%.0fm"),
                "contract_until": "Contract until", "signed_from": "Signed from",
            },
        )
