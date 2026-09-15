"""
The pre-season forecast against a weekly-updating one -- the question of
whether not updating costs anything, tracked to May.

Reads output/series_log.csv (src/weekly) and output/prior_strength.csv
(src/prior_strength). Both are written by scripts that are safe to re-run
every week; this page just renders whatever is there.
"""

import numpy as np
import pandas as pd
import streamlit as st

import config
import dashboard_data as dd
import dashboard_logos as logos
from dashboard_data import st_echarts

st.title(":material/update: Live vs Pre-season")

log = dd.series_log()
prior = dd.prior_strength()

if log.empty:
    st.info(
        "No weekly run logged yet. Run `python -m src.weekly` to fold this "
        "season's results into a second forecast and compare the two here.",
        icon=":material/info:",
    )
    st.stop()

matchday = int(log["matchday"].max())
runs = sorted(log["run_date"].unique())

st.markdown(
    "**The headline forecast on this site never looks at a single result.** It is built "
    "from three seasons of past matches and stops there, on purpose — it was never set "
    "up to update mid-season, and folding in partial data without a real update "
    "mechanism trades one bias for another.\n\n"
    f"This page is the check on that choice. The same model, same layers, is also run "
    f"**conditioned on every match played so far** ({matchday} matchweeks). Where the two "
    "disagree is where not updating is currently costing — or saving — the forecast "
    "something. In May, the two series together answer whether the pre-season call was "
    "worth making."
)

# --- how the two series differ, right now --------------------------------------
latest = log[log["run_date"] == runs[-1]]
pre = latest[latest["series"] == "preseason"].set_index("team")
wk = latest[latest["series"] == "weekly"].set_index("team")

# Subtract on the shared index, THEN reset -- never mix one frame's order with
# another's .values.
diff = pd.DataFrame({
    "pts": wk["exp_points"] - pre["exp_points"],
    "releg": wk["P_bottom3"] - pre["P_bottom3"],
}).reset_index().sort_values("pts")

k = st.columns(3)
up = diff.iloc[-1]
down = diff.iloc[0]
k[0].metric("Most lifted by results", up.team, f"+{up.pts:.1f} pts", delta_color="off")
k[1].metric("Most dragged by results", down.team, f"{down.pts:.1f} pts", delta_color="off")
moved = diff.assign(a=diff["pts"].abs()).query("a > 0.5")
k[2].metric("Clubs the results move", f"{len(moved)} of 20",
            "the rest look the same either way", delta_color="off")

st.subheader(":material/swap_vert: What the played matches have changed")
st.caption(
    "Weekly forecast minus pre-season, in projected points. A long bar means this "
    "club's start has pulled the forecast well away from where the pre-season model "
    "left it. The promoted clubs move most — they had the least evidence to begin with, "
    "so a few real games count for a lot."
)

promoted = set(config.PROMOTED)
with st.container(border=True):
    st_echarts(
        {
            "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
            "grid": {"left": "3%", "right": "6%", "top": 20, "bottom": 30, "containLabel": True},
            "xAxis": {"type": "value", "name": "weekly − pre-season (points)",
                      "nameLocation": "middle", "nameGap": 28},
            "yAxis": {"type": "category", "data": diff["team"].tolist(),
                      "axisLabel": {"fontSize": 11}},
            "series": [{
                "type": "bar",
                "data": [
                    {"value": round(float(r.pts), 2),
                     "itemStyle": {"color": dd.club_colour(r.team),
                                   "opacity": 1.0 if r.team in promoted else 0.55}}
                    for r in diff.itertuples(index=False)
                ],
                "markLine": {"silent": True, "symbol": "none",
                             "lineStyle": {"color": dd.MUTED},
                             "data": [{"xAxis": 0}]},
            }],
        },
        height="560px",
        key="series_gap",
    )
    st.caption("Solid bars are the three promoted clubs; faded bars are the fitted 17.")

""

# --- prior strength in matches -----------------------------------------------
if not prior.empty:
    st.subheader(":material/anchor: How strong is each club's prior?")
    st.caption(
        "One way to read the gap above: how many matchdays would it take for real "
        "results to move a club's relegation probability by ten points? That number is "
        "the pre-season prior's strength, measured in matches. A promoted club's prior "
        "gives way in about a game — it was a league-average guess to begin with. An "
        "established club's barely moves at all: three seasons of its own results are "
        "not overturned by three weeks."
    )

    p = prior.copy()

    def label(r):
        if r["reached"]:
            return f"{r['matchdays_to_10pt']:.1f}"
        if not np.isfinite(r["matchdays_to_10pt"]) or r["matchdays_to_10pt"] > 38 \
                or abs(r["slope_per_md"]) < 0.005:
            return "wall"
        return f"~{r['matchdays_to_10pt']:.0f} (proj.)"

    p["Matchdays to a 10-pt move"] = p.apply(label, axis=1)
    p["Basis"] = np.where(p["promoted"], "Newly promoted", "Proven in the PL")
    p = p.sort_values("matchdays_to_10pt").reset_index(drop=True)

    prom = p[p["promoted"]]
    c = st.columns(2)
    c[0].metric("Promoted-club prior", f"{prom['matchdays_to_10pt'].median():.1f} matchdays",
                "to a 10-point relegation swing", delta_color="off")
    walls = (~p["promoted"] & ~p["reached"]).sum()
    c[1].metric("Fitted clubs unmoved", f"{walls} of 17",
                f"none shifted 10 points in {matchday} real matchdays", delta_color="off")

    # Same shape as the predicted table on the Table page: rank, crest, club,
    # then the numbers, with relegation as a progress bar.
    p.insert(0, "#", range(1, len(p) + 1))
    p["Crest"] = [logos.logo_data_uri(t) for t in p["team"]]
    rmax = float(p[["releg_start", "releg_now"]].to_numpy().max())

    st.dataframe(
        p[["#", "Crest", "team", "Basis", "releg_start", "releg_now",
           "Matchdays to a 10-pt move"]],
        hide_index=True, width="stretch", height=740,
        column_config={
            "#": st.column_config.NumberColumn(width="small"),
            "Crest": st.column_config.ImageColumn(width="small"),
            "team": st.column_config.TextColumn("Club"),
            "Basis": st.column_config.TextColumn(
                "Basis", help="Proven in the PL = built from real match results; "
                              "Newly promoted = priced off past promoted clubs"),
            "releg_start": st.column_config.ProgressColumn(
                "Relegation, pre-season", format="%.1f%%", min_value=0, max_value=rmax),
            "releg_now": st.column_config.ProgressColumn(
                f"Relegation, MW{matchday}", format="%.1f%%", min_value=0, max_value=rmax),
            "Matchdays to a 10-pt move": st.column_config.TextColumn(
                "Matchdays to a 10-pt move",
                help="How many matchdays of real results it takes to shift this club's "
                     "relegation probability by ten points. 'wall' = it barely moves."),
        },
    )

""

# --- the May test -----------------------------------------------------------
st.subheader(":material/calendar_clock: The test in May")
st.caption(
    f"Both series are logged every time `python -m src.weekly` runs — so far on "
    f"{len(runs)} date(s). By May there will be a full trace of each: the pre-season "
    "one that was committed to in August, and the weekly one that kept updating. The "
    "comparison is not just whether the model was right, but whether standing still "
    "cost anything a moving model would have caught."
)

if len(runs) > 1:
    trace = log.pivot_table(index="run_date", columns="series", values="exp_points",
                            aggfunc="mean")
    st.line_chart(trace)
else:
    st.caption(f"_Only one run logged ({runs[0]}). The trace fills in as the season goes._")

st.info(
    "**On the manager layer specifically.** The one thing this does that copying last "
    "season's table cannot is price in a manager the results have not seen yet. Run "
    "against the two seasons there is data for, turning that layer on improved every "
    "score that was measured — rank correlation 0.50 → 0.52, mean rank error 4.35 → "
    "4.30, probabilistic score 0.144 → 0.142 — and hurt nothing. It is the first "
    "evidence either way, though two seasons is a direction, not a calibration.",
    icon=":material/lightbulb:",
)

with st.expander("Raw data preview", icon=":material/table_view:"):
    tab1, tab2 = st.tabs(["Series log", "Prior strength"])
    with tab1:
        st.dataframe(log, hide_index=True, width="stretch")
    with tab2:
        st.dataframe(prior, hide_index=True, width="stretch")
