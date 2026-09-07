import numpy as np
import pandas as pd
import streamlit as st
from streamlit_echarts import JsCode

import config
import dashboard_data as dd
import dashboard_logos as logos
from dashboard_data import st_echarts

table, grid, merged, squads = dd.load()

st.title(":material/sports_soccer: Premier League 2026/27 Forecast")
# Counted from the simulations every time the page loads. Hand-typed versions of
# these sentences quoted a title count, a favourite's percentage and a promoted
# club's spread that had all drifted since they were written.
_fav, _second, _third = table.nlargest(3, "P_title").itertuples(index=False)
_wins = round(_fav.P_title * config.N_SIMULATIONS)
_spans = table.assign(s=table.points_p90 - table.points_p10)
_drama = _spans.nlargest(1, "s").iloc[0]
_drama_short = _drama.team.replace(" City", "").replace(" United", "").replace(" Town", "")
# The widest bars among clubs that DO have a Premier League record -- a promoted
# club being uncertain is not news; a club with three seasons of evidence being
# this uncertain is the point.
_w1, _w2 = _spans[_spans.source == "fitted"].nlargest(2, "s").itertuples(index=False)

st.markdown(
    f"**We played this season {config.N_SIMULATIONS:,} times. {_fav.team} won it "
    f"{_wins:,} times — and lost it {config.N_SIMULATIONS - _wins:,}.**\n\n"
    "That is the whole idea of this page. Not one confident table, but every table "
    f"the season could plausibly produce, counted up. {_fav.team} come out favourites "
    f"at {_fav.P_title:.0%}, {_second.team} close behind at {_second.P_title:.0%}, and "
    f"nobody else clears {_third.P_title:.0%}.\n\n"
    f"The real drama is at the other end. {_drama_short} finish bottom three in "
    f"{_drama.P_bottom3:.0%} of seasons — but they also reach "
    f"{_drama.points_p90:.0f} points in the best 10% of them. Same squad, "
    "same fixtures, wildly different Mays. Every chart below is a way of looking at "
    "that spread instead of hiding it behind a single number."
)

# --- SIDEBAR: filters ---
with st.sidebar:
    st.title(":material/filter_alt: Filters")
    zone = st.multiselect(
        "Zone",
        options=["Title race", "European push", "Mid-table", "Relegation battle"],
        default=[],
        key="zone",
        bind="query-params",
        help="Leave empty to show every club",
    )
    basis = st.multiselect(
        "Strength basis",
        options=["Proven in the Premier League", "Newly promoted"],
        default=[],
        key="basis",
        bind="query-params",
    )

SRC_ID = {"fitted": "Proven in the Premier League", "promoted": "Newly promoted"}
merged["Basis"] = merged["source"].map(SRC_ID)
merged["Zone"] = [
    "Title race" if r.P_title >= 0.03
    else "European push" if r.P_top4 >= 0.10
    else "Relegation battle" if r.P_bottom3 >= 0.15
    else "Mid-table"
    for r in merged.itertuples(index=False)
]

view = merged.copy()
if zone:
    view = view[view["Zone"].isin(zone)]
if basis:
    view = view[view["Basis"].isin(basis)]

if view.empty:
    st.warning("No club matches these filters.", icon=":material/filter_alt_off:")
    st.stop()

st.caption(f"Showing {len(view)} of {len(merged)} clubs.")

# --- KPIs ---
champ = view.sort_values("P_title", ascending=False).iloc[0]
doomed = view.sort_values("P_bottom3", ascending=False).iloc[0]
widest = view.assign(w=view.points_p90 - view.points_p10).sort_values("w").iloc[-1]
thin = view.sort_values("depth_ratio").iloc[0]

k = st.columns(4)
k[0].metric("Title favourite", champ.team, f"{champ.P_title:.1%}", delta_color="off")
k[1].metric("Relegation favourite", doomed.team, f"{doomed.P_bottom3:.1%}", delta_color="inverse")
k[2].metric("Hardest to call", widest.team,
            f"{widest.points_p10:.0f}–{widest.points_p90:.0f} pts", delta_color="off")
k[3].metric("Thinnest squad", thin.team, f"{thin.depth_ratio:.2f}", delta_color="off")

st.divider()

# ---------------------------------------------------------------- season trend
st.subheader(":material/trending_up: Which way is the season pulling?")
st.caption(
    "Nothing is decided in August. Watch the lines fan apart: by Christmas the top "
    "two have separated from the chase, and by March the bottom three have stopped "
    "climbing. The season doesn't turn — it drifts, and it drifts early. Switch to "
    "league position to see the same story as a table forming."
)

traj = dd.trajectory()
traj_view = traj[traj.team.isin(view.team)]
weeks = sorted(traj_view.matchweek.unique())

metric = st.radio(
    "Show",
    ["Cumulative points", "League position"],
    horizontal=True,
    key="trend_metric",
    bind="query-params",
)
col, y_title, inverted = (
    ("points_mean", "Cumulative points", False) if metric == "Cumulative points"
    else ("rank_mean", "League position", True)
)

order = view.sort_values("exp_points", ascending=False).team.tolist()
st_echarts(
    {
        "tooltip": {"trigger": "axis", "order": "valueDesc" if not inverted else "valueAsc"},
        "legend": {"type": "scroll", "bottom": 0, "data": order},
        "grid": {"left": "3%", "right": "5%", "top": 30, "bottom": 60, "containLabel": True},
        "xAxis": {"type": "category", "data": [str(w) for w in weeks],
                  "name": "Matchweek", "boundaryGap": False},
        "yAxis": {"type": "value", "name": y_title, "inverse": inverted,
                  **({"min": 1, "max": 20} if inverted else {})},
        "series": [
            {
                "name": t,
                "type": "line",
                "smooth": True,
                "showSymbol": False,
                "lineStyle": {"width": 2.5 if i < 4 else 1.4},
                "emphasis": {"focus": "series"},
                "data": (vals := [round(v, 2) for v in
                         traj_view[traj_view.team == t].sort_values("matchweek")[col]]),
                # Club crest at the end of each line, so 20 colours don't
                # have to be memorised -- just follow the badge to week 38.
                "markPoint": {
                    "symbol": logos.logo_data_uri(t) or "circle",
                    "symbolSize": 22,
                    "itemStyle": {"color": "transparent"},
                    "label": {"show": False},
                    "data": [{"coord": [len(weeks) - 1, vals[-1]]}],
                },
            }
            for i, t in enumerate(order)
        ],
        "color": dd.club_colours(order),
    },
    height="520px",
    key="season_trend",
)

# ---------------------------------------------------------------- raw spread
st.subheader(":material/candlestick_chart: How wide is the uncertainty, really?")
st.caption(
    f"{_drama_short}'s season swings {_drama.points_p90 - _drama.points_p10:.0f} points "
    f"— {_drama.points_p10:.0f} in the worst tenth, {_drama.points_p90:.0f} in the "
    "best. That is "
    "the difference between the record for fewest points ever and a comfortable "
    f"mid-table finish, and it is the *same team* both times. Nobody can tell you which "
    f"{_drama_short} shows up; anyone claiming otherwise is guessing. The box holds the middle "
    "half of their seasons, the line is the median, the whiskers reach the realistic "
    "extremes."
)

sim_teams, draws = dd.draws()
box_order = view.sort_values("exp_points", ascending=False).team.tolist()
box_data = []
for t in box_order:
    v = draws[:, sim_teams.index(t)]
    q1, med, q3 = (float(x) for x in np.percentile(v, [25, 50, 75]))
    iqr = q3 - q1
    lo = float(v[v >= q1 - 1.5 * iqr].min())
    hi = float(v[v <= q3 + 1.5 * iqr].max())
    box_data.append([round(lo), round(q1), round(med), round(q3), round(hi)])

st_echarts(
    {
        "tooltip": {"trigger": "item"},
        "grid": {"left": "3%", "right": "5%", "top": 30, "bottom": 90, "containLabel": True},
        # Crests as axis labels, via ECharts rich text: each category gets a
        # style named "l<index>" whose background is that club's thumbnail.
        # The formatter has to be JS because a plain string can't pick a
        # different style per category.
        "xAxis": {
            "type": "category",
            "data": box_order,
            "axisLabel": {
                "formatter": JsCode(
                    "function (value, index) { return '{l' + index + '|}'; }"
                ),
                "rich": {
                    f"l{i}": {"height": 22, "width": 22, "align": "center",
                              "backgroundColor": {"image": logos.logo_data_uri(t) or ""}}
                    for i, t in enumerate(box_order)
                },
            },
        },
        "yAxis": {"type": "value", "name": "Season points"},
        "series": [{
            "type": "boxplot",
            # Each box carries its own club's colour.
            "data": [
                {"value": v, "itemStyle": {"color": dd.club_colour(t) + "45",
                                           "borderColor": dd.club_colour(t),
                                           "borderWidth": 2}}
                for t, v in zip(box_order, box_data)
            ],
        }],
    },
    height="480px",
    key="points_boxplot",
)

# ---------------------------------------------------------------- projection
st.subheader(":material/leaderboard: Who finishes where?")
st.caption(
    f"{_w1.team} have played three full Premier League seasons and the model still "
    f"can't place them: their bar spans {_w1.s:.0f} points, as wide as a club that has "
    "never played in this division at all. New manager, half a new squad — the record stops "
    "being evidence. The dot is the expected finish; the bar covers 8 seasons in 10. "
    "Long bar means the honest answer is *we don't know yet*."
)

ordered = view.sort_values("exp_points")
st_echarts(
    {
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
        "legend": {"bottom": 0},
        "grid": {"left": "3%", "right": "5%", "top": 30, "bottom": 40, "containLabel": True},
        "xAxis": {"type": "value", "name": "Points", "max": 100},
        "yAxis": {"type": "category", "data": ordered.team.tolist()},
        "series": [
            {
                "name": "Lower bound",
                "type": "bar",
                "stack": "range",
                "itemStyle": {"color": "transparent"},
                "data": [round(v, 1) for v in ordered.points_p10],
                "tooltip": {"show": False},
            },
            {
                "name": "80% range",
                "type": "bar",
                "stack": "range",
                "itemStyle": {"color": dd.MUTED, "opacity": 0.35, "borderRadius": 4},
                "data": [round(hi - lo, 1) for lo, hi in
                         zip(ordered.points_p10, ordered.points_p90)],
            },
            {
                "name": "Expected points",
                "type": "scatter",
                "symbolSize": 14,
                "data": [
                    {"value": [round(p, 1), t],
                     "itemStyle": {"color": dd.club_colour(t),
                                   "borderColor": dd.INK, "borderWidth": 1}}
                    for p, t in zip(ordered.exp_points, ordered.team)
                ],
            },
        ],
    },
    height="640px",
    key="points_range",
)

# ---------------------------------------------------------------- position map
st.subheader(":material/grid_on: The position map")
st.caption(
    f"Read each row as one club's twenty possible futures. {_fav.team}'s darkness pools "
    f"in the first two columns and thins out fast. {_w1.team}'s smears across ten. A row "
    "that never gets dark anywhere is a club nobody should be confident about — the "
    "model is telling you it has no strong opinion, which is more useful than a "
    "fake one."
)

heat_teams = view.sort_values("exp_rank").team.tolist()
heat_data = [
    [p - 1, i, round(grid.loc[t].iloc[p - 1] * 100, 1)]
    for i, t in enumerate(reversed(heat_teams))
    for p in range(1, 21)
]
st_echarts(
    {
        "tooltip": {"position": "top"},
        "grid": {"left": "3%", "right": "5%", "top": 40, "bottom": 70, "containLabel": True},
        "xAxis": {"type": "category", "data": [str(i) for i in range(1, 21)],
                  "name": "Final position", "splitArea": {"show": True}},
        "yAxis": {"type": "category", "data": list(reversed(heat_teams)),
                  "splitArea": {"show": True}},
        "visualMap": {
            "min": 0, "max": 50, "calculable": True, "orient": "horizontal",
            "left": "center", "bottom": 5,
            "inRange": {"color": ["#F3F5F7", dd.PLOT, dd.INK]},
            "text": ["often", "rarely"],
        },
        "series": [{
            "type": "heatmap",
            "data": heat_data,
            "label": {"show": False},
            "emphasis": {"itemStyle": {"shadowBlur": 10, "shadowColor": "rgba(0,0,0,.4)"}},
        }],
    },
    height=f"{max(420, 34 * len(heat_teams))}px",
    key="position_map",
)

# ---------------------------------------------------------------- club profile
st.subheader(":material/radar: What each forecast is built from")
st.caption("Five traits behind every forecast, scaled 0–100 against the rest of the league.")

selected = st.multiselect(
    "Clubs on the radar",
    options=view.team.tolist(),
    default=view.sort_values("exp_points", ascending=False).team.head(4).tolist(),
    max_selections=6,
    key="radar_clubs",
    bind="query-params",
)

if selected:
    axes = {
        "Attack": "attack",
        "Defence": "defence",
        "Squad depth": "depth_ratio",
        "Squad value": "squad_value_m",
        "Model confidence": "sigma",
    }
    norm = pd.DataFrame({"team": merged.team})
    for label, col in axes.items():
        v = merged[col].astype(float)
        if col == "sigma":  # lower sigma = more confident, so this axis is flipped
            v = v.max() - v
        span = v.max() - v.min()
        norm[label] = 100 * (v - v.min()) / span if span else 50.0
    norm = norm.set_index("team")

    st_echarts(
        {
            "tooltip": {"trigger": "item"},
            "legend": {"bottom": 0, "type": "scroll"},
            "radar": {
                "indicator": [{"name": a, "max": 100} for a in axes],
                "radius": "68%",
                "splitArea": {"areaStyle": {"opacity": 0.04}},
            },
            "series": [{
                "type": "radar",
                "areaStyle": {"opacity": 0.12},
                "lineStyle": {"width": 2},
                "data": [
                    {"name": c, "value": [round(norm.loc[c, a], 1) for a in axes]}
                    for c in selected
                ],
            }],
            "color": dd.club_colours(selected),
        },
        height="480px",
        key="radar_profile",
    )
else:
    st.info("Pick at least one club for the radar.", icon=":material/info:")

# ---------------------------------------------------------------- strength
st.subheader(":material/scatter_plot: Attack vs. defence")
st.caption(
    "Titles live in the top-right corner: score a lot, concede little. Almost nobody "
    "is there. Most of the league clusters in the middle, which is exactly why most "
    "seasons come down to a handful of matches. Crest size is squad value — note how "
    "poorly it predicts position on its own. The dashed rings are the three promoted "
    "sides, sitting where you would expect."
)

def _bubble(r):
    return max(24, (r.squad_value_m / view.squad_value_m.max()) * 60)

st_echarts(
    {
        "tooltip": {"trigger": "item", "formatter": "{b}"},
        "grid": {"left": "3%", "right": "6%", "top": 30, "bottom": 50, "containLabel": True},
        "xAxis": {"type": "value", "name": "Attack", "nameLocation": "middle",
                  "nameGap": 28, "splitLine": {"lineStyle": {"opacity": 0.35}}},
        "yAxis": {"type": "value", "name": "Defence", "nameLocation": "middle",
                  "nameGap": 38, "splitLine": {"lineStyle": {"opacity": 0.35}}},
        "series": [
            {
                # A dashed ring behind the crest, promoted clubs only --
                # status is marked by shape, not by overwriting club colour.
                "name": "Promoted clubs",
                "type": "scatter",
                "symbol": "circle",
                "itemStyle": {"color": "transparent", "borderColor": dd.CORAL,
                              "borderWidth": 2, "borderType": "dashed"},
                "symbolSize": [_bubble(r) + 14 for r in view.itertuples(index=False)
                               if r.source == "promoted"],
                "data": [[round(r.attack, 3), round(r.defence, 3)]
                         for r in view.itertuples(index=False) if r.source == "promoted"],
                "tooltip": {"show": False},
                "silent": True,
            },
            {
                "name": "Clubs",
                "type": "scatter",
                "data": [
                    {
                        "name": r.team,
                        "value": [round(r.attack, 3), round(r.defence, 3)],
                        "symbol": logos.logo_data_uri(r.team) or "circle",
                        "symbolSize": _bubble(r),
                    }
                    for r in view.itertuples(index=False)
                ],
            },
        ],
    },
    height="520px",
    key="attack_defence",
)

# ---------------------------------------------------------------- squad value
st.subheader(":material/account_tree: Where the money actually went")
st.caption(
    "Every player in the league, sized by what he is worth. Click a club to break it "
    "open by position — and notice how lopsided most squads are once you do. A club "
    "can look wealthy in total while its money sits in one line and its cover somewhere "
    "else entirely. That imbalance is what shows up in December."
)

POS_LINE = {p: line for line, poss in config.POSITION_LINES.items() for p in poss}
tree = []
for club in view.team:
    sub = squads[club].copy()
    sub["line"] = sub["position"].map(POS_LINE).map(dd.LINE_ID)
    children = []
    for line, grp in sub.groupby("line"):
        children.append({
            "name": line,
            "children": [
                {"name": r.player, "value": round(r.market_value_m, 1)}
                for r in grp.sort_values("market_value_m", ascending=False).itertuples(index=False)
            ],
        })
    tree.append({"name": club, "children": children})

st_echarts(
    {
        "tooltip": {"formatter": "{b}: €{c}m"},
        "series": [{
            "type": "treemap",
            "roam": False,
            "nodeClick": "zoomToNode",
            "breadcrumb": {"show": True, "bottom": 4},
            "levels": [
                {"itemStyle": {"borderColor": "#fff", "borderWidth": 3, "gapWidth": 3}},
                {"colorSaturation": [0.32, 0.62],
                 "itemStyle": {"borderColorSaturation": 0.7, "gapWidth": 2, "borderWidth": 2}},
                {"colorSaturation": [0.32, 0.55], "itemStyle": {"gapWidth": 1}},
            ],
            "data": tree,
        }],
    },
    height="560px",
    key="value_treemap",
)

# ---------------------------------------------------------------- schedule
st.subheader(":material/calendar_month: How packed is the calendar?")
st.caption(
    "Look at the dark cluster over Christmas. English football plays its heaviest "
    "month exactly when squads are most worn down, and there is no winter break to "
    "hide behind. A club with eleven good players and nothing behind them survives "
    "September comfortably and then loses the season in three weeks of December — "
    "which is precisely where the fatigue penalty bites hardest."
)

fixtures = pd.read_csv(config.FIXTURES_FILE, parse_dates=["date"])
per_day = fixtures.groupby(fixtures.date.dt.date).size()
cal = [[d.isoformat(), int(n)] for d, n in per_day.items()]
years = sorted({d.year for d in per_day.index})

st_echarts(
    {
        "tooltip": {"formatter": "{c0} matches"},
        "visualMap": {
            "min": 1, "max": int(per_day.max()), "calculable": True,
            "orient": "horizontal", "left": "center", "bottom": 5,
            "inRange": {"color": ["#E6EAEF", dd.TEAL, dd.INK]},
        },
        "calendar": [
            {"range": str(y), "cellSize": ["auto", 15], "top": 40 + i * 150,
             "left": 60, "right": 30, "yearLabel": {"show": True}}
            for i, y in enumerate(years)
        ],
        "series": [
            {"type": "heatmap", "coordinateSystem": "calendar", "calendarIndex": i,
             "data": [c for c in cal if c[0].startswith(str(y))]}
            for i, y in enumerate(years)
        ],
    },
    height=f"{80 + len(years) * 160}px",
    key="fixture_calendar",
)

st.info(
    f"**Where to actually look first.** Ignore the top of the table — {_fav.team} and "
    f"{_second.team} are the least interesting clubs here, because everyone already "
    f"agrees about them. The clubs worth watching are {_w1.team} and {_w2.team}, whose "
    f"bars span {_w1.s:.0f} and {_w2.s:.0f} points. They are the two the model knows "
    "least about, which means they are also the two that will move most when real "
    "results start landing. If you check back after ten matchweeks, theirs are the "
    "numbers that will have changed.",
    icon=":material/lightbulb:",
)

# ---------------------------------------------------------------- raw data
with st.expander("Raw data preview", icon=":material/table_view:"):
    st.dataframe(
        view[["team", "Zone", "Basis", "exp_points", "points_p10", "points_p90",
              "P_title", "P_top4", "P_bottom3", "attack", "defence",
              "depth_ratio", "squad_value_m", "sigma"]],
        hide_index=True, width="stretch",
        column_config={
            "team": "Club",
            "exp_points": st.column_config.NumberColumn("Points", format="%.1f"),
            "points_p10": st.column_config.NumberColumn("p10", format="%.0f"),
            "points_p90": st.column_config.NumberColumn("p90", format="%.0f"),
            "P_title": st.column_config.NumberColumn("Title", format="percent"),
            "P_top4": st.column_config.NumberColumn("Top 4", format="percent"),
            "P_bottom3": st.column_config.NumberColumn("Relegation", format="percent"),
            "attack": st.column_config.NumberColumn("Attack", format="%.3f"),
            "defence": st.column_config.NumberColumn("Defence", format="%.3f"),
            "depth_ratio": st.column_config.NumberColumn("Depth", format="%.2f"),
            "squad_value_m": st.column_config.NumberColumn("Squad value", format="%.0fm"),
            "sigma": st.column_config.NumberColumn("σ", format="%.3f"),
        },
    )
