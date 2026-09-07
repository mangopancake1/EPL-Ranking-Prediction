import math

import pandas as pd
import streamlit as st
from streamlit_echarts import JsCode

import dashboard_data as dd
import dashboard_logos as logos
from dashboard_data import st_echarts

table, grid, merged, _ = dd.load()

st.title(":material/leaderboard: Predicted Table")
st.markdown(
    "**This is the table that never happens.** No single season ends exactly here — "
    "it is the average of 10,000 of them, which makes it the best guess and, at the "
    "same time, a finish nobody will ever see. Read the range bars next to each club, "
    "not the points."
)

# --- table ---
st.subheader(":material/table_rows: Season projection")
st.caption(
    "One honest warning before you read it: the three promoted clubs are ordered using "
    "the single least-tested number in the whole model. Their gap to the rest of the "
    "league is measured from real history; the gap *between the three of them* is not. "
    "Trust that they are the bottom three far more than you trust which of them is 18th.\n\n"
    "The order is by **average finishing position** across 10,000 seasons, not by points, "
    "so a club can sit above another with a fraction of a point fewer. That is not a "
    "typo: finishing 8th in most seasons is worth more than a handful of points piled up "
    "in the few where everything goes right. The average-position column shows why."
)

view = table.copy()
view.insert(0, "#", range(1, len(view) + 1))
view["Crest"] = [logos.logo_data_uri(t) for t in view.team]
view["Range"] = [[r.points_p10, r.points_p90] for r in table.itertuples(index=False)]
view["Basis"] = view["source"].map({"fitted": "Proven in the PL", "promoted": "Newly promoted"})

st.dataframe(
    view[["#", "Crest", "team", "exp_points", "exp_rank", "Range",
          "P_title", "P_top4", "P_bottom3", "Basis"]],
    hide_index=True, width="stretch", height=740,
    column_config={
        "#": st.column_config.NumberColumn(width="small"),
        "Crest": st.column_config.ImageColumn(width="small"),
        "team": st.column_config.TextColumn("Club"),
        "exp_points": st.column_config.NumberColumn("Points", format="%.1f"),
        "exp_rank": st.column_config.NumberColumn(
            "Avg. position", format="%.2f",
            help="Mean finishing position across 10,000 simulated seasons. "
                 "This is what the table is sorted by."),
        "Range": st.column_config.BarChartColumn(
            "80% range", y_min=0, y_max=100,
            help="10th to 90th percentile across 10,000 simulations"),
        "P_title": st.column_config.ProgressColumn("Title", format="%.1f%%",
                                                   min_value=0, max_value=1),
        "P_top4": st.column_config.ProgressColumn("Top 4", format="%.1f%%",
                                                  min_value=0, max_value=1),
        "P_bottom3": st.column_config.ProgressColumn("Relegation", format="%.1f%%",
                                                     min_value=0, max_value=1),
        "Basis": st.column_config.TextColumn(
            "Basis", help="Proven in the PL = built from real match results; "
                          "Newly promoted = priced off past promoted clubs"),
    },
)

# --- single-club distribution ---
st.subheader(":material/bar_chart: One club, up close")
st.caption(
    "Along the bottom: every possible final position, 1st (champions) through "
    "20th (relegated). Height of each bar: how likely the model thinks that "
    "finish is, in %. The dashed line marks 5% — what every bar would show if "
    "the model had no opinion at all and treated all 20 spots as equally "
    "likely — so a bar rising clearly above the line is a real, confident "
    "prediction, not noise."
)

c_pick, c_logo = st.columns([5, 1], vertical_alignment="center")
pick = c_pick.selectbox("Club", grid.index.tolist(), key="club", bind="query-params")
logo_file = logos.logo_path(pick)
if logo_file:
    c_logo.image(str(logo_file), width=64)

series = (grid.loc[pick] * 100).values
# Club kit colour, with the top-4 and relegation zones picked out by opacity
# -- club identity stays readable without throwing away the zone information.
kit = dd.club_colour(pick)
colours = [kit if p <= 4 or p >= 18 else kit + "70" for p in range(1, 21)]

# Fixed to the highest bar any club ever needs, over the whole grid -- not
# just this club's own row. Letting ECharts auto-scale the axis per club
# would rescale the y-axis every time the selection changes, so the same
# bar height could mean a different % from one club to the next. The axis
# itself (what each side means, and its scale) has to stay identical no
# matter which club is picked; only the bars and their colour should move.
Y_MAX = math.ceil(grid.values.max() * 100 / 5) * 5

st_echarts(
    {
        "title": {"text": "Chance of finishing in each position", "left": "center",
                  "textStyle": {"fontSize": 13, "fontWeight": "normal"}},
        "tooltip": {
            "trigger": "item",  # fires on tap, not just mouse hover -- works on touchscreens
            "formatter": JsCode(
                "function (p) {"
                "  var v = p.value, diff = Math.round((v - 5) * 10) / 10;"
                "  var cmp = diff > 0 ? '+' + diff + ' vs. equal chance'"
                "          : diff < 0 ? diff + ' vs. equal chance'"
                "          : 'exactly equal chance';"
                "  return 'Position ' + p.name + ': ' + v + '%<br/>' + cmp;"
                "}"
            ),
        },
        "grid": {"left": "3%", "right": "4%", "top": 60, "bottom": 50, "containLabel": True},
        "xAxis": {"type": "category", "data": [str(i) for i in range(1, 21)],
                  "name": "Final league position (1st = champions, 20th = relegated)",
                  "nameLocation": "middle", "nameGap": 28},
        "yAxis": {"type": "value", "name": "Chance of that finish (%)", "max": Y_MAX,
                  "nameLocation": "middle", "nameGap": 40},
        "series": [
            {
                "type": "bar",
                "data": [{"value": round(v, 2), "itemStyle": {"color": c, "borderRadius": [4, 4, 0, 0]}}
                         for v, c in zip(series, colours)],
                "markLine": {
                    "silent": True,
                    "symbol": "none",
                    "lineStyle": {"type": "dashed", "color": dd.MUTED},
                    "data": [{"yAxis": 5, "label": {
                        "formatter": "if every position were equally likely (5%)"}}],
                },
            }
        ],
    },
    height="400px",
    key="club_distribution",
)

r = merged[merged.team == pick].iloc[0]
c = st.columns(5)
c[0].metric("Expected points", f"{r.exp_points:.1f}")
c[1].metric("80% range", f"{r.points_p10:.0f}–{r.points_p90:.0f}", delta_color="off")
c[2].metric("Title chance", f"{r.P_title:.1%}", delta_color="off")
c[3].metric("Top 4 chance", f"{r.P_top4:.1%}", delta_color="off")
c[4].metric("Relegation chance", f"{r.P_bottom3:.1%}", delta_color="inverse")

# --- depth ---
st.subheader(":material/groups: Squad depth")
st.caption(
    "Depth is not squad size — these squads run from 21 players to 35. It is what "
    "happens when the best one in a position gets injured. Each line is scored twice, "
    "against the club's "
    "own first choice and against the league's standard, and the lower score wins: a "
    "squad where everyone is equally cheap is not deep, it is uniformly thin."
)

d = merged.sort_values("depth_ratio", ascending=False)
st_echarts(
    {
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
        "legend": {"bottom": 0},
        "grid": {"left": "3%", "right": "5%", "top": 30, "bottom": 90, "containLabel": True},
        "xAxis": {"type": "category", "data": d.team.tolist(), "axisLabel": {"rotate": 40}},
        "yAxis": {"type": "value", "name": "Depth (0–1)"},
        "series": [
            {"name": line, "type": "bar",
             "itemStyle": {"color": colour, "borderRadius": [3, 3, 0, 0]},
             "data": [round(v, 3) for v in d[line]]}
            for line, colour in zip(dd.LINES, [dd.MUTED, dd.PLOT, dd.TEAL, dd.GOLD])
        ],
    },
    height="440px",
    key="depth_by_line",
)

with st.expander("Raw data preview", icon=":material/table_view:"):
    st.dataframe(
        merged[["team", "depth_ratio", "weakest_line"] + dd.LINES],
        hide_index=True, width="stretch",
        column_config={
            "team": "Club",
            # Scaled to the data, not a guessed ceiling: Arsenal reached 0.565
            # against a hardcoded max of 0.5 and its bar simply overflowed.
            "depth_ratio": st.column_config.ProgressColumn(
                "Total depth", format="%.2f", min_value=0.0,
                max_value=float(merged.depth_ratio.max())),
            "weakest_line": "Thinnest line",
            **{l: st.column_config.NumberColumn(l, format="%.2f") for l in dd.LINES},
        },
    )
