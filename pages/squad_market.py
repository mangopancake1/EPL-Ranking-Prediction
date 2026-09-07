import numpy as np
import pandas as pd
import streamlit as st
from streamlit_echarts import JsCode

import config
import dashboard_data as dd
import dashboard_logos as logos
from src import player_ratings as pr
from dashboard_data import st_echarts

table, grid, merged, squads = dd.load()
transfers = dd.transfers()
ratings = dd.player_ratings_table()

st.title(":material/swap_horiz: Squad & Market")

# Every figure in the copy below is computed, not typed. The last version of
# this page quoted a fee, a headcount and a league position that had all moved
# on since they were written.
_T = set(config.TEAMS)
_intra = transfers[transfers["from"].isin(_T) & transfers.to.isin(_T) & transfers.fee_m.notna()]
_intra = _intra.drop_duplicates(["player", "from", "to"]).nlargest(1, "fee_m").iloc[0]
_moved = sum(
    len({a["player"] for a in pr.roster_delta(t)[1]} & set(squads[t].player))
    for t in config.TEAMS
)

st.markdown(
    f"**{_intra.player} is worth €{_intra.fee_m:.0f}m, and until recently this model "
    f"didn't notice he had joined {_intra.to}.**\n\n"
    "Not a bug so much as the central problem with predicting football from results: "
    "a club's record is a record of the squad that played it. He earned his rating at "
    f"{_intra['from']}, so {_intra.to}'s three seasons of matches contain no trace of "
    f"him — and {_moved} players across the league moved this summer under exactly the "
    "same blind spot.\n\n"
    "This page is the correction. Who was actually bought and sold, how good the "
    "players now in each squad really are, and which clubs the results are therefore "
    "describing least well."
)

# --- KPIs -------------------------------------------------------------------
spend = transfers[transfers.in_epl & transfers.fee_m.notna()].groupby("to").fee_m.sum()
sales = transfers[transfers.out_epl & transfers.fee_m.notna()].groupby("from").fee_m.sum()
net = (spend.reindex(config.TEAMS).fillna(0) - sales.reindex(config.TEAMS).fillna(0)).sort_values()

k = st.columns(4)
k[0].metric("Biggest net spend", net.idxmin() if net.min() < 0 else net.idxmax(),
            f"€{abs(net.min()) if net.min() < 0 else net.max():.0f}m", delta_color="off")
k[1].metric("Total moved", f"€{transfers.fee_m.sum():,.0f}m",
            f"{len(transfers)} transfers", delta_color="off")
best_rated = merged.dropna(subset=["squad_rating"]).sort_values("squad_rating").iloc[-1]
k[2].metric("Best squad rating", best_rated.team, f"{best_rated.squad_rating:.2f}", delta_color="off")
k[3].metric("Most corrected by it",
            merged.assign(a=merged.rating_shift.abs()).sort_values("a").iloc[-1].team,
            "results vs squad disagree most", delta_color="off")

st.divider()

# --- 1. squad rating vs fitted strength -------------------------------------
st.subheader(":material/scatter_plot: Do the results match the squad?")

# The line is fitted the way the model fits it: over the clubs that HAVE a
# record, against the strength their record earned BEFORE any adjustment.
# Plotting the final strength instead would be circular -- the rating shift has
# already pulled each club toward the line the chart then draws, which made an
# honest r of 0.86 read as 0.91.
fitted = merged[(merged.source == "fitted") & merged.squad_rating.notna()].copy()
promoted = merged[(merged.source != "fitted") & merged.squad_rating.notna()].copy()
x, y = fitted.squad_rating.to_numpy(), fitted.fitted_combined.to_numpy()
slope, intercept = np.polyfit(x, y, 1)
r = float(np.corrcoef(x, y)[0, 1])

_rank_rating = fitted.squad_rating.rank(ascending=False).astype(int)
_rank_record = fitted.fitted_combined.rank(ascending=False).astype(int)
_gap = (_rank_record - _rank_rating)
_under = fitted.team[_gap.idxmax()]           # squad far better than record
_over = fitted.team[_gap.idxmin()]            # record far better than squad
_ordinal = lambda n: f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"

st.caption(
    "The single most useful chart here. Horizontal: how good each squad's players "
    "actually are, from real ratings. Vertical: how good the club's own results say it "
    "is, before the model adjusts anything. The dashed line is the relationship across "
    "the league — and the interesting clubs are the ones nowhere near it.\n\n"
    f"Sitting **below** the line means the squad looks stronger than the results — "
    f"{_under} are the clearest case, with the "
    f"{_ordinal(int(_rank_rating[_gap.idxmax()]))}-best squad of these "
    f"{len(fitted)} clubs and the {_ordinal(int(_rank_record[_gap.idxmax()]))}-best "
    f"record. **Above** it means the opposite: results flattering the players, which "
    f"is where {_over} sit. The forecast pulls every club "
    f"{1 - config.SQUAD_RATING_BLEND:.0%} of the way back toward this line.\n\n"
    "The three promoted clubs are marked separately. They have squad ratings but no "
    "Premier League record to plot against, so they sit on the line rather than "
    "helping to define it."
)

line_x = [float(x.min()), float(x.max())]

with st.container(border=True):
    st_echarts(
        {
            "tooltip": {"trigger": "item", "formatter": "{b}"},
            "grid": {"left": "3%", "right": "6%", "top": 40, "bottom": 50, "containLabel": True},
            "xAxis": {"type": "value", "name": "Squad rating (real player ratings)",
                      "nameLocation": "middle", "nameGap": 30,
                      "min": round(float(merged.squad_rating.min()) - 0.05, 2),
                      "max": round(float(merged.squad_rating.max()) + 0.05, 2),
                      "splitLine": {"lineStyle": {"opacity": 0.35}}},
            "yAxis": {"type": "value", "name": "Strength earned on the pitch",
                      "nameLocation": "middle", "nameGap": 45,
                      "splitLine": {"lineStyle": {"opacity": 0.35}}},
            "series": [
                {
                    "name": "What the league implies",
                    "type": "line",
                    "showSymbol": False,
                    "lineStyle": {"type": "dashed", "color": dd.MUTED, "width": 2},
                    "data": [[round(v, 3), round(slope * v + intercept, 3)] for v in line_x],
                    "tooltip": {"show": False},
                    "silent": True,
                },
                {
                    "name": "Clubs",
                    "type": "scatter",
                    "data": [
                        {"name": r_.team,
                         "value": [round(r_.squad_rating, 3), round(r_.fitted_combined, 3)],
                         "symbol": logos.logo_data_uri(r_.team) or "circle",
                         "symbolSize": 34}
                        for r_ in fitted.itertuples(index=False)
                    ],
                },
                {
                    "name": "Promoted (no PL record)",
                    "type": "scatter",
                    "symbol": "diamond",
                    "symbolSize": 16,
                    "itemStyle": {"color": dd.MUTED, "opacity": 0.85},
                    "data": [
                        {"name": f"{r_.team} (promoted)",
                         "value": [round(r_.squad_rating, 3),
                                   round(slope * r_.squad_rating + intercept, 3)]}
                        for r_ in promoted.itertuples(index=False)
                    ],
                },
            ],
        },
        height="520px",
        key="rating_vs_strength",
    )
    st.caption(f"Correlation r = {r:.2f} — one rating point is worth about "
               f"{slope:.2f} in combined strength, measured from these {len(fitted)} "
               "clubs, not assumed.")

""

# --- 2. transfer flows ------------------------------------------------------
st.subheader(":material/account_tree: Where the money went")
st.caption(
    "The twenty biggest deals of the summer, thickest band = biggest fee. Follow any "
    "strand from left (who sold) to right (who bought). Most of the width never leaves "
    "the picture: the Premier League's expensive habit is largely buying from itself, "
    "which is why a single player can appear on both sides of the same window."
)

top_moves = transfers[transfers.fee_m.notna()].nlargest(20, "fee_m")
nodes, links = {}, []
# "from" is a Python keyword, so this reads the columns directly rather than
# through itertuples, where pandas would rename it to a positional _0.
for seller, buyer, fee in zip(top_moves["from"], top_moves["to"], top_moves["fee_m"]):
    src, dst = f"{seller} ", f" {buyer}"  # padded so a club that both sells and buys stays two nodes
    nodes.setdefault(src, {"name": src})
    nodes.setdefault(dst, {"name": dst})
    links.append({"source": src, "target": dst, "value": round(float(fee), 1)})

with st.container(border=True):
    st_echarts(
        {
            "tooltip": {"trigger": "item", "triggerOn": "mousemove",
                        "formatter": "{b}: €{c}m"},
            "series": [{
                "type": "sankey",
                "left": "3%", "right": "16%", "top": "3%", "bottom": "3%",
                "data": list(nodes.values()),
                "links": links,
                "emphasis": {"focus": "adjacency"},
                "lineStyle": {"color": "gradient", "curveness": 0.5, "opacity": 0.45},
                "label": {"fontSize": 11},
            }],
        },
        height="620px",
        key="transfer_sankey",
    )

""

# --- 3. net spend -----------------------------------------------------------
st.subheader(":material/payments: Net spend, club by club")
st.caption(
    "Paid minus received. Bars to the right bought more than they sold; to the left, the "
    "other way. Worth holding this next to the scatter above: heavy spending and a squad "
    "the results haven't recognised yet are usually the same story told twice.\n\n"
    "Free transfers, loans and undisclosed fees have no number attached, so they are "
    "left out rather than quietly counted as zero — a club can be busier than its bar "
    "suggests."
)

net_df = net.reset_index()
net_df.columns = ["team", "net_m"]
net_df = net_df.sort_values("net_m")

with st.container(border=True):
    st_echarts(
        {
            "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"},
                        "valueFormatter": JsCode("function (v) { return '€' + v + 'm'; }")},
            "grid": {"left": "3%", "right": "5%", "top": 30, "bottom": 90, "containLabel": True},
            "xAxis": {"type": "category", "data": net_df.team.tolist(),
                      "axisLabel": {"rotate": 40}},
            "yAxis": {"type": "value", "name": "Net spend (€m)"},
            "series": [{
                "type": "bar",
                "data": [
                    {"value": round(v, 1),
                     "itemStyle": {"color": dd.CORAL if v > 0 else dd.TEAL,
                                   "borderRadius": [3, 3, 0, 0] if v > 0 else [0, 0, 3, 3]}}
                    for v in net_df.net_m
                ],
                "markLine": {"silent": True, "symbol": "none",
                             "lineStyle": {"color": dd.MUTED},
                             "data": [{"yAxis": 0, "label": {"formatter": "break even"}}]},
            }],
        },
        height="440px",
        key="net_spend",
    )

""

# --- 4. rating spread per club ---------------------------------------------
st.subheader(":material/groups_2: How good is each squad, player by player")
st.caption(
    "Averages hide the thing that decides seasons. What matters is not a squad's best "
    "player but its *floor* — the level it drops to when three regulars are injured in "
    "November. Read the bottom of each box, not the top: a club with a high floor keeps "
    "collecting points through a bad month, and a club with a long tail below the box "
    "does not.\n\n"
    "Promoted clubs look sparse here for an honest reason: most of their players have "
    "never played a Premier League minute, so there is no rating to show. Absence of "
    "data, not absence of quality."
)

order = (ratings.groupby("team").rating.median().sort_values(ascending=False).index.tolist())
box_data, counts = [], []
for t in order:
    v = ratings[ratings.team == t].rating.to_numpy()
    q1, med, q3 = (float(z) for z in np.percentile(v, [25, 50, 75]))
    iqr = q3 - q1
    lo = float(v[v >= q1 - 1.5 * iqr].min())
    hi = float(v[v <= q3 + 1.5 * iqr].max())
    box_data.append([round(lo, 2), round(q1, 2), round(med, 2), round(q3, 2), round(hi, 2)])
    counts.append(len(v))

with st.container(border=True):
    st_echarts(
        {
            "tooltip": {"trigger": "item"},
            "grid": {"left": "3%", "right": "5%", "top": 30, "bottom": 95, "containLabel": True},
            "xAxis": {"type": "category", "data": order, "axisLabel": {"rotate": 40}},
            "yAxis": {"type": "value", "name": "Player rating", "scale": True},
            "series": [{
                "type": "boxplot",
                "data": [
                    {"value": v, "itemStyle": {"color": dd.club_colour(t) + "45",
                                               "borderColor": dd.club_colour(t), "borderWidth": 2}}
                    for t, v in zip(order, box_data)
                ],
            }],
        },
        height="460px",
        key="rating_boxes",
    )
    st.caption("Players covered per club: " + ", ".join(
        f"{t} {n}" for t, n in sorted(zip(order, counts), key=lambda z: -z[1])[:6]) + " …")

""

# --- 5. midweek load --------------------------------------------------------
st.subheader(":material/calendar_month: Who actually plays midweek")
st.caption(
    "Qualifying for Europe is a reward that arrives disguised as a punishment. These are "
    "the real dates — actual UEFA and EFL Cup fixtures, not an assumed calendar — and "
    "the darker a club's row, the more often it plays Thursday and then Sunday while a "
    "rival rests all week.\n\n"
    "The clubs with the emptiest rows have a quiet advantage nobody celebrates: a full "
    "week to prepare, every week, all season."
)

mid = dd.midweek_dates()
mid_rows = []
for team, dates in mid.items():
    for d in dates:
        mid_rows.append({"team": team, "date": pd.Timestamp(d)})
mid_df = pd.DataFrame(mid_rows)

if not mid_df.empty:
    mid_df["month"] = mid_df.date.dt.strftime("%b %Y")
    months = sorted(mid_df.month.unique(), key=lambda m: pd.Timestamp(m))
    teams_by_load = mid_df.groupby("team").size().sort_values(ascending=False).index.tolist()
    heat = [
        [months.index(m), teams_by_load.index(t), int(n)]
        for (t, m), n in mid_df.groupby(["team", "month"]).size().items()
    ]
    with st.container(border=True):
        st_echarts(
            {
                "tooltip": {"position": "top", "formatter": "{c0} midweek matches"},
                "grid": {"left": "3%", "right": "5%", "top": 30, "bottom": 70, "containLabel": True},
                "xAxis": {"type": "category", "data": months, "splitArea": {"show": True}},
                "yAxis": {"type": "category", "data": teams_by_load, "splitArea": {"show": True}},
                "visualMap": {"min": 0, "max": int(mid_df.groupby(["team", "month"]).size().max()),
                              "calculable": True, "orient": "horizontal", "left": "center",
                              "bottom": 5, "inRange": {"color": ["#F3F5F7", dd.TEAL, dd.INK]}},
                "series": [{"type": "heatmap", "data": heat,
                            "label": {"show": True, "fontSize": 10},
                            "emphasis": {"itemStyle": {"shadowBlur": 8}}}],
            },
            height=f"{max(400, 26 * len(teams_by_load))}px",
            key="midweek_heat",
        )

""

st.info(
    "**One test you can run yourself in May.** Find the clubs sitting below the line on "
    "the first chart *and* on the spending side of the third. Those are the sides whose "
    "squads have outgrown their record — the model says they should climb, and it is "
    "betting real forecast points on it.\n\n"
    "If they finish where their old results suggested instead, this correction is worth "
    "less than it claims. That is a falsifiable prediction, which is more than most "
    "forecasts offer.",
    icon=":material/lightbulb:",
)

with st.expander("Raw data preview", icon=":material/table_view:"):
    tab1, tab2 = st.tabs(["Transfers", "Player ratings"])
    with tab1:
        st.dataframe(transfers.sort_values("fee_m", ascending=False), hide_index=True,
                     width="stretch", column_config={
                         "from": "From", "to": "To", "player": "Player",
                         "fee_m": st.column_config.NumberColumn("Fee", format="%.1fm"),
                         "in_epl": "Into the PL", "out_epl": "Out of the PL"})
    with tab2:
        st.dataframe(ratings.sort_values("rating", ascending=False), hide_index=True,
                     width="stretch", column_config={
                         "team": "Club", "player": "Player", "position": "Pos",
                         "market_value_m": st.column_config.NumberColumn("Value", format="%.0fm"),
                         "rating": st.column_config.NumberColumn("Rating", format="%.2f")})
