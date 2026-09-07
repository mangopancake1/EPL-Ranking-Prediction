import altair as alt
import pandas as pd
import streamlit as st

import config
import dashboard_data as dd
from dashboard_data import st_echarts

table, grid, merged, _ = dd.load()

# Counted, never typed. Every figure this page used to assert by hand had gone
# stale: the manager count, the number of new appointments, the backtest verdict.
N_PRIORS = int(merged.prior_ppg.notna().sum())
N_NEW = sum(1 for m in config.MANAGERS.values()
            if m.get("status") in ("new", "interim_to_full"))

st.title(":material/query_stats: Why the Model Thinks This")
st.markdown(
    "**A forecast you can't argue with is a forecast you shouldn't trust.**\n\n"
    "So here is where every number came from. Almost all of it is just results: three "
    "seasons of real Premier League matches, weighted so last spring counts more than "
    "two winters ago. Everything on this page is the small remainder — the handful of "
    "things that already happened but the results haven't caught up with. A manager "
    "who arrived in June. A squad half-rebuilt in August. Those are the only licences "
    "the model has to disagree with the recent past."
)

cols = st.columns([1, 3], vertical_alignment="center")

with cols[0].container(border=True):
    st.metric("Reset furthest toward average",
              merged.sort_values("changeover_shrink").iloc[-1].team,
              "new manager, little history here", delta_color="off")
    st.metric("Biggest new-manager boost",
              merged.assign(a=merged.manager_delta.abs()).sort_values("a").iloc[-1].team,
              "track record moved the needle most", delta_color="off")
    st.metric("Hardest to call",
              merged.sort_values("sigma").iloc[-1].team,
              "widest range of outcomes", delta_color="off")

with cols[1].container(border=True):
    mgr = merged[merged.manager_delta.abs() > 1e-9].copy()
    if mgr.empty:
        st.info("No club received a manager adjustment on this run.")
    else:
        long = mgr.melt(
            id_vars=["team", "manager"],
            value_vars=["manager_delta_atk", "manager_delta_def"],
            var_name="Side", value_name="Delta",
        )
        long["Side"] = long["Side"].map({"manager_delta_atk": "Attack",
                                         "manager_delta_def": "Defence"})
        order = mgr.sort_values("manager_delta").team.tolist()
        chart = alt.Chart(long).mark_bar().encode(
            alt.Y("team:N", sort=order, title=None),
            alt.X("Delta:Q", title="Impact on club strength", axis=alt.Axis(labels=False)),
            alt.Color("Side:N", scale=alt.Scale(domain=["Attack", "Defence"],
                                                range=[dd.GOLD, dd.TEAL]),
                      legend=alt.Legend(orient="top", title=None)),
            tooltip=["team", "manager", "Side"],
        ).properties(height=440, title="Manager adjustment, by club")
        st.altair_chart(chart, width="stretch")
        st.caption(
            "Notice how few clubs appear here at all. A manager who has been in the "
            "job two years doesn't get a bar, because his record *is* the club's "
            "record — the results already contain him, and adding his reputation on "
            "top would be counting the same man twice. Only someone the fit has never "
            "seen in this dugout earns an adjustment."
        )

""
""

st.subheader(":material/stacked_bar_chart: What moved each club away from its own results")
st.caption(
    "Every club starts at what three seasons of real matches earned it. These are the "
    "three things allowed to move it from there — how good the current squad actually is, "
    "how much of it is brand new to this league, and who is now in the dugout. Bars to the "
    "right made a club stronger than its results alone; to the left, weaker. A club with "
    "no bars is one the results already describe."
)

adj = merged[merged.source == "fitted"].copy()
adj["squad rating"] = adj["rating_shift"] * 2      # applied to attack and defence alike
adj["new to the league"] = -adj["adaptation_drag"] * 2
adj["manager"] = adj["manager_delta"]

# A component that is identical for every club separates nobody, and drawing it
# as a bar implies it does. "New to the league" collapsed to a flat -0.002 once
# nearly every arrival gained a rating from the career file, so it is dropped
# rather than shown as a driver it no longer is.
COMPONENTS = [("squad rating", dd.TEAL), ("new to the league", dd.CORAL), ("manager", dd.GOLD)]
FLAT = [c for c, _ in COMPONENTS if adj[c].std() < 1e-9]
COMPONENTS = [(c, colour) for c, colour in COMPONENTS if c not in FLAT]

adj["total"] = sum(adj[c] for c, _ in COMPONENTS)
adj = adj.sort_values("total")

if FLAT:
    st.caption(
        f"_{', '.join(FLAT).capitalize()} is not drawn: it currently works out the same "
        f"for all {len(adj)} clubs, so it moves everyone together and separates nobody. It stays "
        "in the model, but showing it as a bar would imply a distinction that isn't there._"
    )

with st.container(border=True):
    st_echarts(
        {
            "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
            "legend": {"bottom": 0},
            "grid": {"left": "3%", "right": "6%", "top": 30, "bottom": 50, "containLabel": True},
            "xAxis": {"type": "value", "name": "Change in combined strength",
                      "nameLocation": "middle", "nameGap": 28},
            "yAxis": {"type": "category", "data": adj.team.tolist()},
            "series": [
                {"name": label, "type": "bar", "stack": "total",
                 "itemStyle": {"color": colour},
                 "data": [round(v, 4) for v in adj[label]]}
                for label, colour in COMPONENTS
            ],
        },
        height="620px",
        key="adjustment_breakdown",
    )

""
""

st.subheader(":material/rule: What's real data, and what's still a best guess")
st.caption(
    "Most forecasts don't show you this table, which is the main reason to distrust "
    "them. Five of these inputs are measured from things that actually happened. Three "
    "are reasoned estimates that nobody — not this model, not anyone — currently has "
    "the history to pin down. They are labelled rather than buried."
)

st.dataframe(
    pd.DataFrame([
        ("Match results & schedule", "Real data",
         "1,140 real Premier League results; the official 2026/27 fixture list"),
        ("Squad market values", "Real data",
         "Full squads for all 20 clubs, cross-checked against real playing history"),
        ("Manager track records", "Real data",
         f"{N_PRIORS} of {len(config.TEAMS)} clubs have a manager with a real "
         "win-draw-loss record to price; the rest have been in the job long enough "
         "that the club's own results already contain them"),
        ("Promoted-club level", "Real data",
         "Based on how every promoted club has actually performed over the last three seasons"),
        ("Squad depth", "Real data",
         "Real market values per position group, judged against the league's own standard"),
        ("Ranking the 3 promoted clubs against each other", "Best guess",
         "There isn't enough history yet to know exactly how much squad value should "
         "separate them — the shakiest part of the whole model"),
        ("New-manager boost", "Best guess",
         "Nothing pins down exactly how much of a club's results are down to the "
         "manager versus the players, so this is a reasoned assumption"),
        ("Busy-schedule & thin-squad penalties", "Best guess",
         "How much a packed fixture list or a shallow squad should cost a club "
         "is a judgment call, not yet backed by enough real data"),
    ], columns=["What it's about", "Status", "In plain terms"]),
    hide_index=True, width="stretch",
    column_config={"Status": st.column_config.TextColumn(width="small")},
)

st.warning(
    "**The most important sentence on this site: this model has not clearly beaten "
    "copying last season's table.**\n\n"
    "Tested against the real 2024/25 and 2025/26 finishes, the naive approach — take "
    "last year's standings, put the promoted clubs at the bottom, go home — won 2024/25 "
    "outright and finished closer on average position in both seasons. The model edged "
    "it on rank correlation in 2025/26 (0.58 against 0.57) and nowhere else. That is a "
    "genuinely uncomfortable result and it is printed here rather than left out.\n\n"
    "The defence is narrow but real: both test seasons were quiet ones for managerial "
    f"change, and pricing in a new manager is the main thing this model does that the "
    f"naive table can't. This season has {N_NEW} new appointments. That is the test the "
    "backtest never got to run — and the reason to check back in May rather than take "
    "any of this on faith now.",
    icon=":material/science:",
)
