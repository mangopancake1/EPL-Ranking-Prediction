import math

import altair as alt
import pandas as pd
import streamlit as st

import dashboard_data as dd
import dashboard_valuable as valuable

table, grid, merged, squads = dd.load()

st.title(":material/compare_arrows: Compare Clubs")
st.markdown(
    "**Two clubs can share an expected finish and have almost nothing else in common.** "
    "One arrives there the same way every time; the other is a coin flip between Europe "
    "and a relegation scrap that happens to average out in the middle. Put them side by "
    "side here and the difference is obvious — it just never shows up in a league table."
)
st.caption("Your picks are saved in the URL, so any comparison here can be shared as a link.")

default = dd.clubs_from_query("clubs", table.team.head(3).tolist())

picked = st.multiselect(
    "Clubs to compare",
    options=table.team.tolist(),
    default=default,
    max_selections=6,
    placeholder="Pick clubs to compare",
)
dd.sync_query("clubs", picked)

if not picked:
    st.info("Pick at least one club.", icon=":material/info:")
    st.stop()

# Each club's real kit colour, not a generic palette.
PALETTE = dd.club_colours(picked)
colours = alt.Scale(domain=picked, range=PALETTE)

st.caption(
    "Along the bottom: every possible final position, 1st (champions) through "
    "20th (relegated). Height of each line: how likely the model thinks that "
    "finish is, in %. The dashed line marks 5% — what every club would show if "
    "the model had no opinion and treated all 20 spots as equally likely."
)

with st.container(border=True):
    long = (grid.loc[picked] * 100).reset_index().melt(
        id_vars=grid.index.name or "index", var_name="Position", value_name="Chance"
    )
    long.columns = ["Club", "Position", "Chance"]
    long["Position"] = long["Position"].astype(int)

    # Fixed to the highest line any club ever needs, over every club in the
    # league -- not just the ones currently picked. Letting Altair auto-scale
    # to only the picked clubs would rescale the y-axis every time the
    # selection changes, so the same line height could mean a different %
    # depending who else is on the chart. The axis itself has to stay
    # identical no matter which clubs are picked; only the lines should move.
    y_max = math.ceil(grid.values.max() * 100 / 5) * 5

    chart = alt.Chart(long).mark_line(point=True, strokeWidth=2.5).encode(
        alt.X("Position:O", title="Final league position (1st = champions, 20th = relegated)"),
        alt.Y("Chance:Q", title="Chance of that finish (%)", scale=alt.Scale(domain=[0, y_max])),
        alt.Color("Club:N", scale=colours, legend=alt.Legend(orient="top", title=None)),
        tooltip=["Club", "Position", alt.Tooltip("Chance:Q", format=".1f")],
    ).properties(height=360, title="Chance of finishing in each position")

    baseline = pd.DataFrame({"y": [5], "label": ["if every position were equally likely (5%)"]})
    rule = alt.Chart(baseline).mark_rule(strokeDash=[5, 4], color=dd.MUTED).encode(alt.Y("y:Q"))
    label = alt.Chart(baseline).mark_text(
        align="left", dx=4, dy=-6, color=dd.MUTED, fontSize=11,
    ).encode(alt.Y("y:Q"), alt.X(datum=0), text="label:N")

    st.altair_chart(chart + rule + label, width="stretch")

""
""

st.subheader(":material/tag: Key numbers")

cmp_cols = ["team", "exp_points", "P_title", "P_top4", "P_bottom3", "depth_ratio", "weakest_line"]
st.dataframe(
    merged[merged.team.isin(picked)][cmp_cols],
    hide_index=True, width="stretch",
    column_config={
        "team": "Club",
        "exp_points": st.column_config.NumberColumn("Points", format="%.1f"),
        "P_title": st.column_config.NumberColumn("Title", format="percent"),
        "P_top4": st.column_config.NumberColumn("Top 4", format="percent"),
        "P_bottom3": st.column_config.NumberColumn("Relegation", format="percent"),
        "depth_ratio": st.column_config.NumberColumn("Depth", format="%.2f"),
        "weakest_line": "Thinnest line",
    },
)

""

valuable.render(squads, default_club=picked[0] if picked else None)
