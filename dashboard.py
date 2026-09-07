"""
Premier League 2026/27 forecast dashboard.

    streamlit run dashboard.py

Layout follows streamlit-echarts-demo: one showcase page as the front door,
supporting pages behind it, sidebar filters bound to URL query params, and a
raw-data preview at the foot of every page.

Every number is read straight from output/ and data/ -- nothing here is
typed in by hand, so after `python -m src.run_pipeline` a click on "Reload
data" is the whole update.
"""

from datetime import datetime

import streamlit as st

import config
import dashboard_data as dd

st.set_page_config(
    page_title="EPL 2026/27 Forecast",
    page_icon=":material/sports_soccer:",
    layout="wide",
)

pg = st.navigation(
    [
        st.Page("pages/showcase.py", title="Showcase",
                icon=":material/dashboard:", default=True),
        st.Page("pages/standings.py", title="Table", icon=":material/leaderboard:"),
        st.Page("pages/squad_market.py", title="Squad & Market",
                icon=":material/swap_horiz:"),
        st.Page("pages/under_the_hood.py", title="Under the Hood", icon=":material/query_stats:"),
        st.Page("pages/compare.py", title="Compare Clubs",
                icon=":material/compare_arrows:"),
    ]
)
pg.run()

with st.sidebar:
    st.divider()
    if st.button(":material/refresh: Reload data", width="stretch"):
        st.cache_data.clear()
        st.rerun()

    st.caption(
        f"{datetime.fromtimestamp(dd.stamp()):%d %b %Y, %H:%M} · "
        f"{config.N_SIMULATIONS:,} simulations"
    )
    # Without this the forecast looks like it knows the current table. It does
    # not, on purpose: every one of the 380 matches is simulated from scratch,
    # so results already played are neither counted nor peeked at.
    st.caption(
        f"Forecast as of **{datetime.strptime(config.TODAY, '%Y-%m-%d'):%d %b %Y}** — "
        "a clean pre-season baseline. Matches already played are deliberately left "
        "out, so this is a prediction to judge in May, not a live table."
    )
