# nba_lockbot_app.py
# Streamlit single-file NBA prediction model (LockBot NBA v1)
# -----------------------------------------------------------
# What it does
# - Single-game mode: enter matchup features → model outputs ML pick, spread pick, O/U pick + confidence.
# - Batch mode: upload a CSV of games → model scores every game and marks a 🔒 Lock of the Day.
# - All weights are adjustable in the sidebar so you can tune and calibrate.

import math
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="LockBot NBA v1", layout="wide")
st.title("🏀 LockBot NBA v1 — Single-File Streamlit Model")
st.caption("Transparent, tunable NBA picks: ML / Spread / Total with confidence. Batch mode + Lock of the Day 🔒")

# -----------------------------
# Sidebar — Global Settings
# -----------------------------
st.sidebar.header("Global Settings & Weights")
league_avg_pace = st.sidebar.number_input("League Avg Pace (poss/48)", 80.0, 110.0, 99.5, 0.1)
scale_win = st.sidebar.number_input("Sigmoid Scale (Win)", 5.0, 60.0, 25.0, 1.0)
scale_total = st.sidebar.number_input("Sigmoid Scale (Total)", 2.0, 40.0, 15.0, 1.0)
lock_min_conf = st.sidebar.slider("Lock Threshold %", 50, 90, 66, 1)

st.sidebar.markdown("---")
st.sidebar.subheader("Weights — Moneyline/Spread (WinScore)")
w = {
    "elo":    st.sidebar.slider("ELO Differential", 0.0, 2.0, 1.0, 0.05),
    "form":   st.sidebar.slider("Recent Form (W% last 10)", 0.0, 2.0, 0.6, 0.05),
    "matchup":st.sidebar.slider("Matchup (ORtg vs opp DRtg)", 0.0, 2.0, 0.8, 0.05),
    "pace":   st.sidebar.slider("Pace Edge", 0.0, 1.5, 0.3, 0.05),
    "home":   st.sidebar.slider("Home Edge", 0.0, 2.0, 0.8, 0.05),
    "rest":   st.sidebar.slider("Rest Edge (days diff)", 0.0, 2.0, 0.6, 0.05),
    "b2b":    st.sidebar.slider("Back-to-Back Penalty", 0.0, 2.0, 0.7, 0.05),
    "inj_off":st.sidebar.slider("Injuries — Offensive Impact", 0.0, 2.0, 0.7, 0.05),
    "inj_def":st.sidebar.slider("Injuries — Defensive Impact", 0.0, 2.0, 0.6, 0.05),
    "travel": st.sidebar.slider("Travel Fatigue (miles diff scaled)", 0.0, 2.0, 0.4, 0.05),
    "h2h":    st.sidebar.slider("Head-to-Head (recent)", 0.0, 1.5, 0.3, 0.05),
    "market": st.sidebar.slider("Market Spread Sanity (sign check)", 0.0, 1.5, 0.5, 0.05),
}

st.sidebar.subheader("Weights — Totals (TotalScore)")
wt = {
    "pace":   st.sidebar.slider("Pace vs League Avg", 0.0, 2.0, 0.9, 0.05, key="wt_pace"),
    "eff":    st.sidebar.slider("Off vs Def Efficiency", 0.0, 2.0, 1.1, 0.05, key="wt_eff"),
    "recent": st.sidebar.slider("Recent O/U Trend", 0.0, 2.0, 0.6, 0.05, key="wt_recent"),
    "inj":    st.sidebar.slider("Injuries (Off & Def)", 0.0, 2.0, 0.6, 0.05, key="wt_inj"),
}

st.sidebar.markdown("---")
FOLLOW_ML_MARGIN = st.sidebar.slider("ATS follows ML unless opposite edge ≥ (%)", 0.0, 6.0, 3.0, 0.5) / 100.0
st.sidebar.caption("Tip: Start with defaults, then tune using recent slates. Keep a notebook of your best-performing presets.")

# -----------------------------
# Helpers
# -----------------------------
def sigmoid(x, scale):
    return 1 / (1 + math.exp(-x / max(scale, 1e-6)))

def win_score_features(row):
    elo_diff  = row.get("elo_home", 0) - row.get("elo_away", 0)
    form_diff = row.get("last10_home_winpct", 0) - row.get("last10_away_winpct", 0)
    matchup   = (row.get("off_home", 110) - row.get("def_away", 110)) - (row.get("off_away", 110) - row.get("def_home", 110))
    pace_edge = (row.get("pace_home", league_avg_pace) - row.get("pace_away", league_avg_pace)) / 10.0
    home_edge = 1.0
    rest_edge = (row.get("rest_home", 0) - row.get("rest_away", 0))
    b2b_pen   = (1 if row.get("b2b_home", 0) else 0) - (1 if row.get("b2b_away", 0) else 0)
    inj_off   = (row.get("injuries_off_away", 0) - row.get("injuries_off_home", 0))
    inj_def   = (row.get("injuries_def_away", 0) - row.get("injuries_def_home", 0))
    travel    = (row.get("travel_away", 0) - row.get("travel_home", 0)) / 500.0
    h2h       = row.get("h2h_home", 0) - 0.5
    market    = -row.get("spread_home", 0) / 10.0

    return {
        "elo": elo_diff, "form": form_diff, "matchup": matchup, "pace": pace_edge, "home": home_edge,
        "rest": rest_edge, "b2b": -b2b_pen, "inj_off": inj_off, "inj_def": inj_def, "travel": travel,
        "h2h": h2h, "market": market
    }

def compute_win(row):
    comps = win_score_features(row)
    winscore = sum(w[k] * comps[k] for k in comps)
    winprob_home = sigmoid(winscore, scale_win)
    ml_team = row.get("home") if winprob_home >= 0.5 else row.get("away")

    # ---------- FIXED ATS LOGIC ----------
    # Spread is entered from the HOME perspective (negative = home favored; positive = home dog).
    spread_home = row.get("spread_home", 0.0)

    # Rough mapping: each spread point ≈ 2.7% in win probability
    implied_shift = -spread_home * 0.027
    spread_edge = (winprob_home - 0.5) - implied_shift  # >0 = model likes HOME more than market

    def fmt_home(spread):
        return f"{row.get('home')} {spread:+g}"

    def fmt_away(spread):
        return f"{row.get('away')} {(-spread):+g}"  # away gets opposite sign

    ml_side = row.get("home") if winprob_home >= 0.5 else row.get("away")

    # Make ATS follow ML unless the opposite edge is big enough
    if abs(spread_edge) < FOLLOW_ML_MARGIN:
        spread_pick = fmt_home(spread_home) if ml_side == row.get("home") else fmt_away(spread_home)
    else:
        if spread_edge > 0.02:
            # Model likes HOME more than market → take HOME side of spread (home -X if fav, +X if dog)
            spread_pick = fmt_home(spread_home)
        elif spread_edge < -0.02:
            # Model likes AWAY more than market → take AWAY side of spread
            spread_pick = fmt_away(spread_home)
        else:
            # Very close: default to ML on the market favorite side
            fav = row.get('home') if spread_home < 0 else row.get('away')
            spread_pick = f"{fav} ML"
    # -------------------------------------

    conf_pct = round(abs(winprob_home - 0.5) * 200, 1)
    return {
        "WinScore": winscore,
        "HomeWinProb": winprob_home,
        "ML_Pick": f"{ml_team} ML",
        "Spread_Pick": spread_pick,
        "Win_Confidence_%": conf_pct,
    }

def compute_total(row):
    pace_avg   = (row.get("pace_home", league_avg_pace) + row.get("pace_away", league_avg_pace)) / 2
    pace_term  = (pace_avg - league_avg_pace) / 2.0
    eff_term   = ((row.get("off_home", 110) + row.get("off_away", 110)) - (row.get("def_home", 110) + row.get("def_away", 110))) / 10.0
    recent_ou  = ((row.get("recent_ou_home", 0) + row.get("recent_ou_away", 0)) / 2.0)
    inj_term   = ((row.get("injuries_off_home", 0) + row.get("injuries_off_away", 0)) - (row.get("injuries_def_home", 0) + row.get("injuries_def_away", 0))) / 2.0

    total_score  = wt["pace"]*pace_term + wt["eff"]*eff_term + wt["recent"]*recent_ou + wt["inj"]*inj_term
    market_total = row.get("total", 220.0)

    projected_delta = total_score * 2.5  # each unit ≈ 2.5 pts inclination vs market
    over_prob = sigmoid(projected_delta, scale_total)
    pick = "Over" if over_prob >= 0.5 else "Under"
    conf_pct = round(abs(over_prob - 0.5) * 200, 1)

    return {
        "TotalScore": total_score,
        "ProjectedDeltaPts": projected_delta,
        "OU_Pick": f"{pick} {market_total}",
        "OU_Confidence_%": conf_pct,
    }

def score_game(row):
    win = compute_win(row)
    tot = compute_total(row)
    lock_score = 0.6 * win["Win_Confidence_%"] + 0.4 * tot["OU_Confidence_%"]
    return {
        "home": row.get("home"),
        "away": row.get("away"),
        "ML_Pick": win["ML_Pick"],
        "Spread_Pick": win["Spread_Pick"],
        "Win_Confidence_%": win["Win_Confidence_%"],
        "OU_Pick": tot["OU_Pick"],
        "OU_Confidence_%": tot["OU_Confidence_%"],
        "LockScore": round(lock_score, 1),
    }

# -----------------------------
# UI — Single Game / Batch
# -----------------------------
t1, t2 = st.tabs(["Single Game", "Batch (CSV Upload)"])

with t1:
    st.subheader("Single Game Inputs")
    c1, c2, c3 = st.columns(3)
    with c1:
        home = st.text_input("Home Team", "BOS Celtics")
        away = st.text_input("Away Team", "NY Knicks")
        spread_home = st.number_input("Market Spread (Home, negative if favored)", -20.0, 20.0, -4.5, 0.5)
        total = st.number_input("Market Total (O/U)", 160.0, 280.0, 223.5, 0.5)
        h2h_home = st.slider("Head-to-Head (home win% last 8)", 0.0, 1.0, 0.5, 0.05)
        travel_home = st.number_input("Travel Miles (Home)", 0.0, 4000.0, 0.0, 50.0)
        travel_away = st.number_input("Travel Miles (Away)", 0.0, 4000.0, 800.0, 50.0)
    with c2:
        elo_home = st.number_input("Home ELO", 1200.0, 2000.0, 1650.0, 10.0)
        elo_away = st.number_input("Away ELO", 1200.0, 2000.0, 1600.0, 10.0)
        off_home = st.number_input("Home ORtg (per 100)", 95.0, 130.0, 116.0, 0.5)
        def_home = st.number_input("Home DRtg (per 100)", 95.0, 130.0, 111.0, 0.5)
        off_away = st.number_input("Away ORtg (per 100)", 95.0, 130.0, 113.0, 0.5)
        def_away = st.number_input("Away DRtg (per 100)", 95.0, 130.0, 112.5, 0.5)
        pace_home = st.number_input("Home Pace (poss/48)", 80.0, 110.0, 100.2, 0.1)
        pace_away = st.number_input("Away Pace (poss/48)", 80.0, 110.0, 98.9, 0.1)
    with c3:
        last10_home = st.slider("Home Win% (last 10)", 0.0, 1.0, 0.7, 0.05)
        last10_away = st.slider("Away Win% (last 10)", 0.0, 1.0, 0.55, 0.05)
        rest_home = st.number_input("Home Rest Days", 0.0, 5.0, 2.0, 0.5)
        rest_away = st.number_input("Away Rest Days", 0.0, 5.0, 1.0, 0.5)
        b2b_home = st.checkbox("Home on B2B", False)
        b2b_away = st.checkbox("Away on B2B", True)
        injuries_off_home = st.slider("Home Offense Missing (pts/100)", 0.0, 20.0, 2.0, 0.5)
        injuries_off_away = st.slider("Away Offense Missing (pts/100)", 0.0, 20.0, 4.0, 0.5)
        injuries_def_home = st.slider("Home Defense Missing (pts/100)", 0.0, 20.0, 1.0, 0.5)
        injuries_def_away = st.slider("Away Defense Missing (pts/100)", 0.0, 20.0, 2.0, 0.5)
        recent_ou_home = st.slider("Home Recent O/U Trend (avg delta, +Over)", -10.0, 10.0, 1.0, 0.5)
        recent_ou_away = st.slider("Away Recent O/U Trend (avg delta, +Over)", -10.0, 10.0, -0.5, 0.5)

    if st.button("Score Game", type="primary"):
        row = {
            "home": home, "away": away,
            "spread_home": spread_home, "total": total,
            "h2h_home": h2h_home,
            "travel_home": travel_home, "travel_away": travel_away,
            "elo_home": elo_home, "elo_away": elo_away,
            "off_home": off_home, "def_home": def_home,
            "off_away": off_away, "def_away": def_away,
            "pace_home": pace_home, "pace_away": pace_away,
            "last10_home_winpct": last10_home, "last10_away_winpct": last10_away,
            "rest_home": rest_home, "rest_away": rest_away,
            "b2b_home": int(b2b_home), "b2b_away": int(b2b_away),
            "injuries_off_home": injuries_off_home, "injuries_off_away": injuries_off_away,
            "injuries_def_home": injuries_def_home, "injuries_def_away": injuries_def_away,
            "recent_ou_home": recent_ou_home, "recent_ou_away": recent_ou_away,
        }
        result = score_game(row)
        with st.container(border=True):
            st.markdown(f"**Matchup:** {home} vs {away}")
            cA, cB, cC = st.columns(3)
            cA.metric("ML Pick", result["ML_Pick"], f"Confidence {result['Win_Confidence_%']}%")
            cB.metric("Spread Pick", result["Spread_Pick"])
            cC.metric("Total Pick", result["OU_Pick"], f"Confidence {result['OU_Confidence_%']}%")
            st.markdown(f"**Lock Score:** {result['LockScore']} (Lock threshold {lock_min_conf}%)")
            if result["LockScore"] >= lock_min_conf:
                st.success("🔒 Consider this a Lock of the Day candidate.")
            else:
                st.info("Not strong enough for 🔒 by your threshold.")

with t2:
    st.subheader("Batch Upload — Score a Full Slate")
    st.caption("Upload a CSV with columns (lowercase): home, away, spread_home (home negative if favored), total, elo_home, elo_away, off_home, def_home, off_away, def_away, pace_home, pace_away, last10_home_winpct, last10_away_winpct, rest_home, rest_away, b2b_home, b2b_away, injuries_off_home, injuries_off_away, injuries_def_home, injuries_def_away, recent_ou_home, recent_ou_away, h2h_home, travel_home, travel_away")

    def example_df():
        return pd.DataFrame([
            {
                "home": "BOS Celtics", "away": "NY Knicks", "spread_home": -4.5, "total": 223.5,
                "elo_home": 1660, "elo_away": 1610,
                "off_home": 117.0, "def_home": 110.0, "off_away": 113.0, "def_away": 112.0,
                "pace_home": 100.2, "pace_away": 98.9,
                "last10_home_winpct": 0.7, "last10_away_winpct": 0.55,
                "rest_home": 2, "rest_away": 1,
                "b2b_home": 0, "b2b_away": 1,
                "injuries_off_home": 2.0, "injuries_off_away": 4.0,
                "injuries_def_home": 1.0, "injuries_def_away": 2.0,
                "recent_ou_home": 1.0, "recent_ou_away": -0.5,
                "h2h_home": 0.5, "travel_home": 0, "travel_away": 800,
            },
            {
                "home": "DEN Nuggets", "away": "DAL Mavericks", "spread_home": -6.5, "total": 228.5,
                "elo_home": 1710, "elo_away": 1665,
                "off_home": 118.5, "def_home": 112.0, "off_away": 117.5, "def_away": 114.5,
                "pace_home": 98.5, "pace_away": 99.3,
                "last10_home_winpct": 0.8, "last10_away_winpct": 0.6,
                "rest_home": 2, "rest_away": 2,
                "b2b_home": 0, "b2b_away": 0,
                "injuries_off_home": 1.0, "injuries_off_away": 3.0,
                "injuries_def_home": 0.0, "injuries_def_away": 1.0,
                "recent_ou_home": -1.0, "recent_ou_away": 0.5,
                "h2h_home": 0.6, "travel_home": 200, "travel_away": 900,
            },
        ])

    ex = example_df()
    ex_csv = ex.to_csv(index=False).encode()
    st.download_button("Download CSV Template (with examples)", data=ex_csv, file_name="nba_lockbot_template.csv", mime="text/csv")

    uploaded = st.file_uploader("Upload your slate CSV", type=["csv"])
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            needed = [
                "home","away","spread_home","total","elo_home","elo_away","off_home","def_home","off_away","def_away","pace_home","pace_away","last10_home_winpct","last10_away_winpct","rest_home","rest_away","b2b_home","b2b_away","injuries_off_home","injuries_off_away","injuries_def_home","injuries_def_away","recent_ou_home","recent_ou_away","h2h_home","travel_home","travel_away"
            ]
            missing = [c for c in needed if c not in df.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                results = []
                for _, row in df.iterrows():
                    results.append(score_game(row))
                out = pd.DataFrame(results)
                out = out.sort_values("LockScore", ascending=False).reset_index(drop=True)
                if len(out):
                    if out.loc[0, "LockScore"] >= lock_min_conf:
                        out.loc[0, "Lock"] = "🔒"
                    else:
                        out.loc[0, "Lock"] = "(no 🔒)"
                st.dataframe(out)

                out_csv = out.to_csv(index=False).encode()
                st.download_button("Download Picks CSV", data=out_csv, file_name="nba_lockbot_picks.csv", mime="text/csv")
        except Exception as e:
            st.exception(e)
