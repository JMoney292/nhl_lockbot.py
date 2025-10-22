# nhl_lockbot.py
# -----------------------------------------------------------
# LockBot NHL v1 — Single-File Streamlit Model
# - Single-game mode: ML, Puckline (ATS), and Total (O/U) picks + confidence
# - Batch mode: CSV upload to score an entire slate & flag a 🔒 Lock of the Day
# - Tunable weights (ratings, form, matchup, rest/B2B, injuries, goalie, travel)
#
# Notes:
# - This is an interpretable handicapping helper. Calibrate weights to your data.
# - Inputs are your expected values for tonight (not blindly season averages).

import math
import pandas as pd
import streamlit as st

st.set_page_config(page_title="LockBot NHL", layout="wide")
st.title("🏒 LockBot NHL v1 — Single-File Streamlit Model")
st.caption("Transparent, tunable NHL picks: ML / Puckline / Total with confidence. Batch mode + Lock of the Day 🔒")

# -----------------------------
# Sidebar — Global Settings
# -----------------------------
st.sidebar.header("Global Settings & Weights")

# Win / ATS scaling
scale_win   = st.sidebar.number_input("Sigmoid Scale (Win)", 5.0, 60.0, 22.0, 1.0)
scale_total = st.sidebar.number_input("Sigmoid Scale (Total)", 2.0, 40.0, 12.0, 1.0)

# How much one goal on the puckline maps to win-prob shift (heuristic; per goal)
goal_to_winprob = st.sidebar.slider("Win Prob shift per 1.0 goal (puckline)", 0.06, 0.22, 0.14, 0.01)

# Force ATS to follow ML unless opposite edge exceeds X%
FOLLOW_ML_MARGIN = st.sidebar.slider("ATS follows ML unless opposite edge ≥ (%)", 0.0, 8.0, 3.0, 0.5)/100.0

lock_min_conf = st.sidebar.slider("Lock Threshold %", 50, 90, 66, 1)

st.sidebar.markdown("---")
st.sidebar.subheader("Weights — Moneyline/ATS (WinScore)")
w = {
    "rating": st.sidebar.slider("Rating Differential (ELO/Power)", 0.0, 2.5, 1.0, 0.05),
    "form":   st.sidebar.slider("Recent Form (W% last 10)",        0.0, 2.0, 0.6, 0.05),
    "match":  st.sidebar.slider("5v5 Matchup (xGF vs opp xGA)",     0.0, 2.0, 0.9, 0.05),
    "spec":   st.sidebar.slider("Special Teams (PP vs PK)",         0.0, 1.5, 0.5, 0.05),
    "home":   st.sidebar.slider("Home Ice Edge",                    0.0, 1.5, 0.5, 0.05),
    "rest":   st.sidebar.slider("Rest Edge (days diff)",            0.0, 2.0, 0.6, 0.05),
    "b2b":    st.sidebar.slider("Back-to-Back Penalty",             0.0, 2.0, 0.8, 0.05),
    "inj_off":st.sidebar.slider("Injuries — Off Impact",            0.0, 2.0, 0.6, 0.05),
    "inj_def":st.sidebar.slider("Injuries — Def Impact",            0.0, 2.0, 0.6, 0.05),
    "goalie": st.sidebar.slider("Goalie (GSAx/quality)",            0.0, 2.0, 0.9, 0.05),
    "travel": st.sidebar.slider("Travel Fatigue",                   0.0, 1.5, 0.4, 0.05),
    "market": st.sidebar.slider("Market Sanity (puckline sign)",    0.0, 1.5, 0.4, 0.05),
}

st.sidebar.subheader("Weights — Totals (TotalScore)")
wt = {
    "xgpace": st.sidebar.slider("xG Pace (xGF60+xGA60 vs avg)",     0.0, 2.0, 1.0, 0.05),
    "finish": st.sidebar.slider("Finishing / Shooting%",            0.0, 1.5, 0.5, 0.05),
    "save":   st.sidebar.slider("Goalie Save Impact",               0.0, 1.5, 0.7, 0.05),
    "recent": st.sidebar.slider("Recent O/U Trend",                 0.0, 1.5, 0.5, 0.05),
    "inj":    st.sidebar.slider("Injuries (Off & Def)",             0.0, 1.5, 0.5, 0.05),
}

st.sidebar.caption("Tip: Tune on a week of games; raise Lock threshold for fewer, stronger 🔒.")

# -----------------------------
# Helpers
# -----------------------------
def sigmoid(x, scale):
    return 1 / (1 + math.exp(-x / max(scale, 1e-6)))

def win_score_components(row):
    """Build interpretable components for WinScore from a dict/Series."""
    # Basic team-strength differential
    rating_diff = row.get("rating_home", 0) - row.get("rating_away", 0)

    # Recent form: W% last 10
    form_diff = row.get("l10_home_winpct", 0) - row.get("l10_away_winpct", 0)

    # 5v5 expected goals matchup signal
    # positive means home offensive xG vs away defensive xG is favorable
    match_term = (
        (row.get("xgf60_home", 2.5) - row.get("xga60_away", 2.5)) -
        (row.get("xgf60_away", 2.5) - row.get("xga60_home", 2.5))
    )

    # Special teams rough edge: PP% vs PK%
    spec_term = (row.get("pp_home", 20.0) - row.get("pk_away", 80.0)/4.0) - \
                (row.get("pp_away", 20.0) - row.get("pk_home", 80.0)/4.0)

    home_edge = 1.0  # constant (scaled by w["home"])
    rest_edge = row.get("rest_home", 0) - row.get("rest_away", 0)
    b2b_pen  = (1 if row.get("b2b_home", 0) else 0) - (1 if row.get("b2b_away", 0) else 0)

    # Injuries: positive numbers mean MISSING impact (pts/100 or xG/60 equivalent scalar)
    inj_off  = row.get("inj_off_away", 0) - row.get("inj_off_home", 0)   # away missing offense helps home
    inj_def  = row.get("inj_def_away", 0) - row.get("inj_def_home", 0)   # away missing defense helps home

    # Goalie: positive means away goalie worse / home goalie better (GSAx-like)
    goalie   = row.get("goalie_edge_home", 0)  # e.g., home GSAx − away GSAx, or your custom scale

    travel   = (row.get("travel_away", 0) - row.get("travel_home", 0)) / 1000.0  # scale: per 1000 miles

    # Market sanity: if home is puckline favorite (-1.5), boost a bit; if dog (+1.5), trim a bit
    market   = -row.get("puckline_home", 0) * 0.2

    comps = {
        "rating": rating_diff,
        "form":   form_diff,
        "match":  match_term,
        "spec":   spec_term,
        "home":   home_edge,
        "rest":   rest_edge,
        "b2b":   -b2b_pen,   # penalty when home is on B2B vs away not
        "inj_off":inj_off,
        "inj_def":inj_def,
        "goalie": goalie,
        "travel": travel,
        "market": market,
    }
    return comps

def compute_win(row):
    comps = win_score_components(row)
    winscore = sum(w[k] * comps[k] for k in comps)
    winprob_home = sigmoid(winscore, scale_win)
    ml_team = row.get("home") if winprob_home >= 0.5 else row.get("away")

    # ---------- ATS / Puckline decision ----------
    # Puckline entered from HOME perspective in goals (typical: -1.5 if favored, +1.5 if dog)
    puckline_home = float(row.get("puckline_home", -1.5))

    # map puckline (goals) into implied win-prob shift
    implied_shift = -puckline_home * goal_to_winprob     # negative line (home -1.5) increases implied home win
    spread_edge = (winprob_home - 0.5) - implied_shift   # >0 => model likes HOME more than market

    def fmt_home(pl):
        return f"{row.get('home')} {pl:+.1f}"

    def fmt_away(pl):
        return f"{row.get('away')} {(-pl):+.1f}"  # away gets opposite sign

    ml_side = row.get("home") if winprob_home >= 0.5 else row.get("away")

    if abs(spread_edge) < FOLLOW_ML_MARGIN:
        spread_pick = fmt_home(puckline_home) if ml_side == row.get("home") else fmt_away(puckline_home)
    else:
        if spread_edge > 0.02:
            spread_pick = fmt_home(puckline_home)   # take the home side of the puckline
        elif spread_edge < -0.02:
            spread_pick = fmt_away(puckline_home)   # take the away side
        else:
            fav = row.get('home') if puckline_home < 0 else row.get('away')
            spread_pick = f"{fav} ML"
    # ---------------------------------------------

    conf_pct = round(abs(winprob_home - 0.5) * 200, 1)  # 50→0, 100→100
    return {
        "WinScore": winscore,
        "HomeWinProb": winprob_home,
        "ML_Pick": f"{ml_team} ML",
        "Spread_Pick": spread_pick,
        "Win_Confidence_%": conf_pct,
    }

def compute_total(row):
    # xG pace proxy: sum of offensive and defensive xG rates
    xgpace_home = row.get("xgf60_home", 2.5) + row.get("xga60_home", 2.5)
    xgpace_away = row.get("xgf60_away", 2.5) + row.get("xga60_away", 2.5)
    pace_term   = (xgpace_home + xgpace_away) / 2.0 - 5.0  # 5.0 is a rough league baseline (xG/60 total both teams)

    # Finishing and goaltending adjustments
    finish_term = (row.get("fin_home", 0) + row.get("fin_away", 0)) / 2.0   # shooting/run-hot/cold scalar
    save_term   = -(row.get("goalie_edge_home", 0))                          # better home goalie reduces totals a bit

    inj_term = (row.get("inj_off_home", 0) + row.get("inj_off_away", 0) -
                row.get("inj_def_home", 0) - row.get("inj_def_away", 0)) / 2.0

    recent_ou = (row.get("recent_ou_home", 0) + row.get("recent_ou_away", 0)) / 2.0  # + means trending Over

    total_score = wt["xgpace"]*pace_term + wt["finish"]*finish_term + wt["save"]*save_term + wt["inj"]*inj_term + wt["recent"]*recent_ou
    market_total = float(row.get("total", 6.0))

    # Convert score to delta vs market (heuristic: each unit ~ 0.6 goals inclination)
    projected_delta = total_score * 0.6
    over_prob = sigmoid(projected_delta, scale_total)

    pick = "Over" if over_prob >= 0.5 else "Under"
    conf_pct = round(abs(over_prob - 0.5) * 200, 1)

    return {
        "TotalScore": total_score,
        "ProjectedDeltaGoals": projected_delta,
        "OU_Pick": f"{pick} {market_total:.1f}",
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
tabs = st.tabs(["Single Game", "Batch (CSV Upload)"])

with tabs[0]:
    st.subheader("Single Game Inputs")

    c1, c2, c3 = st.columns(3)
    with c1:
        home = st.text_input("Home Team", "BOS Bruins")
        away = st.text_input("Away Team", "NY Rangers")
        puckline_home = st.number_input("Puckline (Home; negative if favored)", -3.0, 3.0, -1.5, 0.5)
        total = st.number_input("Market Total (O/U, goals)", 4.5, 8.5, 6.0, 0.5)
        travel_home = st.number_input("Travel Miles (Home)", 0.0, 5000.0, 0.0, 50.0)
        travel_away = st.number_input("Travel Miles (Away)", 0.0, 5000.0, 600.0, 50.0)

    with c2:
        rating_home = st.number_input("Home Rating / ELO", 1200.0, 2000.0, 1610.0, 5.0)
        rating_away = st.number_input("Away Rating / ELO", 1200.0, 2000.0, 1590.0, 5.0)
        xgf60_home  = st.number_input("Home xGF/60 (5v5)", 1.5, 4.0, 2.9, 0.05)
        xga60_home  = st.number_input("Home xGA/60 (5v5)", 1.5, 4.0, 2.5, 0.05)
        xgf60_away  = st.number_input("Away xGF/60 (5v5)", 1.5, 4.0, 2.6, 0.05)
        xga60_away  = st.number_input("Away xGA/60 (5v5)", 1.5, 4.0, 2.7, 0.05)

    with c3:
        l10_home = st.slider("Home Win% (last 10)", 0.0, 1.0, 0.6, 0.05)
        l10_away = st.slider("Away Win% (last 10)", 0.0, 1.0, 0.55, 0.05)
        rest_home = st.number_input("Home Rest Days", 0.0, 5.0, 2.0, 0.5)
        rest_away = st.number_input("Away Rest Days", 0.0, 5.0, 1.0, 0.5)
        b2b_home  = st.checkbox("Home on B2B", False)
        b2b_away  = st.checkbox("Away on B2B", False)
        inj_off_home = st.slider("Home Offense Missing (rel. scale)", 0.0, 5.0, 0.5, 0.1)
        inj_off_away = st.slider("Away Offense Missing (rel. scale)", 0.0, 5.0, 1.0, 0.1)
        inj_def_home = st.slider("Home Defense Missing (rel. scale)", 0.0, 5.0, 0.3, 0.1)
        inj_def_away = st.slider("Away Defense Missing (rel. scale)", 0.0, 5.0, 0.6, 0.1)
        goalie_edge_home = st.slider("Goalie Edge (Home − Away, + favors home)", -3.0, 3.0, 0.5, 0.1)
        fin_home = st.slider("Home Finishing (± hot/cold)", -2.0, 2.0, 0.2, 0.1)
        fin_away = st.slider("Away Finishing (± hot/cold)", -2.0, 2.0, -0.1, 0.1)
        recent_ou_home = st.slider("Home Recent O/U Trend (avg delta, +Over)", -2.0, 2.0, 0.3, 0.1)
        recent_ou_away = st.slider("Away Recent O/U Trend (avg delta, +Over)", -2.0, 2.0, -0.2, 0.1)

    if st.button("Score Game", type="primary"):
        row = {
            "home": home, "away": away,
            "puckline_home": puckline_home, "total": total,
            "travel_home": travel_home, "travel_away": travel_away,

            "rating_home": rating_home, "rating_away": rating_away,
            "xgf60_home": xgf60_home, "xga60_home": xga60_home,
            "xgf60_away": xgf60_away, "xga60_away": xga60_away,

            "l10_home_winpct": l10_home, "l10_away_winpct": l10_away,
            "rest_home": rest_home, "rest_away": rest_away,
            "b2b_home": int(b2b_home), "b2b_away": int(b2b_away),

            "inj_off_home": inj_off_home, "inj_off_away": inj_off_away,
            "inj_def_home": inj_def_home, "inj_def_away": inj_def_away,
            "goalie_edge_home": goalie_edge_home,
            "fin_home": fin_home, "fin_away": fin_away,
            "recent_ou_home": recent_ou_home, "recent_ou_away": recent_ou_away,
        }
        result = score_game(row)
        with st.container(border=True):
            st.markdown(f"**Matchup:** {home} vs {away}")
            cA, cB, cC = st.columns(3)
            cA.metric("ML Pick", result["ML_Pick"], f"Confidence {result['Win_Confidence_%']}%")
            cB.metric("Puckline Pick", result["Spread_Pick"])
            cC.metric("Total Pick", result["OU_Pick"], f"Confidence {result['OU_Confidence_%']}%")
            st.markdown(f"**Lock Score:** {result['LockScore']} (Lock threshold {lock_min_conf}%)")
            if result["LockScore"] >= lock_min_conf:
                st.success("🔒 Consider this a Lock of the Day candidate.")
            else:
                st.info("Not strong enough for 🔒 by your threshold.")

with tabs[1]:
    st.subheader("Batch Upload — Score a Full Slate")
    st.caption(
        "Upload a CSV with columns: "
        "home,away,puckline_home,total,rating_home,rating_away,xgf60_home,xga60_home,xgf60_away,xga60_away,"
        "l10_home_winpct,l10_away_winpct,rest_home,rest_away,b2b_home,b2b_away,inj_off_home,inj_off_away,"
        "inj_def_home,inj_def_away,goalie_edge_home,fin_home,fin_away,recent_ou_home,recent_ou_away,travel_home,travel_away"
    )

    def example_df():
        return pd.DataFrame([
            {
                "home":"BOS Bruins","away":"NY Rangers","puckline_home":-1.5,"total":6.0,
                "rating_home":1620,"rating_away":1605,
                "xgf60_home":2.90,"xga60_home":2.40,"xgf60_away":2.70,"xga60_away":2.60,
                "l10_home_winpct":0.7,"l10_away_winpct":0.6,
                "rest_home":2,"rest_away":1,"b2b_home":0,"b2b_away":0,
                "inj_off_home":0.4,"inj_off_away":1.0,"inj_def_home":0.2,"inj_def_away":0.5,
                "goalie_edge_home":0.6,"fin_home":0.2,"fin_away":-0.1,
                "recent_ou_home":0.3,"recent_ou_away":-0.2,"travel_home":0,"travel_away":450
            },
            {
                "home":"EDM Oilers","away":"SEA Kraken","puckline_home":-1.5,"total":6.5,
                "rating_home":1640,"rating_away":1580,
                "xgf60_home":3.20,"xga60_home":2.80,"xgf60_away":2.50,"xga60_away":2.90,
                "l10_home_winpct":0.65,"l10_away_winpct":0.5,
                "rest_home":1,"rest_away":1,"b2b_home":0,"b2b_away":1,
                "inj_off_home":0.3,"inj_off_away":1.2,"inj_def_home":0.1,"inj_def_away":0.6,
                "goalie_edge_home":0.4,"fin_home":0.3,"fin_away":-0.2,
                "recent_ou_home":0.1,"recent_ou_away":-0.1,"travel_home":200,"travel_away":800
            }
        ])

    ex = example_df()
    ex_csv = ex.to_csv(index=False).encode()
    st.download_button("Download CSV Template (with examples)", data=ex_csv, file_name="nhl_lockbot_template.csv", mime="text/csv")

    uploaded = st.file_uploader("Upload your slate CSV", type=["csv"])
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            needed = [
                "home","away","puckline_home","total","rating_home","rating_away","xgf60_home","xga60_home",
                "xgf60_away","xga60_away","l10_home_winpct","l10_away_winpct","rest_home","rest_away","b2b_home",
                "b2b_away","inj_off_home","inj_off_away","inj_def_home","inj_def_away","goalie_edge_home",
                "fin_home","fin_away","recent_ou_home","recent_ou_away","travel_home","travel_away"
            ]
            missing = [c for c in needed if c not in df.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                results = []
                for _, row in df.iterrows():
                    results.append(score_game(row))
                out = pd.DataFrame(results).sort_values("LockScore", ascending=False).reset_index(drop=True)
                if len(out):
                    out.loc[0, "Lock"] = "🔒" if out.loc[0, "LockScore"] >= lock_min_conf else "(no 🔒)"
                st.dataframe(out)
                st.download_button("Download Picks CSV", data=out.to_csv(index=False).encode(),
                                   file_name="nhl_lockbot_picks.csv", mime="text/csv")
        except Exception as e:
            st.exception(e)
