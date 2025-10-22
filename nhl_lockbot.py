# nhl_lockbot.py
# LockBot NHL — Simple + Advanced + Batch
# Adds: Home/Away Win %, Predicted Score, Puckline & OU, L5 Totals, Blend/Rule

import math
import pandas as pd
import streamlit as st

# ---------- App config ----------
st.set_page_config(page_title="LockBot NHL (NFL-style)", layout="wide")
st.title("🏒 LockBot NHL — NFL-style Simple Mode")
st.caption("Home/Away win %, projected score, puckline & total picks. Tunable totals (goals vs heuristic), L5 totals, and batch mode.")

# ---------- Sidebar (globals) ----------
st.sidebar.header("Global Settings")
scale_win   = st.sidebar.number_input("Sigmoid Scale (Win)", 5.0, 60.0, 22.0, 1.0, key="g_scale_win")
scale_total = st.sidebar.number_input("Sigmoid Scale (Total)", 2.0, 40.0, 10.0, 1.0, key="g_scale_tot")
goal_to_winprob = st.sidebar.slider("Win Prob shift per 1.0 goal (puckline)", 0.06, 0.22, 0.14, 0.01, key="g_goalmap")
FOLLOW_ML_MARGIN = st.sidebar.slider("ATS follows ML unless opposite edge ≥ (%)", 0.0, 8.0, 3.0, 0.5, key="g_follow") / 100.0
lock_min_conf = st.sidebar.slider("Lock Threshold %", 50, 90, 66, 1, key="g_lockthresh")

st.sidebar.markdown("---")
st.sidebar.subheader("Weights — Win/ATS")
w = {
    "rating": st.sidebar.slider("Rating Differential",        0.0, 2.5, 0.8, 0.05, key="w_rating"),
    "form":   st.sidebar.slider("Recent Form (W% L10)",       0.0, 2.0, 0.9, 0.05, key="w_form"),
    "match":  st.sidebar.slider("5v5 Matchup (xG)",           0.0, 2.0, 0.9, 0.05, key="w_match"),
    "spec":   st.sidebar.slider("Special Teams (PP vs PK)",   0.0, 1.5, 0.5, 0.05, key="w_spec"),
    "home":   st.sidebar.slider("Home Ice Edge",              0.0, 1.5, 0.6, 0.05, key="w_home"),
    "rest":   st.sidebar.slider("Rest Edge (days diff)",      0.0, 2.0, 0.6, 0.05, key="w_rest"),
    "b2b":    st.sidebar.slider("Back-to-Back Penalty",       0.0, 2.0, 0.8, 0.05, key="w_b2b"),
    "inj_off":st.sidebar.slider("Injuries — Off",             0.0, 2.0, 0.6, 0.05, key="w_injoff"),
    "inj_def":st.sidebar.slider("Injuries — Def",             0.0, 2.0, 0.6, 0.05, key="w_injdef"),
    "goalie": st.sidebar.slider("Goalie Edge",                0.0, 2.0, 0.9, 0.05, key="w_goalie"),
    "travel": st.sidebar.slider("Travel Fatigue",             0.0, 1.5, 0.4, 0.05, key="w_travel"),
    "market": st.sidebar.slider("Market Sanity (sign check)", 0.0, 1.5, 0.4, 0.05, key="w_market"),
}

st.sidebar.subheader("Weights — Totals (heuristic piece)")
wt = {
    "xgpace": st.sidebar.slider("xG Pace vs avg",       0.0, 2.0, 1.3, 0.05, key="wt_pace"),
    "finish": st.sidebar.slider("Finishing (hot/cold)", 0.0, 1.5, 0.8, 0.05, key="wt_finish"),
    "save":   st.sidebar.slider("Goalie Save Impact",   0.0, 1.5, 0.5, 0.05, key="wt_save"),
    "recent": st.sidebar.slider("Recent O/U Trend",     0.0, 1.5, 0.8, 0.05, key="wt_recent"),
    "inj":    st.sidebar.slider("Injuries (O + D)",     0.0, 1.5, 0.6, 0.05, key="wt_inj"),
}

st.sidebar.markdown("---")
st.sidebar.subheader("Totals Blending & Calibration")
totals_alpha = st.sidebar.slider("Blend α (Goals vs Heuristic)", 0.0, 1.0, 0.75, 0.05, key="g_alpha")
goals_scale  = st.sidebar.slider("Goals scale (γ)", 0.9, 1.2, 1.05, 0.01, key="g_goals_scale")
goals_bias   = st.sidebar.slider("Goals bias (β, adds to total)", 0.0, 0.5, 0.15, 0.01, key="g_goals_bias")
pick_rule = st.sidebar.selectbox(
    "Totals pick rule",
    ["Blend (use α)", "Goals only", "Heuristic only", "Consensus (both must agree)"],
    index=0, key="g_totals_rule"
)
l5_weight = st.sidebar.slider("Weight of L5 game totals in heuristic", 0.0, 1.0, 0.50, 0.05, key="g_l5_weight")

# ---------- Helpers ----------
def sigmoid(x, scale):
    return 1 / (1 + math.exp(-x / max(scale, 1e-6)))

def fill_defaults_minimal(row):
    """Neutral defaults so Simple mode needs only a few inputs."""
    defaults = {
        "xgf60_home": 2.7, "xga60_home": 2.7, "xgf60_away": 2.7, "xga60_away": 2.7,
        "pp_home": 22.0, "pk_home": 80.0, "pp_away": 22.0, "pk_away": 80.0,
        "l10_home_winpct": 0.55, "l10_away_winpct": 0.50,
        "rest_home": 1.0, "rest_away": 1.0,
        "b2b_home": 0, "b2b_away": 0,
        "inj_off_home": 0.0, "inj_off_away": 0.0, "inj_def_home": 0.0, "inj_def_away": 0.0,
        "goalie_edge_home": 0.0,
        "fin_home": 0.0, "fin_away": 0.0,
        "recent_ou_home": 0.0, "recent_ou_away": 0.0,
        "travel_home": 0.0, "travel_away": 0.0,
        "l5_total_home": None, "l5_total_away": None,
    }
    for k, v in defaults.items():
        row.setdefault(k, v)
    return row

def win_score_components(row):
    rating_diff = row.get("rating_home", 0) - row.get("rating_away", 0)
    form_diff   = row.get("l10_home_winpct", 0) - row.get("l10_away_winpct", 0)
    match_term  = ((row.get("xgf60_home",2.7) - row.get("xga60_away",2.7)) -
                   (row.get("xgf60_away",2.7) - row.get("xga60_home",2.7)))
    spec_term   = (row.get("pp_home",22.0) - row.get("pk_away",80.0)/4.0) - \
                  (row.get("pp_away",22.0) - row.get("pk_home",80.0)/4.0)
    home_edge   = 1.0
    rest_edge   = row.get("rest_home", 0) - row.get("rest_away", 0)
    b2b_pen     = (1 if row.get("b2b_home",0) else 0) - (1 if row.get("b2b_away",0) else 0)
    inj_off     = row.get("inj_off_away",0) - row.get("inj_off_home",0)
    inj_def     = row.get("inj_def_away",0) - row.get("inj_def_home",0)
    goalie      = row.get("goalie_edge_home",0)
    travel      = (row.get("travel_away",0) - row.get("travel_home",0)) / 1000.0
    market      = -row.get("puckline_home",0) * 0.2
    return {
        "rating": rating_diff, "form": form_diff, "match": match_term, "spec": spec_term,
        "home": home_edge, "rest": rest_edge, "b2b": -b2b_pen,
        "inj_off": inj_off, "inj_def": inj_def, "goalie": goalie, "travel": travel, "market": market
    }

# ---------- Win model ----------
def compute_win(row):
    comps = win_score_components(row)
    winscore = sum(w[k] * comps[k] for k in comps)

    # model probability that HOME wins
    p_home = sigmoid(winscore, scale_win)
    p_away = 1.0 - p_home

    # ML side text
    ml_team = row.get("home") if p_home >= 0.5 else row.get("away")

    # puckline is from HOME perspective (negative if home favored)
    puckline_home = float(row.get("puckline_home", -1.5))
    implied_shift = -puckline_home * goal_to_winprob
    spread_edge = (p_home - 0.5) - implied_shift  # >0 → model likes HOME vs market

    def fmt_home(pl): return f"{row.get('home')} {pl:+.1f}"
    def fmt_away(pl): return f"{row.get('away')} {(-pl):+.1f}"

    ml_side = row.get("home") if p_home >= 0.5 else row.get("away")
    if abs(spread_edge) < FOLLOW_ML_MARGIN:
        spread_pick = fmt_home(puckline_home) if ml_side == row.get("home") else fmt_away(puckline_home)
    else:
        if spread_edge > 0.02:
            spread_pick = fmt_home(puckline_home)
        elif spread_edge < -0.02:
            spread_pick = fmt_away(puckline_home)
        else:
            fav = row.get('home') if puckline_home < 0 else row.get('away')
            spread_pick = f"{fav} ML"

    # format percents
    home_pct = round(p_home * 100, 1)
    away_pct = round(p_away * 100, 1)
    conf_pct = round(abs(p_home - 0.5) * 200, 1)

    return {
        "WinScore": winscore,
        "HomeWinProb": p_home,
        "AwayWinProb": p_away,
        "Home_Win_%": home_pct,
        "Away_Win_%": away_pct,
        "ML_Pick": f"{ml_team} ML",
        "Spread_Pick": spread_pick,
        "Win_Confidence_%": conf_pct,
    }

# ---------- Totals: heuristic (with L5) ----------
def compute_total_heuristic(row):
    """Heuristic projection around the market total + L5 game totals blend."""
    xgpace_home = row.get("xgf60_home",2.7) + row.get("xga60_home",2.7)
    xgpace_away = row.get("xgf60_away",2.7) + row.get("xga60_away",2.7)
    pace_term   = (xgpace_home + xgpace_away)/2.0 - 5.0

    finish_term = (row.get("fin_home",0) + row.get("fin_away",0))/2.0
    save_term   = -(row.get("goalie_edge_home",0))
    inj_term    = (row.get("inj_off_home",0)+row.get("inj_off_away",0) -
                   row.get("inj_def_home",0)-row.get("inj_def_away",0)) / 2.0
    recent_ou   = (row.get("recent_ou_home",0)+row.get("recent_ou_away",0))/2.0

    total_score = wt["xgpace"]*pace_term + wt["finish"]*finish_term + wt["save"]*save_term + wt["inj"]*inj_term + wt["recent"]*recent_ou
    market_total = float(row.get("total", 6.0))
    base_heuristic_total = market_total + total_score * 0.6

    # NFL-style L5 average game totals
    l5_home = row.get("l5_total_home", None)
    l5_away = row.get("l5_total_away", None)
    l5_vals = [v for v in [l5_home, l5_away] if v is not None]
    if len(l5_vals) > 0 and l5_weight > 0:
        l5_avg = sum(l5_vals) / len(l5_vals)
        heuristic_total = (1.0 - l5_weight) * base_heuristic_total + l5_weight * l5_avg
    else:
        heuristic_total = base_heuristic_total

    return heuristic_total, total_score

# ---------- Projected score (goals) ----------
def predict_goals(row):
    """Projected goals for each side with calibration (γ scale, β bias)."""
    home_base = max(0.5, (row.get("xgf60_home", 2.7) + row.get("xga60_away", 2.7)) / 2.0)
    away_base = max(0.5, (row.get("xgf60_away", 2.7) + row.get("xga60_home", 2.7)) / 2.0)

    home_base += 0.10 * row.get("fin_home", 0.0)
    away_base += 0.10 * row.get("fin_away", 0.0)

    home_base += 0.10 * row.get("inj_off_home", 0.0) + 0.10 * row.get("inj_def_away", 0.0)
    away_base += 0.10 * row.get("inj_off_away", 0.0) + 0.10 * row.get("inj_def_home", 0.0)

    g = row.get("goalie_edge_home", 0.0)
    home_base += 0.08 * g
    away_base -= 0.08 * g

    pl = float(row.get("puckline_home", -1.5))
    home_base += -0.10 * pl
    away_base  -= -0.10 * pl

    home_g = max(0.8, home_base) * goals_scale
    away_g = max(0.8, away_base) * goals_scale
    total_g = home_g + away_g + goals_bias

    return {
        "Pred_Home_Goals": round(home_g, 2),
        "Pred_Away_Goals": round(away_g, 2),
        "Pred_Total_Goals": round(total_g, 2),
    }

# ---------- Totals: rule/blend ----------
def compute_total(row):
    """Totals side using rule: Blend / Goals only / Heuristic only / Consensus."""
    market_total = float(row.get("total", 6.0))
    heuristic_total, total_score = compute_total_heuristic(row)
    goals_proj = predict_goals(row)["Pred_Total_Goals"]

    rule = st.session_state.get("g_totals_rule", "Blend (use α)")
    if rule == "Goals only":
        chosen_total = goals_proj
        rationale = "goals"
    elif rule == "Heuristic only":
        chosen_total = heuristic_total
        rationale = "heuristic"
    elif rule == "Consensus (both must agree)":
        side_goals = goals_proj - market_total
        side_heu   = heuristic_total - market_total
        if side_goals * side_heu > 0:
            chosen_total = 0.5 * (goals_proj + heuristic_total)
            rationale = "consensus"
        else:
            return {
                "TotalScore": total_score,
                "ProjectedDeltaGoals": 0.0,
                "OU_Pick": "No Pick",
                "OU_Confidence_%": 0.0,
                "Blended_Total": 0.5 * (goals_proj + heuristic_total),
                "Chosen_Total": 0.5 * (goals_proj + heuristic_total),
                "Rationale": "disagree",
            }
    else:  # Blend
        chosen_total = totals_alpha * goals_proj + (1 - totals_alpha) * heuristic_total
        rationale = "blend"

    over_prob = sigmoid(chosen_total - market_total, scale_total)
    pick = "Over" if chosen_total >= market_total else "Under"
    conf_pct = round(abs(over_prob - 0.5) * 200, 1)

    return {
        "TotalScore": total_score,
        "ProjectedDeltaGoals": chosen_total - market_total,
        "OU_Pick": f"{pick} {market_total:.1f}",
        "OU_Confidence_%": conf_pct,
        "Blended_Total": totals_alpha * goals_proj + (1 - totals_alpha) * heuristic_total,
        "Chosen_Total": chosen_total,
        "Rationale": rationale,
    }

# ---------- Wrapper ----------
def score_game(row):
    win = compute_win(row)
    tot = compute_total(row)
    proj = predict_goals(row)
    lock_score = 0.6 * win["Win_Confidence_%"] + 0.4 * tot["OU_Confidence_%"]
    return {
        "home": row.get("home"), "away": row.get("away"),

        # Win %
        "Home_Win_%": win["Home_Win_%"],
        "Away_Win_%": win["Away_Win_%"],

        # Picks
        "ML_Pick": win["ML_Pick"], "Spread_Pick": win["Spread_Pick"],
        "OU_Pick": tot["OU_Pick"],

        # Confidence
        "Win_Confidence_%": win["Win_Confidence_%"],
        "OU_Confidence_%": tot["OU_Confidence_%"],

        # Predicted score
        "Pred_Home_Goals": proj["Pred_Home_Goals"],
        "Pred_Away_Goals": proj["Pred_Away_Goals"],
        "Pred_Total_Goals": proj["Pred_Total_Goals"],

        # Totals internals
        "Model_Total": tot["Chosen_Total"],
        "Model_Total_Rationale": tot["Rationale"],

        "LockScore": round(lock_score, 1),
    }

# ---------- UI ----------
tabs = st.tabs(["Simple (NFL-style)", "Advanced", "Batch (CSV Upload)"])

# --- Simple (NFL-style) ---
with tabs[0]:
    st.subheader("Minimal Inputs")
    c1, c2 = st.columns(2)
    with c1:
        home = st.text_input("Home Team", "BOS Bruins", key="simp_home")
        away = st.text_input("Away Team", "NY Rangers", key="simp_away")
        puckline_home = st.number_input("Puckline (Home; negative if favored)", -3.0, 3.0, -1.5, 0.5, key="simp_pl")
        total = st.number_input("Market Total (O/U, goals)", 4.5, 8.5, 6.0, 0.5, key="simp_total")
        # NFL-style L5 totals
        l5_home = st.number_input("Home L5 avg game total (goals)", 3.0, 10.0, 6.2, 0.1, key="simp_l5_home")
        l5_away = st.number_input("Away L5 avg game total (goals)", 3.0, 10.0, 6.0, 0.1, key="simp_l5_away")
    with c2:
        rating_home = st.number_input("Home Rating / Power", 1200.0, 2000.0, 1610.0, 5.0, key="simp_rate_h")
        rating_away = st.number_input("Away Rating / Power", 1200.0, 2000.0, 1590.0, 5.0, key="simp_rate_a")
        l10_home = st.slider("Home Win% (last 10)", 0.0, 1.0, 0.60, 0.05, key="simp_l10_h")
        l10_away = st.slider("Away Win% (last 10)", 0.0, 1.0, 0.55, 0.05, key="simp_l10_a")
        b2b_home = st.checkbox("Home on B2B", False, key="simp_b2b_h")
        b2b_away = st.checkbox("Away on B2B", False, key="simp_b2b_a")

    if st.button("Score (Simple)", type="primary", key="simp_score_btn"):
        row = {
            "home": home, "away": away,
            "puckline_home": puckline_home, "total": total,
            "rating_home": rating_home, "rating_away": rating_away,
            "l10_home_winpct": l10_home, "l10_away_winpct": l10_away,
            "b2b_home": int(b2b_home), "b2b_away": int(b2b_away),
            "l5_total_home": l5_home, "l5_total_away": l5_away,
        }
        row = fill_defaults_minimal(row)
        result = score_game(row)
        with st.container(border=True):
            st.markdown(f"**Matchup:** {home} vs {away}")

            # Row: win %, puckline, total
            c1a, c2a, c3a, c4a = st.columns(4)
            c1a.metric(f"{home} Win %", f"{result['Home_Win_%']}%")
            c2a.metric(f"{away} Win %", f"{result['Away_Win_%']}%")
            c3a.metric("Puckline", result["Spread_Pick"])
            c4a.metric("Total", result["OU_Pick"], f"Conf {result['OU_Confidence_%']}%")

            cA, cB = st.columns(2)
            cA.metric("ML Pick", result["ML_Pick"])
            cB.metric("Win Confidence", f"{result['Win_Confidence_%']}%")

            st.markdown(
                f"**Projected Score:** {home} {result['Pred_Home_Goals']:.2f} — "
                f"{away} {result['Pred_Away_Goals']:.2f} "
                f"*(Projected Total {result['Pred_Total_Goals']:.2f}; "
                f"Model Total {result['Model_Total']:.2f} via {result['Model_Total_Rationale']})*"
            )

            st.markdown(f"**Lock Score:** {result['LockScore']} (Lock threshold {lock_min_conf}%)")
            if result["LockScore"] >= lock_min_conf:
                st.success("🔒 Consider this a Lock of the Day candidate.")
            else:
                st.info("Not strong enough for 🔒 by your threshold.")

# --- Advanced (full inputs) ---
with tabs[1]: 
    st.subheader("Full Inputs (optional)")
    c1, c2, c3 = st.columns(3)
    with c1:
        home_a = st.text_input("Home Team", "BOS Bruins", key="adv_home")
        away_a = st.text_input("Away Team", "NY Rangers", key="adv_away")
        puckline_home_a = st.number_input("Puckline (Home; negative if favored)", -3.0, 3.0, -1.5, 0.5, key="adv_pl")
        total_a = st.number_input("Market Total (O/U, goals)", 4.5, 8.5, 6.0, 0.5, key="adv_total")
        l5_home_a = st.number_input("Home L5 avg game total (goals)", 3.0, 10.0, 6.2, 0.1, key="adv_l5_home")
        l5_away_a = st.number_input("Away L5 avg game total (goals)", 3.0, 10.0, 6.0, 0.1, key="adv_l5_away")
        travel_home = st.number_input("Travel Miles (Home)", 0.0, 5000.0, 0.0, 50.0, key="adv_travel_h")
        travel_away = st.number_input("Travel Miles (Away)", 0.0, 5000.0, 600.0, 50.0, key="adv_travel_a")
    with c2:
        rating_home_a = st.number_input("Home Rating / ELO", 1200.0, 2000.0, 1610.0, 5.0, key="adv_rate_h")
        rating_away_a = st.number_input("Away Rating / ELO", 1200.0, 2000.0, 1590.0, 5.0, key="adv_rate_a")
        xgf60_home  = st.number_input("Home xGF/60 (5v5)", 1.5, 4.0, 2.9, 0.05, key="adv_xgf_h")
        xga60_home  = st.number_input("Home xGA/60 (5v5)", 1.5, 4.0, 2.5, 0.05, key="adv_xga_h")
        xgf60_away  = st.number_input("Away xGF/60 (5v5)", 1.5, 4.0, 2.6, 0.05, key="adv_xgf_a")
        xga60_away  = st.number_input("Away xGA/60 (5v5)", 1.5, 4.0, 2.7, 0.05, key="adv_xga_a")
    with c3:
        l10_home_a = st.slider("Home Win% (last 10)", 0.0, 1.0, 0.60, 0.05, key="adv_l10_h")
        l10_away_a = st.slider("Away Win% (last 10)", 0.0, 1.0, 0.55, 0.05, key="adv_l10_a")
        rest_home = st.number_input("Home Rest Days", 0.0, 5.0, 2.0, 0.5, key="adv_rest_h")
        rest_away = st.number_input("Away Rest Days", 0.0, 5.0, 1.0, 0.5, key="adv_rest_a")
        b2b_home  = st.checkbox("Home on B2B", False, key="adv_b2b_h")
        b2b_away  = st.checkbox("Away on B2B", False, key="adv_b2b_a")
        inj_off_home = st.slider("Home Offense Missing (rel.)", 0.0, 5.0, 0.5, 0.1, key="adv_injoff_h")
        inj_off_away = st.slider("Away Offense Missing (rel.)", 0.0, 5.0, 1.0, 0.1, key="adv_injoff_a")
        inj_def_home = st.slider("Home Defense Missing (rel.)", 0.0, 5.0, 0.3, 0.1, key="adv_injdef_h")
        inj_def_away = st.slider("Away Defense Missing (rel.)", 0.0, 5.0, 0.6, 0.1, key="adv_injdef_a")
        goalie_edge_home = st.slider("Goalie Edge (Home − Away, + favors home)", -3.0, 3.0, 0.5, 0.1, key="adv_goalie")
        fin_home = st.slider("Home Finishing (± hot/cold)", -2.0, 2.0, 0.2, 0.1, key="adv_fin_h")
        fin_away = st.slider("Away Finishing (± hot/cold)", -2.0, 2.0, -0.1, 0.1, key="adv_fin_a")
        recent_ou_home = st.slider("Home Recent O/U Trend (avg delta, +Over)", -2.0, 2.0, 0.3, 0.1, key="adv_ou_h")
        recent_ou_away = st.slider("Away Recent O/U Trend (avg delta, +Over)", -2.0, 2.0, -0.2, 0.1, key="adv_ou_a")

    if st.button("Score (Advanced)", key="adv_score_btn"):
        row = {
            "home": home_a, "away": away_a,
            "puckline_home": puckline_home_a, "total": total_a,
            "travel_home": travel_home, "travel_away": travel_away,
            "rating_home": rating_home_a, "rating_away": rating_away_a,
            "xgf60_home": xgf60_home, "xga60_home": xga60_home,
            "xgf60_away": xgf60_away, "xga60_away": xga60_away,
            "l10_home_winpct": l10_home_a, "l10_away_winpct": l10_away_a,
            "rest_home": rest_home, "rest_away": rest_away,
            "b2b_home": int(b2b_home), "b2b_away": int(b2b_away),
            "inj_off_home": inj_off_home, "inj_off_away": inj_off_away,
            "inj_def_home": inj_def_home, "inj_def_away": inj_def_away,
            "goalie_edge_home": goalie_edge_home, "fin_home": fin_home, "fin_away": fin_away,
            "recent_ou_home": recent_ou_home, "recent_ou_away": recent_ou_away,
            "l5_total_home": l5_home_a, "l5_total_away": l5_away_a,
        }
        result = score_game(row)
        with st.container(border=True):
            st.markdown(f"**Matchup:** {home_a} vs {away_a}")

            c1a, c2a, c3a, c4a = st.columns(4)
            c1a.metric(f"{home_a} Win %", f"{result['Home_Win_%']}%")
            c2a.metric(f"{away_a} Win %", f"{result['Away_Win_%']}%")
            c3a.metric("Puckline", result["Spread_Pick"])
            c4a.metric("Total", result["OU_Pick"], f"Conf {result['OU_Confidence_%']}%")

            cA, cB = st.columns(2)
            cA.metric("ML Pick", result["ML_Pick"])
            cB.metric("Win Confidence", f"{result['Win_Confidence_%']}%")

            st.markdown(
                f"**Projected Score:** {home_a} {result['Pred_Home_Goals']:.2f} — "
                f"{away_a} {result['Pred_Away_Goals']:.2f} "
                f"*(Projected Total {result['Pred_Total_Goals']:.2f}; "
                f"Model Total {result['Model_Total']:.2f} via {result['Model_Total_Rationale']})*"
            )

            st.markdown(f"**Lock Score:** {result['LockScore']} (Lock threshold {lock_min_conf}%)")
            if result["LockScore"] >= lock_min_conf:
                st.success("🔒 Consider this a Lock of the Day candidate.")
            else:
                st.info("Not strong enough for 🔒 by your threshold.")

# --- Batch (CSV) ---
with tabs[2]:
    st.subheader("Batch — Upload CSV")
    simple_csv = st.toggle("Use Simple CSV format", value=True, key="batch_simple_toggle")
    if simple_csv:
        st.caption("Simple CSV needs: home,away,puckline_home,total,rating_home,rating_away (optional l10/b2b,l5_totals).")
        ex = pd.DataFrame([{
            "home":"BOS Bruins","away":"NY Rangers","puckline_home":-1.5,"total":6.0,
            "rating_home":1620,"rating_away":1605,"l10_home_winpct":0.60,"l10_away_winpct":0.55,
            "b2b_home":0,"b2b_away":0,"l5_total_home":6.1,"l5_total_away":6.0
        }]])
    else:
        st.caption("Advanced CSV requires all columns used by the Advanced tab (L5 optional).")
        ex = pd.DataFrame([{
            "home":"BOS Bruins","away":"NY Rangers","puckline_home":-1.5,"total":6.0,
            "rating_home":1620,"rating_away":1605,
            "xgf60_home":2.9,"xga60_home":2.5,"xgf60_away":2.6,"xga60_away":2.7,
            "l10_home_winpct":0.60,"l10_away_winpct":0.55,"rest_home":2,"rest_away":1,
            "b2b_home":0,"b2b_away":0,"inj_off_home":0.4,"inj_off_away":1.0,"inj_def_home":0.2,"inj_def_away":0.5,
            "goalie_edge_home":0.6,"fin_home":0.2,"fin_away":-0.1,"recent_ou_home":0.3,"recent_ou_away":-0.2,
            "travel_home":0,"travel_away":450,"l5_total_home":6.1,"l5_total_away":6.0
        }]])
    st.download_button("Download CSV Template", ex.to_csv(index=False).encode(),
                       "nhl_lockbot_template.csv", "text/csv", key="batch_dl")

    uploaded = st.file_uploader("Upload CSV", type=["csv"], key="batch_uploader")
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            if simple_csv:
                need = ["home","away","puckline_home","total","rating_home","rating_away"]
                miss = [c for c in need if c not in df.columns]
                if miss:
                    st.error(f"Missing columns: {miss}")
                    st.stop()
                rows = []
                for _, r in df.iterrows():
                    row = fill_defaults_minimal({k: r[k] for k in df.columns})
                    rows.append(score_game(row))
                out = pd.DataFrame(rows).sort_values("LockScore", ascending=False).reset_index(drop=True)
            else:
                need = ["home","away","puckline_home","total","rating_home","rating_away",
                        "xgf60_home","xga60_home","xgf60_away","xga60_away",
                        "l10_home_winpct","l10_away_winpct","rest_home","rest_away",
                        "b2b_home","b2b_away","inj_off_home","inj_off_away",
                        "inj_def_home","inj_def_away","goalie_edge_home","fin_home","fin_away",
                        "recent_ou_home","recent_ou_away","travel_home","travel_away"]
                miss = [c for c in need if c not in df.columns]
                if miss:
                    st.error(f"Missing columns: {miss}")
                    st.stop()
                out = pd.DataFrame([score_game(fill_defaults_minimal(dict(r))) for _, r in df.iterrows()]) \
                        .sort_values("LockScore", ascending=False).reset_index(drop=True)

            # Reorder columns nicely
            cols = ["home","away","Home_Win_%","Away_Win_%","ML_Pick","Spread_Pick","OU_Pick",
                    "Pred_Home_Goals","Pred_Away_Goals","Pred_Total_Goals",
                    "Model_Total","Model_Total_Rationale",
                    "Win_Confidence_%","OU_Confidence_%","LockScore"]
            out = out[[c for c in cols if c in out.columns] + [c for c in out.columns if c not in cols]]

            if len(out):
                out.loc[0, "Lock"] = "🔒" if out.loc[0, "LockScore"] >= lock_min_conf else "(no 🔒)"
            st.dataframe(out, use_container_width=True)
            st.download_button("Download Picks CSV", out.to_csv(index=False).encode(),
                               "nhl_lockbot_picks.csv", "text/csv", key="batch_out_dl")
        except Exception as e:
            st.exception(e)
