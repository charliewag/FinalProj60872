"""
make_figures.py

Generates all five analysis figures and saves them to analysis/figures/.
Clears the folder first so only the five canonical figures remain.

  EV1_llm_distributions.png   — violin: one-axis vs three-axis agg_score
  FA1_company_boxplots.png    — agg_score by company (hand-edited, top 15)
  FA2_temporal_trends.png     — agg_score over time
  FA3_failure_taxonomy.png    — mean agg_score by failure type (all sources, 10-15 entries)
  FA4c_keyword_drivers.png    — TF-IDF Ridge keyword drivers (technical vs political)
"""

import json, sys, shutil
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

sys.stdout.reconfigure(encoding="utf-8")

ROOT   = Path(__file__).parent.parent
DATA   = ROOT / "data"
FIGDIR = Path(__file__).parent / "figures"

HAND   = {"danluu", "aws", "manual", "icco"}
TECH_COL = "#1565C0"
POL_COL  = "#C62828"
AGG_COL  = "#2E7D32"
BG       = "#FAFAFA"


# ─────────────────────────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────────────────────────

def setup():
    if FIGDIR.exists():
        shutil.rmtree(FIGDIR)
    FIGDIR.mkdir(parents=True)
    print(f"Cleared and recreated {FIGDIR}")


def load():
    with open(DATA / "postmortems_agg_oneaxis.json",   encoding="utf-8") as f:
        oa = json.load(f)
    with open(DATA / "postmortems_agg_threeaxis.json", encoding="utf-8") as f:
        ta = json.load(f)
    with open(DATA / "model_combined.json",            encoding="utf-8") as f:
        kw = json.load(f)
    return oa, ta, kw


# ─────────────────────────────────────────────────────────────────────────────
# EV1 — violin: one-axis vs three-axis agg_score distributions
# ─────────────────────────────────────────────────────────────────────────────

def ev1_violin(oa, ta):
    oa_hand = [r["agg_score"] for r in oa if r.get("agg_score") is not None and r.get("source") in HAND]
    oa_news = [r["agg_score"] for r in oa if r.get("agg_score") is not None and r.get("source") == "fail_news"]
    ta_hand = [r["agg_score"] for r in ta if r.get("agg_score") is not None and r.get("source") in HAND]
    ta_news = [r["agg_score"] for r in ta if r.get("agg_score") is not None and r.get("source") == "fail_news"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 7), sharey=True)
    fig.patch.set_facecolor(BG)
    fig.suptitle("LLM Score Distributions", fontsize=13, fontweight="bold")

    configs = [
        (axes[0], oa_hand, oa_news, "One-Axis"),
        (axes[1], ta_hand, ta_news, "Three-Axis"),
    ]

    for ax, hand, news, title in configs:
        parts = ax.violinplot([hand, news], positions=[1, 2],
                              showmedians=True, showextrema=True)
        for i, (pc, col) in enumerate(zip(parts["bodies"], [TECH_COL, POL_COL])):
            pc.set_facecolor(col)
            pc.set_alpha(0.55)
            pc.set_edgecolor(col)
        parts["cmedians"].set_colors([TECH_COL, POL_COL])
        parts["cmedians"].set_linewidth(2.5)
        for key in ("cbars", "cmins", "cmaxes"):
            parts[key].set_colors(["#888888"])
            parts[key].set_linewidth(1)

        # overlay box quartiles
        for vals, x, col in [(hand, 1, TECH_COL), (news, 2, POL_COL)]:
            q1, med, q3 = np.percentile(vals, [25, 50, 75])
            ax.plot([x - 0.06, x + 0.06], [med, med], color="white", lw=2.5, zorder=5)
            ax.add_patch(plt.Rectangle((x - 0.06, q1), 0.12, q3 - q1,
                                       facecolor=col, edgecolor="white",
                                       linewidth=1.2, alpha=0.85, zorder=4))
            mean = np.mean(vals)
            ax.scatter([x], [mean], color="white", s=28, zorder=6)

        # stats annotation
        for vals, x, col, label in [(hand, 1, TECH_COL, "tech postmortems"),
                                    (news, 2, POL_COL,  "fail-news")]:
            ax.text(x, -0.6,
                    f"{label}\nn={len(vals)}  μ={np.mean(vals):.2f}  σ={np.std(vals):.2f}",
                    ha="center", fontsize=8.5, color=col)

        ax.set_xticks([]); ax.set_xlim(0.4, 2.6)
        ax.set_ylim(-1, 9); ax.set_ylabel("Score (0–8)", fontsize=10)
        ax.set_title(title, fontsize=10, fontweight="bold", pad=10)
        ax.set_facecolor(BG)
        ax.spines[["top", "right"]].set_visible(False)
        ax.axhline(0, color="#DDDDDD", lw=0.8); ax.axhline(8, color="#DDDDDD", lw=0.8)
        ax.axhline(4, color="#CCCCCC", lw=0.8, linestyle="--")

    plt.tight_layout()
    out = FIGDIR / "EV1_llm_distributions.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"Saved {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# FA1 — company boxplots (three-axis agg_score, hand-edited only)
# ─────────────────────────────────────────────────────────────────────────────

def fa1_company(ta):
    hand = [r for r in ta if r.get("source") in HAND and r.get("agg_score") is not None]

    by_co = defaultdict(list)
    for r in hand:
        by_co[r.get("company", "Unknown")].append(r["agg_score"])

    ranked = [(c, s) for c, s in by_co.items() if len(s) >= 2]
    ranked.sort(key=lambda x: np.median(x[1]), reverse=True)
    ranked = ranked[:15]

    companies = [c for c, _ in ranked]
    data      = [s for _, s in ranked]
    counts    = [len(s) for s in data]

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor(BG)

    bp = ax.boxplot(data, patch_artist=True, vert=True, showfliers=True,
                    medianprops=dict(color="white", linewidth=2.2),
                    flierprops=dict(marker="o", markersize=4, alpha=0.4,
                                   markerfacecolor="#555555", markeredgecolor="none"),
                    widths=0.55)

    for i, (box, d) in enumerate(zip(bp["boxes"], data)):
        intensity = 0.3 + 0.55 * (np.median(d) / 8.0)
        box.set_facecolor(cm.Blues(intensity))
        box.set_edgecolor("#1565C0")
        box.set_linewidth(0.8)
    for w in bp["whiskers"] + bp["caps"]:
        w.set_color("#888888"); w.set_linewidth(0.9)

    ax.set_xticks(range(1, len(companies) + 1))
    ax.set_xticklabels(
        [f"{c}\n(n={n})" for c, n in zip(companies, counts)],
        rotation=35, ha="right", fontsize=9
    )
    ax.set_ylabel("Score (0–8)", fontsize=11)
    ax.set_ylim(-0.5, 9)
    ax.axhline(5.5, color="#1565C0", linestyle="--", linewidth=1, alpha=0.5, label="Technical threshold (5.5)")
    ax.axhline(3.0, color="#C62828", linestyle="--", linewidth=1, alpha=0.5, label="Political threshold (3.0)")
    ax.set_title("Score by Company", fontsize=12, fontweight="bold")
    ax.set_facecolor(BG)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9)

    plt.tight_layout()
    out = FIGDIR / "FA1_company_boxplots.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"Saved {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# FA2 — agg_score over time (three-axis, hand-edited only)
# ─────────────────────────────────────────────────────────────────────────────

def fa2_temporal(ta):
    def parse_year(d):
        try: return int(str(d)[:4])
        except: return None

    hand = [r for r in ta if r.get("source") in HAND
            and r.get("agg_score") is not None and r.get("date")]

    by_year = defaultdict(list)
    for r in hand:
        y = parse_year(r["date"])
        if y and 2000 <= y <= 2026:
            by_year[y].append(r["agg_score"])

    years = sorted(y for y, s in by_year.items() if len(s) >= 2)
    if len(years) < 3:
        print("  Not enough years for temporal plot"); return

    data_yr  = [by_year[y] for y in years]
    counts   = [len(d) for d in data_yr]
    medians  = [np.median(d) for d in data_yr]
    means    = [np.mean(d)   for d in data_yr]
    q1s      = [np.percentile(d, 25) for d in data_yr]
    q3s      = [np.percentile(d, 75) for d in data_yr]

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor(BG)

    # IQR shading
    ax.fill_between(years, q1s, q3s, alpha=0.15, color=AGG_COL, label="IQR (25–75th pct)")
    ax.plot(years, medians, "o-", color=AGG_COL, linewidth=2.2, markersize=6,
            label="Median agg score", zorder=4)
    ax.plot(years, means,   "s--", color=AGG_COL, linewidth=1.2, markersize=4,
            alpha=0.6, label="Mean agg score")

    # doc count labels
    for y, n, med in zip(years, counts, medians):
        ax.text(y, med + 0.25, f"n={n}", ha="center", fontsize=7.5,
                color="#555555", zorder=5)

    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Score (0–8)", fontsize=11)
    ax.set_ylim(0, 9)
    ax.set_xticks(years); ax.set_xticklabels(years, rotation=45, ha="right", fontsize=9)
    ax.axhline(5.5, color=TECH_COL, linestyle=":", linewidth=1, alpha=0.5)
    ax.axhline(3.0, color=POL_COL,  linestyle=":", linewidth=1, alpha=0.5)
    ax.set_title("Score Over Time Tech Postmortems", fontsize=12, fontweight="bold")
    ax.set_facecolor(BG)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9, labels=["IQR (25–75th pct)", "Median score", "Mean score"])

    plt.tight_layout()
    out = FIGDIR / "FA2_temporal_trends.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"Saved {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# FA3 — failure type (all sources, sorted most→least technical, 12 entries)
# ─────────────────────────────────────────────────────────────────────────────

def fa3_failure_type(ta):
    def norm(ft):
        ft = ft.lower().strip()
        ft = ft.replace("deployment failure", "deployment error")
        ft = ft.replace("dns configuration error", "dns error")
        ft = ft.replace("dns resolution error",   "dns error")
        ft = ft.replace("network outage", "network failure")
        return ft

    by_type = defaultdict(list)
    by_src  = defaultdict(lambda: defaultdict(int))

    for r in ta:
        if not r.get("failure_type") or r.get("agg_score") is None:
            continue
        ft  = norm(r["failure_type"])
        src = "hand" if r.get("source") in HAND else "news"
        by_type[ft].append(r["agg_score"])
        by_src[ft][src] += 1

    # require >= 3 docs total and at least 1 tech postmortem, keep top 12 by mean score
    ranked = [(ft, scores) for ft, scores in by_type.items()
              if len(scores) >= 3 and by_src[ft]["hand"] >= 1]
    ranked.sort(key=lambda x: np.mean(x[1]), reverse=True)
    ranked = ranked[:12]

    types  = [ft for ft, _ in ranked]
    means  = [np.mean(s) for _, s in ranked]
    errors = [np.std(s) / len(s)**0.5 for _, s in ranked]   # SEM
    n_hand = [by_src[ft]["hand"] for ft in types]
    n_news = [by_src[ft]["news"] for ft in types]
    totals = [h + nw for h, nw in zip(n_hand, n_news)]

    # colour by mean: green=technical, red=political
    colours = [TECH_COL if m >= 5.5 else (POL_COL if m <= 3.0 else "#F57F17")
               for m in means]

    fig, ax = plt.subplots(figsize=(13, 7))
    fig.patch.set_facecolor(BG)

    y = np.arange(len(types))
    bars = ax.barh(y, means, xerr=errors, color=colours, alpha=0.82,
                   edgecolor="white", linewidth=0.5,
                   error_kw=dict(elinewidth=1.2, capsize=3, capthick=1.2,
                                 ecolor="#555555"))

    # labels: placed past the error bar tip so they never overlap
    for i, (mean, err, h, nw, tot) in enumerate(zip(means, errors, n_hand, n_news, totals)):
        ax.text(mean + err + 0.18, i, f"{mean:.2f}  (n={tot}: {h} tpm, {nw} news)",
                va="center", ha="left", fontsize=8.5, color="#333333")

    ax.set_yticks(y)
    ax.set_yticklabels(types, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Mean score (0–8, ±SEM)", fontsize=10)
    ax.set_xlim(0, 10.5)
    ax.axvline(5.5, color=TECH_COL, linestyle="--", linewidth=1, alpha=0.5,
               label="Technical (5.5)")
    ax.axvline(3.0, color=POL_COL,  linestyle="--", linewidth=1, alpha=0.5,
               label="Political (3.0)")
    ax.set_title("Score by Failure Type", fontsize=12, fontweight="bold")
    ax.set_facecolor(BG)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9, loc="lower right")

    plt.tight_layout()
    out = FIGDIR / "FA3_failure_taxonomy.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"Saved {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# FA3b — failure type, tech postmortems only
# ─────────────────────────────────────────────────────────────────────────────

def fa3b_failure_type_tpm(ta):
    def norm(ft):
        ft = ft.lower().strip()
        ft = ft.replace("deployment failure", "deployment error")
        ft = ft.replace("dns configuration error", "dns error")
        ft = ft.replace("dns resolution error",   "dns error")
        ft = ft.replace("network outage", "network failure")
        return ft

    by_type = defaultdict(list)

    for r in ta:
        if not r.get("failure_type") or r.get("agg_score") is None:
            continue
        if r.get("source") not in HAND:
            continue
        ft = norm(r["failure_type"])
        by_type[ft].append(r["agg_score"])

    # require >= 2 docs (smaller corpus), top 12 by mean score
    ranked = [(ft, scores) for ft, scores in by_type.items() if len(scores) >= 2]
    ranked.sort(key=lambda x: np.mean(x[1]), reverse=True)
    ranked = ranked[:12]

    types  = [ft for ft, _ in ranked]
    means  = [np.mean(s) for _, s in ranked]
    errors = [np.std(s) / len(s)**0.5 for _, s in ranked]
    counts = [len(s) for _, s in ranked]

    colours = [TECH_COL if m >= 5.5 else (POL_COL if m <= 3.0 else "#F57F17")
               for m in means]

    fig, ax = plt.subplots(figsize=(13, 7))
    fig.patch.set_facecolor(BG)

    y = np.arange(len(types))
    ax.barh(y, means, xerr=errors, color=colours, alpha=0.82,
            edgecolor="white", linewidth=0.5,
            error_kw=dict(elinewidth=1.2, capsize=3, capthick=1.2, ecolor="#555555"))

    for i, (mean, err, n) in enumerate(zip(means, errors, counts)):
        ax.text(mean + err + 0.18, i, f"{mean:.2f}  (n={n} tpm)",
                va="center", ha="left", fontsize=8.5, color="#333333")

    ax.set_yticks(y)
    ax.set_yticklabels(types, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Mean score (0–8, ±SEM)", fontsize=10)
    ax.set_xlim(0, 10.5)
    ax.axvline(5.5, color=TECH_COL, linestyle="--", linewidth=1, alpha=0.5,
               label="Technical (5.5)")
    ax.axvline(3.0, color=POL_COL,  linestyle="--", linewidth=1, alpha=0.5,
               label="Political (3.0)")
    ax.set_title("Score by Failure Type  (tech postmortems only)", fontsize=12, fontweight="bold")
    ax.set_facecolor(BG)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9, loc="lower right")

    plt.tight_layout()
    out = FIGDIR / "FA3b_failure_taxonomy_tpm.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"Saved {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# FA4c — keyword drivers (technical vs political, combined model)
# ─────────────────────────────────────────────────────────────────────────────

def fa4c_keywords(kw):
    tech = [(x["keyword"], x["coefficient"])       for x in kw["technical"][:20]]
    pol  = [(x["keyword"], abs(x["coefficient"]))  for x in kw["political"][:20]]

    fig, (ax_t, ax_p) = plt.subplots(1, 2, figsize=(16, 9))
    fig.patch.set_facecolor(BG)
    fig.suptitle("Keyword Drivers", fontsize=13, fontweight="bold")

    def panel(ax, kws, colour, title, xlabel):
        words = [w for w, _ in kws]; vals = [v for _, v in kws]
        y = np.arange(len(words))
        ax.barh(y, vals, color=colour, alpha=0.85, edgecolor="white", linewidth=0.4)
        max_v = max(vals)
        for i, v in enumerate(vals):
            ax.text(v + max_v * 0.012, i, f"{v:.3f}", va="center", ha="left",
                    fontsize=8.5, color="#333333")
        ax.set_yticks(y); ax.set_yticklabels(words, fontsize=11)
        ax.invert_yaxis()
        ax.set_xlabel(xlabel, fontsize=10, labelpad=6)
        ax.set_title(title, fontsize=13, fontweight="bold", color=colour, pad=10)
        ax.set_facecolor(BG)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines["left"].set_color("#CCCCCC"); ax.spines["bottom"].set_color("#CCCCCC")
        ax.tick_params(axis="x", labelsize=9); ax.set_xlim(0, max_v * 1.18)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f}"))

    panel(ax_t, tech, TECH_COL, "Technical",
          "Ridge coefficient  (binary label 1, higher = more technical)")
    panel(ax_p, pol,  POL_COL,  "Political",
          "Ridge coefficient magnitude  (binary label 0, higher = more political)")

    plt.tight_layout()
    out = FIGDIR / "FA4c_keyword_drivers.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"Saved {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    setup()
    print("\nLoading data...")
    oa, ta, kw = load()
    print(f"  one-axis:   {sum(1 for r in oa if r.get('agg_score') is not None)} scored")
    print(f"  three-axis: {sum(1 for r in ta if r.get('agg_score') is not None)} scored")

    print("\nGenerating figures...")
    ev1_violin(oa, ta)
    fa1_company(ta)
    fa2_temporal(ta)
    fa3_failure_type(ta)
    fa3b_failure_type_tpm(ta)
    fa4c_keywords(kw)

    print(f"\nDone — {len(list(FIGDIR.glob('*.png')))} figures in {FIGDIR}")


if __name__ == "__main__":
    main()
