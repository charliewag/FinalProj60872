"""
final_analysis.py

Deep analysis of the final (V4) pipeline outputs.

Sections
--------
1. Company breakdown — agg_score and axis scores by company (top-N)
2. Temporal trends  — scores by year
3. Failure taxonomy — failure_type distribution and axis profiles
4. Keyword showcase — top technical / political / pseudo_technical terms with bar charts
5. Exemplar sources — highest and lowest scoring docs per axis with quotes

Outputs saved to analysis/figures/:
  FA1_company_boxplots.png
  FA2_temporal_trends.png
  FA3_failure_taxonomy.png
  FA4_keyword_showcase.png
  FA5_exemplar_sources.png
  FA6_axis_profiles.png        — scatter: tech vs political, coloured by pseudo_technical
  FA7_source_comparison.png    — mean scores per data source (all 5 sources)
"""

import json, sys, warnings
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import Ridge

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

EXTRA_SW = {
    "fv","ln","pq","ap","afp","said","says","told","reuters","mr","ms","mrs",
    "dr","spokesman","spokeswoman","spokesperson","copyright","rights",
    "reserved","associated","reprinted","permission","article","story","wrote",
    "writing","editor","editors","yesterday","monday","tuesday","wednesday",
    "thursday","friday","saturday","sunday","jan","feb","mar","apr","jun",
    "jul","aug","sep","oct","nov","dec","pst","est","cst","mst","gmt","utc",
    "2019","2020","2021","2022","2023","2024","ve","ll","re","don","didn",
    "doesn","wasn","aren","isn","couldn","wouldn","shouldn","won",
}
SW = list(ENGLISH_STOP_WORDS.union(EXTRA_SW))

ROOT   = Path(__file__).parent.parent
DATA   = ROOT / "data"
FIGDIR = Path(__file__).parent / "figures"
FIGDIR.mkdir(exist_ok=True)

HAND_SOURCES = {"danluu", "aws", "manual", "icco"}
SOURCE_LABELS = {
    "danluu":    "danluu.com",
    "aws":       "AWS",
    "manual":    "Manual",
    "icco":      "ICCO",
    "fail_news": "Fail News",
}
SOURCE_COLOURS = {
    "danluu":    "#1565C0",
    "aws":       "#0288D1",
    "manual":    "#00838F",
    "icco":      "#558B2F",
    "fail_news": "#9C27B0",
}

STYLE = {
    "technical":        "#1565C0",
    "political":        "#C62828",
    "pseudo_technical": "#E65100",
    "agg":              "#2E7D32",
}

# ─────────────────────────────────────────────────────────────────────────────
# Load
# ─────────────────────────────────────────────────────────────────────────────

HAND_SOURCES_SET = {"danluu", "aws", "manual", "icco"}

def train_3axis_keywords(agg):
    """
    Retrain three Ridge models on the 3-axis LLM scores using a shared
    TF-IDF vectorizer (2000 features, min_df=2) fitted on hand-edited docs.

    Returns dict: axis -> (sorted_high_keywords, sorted_low_keywords)
      each entry: list of (word, coef) tuples, top 25
    """
    hand = [r for r in agg
            if r.get("source") in HAND_SOURCES_SET
            and r.get("technical") is not None
            and r.get("text")]

    texts = [r["text"] for r in hand]

    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=2000,
                          stop_words=SW, min_df=2, sublinear_tf=True)
    X = vec.fit_transform(texts)
    names = vec.get_feature_names_out()

    result = {}
    for axis, key in [("technical",       "technical"),
                      ("pseudo_technical", "pseudo_technical"),
                      ("political",        "political")]:
        targets = np.array([r[key] for r in hand], dtype=float)
        clf = Ridge(alpha=10.0)
        clf.fit(X, targets)
        coefs = clf.coef_
        ranked = sorted(zip(coefs, names), reverse=True)
        high = [(w, float(c)) for c, w in ranked if c > 0.005][:25]
        low  = [(w, float(c)) for c, w in ranked if c < -0.005]
        low  = sorted(low, key=lambda x: x[1])[:25]
        result[axis] = (high, low)
        print(f"  3-axis kw [{axis}]: top+ = {high[0][0] if high else 'none'}  "
              f"top- = {low[0][0] if low else 'none'}")

    return result


def load():
    with open(DATA / "postmortems_agg_threeaxis.json") as f: agg = json.load(f)
    with open(DATA / "model_comparison.json")   as f: m_cmp = json.load(f)

    print("  Training 3-axis keyword models for interpretable keyword display...")
    axis_kw = train_3axis_keywords(agg)

    return agg, m_cmp, axis_kw


# ─────────────────────────────────────────────────────────────────────────────
# 1. Company breakdown
# ─────────────────────────────────────────────────────────────────────────────

def plot_company_boxplots(agg):
    hand = [r for r in agg if r.get("source") in HAND_SOURCES
            and r.get("agg_score") is not None]

    # keep companies with >= 2 docs
    by_company = defaultdict(list)
    for r in hand:
        by_company[r.get("company", "Unknown")].append(r["agg_score"])

    # sort by median, keep top 15
    ranked = sorted(by_company.items(), key=lambda x: np.median(x[1]), reverse=True)
    ranked = [(c, s) for c, s in ranked if len(s) >= 2][:15]

    companies = [c for c, _ in ranked]
    data      = [s for _, s in ranked]
    counts    = [len(s) for s in data]

    fig, ax = plt.subplots(figsize=(14, 6))
    bp = ax.boxplot(data, patch_artist=True, vert=True, showfliers=True,
                    medianprops=dict(color="black", linewidth=2),
                    flierprops=dict(marker="o", markersize=4, alpha=0.4),
                    widths=0.55)

    cmap = cm.Blues
    for i, (box, d) in enumerate(zip(bp["boxes"], data)):
        intensity = 0.35 + 0.5 * (np.median(d) / 8.0)
        box.set_facecolor(cmap(intensity))

    ax.set_xticks(range(1, len(companies)+1))
    ax.set_xticklabels(
        [f"{c}\n(n={n})" for c, n in zip(companies, counts)],
        rotation=35, ha="right", fontsize=9
    )
    ax.set_ylabel("Aggregate engineer-utility score  (0–8)", fontsize=11)
    ax.set_ylim(-0.5, 9)
    ax.axhline(5.5, color="gray", linestyle="--", linewidth=1, alpha=0.7,
               label="'Technical' threshold (5.5)")
    ax.axhline(2.5, color="salmon", linestyle="--", linewidth=1, alpha=0.7,
               label="'Political' threshold (2.5)")
    ax.set_title("Aggregate Score by Company  (hand-edited postmortems only, ≥2 docs)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)

    plt.tight_layout()
    out = FIGDIR / "FA1_company_boxplots.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Temporal trends
# ─────────────────────────────────────────────────────────────────────────────

def plot_temporal(agg):
    hand = [r for r in agg if r.get("source") in HAND_SOURCES
            and r.get("agg_score") is not None and r.get("date")]

    def parse_year(d):
        try: return int(str(d)[:4])
        except: return None

    by_year = defaultdict(list)
    by_year_axes = defaultdict(lambda: {"technical": [], "political": [], "pseudo_technical": []})
    for r in hand:
        y = parse_year(r.get("date"))
        if y and 2010 <= y <= 2026:
            by_year[y].append(r["agg_score"])
            for ax in ("technical", "political", "pseudo_technical"):
                if r.get(ax) is not None:
                    by_year_axes[y][ax].append(r[ax])

    years = sorted(by_year.keys())
    if len(years) < 3:
        print("  Not enough years for temporal plot")
        return

    fig, axes = plt.subplots(2, 1, figsize=(13, 10))

    # Panel 1: agg_score over time (box per year)
    ax = axes[0]
    data_yr = [by_year[y] for y in years]
    counts   = [len(d) for d in data_yr]
    bp = ax.boxplot(data_yr, positions=years, patch_artist=True,
                    widths=0.5, showfliers=False,
                    medianprops=dict(color="black", linewidth=2))
    for box in bp["boxes"]:
        box.set_facecolor(STYLE["agg"] + "88")
    # trend line
    medians = [np.median(d) for d in data_yr]
    ax.plot(years, medians, "o-", color=STYLE["agg"], linewidth=2, markersize=6,
            label="Median agg_score")
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Agg score  (0–8)", fontsize=11)
    ax.set_title("Engineer-Utility Score Over Time  (hand-edited postmortems)",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(-0.5, 9)
    for y, n in zip(years, counts):
        ax.text(y, -0.3, f"n={n}", ha="center", fontsize=7.5, color="gray")
    ax.legend(fontsize=9)

    # Panel 2: mean axis scores over time
    ax = axes[1]
    for axis, colour in [("technical", STYLE["technical"]),
                         ("political", STYLE["political"]),
                         ("pseudo_technical", STYLE["pseudo_technical"])]:
        means = [np.mean(by_year_axes[y][axis]) if by_year_axes[y][axis] else np.nan
                 for y in years]
        ax.plot(years, means, "o-", color=colour, linewidth=2, markersize=5,
                label=axis.replace("_", " ").title())
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Mean LLM score  (0–4 per axis)", fontsize=11)
    ax.set_title("Three-Axis LLM Scores Over Time", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 4.5)
    ax.legend(fontsize=9)
    ax.axhline(2, color="gray", linestyle=":", linewidth=1, alpha=0.5)

    plt.tight_layout()
    out = FIGDIR / "FA2_temporal_trends.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Failure taxonomy
# ─────────────────────────────────────────────────────────────────────────────

def plot_failure_taxonomy(agg):
    hand = [r for r in agg if r.get("source") in HAND_SOURCES
            and r.get("failure_type") and r.get("agg_score") is not None]

    # Normalise failure types (lowercase, strip)
    def norm(ft):
        ft = ft.lower().strip()
        # merge near-duplicates
        ft = ft.replace("deployment failure", "deployment error")
        ft = ft.replace("dns configuration error", "dns error")
        ft = ft.replace("dns resolution error", "dns error")
        return ft

    by_type = defaultdict(list)
    by_type_axes = defaultdict(lambda: {"technical":[],"political":[],"pseudo_technical":[]})
    for r in hand:
        ft = norm(r["failure_type"])
        by_type[ft].append(r["agg_score"])
        for ax in ("technical","political","pseudo_technical"):
            if r.get(ax) is not None:
                by_type_axes[ft][ax].append(r[ax])

    # keep types with >= 2 docs
    ranked = sorted(by_type.items(), key=lambda x: -np.mean(x[1]))
    ranked = [(t, s) for t, s in ranked if len(s) >= 2][:14]

    types  = [t for t, _ in ranked]
    counts = [len(s) for _, s in ranked]
    means  = [np.mean(s) for _, s in ranked]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Panel 1: mean agg_score per type
    ax = axes[0]
    colours = [STYLE["agg"] if m >= 5 else (STYLE["political"] if m <= 3 else "#F9A825")
               for m in means]
    bars = ax.barh(range(len(types)), means, color=colours,
                   edgecolor="black", linewidth=0.6)
    ax.set_yticks(range(len(types)))
    ax.set_yticklabels([f"{t}  (n={n})" for t, n in zip(types, counts)], fontsize=9)
    ax.set_xlabel("Mean agg score  (0–8)", fontsize=10)
    ax.set_title("Mean Engineer-Utility Score by Failure Type", fontsize=11, fontweight="bold")
    ax.set_xlim(0, 8.5)
    ax.axvline(4, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    for bar, val in zip(bars, means):
        ax.text(val + 0.1, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}", va="center", fontsize=8)

    # Panel 2: 3-axis heatmap
    ax = axes[1]
    heat_data = np.array([
        [np.mean(by_type_axes[t]["technical"]) if by_type_axes[t]["technical"] else 0,
         np.mean(by_type_axes[t]["pseudo_technical"]) if by_type_axes[t]["pseudo_technical"] else 0,
         np.mean(by_type_axes[t]["political"]) if by_type_axes[t]["political"] else 0]
        for t in types
    ])
    im = ax.imshow(heat_data, aspect="auto", cmap="RdYlGn",
                   vmin=0, vmax=4, origin="upper")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Technical", "Pseudo-Tech", "Political"], fontsize=10)
    ax.set_yticks(range(len(types)))
    ax.set_yticklabels(types, fontsize=9)
    ax.set_title("Mean Axis Scores by Failure Type\n(green=high, red=low)", fontsize=11,
                 fontweight="bold")
    plt.colorbar(im, ax=ax, label="Mean score (0–4)", shrink=0.8)
    for i in range(len(types)):
        for j, val in enumerate(heat_data[i]):
            ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=8,
                    color="black" if 1.5 < val < 3.5 else "white")

    plt.tight_layout()
    out = FIGDIR / "FA3_failure_taxonomy.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Keyword showcase
# ─────────────────────────────────────────────────────────────────────────────

def plot_keyword_showcase(axis_kw):
    """
    Show top keywords from the 3-axis Ridge models.

    These are trained directly on the LLM's per-axis scores (0-4 each), so
    each keyword list answers a clean question:
      Technical    — words the LLM associated with genuine failure explanation
      Political    — words the LLM associated with PR / stakeholder framing
      Pseudo-tech  — words the LLM associated with jargon that sounds technical
                     but doesn't actually explain the failure

    Using 3-axis keywords rather than the 1-axis agg_score keywords avoids
    the noise of a composite target trained on only 171 docs.
    """
    axes_specs = [
        ("technical",       STYLE["technical"],
         "Technical Keywords\n(LLM: genuine failure explanation)",
         "words the LLM scored high on TECHNICAL axis"),
        ("political",       STYLE["political"],
         "Political Keywords\n(LLM: PR / stakeholder framing)",
         "words the LLM scored high on POLITICAL axis"),
        ("pseudo_technical", STYLE["pseudo_technical"],
         "Pseudo-Technical Keywords\n(LLM: jargon without substance)",
         "words the LLM scored high on PSEUDO_TECHNICAL axis"),
    ]

    fig, panel_axes = plt.subplots(1, 3, figsize=(20, 8))
    fig.suptitle(
        "Keyword Drivers — 3-Axis Models  (trained directly on LLM axis scores)\n"
        "Each model answers one question: does this word appear in documents "
        "the LLM rated high on this axis?",
        fontsize=12, fontweight="bold"
    )

    for ax, (axis_key, colour, title, subtitle) in zip(panel_axes, axes_specs):
        high, low = axis_kw[axis_key]
        # Show top 18 high-driving words
        kws   = high[:18]
        words = [w for w, _ in kws]
        coefs = [c for _, c in kws]
        y_pos = range(len(words))

        bars = ax.barh(list(y_pos), coefs, color=colour + "cc",
                       edgecolor="black", linewidth=0.5)
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(words, fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel("Ridge coefficient  (higher = stronger driver)", fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold", color=colour)
        ax.text(0.5, -0.08, subtitle, ha="center", transform=ax.transAxes,
                fontsize=8, color="gray", style="italic")

        max_c = max(coefs) if coefs else 1
        for bar, val in zip(bars, coefs):
            ax.text(val + max_c * 0.015,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=8)

        # Add a small separator and show top 5 low-driving (opposing) words
        if low:
            ax2 = ax.inset_axes([0.55, 0.02, 0.44, 0.28])
            low5  = low[:5]
            w5    = [w for w, _ in low5]
            c5    = [abs(c) for _, c in low5]
            ax2.barh(range(len(w5)), c5, color="#78909C" + "aa",
                     edgecolor="black", linewidth=0.4)
            ax2.set_yticks(range(len(w5)))
            ax2.set_yticklabels(w5, fontsize=7.5)
            ax2.invert_yaxis()
            ax2.set_title("Top opposing", fontsize=7, color="gray")
            ax2.tick_params(axis="x", labelsize=6)

    plt.tight_layout()
    out = FIGDIR / "FA4_keyword_showcase.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 4b. Keyword showcase — new 1-axis model (technical vs political only)
# ─────────────────────────────────────────────────────────────────────────────

MODEL_PARAMS = dict(
    labels      = "3-axis agg_score (0–8)",
    ngram       = "(1, 2)",
    max_features= "None  (no cap)",
    min_df      = 3,
    sublinear_tf= True,
    alpha       = 10.0,
    stopwords   = "sklearn ENGLISH + contraction artifacts only",
    seeds       = "none",
)

def plot_keyword_new_model():
    """
    FA4-style 2×2 grid using the trained 1-axis TF-IDF+Ridge models.
    Rows: combined model (n=1733) / hand-edited model (n=171).
    Cols: Technical (positive coefs) / Political (negative coefs).
    Parameters printed at top.
    """
    def load_kw(path):
        with open(ROOT / path, encoding="utf-8") as f:
            obj = json.load(f)
        tech = [(d["keyword"], d["coefficient"]) for d in obj["technical"]]
        pol  = [(d["keyword"], d["coefficient"]) for d in obj["political"]]
        return tech, pol

    tech_co, pol_co = load_kw("data/model_combined.json")
    tech_he, pol_he = load_kw("data/model_handedited.json")

    N = 20
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    param_str = (
        f"Labels: {MODEL_PARAMS['labels']}  |  "
        f"ngram_range={MODEL_PARAMS['ngram']}  |  "
        f"max_features={MODEL_PARAMS['max_features']}  |  "
        f"min_df={MODEL_PARAMS['min_df']}  |  "
        f"sublinear_tf={MODEL_PARAMS['sublinear_tf']}  |  "
        f"Ridge alpha={MODEL_PARAMS['alpha']}  |  "
        f"seeds={MODEL_PARAMS['seeds']}"
    )
    fig.suptitle(
        "TF-IDF + Ridge Keyword Drivers  —  New 1-Axis Model\n" + param_str,
        fontsize=11, fontweight="bold"
    )

    specs = [
        (0, 0, tech_co, STYLE["technical"],  "TECHNICAL  →  score toward 8",
         "Combined model  (n=1733, hand + fail_news)"),
        (0, 1, pol_co,  STYLE["political"],  "POLITICAL  →  score toward 0",
         "Combined model  (n=1733, hand + fail_news)"),
        (1, 0, tech_he, STYLE["technical"],  "TECHNICAL  →  score toward 8",
         "Hand-edited model  (n=171, engineering postmortems only)"),
        (1, 1, pol_he,  STYLE["political"],  "POLITICAL  →  score toward 0",
         "Hand-edited model  (n=171, engineering postmortems only)"),
    ]

    for row, col, kws, colour, axis_title, model_title in specs:
        ax    = axes[row, col]
        top   = kws[:N]
        words = [w for w, _ in top]
        coefs = [abs(c) for _, c in top]
        y_pos = list(range(len(words)))

        ax.barh(y_pos, coefs, color=colour + "cc",
                edgecolor="black", linewidth=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(words, fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel("Ridge coefficient magnitude", fontsize=9)
        ax.set_title(f"{axis_title}\n{model_title}",
                     fontsize=10, fontweight="bold", color=colour)

        max_c = max(coefs) if coefs else 1.0
        for y, (word, coef_raw) in zip(y_pos, top):
            ax.text(abs(coef_raw) + max_c * 0.015, y,
                    f"{abs(coef_raw):.4f}", va="center", fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = FIGDIR / "FA4b_keyword_new_model.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Exemplar sources
# ─────────────────────────────────────────────────────────────────────────────

def plot_exemplar_sources(agg):
    """Show top/bottom scoring documents per axis with metadata."""
    hand = [r for r in agg if r.get("source") in HAND_SOURCES
            and r.get("agg_score") is not None
            and r.get("technical") is not None]

    def fmt(r, score_label, score_val):
        company = r.get("company", "Unknown")[:22]
        title   = r.get("title", "")[:55]
        return f"{company:22s}  {score_label}={score_val:.2f}  {title}"

    def top_bottom(docs, key, n=5):
        s = sorted(docs, key=lambda x: x.get(key, 0))
        return s[:n], s[-n:]

    fig, axes = plt.subplots(1, 3, figsize=(22, 8))
    fig.suptitle("Exemplar Documents per Axis  (hand-edited postmortems)",
                 fontsize=14, fontweight="bold")

    specs = [
        ("agg_score", "agg", STYLE["agg"],
         "Aggregate Score\n(0=political, 8=engineering utility)"),
        ("technical", "T", STYLE["technical"],
         "Technical Axis\n(0–4, genuine explanation of failure)"),
        ("political", "P", STYLE["political"],
         "Political Axis\n(0–4, PR / stakeholder framing)"),
    ]

    for ax, (key, short, colour, title) in zip(axes, specs):
        bots, tops = top_bottom(hand, key, n=5)

        ax.axis("off")
        y = 0.97
        ax.text(0.0, y, title, fontsize=10, fontweight="bold", color=colour,
                transform=ax.transAxes, va="top")
        y -= 0.06

        ax.text(0.0, y, "── HIGHEST (most " +
                ("engineering utility" if key=="agg_score" else
                 "technical" if key=="technical" else "political") + ") ──",
                fontsize=8, transform=ax.transAxes, va="top", color="gray")
        y -= 0.04

        for r in reversed(tops):
            val  = r.get(key, 0)
            line = fmt(r, short, val)
            ax.text(0.0, y, f"  {val:.2f}  {r.get('company','?')[:20]:20s}  "
                    f"{r.get('title','')[:50]}",
                    fontsize=7.5, transform=ax.transAxes, va="top",
                    color=colour if val > (2 if key=="political" else 4) else "black")
            y -= 0.045

        y -= 0.04
        ax.text(0.0, y, "── LOWEST ──", fontsize=8,
                transform=ax.transAxes, va="top", color="gray")
        y -= 0.04

        for r in bots:
            val  = r.get(key, 0)
            ax.text(0.0, y, f"  {val:.2f}  {r.get('company','?')[:20]:20s}  "
                    f"{r.get('title','')[:50]}",
                    fontsize=7.5, transform=ax.transAxes, va="top", color="#555555")
            y -= 0.045

    plt.tight_layout()
    out = FIGDIR / "FA5_exemplar_sources.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Tech vs Political scatter, coloured by pseudo-technical
# ─────────────────────────────────────────────────────────────────────────────

def plot_axis_profiles(agg):
    hand = [r for r in agg if r.get("source") in HAND_SOURCES
            and r.get("technical") is not None and r.get("political") is not None]
    news = [r for r in agg if r.get("source") == "fail_news"
            and r.get("technical") is not None and r.get("political") is not None]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Technical vs Political Axes  (each dot = one document)",
                 fontsize=13, fontweight="bold")

    for ax, docs, title in [(axes[0], hand, "Hand-edited postmortems"),
                             (axes[1], news[:600], "Fail-news articles (sample n=600)")]:
        pt   = np.array([r.get("pseudo_technical", 0) for r in docs])
        tech = np.array([r.get("technical",        0) for r in docs])
        pol  = np.array([r.get("political",         0) for r in docs])

        sc = ax.scatter(pol, tech, c=pt, cmap="YlOrRd", s=28, alpha=0.65,
                        vmin=0, vmax=4, edgecolors="none")
        plt.colorbar(sc, ax=ax, label="Pseudo-technical score (0–4)")

        ax.set_xlabel("Political score  (0–4)", fontsize=11)
        ax.set_ylabel("Technical score  (0–4)", fontsize=11)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlim(-0.2, 4.5)
        ax.set_ylim(-0.2, 4.5)

        # quadrant labels
        ax.axvline(2, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.axhline(2, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.text(3.4, 3.7, "High T\nHigh P", fontsize=7.5, color="gray", ha="center")
        ax.text(0.6, 3.7, "High T\nLow P",  fontsize=7.5, color="steelblue", ha="center")
        ax.text(3.4, 0.3, "Low T\nHigh P",  fontsize=7.5, color="firebrick", ha="center")
        ax.text(0.6, 0.3, "Low T\nLow P",   fontsize=7.5, color="gray", ha="center")

    plt.tight_layout()
    out = FIGDIR / "FA6_axis_profiles.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Source comparison
# ─────────────────────────────────────────────────────────────────────────────

def plot_source_comparison(agg):
    sources = ["danluu", "aws", "manual", "icco", "fail_news"]
    axes_keys = ["agg_score", "technical", "pseudo_technical", "political"]
    axes_labs = ["Agg score\n(0–8)", "Technical\n(0–4)", "Pseudo-tech\n(0–4)", "Political\n(0–4)"]
    axes_scales = [8, 4, 4, 4]

    by_src = {s: {k: [] for k in axes_keys} for s in sources}
    for r in agg:
        src = r.get("source")
        if src not in by_src: continue
        for k in axes_keys:
            if r.get(k) is not None:
                by_src[src][k].append(r[k])

    x   = np.arange(len(sources))
    w   = 0.18
    fig, axes_ax = plt.subplots(1, 4, figsize=(20, 6))
    fig.suptitle("Score Distribution by Data Source  (all 5 sources)",
                 fontsize=13, fontweight="bold")

    for col_ax, key, label, scale in zip(axes_ax, axes_keys, axes_labs, axes_scales):
        means = [np.mean(by_src[s][key]) if by_src[s][key] else 0 for s in sources]
        stds  = [np.std(by_src[s][key])  if by_src[s][key] else 0 for s in sources]
        ns    = [len(by_src[s][key]) for s in sources]
        colours = [SOURCE_COLOURS[s] for s in sources]

        bars = col_ax.bar(x, means, yerr=stds, capsize=4,
                          color=[c + "cc" for c in colours],
                          edgecolor="black", linewidth=0.7, width=0.55,
                          error_kw=dict(elinewidth=1.2, capthick=1.2))
        col_ax.set_xticks(x)
        col_ax.set_xticklabels(
            [f"{SOURCE_LABELS[s]}\n(n={n})" for s, n in zip(sources, ns)],
            fontsize=8, rotation=20, ha="right"
        )
        col_ax.set_ylabel(f"Mean {label}", fontsize=9)
        col_ax.set_ylim(0, scale * 1.3)
        col_ax.set_title(label, fontsize=10, fontweight="bold")
        for bar, mean, std in zip(bars, means, stds):
            col_ax.text(bar.get_x() + bar.get_width()/2,
                        mean + std + scale * 0.02,
                        f"{mean:.2f}", ha="center", fontsize=8)

    plt.tight_layout()
    out = FIGDIR / "FA7_source_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=== Loading data ===")
    agg, m_cmp, axis_kw = load()
    print(f"  {len(agg)} records loaded")

    hand = [r for r in agg if r.get("source") in HAND_SOURCES]
    news = [r for r in agg if r.get("source") == "fail_news"]
    print(f"  hand-edited: {len(hand)}   fail_news: {len(news)}")

    print("\n=== Summary stats (final V4) ===")
    for subset, label in [(hand, "handedited"), (news, "fail_news")]:
        agg_sc = [r["agg_score"] for r in subset if r.get("agg_score") is not None]
        tech   = [r["technical"]  for r in subset if r.get("technical") is not None]
        pol    = [r["political"]  for r in subset if r.get("political") is not None]
        pt     = [r["pseudo_technical"] for r in subset if r.get("pseudo_technical") is not None]
        print(f"  {label}:")
        print(f"    agg_score: mean={np.mean(agg_sc):.2f}  std={np.std(agg_sc):.2f}")
        print(f"    technical: mean={np.mean(tech):.2f}")
        print(f"    political: mean={np.mean(pol):.2f}")
        print(f"    pseudo_technical: mean={np.mean(pt):.2f}")

    print("\n=== Generating figures ===")
    plot_company_boxplots(agg)
    plot_temporal(agg)
    plot_failure_taxonomy(agg)
    plot_keyword_showcase(axis_kw)
    plot_keyword_new_model()
    plot_exemplar_sources(agg)
    plot_axis_profiles(agg)
    plot_source_comparison(agg)

    print("\nDone. All figures saved to", FIGDIR)


if __name__ == "__main__":
    main()
