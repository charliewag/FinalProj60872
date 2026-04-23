"""
Post-Mortem Analysis: Political vs Technical Classification
Generates all plots and statistics
Saves all figures to analysis/figures/ and a summary stats CSV.
"""

import json
import os
import statistics
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
OUT  = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)

STYLE = {
    "technical":        "#2196F3",   # blue
    "political":        "#F44336",   # red
    "pseudo_technical": "#FF9800",   # orange
    "neutral":          "#9E9E9E",   # grey
    "fail_news":        "#9C27B0",   # purple
}

# ── load data ──────────────────────────────────────────────────────────────
print("Loading data …")
with open(DATA / "postmortems_scored.json") as f:
    scored = json.load(f)

with open(DATA / "postmortems_scored_chunks_oneaxis.json") as f:
    oneaxis = json.load(f)

with open(DATA / "bertopic_results.json") as f:
    bert = json.load(f)

with open(DATA / "bertopic_topics.json") as f:
    bert_topics = json.load(f)

with open(DATA / "model_handedited.json") as f:
    model_he = json.load(f)

with open(DATA / "model_comparison.json") as f:
    model_cmp = json.load(f)

HAND_SOURCES = {"danluu", "aws", "manual", "icco"}

hand    = [d for d in scored   if d.get("source") in HAND_SOURCES]
news    = [d for d in scored   if d.get("source") == "fail_news"]
hand_oa = [d for d in oneaxis  if d.get("source") in HAND_SOURCES]
bert_hand = [d for d in bert   if d.get("source") in HAND_SOURCES]

print(f"  Hand-edited postmortems : {len(hand)}")
print(f"  Fail-news articles      : {len(news)}")
print(f"  BERTopic hand entries   : {len(bert_hand)}")

# ── helpers ────────────────────────────────────────────────────────────────

def savefig(name, tight=True):
    path = OUT / name
    if tight:
        plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {path.name}")


def year_of(d):
    raw = str(d.get("date", ""))
    m = re.search(r"(19|20)\d{2}", raw)
    return int(m.group()) if m else None


# ═══════════════════════════════════════════════════════════════════════════
# 1. SCORE DISTRIBUTIONS  (3-axis hand-edited)
# ═══════════════════════════════════════════════════════════════════════════
print("\n[1] Score distributions …")

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
bins = np.linspace(0, 4, 17)

for ax, (field, label, color) in zip(axes, [
    ("technical_max",        "Technical",        STYLE["technical"]),
    ("political_max",        "Political",        STYLE["political"]),
    ("pseudo_technical_max", "Pseudo-Technical", STYLE["pseudo_technical"]),
]):
    vals = [d[field] for d in hand if d.get(field) is not None]
    ax.hist(vals, bins=bins, color=color, edgecolor="white", alpha=0.85)
    ax.axvline(statistics.mean(vals), color="black", linestyle="--", linewidth=1.2,
               label=f"mean={statistics.mean(vals):.2f}")
    ax.axvline(statistics.median(vals), color="black", linestyle=":", linewidth=1.2,
               label=f"med={statistics.median(vals):.2f}")
    ax.set_title(f"{label} Score Distribution", fontsize=11)
    ax.set_xlabel("Score (0–4)")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)

fig.suptitle("3-Axis Score Distributions — Hand-Edited Postmortems (n=171)", fontsize=12, y=1.02)
savefig("01_score_distributions.png")


# ═══════════════════════════════════════════════════════════════════════════
# 2. TECHNICAL vs POLITICAL SCATTER (hand-edited, colored by pseudo-tech)
# ═══════════════════════════════════════════════════════════════════════════
print("[2] Technical vs Political scatter …")

fig, ax = plt.subplots(figsize=(8, 7))
tech_vals  = [d["technical_max"]        for d in hand if "technical_max" in d]
pol_vals   = [d["political_max"]        for d in hand if "political_max"  in d]
pseudo_vals= [d["pseudo_technical_max"] for d in hand if "pseudo_technical_max" in d]

sc = ax.scatter(tech_vals, pol_vals, c=pseudo_vals, cmap="YlOrRd",
                s=60, alpha=0.75, edgecolors="grey", linewidths=0.4)
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label("Pseudo-Technical Score", fontsize=9)
ax.axhline(2, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
ax.axvline(2, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)

# Annotate quadrants
for (x, y, txt) in [(0.2, 3.7, "Political\nonly"), (3.5, 3.7, "Both high"),
                    (0.2, 0.3, "Neither"),          (3.5, 0.3, "Technical\nonly")]:
    ax.text(x, y, txt, fontsize=8, color="dimgrey", ha="left")

ax.set_xlabel("Technical Score (max across chunks)", fontsize=11)
ax.set_ylabel("Political Score (max across chunks)", fontsize=11)
ax.set_title("Technical vs Political — Hand-Edited Postmortems\n(colour = pseudo-technical score)", fontsize=11)
ax.set_xlim(0, 4.1); ax.set_ylim(0, 4.1)
savefig("02_tech_vs_pol_scatter.png")


# ═══════════════════════════════════════════════════════════════════════════
# 3. COMPANY-LEVEL BOX PLOTS (top companies, ≥3 postmortems)
# ═══════════════════════════════════════════════════════════════════════════
print("[3] Company-level box plots …")

by_company = defaultdict(list)
for d in hand:
    c = d.get("company", "").strip()
    if c:
        by_company[c].append(d)

# Normalise duplicates
ALIASES = {"Amazon Web Services": "Amazon", "AWS": "Amazon"}
merged = defaultdict(list)
for c, docs in by_company.items():
    merged[ALIASES.get(c, c)].extend(docs)

companies_sorted = sorted(
    [(c, docs) for c, docs in merged.items() if len(docs) >= 3],
    key=lambda x: -statistics.median([d["technical_max"] for d in x[1]
                                       if d.get("technical_max") is not None])
)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, (field, color, label) in zip(axes, [
    ("technical_max", STYLE["technical"], "Technical"),
    ("political_max", STYLE["political"], "Political"),
]):
    data_lists = [[d[field] for d in docs if d.get(field) is not None]
                  for _, docs in companies_sorted]
    labels     = [c for c, _ in companies_sorted]
    bp = ax.boxplot(data_lists, vert=True, patch_artist=True,
                    medianprops=dict(color="black", linewidth=1.8))
    for patch in bp["boxes"]:
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel(f"{label} Score", fontsize=10)
    ax.set_title(f"{label} Score by Company (≥3 postmortems)", fontsize=10)
    ax.set_ylim(0, 4.2)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)

fig.suptitle("Score Distribution by Company — Hand-Edited Postmortems", fontsize=12)
savefig("03_company_boxplots.png")


# ═══════════════════════════════════════════════════════════════════════════
# 4. TEMPORAL TRENDS — mean scores by year
# ═══════════════════════════════════════════════════════════════════════════
print("[4] Temporal trends …")

year_tech = defaultdict(list)
year_pol  = defaultdict(list)
for d in hand:
    yr = year_of(d)
    if yr and 2010 <= yr <= 2025:
        if d.get("technical_max") is not None:
            year_tech[yr].append(d["technical_max"])
        if d.get("political_max") is not None:
            year_pol[yr].append(d["political_max"])

years = sorted(set(year_tech) | set(year_pol))
t_means = [statistics.mean(year_tech[y]) if year_tech[y] else None for y in years]
p_means = [statistics.mean(year_pol[y])  if year_pol[y]  else None for y in years]
counts  = [len(year_tech.get(y, [])) for y in years]

fig, ax1 = plt.subplots(figsize=(11, 5))
ax2 = ax1.twinx()

ax1.plot(years, t_means, "o-", color=STYLE["technical"], linewidth=2, label="Technical mean")
ax1.plot(years, p_means, "s--",color=STYLE["political"],  linewidth=2, label="Political mean")
ax2.bar(years, counts, alpha=0.18, color="grey", label="n postmortems")

ax1.set_xlabel("Year", fontsize=11)
ax1.set_ylabel("Mean Score (0–4)", fontsize=11)
ax2.set_ylabel("Number of Postmortems", fontsize=10, color="grey")
ax1.set_ylim(0, 4)
ax1.set_title("Technical vs Political Score Trends Over Time — Hand-Edited Postmortems", fontsize=11)
ax1.legend(loc="upper left", fontsize=9)
ax2.legend(loc="upper right", fontsize=9)
ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
savefig("04_temporal_trends.png")


# ═══════════════════════════════════════════════════════════════════════════
# 5. FAILURE TYPE HEATMAP
# ═══════════════════════════════════════════════════════════════════════════
print("[5] Failure type heatmap …")

FTYPE_CANONICAL = {
    "config error": "Config Error",
    "configuration error": "Config Error",
    "storage config error": "Config Error",
    "pacemaker config error": "Config Error",
    "dns configuration error": "DNS/Routing",
    "dns resolution error": "DNS/Routing",
    "bgp community error": "DNS/Routing",
    "network routing failure": "DNS/Routing",
    "network outage": "Network Outage",
    "network partition": "Network Outage",
    "power failure": "Power/Infra",
    "infrastructure failure": "Power/Infra",
    "deployment error": "Deployment",
    "deployment failure": "Deployment",
    "certificate expiration": "Certificate",
    "ddos attack": "DDoS/Security",
    "data corruption": "Data Corruption",
    "control plane config": "Config Error",
    "tiered cache error": "Cache/Storage",
    "token overwrite": "Data Corruption",
    "http protocol error": "Protocol Error",
    "software bug": "Software Bug",
    "database failure": "Database",
    "memory leak": "Software Bug",
    "race condition": "Software Bug",
}

ftype_docs = defaultdict(list)
for d in hand:
    ft = d.get("failure_type", "").lower().strip()
    canon = FTYPE_CANONICAL.get(ft, "Other" if ft else "Unknown")
    ftype_docs[canon].append(d)

ftypes_sorted = sorted(ftype_docs.items(), key=lambda x: -len(x[1]))
ft_labels = [f"{ft} (n={len(docs)})" for ft, docs in ftypes_sorted]
t_means_ft = [statistics.mean([d["technical_max"] for d in docs if d.get("technical_max") is not None] or [0])
              for _, docs in ftypes_sorted]
p_means_ft = [statistics.mean([d["political_max"]  for d in docs if d.get("political_max")  is not None] or [0])
              for _, docs in ftypes_sorted]
ps_means_ft = [statistics.mean([d["pseudo_technical_max"] for d in docs if d.get("pseudo_technical_max") is not None] or [0])
               for _, docs in ftypes_sorted]

matrix = np.array([t_means_ft, p_means_ft, ps_means_ft])
row_labels = ["Technical", "Political", "Pseudo-Tech"]

fig, ax = plt.subplots(figsize=(max(10, len(ftypes_sorted)*0.9 + 2), 4))
im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=4)
ax.set_xticks(range(len(ft_labels))); ax.set_xticklabels(ft_labels, rotation=40, ha="right", fontsize=8)
ax.set_yticks(range(3));              ax.set_yticklabels(row_labels, fontsize=10)
plt.colorbar(im, ax=ax, label="Mean Score (0–4)")
ax.set_title("Mean Scores by Failure Type — Hand-Edited Postmortems", fontsize=11)
for i in range(3):
    for j in range(len(ftypes_sorted)):
        ax.text(j, i, f"{matrix[i,j]:.1f}", ha="center", va="center", fontsize=7, color="black")
savefig("05_failure_type_heatmap.png")


# ═══════════════════════════════════════════════════════════════════════════
# 6. HAND-EDITED vs FAIL-NEWS SCORE COMPARISON
# ═══════════════════════════════════════════════════════════════════════════
print("[6] Hand-edited vs fail-news comparison …")

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
bins = np.linspace(0, 4, 17)

for ax, field, title in zip(axes,
    ["technical_max", "political_max", "pseudo_technical_max"],
    ["Technical", "Political", "Pseudo-Technical"]):
    h_vals = [d[field] for d in hand if d.get(field) is not None]
    n_vals = [d[field] for d in news if d.get(field) is not None]
    ax.hist(h_vals, bins=bins, alpha=0.65, color=STYLE["technical"],  label=f"Hand-edited (n={len(h_vals)})")
    ax.hist(n_vals, bins=bins, alpha=0.55, color=STYLE["fail_news"],  label=f"Fail-news (n={len(n_vals)})")
    ax.set_title(f"{title} Score", fontsize=10)
    ax.set_xlabel("Score (0–4)")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)

fig.suptitle("Score Distributions: Hand-Edited Postmortems vs Fail-News Articles", fontsize=12, y=1.02)
savefig("06_handedited_vs_failnews.png")


# ═══════════════════════════════════════════════════════════════════════════
# 7. ONE-AXIS SCORE DISTRIBUTION + VIOLIN BY SOURCE
# ═══════════════════════════════════════════════════════════════════════════
print("[7] One-axis distribution and violin …")

hand_oa_scores = [d["score"] for d in hand_oa if d.get("score") is not None]
news_oa_scores = [d["score"] for d in oneaxis if d.get("source") == "fail_news" and d.get("score") is not None]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: overlapping histograms
bins_oa = np.linspace(0, 8, 25)
axes[0].hist(hand_oa_scores, bins=bins_oa, alpha=0.7, color=STYLE["technical"], label=f"Hand-edited (n={len(hand_oa_scores)})")
axes[0].hist(news_oa_scores, bins=bins_oa, alpha=0.55, color=STYLE["fail_news"],  label=f"Fail-news (n={len(news_oa_scores)})")
axes[0].axvline(4, color="black", linestyle="--", linewidth=1, label="midpoint (4)")
axes[0].set_xlabel("Score (0=Political → 8=Technical)", fontsize=10)
axes[0].set_ylabel("Count")
axes[0].set_title("One-Axis Score Distribution", fontsize=11)
axes[0].legend(fontsize=9)

# Right: violin
parts = axes[1].violinplot([hand_oa_scores, news_oa_scores], positions=[1, 2],
                            showmedians=True, showextrema=True)
for pc, color in zip(parts["bodies"], [STYLE["technical"], STYLE["fail_news"]]):
    pc.set_facecolor(color); pc.set_alpha(0.7)
axes[1].set_xticks([1, 2])
axes[1].set_xticklabels(["Hand-Edited\nPostmortems", "Fail-News\nArticles"], fontsize=10)
axes[1].set_ylabel("Score (0–8)", fontsize=10)
axes[1].set_title("One-Axis Score Violin by Source", fontsize=11)
axes[1].axhline(4, color="grey", linestyle="--", linewidth=0.8)
axes[1].set_ylim(0, 8.5)

fig.suptitle("One-Axis (0=Political / 8=Technical) Scoring Comparison", fontsize=12)
savefig("07_oneaxis_violin.png")


# ═══════════════════════════════════════════════════════════════════════════
# 8. DOCUMENT LENGTH vs TECHNICAL SCORE
# ═══════════════════════════════════════════════════════════════════════════
print("[8] Length vs technical score …")

lengths = [len(d.get("text", "")) for d in hand]
tech_s  = [d.get("technical_max", 0) for d in hand]
pol_s   = [d.get("political_max",  0) for d in hand]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, (scores, color, label) in zip(axes, [
    (tech_s, STYLE["technical"], "Technical"),
    (pol_s,  STYLE["political"], "Political"),
]):
    ax.scatter(lengths, scores, alpha=0.55, color=color, edgecolors="grey", linewidths=0.4, s=50)
    # regression line
    z = np.polyfit(lengths, scores, 1)
    p = np.poly1d(z)
    xs = np.linspace(min(lengths), max(lengths), 200)
    ax.plot(xs, p(xs), "k--", linewidth=1.2, label=f"slope={z[0]*1000:.2f}e-3")
    corr = np.corrcoef(lengths, scores)[0, 1]
    ax.set_title(f"{label} Score vs Document Length\n(r={corr:.2f})", fontsize=10)
    ax.set_xlabel("Document Length (characters)", fontsize=10)
    ax.set_ylabel(f"{label} Score", fontsize=10)
    ax.legend(fontsize=9)

fig.suptitle("Document Length vs Score — Hand-Edited Postmortems", fontsize=12)
savefig("08_length_vs_score.png")


# ═══════════════════════════════════════════════════════════════════════════
# 9. TOP TECHNICAL & POLITICAL KEYWORDS (model coefficients)
# ═══════════════════════════════════════════════════════════════════════════
print("[9] Keyword coefficient chart …")

def top_keywords(axis_dict, n=20):
    high = [(e["keyword"], e["coefficient"]) for e in axis_dict.get("high", [])[:n]]
    low  = [(e["keyword"], -e["coefficient"]) for e in axis_dict.get("low",  [])[:n]]
    return high, low

fig, axes = plt.subplots(1, 3, figsize=(17, 6))

for ax, (axis_name, color) in zip(axes, [
    ("technical", STYLE["technical"]),
    ("political", STYLE["political"]),
    ("pseudo_technical", STYLE["pseudo_technical"]),
]):
    axis_data = model_he[axis_name]
    high_items = axis_data.get("high", [])[:18]
    keywords   = [e["keyword"]     for e in high_items]
    coefs      = [e["coefficient"] for e in high_items]
    y_pos = range(len(keywords))
    ax.barh(y_pos, coefs, color=color, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(keywords, fontsize=9)
    ax.set_xlabel("Ridge Coefficient", fontsize=9)
    ax.set_title(f"Top '{axis_name.replace('_',' ').title()}'\nKeywords (hand-edited model)", fontsize=10)
    ax.invert_yaxis()

fig.suptitle("Keyword Coefficients from TF-IDF + Ridge Model — Hand-Edited Postmortems", fontsize=12)
savefig("09_keyword_coefficients.png")


# ═══════════════════════════════════════════════════════════════════════════
# 10. SCORE UNCERTAINTY (std dev across LLM runs)
# ═══════════════════════════════════════════════════════════════════════════
print("[10] Score uncertainty …")

stds  = [d.get("technical_std") for d in hand if d.get("technical_std") is not None]
techs = [d.get("technical_max") for d in hand if d.get("technical_std") is not None]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].hist(stds, bins=20, color="#607D8B", edgecolor="white", alpha=0.85)
axes[0].set_xlabel("Std Dev of Technical Score Across LLM Runs", fontsize=10)
axes[0].set_ylabel("Count")
axes[0].set_title("LLM Scoring Uncertainty — Standard Deviation", fontsize=11)
axes[0].axvline(statistics.mean(stds), color="black", linestyle="--",
                label=f"mean={statistics.mean(stds):.2f}")
axes[0].legend()

axes[1].scatter(techs, stds, alpha=0.55, color="#607D8B", s=50, edgecolors="grey", linewidths=0.4)
axes[1].set_xlabel("Technical Score (max)", fontsize=10)
axes[1].set_ylabel("Std Dev Across Runs", fontsize=10)
axes[1].set_title("Technical Score vs LLM Uncertainty", fontsize=11)

fig.suptitle("LLM Scoring Reliability — Variation Across Multiple Runs", fontsize=12)
savefig("10_score_uncertainty.png")


# ═══════════════════════════════════════════════════════════════════════════
# 11. COMPANY TYPE GROUPING (cloud / social / gaming / other)
# ═══════════════════════════════════════════════════════════════════════════
print("[11] Company type grouping …")

COMPANY_TYPES = {
    "Google": "Cloud/Infra", "Amazon": "Cloud/Infra", "Amazon Web Services": "Cloud/Infra",
    "AWS": "Cloud/Infra", "Cloudflare": "Cloud/Infra", "Fastly": "Cloud/Infra",
    "Heroku": "Cloud/Infra", "TravisCI": "Cloud/Infra", "CircleCI": "Cloud/Infra",
    "GitHub": "Cloud/Infra", "Joyent": "Cloud/Infra",
    "Facebook": "Social/Comms", "Slack": "Social/Comms", "Discord": "Social/Comms",
    "Reddit": "Social/Comms", "Stack Exchange": "Social/Comms", "Stack Overflow": "Social/Comms",
    "CCP Games": "Gaming", "Roblox": "Gaming",
    "Honeycomb": "Observability/Dev", "Basecamp": "Observability/Dev",
    "GoCardless": "Fintech", "Instapaper": "Media/Other",
    "Firefox": "Browser/OS",
}

type_docs = defaultdict(list)
for d in hand:
    c    = d.get("company", "").strip()
    c    = ALIASES.get(c, c)
    ctype = COMPANY_TYPES.get(c, "Other")
    type_docs[ctype].append(d)

types_sorted = sorted(type_docs.items(), key=lambda x: -len(x[1]))
type_labels  = [f"{t}\n(n={len(docs)})" for t, docs in types_sorted]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, (field, color, label) in zip(axes, [
    ("technical_max", STYLE["technical"], "Technical"),
    ("political_max", STYLE["political"], "Political"),
]):
    data_lists = [[d[field] for d in docs if d.get(field) is not None]
                  for _, docs in types_sorted]
    bp = ax.boxplot(data_lists, patch_artist=True,
                    medianprops=dict(color="black", linewidth=1.8))
    for patch in bp["boxes"]:
        patch.set_facecolor(color); patch.set_alpha(0.55)
    ax.set_xticklabels(type_labels, fontsize=9)
    ax.set_ylabel(f"{label} Score", fontsize=10)
    ax.set_title(f"{label} Score by Company Type", fontsize=10)
    ax.set_ylim(0, 4.2)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)

fig.suptitle("Scores by Company Type — Hand-Edited Postmortems", fontsize=12)
savefig("11_company_type_boxplots.png")


# ═══════════════════════════════════════════════════════════════════════════
# 12. FAILURE MODE TAXONOMY BAR CHART (counts + mean technical)
# ═══════════════════════════════════════════════════════════════════════════
print("[12] Failure taxonomy bar chart …")

# Use the canonical types from plot 5
canon_counts = {ft: len(docs) for ft, docs in ftype_docs.items()}
canon_tech   = {ft: statistics.mean([d["technical_max"] for d in docs if d.get("technical_max") is not None] or [0])
                for ft, docs in ftype_docs.items()}
canon_pol    = {ft: statistics.mean([d["political_max"] for d in docs if d.get("political_max")  is not None] or [0])
                for ft, docs in ftype_docs.items()}

sorted_ft = sorted(canon_counts.items(), key=lambda x: -x[1])
fnames  = [x[0] for x in sorted_ft]
fcounts = [x[1] for x in sorted_ft]
ftech   = [canon_tech[x[0]] for x in sorted_ft]
fpol    = [canon_pol[x[0]] for x in sorted_ft]

x = np.arange(len(fnames))
width = 0.28

fig, ax1 = plt.subplots(figsize=(13, 5))
ax2 = ax1.twinx()

ax1.bar(x - width, ftech,   width, color=STYLE["technical"], alpha=0.75, label="Mean Technical")
ax1.bar(x,         fpol,    width, color=STYLE["political"],  alpha=0.75, label="Mean Political")
ax2.bar(x + width, fcounts, width, color="grey",              alpha=0.45, label="Count")

ax1.set_xticks(x)
ax1.set_xticklabels(fnames, rotation=35, ha="right", fontsize=9)
ax1.set_ylabel("Mean Score (0–4)", fontsize=10)
ax2.set_ylabel("Number of Postmortems", fontsize=10, color="grey")
ax1.set_ylim(0, 4.5)
ax1.set_title("Failure Mode Taxonomy — Count and Mean Scores", fontsize=11)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper right")

savefig("12_failure_taxonomy.png")


# ═══════════════════════════════════════════════════════════════════════════
# 13. BERTOPIC TOPIC DISTRIBUTION (mean score per topic)
# ═══════════════════════════════════════════════════════════════════════════
print("[13] BERTopic topic analysis …")

# Map topic_id -> mean score and count from full dataset
topic_data = defaultdict(list)
for d in bert:
    if d.get("topic_id") is not None and d.get("score") is not None:
        topic_data[d["topic_id"]].append(d["score"])

# Merge topic metadata
topic_meta = {t["topic_id"]: t for t in bert_topics}

# Build rows (skip outlier -1 if it's huge)
rows = []
for tid, scores in topic_data.items():
    if tid == -1:
        continue
    meta = topic_meta.get(tid, {})
    rows.append({
        "topic_id": tid,
        "label": meta.get("label", f"Topic {tid}"),
        "keywords": meta.get("keywords", [])[:3],
        "count": len(scores),
        "mean_score": statistics.mean(scores),
    })
rows.sort(key=lambda x: -x["mean_score"])

t_labels = [f"T{r['topic_id']}: {', '.join(r['keywords'][:2])} (n={r['count']})"
            for r in rows]
t_scores = [r["mean_score"] for r in rows]
t_colors = [STYLE["technical"] if s > 4 else STYLE["political"] for s in t_scores]

fig, ax = plt.subplots(figsize=(max(8, len(rows)*0.55 + 2), 5))
bars = ax.bar(range(len(rows)), t_scores, color=t_colors, alpha=0.78, edgecolor="white")
ax.axhline(4, color="black", linestyle="--", linewidth=0.9, label="midpoint (4)")
ax.set_xticks(range(len(rows)))
ax.set_xticklabels(t_labels, rotation=40, ha="right", fontsize=7.5)
ax.set_ylabel("Mean One-Axis Score (0=Political / 8=Technical)", fontsize=9)
ax.set_title("BERTopic Cluster Analysis — Mean Score per Topic\n(Blue=Technical, Red=Political)", fontsize=11)
ax.set_ylim(0, 8)
ax.legend(fontsize=9)
savefig("13_bertopic_topics.png")


# ═══════════════════════════════════════════════════════════════════════════
# 14. SCORE BREAKDOWN BY SOURCE (stacked bar)
# ═══════════════════════════════════════════════════════════════════════════
print("[14] Score by data source …")

sources_all = ["danluu", "aws", "manual", "icco", "fail_news"]
source_labels = {"danluu": "Dan Luu\n(n=135)", "aws": "AWS Blog\n(n=4)",
                 "manual": "Manual\n(n=26)", "icco": "ICCO\n(n=6)",
                 "fail_news": "Fail-News\n(n=1562)"}

src_docs = {s: [d for d in scored if d.get("source") == s] for s in sources_all}

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, (field, label, color) in zip(axes, [
    ("technical_max",        "Technical",        STYLE["technical"]),
    ("political_max",        "Political",        STYLE["political"]),
    ("pseudo_technical_max", "Pseudo-Technical", STYLE["pseudo_technical"]),
]):
    means = []
    stds  = []
    counts = []
    for s in sources_all:
        vals = [d[field] for d in src_docs[s] if d.get(field) is not None]
        means.append(statistics.mean(vals) if vals else 0)
        stds.append(statistics.stdev(vals) if len(vals) > 1 else 0)
        counts.append(len(vals))
    x = range(len(sources_all))
    ax.bar(x, means, color=color, alpha=0.75, yerr=stds, capsize=4,
           error_kw={"elinewidth": 1.2})
    ax.set_xticks(x)
    ax.set_xticklabels([source_labels[s] for s in sources_all], fontsize=8)
    ax.set_ylabel(f"Mean {label} Score", fontsize=9)
    ax.set_title(f"{label} by Source", fontsize=10)
    ax.set_ylim(0, 4.5)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)

fig.suptitle("Mean Scores by Data Source (±1 std dev)", fontsize=12)
savefig("14_score_by_source.png")


# ═══════════════════════════════════════════════════════════════════════════
# 15. CLASSIFICATION QUADRANT — two-class summary
# ═══════════════════════════════════════════════════════════════════════════
print("[15] Quadrant classification pie/bar …")

THRESHOLD = 2.0   # ≥ threshold = "high"

quad_counts = Counter()
for d in hand:
    t = d.get("technical_max", 0)
    p = d.get("political_max",  0)
    th = t >= THRESHOLD; ph = p >= THRESHOLD
    if th and not ph:
        quad_counts["Primarily Technical"] += 1
    elif ph and not th:
        quad_counts["Primarily Political"] += 1
    elif th and ph:
        quad_counts["Mixed (Both High)"] += 1
    else:
        quad_counts["Low Both"] += 1

QUAD_COLORS = {
    "Primarily Technical": STYLE["technical"],
    "Primarily Political": STYLE["political"],
    "Mixed (Both High)":   "#FF9800",
    "Low Both":            STYLE["neutral"],
}

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
labels_q = list(quad_counts.keys())
values_q = [quad_counts[l] for l in labels_q]
colors_q = [QUAD_COLORS[l] for l in labels_q]

axes[0].pie(values_q, labels=labels_q, colors=colors_q, autopct="%1.0f%%",
            startangle=90, textprops={"fontsize": 10})
axes[0].set_title(f"Document Classification (threshold={THRESHOLD})", fontsize=11)

axes[1].barh(labels_q, values_q, color=colors_q, alpha=0.8, edgecolor="white")
axes[1].set_xlabel("Number of Postmortems", fontsize=10)
axes[1].set_title("Classification Count", fontsize=11)
for i, v in enumerate(values_q):
    axes[1].text(v + 0.3, i, str(v), va="center", fontsize=10)

fig.suptitle("Quadrant Classification of Hand-Edited Postmortems\n(Technical/Political threshold ≥ 2.0)", fontsize=12)
savefig("15_quadrant_classification.png")


# ═══════════════════════════════════════════════════════════════════════════
# 16. SUMMARY STATISTICS TABLE (save as CSV)
# ═══════════════════════════════════════════════════════════════════════════
print("[16] Summary statistics CSV …")

import csv

rows_csv = []
for d in hand:
    rows_csv.append({
        "id":                   d.get("id", ""),
        "company":              d.get("company", ""),
        "title":                d.get("title", "")[:80],
        "date":                 d.get("date", ""),
        "source":               d.get("source", ""),
        "failure_type":         d.get("failure_type", ""),
        "technical_max":        d.get("technical_max", ""),
        "political_max":        d.get("political_max", ""),
        "pseudo_technical_max": d.get("pseudo_technical_max", ""),
        "n_chunks":             d.get("n_chunks", ""),
        "technical_std":        d.get("technical_std", ""),
        "text_length":          len(d.get("text", "")),
    })

csv_path = Path(__file__).parent / "summary_stats.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows_csv[0].keys()))
    writer.writeheader()
    writer.writerows(rows_csv)
print(f"  saved {csv_path.name}")

print("\nAll done — figures saved to analysis/figures/")
