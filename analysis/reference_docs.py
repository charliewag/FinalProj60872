"""
Reference Document Analysis: Challenger, Log4J, Toyota vs Hand-Edited Postmortems.
Scores each reference doc using the trained TF-IDF + Ridge model, then compares
to the distribution of hand-edited postmortems across all three axes.
Also overlays scores from the LLM-based scorer (from postmortems_scored.json).
Saves all figures to analysis/figures/.
"""

import json
import re
import statistics
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pickle

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
OUT  = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)

STYLE = {
    "technical":        "#2196F3",
    "political":        "#F44336",
    "pseudo_technical": "#FF9800",
    "neutral":          "#9E9E9E",
    "challenger":       "#673AB7",
    "log4j":            "#009688",
    "toyota":           "#FF5722",
}

# ── load reference docs ────────────────────────────────────────────────────
print("Loading reference documents …")
ref_files = {
    "Challenger\n(NASA 1986)":      ROOT / "challenger.txt",
    "Log4J CSRB\n(DHS 2022)":      ROOT / "csrb_log4j.txt",
    "Toyota Accel\n(NHTSA)":       ROOT / "toyotaaccel.txt",
}

ref_texts = {}
for label, path in ref_files.items():
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        ref_texts[label] = f.read()
    print(f"  {label.split(chr(10))[0]:25s} {len(ref_texts[label]):>8,} chars")


# ── load scored postmortems ────────────────────────────────────────────────
print("Loading scored postmortems …")
with open(DATA / "postmortems_scored.json") as f:
    scored = json.load(f)

HAND = {"danluu", "aws", "manual", "icco"}
hand = [d for d in scored if d.get("source") in HAND]

with open(DATA / "model_handedited.json") as f:
    model_he = json.load(f)


# ── load trained models ────────────────────────────────────────────────────
print("Loading trained Ridge models …")
try:
    with open(DATA / "vectorizer_handedited.pkl", "rb") as f:
        vec = pickle.load(f)
    with open(DATA / "ridge_technical_handedited.pkl", "rb") as f:
        ridge_tech = pickle.load(f)
    with open(DATA / "ridge_political_handedited.pkl", "rb") as f:
        ridge_pol = pickle.load(f)
    with open(DATA / "ridge_pseudo_technical_handedited.pkl", "rb") as f:
        ridge_ps = pickle.load(f)
    MODELS_LOADED = True
    print("  Models loaded.")
except Exception as e:
    print(f"  Could not load models ({e}) — will use keyword scoring fallback.")
    MODELS_LOADED = False


def score_text_models(text):
    """Returns (technical, political, pseudo_technical) scores via Ridge model."""
    X = vec.transform([text])
    return (
        float(np.clip(ridge_tech.predict(X)[0], 0, 4)),
        float(np.clip(ridge_pol.predict(X)[0], 0, 4)),
        float(np.clip(ridge_ps.predict(X)[0], 0, 4)),
    )


def chunk_text(text, chunk_size=4000, overlap=500):
    chunks = []
    step = chunk_size - overlap
    for i in range(0, max(1, len(text) - overlap), step):
        chunks.append(text[i: i + chunk_size])
        if i + chunk_size >= len(text):
            break
    return chunks if chunks else [text]


def score_doc_chunked(text):
    chunks = chunk_text(text)
    scores = [score_text_models(c) for c in chunks]
    t  = max(s[0] for s in scores)
    p  = max(s[1] for s in scores)
    ps = max(s[2] for s in scores)
    return t, p, ps


def savefig(name):
    plt.tight_layout()
    plt.savefig(OUT / name, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {name}")


# ── score reference docs ───────────────────────────────────────────────────
ref_scores = {}
if MODELS_LOADED:
    print("\nScoring reference documents …")
    for label, text in ref_texts.items():
        t, p, ps = score_doc_chunked(text)
        ref_scores[label] = {"technical": t, "political": p, "pseudo_technical": ps}
        print(f"  {label.split(chr(10))[0]:25s}  tech={t:.2f}  pol={p:.2f}  pseudo={ps:.2f}")
else:
    # Fallback: use simple keyword counts normalised to 0-4
    TECH_WORDS = {"configuration","error","failure","code","system","software",
                  "buffer","memory","stack","driver","module","firmware",
                  "protocol","database","network","packet","thread","process"}
    POL_WORDS  = {"safety","concern","regulatory","recommend","stakeholder",
                  "oversight","committee","review","policy","ensure","improve",
                  "commitment","agency","federal","government","report"}
    for label, text in ref_texts.items():
        tokens = set(re.findall(r"[a-z]+", text.lower()))
        t  = min(4, len(tokens & TECH_WORDS) / 3)
        p  = min(4, len(tokens & POL_WORDS)  / 3)
        ps = min(4, (t + p) / 2 * 0.5)
        ref_scores[label] = {"technical": t, "political": p, "pseudo_technical": ps}

# ══════════════════════════════════════════════════════════════════════════
# REF-1.  3-AXIS BAR CHART: reference docs side by side
# ══════════════════════════════════════════════════════════════════════════
print("\n[REF-1] Reference doc 3-axis bar chart …")

ref_labels = list(ref_scores.keys())
axes_info  = [
    ("technical",        STYLE["technical"],        "Technical"),
    ("political",        STYLE["political"],         "Political"),
    ("pseudo_technical", STYLE["pseudo_technical"],  "Pseudo-Technical"),
]
x     = np.arange(len(ref_labels))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 5))
for i, (field, color, label) in enumerate(axes_info):
    vals = [ref_scores[r][field] for r in ref_labels]
    ax.bar(x + (i - 1) * width, vals, width, label=label, color=color, alpha=0.82)

# Overlay hand-edited means as dashed lines
hand_means = {
    "technical":        statistics.mean([d["technical_max"]        for d in hand if d.get("technical_max") is not None]),
    "political":        statistics.mean([d["political_max"]        for d in hand if d.get("political_max") is not None]),
    "pseudo_technical": statistics.mean([d["pseudo_technical_max"] for d in hand if d.get("pseudo_technical_max") is not None]),
}
for field, color, _ in axes_info:
    ax.axhline(hand_means[field], color=color, linestyle="--", linewidth=1.2, alpha=0.7)

ax.set_xticks(x)
ax.set_xticklabels(ref_labels, fontsize=10)
ax.set_ylabel("Score (0–4)", fontsize=11)
ax.set_ylim(0, 4.3)
ax.set_title("Reference Documents: 3-Axis Scores vs Hand-Edited Postmortem Means\n(dashed lines = hand-edited corpus mean)", fontsize=11)
ax.legend(fontsize=9)
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
savefig("REF1_reference_3axis.png")


# ══════════════════════════════════════════════════════════════════════════
# REF-2.  REFERENCE DOCS ON TECHNICAL vs POLITICAL SCATTER
# ══════════════════════════════════════════════════════════════════════════
print("[REF-2] Reference docs on scatter …")

tech_h = [d["technical_max"]  for d in hand if d.get("technical_max") is not None]
pol_h  = [d["political_max"]  for d in hand if d.get("political_max")  is not None]
ps_h   = [d["pseudo_technical_max"] for d in hand if d.get("pseudo_technical_max") is not None]

fig, ax = plt.subplots(figsize=(8, 7))
sc = ax.scatter(tech_h, pol_h, c=ps_h, cmap="YlOrRd", s=45, alpha=0.55,
                edgecolors="grey", linewidths=0.3, zorder=2, label="Hand-edited postmortems")
cbar = plt.colorbar(sc)
cbar.set_label("Pseudo-Technical Score", fontsize=9)

REF_MARKERS = {
    "Challenger\n(NASA 1986)":  ("*", STYLE["challenger"], 250),
    "Log4J CSRB\n(DHS 2022)":  ("D", STYLE["log4j"],      120),
    "Toyota Accel\n(NHTSA)":   ("^", STYLE["toyota"],     140),
}
for label, scores in ref_scores.items():
    marker, color, size = REF_MARKERS[label]
    ax.scatter(scores["technical"], scores["political"],
               marker=marker, color=color, s=size, zorder=5,
               edgecolors="black", linewidths=1.2,
               label=label.replace("\n", " "))

ax.axhline(2, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
ax.axvline(2, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
ax.set_xlabel("Technical Score", fontsize=11)
ax.set_ylabel("Political Score", fontsize=11)
ax.set_xlim(0, 4.1); ax.set_ylim(0, 4.1)
ax.set_title("Reference Documents Positioned in Score Space\nvs Hand-Edited Postmortems", fontsize=11)
ax.legend(fontsize=9, loc="upper left")
savefig("REF2_reference_scatter.png")


# ══════════════════════════════════════════════════════════════════════════
# REF-3.  REFERENCE DOC CHUNK-LEVEL SCORES (technical score along doc length)
# ══════════════════════════════════════════════════════════════════════════
print("[REF-3] Chunk-level score profiles …")

if MODELS_LOADED:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for ax, (label, text) in zip(axes, ref_texts.items()):
        chunks = chunk_text(text, chunk_size=3000, overlap=300)
        chunk_scores = [score_text_models(c) for c in chunks]
        t_scores  = [s[0] for s in chunk_scores]
        p_scores  = [s[1] for s in chunk_scores]
        x_pct = [100 * i / max(1, len(chunks) - 1) for i in range(len(chunks))]
        ax.plot(x_pct, t_scores, "o-", color=STYLE["technical"], linewidth=2, label="Technical")
        ax.plot(x_pct, p_scores, "s--",color=STYLE["political"],  linewidth=2, label="Political")
        ax.axhline(statistics.mean(t_scores), color=STYLE["technical"], linestyle=":", alpha=0.5)
        ax.axhline(statistics.mean(p_scores), color=STYLE["political"],  linestyle=":", alpha=0.5)
        ax.set_xlabel("Document Position (%)", fontsize=9)
        ax.set_title(label.replace("\n", " "), fontsize=10)
        ax.set_ylim(0, 4.2)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        if ax == axes[0]:
            ax.set_ylabel("Score (0–4)", fontsize=10)
            ax.legend(fontsize=9)

    fig.suptitle("Score Profile Along Document — Reference Documents\n(technical vs political per chunk)", fontsize=11)
    savefig("REF3_chunk_profiles.png")


# ══════════════════════════════════════════════════════════════════════════
# REF-4.  KEY TERM OVERLAP: reference docs vs model top keywords
# ══════════════════════════════════════════════════════════════════════════
print("[REF-4] Keyword overlap with model …")

def tokenise(text):
    return Counter(re.findall(r"[a-z']+", text.lower()))

top_tech_kws = [e["keyword"] for e in model_he["technical"]["high"][:40]]
top_pol_kws  = [e["keyword"] for e in model_he["political"]["high"][:40]]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, (label, text) in zip(axes, ref_texts.items()):
    freqs = tokenise(text)
    total_tokens = sum(freqs.values())

    tech_matches = [(kw, freqs.get(kw, 0) / total_tokens * 1000) for kw in top_tech_kws if freqs.get(kw, 0) > 0]
    pol_matches  = [(kw, freqs.get(kw, 0) / total_tokens * 1000) for kw in top_pol_kws  if freqs.get(kw, 0) > 0]
    tech_matches.sort(key=lambda x: -x[1])
    pol_matches.sort(key=lambda x: -x[1])

    top_n = 8
    all_kws = ([(kw, rate, "tech") for kw, rate in tech_matches[:top_n]] +
               [(kw, rate, "pol")  for kw, rate in pol_matches[:top_n]])
    all_kws.sort(key=lambda x: -x[1])

    kw_labels = [f"{kw} ({typ})" for kw, _, typ in all_kws]
    kw_rates  = [rate            for _, rate, _ in all_kws]
    kw_colors = [STYLE["technical"] if t == "tech" else STYLE["political"] for _, _, t in all_kws]

    y = range(len(kw_labels))
    ax.barh(y, kw_rates, color=kw_colors, alpha=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(kw_labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Rate per 1000 tokens", fontsize=9)
    ax.set_title(label.replace("\n", " "), fontsize=10)

handles = [mpatches.Patch(color=STYLE["technical"], label="Technical keyword"),
           mpatches.Patch(color=STYLE["political"],  label="Political keyword")]
fig.legend(handles=handles, fontsize=9, loc="lower center", ncol=2)
fig.suptitle("Model Keyword Frequency in Reference Documents\n(tokens per 1000, Blue=technical, Red=political)", fontsize=11)
savefig("REF4_keyword_overlap.png")


# ══════════════════════════════════════════════════════════════════════════
# REF-5.  REFERENCE DOCS vs DISTRIBUTION (percentile annotation)
# ══════════════════════════════════════════════════════════════════════════
print("[REF-5] Percentile comparison …")

hand_tech_sorted = sorted([d["technical_max"] for d in hand if d.get("technical_max") is not None])
hand_pol_sorted  = sorted([d["political_max"]  for d in hand if d.get("political_max")  is not None])

def percentile(sorted_vals, v):
    below = sum(1 for x in sorted_vals if x <= v)
    return 100 * below / len(sorted_vals)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
bins = np.linspace(0, 4, 17)

for ax, (field, sorted_hand, color, label) in zip(axes, [
    ("technical", hand_tech_sorted, STYLE["technical"], "Technical"),
    ("political", hand_pol_sorted,  STYLE["political"], "Political"),
]):
    ax.hist(sorted_hand, bins=bins, color=color, alpha=0.6, edgecolor="white", label="Hand-edited corpus")
    for doc_label, scores in ref_scores.items():
        v   = scores[field]
        pct = percentile(sorted_hand, v)
        short = doc_label.split("\n")[0]
        marker_color = REF_MARKERS[doc_label][1]
        ax.axvline(v, color=marker_color, linewidth=2, linestyle="-",
                   label=f"{short} ({pct:.0f}th pct)")
    ax.set_xlabel(f"{label} Score", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title(f"Reference Docs vs Hand-Edited Distribution — {label}", fontsize=10)
    ax.legend(fontsize=8)

fig.suptitle("Where Do Reference Documents Fall in the Postmortem Score Distribution?", fontsize=12)
savefig("REF5_percentile_comparison.png")

print("\nReference doc analysis done — figures saved to analysis/figures/")
