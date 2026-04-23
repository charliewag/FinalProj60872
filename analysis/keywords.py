"""
Keyword Analysis for Post-Mortem Scoring Project.
Focuses on:
  - "glitch" and similar evasive words as political tells
  - Model keyword comparison (hand-edited vs combined)
  - Top keyword drivers per axis
  - Keyword frequency: hand-edited vs fail-news
Saves all figures to analysis/figures/.
"""

import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT  = Path(__file__).parent.parent
DATA  = ROOT / "data"
OUT   = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)

STYLE = {
    "technical":        "#2196F3",
    "political":        "#F44336",
    "pseudo_technical": "#FF9800",
    "fail_news":        "#9C27B0",
    "neutral":          "#9E9E9E",
}

# ── load ───────────────────────────────────────────────────────────────────
print("Loading data …")
with open(DATA / "postmortems_scored.json") as f:
    scored = json.load(f)

with open(DATA / "model_handedited.json") as f:
    model_he = json.load(f)

with open(DATA / "model_combined.json") as f:
    model_co = json.load(f)

with open(DATA / "model_comparison.json") as f:
    model_cmp = json.load(f)

HAND = {"danluu", "aws", "manual", "icco"}
hand = [d for d in scored if d.get("source") in HAND]
news = [d for d in scored if d.get("source") == "fail_news"]


def savefig(name):
    path = OUT / name
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {path.name}")


def tokenise(text):
    return re.findall(r"[a-z']+", text.lower())


# ═══════════════════════════════════════════════════════════════════════════
# KW-1.  "GLITCH" AND EVASIVE-WORD FREQUENCY
# ═══════════════════════════════════════════════════════════════════════════
print("\n[KW-1] Glitch / evasive word analysis …")

EVASIVE = [
    "glitch", "issue", "incident", "disruption", "anomaly",
    "hiccup", "irregularity", "outage", "event", "problem",
]

TECHNICAL_SPECIFIC = [
    "bug", "null", "deadlock", "overflow", "exception", "segfault",
    "race condition", "timeout", "heap", "stack", "kernel",
    "certificate", "dns", "bgp", "oom", "panic", "abort",
]

def word_rate(docs, word):
    total_docs = len(docs)
    hits = sum(1 for d in docs if word in d.get("text", "").lower())
    return 100.0 * hits / total_docs if total_docs else 0

# Rates in hand-edited vs fail-news
ev_hand = [word_rate(hand, w) for w in EVASIVE]
ev_news = [word_rate(news, w) for w in EVASIVE]

x = np.arange(len(EVASIVE))
w = 0.35
fig, ax = plt.subplots(figsize=(13, 5))
ax.bar(x - w/2, ev_hand, w, color=STYLE["technical"], alpha=0.8, label="Hand-Edited Postmortems")
ax.bar(x + w/2, ev_news, w, color=STYLE["fail_news"],  alpha=0.75, label="Fail-News Articles")
ax.set_xticks(x)
ax.set_xticklabels(EVASIVE, rotation=25, ha="right", fontsize=10)
ax.set_ylabel("% of Documents Containing Word", fontsize=10)
ax.set_title("Evasive / Non-Specific Word Usage\nHand-Edited Postmortems vs Fail-News Articles", fontsize=11)
ax.legend(fontsize=10)
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
savefig("KW1_evasive_words.png")

# ── within hand-edited: does "glitch" correlate with lower tech score? ────
glitch_docs   = [d for d in hand if "glitch" in d.get("text","").lower()]
noglitch_docs = [d for d in hand if "glitch" not in d.get("text","").lower()]

print(f"  Hand-edited with 'glitch': {len(glitch_docs)}, without: {len(noglitch_docs)}")
if glitch_docs:
    g_tech = [d["technical_max"] for d in glitch_docs if d.get("technical_max") is not None]
    ng_tech= [d["technical_max"] for d in noglitch_docs if d.get("technical_max") is not None]
    print(f"  Mean technical (with glitch): {statistics.mean(g_tech):.2f}")
    print(f"  Mean technical (no glitch):   {statistics.mean(ng_tech):.2f}")


# ═══════════════════════════════════════════════════════════════════════════
# KW-2.  TECHNICAL-SPECIFIC vs EVASIVE WORD RATES — SIDE BY SIDE
# ═══════════════════════════════════════════════════════════════════════════
print("[KW-2] Technical vs evasive word rates …")

ev_hand_h = [word_rate(hand, w) for w in EVASIVE[:8]]
ev_news_h = [word_rate(news, w) for w in EVASIVE[:8]]
ts_hand_h = [word_rate(hand, w) for w in TECHNICAL_SPECIFIC[:8]]
ts_news_h = [word_rate(news, w) for w in TECHNICAL_SPECIFIC[:8]]

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
for ax, (ev, ts, title) in zip(axes, [
    (ev_hand_h, ts_hand_h, "Hand-Edited Postmortems"),
    (ev_news_h, ts_news_h, "Fail-News Articles"),
]):
    x = np.arange(8)
    w = 0.35
    ax.bar(x - w/2, ev, w, color=STYLE["political"], alpha=0.8, label="Evasive words")
    ax.bar(x + w/2, ts, w, color=STYLE["technical"], alpha=0.75, label="Technical-specific words")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{e}/{t}" for e, t in zip(EVASIVE[:8], TECHNICAL_SPECIFIC[:8])],
                       rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("% Documents Containing Word", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)

fig.suptitle("Evasive vs Technical-Specific Word Usage Rates", fontsize=12)
savefig("KW2_evasive_vs_technical_words.png")


# ═══════════════════════════════════════════════════════════════════════════
# KW-3.  MODEL KEYWORD COMPARISON (hand-edited vs combined)
# ═══════════════════════════════════════════════════════════════════════════
print("[KW-3] Model keyword comparison …")

def get_top_words(model_dict, axis, side, n=15):
    return [(e["keyword"], e["coefficient"])
            for e in model_dict[axis].get(side, [])[:n]]

fig, axes = plt.subplots(3, 2, figsize=(14, 14))

for col, (model, model_name) in enumerate([(model_he, "Hand-Edited Model"),
                                            (model_co, "Combined Model")]):
    for row, (axis, color) in enumerate([("technical", STYLE["technical"]),
                                          ("political", STYLE["political"]),
                                          ("pseudo_technical", STYLE["pseudo_technical"])]):
        ax = axes[row][col]
        items = get_top_words(model, axis, "high", 15)
        if not items:
            ax.set_visible(False)
            continue
        kws   = [i[0] for i in items]
        coefs = [i[1] for i in items]
        y     = range(len(kws))
        ax.barh(y, coefs, color=color, alpha=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(kws, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Coefficient", fontsize=8)
        ax.set_title(f"{axis.replace('_',' ').title()} — {model_name}", fontsize=9)

fig.suptitle("Top Keywords by Axis and Model (Hand-Edited vs Combined Dataset)", fontsize=12)
savefig("KW3_model_comparison.png")


# ═══════════════════════════════════════════════════════════════════════════
# KW-4.  FLIPPED KEYWORDS (opposite signal across models)
# ═══════════════════════════════════════════════════════════════════════════
print("[KW-4] Flipped keywords …")

flipped = model_cmp.get("flipped", [])
if flipped:
    fig, ax = plt.subplots(figsize=(10, max(4, len(flipped[:20]) * 0.4)))
    kws  = [e.get("keyword", e.get("word", "")) for e in flipped[:20]]
    he_c = [e.get("handedited_coefficient", e.get("he_coef", 0)) for e in flipped[:20]]
    co_c = [e.get("combined_coefficient",   e.get("co_coef", 0)) for e in flipped[:20]]
    x    = np.arange(len(kws))
    ax.barh(x - 0.2, he_c, 0.35, color=STYLE["technical"], alpha=0.8, label="Hand-Edited")
    ax.barh(x + 0.2, co_c, 0.35, color=STYLE["fail_news"],  alpha=0.8, label="Combined")
    ax.set_yticks(x)
    ax.set_yticklabels(kws, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Ridge Coefficient", fontsize=10)
    ax.set_title("'Flipped' Keywords — Opposite Signals in Hand-Edited vs Combined Models", fontsize=10)
    ax.legend(fontsize=9)
    savefig("KW4_flipped_keywords.png")
else:
    print("  No flipped keyword data found — skipping KW4.")


# ═══════════════════════════════════════════════════════════════════════════
# KW-5.  TOP WORD FREQUENCY IN HAND-EDITED (word cloud proxy — sorted bar)
# ═══════════════════════════════════════════════════════════════════════════
print("[KW-5] Top word frequencies in hand-edited corpus …")

STOP = {
    "the","a","an","and","or","of","to","in","is","was","that","it","for",
    "we","our","this","at","by","with","on","as","from","were","have","been",
    "had","all","are","has","be","not","they","which","their","will","would",
    "but","when","also","more","after","during","about","time","some","can",
    "one","two","three","four","five","new","into","than","then","there",
    "if","so","its","out","up","no","any","each","may","what","who","how",
    "other","these","those","am","do","did","he","she","him","his","her",
    "my","me","us","you","your","been","could","should","over","under","via",
    "s","t","i","re","ve","ll","d","m","n",
}

all_text = " ".join(d.get("text","") for d in hand)
tokens   = tokenise(all_text)
freq     = Counter(t for t in tokens if t not in STOP and len(t) > 2)

top50  = freq.most_common(50)
words  = [w for w, _ in top50]
counts = [c for _, c in top50]

fig, ax = plt.subplots(figsize=(14, 6))
colors = []
tech_kws = {e["keyword"] for e in model_he["technical"]["high"][:30]}
pol_kws  = {e["keyword"] for e in model_he["political"]["high"][:30]}
ps_kws   = {e["keyword"] for e in model_he["pseudo_technical"]["high"][:30]}
for w in words:
    if w in tech_kws:
        colors.append(STYLE["technical"])
    elif w in pol_kws:
        colors.append(STYLE["political"])
    elif w in ps_kws:
        colors.append(STYLE["pseudo_technical"])
    else:
        colors.append(STYLE["neutral"])

ax.bar(range(len(words)), counts, color=colors, alpha=0.85, edgecolor="white")
ax.set_xticks(range(len(words)))
ax.set_xticklabels(words, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Frequency in Corpus", fontsize=10)
ax.set_title("Top 50 Words in Hand-Edited Postmortems\n(Blue=technical signal, Red=political signal, Orange=pseudo-tech, Grey=neutral)",
             fontsize=11)
handles = [
    mpatches.Patch(color=STYLE["technical"],        label="Technical signal"),
    mpatches.Patch(color=STYLE["political"],         label="Political signal"),
    mpatches.Patch(color=STYLE["pseudo_technical"],  label="Pseudo-tech signal"),
    mpatches.Patch(color=STYLE["neutral"],           label="Neutral"),
]
ax.legend(handles=handles, fontsize=9)
savefig("KW5_top_words_handedited.png")


# ═══════════════════════════════════════════════════════════════════════════
# KW-6.  GLITCH SCORE IMPACT — box plot comparison
# ═══════════════════════════════════════════════════════════════════════════
print("[KW-6] Glitch / evasive word -> score impact ...")

EVASIVE_SUBSET = ["glitch", "issue", "incident", "disruption", "anomaly"]

fig, axes = plt.subplots(1, len(EVASIVE_SUBSET), figsize=(16, 5), sharey=True)
for ax, word in zip(axes, EVASIVE_SUBSET):
    with_word    = [d["technical_max"] for d in hand
                    if word in d.get("text","").lower() and d.get("technical_max") is not None]
    without_word = [d["technical_max"] for d in hand
                    if word not in d.get("text","").lower() and d.get("technical_max") is not None]
    bp = ax.boxplot([with_word, without_word], patch_artist=True,
                    medianprops=dict(color="black", linewidth=1.8))
    bp["boxes"][0].set_facecolor(STYLE["political"])
    bp["boxes"][1].set_facecolor(STYLE["technical"])
    bp["boxes"][0].set_alpha(0.65); bp["boxes"][1].set_alpha(0.65)
    ax.set_xticklabels([f"With\n'{word}'\n(n={len(with_word)})",
                        f"Without\n(n={len(without_word)})"], fontsize=8)
    ax.set_ylim(0, 4.2)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_title(f"'{word}'", fontsize=10)

axes[0].set_ylabel("Technical Score (max)", fontsize=10)
fig.suptitle("Impact of Evasive Words on Technical Score — Hand-Edited Postmortems", fontsize=11)
savefig("KW6_evasive_word_impact.png")

print("\nKeyword analysis done — figures saved to analysis/figures/")
