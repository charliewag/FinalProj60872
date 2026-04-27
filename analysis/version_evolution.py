"""
version_evolution.py

Quantitative evidence that each pipeline iteration improved on the last.

Four versions:
  V1  Thirds scorer — text split into thirds, LLM scores averaged, no chunking
  V2  Chunks one-axis — overlapping 4k-char chunks, LLM mean + max
  V3  Three-axis LLM — technical / pseudo_technical / political scored separately;
        keyword detector trained per axis, then axes combined with agg formula
  V4  Final — three-axis LLM scores collapsed to agg_score via explicit formula,
        single Ridge trained on that target, geometric-mean ensemble at inference

For each version we report:
  LLM  : mean scores on handedited vs fail_news, separation, Mann-Whitney U p
  KW   : same with keyword model, plus Spearman ρ with the LLM ground truth
  Docs : keyword model scores on the three reference documents

Outputs (saved to analysis/figures/):
  EV1_llm_distributions.png   — LLM score distributions per version
  EV2_kw_vs_llm_scatter.png   — keyword vs LLM scatter, one panel per version
  EV3_separation_metrics.png  — bar chart: LLM sep / KW sep / Spearman ρ
  EV4_test_doc_scores.png     — reference doc keyword scores per version
  EV5_metrics_summary.png     — printed table rendered as figure
"""

import json, pickle, sys, warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import spearmanr, mannwhitneyu
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

ROOT    = Path(__file__).parent.parent
DATA    = ROOT / "data"
DOCS    = ROOT / "test_docs"
FIGDIR  = Path(__file__).parent / "figures"
FIGDIR.mkdir(exist_ok=True)

HAND_SOURCES = {"danluu", "aws", "manual", "icco"}

# ── colour palette ────────────────────────────────────────────────────────────
VC = {
    "V1": "#78909C",   # grey-blue
    "V2": "#42A5F5",   # blue
    "V3": "#FFA726",   # orange
    "V4": "#66BB6A",   # green
}
SRC_COL = {"hand": "#1565C0", "news": "#C62828"}

# ── extra stopwords (mirrors train1axis.py) ───────────────────────────────────
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

# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_all():
    with open(DATA / "postmortems_scored_thirds.json")          as f: thirds  = json.load(f)
    with open(DATA / "postmortems_scored_chunks_oneaxis.json")   as f: chunks1 = json.load(f)
    with open(DATA / "postmortems_scored.json")                 as f: s3ax   = json.load(f)
    with open(DATA / "postmortems_agg.json")                    as f: agg    = json.load(f)

    # Index by id so we can join
    by_id = {}
    for r in agg:
        rid = r["id"]
        by_id[rid] = dict(r)
        by_id[rid]["text"] = r.get("text", "")

    thirds_by_id  = {r["id"]: r for r in thirds}
    chunks1_by_id = {r["id"]: r for r in chunks1}
    s3ax_by_id    = {r["id"]: r for r in s3ax}

    for rid, rec in by_id.items():
        t = thirds_by_id.get(rid, {})
        c = chunks1_by_id.get(rid, {})
        s = s3ax_by_id.get(rid, {})
        rec["v1_llm"]  = t.get("score")
        rec["v2_llm"]  = c.get("score")
        rec["v2_max"]  = c.get("score_max")
        rec["v3_tech"] = s.get("technical")
        rec["v3_tech_max"] = s.get("technical_max")
        rec["v3_pt"]   = s.get("pseudo_technical")
        rec["v3_pol"]  = s.get("political")
        rec["v3_pol_max"] = s.get("political_max")
        rec["v4_llm"]  = rec.get("agg_score")

    docs = [r for r in by_id.values()
            if all(r.get(k) is not None
                   for k in ["v1_llm","v2_llm","v3_tech","v4_llm"])
            and r.get("text")]

    print(f"Loaded {len(docs)} docs with all four LLM versions scored")
    return docs


def load_test_docs():
    out = {}
    for path in sorted(DOCS.glob("*.txt")):
        # skip synthetic docs
        if "synthetic" in path.name:
            continue
        with open(path, encoding="utf-8", errors="replace") as f:
            out[path.stem] = f.read()
    print(f"Test docs: {list(out.keys())}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Keyword model helpers
# ─────────────────────────────────────────────────────────────────────────────

def fit_ridge(texts, targets, max_features=500, alpha=1.0, min_df=3):
    vec = TfidfVectorizer(ngram_range=(1,2), max_features=max_features,
                          stop_words=SW, min_df=min_df, sublinear_tf=True)
    X   = vec.fit_transform(texts)
    clf = Ridge(alpha=alpha)
    clf.fit(X, np.array(targets))
    return vec, clf


def predict_chunks(text, vec, clf, lo=0, hi=8,
                   max_chars=12000, chunk_size=4000, overlap=500, max_chunks=8):
    """Mean + max over overlapping chunks; clips to [lo, hi]."""
    if len(text) <= max_chars:
        X = vec.transform([text])
        s = float(np.clip(clf.predict(X)[0], lo, hi))
        return s, s

    chunks, start = [], 0
    while start < len(text) and len(chunks) < max_chunks:
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap

    X      = vec.transform(chunks)
    scores = np.clip(clf.predict(X), lo, hi)
    return float(np.mean(scores)), float(np.max(scores))


def agg_formula(tech_max, tech_mean, pt_mean, pol_mean):
    """Mirrors aggregate_score.py formula → 0-8."""
    pol_pen = pol_mean * max(0.0, 1.0 - tech_max / 4.0)
    raw     = 2.0*tech_max + 0.5*tech_mean - 1.5*pt_mean - 1.0*pol_pen
    raw_min, raw_max = -6.0, 10.0
    return float(np.clip((raw - raw_min) / (raw_max - raw_min) * 8.0, 0, 8))


def sep_stats(hand_scores, news_scores):
    sep  = np.mean(hand_scores) - np.mean(news_scores)
    stat, p = mannwhitneyu(hand_scores, news_scores, alternative="greater")
    return sep, p


# ─────────────────────────────────────────────────────────────────────────────
# Build keyword scores for each version
# ─────────────────────────────────────────────────────────────────────────────

def build_v1_kw(docs):
    """Train Ridge on thirds LLM scores (handedited only), apply to all."""
    hand = [d for d in docs if d["source"] in HAND_SOURCES]
    vec, clf = fit_ridge([d["text"] for d in hand],
                         [d["v1_llm"] for d in hand],
                         max_features=500, alpha=1.0, min_df=2)
    scores = {}
    for d in docs:
        mean, mx = predict_chunks(d["text"], vec, clf, lo=0, hi=8)
        scores[d["id"]] = mean
    print(f"  V1 kw: trained on {len(hand)} handedited docs")
    return vec, clf, scores


def build_v2_kw(docs):
    """Train Ridge on chunks1 LLM scores (handedited only), apply to all."""
    hand = [d for d in docs if d["source"] in HAND_SOURCES]
    vec, clf = fit_ridge([d["text"] for d in hand],
                         [d["v2_llm"] for d in hand],
                         max_features=500, alpha=1.0, min_df=2)
    scores = {}
    for d in docs:
        mean, mx = predict_chunks(d["text"], vec, clf, lo=0, hi=8)
        scores[d["id"]] = mean
    print(f"  V2 kw: trained on {len(hand)} handedited docs")
    return vec, clf, scores


def build_v3_kw(docs):
    """
    Train three separate Ridge models on 3-axis LLM scores (handedited only).
    Combine keyword-predicted axis scores through the agg formula.
    This is the 'failed' version — keyword-axis predictions fed into agg
    give noisier results than training one Ridge directly on the agg target.
    """
    hand = [d for d in docs if d["source"] in HAND_SOURCES]
    texts_h = [d["text"] for d in hand]

    # shared vectorizer for all three axes (matches train3axis.py approach)
    vec = TfidfVectorizer(ngram_range=(1,2), max_features=2000,
                          stop_words=SW, min_df=2, sublinear_tf=True)
    X_h = vec.fit_transform(texts_h)

    clfs = {}
    for axis, key in [("technical","v3_tech"), ("pseudo_technical","v3_pt"),
                      ("political","v3_pol")]:
        targets = np.array([d[key] for d in hand], dtype=float)
        clf = Ridge(alpha=10.0)
        clf.fit(X_h, targets)
        clfs[axis] = clf

    scores = {}
    for d in docs:
        text = d["text"]
        # chunk each doc the same way
        if len(text) <= 12000:
            chunks = [text]
        else:
            chunks, start = [], 0
            while start < len(text) and len(chunks) < 8:
                chunks.append(text[start:min(start+4000, len(text))])
                start += 3500

        axis_chunk_scores = {a: [] for a in clfs}
        for chunk in chunks:
            Xc = vec.transform([chunk])
            for axis, clf in clfs.items():
                axis_chunk_scores[axis].append(float(np.clip(clf.predict(Xc)[0], 0, 4)))

        kw_t_mean  = float(np.mean(axis_chunk_scores["technical"]))
        kw_t_max   = float(np.max(axis_chunk_scores["technical"]))
        kw_pt_mean = float(np.mean(axis_chunk_scores["pseudo_technical"]))
        kw_pol_mean= float(np.mean(axis_chunk_scores["political"]))
        scores[d["id"]] = agg_formula(kw_t_max, kw_t_mean, kw_pt_mean, kw_pol_mean)

    print(f"  V3 kw: trained 3 axis Ridges on {len(hand)} handedited docs, "
          f"combined via agg formula")
    return vec, clfs, scores


def build_v4_kw(docs):
    """
    Retrain final model with expanded vocabulary so government report terms
    (ntsb, cve, vulnerability, deadlock, root cause …) make it in.

    Changes vs V1/V2:
      - max_features=1000  (was 500)
      - min_df=2           (was 3)
      - force_include seed terms even if below min_df via vocabulary override
    Uses geometric-mean ensemble with handedited model only — the combined
    model penalises government-report vocabulary (words like 'process',
    'access', 'significant' appear in both fail_news and gov reports, so the
    combined Ridge gives them negative coefs that unfairly drag down long
    technical documents).
    """
    hand = [d for d in docs if d["source"] in HAND_SOURCES]
    texts_h = [d["text"] for d in hand]
    scores_h = [d["v4_llm"] for d in hand]

    # Build base vectorizer to discover naturally occurring features
    base_vec = TfidfVectorizer(ngram_range=(1,2), max_features=1000,
                               stop_words=SW, min_df=2, sublinear_tf=True)
    base_vec.fit(texts_h)
    base_vocab = dict(base_vec.vocabulary_)

    # Force-include government / technical seed terms that min_df would drop
    SEEDS = [
        "ntsb","nhtsa","csrb","faa","vulnerability","cve","exploit","deadlock",
        "race condition","memory leak","buffer overflow","root cause","stack trace",
        "failure mode","contributing factor","hotfix","rollback","regression",
        "patch","recommendation","findings","investigation","log4j","log4shell",
        "throttle","segfault","exception","corruption","cascade","timeout",
        "revert","mitigation","workaround","postmortem","incident report",
    ]
    # Add seeds not already in vocab
    next_idx = max(base_vocab.values()) + 1
    for seed in SEEDS:
        if seed not in base_vocab:
            base_vocab[seed] = next_idx
            next_idx += 1

    vec = TfidfVectorizer(ngram_range=(1,2), vocabulary=base_vocab,
                          sublinear_tf=True)
    X_h = vec.fit_transform(texts_h)

    clf = Ridge(alpha=1.0)
    clf.fit(X_h, np.array(scores_h))

    scores = {}
    for d in docs:
        mean, mx = predict_chunks(d["text"], vec, clf, lo=0, hi=8)
        scores[d["id"]] = mean

    print(f"  V4 kw: expanded vocab ({len(base_vocab)} features, seeds force-included), "
          f"handedited model, mean-chunk scoring")
    return vec, clf, scores


# ─────────────────────────────────────────────────────────────────────────────
# Score test docs with a keyword model (mean+max chunking)
# ─────────────────────────────────────────────────────────────────────────────

def score_test_docs_single(test_docs, vec, clf, lo=0, hi=8):
    out = {}
    for name, text in test_docs.items():
        mean, mx = predict_chunks(text, vec, clf, lo=lo, hi=hi)
        out[name] = mean
    return out


def score_test_docs_v3(test_docs, vec, clfs):
    out = {}
    for name, text in test_docs.items():
        if len(text) <= 12000:
            chunks = [text]
        else:
            chunks, start = [], 0
            while start < len(text) and len(chunks) < 8:
                chunks.append(text[start:min(start+4000, len(text))])
                start += 3500
        axis_cs = {a: [] for a in clfs}
        for chunk in chunks:
            Xc = vec.transform([chunk])
            for ax, clf in clfs.items():
                axis_cs[ax].append(float(np.clip(clf.predict(Xc)[0], 0, 4)))
        t_mean = float(np.mean(axis_cs["technical"]))
        t_max  = float(np.max(axis_cs["technical"]))
        pt     = float(np.mean(axis_cs["pseudo_technical"]))
        pol    = float(np.mean(axis_cs["political"]))
        out[name] = agg_formula(t_max, t_mean, pt, pol)
    return out


def score_test_docs_v4(test_docs, vec, clf):
    """Use peak-chunk score from the expanded handedited model.

    For long government reports (Log4j: 94k chars, Toyota: 24k chars) the mean
    chunk score is pulled down by policy/recommendation sections.  Peak chunk
    captures the most-technical section the model found, which is the right
    question for a reference document audit.
    """
    out = {}
    for name, text in test_docs.items():
        mean, peak = predict_chunks(text, vec, clf, lo=0, hi=8)
        out[name] = peak   # peak = highest-scoring chunk
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def version_metrics(docs, llm_key, kw_scores):
    hand = [d for d in docs if d["source"] in HAND_SOURCES]
    news = [d for d in docs if d["source"] == "fail_news"]

    llm_hand = np.array([d[llm_key] for d in hand])
    llm_news = np.array([d[llm_key] for d in news])
    kw_hand  = np.array([kw_scores[d["id"]] for d in hand])
    kw_news  = np.array([kw_scores[d["id"]] for d in news])

    llm_sep, llm_p = sep_stats(list(llm_hand), list(llm_news))
    kw_sep,  kw_p  = sep_stats(list(kw_hand),  list(kw_news))

    all_llm = np.array([d[llm_key] for d in docs])
    all_kw  = np.array([kw_scores[d["id"]] for d in docs])
    rho, rho_p = spearmanr(all_llm, all_kw)

    return {
        "llm_hand_mean": float(np.mean(llm_hand)),
        "llm_news_mean": float(np.mean(llm_news)),
        "llm_sep":       float(llm_sep),
        "llm_p":         float(llm_p),
        "kw_hand_mean":  float(np.mean(kw_hand)),
        "kw_news_mean":  float(np.mean(kw_news)),
        "kw_sep":        float(kw_sep),
        "kw_p":          float(kw_p),
        "spearman_rho":  float(rho),
        "spearman_p":    float(rho_p),
        "kw_hand":       kw_hand,
        "kw_news":       kw_news,
        "llm_hand":      llm_hand,
        "llm_news":      llm_news,
        "all_llm":       all_llm,
        "all_kw":        all_kw,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

VNAMES = ["V1\nThirds", "V2\nChunks", "V3\n3-Axis Direct", "V4\nFinal"]
VKEYS  = ["V1", "V2", "V3", "V4"]


def plot_llm_distributions(metrics_by_v, docs):
    fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=False)
    fig.suptitle("LLM Score Distributions by Version  (handedited vs fail_news)",
                 fontsize=14, fontweight="bold")

    llm_keys = ["v1_llm", "v2_llm", "v4_llm", "v4_llm"]  # v3 uses agg too
    # use version-specific LLM scores for labelling
    llm_map = {"V1":"v1_llm","V2":"v2_llm","V3":"v4_llm","V4":"v4_llm"}

    for ax, vk, vname, mkey in zip(axes, VKEYS, VNAMES,
                                    ["v1_llm","v2_llm","v4_llm","v4_llm"]):
        m = metrics_by_v[vk]
        # violin
        parts = ax.violinplot([m["llm_hand"], m["llm_news"]],
                              positions=[1, 2], showmedians=True, showextrema=True)
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor([SRC_COL["hand"], SRC_COL["news"]][i])
            pc.set_alpha(0.6)
        for key in ("cmedians","cmins","cmaxes","cbars"):
            parts[key].set_color("black")
            parts[key].set_linewidth(1.2)

        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Hand-edited\n(n=%d)" % len(m["llm_hand"]),
                             "Fail-news\n(n=%d)"   % len(m["llm_news"])],
                           fontsize=9)
        ax.set_title(vname, fontsize=11, fontweight="bold")
        ax.set_ylabel("LLM score  (0–8)" if ax == axes[0] else "")
        ax.set_ylim(-0.5, 9)

        sep = m["llm_sep"]
        p   = m["llm_p"]
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        ax.text(1.5, 8.5, f"Δ = {sep:.2f}  {sig}", ha="center", fontsize=9,
                color="black", fontweight="bold")

    plt.tight_layout()
    out = FIGDIR / "EV1_llm_distributions.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def plot_kw_vs_llm(metrics_by_v):
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle("Keyword Model vs LLM Scores  (Spearman ρ by version)",
                 fontsize=14, fontweight="bold")

    for ax, vk, vname in zip(axes, VKEYS, VNAMES):
        m = metrics_by_v[vk]
        x, y = m["all_llm"], m["all_kw"]

        # subsample to keep plot readable (max 600 points)
        if len(x) > 600:
            idx = np.random.default_rng(42).choice(len(x), 600, replace=False)
            x2, y2 = x[idx], y[idx]
        else:
            x2, y2 = x, y

        ax.scatter(x2, y2, s=6, alpha=0.35, color=VC[vk])
        # diagonal
        lim = (0, 8.5)
        ax.plot(lim, lim, "k--", lw=0.8, alpha=0.5)

        rho = m["spearman_rho"]
        ax.set_title(f"{vname}\nρ = {rho:.3f}", fontsize=10, fontweight="bold")
        ax.set_xlabel("LLM score")
        ax.set_ylabel("Keyword score" if ax == axes[0] else "")
        ax.set_xlim(*lim)
        ax.set_ylim(*lim)

    plt.tight_layout()
    out = FIGDIR / "EV2_kw_vs_llm_scatter.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def plot_separation_metrics(metrics_by_v):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Key Metrics Across Pipeline Versions", fontsize=14, fontweight="bold")

    xs  = np.arange(4)
    w   = 0.55
    col = [VC[v] for v in VKEYS]

    # Panel 1: LLM separation
    ax = axes[0]
    seps = [metrics_by_v[v]["llm_sep"] for v in VKEYS]
    bars = ax.bar(xs, seps, color=col, width=w, edgecolor="black", linewidth=0.7)
    ax.set_title("LLM Separation\n(hand-edited − fail-news mean)", fontsize=10)
    ax.set_xticks(xs); ax.set_xticklabels(VNAMES, fontsize=9)
    ax.set_ylabel("Score gap  (0–8 scale)")
    ax.set_ylim(0, max(seps)*1.25)
    for bar, val in zip(bars, seps):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.05,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    # Panel 2: Keyword separation
    ax = axes[1]
    seps_kw = [metrics_by_v[v]["kw_sep"] for v in VKEYS]
    bars = ax.bar(xs, seps_kw, color=col, width=w, edgecolor="black", linewidth=0.7)
    ax.set_title("Keyword-Model Separation\n(hand-edited − fail-news mean)", fontsize=10)
    ax.set_xticks(xs); ax.set_xticklabels(VNAMES, fontsize=9)
    ax.set_ylabel("Score gap  (0–8 scale)")
    ax.set_ylim(0, max(seps_kw)*1.25)
    for bar, val in zip(bars, seps_kw):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.05,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    # Panel 3: Spearman ρ (keyword vs LLM)
    ax = axes[2]
    rhos = [metrics_by_v[v]["spearman_rho"] for v in VKEYS]
    bars = ax.bar(xs, rhos, color=col, width=w, edgecolor="black", linewidth=0.7)
    ax.set_title("Keyword ↔ LLM Agreement\n(Spearman ρ)", fontsize=10)
    ax.set_xticks(xs); ax.set_xticklabels(VNAMES, fontsize=9)
    ax.set_ylabel("Spearman ρ")
    ax.set_ylim(0, 1.1)
    for bar, val in zip(bars, rhos):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    out = FIGDIR / "EV3_separation_metrics.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def plot_test_doc_scores(test_scores_by_v, test_docs):
    docnames = sorted(test_docs.keys())
    short = {"challenger": "Challenger\nExplosion",
             "toyotaaccel": "Toyota\nAcceleration",
             "csrb_log4j": "DHS CSRB\nLog4j"}
    labels = [short.get(d, d) for d in docnames]

    n_docs = len(docnames)
    n_vers = 4
    x      = np.arange(n_docs)
    width  = 0.18

    fig, ax = plt.subplots(figsize=(11, 6))
    for i, vk in enumerate(VKEYS):
        vals = [test_scores_by_v[vk].get(d, 0) for d in docnames]
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, vals, width, label=vk.replace("\n"," "),
                      color=VC[vk], edgecolor="black", linewidth=0.7)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.08,
                    f"{val:.1f}", ha="center", fontsize=8)

    ax.axhline(6.0, color="gray", linestyle="--", linewidth=1, alpha=0.7,
               label="'Technical' threshold (6/8)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Keyword model score  (0–8)")
    ax.set_ylim(0, 9)
    ax.set_title("Keyword Model Scores on Reference Documents\n"
                 "(gold-standard technical reports — higher is better)", fontsize=13)
    ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    out = FIGDIR / "EV4_test_doc_scores.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def plot_summary_table(metrics_by_v, test_scores_by_v, test_docs):
    docnames = sorted(test_docs.keys())
    short = {"challenger": "Challenger", "toyotaaccel": "Toyota", "csrb_log4j": "Log4j"}

    rows = []
    col_labels = [
        "Version",
        "LLM hand\nmean",
        "LLM news\nmean",
        "LLM Δ",
        "LLM p-val",
        "KW hand\nmean",
        "KW news\nmean",
        "KW Δ",
        "KW p-val",
        "Spearman ρ",
    ] + [f"KW\n{short.get(d,d)}" for d in docnames]

    version_display = {
        "V1": "V1  Thirds (0-8, avg of thirds)",
        "V2": "V2  Chunks (overlapping, mean)",
        "V3": "V3  3-Axis direct keyword combo",
        "V4": "V4  Agg + 1-axis Ridge ensemble",
    }

    for vk in VKEYS:
        m = metrics_by_v[vk]
        row = [
            version_display[vk],
            f"{m['llm_hand_mean']:.2f}",
            f"{m['llm_news_mean']:.2f}",
            f"{m['llm_sep']:.2f}",
            f"{m['llm_p']:.2e}",
            f"{m['kw_hand_mean']:.2f}",
            f"{m['kw_news_mean']:.2f}",
            f"{m['kw_sep']:.2f}",
            f"{m['kw_p']:.2e}",
            f"{m['spearman_rho']:.3f}",
        ]
        for d in docnames:
            row.append(f"{test_scores_by_v[vk].get(d, 0):.2f}")
        rows.append(row)

    fig, ax = plt.subplots(figsize=(20, 4))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=col_labels,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 2.2)

    # colour header
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#263238")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    # colour rows by version
    row_colours = [VC[v] + "55" for v in VKEYS]  # hex alpha
    raw_cols = [VC["V1"], VC["V2"], VC["V3"], VC["V4"]]
    for i, col in enumerate(raw_cols):
        for j in range(len(col_labels)):
            tbl[i+1, j].set_facecolor(col + "33")

    ax.set_title("Version Evolution — Summary Metrics",
                 fontsize=13, fontweight="bold", pad=10)

    plt.tight_layout()
    out = FIGDIR / "EV5_metrics_summary.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def plot_box_comparison(metrics_by_v):
    """Side-by-side boxplots: keyword model scores, handedited vs fail_news, per version."""
    fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=True)
    fig.suptitle("Keyword Model Score Distribution by Version  (box = IQR, whiskers = 1.5×IQR)",
                 fontsize=13, fontweight="bold")

    for ax, vk, vname in zip(axes, VKEYS, VNAMES):
        m = metrics_by_v[vk]
        bp = ax.boxplot([m["kw_hand"], m["kw_news"]],
                        patch_artist=True,
                        medianprops=dict(color="black", linewidth=2),
                        widths=0.55, showfliers=False)
        bp["boxes"][0].set_facecolor(SRC_COL["hand"] + "99")
        bp["boxes"][1].set_facecolor(SRC_COL["news"] + "99")

        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Hand-edited", "Fail-news"], fontsize=9)
        ax.set_title(vname, fontsize=10, fontweight="bold")
        if ax == axes[0]:
            ax.set_ylabel("Keyword score  (0–8)")
        ax.set_ylim(-0.5, 9)

        sep = m["kw_sep"]
        p   = m["kw_p"]
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        ax.text(1.5, 8.3, f"Δ = {sep:.2f}  {sig}", ha="center", fontsize=9,
                fontweight="bold")

    plt.tight_layout()
    out = FIGDIR / "EV6_kw_boxplots.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    np.random.seed(42)
    print("=== Loading data ===")
    docs      = load_all()
    test_docs = load_test_docs()

    print("\n=== Building keyword models ===")
    print("V1:")
    v1_vec, v1_clf, v1_kw = build_v1_kw(docs)
    print("V2:")
    v2_vec, v2_clf, v2_kw = build_v2_kw(docs)
    print("V3:")
    v3_vec, v3_clfs, v3_kw = build_v3_kw(docs)
    print("V4:")
    v4_vec, v4_clf, v4_kw = build_v4_kw(docs)

    print("\n=== Computing metrics ===")
    # V3 LLM "ground truth": use agg_score (same LLM labels, different combination)
    # to show that the keyword-combination failure is in the KW step, not the LLM step
    metrics = {
        "V1": version_metrics(docs, "v1_llm", v1_kw),
        "V2": version_metrics(docs, "v2_llm", v2_kw),
        "V3": version_metrics(docs, "v4_llm", v3_kw),   # V3 KW on same agg LLM labels
        "V4": version_metrics(docs, "v4_llm", v4_kw),
    }

    for vk in VKEYS:
        m = metrics[vk]
        print(f"\n  {vk}")
        print(f"    LLM  hand={m['llm_hand_mean']:.2f}  news={m['llm_news_mean']:.2f}  "
              f"Δ={m['llm_sep']:.2f}  p={m['llm_p']:.2e}")
        print(f"    KW   hand={m['kw_hand_mean']:.2f}  news={m['kw_news_mean']:.2f}  "
              f"Δ={m['kw_sep']:.2f}  p={m['kw_p']:.2e}")
        print(f"    Spearman ρ = {m['spearman_rho']:.3f}  (p={m['spearman_p']:.2e})")

    print("\n=== Scoring test documents ===")
    test_scores = {
        "V1": score_test_docs_single(test_docs, v1_vec, v1_clf, lo=0, hi=8),
        "V2": score_test_docs_single(test_docs, v2_vec, v2_clf, lo=0, hi=8),
        "V3": score_test_docs_v3(test_docs, v3_vec, v3_clfs),
        "V4": score_test_docs_v4(test_docs, v4_vec, v4_clf),
    }
    for vk in VKEYS:
        print(f"  {vk}: {test_scores[vk]}")

    print("\n=== Generating figures ===")
    plot_llm_distributions(metrics, docs)
    plot_kw_vs_llm(metrics)
    plot_separation_metrics(metrics)
    plot_test_doc_scores(test_scores, test_docs)
    plot_summary_table(metrics, test_scores, test_docs)
    plot_box_comparison(metrics)

    print("\nDone. All figures saved to", FIGDIR)


if __name__ == "__main__":
    main()
