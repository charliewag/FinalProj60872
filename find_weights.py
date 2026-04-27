"""
find_weights.py

Explores how different weight choices shape the score distribution for both
aggregation modes, then compares them directly on the same documents.

Goal
----
Use three descriptive diagnostics to judge whether a weight
combination is behaving sensibly:

  score_std      Standard deviation across all documents.
                 Too low → scores are bunched (formula not discriminating).
                 Too high → formula is oversensitive to noise.
                 Target: roughly 1.5–2.5 on a 0-8 scale.

  p90_hand       90th percentile of hand-edited scores.
                 Should be ≥ 6.0.  If the best postmortems only reach 5,
                 the scale is being wasted.

  p10_news       10th percentile of fail-news scores.
                 Should be ≤ 2.0.  If even the worst news articles score 3,
                 the formula isn't sensitive enough to political content.

  These are not combined into a single optimisation target.  The grid output
  lets you read the tradeoffs directly and pick weights that make sense.

Grid search
-----------
  One-axis:
    gap_penalty   in [0.0, 0.25, 0.5, 0.75, 1.0]
    gap_tolerance in [0.5, 1.0, 1.5, 2.0, 2.5]

  Three-axis:
    w_tm  in [1.5, 2.0, 2.5, 3.0]
    w_t   in [0.0, 0.25, 0.5, 0.75]
    w_pt  in [0.25, 0.5, 0.75, 1.0]
    w_pol in [0.5, 0.75, 1.0, 1.25]

Output
------
  Prints full grid tables for both modes.
  Saves analysis/figures/FW_weight_grid.png
  Saves data/best_weights.json with the defaults from aggregate_unified.py
  (not "best" by any single metric — just the semantically-chosen defaults
  documented there, confirmed or revised by inspection of this output).
"""

import json
import sys
import itertools
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

from aggregate_unified import agg_oneaxis, agg_threeaxis

sys.stdout.reconfigure(encoding="utf-8")

DATA   = Path("data")
FIGDIR = Path("analysis/figures")
FIGDIR.mkdir(parents=True, exist_ok=True)

HAND = {"danluu", "aws", "manual", "icco"}

TEST_DOCS = {
    "challenger": "test_docs/challenger.txt",
    "csrb_log4j": "test_docs/csrb_log4j.txt",
    "toyotaaccel": "test_docs/toyotaaccel.txt",
}

# ─────────────────────────────────────────────────────────────────────────────
# Load
# ─────────────────────────────────────────────────────────────────────────────

def load_corpus():
    with open(DATA / "postmortems_scored_chunks_oneaxis.json", encoding="utf-8") as f:
        oa = json.load(f)
    with open(DATA / "postmortems_scored.json", encoding="utf-8") as f:
        ta = json.load(f)

    oa_valid = [r for r in oa if r.get("score") is not None]
    ta_valid = [r for r in ta if r.get("technical") is not None]

    # align by id so we can compare the same documents
    oa_by_id = {r["id"]: r for r in oa_valid}
    ta_by_id = {r["id"]: r for r in ta_valid}
    common   = sorted(set(oa_by_id) & set(ta_by_id))

    oa_aligned = [oa_by_id[i] for i in common]
    ta_aligned = [ta_by_id[i] for i in common]

    print(f"Corpus: {len(common)} docs with both one-axis and three-axis scores")
    src = defaultdict(int)
    for r in oa_aligned:
        src[r.get("source", "?")] += 1
    print(f"  sources: {dict(src)}")

    return oa_aligned, ta_aligned


def load_test_docs():
    out = {}
    for name, path in TEST_DOCS.items():
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                out[name] = f.read()
        except FileNotFoundError:
            print(f"  [warn] test doc not found: {path}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostics  (descriptive, not a single optimisation target)
# ─────────────────────────────────────────────────────────────────────────────

def diagnostics(scores_dict, records):
    """
    scores_dict : {id: agg_score}
    records     : list of dicts with 'id' and 'source'
    Returns dict of descriptive metrics.
    """
    all_s  = [scores_dict[r["id"]] for r in records]
    hand_s = [scores_dict[r["id"]] for r in records if r.get("source") in HAND]
    news_s = [scores_dict[r["id"]] for r in records if r.get("source") == "fail_news"]

    return dict(
        std_all   = float(np.std(all_s)),
        mean_all  = float(np.mean(all_s)),
        p10_all   = float(np.percentile(all_s, 10)),
        p90_all   = float(np.percentile(all_s, 90)),
        p90_hand  = float(np.percentile(hand_s, 90)) if hand_s else 0,
        p50_hand  = float(np.percentile(hand_s, 50)) if hand_s else 0,
        p10_news  = float(np.percentile(news_s, 10)) if news_s else 8,
        p50_news  = float(np.percentile(news_s, 50)) if news_s else 8,
        hand_mean = float(np.mean(hand_s)) if hand_s else 0,
        news_mean = float(np.mean(news_s)) if news_s else 8,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Grid search
# ─────────────────────────────────────────────────────────────────────────────

def grid_oneaxis(oa_docs):
    gap_penalties  = [0.0, 0.25, 0.5, 0.75, 1.0]
    gap_tolerances = [0.5, 1.0, 1.5, 2.0, 2.5]

    results = []
    for gp, gt in itertools.product(gap_penalties, gap_tolerances):
        sc = {r["id"]: agg_oneaxis(r, gap_penalty=gp, gap_tolerance=gt)
              for r in oa_docs}
        d = diagnostics(sc, oa_docs)
        results.append(dict(gap_penalty=gp, gap_tolerance=gt, **d))

    return results


def grid_threeaxis(ta_docs):
    w_tms  = [1.5, 2.0, 2.5, 3.0]
    w_ts   = [0.0, 0.25, 0.5, 0.75]
    w_pts  = [0.25, 0.5, 0.75, 1.0]
    w_pols = [0.5, 0.75, 1.0, 1.25]

    results = []
    total = len(w_tms) * len(w_ts) * len(w_pts) * len(w_pols)
    print(f"  Three-axis grid: {total} combinations...")

    for w_tm, w_t, w_pt, w_pol in itertools.product(w_tms, w_ts, w_pts, w_pols):
        sc = {r["id"]: agg_threeaxis(r, w_tm=w_tm, w_t=w_t, w_pt=w_pt, w_pol=w_pol)
              for r in ta_docs}
        d = diagnostics(sc, ta_docs)
        results.append(dict(w_tm=w_tm, w_t=w_t, w_pt=w_pt, w_pol=w_pol, **d))

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Printing
# ─────────────────────────────────────────────────────────────────────────────

def print_grid(results, mode, sort_by="std_all", n=15):
    results_sorted = sorted(results, key=lambda x: -x[sort_by]
                            if sort_by in ("std_all",) else x[sort_by])
    print(f"\n{'='*90}")
    print(f"GRID  [{mode}]  sorted by {sort_by}  (showing {n} of {len(results)})")
    print(f"{'='*90}")
    if mode == "oneaxis":
        print(f"  {'gap_pen':>7}  {'gap_tol':>7}  "
              f"{'std':>6}  {'p90_hand':>8}  {'p10_news':>8}  "
              f"{'hand_mean':>9}  {'news_mean':>9}  {'p10':>5}  {'p90':>5}")
        for r in results_sorted[:n]:
            print(f"  {r['gap_penalty']:7.2f}  {r['gap_tolerance']:7.2f}  "
                  f"{r['std_all']:6.3f}  {r['p90_hand']:8.3f}  {r['p10_news']:8.3f}  "
                  f"{r['hand_mean']:9.3f}  {r['news_mean']:9.3f}  "
                  f"{r['p10_all']:5.2f}  {r['p90_all']:5.2f}")
    else:
        print(f"  {'w_tm':>5}  {'w_t':>5}  {'w_pt':>5}  {'w_pol':>5}  "
              f"{'std':>6}  {'p90_hand':>8}  {'p10_news':>8}  "
              f"{'hand_mean':>9}  {'news_mean':>9}  {'p10':>5}  {'p90':>5}")
        for r in results_sorted[:n]:
            print(f"  {r['w_tm']:5.2f}  {r['w_t']:5.2f}  {r['w_pt']:5.2f}  {r['w_pol']:5.2f}  "
                  f"{r['std_all']:6.3f}  {r['p90_hand']:8.3f}  {r['p10_news']:8.3f}  "
                  f"{r['hand_mean']:9.3f}  {r['news_mean']:9.3f}  "
                  f"{r['p10_all']:5.2f}  {r['p90_all']:5.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_weight_grid(oa_results, ta_results,
                     oa_default=(0.5, 1.5),
                     ta_default=(2.5, 0.5, 0.5, 0.75)):
    """
    2 × 3 grid of heatmaps — one row per mode, three diagnostic columns:
      std_all, p90_hand, p10_news
    Semantically-chosen defaults are starred on each panel.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Weight Grid — Descriptive Diagnostics  (not a single objective)",
                 fontsize=13, fontweight="bold")

    diag_cols = [
        ("std_all",  "RdYlGn", "Std across all docs\n(target ~1.5–2.5)"),
        ("p90_hand", "RdYlGn", "p90 of hand-edited scores\n(want ≥ 6.0)"),
        ("p10_news", "RdYlGn_r", "p10 of fail-news scores\n(want ≤ 2.0, lower = more sensitive)"),
    ]

    # ── Row 0: one-axis ──────────────────────────────────────────────────────
    gps  = sorted(set(r["gap_penalty"]   for r in oa_results))
    gts  = sorted(set(r["gap_tolerance"] for r in oa_results))

    for col, (metric, cmap, title) in enumerate(diag_cols):
        ax = axes[0, col]
        Z  = np.zeros((len(gts), len(gps)))
        for r in oa_results:
            i = gts.index(r["gap_tolerance"])
            j = gps.index(r["gap_penalty"])
            Z[i, j] = r[metric]
        im = ax.imshow(Z, aspect="auto", cmap=cmap, origin="lower")
        ax.set_xticks(range(len(gps)));  ax.set_xticklabels([f"{v:.2f}" for v in gps])
        ax.set_yticks(range(len(gts)));  ax.set_yticklabels([f"{v:.2f}" for v in gts])
        ax.set_xlabel("gap_penalty", fontsize=9)
        ax.set_ylabel("gap_tolerance", fontsize=9)
        ax.set_title(f"One-axis — {title}", fontsize=9)
        plt.colorbar(im, ax=ax, shrink=0.8)
        # mark default
        gp_def, gt_def = oa_default
        if gp_def in gps and gt_def in gts:
            xi, yi = gps.index(gp_def), gts.index(gt_def)
            ax.plot(xi, yi, "k*", markersize=14,
                    label=f"default ({gp_def},{gt_def})")
            ax.legend(fontsize=7)

    # ── Row 1: three-axis (marginalise w_t, w_pol → best per cell) ───────────
    w_tms = sorted(set(r["w_tm"] for r in ta_results))
    w_pts = sorted(set(r["w_pt"] for r in ta_results))

    for col, (metric, cmap, title) in enumerate(diag_cols):
        ax = axes[1, col]
        # For each (w_tm, w_pt) cell take the value from the row with default w_t/w_pol
        w_t_def, w_pol_def = ta_default[1], ta_default[3]
        Z = np.zeros((len(w_pts), len(w_tms)))
        for r in ta_results:
            if abs(r["w_t"] - w_t_def) < 1e-9 and abs(r["w_pol"] - w_pol_def) < 1e-9:
                i = w_pts.index(r["w_pt"])
                j = w_tms.index(r["w_tm"])
                Z[i, j] = r[metric]
        im = ax.imshow(Z, aspect="auto", cmap=cmap, origin="lower")
        ax.set_xticks(range(len(w_tms))); ax.set_xticklabels([f"{v:.1f}" for v in w_tms])
        ax.set_yticks(range(len(w_pts))); ax.set_yticklabels([f"{v:.2f}" for v in w_pts])
        ax.set_xlabel("w_tm  (tech_max weight)", fontsize=9)
        ax.set_ylabel("w_pt  (pseudo_tech penalty)", fontsize=9)
        ax.set_title(f"Three-axis — {title}\n(w_t={w_t_def}, w_pol={w_pol_def})", fontsize=9)
        plt.colorbar(im, ax=ax, shrink=0.8)
        # mark default
        wt_def, wpt_def = ta_default[0], ta_default[2]
        if wt_def in w_tms and wpt_def in w_pts:
            xi, yi = w_tms.index(wt_def), w_pts.index(wpt_def)
            ax.plot(xi, yi, "k*", markersize=14,
                    label=f"default ({wt_def},{wpt_def})")
            ax.legend(fontsize=7)

    plt.tight_layout()
    out = FIGDIR / "FW_weight_grid.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Show diagnostics for the semantically-chosen defaults
# ─────────────────────────────────────────────────────────────────────────────

def show_defaults(oa_results, ta_results, oa_docs, ta_docs):
    """Report diagnostics for the default weights from aggregate_unified.py."""
    OA_DEFAULTS = dict(gap_penalty=0.5, gap_tolerance=1.5)
    TA_DEFAULTS = dict(w_tm=2.5, w_t=0.5, w_pt=0.5, w_pol=0.75)

    # look up from grid
    def _find(results, **kw):
        for r in results:
            if all(abs(r[k] - v) < 1e-9 for k, v in kw.items()):
                return r
        return None

    d_oa = _find(oa_results, **OA_DEFAULTS)
    d_ta = _find(ta_results, **TA_DEFAULTS)

    print(f"\n{'='*70}")
    print(f"DEFAULTS  (semantically chosen — not optimised)")
    print(f"{'='*70}")

    if d_oa:
        print(f"\n  One-axis  gap_penalty={OA_DEFAULTS['gap_penalty']}  "
              f"gap_tolerance={OA_DEFAULTS['gap_tolerance']}")
        print(f"    std_all   = {d_oa['std_all']:.3f}   (target ~1.5–2.5)")
        print(f"    p90_hand  = {d_oa['p90_hand']:.3f}   (want ≥ 6.0)")
        print(f"    p10_news  = {d_oa['p10_news']:.3f}   (want ≤ 2.0)")
        print(f"    hand_mean = {d_oa['hand_mean']:.3f}   news_mean = {d_oa['news_mean']:.3f}")
    else:
        print("  [one-axis defaults not in grid — run with matching range]")

    if d_ta:
        print(f"\n  Three-axis  w_tm={TA_DEFAULTS['w_tm']}  w_t={TA_DEFAULTS['w_t']}"
              f"  w_pt={TA_DEFAULTS['w_pt']}  w_pol={TA_DEFAULTS['w_pol']}")
        print(f"    std_all   = {d_ta['std_all']:.3f}   (target ~1.5–2.5)")
        print(f"    p90_hand  = {d_ta['p90_hand']:.3f}   (want ≥ 6.0)")
        print(f"    p10_news  = {d_ta['p10_news']:.3f}   (want ≤ 2.0)")
        print(f"    hand_mean = {d_ta['hand_mean']:.3f}   news_mean = {d_ta['news_mean']:.3f}")
    else:
        print("  [three-axis defaults not in grid — run with matching range]")

    # Score percentiles side-by-side
    oa_sc = sorted([agg_oneaxis(r, **OA_DEFAULTS)  for r in oa_docs])
    ta_sc = sorted([agg_threeaxis(r, **TA_DEFAULTS) for r in ta_docs])
    print(f"\n  Score percentiles (defaults):")
    for pct in [10, 25, 50, 75, 90]:
        oi = min(int(pct / 100 * len(oa_sc)), len(oa_sc) - 1)
        ti = min(int(pct / 100 * len(ta_sc)), len(ta_sc) - 1)
        print(f"    p{pct:2d}   oneaxis={oa_sc[oi]:.2f}   threeaxis={ta_sc[ti]:.2f}")

    return d_oa, d_ta


def save_defaults(d_oa, d_ta):
    """Persist the default-weight diagnostic results for reference."""
    OA_DEFAULTS = dict(gap_penalty=0.5, gap_tolerance=1.5)
    TA_DEFAULTS = dict(w_tm=2.5, w_t=0.5, w_pt=0.5, w_pol=0.75)
    out = {
        "note": "Semantically-chosen defaults from aggregate_unified.py — not optimised",
        "oneaxis": {
            **OA_DEFAULTS,
            "diagnostics": {k: d_oa[k] for k in
                            ("std_all","p90_hand","p10_news","hand_mean","news_mean")}
                if d_oa else {},
        },
        "threeaxis": {
            **TA_DEFAULTS,
            "diagnostics": {k: d_ta[k] for k in
                            ("std_all","p90_hand","p10_news","hand_mean","news_mean")}
                if d_ta else {},
        },
    }
    path = DATA / "default_weights.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved default weight diagnostics -> {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=== Loading corpus ===")
    oa_docs, ta_docs = load_corpus()

    print("\n=== Grid search: one-axis ===")
    oa_results = grid_oneaxis(oa_docs)
    print_grid(oa_results, "oneaxis",  sort_by="std_all")
    print_grid(oa_results, "oneaxis",  sort_by="p90_hand", n=10)

    print("\n=== Grid search: three-axis ===")
    ta_results = grid_threeaxis(ta_docs)
    print_grid(ta_results, "threeaxis", sort_by="std_all")
    print_grid(ta_results, "threeaxis", sort_by="p90_hand", n=10)

    plot_weight_grid(oa_results, ta_results)

    d_oa, d_ta = show_defaults(oa_results, ta_results, oa_docs, ta_docs)
    save_defaults(d_oa, d_ta)

    print(f"\n=== Done ===")
    print(f"Inspect the table and FW_weight_grid.png then confirm or revise")
    print(f"the defaults in aggregate_unified.py before retraining.")


if __name__ == "__main__":
    main()
