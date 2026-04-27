"""
aggregate_unified.py

Unified chunk-aware score aggregator for both LLM scoring modes.

One-axis mode
-------------
  Input fields (from postmortems_scored_chunks_oneaxis.json):
    score       — mean score across all chunks × all runs        (0–8)
    score_max   — mean of per-run peak-chunk scores              (0–8)
    score_std   — std of per-run document means  (run consistency)
    n_chunks    — number of chunks the document was split into
    reasoning   — one-sentence rationale from the most-technical chunk

  Formula:
    gap   = score_max - score_mean
    final = score_max - gap_penalty * max(0, gap - gap_tolerance)
    final = clip(final, 0, 8)

  The peak is the primary signal — a postmortem is as useful as its most
  technical section.  But a high isolated peak surrounded by low chunks
  (large gap) suggests the technical content is a brief aside inside a PR
  document, so we apply a gap penalty only when the gap exceeds a tolerance.

  Defaults  gap_penalty=0.5, gap_tolerance=1.5
    gap_tolerance  how many points of peak-vs-mean gap are acceptable
                   before any penalty kicks in.  1.5 is chosen because a
                   long document naturally has some low-content sections
                   (introduction, timeline) that pull the mean down even
                   when the technical sections are strong.
    gap_penalty    for every point of gap beyond the tolerance, the score
                   is reduced by this amount.  0.5 means a 3-point excess
                   gap (e.g., max=7, mean=2.5) costs 1.5 points off the peak.

  Single-chunk documents: gap=0, so final = score_max = score. No penalty.

Three-axis mode
---------------
  Input fields (from postmortems_scored.json):
    technical        — mean TECHNICAL score across chunks × runs  (0–4)
    technical_max    — mean of per-run peak-chunk TECHNICAL        (0–4)
    pseudo_technical — mean PSEUDO_TECHNICAL                       (0–4)
    political        — mean POLITICAL                              (0–4)
    n_chunks
    failure_type     — short failure category from most-technical chunk

  Formula:
    pol_pen = political_mean * max(0, 1 - technical_max / 4)

    raw =   w_tm * technical_max       primary: peak technical content
          + w_t  * technical_mean      secondary: sustained technical signal
          - w_pt * pseudo_technical    jargon-without-substance penalty (mean)
          - w_pol * pol_pen            PR framing penalty, attenuated when
                                       technical_max is high

    final = clip( (raw - RAW_MIN) / (RAW_MAX - RAW_MIN) * 8,  0, 8 )
    RAW_MIN = -(w_pt * 4 + w_pol * 4)
    RAW_MAX =   w_tm * 4 + w_t  * 4

  The political penalty attenuation means: a document can open with a PR
  paragraph and still score high if its technical sections are substantive.
  Pseudo-technical jargon is always penalised regardless of technical quality —
  it signals an intent to mislead rather than a structural compromise.

  Defaults  w_tm=2.5, w_t=0.5, w_pt=0.5, w_pol=1.5
    w_tm  = 2.5  technical_max is the dominant axis; on a 0-4 scale this
                 gives a max raw technical contribution of 10 points before
                 normalisation.
    w_t   = 0.5  technical_mean adds a small "sustaining" signal — confirms
                 the peak is not a one-paragraph anomaly.
    w_pt  = 0.5  pseudo_technical penalty is mild. Jargon is bad but does
                 not negate real technical content in other sections.
    w_pol = 1.5  political penalty uses partial attenuation (1 - tm/8):
                 full weight at tm=0, half weight at tm=4 — political always
                 matters but hurts weak-technical docs most.

Usage
-----
  python aggregate_unified.py --mode oneaxis
  python aggregate_unified.py --mode threeaxis
  python aggregate_unified.py --mode oneaxis   --gap_penalty 0.5 --gap_tolerance 1.5
  python aggregate_unified.py --mode threeaxis --w_tm 2.5 --w_t 0.5 --w_pt 0.5 --w_pol 0.75

Output
------
  data/postmortems_agg_oneaxis.json   (mode=oneaxis)
  data/postmortems_agg_threeaxis.json (mode=threeaxis)
  Each record gets an 'agg_score' field (0–8) plus the source reasoning/failure_type.
"""

import json
import sys
import argparse
import numpy as np
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8")

INPUT_ONEAXIS   = "data/postmortems_scored_chunks_oneaxis.json"
INPUT_THREEAXIS = "data/postmortems_scored.json"
OUTPUT_ONEAXIS   = "data/postmortems_agg_oneaxis.json"
OUTPUT_THREEAXIS = "data/postmortems_agg_threeaxis.json"


# ─────────────────────────────────────────────────────────────────────────────
# Core aggregation functions
# ─────────────────────────────────────────────────────────────────────────────

def agg_oneaxis(record, gap_penalty=0.5, gap_tolerance=1.5):
    """
    Returns a 0-8 engineer-utility score from one-axis LLM fields.

    The peak chunk score is the primary signal.  A gap penalty kicks in only
    when the peak substantially exceeds the mean — indicating isolated rather
    than sustained technical content.

    gap_penalty   : score reduction per point of gap beyond the tolerance.
    gap_tolerance : acceptable peak-vs-mean gap before any penalty.
    """
    s_mean = float(record.get("score",     0) or 0)
    s_max  = float(record.get("score_max", s_mean) or s_mean)

    gap     = max(0.0, s_max - s_mean)
    penalty = gap_penalty * max(0.0, gap - gap_tolerance)
    return float(np.clip(s_max - penalty, 0.0, 8.0))


def agg_threeaxis(record, w_tm=2.5, w_t=0.5, w_pt=0.5, w_pol=1.5):
    """
    Returns a 0-8 engineer-utility score from three-axis LLM fields.

    w_tm  : weight on technical_max  — peak technical content, dominant signal
    w_t   : weight on technical mean — sustaining signal, small contribution
    w_pt  : weight on pseudo_technical mean — jargon penalty, always applied
    w_pol : weight on political penalty — partial attenuation, never fully zeroes
    """
    tm  = float(record.get("technical_max",    0) or 0)
    t   = float(record.get("technical",        0) or 0)
    pt  = float(record.get("pseudo_technical", 0) or 0)
    pol = float(record.get("political",        0) or 0)

    # Political penalty reduces as technical_max rises but never zeroes:
    # at tm=0 full weight, at tm=4 half weight — political always matters.
    pol_pen = pol * (1.0 - tm / 8.0)

    raw     = w_tm * tm + w_t * t - w_pt * pt - w_pol * pol_pen
    raw_min = -(w_pt * 4.0 + w_pol * 4.0)
    raw_max =   w_tm * 4.0 + w_t  * 4.0

    if raw_max == raw_min:
        return 4.0
    return float(np.clip((raw - raw_min) / (raw_max - raw_min) * 8.0, 0.0, 8.0))


# ─────────────────────────────────────────────────────────────────────────────
# Batch processing
# ─────────────────────────────────────────────────────────────────────────────

def process_oneaxis(gap_penalty=0.5, gap_tolerance=1.5, verbose=True):
    with open(INPUT_ONEAXIS, encoding="utf-8") as f:
        data = json.load(f)

    scored = []
    for r in data:
        if r.get("score") is None:
            scored.append(r)
            continue
        rec = dict(r)
        rec["agg_score"] = agg_oneaxis(r, gap_penalty=gap_penalty,
                                           gap_tolerance=gap_tolerance)
        scored.append(rec)

    with open(OUTPUT_ONEAXIS, "w", encoding="utf-8") as f:
        json.dump(scored, f, indent=2)

    if verbose:
        _print_summary(scored, "oneaxis",
                       gap_penalty=gap_penalty, gap_tolerance=gap_tolerance)
    return scored


def process_threeaxis(w_tm=2.5, w_t=0.5, w_pt=0.5, w_pol=1.5, verbose=True):
    with open(INPUT_THREEAXIS, encoding="utf-8") as f:
        data = json.load(f)

    scored = []
    for r in data:
        if r.get("technical") is None:
            scored.append(r)
            continue
        rec = dict(r)
        rec["agg_score"] = agg_threeaxis(r, w_tm=w_tm, w_t=w_t, w_pt=w_pt, w_pol=w_pol)
        scored.append(rec)

    with open(OUTPUT_THREEAXIS, "w", encoding="utf-8") as f:
        json.dump(scored, f, indent=2)

    if verbose:
        _print_summary(scored, "threeaxis", w_tm=w_tm, w_t=w_t, w_pt=w_pt, w_pol=w_pol)
    return scored


# ─────────────────────────────────────────────────────────────────────────────
# Summary printer
# ─────────────────────────────────────────────────────────────────────────────

HAND = {"danluu", "aws", "manual", "icco"}

def _print_summary(scored, mode, **params):
    valid = [r for r in scored if r.get("agg_score") is not None]
    hand  = [r for r in valid if r.get("source") in HAND]
    news  = [r for r in valid if r.get("source") == "fail_news"]

    param_str = "  ".join(f"{k}={v}" for k, v in params.items())
    print(f"\n=== {mode}  [{param_str}] ===")
    print(f"  total={len(valid)}  hand={len(hand)}  news={len(news)}")

    if hand and news:
        hm = np.mean([r["agg_score"] for r in hand])
        nm = np.mean([r["agg_score"] for r in news])
        hs = np.std( [r["agg_score"] for r in hand])
        ns = np.std( [r["agg_score"] for r in news])
        print(f"  hand  mean={hm:.3f}  std={hs:.3f}")
        print(f"  news  mean={nm:.3f}  std={ns:.3f}")
        print(f"  separation (hand-news) = {hm - nm:.3f}")

    # show reasoning / failure_type for top 5 hand-edited
    top5 = sorted(hand, key=lambda r: r["agg_score"], reverse=True)[:5]
    print(f"\n  Top 5 hand-edited:")
    for r in top5:
        extra = r.get("reasoning") or r.get("failure_type") or ""
        print(f"    {r['agg_score']:.2f}  {r.get('company','?')[:20]:20s}  "
              f"{r.get('title','')[:45]:45s}  [{extra[:60]}]")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Unified chunk aggregator")
    p.add_argument("--mode",   required=True, choices=["oneaxis", "threeaxis"])
    # one-axis params
    p.add_argument("--gap_penalty",   type=float, default=0.5,
                   help="Score reduction per point of gap beyond tolerance  [oneaxis, default 0.5]")
    p.add_argument("--gap_tolerance", type=float, default=1.5,
                   help="Acceptable peak-vs-mean gap before penalty kicks in  [oneaxis, default 1.5]")
    # three-axis params
    p.add_argument("--w_tm",  type=float, default=2.5,
                   help="Weight on technical_max            [threeaxis, default 2.5]")
    p.add_argument("--w_t",   type=float, default=0.5,
                   help="Weight on technical mean           [threeaxis, default 0.5]")
    p.add_argument("--w_pt",  type=float, default=0.5,
                   help="Weight on pseudo_technical mean    [threeaxis, default 0.5]")
    p.add_argument("--w_pol", type=float, default=1.5,
                   help="Weight on political penalty        [threeaxis, default 1.5]")
    args = p.parse_args()

    if args.mode == "oneaxis":
        process_oneaxis(gap_penalty=args.gap_penalty,
                        gap_tolerance=args.gap_tolerance)
        print(f"\nSaved -> {OUTPUT_ONEAXIS}")
    else:
        process_threeaxis(w_tm=args.w_tm, w_t=args.w_t,
                          w_pt=args.w_pt, w_pol=args.w_pol)
        print(f"\nSaved -> {OUTPUT_THREEAXIS}")


if __name__ == "__main__":
    main()
