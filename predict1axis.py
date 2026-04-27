"""
predict.py

Scores a text file on the 0-8 technical/political axis using a trained model.
For documents over MAX_CHARS, splits into overlapping chunks and scores each
independently — reporting mean score, peak (max) chunk score, and per-chunk
breakdown alongside keyword drivers.

The mean vs peak distinction matters: a document can have highly technical
sections (high peak) while still averaging out to a mixed score if it also
contains policy/PR framing.  Both numbers are reported.

Usage:
    python predict.py <textfile> [--model handedited|combined|ensemble]

Examples:
    python predict.py csrb_log4j.txt
    python predict.py challenger.txt --model handedited
    python predict.py challenger.txt --model ensemble

ensemble:
    Geometric mean of handedited and combined scores per chunk.
    sqrt(handedited * combined) — stays in 0-8 naturally, penalises
    disagreement, and combines political detection (combined) with
    technical depth detection (handedited).
"""

import sys
import pickle
import argparse
import numpy as np

MODEL_DIR  = 'data'
MAX_CHARS  = 12000   # docs under this are scored as a single chunk
CHUNK_SIZE = 4000    # chars per chunk for long docs
OVERLAP    = 500     # overlap between chunks to avoid cutting mid-sentence
MAX_CHUNKS = 8       # cap to avoid runaway on very long docs


def load_model(label):
    vec_path = f'{MODEL_DIR}/vectorizer_{label}.pkl'
    clf_path = f'{MODEL_DIR}/ridge_{label}.pkl'
    try:
        vectorizer = pickle.load(open(vec_path, 'rb'))
        clf        = pickle.load(open(clf_path, 'rb'))
        return vectorizer, clf
    except FileNotFoundError:
        print(f"Model '{label}' not found. Run train.py first.")
        sys.exit(1)


W_HE = 0.65   # handedited weight — lets technical depth dominate
W_CO = 1 - W_HE  # combined weight — corrective downward pull on political content


def score_chunk_ensemble(text, vec_he, clf_he, vec_co, clf_co, top_n=20):
    """Weighted geometric mean for a single chunk using pre-loaded models.

    he^W_HE * co^W_CO — stays in 0-8 naturally. Handedited drives technical
    recognition; combined provides political correction (co=0 pulls the score
    to 0 regardless of weight, so a strong political signal always wins).
    """
    s_he, tech_he, pol_he = score_text(text, vec_he, clf_he, top_n)
    s_co, _,       _      = score_text(text, vec_co, clf_co, top_n)
    score = float(np.clip((s_he ** W_HE) * (s_co ** W_CO), 0, 8))
    return score, tech_he, pol_he, s_he, s_co


def make_chunks(text):
    """
    Short docs: returned as a single chunk (no label added).
    Long docs: split into overlapping windows.  Each chunk is labeled with its
    position so the keyword output is anchored to a specific section.
    """
    if len(text) <= MAX_CHARS:
        return [(text, None)]   # (chunk_text, label_or_None)

    chunks = []
    start  = 0
    total  = len(text)

    while start < total and len(chunks) < MAX_CHUNKS:
        end   = min(start + CHUNK_SIZE, total)
        chunk = text[start:end]
        label = f"chars {start:,}-{end:,} of {total:,}"
        chunks.append((chunk, label))
        start += CHUNK_SIZE - OVERLAP

    return chunks


def score_text(text, vectorizer, clf, top_n=20):
    """Score a single chunk of text.  Returns (score, tech_drivers, pol_drivers)."""
    X     = vectorizer.transform([text])
    score = clf.predict(X)[0]
    score = float(np.clip(score, 0, 8))

    feature_names = vectorizer.get_feature_names_out()
    coefs         = clf.coef_
    tfidf_weights = np.asarray(X.todense()).flatten()
    contributions = tfidf_weights * coefs

    present_mask     = tfidf_weights > 0
    present_contribs = contributions[present_mask]
    present_names    = feature_names[present_mask]

    ranked = sorted(zip(present_contribs, present_names), reverse=True)

    tech_drivers = [(w, float(c)) for c, w in ranked if c > 0][:top_n]
    pol_drivers  = [(w, float(c)) for c, w in ranked if c < 0]
    pol_drivers  = sorted(pol_drivers, key=lambda x: x[1])[:top_n]

    return score, tech_drivers, pol_drivers


def render_bar(score, width=8):
    filled = int(round(max(0, min(width, score))))
    bar    = '#' * filled + '.' * (width - filled)
    if score <= 2:
        label = 'POLITICAL'
    elif score >= 6:
        label = 'TECHNICAL'
    else:
        label = 'MIXED'
    return f"[{bar}]  {score:.2f}/8  ({label})"


def print_drivers(tech_drivers, pol_drivers, top_n):
    if not tech_drivers and not pol_drivers:
        print(f"    (no known keywords found in this section)")
        return

    all_contribs = [abs(c) for _, c in tech_drivers + pol_drivers]
    max_contrib  = max(all_contribs) if all_contribs else 1.0
    BAR = 28

    if tech_drivers:
        print(f"    KEYWORDS -> TECHNICAL (+)")
        for word, contrib in tech_drivers[:top_n]:
            bar = '+' * int((abs(contrib) / max_contrib) * BAR)
            print(f"      {word:<28s} {contrib:+.5f}  {bar}")

    if pol_drivers:
        print(f"    KEYWORDS -> POLITICAL (-)")
        for word, contrib in pol_drivers[:top_n]:
            bar = '-' * int((abs(contrib) / max_contrib) * BAR)
            print(f"      {word:<28s} {contrib:+.5f}  {bar}")


def main():
    parser = argparse.ArgumentParser(description='Score a postmortem text file')
    parser.add_argument('textfile', help='Path to text file to score')
    parser.add_argument('--model', default='ensemble',
                        choices=['handedited', 'combined', 'ensemble'],
                        help='Which model to use (default: ensemble)')
    parser.add_argument('--top', type=int, default=10,
                        help='Number of keywords to show per chunk (default: 10)')
    args = parser.parse_args()

    try:
        with open(args.textfile, encoding='utf-8', errors='replace') as f:
            text = f.read()
    except FileNotFoundError:
        print(f"File not found: {args.textfile}")
        sys.exit(1)

    is_ensemble = args.model == 'ensemble'
    if is_ensemble:
        vec_he, clf_he = load_model('handedited')
        vec_co, clf_co = load_model('combined')
    else:
        vectorizer, clf = load_model(args.model)

    chunks = make_chunks(text)
    is_chunked = len(chunks) > 1

    print(f"\nfile:    {args.textfile}  ({len(text):,} chars)")
    print(f"model:   {args.model}")
    if is_chunked:
        print(f"chunks:  {len(chunks)}  x ~{CHUNK_SIZE:,} chars  (overlap={OVERLAP})")
    else:
        print(f"chunks:  1  (fits within {MAX_CHARS:,} char limit)")

    # --- score each chunk ---
    chunk_scores     = []
    chunk_results    = []   # (score, tech_drivers, pol_drivers, label, s_he, s_co)
    he_scores        = []   # per-chunk handedited scores (for LLM-style aggregation)
    co_scores        = []   # per-chunk combined scores

    for chunk_text, label in chunks:
        if is_ensemble:
            score, tech_drivers, pol_drivers, s_he, s_co = score_chunk_ensemble(
                chunk_text, vec_he, clf_he, vec_co, clf_co, top_n=args.top
            )
            chunk_scores.append(score)
            he_scores.append(s_he)
            co_scores.append(s_co)
            chunk_results.append((score, tech_drivers, pol_drivers, label, s_he, s_co))
        else:
            score, tech_drivers, pol_drivers = score_text(
                chunk_text, vectorizer, clf, top_n=args.top
            )
            chunk_scores.append(score)
            chunk_results.append((score, tech_drivers, pol_drivers, label, None, None))

    # Aggregate scores — ensemble uses LLM-style aggregation when chunked:
    # handedited: (2×max + 0.5×mean) / 2.5  — rewards peak technical sections
    # combined:   mean                        — political correction reflects whole doc
    if is_ensemble and is_chunked:
        agg_he   = (2.0 * max(he_scores) + 0.5 * float(np.mean(he_scores))) / 2.5
        agg_co   = float(np.mean(co_scores))
        mean_score = float(np.clip(agg_he ** W_HE * agg_co ** W_CO, 0, 8))
    else:
        mean_score = float(np.mean(chunk_scores))

    max_score  = float(np.max(chunk_scores))
    max_idx    = int(np.argmax(chunk_scores))

    # --- summary header ---
    print(f"\n{'='*58}")
    if is_chunked:
        if is_ensemble:
            agg_he_val = (2.0 * max(he_scores) + 0.5 * float(np.mean(he_scores))) / 2.5
            agg_co_val = float(np.mean(co_scores))
            print(f"SCORE:  {render_bar(mean_score)}")
            print(f"  he_agg={agg_he_val:.2f}  co_agg={agg_co_val:.2f}"
                  f"  (2*max+0.5*mean)/2.5 * mean")
        else:
            print(f"MEAN SCORE:  {render_bar(mean_score)}")
        print(f"PEAK CHUNK:  {render_bar(max_score)}"
              f"  <- chunk {max_idx+1}")
        scores_str = '  '.join(f"c{i+1}:{s:.1f}" for i, s in enumerate(chunk_scores))
        label = "per-chunk" if not is_ensemble else "per-chunk ensemble"
        print(f"\n  {label}: {scores_str}")
    else:
        print(f"SCORE:  {render_bar(mean_score)}")
        if is_ensemble:
            _, _, _, _, s_he, s_co = chunk_results[0]
            print(f"  handedited: {s_he:.2f}   combined: {s_co:.2f}"
                  f"   geom-mean: {mean_score:.2f}")
    print(f"{'='*58}")
    print(f"  0 = completely political  (vague reassurance, no mechanism)")
    print(f"  8 = completely technical  (root cause, mechanism, fix, reproducible)")

    # --- per-chunk keyword breakdown ---
    if is_chunked:
        print(f"\n{'='*58}")
        print(f"PER-CHUNK KEYWORD BREAKDOWN")
        print(f"{'='*58}")
        for i, (score, tech_drivers, pol_drivers, label, s_he, s_co) in enumerate(chunk_results):
            peak_tag = '  <- PEAK' if i == max_idx else ''
            ensemble_tag = f"  (he={s_he:.1f} co={s_co:.1f})" if is_ensemble else ""
            print(f"\n  chunk {i+1}  [{label}]  {render_bar(score)}{peak_tag}{ensemble_tag}")
            print_drivers(tech_drivers, pol_drivers, args.top)
    else:
        score, tech_drivers, pol_drivers, _, _, _ = chunk_results[0]
        print()
        print_drivers(tech_drivers, pol_drivers, args.top)

    _, best_tech, best_pol, _, _, _ = chunk_results[max_idx]
    print(f"\n{'='*58}")
    if is_chunked:
        print(f"  peak chunk signal  (chunk {max_idx+1}):")
    tech_sum = sum(c for _, c in best_tech)
    pol_sum  = sum(c for _, c in best_pol)
    print(f"    technical pull:  {tech_sum:+.4f}")
    print(f"    political pull:  {pol_sum:+.4f}")
    print(f"    net signal:      {tech_sum + pol_sum:+.4f}")


if __name__ == '__main__':
    main()