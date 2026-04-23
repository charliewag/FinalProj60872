"""
predict3axis.py

Scores a text file on all three axes (TECHNICAL, PSEUDO_TECHNICAL, POLITICAL)
using the trained Ridge models from train3axis.py.

For documents over MAX_CHARS, splits into overlapping chunks and scores each
independently — reporting mean and peak per axis.

Usage:
    python predict3axis.py <textfile> [--model handedited|combined]

Examples:
    python predict3axis.py csrb_log4j.txt
    python predict3axis.py challenger.txt --model handedited
"""

import sys
import pickle
import argparse
import numpy as np

MODEL_DIR  = 'data'
MAX_CHARS  = 12000
CHUNK_SIZE = 4000
OVERLAP    = 500
MAX_CHUNKS = 8

AXES = ('technical', 'pseudo_technical', 'political')


def load_model(label):
    vec_path = f'{MODEL_DIR}/vectorizer_{label}.pkl'
    try:
        vectorizer = pickle.load(open(vec_path, 'rb'))
    except FileNotFoundError:
        print(f"Vectorizer for '{label}' not found. Run train3axis.py first.")
        sys.exit(1)

    clfs = {}
    for axis in AXES:
        clf_path = f'{MODEL_DIR}/ridge_{axis}_{label}.pkl'
        try:
            clfs[axis] = pickle.load(open(clf_path, 'rb'))
        except FileNotFoundError:
            print(f"Model ridge_{axis}_{label}.pkl not found. Run train3axis.py first.")
            sys.exit(1)

    return vectorizer, clfs


def make_chunks(text):
    if len(text) <= MAX_CHARS:
        return [(text, None)]

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


def score_text(text, vectorizer, clfs, top_n=20):
    """
    Score a single chunk on all three axes.
    Returns {axis: (score, pos_drivers, neg_drivers)}.
    """
    X             = vectorizer.transform([text])
    feature_names = vectorizer.get_feature_names_out()
    tfidf_weights = np.asarray(X.todense()).flatten()
    present_mask  = tfidf_weights > 0
    present_names = feature_names[present_mask]
    present_tfidf = tfidf_weights[present_mask]

    results = {}
    for axis in AXES:
        clf   = clfs[axis]
        score = float(np.clip(clf.predict(X)[0], 0, 4))

        coefs         = clf.coef_[present_mask]
        contributions = present_tfidf * coefs
        ranked        = sorted(zip(contributions, present_names), reverse=True)

        pos_drivers = [(w, float(c)) for c, w in ranked if c > 0][:top_n]
        neg_drivers = [(w, float(c)) for c, w in ranked if c < 0]
        neg_drivers = sorted(neg_drivers, key=lambda x: x[1])[:top_n]

        results[axis] = (score, pos_drivers, neg_drivers)

    return results


def render_bar(score, width=4):
    filled = int(round(max(0, min(width, score))))
    bar    = '#' * filled + '.' * (width - filled)
    if score <= 1.0:
        label = 'LOW'
    elif score >= 3.0:
        label = 'HIGH'
    else:
        label = 'MED'
    return f"[{bar}]  {score:.2f}/4  ({label})"


def print_drivers(pos_drivers, neg_drivers, top_n, axis):
    if not pos_drivers and not neg_drivers:
        print(f"    (no known keywords found)")
        return

    all_contribs = [abs(c) for _, c in pos_drivers + neg_drivers]
    max_contrib  = max(all_contribs) if all_contribs else 1.0
    BAR = 28

    if pos_drivers:
        print(f"    KEYWORDS -> HIGH (+)")
        for word, contrib in pos_drivers[:top_n]:
            bar = '+' * int((abs(contrib) / max_contrib) * BAR)
            print(f"      {word:<28s} {contrib:+.5f}  {bar}")

    if neg_drivers:
        print(f"    KEYWORDS -> LOW (-)")
        for word, contrib in neg_drivers[:top_n]:
            bar = '-' * int((abs(contrib) / max_contrib) * BAR)
            print(f"      {word:<28s} {contrib:+.5f}  {bar}")


def main():
    parser = argparse.ArgumentParser(description='Score a postmortem on three axes')
    parser.add_argument('textfile', help='Path to text file to score')
    parser.add_argument('--model', default='handedited',
                        choices=['handedited', 'combined'],
                        help='Which model to use (default: handedited)')
    parser.add_argument('--top', type=int, default=10,
                        help='Number of keywords to show per chunk (default: 10)')
    args = parser.parse_args()

    try:
        with open(args.textfile, encoding='utf-8', errors='replace') as f:
            text = f.read()
    except FileNotFoundError:
        print(f"File not found: {args.textfile}")
        sys.exit(1)

    vectorizer, clfs = load_model(args.model)
    chunks     = make_chunks(text)
    is_chunked = len(chunks) > 1

    print(f"\nfile:    {args.textfile}  ({len(text):,} chars)")
    print(f"model:   {args.model}")
    if is_chunked:
        print(f"chunks:  {len(chunks)}  x ~{CHUNK_SIZE:,} chars  (overlap={OVERLAP})")
    else:
        print(f"chunks:  1  (fits within {MAX_CHARS:,} char limit)")

    # --- score each chunk ---
    chunk_results = []   # list of {axis: (score, pos, neg)}

    for chunk_text, chunk_label in chunks:
        chunk_results.append(score_text(chunk_text, vectorizer, clfs, top_n=args.top))

    # per-axis mean and peak
    axis_scores = {}
    for axis in AXES:
        scores = [r[axis][0] for r in chunk_results]
        axis_scores[axis] = {
            'mean':    float(np.mean(scores)),
            'max':     float(np.max(scores)),
            'max_idx': int(np.argmax(scores)),
            'all':     scores,
        }

    # --- summary header ---
    print(f"\n{'='*60}")
    print(f"SCORES  (0-4 per axis)")
    print(f"{'='*60}")
    for axis in AXES:
        s = axis_scores[axis]
        if is_chunked:
            print(f"  {axis:<20s}  mean: {render_bar(s['mean'])}  "
                  f"peak: {render_bar(s['max'])}  <- chunk {s['max_idx']+1}")
        else:
            print(f"  {axis:<20s}  {render_bar(s['mean'])}")
    print(f"{'='*60}")
    print(f"  0 = none of this axis  4 = all of this axis")

    if is_chunked:
        print(f"\n  per-chunk scores:")
        for axis in AXES:
            scores_str = '  '.join(f"c{i+1}:{s:.1f}"
                                   for i, s in enumerate(axis_scores[axis]['all']))
            print(f"    {axis:<20s}  {scores_str}")

    # --- per-chunk keyword breakdown ---
    if is_chunked:
        print(f"\n{'='*60}")
        print(f"PER-CHUNK KEYWORD BREAKDOWN")
        print(f"{'='*60}")
        for i, (result, (_, chunk_label)) in enumerate(zip(chunk_results, chunks)):
            print(f"\n  chunk {i+1}  [{chunk_label}]")
            for axis in AXES:
                score, pos, neg = result[axis]
                peak_tag = '  <- PEAK' if i == axis_scores[axis]['max_idx'] else ''
                print(f"    {axis:<20s}  {render_bar(score)}{peak_tag}")
                print_drivers(pos, neg, args.top, axis)
    else:
        # single chunk — show keyword lists per axis
        result = chunk_results[0]
        for axis in AXES:
            score, pos, neg = result[axis]
            print(f"\n  === {axis.upper()} ===")
            print_drivers(pos, neg, args.top, axis)

    # --- net signal summary per axis ---
    print(f"\n{'='*60}")
    print(f"NET SIGNAL SUMMARY  (from peak chunk per axis)")
    for axis in AXES:
        peak_idx         = axis_scores[axis]['max_idx']
        _, pos, neg      = chunk_results[peak_idx][axis]
        pos_sum = sum(c for _, c in pos)
        neg_sum = sum(c for _, c in neg)
        print(f"  {axis:<20s}  pos pull: {pos_sum:+.4f}  "
              f"neg pull: {neg_sum:+.4f}  net: {pos_sum + neg_sum:+.4f}")


if __name__ == '__main__':
    main()
