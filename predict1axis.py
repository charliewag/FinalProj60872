"""
predict1axis.py

Scores a text file using the centroid + number-pattern method.

Centroid method: compares the document's TF-IDF vector to the mean
vector of all technical training docs vs all political training docs.
Closer to the technical centroid = higher score. Vocabulary-independent
number patterns (hex, IPs, versions, sizes) provide an additional signal.

Final score = 0.75 * centroid_agg + 0.25 * number_score
  centroid_agg = 0.8 * chunk_max + 0.2 * chunk_mean
  threshold 0.50 -> TECHNICAL or POLITICAL

Usage:
    python predict1axis.py <textfile>
"""

import sys
import argparse
import pickle
import numpy as np

from train1axis import load, build_centroids, centroid_score, number_score, chunk_texts

CENTROID_WEIGHT = 0.75
NUMBER_WEIGHT   = 0.25
THRESHOLD       = 0.50


def load_vectorizer():
    try:
        with open('data/vectorizer_combined.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        print("Vectorizer not found — run train1axis.py first.")
        sys.exit(1)


def render_bar(score, threshold=THRESHOLD):
    filled = int(round(max(0.0, min(1.0, score)) * 10))
    bar    = '#' * filled + '.' * (10 - filled)
    label  = 'TECHNICAL' if score >= threshold else 'POLITICAL'
    return f"[{bar}]  {score:.3f}  ({label})"


def main():
    parser = argparse.ArgumentParser(description='Score a document with centroid method')
    parser.add_argument('textfile', help='Path to text file')
    args = parser.parse_args()

    try:
        with open(args.textfile, encoding='utf-8', errors='replace') as f:
            text = f.read()
    except FileNotFoundError:
        print(f"File not found: {args.textfile}")
        sys.exit(1)

    vec    = load_vectorizer()
    docs   = load(sources=None)
    c_tech, c_pol, boundary, sigma = build_centroids(vec, docs)

    chunks      = chunk_texts(text)
    cent_scores = []
    for chunk in chunks:
        x = np.asarray(vec.transform([chunk]).todense()).ravel()
        cent_scores.append(centroid_score(x, c_tech, c_pol, boundary, sigma))

    cent_agg = 0.8 * max(cent_scores) + 0.2 * float(np.mean(cent_scores))
    num      = number_score(text)
    final    = CENTROID_WEIGHT * cent_agg + NUMBER_WEIGHT * num

    print(f"\nfile:    {args.textfile}  ({len(text):,} chars)")
    print(f"chunks:  {len(chunks)}")
    print(f"\n{'='*52}")
    print(f"RESULT:  {render_bar(final)}")
    print(f"{'='*52}")
    print(f"  centroid_agg  {cent_agg:.3f}  (0.8*max + 0.2*mean across {len(chunks)} chunk(s))")
    print(f"  number_score  {num:.3f}")
    print(f"  final         {CENTROID_WEIGHT}*{cent_agg:.3f} + {NUMBER_WEIGHT}*{num:.3f} = {final:.3f}")
    if len(chunks) > 1:
        scores_str = '  '.join(f"c{i+1}:{s:.3f}" for i, s in enumerate(cent_scores))
        print(f"\n  per-chunk centroid: {scores_str}")
        print(f"  max={max(cent_scores):.3f}  mean={np.mean(cent_scores):.3f}")
    print(f"\n  0.00 = political  (closer to political centroid, no number signal)")
    print(f"  1.00 = technical  (closer to technical centroid + strong number signal)")


if __name__ == '__main__':
    main()
