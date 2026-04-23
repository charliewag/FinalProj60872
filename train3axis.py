"""
train3axis.py

Trains TF-IDF + Ridge regression on three independent axes:
  TECHNICAL, PSEUDO_TECHNICAL, POLITICAL

One shared vectorizer per dataset, one Ridge model per axis per dataset.

Datasets:
  1. Hand-edited postmortems only  (danluu, aws, manual, icco)
  2. Full combined dataset         (hand-edited + fail_news)

Outputs per label (handedited / combined):
  data/vectorizer_{label}.pkl
  data/ridge_technical_{label}.pkl
  data/ridge_pseudo_technical_{label}.pkl
  data/ridge_political_{label}.pkl
  data/model_{label}.json          — keywords + coefficients for all three axes
  data/model_comparison.json       — side-by-side diff across both datasets
"""

import json
import pickle
import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

SCORED  = 'data/postmortems_scored.json'

HANDEDITED_SOURCES = {'danluu', 'aws', 'manual', 'icco'}

AXES = ('technical', 'pseudo_technical', 'political')

RIDGE_ALPHA    = 10.0
MAX_FEATURES   = 2000
COEF_THRESHOLD = 0.01
TOP_N          = 30

# ---------------------------------------------------------------------------
# Stopwords
# ---------------------------------------------------------------------------

EXTRA_STOPWORDS = {
    'fv', 'ln', 'pq', 'ap', 'afp',
    'said', 'says', 'told', 'reuters', 'mr', 'ms', 'mrs', 'dr',
    'spokesman', 'spokeswoman', 'spokesperson',
    'copyright', 'rights', 'reserved', 'associated',
    'reprinted', 'permission',
    'article', 'story', 'wrote', 'writing', 'editor', 'editors',
    'yesterday', 'monday', 'tuesday', 'wednesday',
    'thursday', 'friday', 'saturday', 'sunday',
    'jan', 'feb', 'mar', 'apr', 'jun', 'jul',
    'aug', 'sep', 'oct', 'nov', 'dec',
    'pst', 'est', 'cst', 'mst', 'gmt', 'utc',
    '2019', '2020', '2021', '2022', '2023', '2024',
    've', 'll', 're', 'don', 'didn', 'doesn',
    'wasn', 'aren', 'isn', 'couldn', 'wouldn', 'shouldn', 'won',
}

STOP_WORDS = list(ENGLISH_STOP_WORDS.union(EXTRA_STOPWORDS))

# ---------------------------------------------------------------------------
# Seed vocabulary
# ---------------------------------------------------------------------------

AXIS_SEEDS = {
    'technical': [
        'deadlock', 'race condition', 'memory leak', 'buffer overflow',
        'null pointer', 'segfault', 'stack overflow', 'integer overflow',
        'corruption', 'crash', 'exception', 'timeout', 'cascade',
        'regression', 'rollback', 'revert',
        'root cause', 'failure mode', 'postmortem', 'contributing factor',
        'stack trace', 'core dump', 'log analysis',
        'patch', 'hotfix', 'mitigation', 'workaround',
        'cve', 'vulnerability', 'exploit',
        'ntsb', 'nhtsa', 'csrb', 'faa', 'investigation',
        'board', 'findings', 'recommendation',
    ],
    'pseudo_technical': [
        'infrastructure', 'platform', 'ecosystem', 'solution', 'pipeline',
        'synergy', 'leverage', 'scalable', 'robust', 'resilient',
        'monitoring', 'observability', 'telemetry', 'dashboard',
        'microservices', 'cloud native', 'distributed system',
        'best practices', 'industry standard',
    ],
    'political': [
        'glitch', 'disruption', 'outage', 'issue', 'incident',
        'hiccup', 'degradation', 'impacted', 'affected',
        'apologize', 'sorry', 'sincerely', 'regret',
        'committed', 'dedication', 'trust', 'transparency',
        'going forward', 'lessons learned',
        'we take', 'working hard', 'top priority',
        'press release', 'press conference',
    ],
}

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load(sources=None):
    with open(SCORED, encoding='utf-8') as f:
        data = json.load(f)

    docs = [r for r in data if r.get('technical') is not None and r.get('text')]
    if sources:
        docs = [r for r in docs if r.get('source') in sources]

    src_dist = defaultdict(int)
    for r in docs:
        src_dist[r.get('source', 'unknown')] += 1

    print(f"  records:  {len(docs)}")
    print(f"  sources:  {dict(src_dist)}")
    for axis in AXES:
        vals = [r[axis] for r in docs]
        print(f"  {axis:<20s}  mean={np.mean(vals):.2f}  std={np.std(vals):.2f}"
              f"  min={min(vals):.2f}  max={max(vals):.2f}")

    return docs


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train_all(docs, label):
    """
    Fit one TF-IDF vectorizer and one Ridge per axis.
    Returns (vectorizer, {axis: clf}).
    """
    texts = [r['text'] for r in docs]

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=MAX_FEATURES,
        stop_words=STOP_WORDS,
        min_df=2,
        sublinear_tf=True,
    )
    X = vectorizer.fit_transform(texts)
    pickle.dump(vectorizer, open(f'data/vectorizer_{label}.pkl', 'wb'))

    clfs = {}
    for axis in AXES:
        scores = np.array([r[axis] for r in docs], dtype=float)
        print(f"\n  [{axis}]  mean={scores.mean():.2f}  std={scores.std():.2f}")

        clf = Ridge(alpha=RIDGE_ALPHA)
        clf.fit(X, scores)

        cv_r2   = cross_val_score(clf, X, scores, cv=5, scoring='r2')
        cv_rmse = cross_val_score(clf, X, scores, cv=5,
                                  scoring='neg_root_mean_squared_error')
        preds      = clf.predict(X)
        train_r2   = r2_score(scores, preds)
        train_rmse = mean_squared_error(scores, preds) ** 0.5

        print(f"  train  R2: {train_r2:.3f}   RMSE: {train_rmse:.3f}")
        print(f"  CV     R2: {cv_r2.mean():.3f} +/- {cv_r2.std():.3f}")
        print(f"  CV   RMSE: {(-cv_rmse).mean():.3f} +/- {cv_rmse.std():.3f}")

        gap = train_r2 - cv_r2.mean()
        if gap > 0.3:
            print(f"  WARNING: train/CV gap={gap:.2f} — overfitting")
        elif cv_r2.mean() < 0.2:
            print(f"  WARNING: CV R2 low — weak generalization")
        else:
            print(f"  train/CV gap={gap:.2f} — ok")

        pickle.dump(clf, open(f'data/ridge_{axis}_{label}.pkl', 'wb'))
        clfs[axis] = clf

    return vectorizer, clfs


# ---------------------------------------------------------------------------
# Extract keywords
# ---------------------------------------------------------------------------

def extract_keywords(vectorizer, clf, top_n=TOP_N):
    """
    Returns (high_drivers, low_drivers) for one axis.
    high_drivers: words with positive coefficients (push axis score up)
    low_drivers:  words with negative coefficients (push axis score down)
    """
    names = vectorizer.get_feature_names_out()
    coefs = clf.coef_

    mask   = np.abs(coefs) > COEF_THRESHOLD
    active = sorted(zip(coefs[mask], names[mask]), reverse=True)

    high = [(w, float(c)) for c, w in active if c > 0][:top_n]
    low  = [(w, float(c)) for c, w in active if c < 0]
    low  = sorted(low, key=lambda x: x[1])[:top_n]

    return high, low


def report_seeds(vectorizer, clfs):
    vocab = vectorizer.vocabulary_
    print(f"\n  --- SEED DIAGNOSTIC ---")
    for axis in AXES:
        coefs = clfs[axis].coef_
        seeds = AXIS_SEEDS[axis]
        found, missing = [], []
        for term in seeds:
            if term in vocab:
                c = float(coefs[vocab[term]])
                found.append((term, c))
            else:
                missing.append(term)

        print(f"\n  [{axis}] seeds found ({len(found)}/{len(seeds)}):")
        for term, c in sorted(found, key=lambda x: x[1]):
            direction = 'HIGH' if c > 0 else 'LOW'
            match = 'OK' if c > 0 else 'WRONG DIR'
            print(f"    {term:<30s}  coef={c:+.4f}  [{direction}]  {match}")

        if missing:
            print(f"\n  [{axis}] seeds NOT in model (too rare or filtered):")
            for term in missing:
                print(f"    {term}")
    print()


# ---------------------------------------------------------------------------
# Print keyword table
# ---------------------------------------------------------------------------

AXIS_LABELS = {
    'technical':       ('TECHNICAL (high)',    'TECHNICAL (low)'),
    'pseudo_technical':('JARGON (high)',        'PLAIN (low)'),
    'political':       ('POLITICAL (high)',     'POLITICAL (low)'),
}

def print_keywords_for_axis(axis, high, low):
    all_coefs = [abs(c) for _, c in high + low]
    max_coef  = max(all_coefs) if all_coefs else 1.0
    BAR       = 35
    hi_label, lo_label = AXIS_LABELS[axis]

    print(f"\n  --- {hi_label} ---")
    for word, coef in high:
        bar = '+' * int((abs(coef) / max_coef) * BAR)
        print(f"    {word:<30s} {coef:+.4f}  {bar}")

    print(f"\n  --- {lo_label} ---")
    for word, coef in low:
        bar = '-' * int((abs(coef) / max_coef) * BAR)
        print(f"    {word:<30s} {coef:+.4f}  {bar}")


# ---------------------------------------------------------------------------
# Compare two keyword sets for one axis
# ---------------------------------------------------------------------------

def compare_axis(axis, kw_he, kw_co):
    """
    kw_he, kw_co: {word: coef} dicts for one axis from each dataset.
    Returns comparison dict.
    """
    all_words = set(kw_he) | set(kw_co)

    both_high, both_low, flipped, he_only, co_only = [], [], [], [], []

    for w in all_words:
        he = kw_he.get(w)
        co = kw_co.get(w)
        if he is not None and co is not None:
            if he > 0 and co > 0:
                both_high.append((w, he, co))
            elif he < 0 and co < 0:
                both_low.append((w, he, co))
            else:
                flipped.append((w, he, co))
        elif he is not None:
            he_only.append((w, he))
        else:
            co_only.append((w, co))

    both_high.sort(key=lambda x: -(x[1] + x[2]))
    both_low.sort(key=lambda x: (x[1] + x[2]))
    flipped.sort(key=lambda x: abs(x[1]) + abs(x[2]), reverse=True)

    hi_label, lo_label = AXIS_LABELS[axis]
    print(f"\n[AGREE — {hi_label} in both models] ({len(both_high)} words)")
    for w, he, co in both_high[:15]:
        print(f"  {w:<30s}  he: {he:+.4f}  co: {co:+.4f}")

    print(f"\n[AGREE — {lo_label} in both models] ({len(both_low)} words)")
    for w, he, co in both_low[:15]:
        print(f"  {w:<30s}  he: {he:+.4f}  co: {co:+.4f}")

    print(f"\n[FLIPPED] ({len(flipped)} words)")
    for w, he, co in flipped[:10]:
        print(f"  {w:<30s}  he: {'HIGH' if he > 0 else 'LOW'} {he:+.4f}  "
              f"co: {'HIGH' if co > 0 else 'LOW'} {co:+.4f}")

    return {
        'agree_high':     [{'word': w, 'handedited': he, 'combined': co} for w, he, co in both_high],
        'agree_low':      [{'word': w, 'handedited': he, 'combined': co} for w, he, co in both_low],
        'flipped':        [{'word': w, 'handedited': he, 'combined': co} for w, he, co in flipped],
        'handedited_only':[{'word': w, 'coef': c} for w, c in he_only],
        'combined_only':  [{'word': w, 'coef': c} for w, c in co_only],
    }


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save(vectorizer, clfs, label):
    obj = {}
    for axis in AXES:
        high, low = extract_keywords(vectorizer, clfs[axis])
        obj[axis] = {
            'high': [{'keyword': w, 'coefficient': c} for w, c in high],
            'low':  [{'keyword': w, 'coefficient': c} for w, c in low],
        }
    path = f'data/model_{label}.json'
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2)
    print(f"  saved to {path}")
    return obj


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"\n{'='*65}")
    print(f"MODEL 1: HAND-EDITED ONLY  {HANDEDITED_SOURCES}")
    print(f"{'='*65}")
    docs_he = load(sources=HANDEDITED_SOURCES)
    vec_he, clfs_he = train_all(docs_he, label='handedited')
    for axis in AXES:
        high, low = extract_keywords(vec_he, clfs_he[axis])
        print(f"\n  === {axis.upper()} ===")
        print_keywords_for_axis(axis, high, low)
    report_seeds(vec_he, clfs_he)
    kw_he_obj = save(vec_he, clfs_he, 'handedited')

    print(f"\n{'='*65}")
    print(f"MODEL 2: COMBINED  (all sources)")
    print(f"{'='*65}")
    docs_co = load(sources=None)
    vec_co, clfs_co = train_all(docs_co, label='combined')
    for axis in AXES:
        high, low = extract_keywords(vec_co, clfs_co[axis])
        print(f"\n  === {axis.upper()} ===")
        print_keywords_for_axis(axis, high, low)
    report_seeds(vec_co, clfs_co)
    kw_co_obj = save(vec_co, clfs_co, 'combined')

    # --- comparison ---
    print(f"\n{'='*65}")
    print(f"MODEL COMPARISON (per axis)")
    print(f"{'='*65}")

    comparison = {}
    for axis in AXES:
        print(f"\n{'='*40}")
        print(f"AXIS: {axis.upper()}")
        print(f"{'='*40}")
        kw_he = {e['keyword']: e['coefficient'] for e in kw_he_obj[axis]['high'] + kw_he_obj[axis]['low']}
        kw_co = {e['keyword']: e['coefficient'] for e in kw_co_obj[axis]['high'] + kw_co_obj[axis]['low']}
        comparison[axis] = compare_axis(axis, kw_he, kw_co)

    with open('data/model_comparison.json', 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2)
    print(f"\ncomparison saved to data/model_comparison.json")
    print(f"\nmodel artifacts saved:")
    for label in ('handedited', 'combined'):
        print(f"  data/vectorizer_{label}.pkl")
        for axis in AXES:
            print(f"  data/ridge_{axis}_{label}.pkl")
    print(f"\nrun predict3axis.py to score new documents")


if __name__ == '__main__':
    main()
