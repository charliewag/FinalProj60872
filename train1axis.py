"""
train1axis.py

Binary classifier: TF-IDF + Ridge on {0=political, 1=technical} labels.
Excludes the ambiguous middle zone (3.0-5.5) so the model learns clean signal
from clearly political docs (<3.0) vs clearly technical docs (>=5.5).

Labels derived from postmortems_agg_threeaxis.json.

Two models:
  handedited  trained on hand-curated postmortems only
  combined    trained on all scored docs (handedited + fail_news)

Outputs:
  data/vectorizer_handedited.pkl   data/ridge_handedited.pkl
  data/vectorizer_combined.pkl     data/ridge_combined.pkl
  data/model_handedited.json       data/model_combined.json
  data/model_comparison.json
"""

import json
import pickle
import sys
import glob
import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score
from scipy.stats import mannwhitneyu

sys.stdout.reconfigure(encoding='utf-8')

SCORED = 'data/postmortems_agg_threeaxis.json'

HANDEDITED_SOURCES = {'danluu', 'aws', 'manual', 'icco'}

LOW_THRESH  = 3.0   # below this → political (0)
HIGH_THRESH = 5.5   # at or above this → technical (1)

RIDGE_ALPHA = 10.0
MIN_DF      = 3
COEF_THRESH = 0.005
TOP_N       = 25

EXTRA_STOPWORDS = {
    've', 'll', 're', 'don', 'didn', 'doesn',
    'wasn', 'aren', 'isn', 'couldn', 'wouldn', 'shouldn', 'won',
    'fv', 'ln', 'pq', 'newstex',
    '2019', '2020', '2021', '2022', '2023', '2024',
    'year', '000',
    'pst', 'est', 'cst', 'mst', 'gmt', 'utc',
    'pdt', 'edt', 'cdt', 'mdt', 'bst', 'cet', 'ist', 'jst', 'aest',
    'jan', 'feb', 'mar', 'apr', 'jun', 'jul',
    'aug', 'sep', 'oct', 'nov', 'dec',
    'mr', 'ms', 'mrs', 'dr',
}

STOP_WORDS = list(ENGLISH_STOP_WORDS.union(EXTRA_STOPWORDS))


# ---------------------------------------------------------------------------
# Load — binary labels, middle excluded
# ---------------------------------------------------------------------------

def load(sources=None):
    with open(SCORED, encoding='utf-8') as f:
        data = json.load(f)

    docs = [r for r in data if r.get('agg_score') is not None and r.get('text')]
    if sources:
        docs = [r for r in docs if r.get('source') in sources]

    before = len(docs)
    docs = [r for r in docs
            if r['agg_score'] < LOW_THRESH or r['agg_score'] >= HIGH_THRESH]

    for r in docs:
        r['label'] = 1 if r['agg_score'] >= HIGH_THRESH else 0

    n_tech = sum(1 for r in docs if r['label'] == 1)
    n_pol  = sum(1 for r in docs if r['label'] == 0)

    src_dist = defaultdict(int)
    for r in docs:
        src_dist[r.get('source', 'unknown')] += 1

    print(f"  total before filter: {before}")
    print(f"  kept:  {len(docs)}  (political={n_pol}, technical={n_tech})")
    print(f"  excluded middle [{LOW_THRESH}, {HIGH_THRESH}): {before - len(docs)}")
    print(f"  sources: {dict(src_dist)}")
    return docs


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train(docs, label):
    texts  = [r['text']  for r in docs]
    labels = np.array([r['label'] for r in docs], dtype=float)

    n_tech = int(labels.sum())
    n_pol  = int(len(labels) - n_tech)
    print(f"\n  political={n_pol}  technical={n_tech}  balance={n_tech/len(labels):.2f}")

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=None,
        stop_words=STOP_WORDS,
        min_df=MIN_DF,
        use_idf=False,
        norm='l2',
    )
    X = vectorizer.fit_transform(texts)
    print(f"  features: {X.shape[1]:,}")

    clf = Ridge(alpha=RIDGE_ALPHA)
    clf.fit(X, labels)

    preds_raw = clf.predict(X)
    preds_bin = (preds_raw >= 0.5).astype(int)
    train_acc = accuracy_score(labels, preds_bin)

    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_accs, cv_aucs = [], []
    for tr, te in kf.split(X):
        m = Ridge(alpha=RIDGE_ALPHA).fit(X[tr], labels[tr])
        p = m.predict(X[te])
        cv_accs.append(accuracy_score(labels[te], (p >= 0.5).astype(int)))
        cv_aucs.append(roc_auc_score(labels[te], p))

    print(f"  train acc: {train_acc:.3f}")
    print(f"  CV    acc: {np.mean(cv_accs):.3f} +/- {np.std(cv_accs):.3f}")
    print(f"  CV    AUC: {np.mean(cv_aucs):.3f} +/- {np.std(cv_aucs):.3f}")

    pickle.dump(vectorizer, open(f'data/vectorizer_{label}.pkl', 'wb'))
    pickle.dump(clf,        open(f'data/ridge_{label}.pkl',      'wb'))

    return vectorizer, clf


# ---------------------------------------------------------------------------
# Keywords
# ---------------------------------------------------------------------------

def extract_keywords(vectorizer, clf, top_n=TOP_N):
    names = vectorizer.get_feature_names_out()
    coefs = clf.coef_

    mask   = np.abs(coefs) > COEF_THRESH
    active = sorted(zip(coefs[mask], names[mask]), reverse=True)

    technical = [(w, float(c)) for c, w in active if c > 0][:top_n]
    political  = [(w, float(c)) for c, w in active if c < 0]
    political  = sorted(political, key=lambda x: x[1])[:top_n]

    return technical, political


def print_keywords(technical, political):
    all_coefs = [abs(c) for _, c in technical + political]
    max_coef  = max(all_coefs) if all_coefs else 1.0
    BAR = 35

    print(f"\n  --- TECHNICAL ---")
    for word, coef in technical:
        bar = '+' * int((abs(coef) / max_coef) * BAR)
        print(f"    {word:<30s} {coef:+.4f}  {bar}")

    print(f"\n  --- POLITICAL ---")
    for word, coef in political:
        bar = '-' * int((abs(coef) / max_coef) * BAR)
        print(f"    {word:<30s} {coef:+.4f}  {bar}")


def save_keywords(technical, political, path):
    obj = {
        'technical': [{'keyword': w, 'coefficient': c} for w, c in technical],
        'political':  [{'keyword': w, 'coefficient': c} for w, c in political],
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2)
    print(f"  saved -> {path}")


# ---------------------------------------------------------------------------
# Compare two keyword lists
# ---------------------------------------------------------------------------

def compare(kw_he, kw_co):
    all_words = set(kw_he) | set(kw_co)
    both_tech, both_pol, flipped, he_only, co_only = [], [], [], [], []

    for w in all_words:
        he = kw_he.get(w)
        co = kw_co.get(w)
        if he is not None and co is not None:
            if   he > 0 and co > 0: both_tech.append((w, he, co))
            elif he < 0 and co < 0: both_pol.append((w, he, co))
            else:                   flipped.append((w, he, co))
        elif he is not None: he_only.append((w, he))
        else:                co_only.append((w, co))

    both_tech.sort(key=lambda x: -(x[1] + x[2]))
    both_pol.sort(key=lambda x: (x[1] + x[2]))
    flipped.sort(key=lambda x: abs(x[1]) + abs(x[2]), reverse=True)

    print(f"\n{'='*65}")
    print(f"MODEL COMPARISON")
    print(f"{'='*65}")
    print(f"\n[AGREE — TECHNICAL]  ({len(both_tech)} words)")
    for w, he, co in both_tech[:20]:
        print(f"  {w:<30s}  he={he:+.4f}  co={co:+.4f}")
    print(f"\n[AGREE — POLITICAL]  ({len(both_pol)} words)")
    for w, he, co in both_pol[:20]:
        print(f"  {w:<30s}  he={he:+.4f}  co={co:+.4f}")
    print(f"\n[FLIPPED]  ({len(flipped)} words)")
    for w, he, co in flipped[:15]:
        print(f"  {w:<30s}  he={'TECH' if he>0 else 'POL'} {he:+.4f}"
              f"  co={'TECH' if co>0 else 'POL'} {co:+.4f}")

    return {
        'agree_technical': [{'word': w, 'he': he, 'co': co} for w, he, co in both_tech],
        'agree_political':  [{'word': w, 'he': he, 'co': co} for w, he, co in both_pol],
        'flipped':          [{'word': w, 'he': he, 'co': co} for w, he, co in flipped],
        'handedited_only':  [{'word': w, 'coef': c} for w, c in he_only],
        'combined_only':    [{'word': w, 'coef': c} for w, c in co_only],
    }


# ---------------------------------------------------------------------------
# Predict test docs
# ---------------------------------------------------------------------------

import re as _re

CHUNK_SIZE = 4000
CHUNK_OVERLAP = 500
MAX_CHUNKS = 10

# Technical number patterns: hex, IPs, version strings, sizes, latencies,
# precise percentages, ALL_CAPS constants, stack-frame addresses.
_TECH_NUM_PATTERNS = [
    r'\b0x[0-9a-fA-F]{2,}\b',              # hex literals
    r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',  # IP addresses
    r'\bv?\d+\.\d+\.\d+\b',                # version numbers
    r'\b\d+\s*(?:ms|us|ns|GB|MB|KB|TB|GiB|MiB|KiB)\b',  # sizes / latencies
    r'\b\d{1,3}\.\d{2,}%',                 # precise percentages
    r'\b[A-Z][A-Z_]{3,}\b',                # CONSTANTS / ERROR_CODES
    r':\d{4,5}\b',                         # port numbers
    r'\b\d{10,}\b',                        # unix timestamps / large IDs
]
_TECH_NUM_RE = _re.compile('|'.join(_TECH_NUM_PATTERNS))


def number_score(text):
    """0-1 signal: density of technical number patterns (vocab-independent)."""
    hits = len(_TECH_NUM_RE.findall(text))
    words = max(1, len(text.split()))
    return min(1.0, hits / words * 15)   # scale so ~1 hit per 15 words → 1.0


def chunk_texts(text):
    if len(text) <= CHUNK_SIZE:
        return [text]
    chunks, start = [], 0
    while start < len(text) and len(chunks) < MAX_CHUNKS:
        chunks.append(text[start:start + CHUNK_SIZE])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def build_centroids(vectorizer, docs):
    """
    Mean L2-normalised TF-IDF vector for each class.
    Also returns logistic calibration params (boundary, sigma) derived from
    the training distribution of cosine-diff scores.
    """
    texts_t = [r['text'] for r in docs if r['label'] == 1]
    texts_p = [r['text'] for r in docs if r['label'] == 0]
    Xt = vectorizer.transform(texts_t)
    Xp = vectorizer.transform(texts_p)
    c_tech = np.asarray(Xt.mean(axis=0)).ravel()
    c_pol  = np.asarray(Xp.mean(axis=0)).ravel()
    c_tech /= (np.linalg.norm(c_tech) + 1e-9)
    c_pol  /= (np.linalg.norm(c_pol)  + 1e-9)

    # Compute per-doc diffs on training data to calibrate the logistic
    def diff_for(texts):
        diffs = []
        for txt in texts:
            x = np.asarray(vectorizer.transform([txt]).todense()).ravel()
            xn = x / (np.linalg.norm(x) + 1e-9)
            diffs.append(float(xn @ c_tech) - float(xn @ c_pol))
        return np.array(diffs)

    d_t = diff_for(texts_t)
    d_p = diff_for(texts_p)
    boundary = (np.mean(d_t) + np.mean(d_p)) / 2      # midpoint of class means
    sigma    = (np.std(d_t) + np.std(d_p)) / 2         # pooled std
    print(f"  centroid boundary={boundary:.4f}  sigma={sigma:.4f}")
    return c_tech, c_pol, boundary, sigma


def centroid_score(x_arr, c_tech, c_pol, boundary, sigma):
    """Logistic-calibrated centroid score (0=political, 1=technical)."""
    xn   = x_arr / (np.linalg.norm(x_arr) + 1e-9)
    diff = float(xn @ c_tech) - float(xn @ c_pol)
    return float(1.0 / (1.0 + np.exp(-(diff - boundary) / (sigma + 1e-9))))


def predict_test_docs(vectorizer, docs_co, number_weight=0.25, centroid_weight=0.75):
    test_files = sorted(glob.glob('test_docs/*.txt'))
    if not test_files:
        return

    c_tech, c_pol, boundary, sigma = build_centroids(vectorizer, docs_co)

    print(f"\n{'='*60}")
    print(f"TEST DOC PREDICTIONS  (combined binary model)")
    print(f"  centroid={centroid_weight:.0%}  numbers={number_weight:.0%}  threshold=0.50")
    print(f"{'='*60}")
    print(f"  {'file':<32}  {'cent':>5}  {'num':>5}  {'final':>6}  label")
    print(f"  {'-'*62}")

    for path in test_files:
        try:
            with open(path, encoding='utf-8', errors='replace') as f:
                text = f.read()

            chunks     = chunk_texts(text)
            cent_scores = []
            for chunk in chunks:
                x_arr = np.asarray(vectorizer.transform([chunk]).todense()).ravel()
                cent_scores.append(centroid_score(x_arr, c_tech, c_pol, boundary, sigma))

            cent_agg = 0.8 * max(cent_scores) + 0.2 * float(np.mean(cent_scores))
            num      = number_score(text)
            final    = centroid_weight * cent_agg + number_weight * num
            label    = 'TECHNICAL' if final >= 0.50 else 'POLITICAL'
            name     = path.replace('\\', '/').split('/')[-1]
            print(f"  {name:<32}  {cent_agg:5.3f}  {num:5.3f}  {final:6.3f}  {label}")
        except Exception as e:
            print(f"  {path}: error — {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"\n{'='*65}")
    print(f"BINARY LABELS  political < {LOW_THRESH}  |  technical >= {HIGH_THRESH}")
    print(f"{'='*65}")

    print(f"\n--- MODEL 1: HAND-EDITED ONLY ---")
    docs_he = load(sources=HANDEDITED_SOURCES)
    vec_he, clf_he = train(docs_he, label='handedited')
    tech_he, pol_he = extract_keywords(vec_he, clf_he)
    print_keywords(tech_he, pol_he)
    save_keywords(tech_he, pol_he, 'data/model_handedited.json')

    print(f"\n--- MODEL 2: COMBINED (all sources) ---")
    docs_co = load(sources=None)
    vec_co, clf_co = train(docs_co, label='combined')
    tech_co, pol_co = extract_keywords(vec_co, clf_co)
    print_keywords(tech_co, pol_co)
    save_keywords(tech_co, pol_co, 'data/model_combined.json')

    kw_he = {w: c for w, c in tech_he + pol_he}
    kw_co = {w: c for w, c in tech_co + pol_co}
    comparison = compare(kw_he, kw_co)
    with open('data/model_comparison.json', 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2)

    predict_test_docs(vec_co, docs_co)

    print(f"\nArtifacts saved:")
    print(f"  data/vectorizer_{{handedited,combined}}.pkl")
    print(f"  data/ridge_{{handedited,combined}}.pkl")
    print(f"  data/model_{{handedited,combined,comparison}}.json")


if __name__ == '__main__':
    main()
