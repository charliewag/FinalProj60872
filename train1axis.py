"""
train.py

Trains TF-IDF + Ridge regression on two datasets:
  1. Hand-edited postmortems only  (danluu, aws, manual, icco)
  2. Full combined dataset         (hand-edited + fail_news)

Outputs:
  data/model_handedited.json   — keywords + coefficients from hand-edited model
  data/model_combined.json     — keywords + coefficients from combined model
  data/model_comparison.json   — side-by-side diff of both keyword lists

Prints a comparison table showing which keywords agree, disagree, or are unique
to each model — useful for analysis section of paper.
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

# hand-edited sources — everything except fail_news
HANDEDITED_SOURCES = {'danluu', 'aws', 'manual', 'icco'}

RIDGE_ALPHA    = 10.0
MAX_FEATURES   = 2000
COEF_THRESHOLD = 0.01   # show keywords with |coef| above this
TOP_N          = 30

# ---------------------------------------------------------------------------
# Stopwords
# ---------------------------------------------------------------------------

EXTRA_STOPWORDS = {
    # news wire codes / byline boilerplate
    'fv', 'ln', 'pq', 'ap', 'afp',
    'said', 'says', 'told', 'reuters', 'mr', 'ms', 'mrs', 'dr',
    'spokesman', 'spokeswoman', 'spokesperson',
    'copyright', 'rights', 'reserved', 'associated',
    # NOTE: 'press' intentionally excluded — "press release", "press conference"
    # are genuine markers of PR framing and should remain a learnable signal.
    'reprinted', 'permission',
    # NOTE: 'report', 'reported', 'reporting' intentionally excluded —
    # government/NTSB/CSRB documents use "report" heavily and it should
    # stay as a signal of formal technical investigation.
    'article', 'story', 'wrote', 'writing', 'editor', 'editors',
    # day names
    'yesterday', 'monday', 'tuesday', 'wednesday',
    'thursday', 'friday', 'saturday', 'sunday',
    # month abbreviations — appear in postmortem timestamps, not meaningful signal
    'jan', 'feb', 'mar', 'apr', 'jun', 'jul',
    'aug', 'sep', 'oct', 'nov', 'dec',
    # timezone abbreviations — same problem as month abbreviations
    'pst', 'est', 'cst', 'mst', 'gmt', 'utc',
    # year numbers as bare tokens — spurious date signals
    '2019', '2020', '2021', '2022', '2023', '2024',
    # contraction artifacts: tokenizer splits "we've" -> "we" + "ve",
    # "we'll" -> "we" + "ll", etc.  These carry no semantic weight.
    've', 'll', 're', 'don', 'didn', 'doesn',
    'wasn', 'aren', 'isn', 'couldn', 'wouldn', 'shouldn', 'won',
}

STOP_WORDS = list(ENGLISH_STOP_WORDS.union(EXTRA_STOPWORDS))

# ---------------------------------------------------------------------------
# Seed vocabulary — terms we know should be meaningful signal
# ---------------------------------------------------------------------------
# These are used for diagnostic reporting after training: we check whether
# each seed actually made it into the model vocabulary and what coefficient
# it received.  They do NOT override the vectorizer — if a term doesn't
# appear in enough documents (min_df) it won't be in the model, which is
# itself informative (the corpus may lack that level of specificity).
#
# "Glitch" insight: vague failure words are political camouflage.
# Specific mechanism words are technical honesty.

POLITICAL_SEEDS = [
    # vague failure language — the "glitch" pattern
    'glitch', 'disruption', 'outage', 'issue', 'incident',
    'hiccup', 'degradation', 'impacted', 'affected',
    # PR / stakeholder reassurance language
    'apologize', 'sorry', 'sincerely', 'regret',
    'committed', 'dedication', 'trust', 'transparency',
    'going forward', 'best practices', 'lessons learned',
    'we take', 'working hard', 'top priority',
    'press release', 'press conference',
]

TECHNICAL_SEEDS = [
    # specific failure mechanism words
    'deadlock', 'race condition', 'memory leak', 'buffer overflow',
    'null pointer', 'segfault', 'stack overflow', 'integer overflow',
    'corruption', 'crash', 'exception', 'timeout', 'cascade',
    'regression', 'rollback', 'revert',
    # root cause / diagnosis language
    'root cause', 'failure mode', 'postmortem', 'contributing factor',
    'stack trace', 'core dump', 'log analysis',
    # remediation specificity
    'patch', 'hotfix', 'mitigation', 'workaround',
    'cve', 'vulnerability', 'exploit',
    # government / formal investigation language
    'ntsb', 'nhtsa', 'csrb', 'faa', 'investigation',
    'board', 'findings', 'recommendation',
]

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load(sources=None):
    """
    Load scored records, optionally filtered to a set of sources.
    sources=None means all sources.
    """
    with open(SCORED, encoding='utf-8') as f:
        data = json.load(f)

    docs = [r for r in data if r.get('score') is not None and r.get('text')]
    if sources:
        docs = [r for r in docs if r.get('source') in sources]

    score_dist = defaultdict(int)
    for r in docs:
        score_dist[r['score']] += 1

    src_dist = defaultdict(int)
    for r in docs:
        src_dist[r.get('source', 'unknown')] += 1

    print(f"  records:  {len(docs)}")
    print(f"  sources:  {dict(src_dist)}")
    print(f"  scores:   ", end='')
    print({s: score_dist[s] for s in range(9) if score_dist[s]})

    return docs

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train(docs, label):
    texts  = [r['text'] for r in docs]
    scores = np.array([r['score'] for r in docs], dtype=float)

    print(f"\n  mean score: {scores.mean():.2f}  std: {scores.std():.2f}")

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=MAX_FEATURES,
        stop_words=STOP_WORDS,
        min_df=2,        # lowered from 3 — corpus is ~170 docs, 3 was too aggressive
        sublinear_tf=True,
    )
    X = vectorizer.fit_transform(texts)

    clf = Ridge(alpha=RIDGE_ALPHA)
    clf.fit(X, scores)

    # cross-validation
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

    # save model artifacts for inference
    pickle.dump(vectorizer, open(f'data/vectorizer_{label}.pkl', 'wb'))
    pickle.dump(clf,        open(f'data/ridge_{label}.pkl',      'wb'))

    return vectorizer, clf


# ---------------------------------------------------------------------------
# Extract keywords
# ---------------------------------------------------------------------------

def extract_keywords(vectorizer, clf, top_n=TOP_N):
    names = vectorizer.get_feature_names_out()
    coefs = clf.coef_

    mask   = np.abs(coefs) > COEF_THRESHOLD
    active = sorted(zip(coefs[mask], names[mask]), reverse=True)

    technical = [(w, float(c)) for c, w in active if c > 0][:top_n]
    political = [(w, float(c)) for c, w in active if c < 0]
    political = sorted(political, key=lambda x: x[1])[:top_n]

    return technical, political


def report_seeds(vectorizer, clf):
    """
    Check how many seed terms made it into the model and what signal
    they carry.  Terms absent from the model vocabulary are either too
    rare in the corpus (below min_df) or filtered as stop words — both
    are meaningful diagnostics about the training data.
    """
    vocab = vectorizer.vocabulary_          # term -> column index
    coefs = clf.coef_

    print(f"\n  --- SEED DIAGNOSTIC ---")
    for label, seeds in [('POLITICAL', POLITICAL_SEEDS), ('TECHNICAL', TECHNICAL_SEEDS)]:
        found, missing = [], []
        for term in seeds:
            if term in vocab:
                c = float(coefs[vocab[term]])
                direction = 'POL' if c < 0 else 'TECH'
                found.append((term, c, direction))
            else:
                missing.append(term)

        print(f"\n  {label} seeds found in model ({len(found)}/{len(seeds)}):")
        for term, c, direction in sorted(found, key=lambda x: x[1]):
            match = 'OK' if (label == 'POLITICAL' and c < 0) or (label == 'TECHNICAL' and c > 0) else 'WRONG DIR'
            print(f"    {term:<30s}  coef={c:+.4f}  [{direction}]  {match}")

        if missing:
            print(f"\n  {label} seeds NOT in model (too rare or filtered):")
            for term in missing:
                print(f"    {term}")
    print()


# ---------------------------------------------------------------------------
# Print keyword table
# ---------------------------------------------------------------------------

def print_keywords(technical, political, label):
    all_coefs = [abs(c) for _, c in technical + political]
    max_coef  = max(all_coefs) if all_coefs else 1.0
    BAR       = 35

    print(f"\n  --- TECHNICAL (toward 8) ---")
    for word, coef in technical:
        bar = '+' * int((abs(coef) / max_coef) * BAR)
        print(f"    {word:<30s} {coef:+.4f}  {bar}")

    print(f"\n  --- POLITICAL (toward 0) ---")
    for word, coef in political:
        bar = '-' * int((abs(coef) / max_coef) * BAR)
        print(f"    {word:<30s} {coef:+.4f}  {bar}")


# ---------------------------------------------------------------------------
# Compare two keyword lists
# ---------------------------------------------------------------------------

def compare(kw_he, kw_co):
    """
    kw_he, kw_co: dicts of {word: coef} from each model.
    Prints a comparison showing agreement, disagreement, and unique terms.
    """
    all_words = set(kw_he.keys()) | set(kw_co.keys())

    both_tech   = []
    both_pol    = []
    flipped     = []
    he_only     = []
    co_only     = []

    for w in all_words:
        he = kw_he.get(w)
        co = kw_co.get(w)
        if he is not None and co is not None:
            if he > 0 and co > 0:
                both_tech.append((w, he, co))
            elif he < 0 and co < 0:
                both_pol.append((w, he, co))
            else:
                flipped.append((w, he, co))
        elif he is not None:
            he_only.append((w, he))
        else:
            co_only.append((w, co))

    both_tech.sort(key=lambda x: -(x[1] + x[2]))
    both_pol.sort(key=lambda x: (x[1] + x[2]))
    flipped.sort(key=lambda x: abs(x[1]) + abs(x[2]), reverse=True)

    print(f"\n{'='*65}")
    print(f"MODEL COMPARISON")
    print(f"{'='*65}")

    print(f"\n[AGREE — TECHNICAL in both models] ({len(both_tech)} words)")
    for w, he, co in both_tech[:20]:
        print(f"  {w:<30s}  handedited: {he:+.4f}  combined: {co:+.4f}")

    print(f"\n[AGREE — POLITICAL in both models] ({len(both_pol)} words)")
    for w, he, co in both_pol[:20]:
        print(f"  {w:<30s}  handedited: {he:+.4f}  combined: {co:+.4f}")

    print(f"\n[FLIPPED — opposite sign between models] ({len(flipped)} words)")
    print(f"  These are the most analytically interesting — a word that")
    print(f"  is technical in one corpus and political in the other.")
    for w, he, co in flipped[:20]:
        he_label = 'TECH' if he > 0 else 'POL'
        co_label = 'TECH' if co > 0 else 'POL'
        print(f"  {w:<30s}  handedited: {he_label} {he:+.4f}  combined: {co_label} {co:+.4f}")

    print(f"\n[UNIQUE to hand-edited model] ({len(he_only)} words)")
    for w, c in sorted(he_only, key=lambda x: -abs(x[1]))[:15]:
        print(f"  {w:<30s}  {c:+.4f}")

    print(f"\n[UNIQUE to combined model] ({len(co_only)} words)")
    for w, c in sorted(co_only, key=lambda x: -abs(x[1]))[:15]:
        print(f"  {w:<30s}  {c:+.4f}")

    return {
        'agree_technical': [{'word': w, 'handedited': he, 'combined': co}
                             for w, he, co in both_tech],
        'agree_political': [{'word': w, 'handedited': he, 'combined': co}
                             for w, he, co in both_pol],
        'flipped':         [{'word': w, 'handedited': he, 'combined': co}
                             for w, he, co in flipped],
        'handedited_only': [{'word': w, 'coef': c} for w, c in he_only],
        'combined_only':   [{'word': w, 'coef': c} for w, c in co_only],
    }


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save(technical, political, path):
    obj = {
        'technical': [{'keyword': w, 'coefficient': c} for w, c in technical],
        'political':  [{'keyword': w, 'coefficient': c} for w, c in political],
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2)
    print(f"  saved to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # --- hand-edited model (primary sources) ---
    print(f"\n{'='*65}")
    print(f"MODEL 1: HAND-EDITED ONLY  {HANDEDITED_SOURCES}")
    print(f"{'='*65}")
    docs_he = load(sources=HANDEDITED_SOURCES)
    vec_he, clf_he = train(docs_he, label='handedited')
    tech_he, pol_he = extract_keywords(vec_he, clf_he)
    print_keywords(tech_he, pol_he, 'handedited')
    report_seeds(vec_he, clf_he)
    save(tech_he, pol_he, 'data/model_handedited.json')

    # --- combined model (handedited + fail_news) ---
    # fail_news provides high volume of political-style text (n=1562, avg=0.70)
    # which strengthens the political side of the combined model's vocabulary.
    # Useful for analysis and comparison but interpret with care — political
    # keywords here may reflect news-wire style rather than corporate PR framing.
    print(f"\n{'='*65}")
    print(f"MODEL 2: COMBINED  (all sources)")
    print(f"{'='*65}")
    docs_co = load(sources=None)
    vec_co, clf_co = train(docs_co, label='combined')
    tech_co, pol_co = extract_keywords(vec_co, clf_co)
    print_keywords(tech_co, pol_co, 'combined')
    report_seeds(vec_co, clf_co)
    save(tech_co, pol_co, 'data/model_combined.json')

    # --- comparison ---
    kw_he = {w: c for w, c in tech_he + pol_he}
    kw_co = {w: c for w, c in tech_co + pol_co}
    comparison = compare(kw_he, kw_co)

    with open('data/model_comparison.json', 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2)
    print(f"\ncomparison saved to data/model_comparison.json")
    print(f"\nmodel artifacts saved:")
    print(f"  data/vectorizer_handedited.pkl")
    print(f"  data/ridge_handedited.pkl")
    print(f"  data/vectorizer_combined.pkl")
    print(f"  data/ridge_combined.pkl")
    print(f"\nrun predict.py to score new documents using either model")


if __name__ == '__main__':
    main()
