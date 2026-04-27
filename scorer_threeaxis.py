import requests
import json
import os
import re
import sys
import numpy as np

# fix unicode printing on windows
sys.stdout.reconfigure(encoding='utf-8')

INPUT = 'data/postmortems_combined.json'
OUTPUT = 'data/postmortems_scored.json'

PROMPT = """You are a senior software engineer evaluating a section of a document about a software failure.

Score this section on THREE independent axes, each from 0 to 4.

--- AXIS DEFINITIONS ---

TECHNICAL (0-4): Genuine explanation of the failure.
  4 = Root cause named specifically, failure mechanism explained, fix described technically,
      reproducible or preventable from this information alone.
  0 = No actionable technical content whatsoever.

PSEUDO_TECHNICAL (0-4): Technical-sounding language that does not actually explain anything.
  4 = Heavy use of jargon, acronyms, product names, internal tools, vague metrics, or process language that
      creates an impression of depth without explaining what broke or how the fix works.
  0 = No jargon masquerading as explanation.

POLITICAL (0-4): Business and stakeholder-driven language.
  4 = Pure PR — reassuring customers, protecting brand, attributing blame externally,
      vague commitments to improvement, SLA language, apologies without substance.
  0 = No business or PR framing at all.

--- SCORING CHECKLIST ---

Before scoring, ask:
  1. Is the ROOT CAUSE named specifically? (not 'an issue with our infrastructure')     → raises TECHNICAL
  2. Is the FAILURE MECHANISM explained? (why it failed, not just that it failed)        → raises TECHNICAL
  3. Is the FIX described technically? (what exactly changed, not 'we improved things')  → raises TECHNICAL
  4. Does the section use technical-sounding terms without explaining them?               → raises PSEUDO_TECHNICAL
  5. Does the section reassure stakeholders or protect the brand?                        → raises POLITICAL
  6. Does the section reference business impact, SLAs, or vague commitments?             → raises POLITICAL

The three scores are INDEPENDENT. A section can score high on multiple axes simultaneously.
Be decisive. Do not hedge toward the middle to be safe.

Respond in JSON only, exactly this format, no other text:
{{
  "technical": <integer 0-4>,
  "pseudo_technical": <integer 0-4>,
  "political": <integer 0-4>,
  "failure_type": "<two words max describing the failure category, e.g. 'memory leak', 'config error', 'network outage'>"
}}

DOCUMENT SECTION:
{text}"""


MAX_CHARS  = 12000   # docs under this are scored as a single chunk
CHUNK_SIZE = 4000    # chars per chunk for long docs
OVERLAP    = 500     # overlap between chunks to avoid cutting mid-sentence
MAX_CHUNKS = 20      # high limit — long docs like CSRB Log4j deserve full coverage;
                     # most short news articles are 1 chunk so this rarely fires
N_RUNS     = 3       # LLM runs per chunk


def get_id(record, index):
    if record.get('id'):
        return str(record['id'])
    if record.get('url'):
        return record['url']
    if record.get('incident_id'):
        return f"{record.get('source', 'unknown')}_{record['incident_id']}_{record.get('article_num', 0)}"
    return f"{record.get('source', 'unknown')}_{index}"


def make_chunks(text):
    """
    Short docs: returned as a single chunk.
    Long docs: split into overlapping windows of CHUNK_SIZE chars.
    Each chunk is labeled so the LLM knows it is reading a section,
    not the complete document.
    """
    if len(text) <= MAX_CHARS:
        return [text]

    chunks = []
    start  = 0
    total  = len(text)

    while start < total and len(chunks) < MAX_CHUNKS:
        end   = min(start + CHUNK_SIZE, total)
        chunk = text[start:end]
        label = f"[SECTION {len(chunks)+1} -- characters {start}-{end} of {total}]\n"
        chunks.append(label + chunk)
        start += CHUNK_SIZE - OVERLAP

    return chunks


def parse_json(raw):
    # strategy 1: direct parse
    try:
        return json.loads(raw)
    except:
        pass
    # strategy 2: find first { ... } block
    try:
        match = re.search(r'\{[^{}]+\}', raw, re.DOTALL)
        if match:
            return json.loads(match.group())
    except:
        pass
    # strategy 3: extract fields manually
    try:
        technical       = re.search(r'"?technical"?\s*:\s*(\d)',         raw)
        pseudo          = re.search(r'"?pseudo_technical"?\s*:\s*(\d)',   raw)
        political       = re.search(r'"?political"?\s*:\s*(\d)',          raw)
        failure_type    = re.search(r'"?failure_type"?\s*:\s*"([^"]*)"', raw)
        if technical and pseudo and political:
            return {
                'technical':        int(technical.group(1)),
                'pseudo_technical': int(pseudo.group(1)),
                'political':        int(political.group(1)),
                'failure_type':     failure_type.group(1).strip() if failure_type else '',
            }
    except:
        pass
    return None


def query_llm(chunk_text):
    """Send one chunk to the LLM. Returns parsed result dict or None."""
    try:
        r = requests.post('http://localhost:11434/api/generate', json={
            "model": "llama3.1:8b",
            "prompt": PROMPT.format(text=chunk_text),
            "stream": False
        }, timeout=120)
        raw    = r.json()['response']
        result = parse_json(raw)
        if result is None:
            print(f"    could not parse: {raw[:100]}")
        return result
    except Exception as e:
        print(f"    request error: {e}")
        return None


def score_document(text):
    """
    Score a full document over N_RUNS runs.

    Each run scores every chunk once, producing per-axis (mean, max) pairs across chunks.
    Those pairs are then averaged across the N_RUNS runs.

    Returns a dict with per-axis mean and max:
      technical           — mean of per-run chunk-means for TECHNICAL
      technical_max       — mean of per-run chunk-maxes for TECHNICAL
      pseudo_technical    — mean of per-run chunk-means for PSEUDO_TECHNICAL
      pseudo_technical_max
      political           — mean of per-run chunk-means for POLITICAL
      political_max
      n_chunks            — number of chunks the document was split into
      technical_std       — std of per-run technical means (run-to-run consistency)
      failure_type        — failure category from the highest-technical chunk in the last run

    For single-chunk docs mean == max for all axes.
    """
    chunks   = make_chunks(text)
    n_chunks = len(chunks)

    AXES = ('technical', 'pseudo_technical', 'political')

    run_means      = {a: [] for a in AXES}
    run_maxes      = {a: [] for a in AXES}
    run_failure_types = []

    for _ in range(N_RUNS):
        chunk_scores       = {a: [] for a in AXES}
        chunk_failure_types = []

        for chunk in chunks:
            result = query_llm(chunk)
            if result:
                for a in AXES:
                    chunk_scores[a].append(result.get(a, 0))
                chunk_failure_types.append(result.get('failure_type', ''))

        if chunk_scores['technical']:
            for a in AXES:
                run_means[a].append(float(np.mean(chunk_scores[a])))
                run_maxes[a].append(float(np.max(chunk_scores[a])))
            # failure_type from the highest-technical chunk this run
            max_idx = int(np.argmax(chunk_scores['technical']))
            run_failure_types.append(chunk_failure_types[max_idx] if chunk_failure_types else '')

    if not run_means['technical']:
        return None

    return {
        'technical':            round(float(np.mean(run_means['technical'])),            2),
        'technical_max':        round(float(np.mean(run_maxes['technical'])),            2),
        'pseudo_technical':     round(float(np.mean(run_means['pseudo_technical'])),     2),
        'pseudo_technical_max': round(float(np.mean(run_maxes['pseudo_technical'])),     2),
        'political':            round(float(np.mean(run_means['political'])),            2),
        'political_max':        round(float(np.mean(run_maxes['political'])),            2),
        'n_chunks':             n_chunks,
        'technical_std':        round(float(np.std(run_means['technical'])),             2),
        'failure_type':         run_failure_types[-1] if run_failure_types else '',
    }


def safe(val, width=None):
    """Return a printable string from a value that may be None."""
    s = str(val) if val is not None else 'unknown'
    return s[:width] if width else s


def main():
    with open(INPUT, encoding='utf-8') as f:
        data = json.load(f)

    for i, record in enumerate(data):
        if not record.get('id'):
            record['id'] = get_id(record, i)

    if os.path.exists(OUTPUT):
        with open(OUTPUT, encoding='utf-8') as f:
            scored = json.load(f)
        for i, record in enumerate(scored):
            if not record.get('id'):
                record['id'] = get_id(record, i)
        done_ids = {r['id'] for r in scored if r.get('technical') is not None}
        print(f"resuming: {len(done_ids)} already scored, {len(data) - len(done_ids)} remaining")
    else:
        scored   = []
        done_ids = set()

    to_score = [r for r in data if r['id'] not in done_ids]
    scored   = [r for r in scored if r.get('technical') is not None]

    print(f"scoring {len(to_score)} documents...")

    for i, record in enumerate(to_score):
        company = safe(record.get('company'), 30)
        title   = safe(record.get('title'),   50)
        source  = safe(record.get('source'))
        if not record.get('text'):
            print(f"    skipping -- no text field")
            continue

        n_chunks = len(make_chunks(record['text']))
        chunks_label = f"{n_chunks} chunk{'s' if n_chunks != 1 else ''}"
        print(f"[{i+1}/{len(to_score)}] {company} -- {title} [{chunks_label}]")

        result = score_document(record['text'])

        if result:
            record['technical']            = result['technical']
            record['technical_max']        = result['technical_max']
            record['pseudo_technical']     = result['pseudo_technical']
            record['pseudo_technical_max'] = result['pseudo_technical_max']
            record['political']            = result['political']
            record['political_max']        = result['political_max']
            record['n_chunks']             = result['n_chunks']
            record['technical_std']        = result['technical_std']
            record['failure_type']         = result['failure_type']
            reasoning = safe(record['failure_type'], 80)
            if n_chunks > 1:
                print(f"    T={record['technical']:.2f}(max {record['technical_max']:.2f})"
                      f"  P={record['pseudo_technical']:.2f}(max {record['pseudo_technical_max']:.2f})"
                      f"  POL={record['political']:.2f}(max {record['political_max']:.2f})"
                      f"  std={record['technical_std']:.2f}  src={source}")
            else:
                print(f"    T={record['technical']:.2f}"
                      f"  P={record['pseudo_technical']:.2f}"
                      f"  POL={record['political']:.2f}"
                      f"  std={record['technical_std']:.2f}  src={source}")
            print(f"    {reasoning}")
            scored.append(record)
        else:
            print(f"    failed, skipping")

        if (i + 1) % 10 == 0:
            with open(OUTPUT, 'w', encoding='utf-8') as f:
                json.dump(scored, f, indent=2)

    # final save
    with open(OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(scored, f, indent=2)

    print(f"\n=== DONE ===")
    print(f"scored: {len(scored)}/{len(data)}")

    if scored:
        n = len(scored)
        avg_t   = sum(r['technical']            for r in scored) / n
        avg_pt  = sum(r['pseudo_technical']     for r in scored) / n
        avg_pol = sum(r['political']            for r in scored) / n
        print(f"avg  technical={avg_t:.2f}/4"
              f"  pseudo_technical={avg_pt:.2f}/4"
              f"  political={avg_pol:.2f}/4")

        from collections import defaultdict
        by_source = defaultdict(list)
        for r in scored:
            by_source[r.get('source', 'unknown')].append(
                (r['technical'], r['pseudo_technical'], r['political'])
            )
        print("\nby source:")
        for src, triples in sorted(by_source.items()):
            ts   = [t  for t, _, _ in triples]
            pts  = [pt for _, pt, _ in triples]
            pols = [p  for _, _, p  in triples]
            print(f"  {src:15s}  n={len(ts):4d}"
                  f"  T={sum(ts)/len(ts):.2f}"
                  f"  PT={sum(pts)/len(pts):.2f}"
                  f"  POL={sum(pols)/len(pols):.2f}")


if __name__ == '__main__':
    main()