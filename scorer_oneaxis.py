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

PROMPT = """You are a senior software engineer evaluating a document about a software failure.

Score this document on a single axis from 0 to 8:
  0 = Completely POLITICAL
  8 = Completely TECHNICAL

--- SCALE DEFINITIONS ---

TECHNICAL (toward 8): The document genuinely explains the failure mechanism problem and solution in enough detail that an outside engineer could understand what broke, why it broke, and how the fix actually prevents recurrence. 

POLITICAL (toward 0): How much does this push the business agenda of the company, dont be fooled by technical-sounding language numbers, and jargon to reassure stakeholders WITHOUT actually explaining the failure mechanism. 

--- ANCHOR EXAMPLES ---

Score 0 example reasoning: A 0 would be reassuring users or stakeholders the problem won't happen again while only referencing internal tools and vague descriptions of real problems/solutions.

Score 8 example reasoning: An 8 looks like a government incident report or a detailed engineering postmortem with specific system behavior, root cause, and verified remediation with the sole goal of pushing the cutting edge of science/technology.

--- SCORING CHECKLIST (use this before deciding) ---

Ask yourself:
  1. Is the ROOT CAUSE named specifically? (not 'an issue with our infrastructure')
  2. Is the FAILURE MECHANISM explained? (why did it fail, not just that it failed)
  3. Is the FIX described technically? (what exactly changed, not 'we improved our systems')
  4. Could an outside engineer REPRODUCE or PREVENT this from the information given?

Each yes moves the score toward 8. Pseudo-technical language that mimics these without delivering them should LOWER the score, not raise it. Be decisive and confident. Do not hedge toward the middle to be safe.

Respond in JSON only, exactly this format, no other text:
{{
  "score": <integer 0-8>,
  "reasoning": "<one sentence max>"
}}

DOCUMENT TEXT:
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
        score  = re.search(r'"?score"?\s*:\s*(\d)', raw)
        reason = re.search(r'"?reasoning"?\s*:\s*"([^"]*)"', raw)
        if score:
            return {
                'score':     int(score.group(1)),
                'reasoning': reason.group(1).strip() if reason else ''
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
    Score a full document over N_RUNS document-level runs.

    Each run scores every chunk once, producing a (mean, max) pair across chunks.
    Those pairs are then averaged across the N_RUNS runs.

    Returns a dict with:
      score        — mean of per-run chunk-means (overall tone)
      score_max    — mean of per-run chunk-maxes (avg peak technical depth)
      n_chunks     — number of chunks the document was split into
      score_std    — std of per-run means (run-to-run consistency)
      reasoning    — LLM reasoning from the highest-scoring chunk in the last run

    For single-chunk docs score == score_max.
    """
    chunks   = make_chunks(text)
    n_chunks = len(chunks)

    run_means      = []
    run_maxes      = []
    run_reasonings = []

    for _ in range(N_RUNS):
        chunk_scores     = []
        chunk_reasonings = []

        for chunk in chunks:
            result = query_llm(chunk)
            if result:
                chunk_scores.append(result['score'])
                chunk_reasonings.append(result.get('reasoning', ''))

        if chunk_scores:
            run_means.append(float(np.mean(chunk_scores)))
            run_maxes.append(float(np.max(chunk_scores)))
            max_idx = int(np.argmax(chunk_scores))
            run_reasonings.append(chunk_reasonings[max_idx] if chunk_reasonings else '')

    if not run_means:
        return None

    return {
        'score':     round(float(np.mean(run_means)), 2),
        'score_max': round(float(np.mean(run_maxes)), 2),
        'n_chunks':  n_chunks,
        'score_std': round(float(np.std(run_means)),  2),
        'reasoning': run_reasonings[-1] if run_reasonings else '',
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
        done_ids = {r['id'] for r in scored if r.get('score') is not None}
        print(f"resuming: {len(done_ids)} already scored, {len(data) - len(done_ids)} remaining")
    else:
        scored   = []
        done_ids = set()

    to_score = [r for r in data if r['id'] not in done_ids]
    scored   = [r for r in scored if r.get('score') is not None]

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
            record['score']     = result['score']
            record['score_max'] = result['score_max']
            record['n_chunks']  = result['n_chunks']
            record['score_std'] = result['score_std']
            record['reasoning'] = result['reasoning']
            reasoning = safe(record['reasoning'], 80)
            if n_chunks > 1:
                print(f"    mean={record['score']:.2f}  max={record['score_max']:.2f}"
                      f"  std={record['score_std']:.2f}  src={source}")
            else:
                print(f"    score={record['score']:.2f}"
                      f"  std={record['score_std']:.2f}  src={source}")
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
        avg     = sum(r['score']                      for r in scored) / len(scored)
        avg_max = sum(r.get('score_max', r['score'])  for r in scored) / len(scored)
        print(f"avg mean score: {avg:.2f}/8   avg peak score: {avg_max:.2f}/8")

        from collections import defaultdict
        by_source = defaultdict(list)
        for r in scored:
            by_source[r.get('source', 'unknown')].append(
                (r['score'], r.get('score_max', r['score']))
            )
        print("\nby source:")
        for src, pairs in sorted(by_source.items()):
            means = [m for m, _ in pairs]
            peaks = [p for _, p in pairs]
            print(f"  {src:15s}  n={len(means):4d}"
                  f"  mean={sum(means)/len(means):.2f}"
                  f"  peak={sum(peaks)/len(peaks):.2f}")


if __name__ == '__main__':
    main()