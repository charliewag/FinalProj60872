"""
predictllm3axis.py

Scores a text file on three independent axes using the same LLM prompt as
scorer.py:  TECHNICAL (0-4), PSEUDO_TECHNICAL (0-4), POLITICAL (0-4).
Also reports failure_type from the most technically informative chunk.

Runs N_RUNS times and reports mean, std, and per-run breakdown for each axis.
For long documents, splits into overlapping chunks and averages across chunks
within each run before averaging across runs.

Usage:
    python predictllm3axis.py <textfile>
    python predictllm3axis.py <textfile> --runs 5
"""

import requests
import json
import re
import sys
import argparse
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

OLLAMA_URL = 'http://localhost:11434/api/generate'
MODEL      = 'llama3.1:8b'
N_RUNS     = 3

AXES = ('technical', 'pseudo_technical', 'political')

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

MAX_CHARS  = 12000
CHUNK_SIZE = 4000
OVERLAP    = 500
MAX_CHUNKS = 8


# ---------------------------------------------------------------------------
# Text handling
# ---------------------------------------------------------------------------

def make_chunks(text):
    if len(text) <= MAX_CHARS:
        return [text]

    chunks = []
    start  = 0
    total  = len(text)

    while start < total and len(chunks) < MAX_CHUNKS:
        end   = min(start + CHUNK_SIZE, total)
        chunk = text[start:end]
        label = f"[SECTION {len(chunks)+1} — characters {start}-{end} of {total}]\n"
        chunks.append(label + chunk)
        start += CHUNK_SIZE - OVERLAP

    return chunks


# ---------------------------------------------------------------------------
# LLM query
# ---------------------------------------------------------------------------

def parse_json(raw):
    try:
        return json.loads(raw)
    except:
        pass
    try:
        match = re.search(r'\{[^{}]+\}', raw, re.DOTALL)
        if match:
            return json.loads(match.group())
    except:
        pass
    try:
        technical    = re.search(r'"?technical"?\s*:\s*(\d)',         raw)
        pseudo       = re.search(r'"?pseudo_technical"?\s*:\s*(\d)',   raw)
        political    = re.search(r'"?political"?\s*:\s*(\d)',          raw)
        failure_type = re.search(r'"?failure_type"?\s*:\s*"([^"]*)"', raw)
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


def query(text):
    try:
        r = requests.post(OLLAMA_URL, json={
            "model":  MODEL,
            "prompt": PROMPT.format(text=text),
            "stream": False,
        }, timeout=120)
        raw    = r.json()['response']
        result = parse_json(raw)
        if result is None:
            print(f"  [warn] could not parse: {raw[:120]}")
        return result
    except Exception as e:
        print(f"  [error] {e}")
        return None


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

def render_bar(score, width=4):
    filled = max(0, min(width, int(round(score))))
    bar    = '#' * filled + '.' * (width - filled)
    if score <= 1.0:
        label = 'LOW'
    elif score >= 3.0:
        label = 'HIGH'
    else:
        label = 'MED'
    return f"[{bar}]  {score:.2f}/4  ({label})"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Score a document on three axes with the LLM')
    parser.add_argument('textfile', help='Path to text file')
    parser.add_argument('--runs', type=int, default=N_RUNS,
                        help=f'Number of LLM runs to average (default: {N_RUNS})')
    args = parser.parse_args()

    try:
        with open(args.textfile, encoding='utf-8', errors='replace') as f:
            text = f.read()
    except FileNotFoundError:
        print(f"File not found: {args.textfile}")
        sys.exit(1)

    chunks     = make_chunks(text)
    is_chunked = len(chunks) > 1

    print(f"\nfile:    {args.textfile}")
    print(f"length:  {len(text):,} chars")
    if is_chunked:
        print(f"chunks:  {len(chunks)}  x ~{CHUNK_SIZE} chars  "
              f"(overlap={OVERLAP}, max={MAX_CHUNKS})")
    else:
        print(f"chunks:  1  (fits within {MAX_CHARS} char limit)")
    print(f"model:   {MODEL}")
    print(f"runs:    {args.runs}")
    print()

    # run_means[axis] = list of per-run chunk-mean scores
    run_means        = {a: [] for a in AXES}
    run_failure_types = []
    all_chunk_data   = []   # run -> list of {axis: score}

    for run in range(args.runs):
        print(f"--- run {run+1}/{args.runs} ---")
        chunk_scores       = {a: [] for a in AXES}
        chunk_failure_types = []

        for j, chunk in enumerate(chunks):
            tag = f"chunk {j+1}/{len(chunks)}" if is_chunked else "full doc"
            print(f"  {tag}...", end=' ', flush=True)
            result = query(chunk)
            if result:
                for a in AXES:
                    chunk_scores[a].append(result.get(a, 0))
                chunk_failure_types.append(result.get('failure_type', ''))
                t  = result.get('technical', 0)
                pt = result.get('pseudo_technical', 0)
                po = result.get('political', 0)
                ft = result.get('failure_type', '')
                print(f"T={t} PT={pt} POL={po}  \"{ft}\"")
            else:
                print("failed")

        if chunk_scores['technical']:
            for a in AXES:
                run_means[a].append(float(np.mean(chunk_scores[a])))
            # failure_type from highest-technical chunk this run
            max_idx = int(np.argmax(chunk_scores['technical']))
            run_failure_types.append(chunk_failure_types[max_idx] if chunk_failure_types else '')
            all_chunk_data.append({a: chunk_scores[a][:] for a in AXES})
            if is_chunked:
                for a in AXES:
                    print(f"  run {run+1} {a} mean: {run_means[a][-1]:.2f}")
        else:
            print(f"  run {run+1} all chunks failed")

    if not run_means['technical']:
        print("\nAll runs failed. Is Ollama running?")
        sys.exit(1)

    final = {a: round(float(np.mean(run_means[a])), 2) for a in AXES}
    stds  = {a: round(float(np.std(run_means[a])),  2) for a in AXES}
    failure_type = run_failure_types[-1] if run_failure_types else ''

    print(f"\n{'='*60}")
    print(f"RESULTS  (mean over {len(run_means['technical'])} runs)")
    print(f"{'='*60}")
    for a in AXES:
        print(f"  {a:<20s}  {render_bar(final[a])}  std={stds[a]:.2f}")
    print(f"  failure type:         {failure_type}")
    print(f"{'='*60}")

    if is_chunked:
        flat_t = [s for run in all_chunk_data for s in run['technical']]
        print(f"  chunks: {len(flat_t)} total scores  "
              f"(technical min={min(flat_t):.1f}, max={max(flat_t):.1f})")

    # variance interpretation (based on technical axis)
    std_t = stds['technical']
    if std_t >= 1.5:
        print(f"\n[!] High variance on technical (std={std_t:.2f}) — document is genuinely ambiguous.")
    elif std_t >= 0.75:
        print(f"\n[~] Moderate variance on technical (std={std_t:.2f}) — mixed signals.")
    else:
        print(f"\n[ok] Low variance on technical (std={std_t:.2f}) — score is stable.")

    # per-run summary
    print(f"\nPer-run scores:")
    for i in range(len(run_means['technical'])):
        scores_str = '  '.join(f"{a[:4]}={run_means[a][i]:.2f}" for a in AXES)
        ft = run_failure_types[i] if i < len(run_failure_types) else ''
        print(f"  run {i+1}  {scores_str}  \"{ft}\"")

    # chunk breakdown for long docs
    if is_chunked:
        print(f"\nChunk score breakdown:")
        for run_i, chunk_data in enumerate(all_chunk_data):
            for a in AXES:
                scores_str = '  '.join(f"c{j+1}={s}" for j, s in enumerate(chunk_data[a]))
                print(f"  run {run_i+1} {a:<20s}: {scores_str}")

    print(f"\n  0 = none of this axis  4 = all of this axis")


if __name__ == '__main__':
    main()
