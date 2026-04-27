"""
predictllm.py

Scores a text file using the same LLM prompt as scorer.py.
Runs N_RUNS times and reports mean, std, and per-run breakdown.

For documents over MAX_CHARS, splits into overlapping chunks and
averages scores across all chunks per run, then averages across runs.

Usage:
    python predictllm.py <textfile>
    python predictllm.py <textfile> --runs 5
"""

import requests
import json
import re
import sys
import argparse
import numpy as np

from aggregate_unified import agg_oneaxis

sys.stdout.reconfigure(encoding='utf-8')

OLLAMA_URL = 'http://localhost:11434/api/generate'
MODEL      = 'llama3.1:8b'
N_RUNS     = 3

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

MAX_CHARS  = 12000   # anything under this is sent as-is
CHUNK_SIZE = 4000    # chars per chunk for long docs
OVERLAP    = 500     # overlap between chunks to avoid cutting mid-sentence
MAX_CHUNKS = 8       # cap — avoids runaway on very long docs


# ---------------------------------------------------------------------------
# Text handling
# ---------------------------------------------------------------------------

def make_chunks(text, max_chunks=MAX_CHUNKS):
    """
    Short docs: returned as a single chunk.
    Long docs: split into overlapping windows of CHUNK_SIZE chars.
    Each chunk is labeled so the LLM knows it's reading a section.
    """
    if len(text) <= MAX_CHARS:
        return [text]

    chunks = []
    start  = 0
    total  = len(text)

    while start < total and len(chunks) < max_chunks:
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

def render_bar(score):
    filled = max(0, min(8, int(round(score))))
    bar    = '#' * filled + '.' * (8 - filled)
    if score <= 2.5:
        label = 'POLITICAL'
    elif score >= 5.5:
        label = 'TECHNICAL'
    else:
        label = 'MIXED'
    return f"[{bar}]  {score:.2f}/8  ({label})"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Score a document with the LLM')
    parser.add_argument('textfile', help='Path to text file')
    parser.add_argument('--runs', type=int, default=N_RUNS,
                        help=f'Number of LLM runs to average (default: {N_RUNS})')
    parser.add_argument('--max-chunks', type=int, default=MAX_CHUNKS,
                        help=f'Max chunks for long docs (default: {MAX_CHUNKS})')
    args = parser.parse_args()

    try:
        with open(args.textfile, encoding='utf-8', errors='replace') as f:
            text = f.read()
    except FileNotFoundError:
        print(f"File not found: {args.textfile}")
        sys.exit(1)

    chunks     = make_chunks(text, max_chunks=args.max_chunks)
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

    run_scores     = []   # mean score per run
    run_peaks      = []   # peak chunk score per run (for score_max)
    all_reasonings = []
    all_chunk_data = []   # for detailed printout

    for run in range(args.runs):
        print(f"--- run {run+1}/{args.runs} ---")
        chunk_scores     = []
        chunk_reasonings = []

        for j, chunk in enumerate(chunks):
            tag = f"chunk {j+1}/{len(chunks)}" if is_chunked else "full doc"
            print(f"  {tag}...", end=' ', flush=True)
            result = query(chunk)
            if result:
                s = result['score']
                r = result.get('reasoning', '')
                chunk_scores.append(s)
                chunk_reasonings.append(r)
                print(f"score={s}  \"{r[:70]}\"")
            else:
                print("failed")

        if chunk_scores:
            run_mean = round(float(np.mean(chunk_scores)), 2)
            run_scores.append(run_mean)
            run_peaks.append(float(max(chunk_scores)))
            all_reasonings.append(chunk_reasonings[-1])
            all_chunk_data.append(chunk_scores)
            if is_chunked:
                print(f"  run {run+1} mean across {len(chunk_scores)} chunks: {run_mean}"
                      f"  peak={max(chunk_scores)}")
        else:
            print(f"  run {run+1} all chunks failed")

    if not run_scores:
        print("\nAll runs failed. Is Ollama running?")
        sys.exit(1)

    mean_score = round(float(np.mean(run_scores)), 2)
    std_score  = round(float(np.std(run_scores)),  2)
    score_max  = round(float(np.mean(run_peaks)),  2)  # mean of per-run peaks

    record = {"score": mean_score, "score_max": score_max}
    agg    = round(agg_oneaxis(record), 2)

    print(f"\n{'='*60}")
    print(f"RESULT:  {render_bar(mean_score)}")
    print(f"std dev: {std_score:.2f}  (across {len(run_scores)} runs)")
    if is_chunked:
        flat = [s for run in all_chunk_data for s in run]
        print(f"chunks:  {len(flat)} total scores averaged  "
              f"(min={min(flat)}, max={max(flat)})")
        print(f"score_max (mean of per-run peaks): {score_max:.2f}")
    print(f"{'='*60}")
    print(f"\nAGG SCORE:  {render_bar(agg)}")
    gap = max(0.0, score_max - mean_score)
    print(f"  formula:  score_max={score_max:.2f}  score_mean={mean_score:.2f}"
          f"  gap={gap:.2f}  -> agg={agg:.2f}")
    print(f"  (gap_penalty=0.5, gap_tolerance=1.5)")
    print(f"{'='*60}")

    # variance interpretation
    if std_score >= 2.0:
        print(f"\n[!] High variance (std={std_score:.2f}) -- document is genuinely")
        print(f"    ambiguous. LLM itself is uncertain about classification.")
    elif std_score >= 1.0:
        print(f"\n[~] Moderate variance (std={std_score:.2f}) -- score is")
        print(f"    reasonably stable but document has mixed signals.")
    else:
        print(f"\n[ok] Low variance (std={std_score:.2f}) -- score is stable.")

    # per-run summary
    print(f"\nPer-run scores:")
    for i, (s, r) in enumerate(zip(run_scores, all_reasonings)):
        print(f"  run {i+1}  score={s}  \"{r[:80]}\"")

    # chunk score breakdown for long docs
    if is_chunked:
        print(f"\nChunk score breakdown:")
        for run_i, chunk_scores in enumerate(all_chunk_data):
            scores_str = '  '.join(f"c{j+1}={s}" for j, s in enumerate(chunk_scores))
            print(f"  run {run_i+1}: {scores_str}  -> mean={run_scores[run_i]}")

    print(f"\n  0 = completely political  (vague reassurance, no mechanism)")
    print(f"  8 = completely technical  (root cause, mechanism, fix, reproducible)")


if __name__ == '__main__':
    main()