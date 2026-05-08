# FinalProj60872

Postmortems are documents that identify and analyze the failures of a specific system. Learning from failure is not only necessary for the improvement of these systems but also fundamental to the progression of technology in society. However, as we know failure is highly stigmatized in our profit and image driven economy. Those that publish postmortems, whether that be the companies of the failure or outside sources looking in, have alternative motives that are driven by goals other than building a knowledge base of engineering faults. This technical writing plays a key role in helping other engineers avoid the same mistakes as others, especially in a world where technology is so ingrained that failures have real consequences in terms of the economy, society, environment, public health, etc. With these consequences at stake, it is hard to say what a perfect postmortem looks like but it does show there is value in differentiating between technical writing that should be kept in mind by engineers in similar fields and political writing that is essential to business goals. It is in this sense where evaluating postmortems on a technical-political axis becomes valuable to help readers understand the true context and purpose of these failure based excerpts.

Evaluating postmortems in this sense is almost more philosophical than mathematical. What makes writing technical? How do these factors play into a scoring system for a postmortem in its entirety? I first created an LLM based tool that can grade postmortems on a technical and political scale and run it on postmortems scraped from the internet ranging from completely technical debriefs to political news articles informing customers on business operations. We evaluate this body of work to see the extent to which a technical body of failure related work really exists. I then try to find simplicities in the scored dataset using simpler NLP models to identify keywords and score postmortems without a complex LLM. As readers having a tool to grade a postmortem on a scale may be useful but being able to sight flags while reading can help prime the brain to recognize the way in which information is being framed. I then suggest pathways forward using this tool to create a real engineering failure database. Acknowledgement: I did use AI tools to help write and debug code where all of the ideas and directions of the project were mine.

---

## How to Run

### Prerequisites

- Python 3.9+
- [Ollama](https://ollama.com/) running locally with `llama3.1:8b` pulled (`ollama pull llama3.1:8b`) which is required for the LLM scorer and predict scripts
- Install dependencies: `pip install requests beautifulsoup4 numpy scikit-learn scipy matplotlib`

---

### Pipeline

#### Step 1 - Scrape postmortems from the ICCO dataset (superset of danluu dataset)
Both are community sourced databases of postmortems.

Pulled from: https://github.com/icco/postmortems/tree/main

Reads markdown files from `icco_postmortems/data/`, fetches the live URLs, and saves scraped text.

```
python data_aggregators/initial_scraper.py
```

Output: `data/postmortems_raw.json`, `data/failed_urls.txt`

I went through failed URLs manually to ensure they were actually unavailable.
---

#### Step 2 - Import FAIL dataset news articles

Reads plain-text articles from the FAIL dataset directory (pulled from respective research paper in references) and converts them to the shared record format.

```
python data_aggregators/importfail.py
```

Output: `data/fail_news_articles.json`

---

#### Step 3 - Combine datasets

Merges hand-edited postmortems (`data/postmortems_handedited.json`) with the FAIL news articles into one file.

```
python data_aggregators/combine.py
```

Output: `data/postmortems_combined.json`

---

#### Step 4 - Score documents with the LLM

Requires Ollama running. Supports resuming where re-running skips already-scored documents.

**Three-axis scorer** (used for model training and figures):
Scores each document on TECHNICAL (0–4), PSEUDO_TECHNICAL (0–4), and POLITICAL (0–4).

```
python scorer_threeaxis.py
```

**One-axis scorer** (original, single 0–8 political-to-technical scale):

```
python scorer_oneaxis.py
```

Both read `data/postmortems_combined.json` and write to `data/postmortems_scored.json`.

---

#### Step 5 - Aggregate scores

Converts raw LLM scores into a final `agg_score` (0–8) using a weighted formula that rewards peak technical content and penalises political framing.

```
python aggregate_unified.py --mode threeaxis
python aggregate_unified.py --mode oneaxis
```

Optional weight flags for three-axis mode: `--w_tm 2.5 --w_t 0.5 --w_pt 0.5 --w_pol 1.5`
Optional gap flags for one-axis mode: `--gap_penalty 0.5 --gap_tolerance 1.5`

Outputs: `data/postmortems_agg_threeaxis.json`, `data/postmortems_agg_oneaxis.json`

---

#### Step 6 - Train the NLP classifier

Trains a TF-L2 Norm + Ridge and TF-L2 Norm Centroid LDA-like binary classifiers (hand-edited only, and combined) on the aggregated scores. Excludes the ambiguous middle zone (3.0–5.5).

```
python train1axis.py
```

Outputs: `data/vectorizer_*.pkl`, `data/ridge_*.pkl`, `data/model_*.json`

---

#### Step 7 - Generate analysis figures

Produces all five figures and saves them to `analysis/figures/`.

```
python analysis/make_figures.py
```

Reads: `data/postmortems_agg_oneaxis.json`, `data/postmortems_agg_threeaxis.json`, `data/model_combined.json`

---

### Scoring a new document

To score a single `.txt` file on the political-technical axis:

**LLM one-axis** (0–8, requires Ollama):
```
python predictllm1axis.py path/to/document.txt
python predictllm1axis.py path/to/document.txt --runs 3
```

**LLM three-axis** (TECHNICAL / PSEUDO_TECHNICAL / POLITICAL, requires Ollama):
```
python predictllm3axis.py path/to/document.txt
python predictllm3axis.py path/to/document.txt --runs 3
```

**Centroid NLP model** (no LLM required, run `train1axis.py` first):
```
python predict1axis.py path/to/document.txt
```
