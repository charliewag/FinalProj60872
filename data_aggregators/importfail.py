import os
import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

STORIES_DIR = 'fail_dataset/PurdueDualityLab-ASE24-FAIL-FailureAnalysisInvestigationWithLLM-2095fb5/tests/ko_test/data/stories'
OUTPUT = 'data/fail_news_articles.json'

records = []

for incident_id in os.listdir(STORIES_DIR):
    incident_path = os.path.join(STORIES_DIR, incident_id)
    if not os.path.isdir(incident_path):
        continue

    for article_file in os.listdir(incident_path):
        if not article_file.endswith('.txt'):
            continue

        filepath = os.path.join(incident_path, article_file)
        try:
            with open(filepath, encoding='utf-8', errors='replace') as f:
                text = f.read().strip()
        except:
            continue

        if len(text) < 200:
            continue

        records.append({
            'source': 'fail_news',
            'company': '',
            'title': '',
            'url': '',
            'date': None,
            'incident_id': incident_id,
            'article_num': article_file.replace('.txt', ''),
            'text': text[:10000]
        })

with open(OUTPUT, 'w', encoding='utf-8') as f:
    json.dump(records, f, indent=2)

print(f"collected {len(records)} news articles from {len(os.listdir(STORIES_DIR))} incidents")