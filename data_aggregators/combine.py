import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open('data/postmortems_handedited.json', encoding='utf-8') as f:
    hand = json.load(f)

with open('data/fail_news_articles.json', encoding='utf-8') as f:
    news = json.load(f)

combined = list(hand)

added = 0
for r in news:
    if len(r.get('text', '')) > 500:
        combined.append(r)
        added += 1

with open('data/postmortems_combined.json', 'w', encoding='utf-8') as f:
    json.dump(combined, f, indent=2)

print(f"hand edited:  {len(hand)}")
print(f"news articles added: {added}")
print(f"total combined: {len(combined)}")