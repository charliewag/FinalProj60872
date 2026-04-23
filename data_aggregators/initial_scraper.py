import requests
import urllib3
from bs4 import BeautifulSoup
import json, time, re, os, sys
from collections import Counter

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
sys.stdout.reconfigure(encoding='utf-8')

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate',
    'Connection': 'keep-alive',
}
ICCO_DIR = 'icco_postmortems/data'
OUTPUT = 'data/postmortems_raw.json'
FAILED_OUTPUT = 'data/failed_urls.txt'

def parse_markdown(filepath):
    with open(filepath, encoding='utf-8') as f:
        content = f.read()

    match = re.match(r'^---\n(.*?)\n---\n?(.*)', content, re.DOTALL)
    if not match:
        return None

    frontmatter = match.group(1)
    summary = match.group(2).strip()

    def get_field(name):
        m = re.search(rf'^{name}:\s*(.+)$', frontmatter, re.MULTILINE)
        return m.group(1).strip().strip('"') if m else None

    def get_list_field(name):
        m = re.search(rf'^{name}:\n((?:- .+\n?)+)', frontmatter, re.MULTILINE)
        if m:
            return [line.strip('- ').strip() for line in m.group(1).strip().split('\n')]
        return []

    url = get_field('url')
    if not url:
        return None

    return {
        'uuid': get_field('uuid'),
        'url': url,
        'company': get_field('company') or 'Unknown',
        'categories': get_list_field('categories'),
        'summary': summary
    }


def fetch_text(url):
    if not url or url.lower().endswith('.pdf'):
        return None, None, 'pdf'
    try:
        r = requests.get(url, headers=HEADERS, timeout=15, verify=False, allow_redirects=True)
        if r.status_code != 200:
            return None, None, f'http_{r.status_code}'
        if 'application/pdf' in r.headers.get('Content-Type', ''):
            return None, None, 'pdf'

        soup = BeautifulSoup(r.text, 'html.parser')

        h1 = soup.find('h1')
        title = h1.get_text().strip() if h1 else None
        if not title:
            t = soup.find('title')
            if t:
                title = re.split(r'\s*[|\-]\s*', t.get_text())[0].strip()

        for tag in soup(['nav', 'footer', 'script', 'style', 'header',
                         'aside', 'form', 'button', 'iframe', 'img',
                         'figure', 'figcaption', 'noscript']):
            tag.decompose()

        junk_patterns = ['cookie', 'banner', 'sidebar', 'related', 'subscribe',
                         'newsletter', 'comment', 'share', 'social', 'ad',
                         'popup', 'modal', 'menu', 'toc', 'table-of-contents']
        for pattern in junk_patterns:
            for el in soup.find_all(attrs={'class': re.compile(pattern, re.I)}):
                el.decompose()
            for el in soup.find_all(attrs={'id': re.compile(pattern, re.I)}):
                el.decompose()

        content = (
            soup.find('article') or
            soup.find(attrs={'class': re.compile(r'post[-_]?(body|content|text)', re.I)}) or
            soup.find(attrs={'class': re.compile(r'article[-_]?(body|content|text)', re.I)}) or
            soup.find('main') or
            soup.find(attrs={'role': 'main'}) or
            soup.find(attrs={'id': re.compile(r'content|main|post|article', re.I)}) or
            soup.find('body')
        )

        if not content:
            return None, None, 'no_content'

        paragraphs = []
        for p in content.find_all(['p', 'li', 'h2', 'h3', 'h4', 'pre', 'code']):
            text = p.get_text().strip()
            if len(text) > 40:
                paragraphs.append(text)

        text = ' '.join(paragraphs)

        if len(text) < 300:
            return None, None, 'too_short'

        return title, text[:10000], None

    except requests.exceptions.ConnectionError:
        return None, None, 'connection_error'
    except requests.exceptions.Timeout:
        return None, None, 'timeout'
    except Exception as e:
        return None, None, str(e)[:50]


def main():
    os.makedirs('data', exist_ok=True)

    files = [f for f in os.listdir(ICCO_DIR) if f.endswith('.md')]
    print(f"found {len(files)} markdown files")

    records = []
    failed = []
    seen_urls = set()

    for i, fname in enumerate(files):
        meta = parse_markdown(os.path.join(ICCO_DIR, fname))
        if not meta:
            continue

        url = meta['url']
        if url in seen_urls:
            continue
        seen_urls.add(url)

        company = meta['company'].encode('ascii', 'replace').decode('ascii')
        print(f"[{i+1}/{len(files)}] {company[:30]} -- {url[:60]}")

        title, text, error = fetch_text(url)

        if not text:
            print(f"  FAILED ({error}): {url}")
            failed.append({
                'url': url,
                'company': meta['company'],
                'reason': error,
                'categories': meta['categories'],
                'summary': meta['summary']
            })
            time.sleep(0.3)
            continue

        records.append({
            'source': 'icco',
            'company': meta['company'],
            'title': title or url,
            'url': url,
            'date': None,
            'categories': meta['categories'],
            'summary': meta['summary'],
            'text': text
        })

        title_clean = (title or url).encode('ascii', 'replace').decode('ascii')
        print(f"  ok: {title_clean[:70]}")
        time.sleep(0.5)

        if len(records) % 25 == 0:
            with open(OUTPUT, 'w', encoding='utf-8') as f:
                json.dump(records, f, indent=2)
            print(f"  -- checkpoint: {len(records)} saved --")

    # save successful records
    with open(OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=2)

    # save failed urls to text file for manual review
    with open(FAILED_OUTPUT, 'w', encoding='utf-8') as f:
        f.write(f"FAILED URLS ({len(failed)} total)\n")
        f.write("=" * 60 + "\n\n")
        for item in failed:
            f.write(f"company:    {item['company']}\n")
            f.write(f"url:        {item['url']}\n")
            f.write(f"reason:     {item['reason']}\n")
            f.write(f"categories: {', '.join(item['categories'])}\n")
            f.write(f"summary:    {item['summary']}\n")
            f.write("\n")

    print(f"\n=== DONE ===")
    print(f"collected: {len(records)}")
    print(f"failed:    {len(failed)} -- see data/failed_urls.txt")

    counts = Counter(c for r in records for c in r['categories'])
    print("\ncategory breakdown:")
    for cat, n in counts.most_common():
        print(f"  {cat}: {n}")


if __name__ == '__main__':
    main()