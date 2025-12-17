import requests

# Replace this with one real encoded URL from your GitHub logs
encoded = "https://news.google.com/rss/articles/CBMiqgFBVV95cUxPZ2loTVRiOXJvMXZ6WGlqOVFMZGFFVXZsOTlpVFlkN2t3Zno..."

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Referer': 'https://news.google.com/',
    'DNT': '1',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
}

print("Testing redirect...\n")

try:
    r = requests.get(encoded, headers=headers, allow_redirects=True, timeout=15)
    print("Final URL ->", r.url)
    print("Status code:", r.status_code)
except Exception as e:
    print("Failed:", e)