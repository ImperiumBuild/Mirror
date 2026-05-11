# debug_reviews.py
from google_play_scraper import reviews, Sort

result, _ = reviews(
    "com.kudabank.app",   # Kuda's exact app ID
    lang="en",
    country="ng",
    sort=Sort.NEWEST,
    count=5,
)
print(f"Count: {len(result)}")
for r in result:
    print(r.get("content", "")[:80])