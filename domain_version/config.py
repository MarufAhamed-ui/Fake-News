"""Configuration copied with the self-contained HTML deployment folder."""

CONFIDENCE_THRESHOLD = 0.60
RANDOM_STATE = 42
MAX_TEXT_CHARS = 25_000
RSS_CACHE_SECONDS = 30 * 60
ARTICLES_PER_FEED = 6

RSS_FEEDS = {
    "BBC News": "https://feeds.bbci.co.uk/news/rss.xml",
    "Google News": "https://news.google.com/rss?hl=en-US&gl=US&ceid=US:en",
}
