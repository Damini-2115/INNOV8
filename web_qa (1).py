from __future__ import annotations

"""
Trusted News Fetcher – India Edition
Fetches latest news from trusted RSS feeds on:
  - Misinformation / Fake News
  - Social Awareness / Cyber Safety
  - Government Schemes
  - Frauds & Scams
  - Bank Frauds / Banking Alerts
"""

import re
import time
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional, Dict
from urllib.parse import urlparse, urljoin
from email.utils import parsedate_to_datetime

import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ──────────────────────────── Constants ────────────────────────────

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)
REQUEST_TIMEOUT = 15
CACHE_TTL = 600  # 10 minutes

# ──────────────────────────── Trusted Domains ─────────────────────

TRUSTED_DOMAINS = {
    # Major Indian News
    "ndtv.com", "thehindu.com", "indianexpress.com", "livemint.com",
    "economictimes.indiatimes.com", "hindustantimes.com",
    "financialexpress.com", "theprint.in", "freepressjournal.in",
    "scroll.in", "thewire.in", "deccanherald.com", "tribuneindia.com",
    # Government
    "pib.gov.in", "mygov.in", "gov.in", "nic.in", "rbi.org.in",
    "sebi.gov.in", "mha.gov.in", "mohfw.gov.in", "pmindia.gov.in",
    # Fact-Checkers
    "boomlive.in", "altnews.in", "factly.in", "factchecker.in",
    "india.factcheck.afp.com", "vishvasnews.com",
    # International Trusted
    "bbc.com", "reuters.com", "apnews.com", "timesofindia.indiatimes.com",
}

# ──────────────────────────── RSS Feed Topics ─────────────────────

# Google News RSS – no API key needed
_GN = "https://news.google.com/rss/search?q={q}&hl=en-IN&gl=IN&ceid=IN:en"

TOPIC_FEEDS: Dict[str, List[str]] = {
    "Fraud Alert": [
        _GN.format(q="cyber+fraud+India"),
        _GN.format(q="online+scam+India+2025"),
        _GN.format(q="bank+fraud+India"),
        _GN.format(q="financial+fraud+India"),
        _GN.format(q="UPI+fraud+India"),
        _GN.format(q="phishing+scam+India"),
    ],
    "Banking": [
        _GN.format(q="RBI+alert+India"),
        _GN.format(q="bank+scheme+India+2025"),
        _GN.format(q="SBI+HDFC+ICICI+banking+alert"),
        _GN.format(q="loan+fraud+India"),
        _GN.format(q="bank+deposit+scheme+India"),
    ],
    "Government Scheme": [
        _GN.format(q="government+scheme+India+2025"),
        _GN.format(q="PM+Kisan+Ayushman+Bharat+update"),
        _GN.format(q="government+yojana+India"),
        _GN.format(q="scheme+fraud+India"),
        _GN.format(q="fake+government+scheme+India"),
    ],
    "Misinformation": [
        _GN.format(q="fake+news+India+fact+check"),
        _GN.format(q="misinformation+India"),
        _GN.format(q="rumour+India+busted"),
        _GN.format(q="fact+check+India+viral"),
        _GN.format(q="false+information+India"),
    ],
    "Social Awareness": [
        _GN.format(q="social+awareness+India+cyber+safety"),
        _GN.format(q="digital+literacy+India"),
        _GN.format(q="consumer+awareness+India"),
        _GN.format(q="cyber+crime+awareness+India"),
        _GN.format(q="online+safety+India"),
    ],
}

# ──────────────────────────── Data Class ──────────────────────────

@dataclass
class NewsItem:
    headline: str
    url: str
    source: str
    image_url: Optional[str] = None
    published_date: Optional[str] = None
    time_ago: str = ""
    summary: str = ""
    category: str = "General"
    trust_source: str = "Trusted Source"

# ──────────────────────────── In-Memory Cache ─────────────────────

_cache: Dict[str, object] = {}

def _cache_get(key: str):
    entry = _cache.get(key)
    if entry and (time.time() - entry["ts"]) < CACHE_TTL:
        return entry["data"]
    return None

def _cache_set(key: str, data):
    _cache[key] = {"ts": time.time(), "data": data}

# ──────────────────────────── Helpers ─────────────────────────────

def _time_ago(pub_date_str: Optional[str]) -> str:
    """Convert ISO or RFC-2822 date string to '2 days ago' format."""
    if not pub_date_str:
        return ""
    try:
        # Try RFC-2822 (RSS format)
        dt = parsedate_to_datetime(pub_date_str)
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    except Exception:
        try:
            dt = datetime.fromisoformat(pub_date_str.replace("Z", "+00:00"))
            dt = dt.replace(tzinfo=None)
        except Exception:
            return ""

    now = datetime.utcnow()
    diff = now - dt
    secs = int(diff.total_seconds())
    if secs < 0:
        return "just now"
    if secs < 3600:
        m = secs // 60
        return f"{m} min ago" if m > 1 else "just now"
    if secs < 86400:
        h = secs // 3600
        return f"{h} hour{'s' if h > 1 else ''} ago"
    d = secs // 86400
    return f"{d} day{'s' if d > 1 else ''} ago"


def _is_trusted(url: str) -> bool:
    netloc = urlparse(url).netloc.lower().lstrip("www.")
    return any(td in netloc for td in TRUSTED_DOMAINS)


def _clean_text(text: str) -> str:
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def _get_trust_label(url: str) -> str:
    netloc = urlparse(url).netloc.lower()
    if any(d in netloc for d in ["gov.in", "nic.in", "rbi.org.in", "sebi.gov.in"]):
        return "Government Official"
    if "pib.gov.in" in netloc:
        return "Press Information Bureau"
    if any(d in netloc for d in ["boomlive", "altnews", "factly", "factchecker", "vishvas"]):
        return "Fact Checker"
    if any(d in netloc for d in ["thehindu", "indianexpress", "ndtv", "livemint", "economictimes"]):
        return "Verified News"
    return "Trusted Source"

# ──────────────────────────── Image Extraction ────────────────────

def _fetch_article_image(url: str) -> Optional[str]:
    """Fetch the OG/Twitter or first article image from the article page."""
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=8,
            allow_redirects=True,
        )
        soup = BeautifulSoup(resp.text, "html.parser")

        # Priority 1: og:image
        og = soup.find("meta", property="og:image")
        if og and og.get("content"):
            src = og["content"].strip()
            if src.startswith("http"):
                return src

        # Priority 2: twitter:image
        tw = soup.find("meta", attrs={"name": "twitter:image"})
        if tw and tw.get("content"):
            src = tw["content"].strip()
            if src.startswith("http"):
                return src

        # Priority 3: first large article image
        for img in soup.select("article img, .article-body img, main img, .story img"):
            src = (img.get("src") or img.get("data-src") or
                   img.get("data-lazy-src") or "")
            if src.startswith("//"):
                src = "https:" + src
            elif src.startswith("/"):
                src = urljoin(url, src)
            if src.startswith("http") and not src.endswith((".gif", ".svg")):
                # Skip tiny icons
                w = img.get("width", "9999")
                try:
                    if int(str(w)) < 100:
                        continue
                except Exception:
                    pass
                return src

    except Exception as e:
        log.debug(f"Image fetch error for {url}: {e}")
    return None

# ──────────────────────────── RSS Parsing ─────────────────────────

def _parse_rss(feed_url: str) -> List[dict]:
    """Parse an RSS feed and return raw items."""
    try:
        resp = requests.get(
            feed_url,
            headers={"User-Agent": USER_AGENT},
            timeout=REQUEST_TIMEOUT,
        )
        soup = BeautifulSoup(resp.content, "xml")
        items = []
        for item in soup.find_all("item"):
            title = item.find("title")
            link = item.find("link")
            desc = item.find("description")
            pub = item.find("pubDate")
            source_tag = item.find("source")

            # Google News wraps real URL in <link> or sometimes in description
            raw_url = link.get_text(strip=True) if link else ""
            # Google News uses a redirect URL like news.google.com/articles/...
            # We keep it; the browser will follow the redirect.

            # Extract source from <source> tag or from title suffix " - SourceName"
            source_name = ""
            if source_tag:
                source_name = source_tag.get_text(strip=True)
            elif title:
                t = title.get_text(strip=True)
                if " - " in t:
                    source_name = t.rsplit(" - ", 1)[-1]

            # Clean title
            headline = title.get_text(strip=True) if title else ""
            if " - " in headline:
                headline = headline.rsplit(" - ", 1)[0].strip()

            summary = _clean_text(desc.get_text(strip=True)) if desc else ""
            # Remove HTML from summary
            summary = re.sub(r'<[^>]+>', ' ', summary)
            summary = re.sub(r'\s+', ' ', summary).strip()
            # Truncate
            summary = summary[:220] + "…" if len(summary) > 220 else summary

            pub_str = pub.get_text(strip=True) if pub else ""

            items.append({
                "headline": headline,
                "url": raw_url,
                "source": source_name,
                "summary": summary,
                "pub_date": pub_str,
            })
        return items
    except Exception as e:
        log.warning(f"RSS parse error [{feed_url}]: {e}")
        return []


def _resolve_google_news_url(gn_url: str) -> str:
    """Follow Google News redirect to get the real article URL."""
    try:
        resp = requests.get(
            gn_url,
            headers={"User-Agent": USER_AGENT},
            timeout=8,
            allow_redirects=True,
        )
        return resp.url
    except Exception:
        return gn_url

# ──────────────────────────── Main Fetcher ────────────────────────

import concurrent.futures

# ──────────────────────────── Optimized Parallel Fetcher ────────────────

def _process_single_article(r, category):
    """Processes a single raw RSS item: resolves URL, fetches image, and categorizes."""
    try:
        url = r["url"]
        # Resolve Google News redirect (fast)
        real_url = _resolve_google_news_url(url)

        if not _is_trusted(real_url):
            return None

        # Fetch image (slowest part - now running in parallel)
        img = _fetch_article_image(real_url)

        return NewsItem(
            headline=r["headline"],
            url=real_url,
            source=r["source"] or urlparse(real_url).netloc.lstrip("www."),
            image_url=img,
            published_date=r["pub_date"],
            time_ago=_time_ago(r["pub_date"]),
            summary=r["summary"],
            category=category,
            trust_source=_get_trust_label(real_url),
        )
    except Exception as e:
        log.debug(f"Error processing article: {e}")
        return None

def _fetch_category(category: str, feeds: List[str], max_per_cat: int = 4) -> List[NewsItem]:
    """Fetch news items for one category using parallel threads."""
    raw_items = []
    for feed_url in feeds:
        raw_items.extend(_parse_rss(feed_url))
    
    # Take more raw items than needed to account for untrusted/failed ones
    candidate_items = raw_items[:max_per_cat * 3]
    
    items: List[NewsItem] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(_process_single_article, r, category) for r in candidate_items]
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res:
                items.append(res)
                if len(items) >= max_per_cat:
                    break
    return items

def get_latest_news(max_total: int = 15) -> List[NewsItem]:
    """Fetch latest news across all topic categories. Now significantly faster."""
    cached = _cache_get("latest_news")
    if cached is not None:
        return cached

    log.info("Fetching fresh news in parallel...")
    all_items: List[NewsItem] = []
    
    # Fetch categories in parallel too
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_cat = {executor.submit(_fetch_category, cat, feeds, 4): cat 
                         for cat, feeds in TOPIC_FEEDS.items()}
        
        for future in concurrent.futures.as_completed(future_to_cat):
            all_items.extend(future.result())

    # Sort by recency
    def sort_key(item: NewsItem):
        try:
            dt = parsedate_to_datetime(item.published_date)
            return dt.timestamp()
        except Exception:
            return 0

    all_items.sort(key=sort_key, reverse=True)
    all_items = all_items[:max_total]

    _cache_set("latest_news", all_items)
    return all_items


def search_news(query: str, max_results: int = 10) -> List[NewsItem]:
    """Search for news matching a user query."""
    cache_key = f"search:{query.lower().strip()}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    # Build topic-aware feeds
    feeds = [
        _GN.format(q=query.replace(" ", "+")),
        _GN.format(q=f"{query.replace(' ', '+')}+India+fraud"),
        _GN.format(q=f"{query.replace(' ', '+')}+India+scheme"),
    ]

    items: List[NewsItem] = []
    seen: set = set()

    for feed_url in feeds:
        if len(items) >= max_results:
            break
        for r in _parse_rss(feed_url):
            if len(items) >= max_results:
                break
            url = r["url"]
            if not url or url in seen:
                continue
            real_url = _resolve_google_news_url(url)
            if not _is_trusted(real_url):
                continue
            seen.add(url)
            seen.add(real_url)
            img = _fetch_article_image(real_url)
            cat = _auto_categorize(r["headline"] + " " + r["summary"])
            items.append(NewsItem(
                headline=r["headline"],
                url=real_url,
                source=r["source"] or urlparse(real_url).netloc.lstrip("www."),
                image_url=img,
                published_date=r["pub_date"],
                time_ago=_time_ago(r["pub_date"]),
                summary=r["summary"],
                category=cat,
                trust_source=_get_trust_label(real_url),
            ))

    _cache_set(cache_key, items)
    return items


def _auto_categorize(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ["fraud", "scam", "cheat", "swindle", "cyber crime", "phishing", "ponzi"]):
        return "Fraud Alert"
    if any(k in t for k in ["bank", "rbi", "sbi", "hdfc", "icici", "loan", "deposit", "interest rate"]):
        return "Banking"
    if any(k in t for k in ["scheme", "yojana", "pm kisan", "ayushman", "subsidy", "pension", "government"]):
        return "Government Scheme"
    if any(k in t for k in ["fake", "misinformation", "fact check", "rumour", "false claim", "hoax"]):
        return "Misinformation"
    if any(k in t for k in ["awareness", "safety", "digital literacy", "cyber safety", "precaution"]):
        return "Social Awareness"
    return "General"


# Backwards-compatible alias
def answer_query(query: str, max_results: int = 8) -> List[NewsItem]:
    return search_news(query, max_results)