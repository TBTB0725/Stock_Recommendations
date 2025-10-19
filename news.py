# news.py
"""
Finviz News Scraping and Normalization
- Output: ["ticker","datetime","headline","source","url","relatedTickers"]
- Data Source: https://finviz.com/quote.ashx?t={TICKER} 的 #news-table
"""

import string
import pandas as pd
import time
from zoneinfo import ZoneInfo
from typing import Iterable, List, Dict, Any, Optional
from datetime import datetime, timedelta, timezone
from urllib.request import Request, urlopen
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup

# Parameters
DEFAULT_LOOKBACK_DAYS = 12         # Days
DEFAULT_PER_TICKER_COUNT = 100     # Initial Count
DEFAULT_FINAL_CAP = 25             # Final Count
DEFAULT_SLEEP_S = 0.3              # Sleep between Tickers
_FINVIZ_BASE = "https://finviz.com/quote.ashx?t="
_FINVIZ_ROOT = "https://finviz.com"
_BS_PARSER = "lxml"

# Convert Time Zone to UTC
_ET = ZoneInfo("America/New_York")

def fetch_finviz_headlines(
    tickers: Iterable[str],
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    per_ticker_count: int = DEFAULT_PER_TICKER_COUNT,
    final_cap: int = DEFAULT_FINAL_CAP,
    sleep_s: float = DEFAULT_SLEEP_S,
) -> pd.DataFrame:
    
    tickers = [t.strip().upper() for t in tickers if t and str(t).strip()]

    # Calculate time period that is acceptable
    cutoff_utc = datetime.now(timezone.utc) - timedelta(days=lookback_days)

    all_rows: List[Dict[str, Any]] = []

    for t in tickers:
        rows = _scrape_finviz_single(t, per_ticker_count)  # Get original data
        normed = [_normalize_row(t, r) for r in rows]      # Standardization

        # Cut news that is out of time period
        keep = [r for r in normed if r and (r["datetime"] is None or r["datetime"] >= cutoff_utc)]
        all_rows.extend(keep)

        # Sleep
        if sleep_s and sleep_s > 0:
            time.sleep(sleep_s)

    df = pd.DataFrame(all_rows)

    # Remove news that does not have headlines
    df = df[~(df["headline"].astype(str).str.len() == 0)]

    # Rerank + cut off the excess
    df = df.sort_values(["ticker", "datetime"], ascending=[True, False])
    df = df.groupby("ticker", group_keys=False).head(final_cap)

    return df.reset_index(drop=True)

# Single ticker scrapping
def _scrape_finviz_single(ticker: str, per_ticker_count: int) -> List[Dict[str, Any]]:
    url = _FINVIZ_BASE + ticker
    
    req = Request(
        url=url,
        headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://finviz.com/",
        },
    )
    
    # Get original html data
    html = urlopen(req, timeout=15).read()

    # Use soup to find what we want
    soup = BeautifulSoup(html, _BS_PARSER)
    table = soup.find(id="news-table") or soup.find("table", {"class": "fullview-news-outer"})

    parsed: List[Dict[str, Any]] = []
    current_date_str: Optional[str] = None

    for row in table.find_all("tr"):
        a = row.find("a"); td = row.find("td")
        if not a or not td:
            continue

        pieces = (td.get_text(strip=True) or "").split()
        if len(pieces) == 2:
            date_str, time_str = pieces[0], pieces[1]
            current_date_str = date_str
        elif len(pieces) == 1:
            time_str = pieces[0]
            date_str = current_date_str or "Today"
        else:
            continue

        headline = a.get_text(strip=True)
        href = a.get("href", "").strip()
        url = href if href.startswith(("http://", "https://")) else urljoin(_FINVIZ_ROOT, href)

        source_tag = row.find("span", class_="news-link-right")
        source = source_tag.get_text(strip=True) if source_tag else ""
        if not source and url:
            try:
                source = urlparse(url).netloc.split(":")[0]
            except Exception:
                source = ""

        parsed.append({"date_str": date_str, "time_str": time_str, "headline": headline, "url": url, "source": source})
        if len(parsed) >= per_ticker_count:
            break

    return parsed

# Standardize
def _normalize_row(ticker: str, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    headline = (item.get("headline") or "").strip()
    url = (item.get("url") or "").strip()
    source = (item.get("source") or "").strip()
    date_str = (item.get("date_str") or "").strip()
    time_str = (item.get("time_str") or "").strip()
    return {
        "ticker": ticker.upper(),
        "datetime": _parse_to_utc(date_str, time_str),
        "headline": headline,
        "source": source,
        "url": url,
        "relatedTickers": [ticker.upper()],
    }



def _parse_to_utc(date_str: str, time_str: str) -> Optional[datetime]:
    if not time_str:
        return None
    now_et = datetime.now(_ET) if _ET else datetime.now()
    if date_str in ("Today", "today", ""):
        d = now_et.date()
    elif date_str in ("Yesterday", "yesterday"):
        d = (now_et - timedelta(days=1)).date()
    else:
        d = None
        for fmt in ("%b-%d-%y", "%b-%d-%Y"):  # e.g., Oct-18-25 / Oct-18-2025
            try:
                d = datetime.strptime(date_str, fmt).date()
                break
            except Exception:
                pass
        if d is None:
            return None
    try:
        t = datetime.strptime(time_str.replace(" ", ""), "%I:%M%p").time()
    except Exception:
        return None
    if _ET:
        return datetime(d.year, d.month, d.day, t.hour, t.minute, tzinfo=_ET).astimezone(timezone.utc)
    return datetime(d.year, d.month, d.day, t.hour, t.minute).replace(tzinfo=timezone.utc)

_PUNC_TABLE = str.maketrans("", "", string.punctuation)
def _normalize_title(s: Any) -> str:
    return " ".join(str(s).lower().translate(_PUNC_TABLE).split()) if isinstance(s, str) else ""

# Test
if __name__ == "__main__":
    df = fetch_finviz_headlines(["AAPL", "MSFT", "NVDA"])
    cols = ["ticker", "datetime", "headline", "source", "url"]
    print(df[cols].head(10).to_string(index=False))
