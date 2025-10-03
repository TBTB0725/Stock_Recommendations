# news.py
from __future__ import annotations
import datetime as dt
import time
from typing import Iterable, List, Optional
import requests
from bs4 import BeautifulSoup
import pandas as pd

_FINVIZ_BASE = "https://finviz.com/quote.ashx?t="

def _request_html(url: str, timeout: int = 20) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.text

def _parse_finviz_table(ticker: str, html: str) -> List[dict]:
    """
    解析 <table id="news-table">，返回 [{'ticker', 'published_at', 'headline', 'url'}...]
    Finviz 的日期/时间规则：遇到新日期行，之后多行只有“时间”需继承最近日期。
    """
    soup = BeautifulSoup(html, "html.parser")
    tbl = soup.find(id="news-table")
    if tbl is None:
        return []

    rows = []
    current_date: Optional[dt.date] = None

    for tr in tbl.find_all("tr"):
        tdc = tr.find("td", {"class": "nn-date"})
        link = tr.find("a")
        if not link or not tr.td:
            continue

        # Finviz 的 “日期 + 时间”/只有“时间” 混排；先取 td 里的文本
        raw = tr.td.get_text(strip=True).split()
        # 取标题与 url
        headline = link.get_text(strip=True)
        url = link.get("href", "")

        # 日期/时间解析
        if len(raw) == 1:
            # 只有时间，沿用 current_date
            if current_date is None:
                # 如果开头第一行就只有时间，保守跳过（很少见）
                continue
            time_str = raw[0]
            try:
                t = dt.datetime.strptime(time_str, "%I:%M%p").time()
            except ValueError:
                # 兜底
                continue
            published_at = dt.datetime.combine(current_date, t)
        else:
            # raw[0]=日期（如 "Feb-02-25"）；raw[1]=时间（如 "09:15AM"）
            date_str, time_str = raw[0], raw[1]
            try:
                d = dt.datetime.strptime(date_str, "%b-%d-%y").date()
                t = dt.datetime.strptime(time_str, "%I:%M%p").time()
            except ValueError:
                # 某些行可能是 “Today”/“Yesterday” 等，这里做个简单兼容
                if date_str.lower() in ("today", "yesterday"):
                    base = dt.date.today()
                    if date_str.lower() == "yesterday":
                        base = base - dt.timedelta(days=1)
                    current_date = base
                    try:
                        t = dt.datetime.strptime(time_str, "%I:%M%p").time()
                        published_at = dt.datetime.combine(current_date, t)
                    except ValueError:
                        continue
                else:
                    continue
            else:
                current_date = d
                published_at = dt.datetime.combine(d, t)

        rows.append(
            {
                "ticker": ticker.upper(),
                "published_at": published_at,
                "headline": headline,
                "url": url,
            }
        )

    return rows

def fetch_finviz_headlines(
    tickers: Iterable[str],
    sleep_sec: float = 0.5,
) -> pd.DataFrame:
    """
    抓取多个 ticker 的新闻；返回 DataFrame[ticker, published_at (datetime), headline, url]
    """
    all_rows: List[dict] = []
    for t in tickers:
        url = f"{_FINVIZ_BASE}{t}"
        try:
            html = _request_html(url)
            rows = _parse_finviz_table(t, html)
            all_rows.extend(rows)
        except Exception:
            # 对单个 ticker 的错误容忍，不中断全部
            continue
        time.sleep(sleep_sec)

    df = pd.DataFrame(all_rows)
    if df.empty:
        return pd.DataFrame(columns=["ticker", "published_at", "headline", "url"])

    # 去重（有时同标题重复）
    df = df.drop_duplicates(subset=["ticker", "headline", "published_at"]).sort_values(
        ["ticker", "published_at"], ascending=[True, False]
    )
    return df.reset_index(drop=True)

def recent_headlines(
    df: pd.DataFrame,
    days_back: int = 10,
    per_ticker: int = 30,
) -> pd.DataFrame:
    """
    过滤最近 N 天并限制每只股票最多条数。
    """
    if df.empty:
        return df
    cutoff = dt.datetime.now() - dt.timedelta(days=days_back)
    out = []
    for t, g in df.groupby("ticker", sort=False):
        g2 = g[g["published_at"] >= cutoff].sort_values("published_at", ascending=False)
        out.append(g2.head(per_ticker))
    return pd.concat(out).reset_index(drop=True) if out else df.head(0)
