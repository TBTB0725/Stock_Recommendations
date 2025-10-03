# sentiment.py
from __future__ import annotations
import os
import time
from typing import Iterable, Dict, List, Tuple
import numpy as np
import pandas as pd

_USE_PROVIDER = os.getenv("NEWS_LLM_PROVIDER", "gemini").lower()

def _extract_json(text: str) -> dict:
    import json, re
    s = text.strip()

    # 1) 去掉围栏 ```json ... ``` 或 ``` ... ```
    fence = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.S | re.I)
    m = fence.match(s)
    if m:
        s = m.group(1).strip()

    # 2) 直接尝试 json.loads
    try:
        return json.loads(s)
    except Exception:
        pass

    # 3) 从文本中提取第一个 {...} 作为 JSON 尝试
    m = re.search(r"\{.*\}", s, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass

    # 4) 失败就返回空
    return {}

_use_gemini = False
try:
    import google.generativeai as genai
    _GEMINI_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if _USE_PROVIDER == "gemini" and _GEMINI_KEY:
        genai.configure(api_key=_GEMINI_KEY)
        _use_gemini = True
except Exception:
    _use_gemini = False


_PROMPT_TMPL = """You are a finance expert.
Given a list of short stock news headlines for one company, output ONE JSON line with fields:
- symbol: the stock ticker (leave empty if unclear)
- name: the company name (leave empty if unclear)
- impact: a real number in [-1.0, 1.0] indicating the expected price impact over the next few days/weeks
  (-1 = very negative, +1 = very positive).

Only output the JSON, nothing else.

Headlines:
{headlines}
"""

def _gemini_call(prompt: str) -> str:
    """
    强制 JSON 输出；优先用 resp.text，兜底拼接 parts。
    """
    model = genai.GenerativeModel(
        "gemini-2.5-flash",
        generation_config={"response_mime_type": "application/json"}  # ← 关键：直接要 JSON
    )
    resp = model.generate_content(prompt)

    # 1) 最常见：直接拿纯文本（已是 JSON 字符串）
    if hasattr(resp, "text") and resp.text:
        return resp.text.strip()

    # 2) 兜底：把第一条候选的所有文本片段拼起来
    try:
        parts = resp.candidates[0].content.parts
        txts = []
        for p in parts:
            if hasattr(p, "text") and p.text:
                txts.append(p.text)
        if txts:
            return "\n".join(txts).strip()
    except Exception:
        pass

    # 3) 最兜底：把 resp 转成字符串
    return str(resp)


def score_headlines_grouped(
    headlines_df: pd.DataFrame,
    sleep_sec: float = 1.0,
    return_raw: bool = False,   # ← 新增：需要时返回原始 LLM 文本
) -> pd.DataFrame:
    """
    输入: DataFrame[ticker, published_at, headline, url]
    输出: DataFrame[ticker, impact, n_headlines, last_ts]（如 return_raw=True，还会带 raw 列）
    """
    if headlines_df.empty:
        cols = ["ticker", "impact", "n_headlines", "last_ts"] + (["raw"] if return_raw else [])
        return pd.DataFrame(columns=cols)

    rows = []
    for t, g in headlines_df.groupby("ticker", sort=False):
        g_sorted = g.sort_values("published_at", ascending=False)
        heads = "\n".join(f"- {h}" for h in g_sorted["headline"].tolist())
        prompt = _PROMPT_TMPL.format(headlines=heads)

        impact = 0.0
        raw_text = ""
        try:
            if _use_gemini:
                raw_text = _gemini_call(prompt)
            else:
                raw_text = '{"symbol": "", "name": "", "impact": 0.0}'

            # 直接解析 JSON（此时大概率是干净的 JSON）
            import json
            j = json.loads(raw_text)
            impact = float(j.get("impact", 0.0))
            impact = max(-1.0, min(1.0, impact))
        except Exception:
            impact = 0.0

        row = {
            "ticker": t,
            "impact": float(impact),
            "n_headlines": int(len(g_sorted)),
            "last_ts": g_sorted["published_at"].max(),
        }
        if return_raw:
            row["raw"] = raw_text
        rows.append(row)
        time.sleep(sleep_sec)

    return pd.DataFrame(rows).reset_index(drop=True)
    

def impact_to_annual_uplift(
    impact: pd.Series,
    horizon_days: int,
    beta_h: float = 0.04,
    tdpy: int = 252,
) -> pd.Series:
    uplift_h = impact.astype(float) * float(beta_h)
    return (1.0 + uplift_h) ** (tdpy / float(horizon_days)) - 1.0
