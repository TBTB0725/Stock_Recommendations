# sentiment.py
from __future__ import annotations
import os
import time
from typing import Iterable, Dict, List, Tuple
import numpy as np
import pandas as pd

# 可选：OpenAI / Anthropic / Gemini
_USE_PROVIDER = os.getenv("NEWS_LLM_PROVIDER", "gemini").lower()

# --- Gemini (google-generativeai) ---
_use_gemini = False
try:
    import google.generativeai as genai
    _GEMINI_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if _USE_PROVIDER == "gemini" and _GEMINI_KEY:
        genai.configure(api_key=_GEMINI_KEY)
        _use_gemini = True
except Exception:
    _use_gemini = False

# 你也可以按需扩展 openai/anthropic，这里先聚焦 gemini 便于直接跑通

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
    model = genai.GenerativeModel("gemini-2.5-flash")
    resp = model.generate_content(prompt)
    # 取第一段文本
    return resp.candidates[0].content.parts[0].text.strip()

def score_headlines_grouped(
    headlines_df: pd.DataFrame,
    sleep_sec: float = 1.0,
) -> pd.DataFrame:
    """
    输入: DataFrame[ticker, published_at, headline, url]
    输出: DataFrame[ticker, impact, n_headlines, last_ts]
    - 将每只股票的若干标题合并，交给 LLM 产出 impact。
    """
    if headlines_df.empty:
        return pd.DataFrame(columns=["ticker", "impact", "n_headlines", "last_ts"])

    rows = []
    for t, g in headlines_df.groupby("ticker", sort=False):
        g_sorted = g.sort_values("published_at", ascending=False)
        heads = "\n".join(f"- {h}" for h in g_sorted["headline"].tolist())
        prompt = _PROMPT_TMPL.format(headlines=heads)

        impact = 0.0
        try:
            if _use_gemini:
                text = _gemini_call(prompt)
            else:
                # 如果没有可用的 provider/key，就返回 0 分，保证流程不断
                text = '{"symbol": "", "name": "", "impact": 0.0}'
            # 尝试解析 impact
            import json
            j = json.loads(text)
            impact = float(j.get("impact", 0.0))
            impact = max(-1.0, min(1.0, impact))
        except Exception:
            impact = 0.0

        rows.append(
            {
                "ticker": t,
                "impact": float(impact),
                "n_headlines": int(len(g_sorted)),
                "last_ts": g_sorted["published_at"].max(),
            }
        )
        time.sleep(sleep_sec)

    return pd.DataFrame(rows).reset_index(drop=True)

def impact_to_annual_uplift(
    impact: pd.Series,
    horizon_days: int,
    beta_h: float = 0.04,
    tdpy: int = 252,
) -> pd.Series:
    """
    将 impact ∈ [-1,1] 映射为 “预测期（horizon_days）的收益增量”：
        uplift_h = impact * beta_h
    再转换成年化增量：
        uplift_ann = (1 + uplift_h) ** (tdpy / horizon_days) - 1
    其中 beta_h 是“强正/负新闻在该预测期内的典型价差幅度”的缩放系数（可在 UI 中调）。
    """
    uplift_h = impact.astype(float) * float(beta_h)
    return (1.0 + uplift_h) ** (tdpy / float(horizon_days)) - 1.0
