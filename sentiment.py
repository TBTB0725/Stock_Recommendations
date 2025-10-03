# sentiment.py
from __future__ import annotations
import os
import time
from typing import Iterable, Dict, List, Tuple
import numpy as np
import pandas as pd
import os, json, time, base64, traceback

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
    目标：
    - 强制模型输出 application/json
    - 兼容三种返回形态：resp.text / parts[].text / parts[].inline_data (base64 JSON)
    - 把本次调用的路径/元数据记录到 LAST_CALL_DEBUG（终端 print + 可回读）
    """
    global LAST_CALL_DEBUG
    LAST_CALL_DEBUG = {"path": "unknown", "parts": 0, "err": None}

    try:
        model = genai.GenerativeModel(
            "gemini-2.5-flash",
            generation_config={
                "response_mime_type": "application/json",
                "temperature": 0,
                "candidate_count": 1,
            },
        )
        resp = model.generate_content(prompt)

        # case 1: 直接 text
        if hasattr(resp, "text") and resp.text:
            LAST_CALL_DEBUG.update({"path": "resp.text"})
            print("[gemini] using resp.text")
            return resp.text.strip()

        # case 2/3: 遍历 parts
        cand = resp.candidates[0]
        parts = getattr(cand.content, "parts", []) or []
        LAST_CALL_DEBUG["parts"] = len(parts)

        texts = []
        for i, p in enumerate(parts):
            inline = getattr(p, "inline_data", None)
            if inline and getattr(inline, "mime_type", "") == "application/json":
                b64 = getattr(inline, "data", "") or ""
                if b64:
                    try:
                        raw = base64.b64decode(b64).decode("utf-8", "ignore")
                        if raw.strip():
                            LAST_CALL_DEBUG.update({"path": f"parts[{i}].inline_data"})
                            print(f"[gemini] using parts[{i}].inline_data(application/json)")
                            return raw.strip()
                    except Exception as e:
                        print("[gemini] inline_data decode error:", e)

            t = getattr(p, "text", None)
            if t:
                texts.append(t)

        if texts:
            LAST_CALL_DEBUG.update({"path": "parts[].text"})
            print("[gemini] using parts[].text join")
            return "\n".join(texts).strip()

        # 兜底
        LAST_CALL_DEBUG.update({"path": "str(resp)"})
        print("[gemini] fallback to str(resp)")
        return str(resp)

    except Exception as e:
        LAST_CALL_DEBUG.update({"err": traceback.format_exc()})
        print("[gemini] EXCEPTION:\n", LAST_CALL_DEBUG["err"])
        return ""




def score_headlines_grouped(
    headlines_df: pd.DataFrame,
    sleep_sec: float = 0.2,
    return_raw: bool = False,
) -> pd.DataFrame:
    """
    输入: DataFrame[ticker, published_at, headline, url]
    输出: DataFrame[ticker, impact, n_headlines, last_ts, raw?, path?, err?]
    - 每个 ticker 把若干 headline 合并一次喂给 LLM
    - 把本次调用的调试信息（path/err）也带回（return_raw=True 时最有用）
    - 终端会打印每个 ticker 的处理日志
    """
    cols = ["ticker", "impact", "n_headlines", "last_ts"]
    if return_raw:
        cols += ["raw", "path", "err"]

    if headlines_df is None or headlines_df.empty:
        return pd.DataFrame(columns=cols)

    out_rows = []
    # 关键：显式保证按 ticker 分组；若分组失败，说明列名/类型不对，会直接抛出错误
    for t, g in headlines_df.groupby("ticker", sort=False):
        try:
            g_sorted = g.sort_values("published_at", ascending=False)
            n = len(g_sorted)
            heads = "\n".join(f"- {h}" for h in g_sorted["headline"].tolist())[:8000]  # 防 prompt 太长
            prompt = _PROMPT_TMPL.format(headlines=heads)

            print(f"[sentiment] scoring ticker={t} headlines={n}")
            raw = ""
            impact = 0.0
            path = None
            err = None

            if _use_gemini:
                raw = _gemini_call(prompt)  # 可能是 json 字符串，也可能为空
                path = (LAST_CALL_DEBUG or {}).get("path")
                if (LAST_CALL_DEBUG or {}).get("err"):
                    err = LAST_CALL_DEBUG["err"]

                if raw:
                    try:
                        j = json.loads(raw)
                        impact = float(j.get("impact", 0.0))
                        impact = max(-1.0, min(1.0, impact))
                    except Exception:
                        err = (err or "") + "\njson.loads failed\n" + traceback.format_exc()
                        print("[sentiment] json.loads failed:", err)
                        impact = 0.0
                else:
                    err = (err or "") + "\n_gemini_call returned empty string"
                    print("[sentiment] empty raw from _gemini_call")
                    impact = 0.0
            else:
                err = "LLM disabled (_use_gemini=False)"
                impact = 0.0

            row = {
                "ticker": t,
                "impact": float(impact),
                "n_headlines": int(n),
                "last_ts": g_sorted["published_at"].max(),
            }
            if return_raw:
                row.update({"raw": raw, "path": path, "err": err})

            out_rows.append(row)
            time.sleep(sleep_sec)

        except Exception:
            # 记录单个 ticker 的错误，不让整个批次失败
            print(f"[sentiment] EXCEPTION in ticker={t}:\n", traceback.format_exc())
            row = {"ticker": t, "impact": 0.0, "n_headlines": len(g), "last_ts": g["published_at"].max()}
            if return_raw:
                row.update({"raw": "", "path": None, "err": traceback.format_exc()})
            out_rows.append(row)

    return pd.DataFrame(out_rows, columns=cols)


def impact_to_annual_uplift(
    impact: pd.Series,
    horizon_days: int,
    beta_h: float = 0.04,
    tdpy: int = 252,
) -> pd.Series:
    uplift_h = impact.astype(float) * float(beta_h)
    return (1.0 + uplift_h) ** (tdpy / float(horizon_days)) - 1.0
