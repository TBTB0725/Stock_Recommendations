# sentiment.py —— ChatGPT(Title to Score)
import os, time, json
from typing import Dict, Any, List, Optional

from openai import OpenAI, APIError, APIStatusError

DEFAULT_MODEL = "gpt-4.1-mini"

_RATE = float(os.getenv("RATE", "0.2"))
_RETRY = int(os.getenv("RETRY", "1"))
_BACKOFF = 2.0

_PROMPT ="""You are an experienced equity analyst.

Goal
Score the expected near-term PRICE IMPACT (next 1-5 trading days) of ONE news headline on ONE specific stock.

Inputs
- Ticker: {ticker}
- Headline: {headline}

Rules (decide sign & magnitude)
1) Company match: If the headline is NOT about {ticker} (or only broadly market/sector-wide), return 0.0.
2) Novelty/materiality: Larger impact only if the info is NEW and MATERIAL for {ticker}'s cash flows, risk, or multiple.
3) Source quality: Rumor/speculation → smaller magnitude; official filings/earnings/regulator actions → larger magnitude.
4) Scope:
   - Company-specific news → sign/magnitude for {ticker}.
   - Peer/supplier/customer news → small spillover (|impact| ≤ 0.2) unless clearly material.
   - Macro/market-wide news (rates, CPI, geopolitics) → 0.0 unless headline is explicitly {ticker}-specific.
5) Mapping the scale (guideline):
   - +0.7 to +1.0: very positive (e.g., acquisition at premium, blowout earnings + strong guide, big regulatory win)
   - +0.3 to +0.6: moderately positive (beat, major partnership win, product launch with clear edge)
   - 0.1 to 0.2: slightly positive (incremental tailwind, minor favorable data point)
   - 0.0: neutral/ambiguous/old news/not about {ticker}/market-wide only
   - -0.1 to -0.2: slightly negative (minor setback, limited issue)
   - -0.3 to -0.6: moderately negative (miss, guidance cut, meaningful probe/recall)
   - -0.7 to -1.0: very negative (fraud/ban/large legal loss/major safety failure)
6) If headline contains immediate price move (“shares +8% premarket”), reflect the sign but still use the above scale (do not copy %).
7) Be concise: Reason ≤ 20 words, factual and specific.

Output (STRICT JSON, no extra text)
{{
  "symbol": "{ticker}",
  "name": "",              // official company name if you are highly confident, else ""
  "impact": <float between -1.0 and 1.0>,
  "reason": "<very brief rationale tied to the headline>"
}}
"""

_client_cache: Dict[str, OpenAI] = {}

def _ensure_client(api_key: Optional[str] = None) -> OpenAI:
    key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI")
    if not key:
        raise RuntimeError("Missing OPENAI_API_KEY (or pass api_key).")
    if "client" not in _client_cache:
        _client_cache["client"] = OpenAI(api_key=key, timeout=20.0)
    return _client_cache["client"]

def _first_json(text: str) -> Dict[str, Any]:
    s = (text or "").strip()
    if s.startswith("```"):
        s = s.strip("`")
    a, b = s.find("{"), s.rfind("}")
    if a == -1 or b == -1 or b <= a:
        return {}
    frag = s[a : b + 1]
    try:
        return json.loads(frag)
    except Exception:
        try:
            return json.loads(frag.replace("\n", " ").replace(",}", "}"))
        except Exception:
            return {}

def _score_one(client: OpenAI, model_name: str, ticker: str, headline: str) -> Dict[str, Any]:
    if not headline or not str(headline).strip():
        return {"impact": 0.0, "ai_raw": "", "ai_json": {}}

    prompt = _PROMPT.format(ticker=ticker, headline=str(headline).strip())
    delay = _RATE
    last_err: Optional[Exception] = None

    for attempt in range(_RETRY):
        try:
            print(f"[score] {ticker}: try {attempt+1}", flush=True)
            time.sleep(_RATE)

            completion = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=150,
            )

            ai_raw = (completion.choices[0].message.content or "").strip()

            try:
                data = json.loads(ai_raw) if ai_raw else {}
            except Exception:
                data = _first_json(ai_raw)

            impact = float(data.get("impact", 0.0)) if isinstance(data, dict) else 0.0
            return {
                "impact": max(-1.0, min(1.0, impact)),
                "ai_raw": ai_raw,
                "ai_json": data if isinstance(data, dict) else {},
            }

        except (APIStatusError, APIError) as e:
            last_err = e
            print(f"[error] {ticker}: {type(e).__name__} -> {e}", flush=True)
            time.sleep(delay); delay *= _BACKOFF
        except Exception as e:
            last_err = e
            print(f"[error] {ticker}: {type(e).__name__} -> {e}", flush=True)
            time.sleep(delay); delay *= _BACKOFF

    return {"impact": 0.0, "ai_raw": f"ERROR: {last_err}", "ai_json": {}}

def score_titles(
    items: List[Dict[str, Any]],
    model_name: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Input: list[dict], each item contains at least {"ticker": str, "headline": str}
    Output: list[dict] aligned with the input, with additional {"impact": float, "ai_raw": str, "ai_json": dict}
    """
    client = _ensure_client(api_key)
    out: List[Dict[str, Any]] = []
    for it in items:
        res = _score_one(client, model_name, str(it.get("ticker", "")).strip(), str(it.get("headline", "")).strip())
        row = dict(it); row.update(res)
        out.append(row)
    return out

if __name__ == "__main__":
    demo = [
        {"ticker":"AAPL","headline":"Apple unveils new iPhone with satellite SOS feature"},
        {"ticker":"NVDA","headline":"NVIDIA data-center revenue jumps on strong AI demand"},
        {"ticker":"TSLA","headline":"Tesla recalls vehicles to fix Autopilot warning issue"},
    ]
    out = score_titles(demo, model_name=os.getenv("OPENAI_MODEL","gpt-4.1-mini"))
    print(json.dumps(out, ensure_ascii=False, indent=2))
