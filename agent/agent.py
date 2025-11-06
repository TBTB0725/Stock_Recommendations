# agent/agent.py
from __future__ import annotations
import json, logging, time, uuid
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from dataclasses import asdict, is_dataclass
from openai import OpenAI
from datetime import datetime, date

from agent import tools as T

# -----------------------------
# Logger
# -----------------------------
logger = logging.getLogger("agent.chat")
if not logger.handlers:
    _h = logging.StreamHandler()
    _fmt = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s", "%H:%M:%S")
    _h.setFormatter(_fmt); logger.addHandler(_h)
logger.setLevel(logging.INFO)


# -------------------------------
# Constants & Verification Rules
# -------------------------------
ALLOWED_HORIZONS = {"1D", "5D", "1W", "2W", "1M", "3M", "6M", "1Y"}

REQUIRED_KW = {
    "fetch_prices_tool": {"tickers"},
    "to_returns_tool": {"prices"},
    "forecast_tool": {"prices", "horizon"},
    "risk_tool": {"returns"},
    "optimize_tool": {"objective", "mu_annual", "Sigma_annual"},
    "evaluate_portfolio_tool": {
        "name", "tickers", "weights", "capital",
        "mu_annual", "Sigma_annual", "returns_daily",
    },
    "compile_report_tool": {"results"},
    "fetch_news_tool": {"tickers"},
    "sentiment_score_titles_tool": {"items"},
}

PRIMARY_PARAM = {
    "fetch_prices_tool": (
        "tickers",
        {"tickers", "ticker", "symbols", "symbol", "assets", "universe", "list"},
    ),
    "to_returns_tool": (
        "prices",
        {"price", "prices", "px", "df", "data", "frame", "table"},
    ),
    "forecast_tool": (
        "prices",
        {"price", "prices", "px", "df", "data", "frame"},
    ),
    "risk_tool": (
        "returns",
        {"ret", "rets", "returns", "returns_df", "df", "data"},
    ),
    "optimize_tool": (
        "objective",
        {"objective", "obj", "mode", "target"},
    ),

    "evaluate_portfolio_tool": (
        "weights",
        {"weights", "w", "portfolio", "vector"},
    ),
    "compile_report_tool": (
        "results",
        {"results", "items", "list", "objs", "arr"},
    ),
    "fetch_news_tool": (
        "tickers",
        {"tickers", "ticker", "symbols", "symbol", "universe", "list"},
    ),
    "sentiment_score_titles_tool": (
        "items",
        {"items", "titles", "news", "rows", "df", "data", "list"},
    ),
}


ALLOWED_KW = {
    "fetch_prices_tool": {
        "tickers", "lookback_days", "end",
        "pad_ratio", "auto_business_align", "use_adjusted_close",
    },
    "to_returns_tool": {
        "prices", "method", "dropna",
    },
    "forecast_tool": {
        "prices", "horizon", "tune",
        "cv_metric", "cv_initial_days", "cv_period_days",
        "param_grid", "min_points_for_cv",
    },
    "risk_tool": {
        "returns", "annualize", "trading_days_per_year",
    },
    "optimize_tool": {
        "objective",
        "mu_annual", "Sigma_annual",
        "rf", "cons",
        "eps", "restarts", "seed",
    },
    "evaluate_portfolio_tool": {
        "name", "tickers", "weights", "capital",
        "mu_annual", "Sigma_annual", "returns_daily",
        "rf_annual", "var_alpha", "var_horizon_days", "log_returns",
    },
    "compile_report_tool": {
        "results",
    },
    "fetch_news_tool": {
        "tickers", "lookback_days",
        "per_ticker_count", "final_cap", "sleep_s",
    },
    "sentiment_score_titles_tool": {
        "items", "model_name", "api_key",
        "default_ticker", "limit",
    },
}

# -------------------------
# Tool registration
# -------------------------
TOOLS_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "fetch_prices_tool",
            "description": "Fetch historical prices.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tickers": {"type": "array", "items": {"type": "string"}},
                    "lookback_days": {"type": "integer", "default": 252},
                    "end": {"type": ["string", "null"]},
                    "pad_ratio": {"type": "number", "default": 2.0},
                    "auto_business_align": {"type": "boolean", "default": True},
                    "use_adjusted_close": {"type": "boolean", "default": True},
                },
                "required": ["tickers"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "to_returns_tool",
            "description": "Convert price DataFrame to returns DataFrame. Use {'__ref__': <id>} for 'prices'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prices": {"type": "object"},
                    "method": {"type": "string", "enum": ["log", "simple"], "default": "log"},
                    "dropna": {"type": "boolean", "default": True},
                },
                "required": ["prices"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "forecast_tool",
            "description": "Forecast annualized expected returns (Prophet).",
            "parameters": {
                "type": "object",
                "properties": {
                    "prices": {"type": "object"},
                    "horizon": {
                        "type": "string",
                        "enum": ["1D","5D","1W","2W","1M","3M","6M","1Y"],
                        "default": "3M"
                    },
                    "tune": {"type": "boolean", "default": False},
                },
                "required": ["prices"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "risk_tool",
            "description": "Compute (annualized) covariance matrix from returns.",
            "parameters": {
                "type": "object",
                "properties": {
                    "returns": {"type": "object"},
                    "annualize": {"type": "boolean", "default": True},
                    "trading_days_per_year": {"type": "integer", "default": 252},
                },
                "required": ["returns"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "optimize_tool",
            "description": "Portfolio optimization ('min_var'|'max_ret'|'max_sharpe').",
            "parameters": {
                "type": "object",
                "properties": {
                    "objective": {
                        "type": "string",
                        "enum": ["min_var", "max_ret", "max_sharpe"]
                    },
                    "mu_annual": {"type": ["object", "null"]},
                    "Sigma_annual": {"type": ["object", "null"]},
                    "rf": {"type": "number", "default": 0.0},
                    "cons": {"type": ["object", "null"]},
                    "eps": {"type": "number", "default": 1e-12},
                    "restarts": {"type": "integer", "default": 0},
                    "seed": {"type": "integer", "default": 0},
                },
                "required": ["objective"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "evaluate_portfolio_tool",
            "description": "Evaluate portfolio (VaR, μ, σ, Sharpe, allocations). Use {'__ref__': <id>} for large inputs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "default": "Portfolio"},
                    "tickers": {"type": "array", "items": {"type": "string"}},
                    "weights": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Weight vector aligned with `tickers`.",
                    },
                    "capital": {"type": "number"},
                    "mu_annual": {"type": "object"},
                    "Sigma_annual": {"type": "object"},
                    "returns_daily": {"type": "object"},
                    "rf_annual": {"type": "number", "default": 0.0},
                    "var_alpha": {"type": "number", "default": 0.05},
                    "var_horizon_days": {"type": "integer", "default": 1},
                    "log_returns": {"type": "boolean", "default": True},
                },
                "required": [
                    "name",
                    "tickers",
                    "weights",
                    "capital",
                    "mu_annual",
                    "Sigma_annual",
                    "returns_daily",
                ],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compile_report_tool",
            "description": "Aggregate multiple PortfolioResult into summary tables.",
            "parameters": {
                "type": "object",
                "properties": {
                    "results": {"type": "array", "items": {"type": "object"}}
                },
                "required": ["results"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_news_tool",
            "description": "Fetch recent Finviz headlines for tickers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tickers": {"type": "array", "items": {"type": "string"}},
                    "lookback_days": {"type": "integer", "default": 3},
                    "per_ticker_count": {"type": "integer", "default": 15},
                    "final_cap": {"type": "integer", "default": 200},
                    "sleep_s": {"type": "number", "default": 0.5},
                },
                "required": ["tickers"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sentiment_score_titles_tool",
            "description": "Score news titles' near-term impact. Accepts DataFrame/list[dict]/list[str]/str (or reference).",
            "parameters": {
                "type": "object",
                "properties": {
                    "items": {"type": "object"},
                    "model_name": {"type": "string", "default": "gpt-4.1-mini"},
                    "api_key": {"type": ["string", "null"], "default": None},
                    "default_ticker": {"type": ["string", "null"], "default": None},
                    "limit": {"type": "integer", "default": 200},
                },
                "required": ["items"],
            },
        },
    },
]

TOOL_FUNCS = {
    "fetch_prices_tool": T.fetch_prices_tool,
    "to_returns_tool": T.to_returns_tool,
    "forecast_tool": T.forecast_tool,
    "risk_tool": T.risk_tool,
    "optimize_tool": T.optimize_tool,
    "evaluate_portfolio_tool": T.evaluate_portfolio_tool,
    "compile_report_tool": T.compile_report_tool,
    "fetch_news_tool": T.fetch_news_tool,
    "sentiment_score_titles_tool": T.sentiment_score_titles_tool,
}

# ---------------------
# System prompt
# ---------------------
SCOPE_GUARD_PROMPT = """
You are QuantChat, a conversational quant agent with access to a fixed set of tools.

Hard rules (must follow):

1) Scope
   - Only answer tasks that can be solved using these tools:
     - price data → returns
     - Prophet-based expected returns
     - covariance / risk metrics
     - portfolio optimization (min_var / max_ret / max_sharpe)
     - portfolio evaluation (μ, σ, Sharpe, VaR)
     - news + sentiment scoring
   - For anything else (chitchat, health, travel, generic coding, etc.), reply:
     "I only answer quantitative questions I can compute with my tools."

2) Inputs
   - If key inputs are missing (e.g. tickers, capital, horizon, rf, weights),
     ask a SHORT follow-up question to get them.

3) Tool usage
   - Prefer calling tools whenever a numeric or data-based answer is needed.
   - Use as few tool calls as possible to reliably answer the question.
   - You may use {"__ref__": "<object_id>"} to refer to previous tool outputs.

4) Numerical answers
   - Never invent numbers.
   - Never output a numeric result unless it comes from a successful tool call
     in the current conversation context.
   - If a tool fails or data is insufficient:
       - try a minimal alternative tool chain once;
       - otherwise briefly explain what is missing or that the computation failed.

5) Communication style
   - Be concise, precise, and quantitative.
   - State assumptions explicitly (dates, horizons, confidence levels, units).
   - Refuse gracefully when out-of-scope using one short sentence.

Always obey these rules before answering the user.
"""

# ----------------------------
# Memory object repository
# ----------------------------
class ObjectStore:
    def __init__(self):
        self._store: Dict[str, Any] = {}

    def put(self, obj: Any) -> str:
        rid = f"obj_{uuid.uuid4().hex[:8]}"
        self._store[rid] = obj
        return rid
    
    def get(self, rid: str) -> Any:
        if rid not in self._store:
            raise KeyError(f"Object ref not found: {rid}")
        return self._store[rid]
    
    def resolve_refs(self, args: Any) -> Any:
        if isinstance(args, dict):
            if set(args.keys()) == {"__ref__"}:
                return self.get(args["__ref__"])
            return {k: self.resolve_refs(v) for k, v in args.items()}
        if isinstance(args, list):
            return [self.resolve_refs(v) for v in args]
        return args
    

# ---------------------------------
# Ensure the correctness of param
# ---------------------------------
def _normalize_args(tool_name: str, raw_args: Any) -> Any:
    """
    参数规范化（严格模式）：
      - 如果是裸 {"__ref__": "..."}，包装到该工具的主参数下：
        例如 to_returns_tool -> {"prices": {"__ref__": "..."}}
      - 如果缺主参数且只有一个键，且值是 {"__ref__": "..."}，也做同样包装。
    """
    if not isinstance(raw_args, dict):
        return raw_args

    expected, aliases = PRIMARY_PARAM.get(tool_name, (None, set()))
    if expected is None:
        return raw_args

    # 裸 __ref__：直接映射到主参数
    if set(raw_args.keys()) == {"__ref__"}:
        return {expected: raw_args}

    args = dict(raw_args)

    # 别名 → 主参数（这是显式名字，不是猜数据类型，可以保留）
    if expected not in args:
        for alias in list(args.keys()):
            if alias in aliases:
                args[expected] = args.pop(alias)
                break

    # 单键 + __ref__ → 主参数
    if expected not in args and len(args) == 1:
        _, v = next(iter(args.items()))
        if isinstance(v, dict) and set(v.keys()) == {"__ref__"}:
            args = {expected: v}

    return args

def _prune_kwargs(tool_name: str, args: Any) -> Any:
    """删除工具不认识的键（如遗留的 '__ref__'），避免意外 kwargs 抛错。"""
    if not isinstance(args, dict):
        return args
    allowed = ALLOWED_KW.get(tool_name)
    if not allowed:
        return args
    return {k: v for k, v in args.items() if k in allowed}


# ---------------------------------
# Make objects JSON-serializable
# ---------------------------------
def _json_safe(o):
    """Convert common non-JSON-serializable objects to JSON-safe structures."""
    if o is None or isinstance(o, (str, int, float, bool)):
        return o
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp, datetime, date, np.datetime64)):
        try:
            return pd.Timestamp(o).isoformat()
        except Exception:
            return str(o)
    if isinstance(o, (list, tuple, set)):
        return [_json_safe(x) for x in o]
    if isinstance(o, dict):
        return {str(k): _json_safe(v) for k, v in o.items()}
    # final fallback
    return str(o)

# ----------------------------------
# Build lightweight result preview
# ----------------------------------
def _coerce_preview(x: Any) -> Any:

    if x is None or isinstance(x, (str, int, float, bool)):
        return x

    if isinstance(x, (pd.Timestamp, datetime, date, np.datetime64)):
        return _json_safe(x)

    if isinstance(x, np.ndarray):
        return {"__ndarray__": True, "shape": list(x.shape), "preview": _json_safe(x[:10].tolist())}

    if isinstance(x, pd.Series):
        return {
            "__series__": True,
            "index_preview": _json_safe(x.index[:10].tolist()),
            "values_preview": _json_safe(x.iloc[:10].tolist()),
        }

    if isinstance(x, pd.DataFrame):
        head_split = x.head(5).to_dict(orient="split")
        return {
            "__dataframe__": True,
            "shape": [int(x.shape[0]), int(x.shape[1])],
            "columns": _json_safe(x.columns.tolist()[:20]),
            "head": _json_safe(head_split),
        }

    if is_dataclass(x):
        fields = {k: _coerce_preview(v) for k, v in asdict(x).items()}
        return {"__dataclass__": True, "fields": fields}

    if isinstance(x, (list, tuple)):
        return [_coerce_preview(v) for v in x]
    if isinstance(x, dict):
        return {k: _coerce_preview(v) for k, v in x.items()}

    return str(type(x))

# ---------------------
# Dialogue Agent
# ---------------------
class ChatStockAgent:
    def __init__(self, model: str = "gpt-4.1-mini", verbose: bool = True, system_prompt: Optional[str] = None):
        self.client = OpenAI()
        self.model = model
        self.verbose = verbose
        self.system_prompt = system_prompt or SCOPE_GUARD_PROMPT
        self.reset()

    def reset(self):
        self.messages: List[Dict[str, str]] = [{"role": "system", "content": self.system_prompt}]
        self.store = ObjectStore()

    def _log(self, msg: str):
        if self.verbose: logger.info(msg)
    
    def _sanitize_forecast_args(self, args: Dict[str, Any]) -> Dict[str, Any]:
        if "horizon" not in args:
            raise T.ToolExecutionError(
                "forecast_tool missing required arg: ['horizon']."
            )
        hz = str(args["horizon"]).upper().strip()

        aliases = {
            "1DAY": "1D",
            "DAILY": "1D",
            "5DAY": "5D",
            "1WEEK": "1W",
            "2WEEK": "2W",
            "1MONTH": "1M",
            "3MONTH": "3M",
            "6MONTH": "6M",
            "12M": "1Y",
            "1YEAR": "1Y",
            "12MONTH": "1Y",
        }

        hz = aliases.get(hz, hz)
        if hz not in ALLOWED_HORIZONS:
            raise T.ToolExecutionError(
                f"Invalid forecast horizon '{hz}'. "
                f"Allowed horizons: {sorted(ALLOWED_HORIZONS)}"
            )

        args["horizon"] = hz
        return args
    
    def _parse_rf(self, x) -> float:
        """严格解析 rf；无效输入直接报错，不回退 0."""
        if isinstance(x, str):
            s = x.strip()
            if s.endswith("%"):
                s = s[:-1].strip()
                try:
                    return float(s) / 100.0
                except Exception:
                    raise T.ToolExecutionError(
                        f"Invalid rf value '{x}'. Expect like 2% or 0.02."
                    )
            try:
                return float(s)
            except Exception:
                raise T.ToolExecutionError(
                    f"Invalid rf value '{x}'. Expect like 2% or 0.02."
                )

        try:
            return float(x)
        except Exception:
            raise T.ToolExecutionError(
                f"Invalid rf value '{x}'. Expect numeric, e.g. 0.02."
            )

    def _run_tool(self, name: str, raw_args: Dict[str, Any]) -> Dict[str, Any]:
        fn = TOOL_FUNCS[name]

        # 1) 归一化当前调用参数（只用调用里给的东西，不看历史）
        fixed_args = _normalize_args(name, raw_args)

        # 2) 解析 {"__ref__": "..."} 为真实对象
        args = self.store.resolve_refs(fixed_args)

        # 3) 特定工具的小范围参数清洗（不引入新数据）
        if name == "forecast_tool":
            args = self._sanitize_forecast_args(args)
        if name == "optimize_tool":
            # 只把 rf 从 "2%" 这类格式转成数值，不自动造 mu/Sigma
            if "rf" in args:
                args["rf"] = self._parse_rf(args["rf"])

        # 4) 删除未在 ALLOWED_KW 声明的多余参数，避免脏键
        args = _prune_kwargs(name, args)

        # 5) 检查必填参数是否都在（严格模式）
        must = REQUIRED_KW.get(name, set())
        missing = [k for k in must if k not in args]
        if missing:
            raise T.ToolExecutionError(f"{name} missing required args: {missing}.")

        # 6) 真正调用工具
        try:
            result = fn(**args)
        except Exception as e:
            raise

        # 7) 存入对象仓库 + 提供 JSON-safe preview 给后续工具/模型参考
        ref = self.store.put(result)
        preview = _coerce_preview(result)
        return {"ok": True, "ref": ref, "preview": preview}


    def ask(self, user_text: str) -> str:
        self.messages.append({"role": "user", "content": user_text})
        hops = 0
        while True:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=TOOLS_SPEC,
                tool_choice="auto",
                temperature=0.2,
            )
            msg = resp.choices[0].message

            if msg.tool_calls:
                hops += 1
                if self.verbose:
                    self._log(f"Tool calls: {[tc.function.name for tc in msg.tool_calls]}")
                if hops > 8:
                    text = "Tool hop limit reached. Please narrow the request."
                    self.messages.append({"role": "assistant", "content": text})
                    return text

                self.messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": msg.tool_calls})
                for tc in msg.tool_calls:
                    name = tc.function.name
                    args = json.loads(tc.function.arguments or "{}")
                    try:
                        tool_out = self._run_tool(name, args)
                    except Exception as e:
                        tool_out = {"ok": False, "error": f"{type(e).__name__}: {e}"}
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": name,
                        "content": json.dumps(tool_out, ensure_ascii=False),
                    })
                continue

            text = (msg.content or "").strip()
            self.messages.append({"role": "assistant", "content": text})
            return text


# ----------------------------
# CLI 示例
# ----------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="QuantChat — Chat-style Quant Agent with tools")
    parser.add_argument("--model", type=str, default="gpt-4.1-mini")
    args = parser.parse_args()

    agent = ChatStockAgent(model=args.model, verbose=True)
    print("QuantChat ready. Ask things like:")
    print("  - 'Compute 1d 95% VaR for AAPL with $100k using last 252 days.'")
    print("  - 'Optimize max_sharpe for AAPL, MSFT, NVDA; rf=2%.'")
    print("-"*60)

    while True:
        try:
            q = input("\nYou: ").strip()
            if not q: continue
            if q.lower() in {"exit", "quit"}: break
            ans = agent.ask(q)
            print("\nAgent:", ans)
        except KeyboardInterrupt:
            break
