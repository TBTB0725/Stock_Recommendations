# agent/agent.py
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import re
from openai import OpenAI

# === 你在第二步里实现的工具库 ===
from agent import tools as T

# ------------------------
# 日志配置
# ------------------------
logger = logging.getLogger("agent.brain")
if not logger.handlers:
    _h = logging.StreamHandler()
    _fmt = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s", "%H:%M:%S")
    _h.setFormatter(_fmt)
    logger.addHandler(_h)
logger.setLevel(logging.INFO)

# ------------------------
# LLM 配置
# ------------------------
DEFAULT_MODEL = "gpt-4.1-mini"

# 可被调用的工具注册表（给执行器用）
AVAILABLE_TOOLS: Dict[str, Any] = {
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

# 给 LLM 的工具“说明书”（告诉模型有哪些工具、参数含义）
TOOLS_SPEC = """
You are a financial analysis agent. You must produce a single JSON object with:

{
  "plan": "<brief human summary of your plan>",
  "calls": [
    {"tool": "<one of AVAILABLE_TOOLS>", "args": { ... }},
    ...
  ]
}

AVAILABLE_TOOLS (and expected args):

1) fetch_prices_tool:
   args: {
     "tickers": [str, ...],
     "lookback_days": int (e.g., 252),
     "end": null or "YYYY-MM-DD",
     "pad_ratio": float (default 2.0),
     "auto_business_align": bool (default true),
     "use_adjusted_close": bool (default true)
   }
   returns: price DataFrame (index=dates, columns=tickers)

2) to_returns_tool:
   args: {
     "prices": "<REF: name of previous call result>",
     "method": "log" or "simple" (default "log"),
     "dropna": true
   }
   returns: returns DataFrame

3) forecast_tool:
   args: {
     "prices": "<REF: name of previous call result>",
     "horizon": "1M"|"3M"|"6M" (default "3M"),
     "tune": false
   }
   returns: expected annualized returns (pd.Series indexed by tickers)

4) risk_tool:
   args: {
     "returns": "<REF: name of previous call result>",
     "annualize": true,
     "trading_days_per_year": 252
   }
   returns: annualized covariance matrix (pd.DataFrame)

5) optimize_tool:
   args: {
     "objective": "min_var"|"max_ret"|"max_sharpe",
     "mu_annual": "<REF to pd.Series>" (needed for max_ret / max_sharpe),
     "Sigma_annual": "<REF to pd.DataFrame>" (needed for min_var / max_sharpe),
     "rf": 0.0
   }
   returns: np.ndarray weights

6) evaluate_portfolio_tool:
   args: {
     "name": str,
     "tickers": [str, ...],
     "weights": "<REF to np.ndarray>",
     "capital": float,
     "mu_annual": "<REF to pd.Series>",
     "Sigma_annual": "<REF to pd.DataFrame>",
     "returns_daily": "<REF to pd.DataFrame>",
     "rf_annual": 0.0,
     "var_alpha": 0.05,
     "var_horizon_days": 1,
     "log_returns": true
   }
   returns: PortfolioResult dataclass

7) compile_report_tool:
   args: {
     "results": ["<REF to PortfolioResult>", ...]
   }
   returns: (summary_df, risk_df, allocation_df)

8) fetch_news_tool:
   args: {
     "tickers": [str, ...],
     "lookback_days": 3,
     "per_ticker_count": 15,
     "final_cap": 200,
     "sleep_s": 0.5
   }
   returns: news DataFrame (ticker, headline, source, url, datetime, ...)

9) sentiment_score_titles_tool:
   args: {
     "items": [{"ticker": str, "headline": str}, ...],
     "model_name": "gpt-4.1-mini",
     "api_key": null  // if null, use env OPENAI_API_KEY/OPENAI
   }
   returns: list of dicts each with {"impact": float, "ai_raw": str, "ai_json": dict, ...}

RULES:
- Always output STRICT JSON only (no markdown, no extra text).
- When referring to the output of a previous call as an argument, pass it by a string token
  of the form: {"__ref__": "<CALL_NAME>"} where CALL_NAME is a unique name you assign for that call.
- Include a unique "name" field for each call object, e.g. {"name":"step1_prices", "tool":..., "args":{...}}
- Make sure dependencies are ordered (a call must come after calls whose outputs it references).
- If the user did not provide tickers or dates, choose reasonable defaults and explain them in "plan".
"""

SYSTEM_PROMPT = f"""
You are a careful planner and tool-calling orchestrator.
{TOOLS_SPEC}
"""

def _parse_hints_from_instruction(text: str) -> Dict[str, Any]:
    """
    从自然语言里尽可能解析：tickers, horizon, capital, objective
    注意：为避免把 '1M'（1个月）误识别为 1,000,000，这里对 capital 的解析采取更严格规则：
      - 先剔除时间单位（\d+(W|M|Y)）
      - 仅当出现 $ 或资金语义关键词时才识别金额
    """
    t = text.upper()

    # ---- tickers ----
    candidates = re.findall(r"\b[A-Z][A-Z0-9.\-]{0,4}\b", t)
    blacklist = {
        "FOR", "AND", "WITH", "NEXT", "THEN", "INCLUDE", "CAPITAL", "EVALUATE",
        "ANALYZE", "OPTIMIZE", "MAX", "SHARPE", "RET", "VAR", "RF", "W", "M", "Y"
    }
    tickers = [c for c in candidates if c not in blacklist]
    seen = set()
    tickers = [x for x in tickers if not (x in seen or seen.add(x))][:10]

    # ---- horizon: 1W/1M/3M/6M/12M/1Y ----
    m = re.search(r"\b(\d{1,2})\s*(W|M|Y)\b", t)
    horizon = None
    if m:
        n, u = int(m.group(1)), m.group(2)
        horizon = f"{n}{u}"

    # ---- capital（严格模式，避免把 1M 时间误判为金额）----
    # 先去掉时间表达，防止 '1M'(1 month) 被当金额
    t_no_time = re.sub(r"\b\d{1,2}\s*(W|M|Y)\b", " ", t)

    cap = None
    # 1) 优先匹配带 $ 的写法：$200000 / $200k / $1.5m / $2b
    m = re.search(r"\$\s*([\d,.]+)\s*([KMB])?\b", t_no_time)

    # 2) 若未匹配到，再匹配带资金语义关键词：CAPITAL/BUDGET/INVEST/ALLOCATE/CASH
    #    例如：capital 200k / invest 1.2m
    if not m:
        m = re.search(
            r"(CAPITAL|BUDGET|INVEST|ALLOCATE|CASH)\s*[:=]?\s*\$?\s*([\d,.]+)\s*([KMB])?\b",
            t_no_time
        )

    if m:
        # 如果命中关键词版本，数字在 group(2)；带 $ 版本在 group(1)
        if m.lastindex == 3:
            num_s, unit = m.group(2), (m.group(3) or "").upper()
        else:
            num_s, unit = m.group(1), (m.group(2) or "").upper()

        try:
            num = float(num_s.replace(",", ""))
            mul = 1.0
            if unit == "K":
                mul = 1e3
            elif unit == "M":
                mul = 1e6
            elif unit == "B":
                mul = 1e9
            cap = float(num * mul)
        except Exception:
            cap = None  # 容错：解析失败则忽略

    # ---- objective ----
    objective = None
    if "MAX_SHARPE" in t or ("MAX" in t and "SHARPE" in t):
        objective = "max_sharpe"
    elif "MIN_VAR" in t or "MIN VARIANCE" in t:
        objective = "min_var"
    elif "MAX_RET" in t or "MAX RETURN" in t:
        objective = "max_ret"

    return {
        "tickers": tickers or None,
        "horizon": horizon or None,
        "capital": cap,          # 若无明确资金写法将保持 None，不会误把 1M 当钱
        "objective": objective or None,
        "lookback_days": 252,
        "rf": 0.0,
    }


def _postprocess_plan(plan: Dict[str, Any], hints: Dict[str, Any]) -> Dict[str, Any]:
    """
    用 hints 修补/覆盖 LLM 计划里的关键参数（如 capital、horizon、tickers、objective）。
    同时保证 evaluate_portfolio_tool 一定有 capital>0。
    """
    calls = plan.get("calls", [])
    if not isinstance(calls, list):
        return plan

    # 把 calls 列表转为 name->call 的索引，方便引用
    name_index = {c.get("name", f"step{i+1}"): c for i, c in enumerate(calls)}

    # 1) 修补 fetch_prices 的 tickers / lookback_days
    for c in calls:
        if c.get("tool") == "fetch_prices_tool":
            args = c.setdefault("args", {})
            if hints.get("tickers"):
                args["tickers"] = hints["tickers"]
            args.setdefault("lookback_days", hints.get("lookback_days", 252))

    # 2) 修补 forecast 的 horizon
    for c in calls:
        if c.get("tool") == "forecast_tool":
            args = c.setdefault("args", {})
            if hints.get("horizon"):
                args["horizon"] = hints["horizon"]

    # 3) 修补 optimize 的 objective / rf
    for c in calls:
        if c.get("tool") == "optimize_tool":
            args = c.setdefault("args", {})
            if hints.get("objective"):
                args["objective"] = hints["objective"]
            args.setdefault("rf", hints.get("rf", 0.0))

    # 4) 修补 evaluate_portfolio 的 capital 和 tickers
    eval_calls = [c for c in calls if c.get("tool") == "evaluate_portfolio_tool"]
    if eval_calls:
        ec = eval_calls[-1]
        eargs = ec.setdefault("args", {})
        # capital
        cap = hints.get("capital")
        if cap and (not isinstance(eargs.get("capital"), (int, float)) or eargs.get("capital", 0) <= 0):
            eargs["capital"] = float(cap)
        # tickers
        if hints.get("tickers"):
            eargs["tickers"] = hints["tickers"]
        # 默认名
        eargs.setdefault("name", "Portfolio (LLM)")
    else:
        # 如果计划里没有评估步骤，自动追加一个
        # 这里默认引用常见名字；若名字不同，LLM 计划里也会给出，我们通常已经执行到那一步了
        eval_call = {
            "name": "evaluate_portfolio",
            "tool": "evaluate_portfolio_tool",
            "args": {
                "name": "Portfolio (LLM)",
                "tickers": hints.get("tickers") or [],
                "weights": {"__ref__": "optimize_max_sharpe"},  # 常见名称；如果不同，用户一跑就能看到 KeyError 来修
                "capital": float(hints.get("capital") or 100000.0),
                "mu_annual": {"__ref__": "forecast_returns"},
                "Sigma_annual": {"__ref__": "risk_cov"},
                "returns_daily": {"__ref__": "to_returns"},
                "rf_annual": hints.get("rf", 0.0),
                "var_alpha": 0.05,
                "var_horizon_days": 1,
                "log_returns": True,
            },
        }
        calls.append(eval_call)

    plan["calls"] = calls
    return plan


def _coerce_for_model(x: Any) -> Any:
    """把复杂对象压缩为可读/轻量对象，供可选的总结用（不传回 LLM也可）。"""
    if x is None:
        return None
    if isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, np.ndarray):
        return {"__ndarray__": True, "shape": list(x.shape), "preview": x[:10].tolist()}
    if isinstance(x, pd.Series):
        return {"__series__": True, "index_preview": x.index[:10].tolist(), "values_preview": x.iloc[:10].tolist()}
    if isinstance(x, pd.DataFrame):
        return {
            "__dataframe__": True,
            "shape": [x.shape[0], x.shape[1]],
            "columns": x.columns.tolist()[:20],
            "head": x.head(5).to_dict(orient="split"),
        }
    if is_dataclass(x):
        return {"__dataclass__": True, "fields": asdict(x)}
    if isinstance(x, (list, tuple)):
        return [_coerce_for_model(v) for v in x]
    if isinstance(x, dict):
        return {k: _coerce_for_model(v) for k, v in x.items()}
    return str(type(x))

def _resolve_arg(value: Any, ctx: Dict[str, Any]) -> Any:
    """把 {"__ref__": "call_name"} 解析为真实对象；其他原样返回。"""
    if isinstance(value, dict) and set(value.keys()) == {"__ref__"}:
        key = value["__ref__"]
        if key not in ctx:
            raise KeyError(f"Reference not found: {key}")
        return ctx[key]
    return value

class StockAgent:
    def __init__(self, model: str = DEFAULT_MODEL, verbose: bool = True):
        self.model = model
        self.verbose = verbose
        self.client = OpenAI()  # 依赖环境变量 OPENAI_API_KEY 或 OPENAI

    def _log(self, msg: str):
        if self.verbose:
            logger.info(msg)

    def plan(self, instruction: str, hints: Dict[str, Any]) -> Dict[str, Any]:
        """让 LLM 产出 JSON 计划（单轮），并给出 HINTS 作为先验"""
        self._log("🧠 Planning with LLM...")
        t0 = time.perf_counter()
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "system", "content": "HINTS: " + json.dumps(hints)},
                {"role": "user", "content": instruction},
            ],
        )
        ms = (time.perf_counter() - t0) * 1000
        self._log(f"🧾 Plan received in {ms:.0f} ms")

        content = resp.choices[0].message.content or "{}"
        try:
            plan = json.loads(content)
        except Exception as e:
            raise RuntimeError(f"LLM did not return valid JSON: {e}\nRaw: {content}") from e

        if "calls" not in plan or not isinstance(plan["calls"], list):
            raise RuntimeError(f"Plan JSON missing 'calls': {plan}")
        for i, c in enumerate(plan["calls"], 1):
            if "name" not in c:
                c["name"] = f"step{i}"
        return plan

    def execute(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """按计划顺序执行每个工具调用。"""
        ctx: Dict[str, Any] = {}
        self._log(f"📋 Plan: {plan.get('plan','(no plan text)')}")
        for call in plan["calls"]:
            name = call.get("name")
            tool = call.get("tool")
            args = call.get("args", {})
            if tool not in AVAILABLE_TOOLS:
                raise RuntimeError(f"Unknown tool: {tool}")

            # 解析引用参数
            real_args = {k: _resolve_arg(v, ctx) for k, v in args.items()}

            self._log(f"🔧 {name} → {tool}({list(real_args.keys())})")
            fn = AVAILABLE_TOOLS[tool]
            result = fn(**real_args)  # 这里的错误会由工具层抛出 ToolExecutionError

            # 保存到上下文供后续引用
            ctx[name] = result

        self._log("✅ All steps completed.")
        return ctx

    def run(self, instruction: str) -> Dict[str, Any]:
        hints = _parse_hints_from_instruction(instruction)
        plan = self.plan(instruction, hints)
        plan = _postprocess_plan(plan, hints)

        # 打印完整的“思考产物”——JSON 计划
        self._log("🧩 Full Plan JSON:")
        print(json.dumps(plan, ensure_ascii=False, indent=2))

        results = self.execute(plan)

        # 生成一个人类可读的总结（不走 LLM，直接基于结果对象）
        summary_text = self._human_summary(plan, results)

        # 轻量结构化摘要（用于前端展示）
        summary = {k: _coerce_for_model(v) for k, v in results.items()}
        return {"plan": plan, "results": results, "summary": summary, "text_summary": summary_text}

    def _human_summary(self, plan: Dict[str, Any], ctx: Dict[str, Any]) -> str:
        """基于执行结果，打印权重/指标的简明总结。"""
        # 找到 evaluate_portfolio_tool 的结果
        eval_names = [c["name"] for c in plan.get("calls", []) if c.get("tool") == "evaluate_portfolio_tool"]
        pr = None
        for nm in reversed(eval_names):
            if nm in ctx:
                pr = ctx[nm]
                break
        lines = []
        if pr is not None:
            # PortfolioResult dataclass
            try:
                name = getattr(pr, "name", "Portfolio")
                weights = getattr(pr, "weights", None)  # pd.Series
                exp = getattr(pr, "exp_return_annual", None)
                vol = getattr(pr, "volatility_annual", None)
                sharpe = getattr(pr, "sharpe", None)
                var_alpha = getattr(pr, "var_alpha", None)
                var_value = getattr(pr, "var_value", None)
                vh = getattr(pr, "var_horizon_days", None)

                lines.append(f"Portfolio: {name}")
                if isinstance(weights, pd.Series):
                    topw = weights.sort_values(ascending=False).head(10)
                    lines.append("Top weights:")
                    for k, v in topw.items():
                        lines.append(f"  - {k}: {v:.2%}")
                if exp is not None and vol is not None and sharpe is not None:
                    lines.append(f"Annualized μ={exp:.2%}, σ={vol:.2%}, Sharpe={sharpe:.2f}")
                if var_alpha is not None and var_value is not None and vh is not None:
                    conf = int((1 - var_alpha) * 100)
                    lines.append(f"VaR({conf}%, {vh}d) ≈ {var_value:.2%} loss")
            except Exception as e:
                lines.append(f"(summary failed: {e})")
        else:
            lines.append("No evaluate_portfolio_tool result found.")

        return "\n".join(lines)



# ------------------------
# 便捷 CLI / 调试入口
# ------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Stock Agent (LLM planning + tool execution)")
    parser.add_argument("instruction", type=str, nargs="*", help="Natural language task")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    args = parser.parse_args()

    instruction = " ".join(args.instruction) or \
        "Analyze AAPL and MSFT for the next 1M, include sentiment, optimize max_sharpe, then evaluate with $100000 capital."

    agent = StockAgent(model=args.model, verbose=True)
    out = agent.run(instruction)
    # 打印一个简短摘要（避免把完整 DataFrame 打到控制台）
    print(json.dumps({"plan": out["plan"], "steps": list(out["summary"].keys())}, ensure_ascii=False, indent=2))
    print("\n=== Summary ===\n" + out.get("text_summary", "(no summary)"))

