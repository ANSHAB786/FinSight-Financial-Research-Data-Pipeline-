# finsight.py
#
# bloomberg-style financial research pipeline
# architecture:
#
#   user question
#       │
#       ▼
#   query router  ──── decides which engines to call
#       │
#       ├── financial metrics engine  ── python computes YoY, CAGR, margins
#       ├── sec retriever             ── management commentary, risk, filings
#       ├── news retriever            ── sentiment, catalysts, recent events
#       └── price engine              ── price history chunks
#       │
#       ▼
#   evidence verifier  ── rejects chunks with no financial numbers for metric queries
#       │
#       ▼
#   context builder    ── assembles ranked evidence with citations
#       │
#       ▼
#   qwen2.5 via ollama ── generates answer from grounded context only
#       │
#       ▼
#   numeric validator  ── flags any % not present in context
#       │
#       ▼
#   answer
#
# the llm is the last step, not the calculator.
# python computes every financial ratio before the llm sees anything.

import feedparser
import html
import logging
import nltk
import numpy as np
import os
import re
import requests
import schedule
import threading
import time
import unicodedata
import yfinance as yf

from bs4 import BeautifulSoup
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from supabase import create_client, Client
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, List, Optional, Tuple
import torch

load_dotenv()
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

SEC_HEADER   = {"User-Agent": "Anshab Shaikh shaikhanshab786@gmail.com", "Accept-Encoding": "gzip, deflate"}
OLLAMA_BASE  = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b-instruct")
FRED_API_KEY = os.getenv("FRED_API_KEY", "")
BEA_API_KEY  = os.getenv("BEA_API_KEY", "")
BLS_API_KEY  = os.getenv("BLS_API_KEY", "")

FRED_SERIES = {
    "fed_funds_rate": "FEDFUNDS",
    "10y_treasury"  : "DGS10",
    "2y_treasury"   : "DGS2",
    "cpi"           : "CPIAUCSL",
    "core_pce"      : "PCEPILFE",
    "unemployment"  : "UNRATE",
    "gdp_growth"    : "A191RL1Q225SBEA",
}

BLS_SERIES = {
    "nonfarm_payrolls": "CES0000000001",
    "cpi_all"         : "CUUR0000SA0",
    "ppi"             : "WPUFD4",
}
# BEA NIPA table codes — each maps to a specific line item within that table
BEA_SERIES = {
    "real_gdp_growth"   : {"table": "T10101", "line": "1"},   # % change, real GDP
    "corporate_profits" : {"table": "T61600D", "line": "1"},  # corporate profits, $
    "personal_income"   : {"table": "T20600", "line": "1"},   # personal income, $
    "trade_balance"     : {"table": "T40100", "line": "1"},   # net exports, $
}

# ──────────────────────────────────────────────────────────────────────────────
# setup
# ──────────────────────────────────────────────────────────────────────────────

def _make_logger():
    os.makedirs("logs", exist_ok=True)
    fname = f"logs/finsight_{datetime.now().strftime('%Y%m%d')}.log"
    logging.basicConfig(
        level    = logging.INFO,
        format   = "%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt  = "%H:%M:%S",
        handlers = [logging.FileHandler(fname), logging.StreamHandler()]
    )
    return logging.getLogger("finsight")

log = _make_logger()

log.info("loading finbert...")
_fb_tok   = AutoTokenizer.from_pretrained("ProsusAI/finbert")
_fb_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
_fb_lbls  = ["positive", "negative", "neutral"]

log.info("loading bge embedder + reranker...")
EMBEDDER = SentenceTransformer("BAAI/bge-base-en-v1.5")
RERANKER = CrossEncoder("BAAI/bge-reranker-large")
log.info("models ready — qwen2.5 runs via ollama")

def _make_db() -> Client:
    url = os.getenv("SUPABASE_URL", "")
    key = os.getenv("SUPABASE_KEY", "")
    if not url or not key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_KEY must be in .env")
    return create_client(url, key)

DB: Client = _make_db()

# bm25 index cache — rebuilt per ticker, reused across queries
_bm25_cache: Dict[str, Tuple[BM25Okapi, list]] = {}

# financial metrics cache — computed once per ticker, invalidated on refresh
_metrics_cache: Dict[str, dict] = {}


# ──────────────────────────────────────────────────────────────────────────────
# utilities
# ──────────────────────────────────────────────────────────────────────────────

def fmt(v) -> str:
    if v is None or v == 0:
        return "N/A"
    if abs(v) >= 1e12: return f"${v/1e12:.2f}T"
    if abs(v) >= 1e9:  return f"${v/1e9:.2f}B"
    if abs(v) >= 1e6:  return f"${v/1e6:.2f}M"
    return f"${v:,.2f}"


def pct(v) -> str:
    if v is None:
        return "N/A"
    return f"{v:.2f}%"


def make_cid(ticker: str, stype: str, idx: int) -> str:
    return f"{ticker}_{stype}_{idx}_{datetime.now().strftime('%Y%m%d%H%M%S')}"


def detect_lang(text: str) -> str:
    ja = sum(1 for c in text if "\u3040" <= c <= "\u9fff")
    hi = sum(1 for c in text if "\u0900" <= c <= "\u097f")
    n  = max(len(text), 1)
    return "ja" if ja/n > 0.1 else "hi" if hi/n > 0.1 else "en"


def clean(text: str) -> str:
    if not text:
        return ""
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)
    for src, dst in [("\u2014"," - "),("\u2013"," - "),("\u2018","'"),("\u2019","'"),("\u201c",'"'),("\u201d",'"')]:
        text = text.replace(src, dst)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)
    text = re.sub(r"[\u200b\u200c\u200d\ufeff\u00ad]", "", text)
    text = re.sub(r"\.{3,}", "...", text)
    text = re.sub(r"-{3,}|_{3,}|\*{2,}", " ", text)
    text = re.sub(r"(?<!\w)Page \d+(?!\w)", " ", text)
    text = re.sub(r"(?<!\w)[♦•◦▪▸►](?!\w)", "\n- ", text)
    text = re.sub(r"(?<!\w)[→←↑↓](?!\w)", " ", text)  # directional arrows stay stripped, not meaningful as list markers
    return re.sub(r"\s+", " ", text).strip()


def strip_corp(name: str) -> str:
    return " ".join(re.sub(r"\b(Inc\.|Corp\.|Co\.|Ltd\.|LLC|PLC|&|,)\b", "", name).split()) or name


def retry(fn, attempts=3, delay=2):
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:
            if i == attempts - 1:
                raise
            log.warning(f"retry {i+1}: {e}")
            time.sleep(delay)
            delay *= 2


def age_boost(date_str: str, score: float) -> float:
    try:
        age = (datetime.now() - datetime.strptime(date_str[:10], "%Y-%m-%d")).days
        if age < 30:  return score * 1.2
        if age < 90:  return score * 1.1
    except Exception:
        pass
    return score


def finbert(text: str) -> Tuple[str, float]:
    inp = _fb_tok(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    with torch.no_grad():
        out = _fb_model(**inp)
    probs = torch.nn.functional.softmax(out.logits, dim=-1)
    return _fb_lbls[torch.argmax(probs)], round(probs.max().item(), 4)


def check_ollama() -> bool:
    try:
        r      = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        if any(OLLAMA_MODEL in m for m in models):
            log.info(f"ollama ready — {OLLAMA_MODEL} available")
            return True
        log.warning(f"ollama running but {OLLAMA_MODEL} not pulled — run: ollama pull {OLLAMA_MODEL}")
        return False
    except requests.exceptions.ConnectionError:
        log.error(f"ollama not reachable at {OLLAMA_BASE} — run: ollama serve")
        return False
    except Exception as e:
        log.error(f"ollama check: {e}")
        return False


# ──────────────────────────────────────────────────────────────────────────────
# chunk quality + validation
# ──────────────────────────────────────────────────────────────────────────────

_FIN_KW = [
    "revenue","profit","income","growth","risk","market","business","operations",
    "strategy","competition","financial","quarter","fiscal","annual","billion",
    "million","percent","increase","decrease","compared","management","outlook","guidance","earnings",
]

_BOILERPLATE = [
    "table of contents","securities and exchange commission","washington d.c. 20549",
    "pursuant to section 13","registrant's telephone number","check the appropriate box",
    "exact name of registrant","commission file number","exhibit index",
    "incorporated by reference","filed herewith","pursuant to the requirements",
    "has duly caused this report","cover page interactive data","inline xbrl",
    "formatted as inline","rule 406 of regulation","the proxy statement",
]

def classify_chunk_type(text: str) -> str:
    """
    Distinguishes financial-table-like content from narrative prose so
    downstream scoring doesn't apply prose heuristics (numeric-density
    penalty, min-length-50) to number-dense financial data, which is
    exactly the content most queries actually want.
    """
    words = text.split()
    if not words:
        return "narrative"
    numeric_tokens = sum(1 for w in words if re.match(r"^\$?[\d,\.]+%?$", w))
    nratio = numeric_tokens / len(words)
    fin_kw_hits = sum(1 for kw in _FIN_KW if kw in text.lower())
    if nratio > 0.3 and fin_kw_hits >= 2:
        return "tabular"
    return "narrative"

def chunk_quality(text: str) -> float:
    words = text.split()
    n     = len(words)
    ctype = classify_chunk_type(text)

    # tabular chunks can legitimately be short — a clean statement excerpt
    min_len = 15 if ctype == "tabular" else 30
    if n < min_len:
        return 0.0

    score = 1.0
    if ctype == "narrative":
        if n < 50:   score *= 0.6
        elif n < 80: score *= 0.8

    noise = (words.count("—") + words.count("-") + words.count("(") + words.count(")")) / n
    if noise > 0.08:
        score *= (1 - noise)

    # numeric density: penalize ONLY narrative chunks — for tabular content
    # the numbers ARE the value, not noise
    if ctype == "narrative":
        nratio = sum(1 for w in words if re.match(r"^\$?[\d,\.%]+$", w)) / n
        if nratio > 0.5:   score *= 0.4
        elif nratio > 0.3: score *= 0.7

    tl   = text.lower()
    hits = sum(1 for kw in _FIN_KW if kw in tl)
    score = min(score + hits * 0.05, 1.0)
    if "risk factors" in tl or "management's discussion" in tl:
        score = min(score + 0.1, 1.0)

    # top-word repetition: exempt financial vocabulary and numeric tokens —
    # "million" appearing 12x in a statement excerpt isn't noise
    counted = Counter(w for w in words if w.lower() not in _FIN_KW
                       and not re.match(r"^\$?[\d,\.%]+$", w))
    if counted:
        top = counted.most_common(1)[0][1]
        if top / n > 0.08:
            score *= (1 - top / n)

    return round(max(score, 0.0), 3)


def is_junk_chunk(text: str) -> bool:
    """
    Minimal, surgical rejection — only for content with genuinely zero
    retrievable value. Everything else is admitted with a quality score
    from chunk_quality() and left to the retrieval layer's quality-weighted
    boost to prioritize, instead of ingestion-time deletion doing it
    irreversibly (which was previously killing dense financial tables).
    """
    words = text.split()
    if len(words) < 10:
        return True
    if "☐" in text or "☑" in text:  # unfilled checkbox forms — pure noise
        return True
    tl = text.lower()
    boilerplate_hits = sum(1 for b in _BOILERPLATE if b in tl)
    # only reject if it's ALMOST ENTIRELY boilerplate AND short —
    # a couple mentions inside real MD&A shouldn't nuke the whole chunk
    if boilerplate_hits > 2 and len(words) < 60:
        return True
    return False


def dedup(chunks: list) -> list:
    seen, out = set(), []
    for c in chunks:
        words = c["content"].split()
        if len(words) <= 20:
            fp = " ".join(words)  # too short to skip a prefix — fingerprint the whole thing
        else:
            body_start = min(20, len(words))
            fp = " ".join(words[body_start:body_start+80])
        if fp not in seen:
            seen.add(fp)
            out.append(c)
    removed = len(chunks) - len(out)
    if removed:
        log.info(f"dedup: removed {removed} chunks")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# section tagging — every chunk gets a section label at creation time
# this is what retrieval uses to pre-filter before reranking
# ──────────────────────────────────────────────────────────────────────────────

def tag_section(text: str, form_type: str = "", source_type: str = "") -> str:
    t = text.lower()

    if source_type in ("sec_content", "xbrl_data", "sec_filing", ""):
        if "item 1a" in t or "risk factors" in t:             return "risk_factors"
        if "item 7" in t or "management's discussion" in t:   return "md&a"
        if "forward-looking" in t:                             return "forward_looking"
        if "legal proceedings" in t or "litigation" in t:     return "legal"
        if form_type == "8-K" and "board" in t:               return "board"
        if "earnings" in t or "quarterly results" in t:       return "earnings"
        if ("balance sheet" in t or "cash flow" in t) and re.search(r"\$|\d[,.]?\d*\s*(million|billion|thousand)", t): return "financials"
        if "deferred revenue" in t or "contract liability" in t: return "balance_sheet"
        if "reality labs" in t or "family of apps" in t:      return "segment"
        if "reportable segment" in t or "operating segment" in t: return "segment"

    if source_type == "earnings_transcript":
        if "operator" in t or "question" in t:  return "qa_session"
        if "prepared remarks" in t:              return "prepared_remarks"
        if "guidance" in t or "outlook" in t:   return "guidance"
        if "revenue" in t or "earnings" in t:   return "earnings_discussion"
        return "earnings_transcript"

    if source_type == "investor_presentation":
        if "strategy" in t or "strategic" in t: return "strategy"
        if "financial target" in t:              return "financial_targets"
        if "market opportunity" in t:            return "market_opportunity"
        return "investor_presentation"

    if source_type == "structured_metrics":
        if "income statement" in t or "revenue" in t: return "income_statement"
        if "balance sheet" in t or "assets" in t:     return "balance_sheet"
        if "cash flow" in t:                           return "cash_flow"
        if "ratio" in t:                               return "key_ratios"
        return "financial_statement"

    if "news" in source_type:
        if re.search(r"\bearnings (call|release|report|results)\b", t) or "quarterly results" in t: return "earnings"
        if "acquisition" in t or "merger" in t:  return "ma_news"
        if "regulation" in t or "lawsuit" in t:  return "regulatory_news"
        if "ceo" in t or "executive" in t:       return "management_news"
        return "general_news"

    if source_type == "price_history":  return "price_history"
    if source_type == "fundamentals":   return "fundamentals"
    if source_type == "computed_metrics": return "computed_metrics"
    return "general"


# ──────────────────────────────────────────────────────────────────────────────
# financial intelligence layer
#
# this is the biggest thing missing from your current pipeline.
# bloomberg doesn't ask the llm what the gross margin is.
# bloomberg computes it in python and hands it to the llm as fact.
#
# structure:
#   financial_history[ticker] = {
#       "revenue":          {"2025-09-28": 94930000000, "2024-09-28": 90753000000, ...},
#       "gross_profit":     {...},
#       "operating_income": {...},
#       "net_income":       {...},
#       "eps_basic":        {...},
#       "free_cash_flow":   {...},
#       "total_assets":     {...},
#       "total_debt":       {...},
#       "equity":           {...},
#   }
#
#   computed[ticker] = {
#       "gross_margin_latest":        47.2,
#       "operating_margin_latest":    31.4,
#       "net_margin_latest":          24.6,
#       "revenue_yoy":                {"2025":  7.4, "2024": 2.0, ...},
#       "gross_profit_yoy":           {...},
#       "operating_income_yoy":       {...},
#       "revenue_cagr_3y":            4.1,
#       "revenue_cagr_5y":            8.2,
#       "eps_growth_yoy":             {...},
#       "roe":                        1.47,
#       "roa":                        0.28,
#       "debt_to_equity":             1.52,
#       "current_ratio":              0.87,
#       "fcf_growth_yoy":             {...},
#   }
# ──────────────────────────────────────────────────────────────────────────────

CANONICAL_FIELD_MAP = {
    "revenue": ["Total Revenue", "Net Sales", "Total Revenues"],
    "gross_profit": ["Gross Profit"],
    "operating_income": ["Operating Income", "Pretax Income", "Income Before Tax"],
    "net_income": ["Net Income", "Net Income Common Stockholders"],
    "ebitda": ["EBITDA"],
    "eps_basic": ["Basic EPS"],
    "eps_diluted": ["Diluted EPS"],
    "r_and_d": ["Research And Development"],
    "operating_expenses": ["Total Operating Expenses"],
    "net_interest_income": ["Net Interest Income"],
    "provision_credit_losses": ["Provision For Credit Losses", "Provision For Loan Lease Losses"],
    "non_interest_income": ["Non Interest Income", "Total Non Interest Income"],
    "non_interest_expense": ["Non Interest Expense", "Total Non Interest Expense"],
    "total_assets": ["Total Assets"],
    "total_liabilities": ["Total Liabilities Net Minority Interest"],
    "equity": ["Stockholders Equity"],
    "total_debt": ["Total Debt"],
    "cash": ["Cash And Cash Equivalents"],
    "current_assets": ["Current Assets"],
    "current_liabilities": ["Current Liabilities"],
    "long_term_debt": ["Long Term Debt"],
    "retained_earnings": ["Retained Earnings"],
    "deposits": ["Total Deposits"],
    "operating_cf": ["Operating Cash Flow"],
    "free_cash_flow": ["Free Cash Flow"],
    "capex": ["Capital Expenditure"],
    "investing_cf": ["Investing Cash Flow"],
    "financing_cf": ["Financing Cash Flow"],
    "share_repurchase": ["Repurchase Of Capital Stock"],
}

CANONICAL_FIELD_MAP_Q = {
    "revenue_q": CANONICAL_FIELD_MAP["revenue"],
    "gross_profit_q": CANONICAL_FIELD_MAP["gross_profit"],
    "operating_income_q": CANONICAL_FIELD_MAP["operating_income"],
    "net_income_q": CANONICAL_FIELD_MAP["net_income"],
    "eps_basic_q": CANONICAL_FIELD_MAP["eps_basic"],
    "operating_cf_q": CANONICAL_FIELD_MAP["operating_cf"],
    "free_cash_flow_q": CANONICAL_FIELD_MAP["free_cash_flow"],
    "capex_q": CANONICAL_FIELD_MAP["capex"],
}


def _extract_canonical(stmt, canonical_key: str, field_map: dict) -> Optional[str]:
    for row_name in field_map.get(canonical_key, []):
        if row_name in stmt.index:
            return row_name
    return None


def build_financial_history(ticker: str) -> dict:
    """
    Pulls multi-period financial data from yfinance.
    Returns a structured dict with time-series for every key metric.
    Row lookup goes through CANONICAL_FIELD_MAP so a bank's "Net Interest
    Income" and a SaaS company's "Total Revenue" both land in
    history["revenue"] via the same code path — no sector-specific branching.
    """
    log.info(f"building financial history: {ticker}")
    t = yf.Ticker(ticker)
    out = defaultdict(dict)

    try:
        info = t.info
        sector = info.get("sector", "Unknown")
        industry = info.get("industry", "Unknown")
        log.info(f"{ticker}: sector={sector} industry={industry}")
    except Exception:
        sector, industry = "Unknown", "Unknown"

    def _pull(stmt_fn, field_map, max_cols, label):
        try:
            stmt = stmt_fn()
            if stmt is None or stmt.empty:
                return
            for col in stmt.columns[:max_cols]:
                period = str(col.date()) if hasattr(col, "date") else str(col)[:10]
                for canonical_key in field_map:
                    row_name = _extract_canonical(stmt, canonical_key, field_map)
                    if row_name is None:
                        continue
                    v = stmt.loc[row_name, col]
                    if v is not None and not (isinstance(v, float) and np.isnan(v)):
                        out[canonical_key][period] = float(v)
        except Exception as e:
            log.warning(f"{label} failed for {ticker}: {e}")

    _pull(lambda: t.income_stmt, CANONICAL_FIELD_MAP, 6, "income stmt")
    _pull(lambda: t.quarterly_income_stmt, CANONICAL_FIELD_MAP_Q, 8, "quarterly income")
    _pull(lambda: t.balance_sheet, CANONICAL_FIELD_MAP, 4, "balance sheet")
    _pull(lambda: t.cashflow, CANONICAL_FIELD_MAP, 4, "cashflow")
    _pull(lambda: t.quarterly_cashflow, CANONICAL_FIELD_MAP_Q, 8, "quarterly cashflow")

    out["_sector"] = sector
    out["_industry"] = industry
    return dict(out)


# ──────────────────────────────────────────────────────────────────────────────
# trend analysis — classifies a metric's time series into a pattern
# (steady growth / recovering / declining / volatile / stable) with a
# confidence and a plain-english reason. used by every engine below.
# ──────────────────────────────────────────────────────────────────────────────

def get_metrics(ticker: str, company: str, fundamentals: dict = None, force_refresh: bool = False) -> dict:
    """
    Entry point for the financial intelligence layer — everything else in
    this section exists to support this one function.

    Returns cached metrics or builds them fresh:
      build_financial_history()  -> raw time series from yfinance
      compute_all_metrics()      -> ratios, trends, and every reasoning
                                     engine (capital allocation, evidence,
                                     contradictions, thesis, risk, scenario,
                                     valuation) built on top of that history
      metrics_to_text()          -> (called later, by build_metrics_chunk)
                                     turns the dict above into the chunk
                                     the LLM actually reads

    Invalidated when refresh_price_and_metrics is called.
    fundamentals (PE ratio etc) is optional — passed through to the
    valuation engine when available, omitted otherwise.
    """
    if ticker in _metrics_cache and not force_refresh:
        return _metrics_cache[ticker]
    history = build_financial_history(ticker)
    metrics = compute_all_metrics(history, ticker, company, fundamentals=fundamentals)
    _metrics_cache[ticker] = metrics
    return metrics


def build_trend_analysis(series: dict, metric_name: str = "") -> dict:
    if len(series) < 2:
        return {
            "classification": "Insufficient Data",
            "direction": "Unknown",
            "confidence": "Low",
            "strength": 0,
            "reason": "Not enough historical data.",
            "growth_rates": [],
            "momentum": "Unknown",
            "inflection": False,
            "rolling_avg_growth_4p": None,
        }

    periods = sorted(series.keys())
    growth = []

    # Safely compute growth rates
    for i in range(1, len(periods)):
        prev, curr = series[periods[i-1]], series[periods[i]]
        if prev is None or curr is None or prev == 0:
            continue
        try:
            growth.append(round((curr - prev) / abs(prev) * 100, 2))
        except Exception:
            continue

    if not growth:
        return {
            "classification": "Insufficient Data",
            "direction": "Unknown",
            "confidence": "Low",
            "strength": 0,
            "reason": "Not enough valid data points.",
            "growth_rates": [],
            "momentum": "Unknown",
            "inflection": False,
            "rolling_avg_growth_4p": None,
        }

    latest = growth[-1]
    sign_changes = sum(1 for i in range(1, len(growth)) if (growth[i] > 0) != (growth[i-1] > 0))
    is_volatile = len(growth) >= 3 and sign_changes >= len(growth) - 1

    # Classification logic
    if is_volatile:
        classification = "Volatile"
    elif all(g > 0 for g in growth):
        classification = "Steady Growth"
    elif latest > 0 and growth[0] < 0:
        classification = "Recovering Growth"
    elif latest < 0 and growth[0] > 0:
        classification = "Declining"
    elif all(g < 0 for g in growth):
        classification = "Continuous Decline"
    else:
        classification = "Stable"

    direction = "Positive" if latest > 0 else "Negative" if latest < 0 else "Neutral"

    if is_volatile:
        confidence = "Low"
    elif abs(latest) >= 8:
        confidence = "High"
    elif abs(latest) >= 3:
        confidence = "Medium"
    else:
        confidence = "Low"

    strength = min(10, round(abs(latest) / 2))

    reasons = {
        "Recovering Growth": "Business recovered after an earlier decline.",
        "Steady Growth": "Growth remained positive across all periods.",
        "Declining": "Growth has weakened after previous expansion.",
        "Continuous Decline": "Business has contracted across every reported period.",
        "Volatile": "Performance swings between gains and losses period to period.",
    }
    reason = reasons.get(classification, "No dominant long-term trend detected.")

    # Momentum & inflection
    momentum = "Unknown"
    inflection = False
    if len(growth) >= 2:
        deltas = [growth[i] - growth[i-1] for i in range(1, len(growth))]
        avg_delta = sum(deltas) / len(deltas)
        if abs(avg_delta) < 1.0:
            momentum = "Steady"
        elif avg_delta > 0:
            momentum = "Accelerating"
        else:
            momentum = "Decelerating"
        if len(deltas) >= 2:
            inflection = any((deltas[i] > 0) != (deltas[i-1] > 0) for i in range(1, len(deltas)))

    rolling_avg_growth = None
    if len(growth) >= 4:
        rolling_avg_growth = round(sum(growth[-4:]) / 4, 2)

    return {
        "classification": classification,
        "direction": direction,
        "confidence": confidence,
        "strength": strength,
        "reason": reason,
        "growth_rates": growth,
        "momentum": momentum,
        "inflection": inflection,
        "rolling_avg_growth_4p": rolling_avg_growth,
    }



def build_financial_insight(analysis: dict) -> dict:
    cls = analysis["classification"]
    table = {
        "Steady Growth": {
            "signal": "Bullish", "risk": "Low",
            "business_insight": "The business is delivering consistent growth without major volatility.",
            "opportunity": "Sustainable expansion supports long-term earnings growth.",
            "investor_interpretation": "Long-term investors generally favor businesses with predictable growth.",
            "executive_summary": "Financial performance remains consistently strong.",
        },
        "Recovering Growth": {
            "signal": "Bullish", "risk": "Medium",
            "business_insight": "The company has recovered from a previous slowdown and growth is accelerating.",
            "opportunity": "Improving momentum could translate into stronger profitability in future periods.",
            "investor_interpretation": "Recovery trends often indicate improving business fundamentals.",
            "executive_summary": "Business momentum has shifted from contraction toward expansion.",
        },
        "Declining": {
            "signal": "Bearish", "risk": "High",
            "business_insight": "Core financial performance is deteriorating over time.",
            "opportunity": "Management must improve execution to restore growth.",
            "investor_interpretation": "Persistent declines usually reduce investor confidence.",
            "executive_summary": "Financial performance continues to weaken.",
        },
        "Continuous Decline": {
            "signal": "Bearish", "risk": "High",
            "business_insight": "The metric has contracted in every reported period with no recovery signal.",
            "opportunity": "A structural turnaround would be required to reverse the trend.",
            "investor_interpretation": "Sustained decline materially raises downside risk.",
            "executive_summary": "Performance has deteriorated consistently across the reporting history.",
        },
        "Volatile": {
            "signal": "Neutral", "risk": "High",
            "business_insight": "Financial performance fluctuates significantly between reporting periods.",
            "opportunity": "Operational stability could unlock stronger valuation.",
            "investor_interpretation": "High volatility increases uncertainty for investors.",
            "executive_summary": "Performance lacks consistency.",
        },
    }
    default = {
        "signal": "Neutral", "risk": "Medium",
        "business_insight": "No dominant long-term financial pattern was detected.",
        "opportunity": "Future financial reports should be monitored.",
        "investor_interpretation": "Current trend provides limited conviction.",
        "executive_summary": "Trend remains inconclusive.",
    }
    return table.get(cls, default)


# ──────────────────────────────────────────────────────────────────────────────
# company health score — replaces ad-hoc thresholds with a single 0-10 score
# ──────────────────────────────────────────────────────────────────────────────

SECTOR_PROFILES = {
    "bank": {
        "gics_sectors": ["Financial Services"],
        "gics_industries": ["Banks", "Banks—Regional", "Banks—Diversified"],
        "primary_metrics": ["net_interest_income", "roe", "roa", "tier1_capital_ratio",
                            "net_interest_margin", "provision_credit_losses"],
        "not_applicable": ["gross_margin", "current_ratio", "debt_to_equity"],
        # debt/equity is structurally high for banks (deposits count as liabilities) —
        # don't penalize it the way you would for an industrial company
        "leverage_note": "Debt/equity is not a meaningful risk signal for banks — "
                          "deposits and borrowings are core to the business model, "
                          "not distress. Use tier1_capital_ratio instead.",
        "health_thresholds": {"roe_strong": 12, "roe_weak": 6},
    },
    "insurance": {
        "gics_sectors": ["Financial Services"],
        "gics_industries": ["Insurance—Life", "Insurance—Property & Casualty",
                            "Insurance—Diversified", "Insurance—Reinsurance"],
        "primary_metrics": ["combined_ratio", "roe", "net_premiums_earned"],
        "not_applicable": ["gross_margin", "current_ratio"],
        "health_thresholds": {"roe_strong": 12, "roe_weak": 6},
    },
    "reit": {
        "gics_sectors": ["Real Estate"],
        "gics_industries": [],  # match on sector alone for REITs
        "primary_metrics": ["ffo", "affo", "revenue", "debt_to_equity"],
        "not_applicable": ["gross_margin", "net_margin", "eps_basic"],
        # REITs are legally required to distribute ~90% of taxable income —
        # high leverage and high payout ratio are normal, not warning signs
        "leverage_note": "REITs carry structurally high leverage by design "
                          "(real estate financing) — compare against sector peers, "
                          "not general-industrial thresholds.",
        "health_thresholds": {"ffo_growth_strong": 5, "ffo_growth_weak": -2},
    },
    "utility": {
        "gics_sectors": ["Utilities"],
        "gics_industries": [],
        "primary_metrics": ["revenue", "operating_margin", "debt_to_equity", "roe"],
        "not_applicable": [],
        "leverage_note": "Utilities carry high, rate-regulated leverage as standard "
                          "practice — elevated debt/equity here is not comparable "
                          "to a general industrial company's leverage risk.",
        "health_thresholds": {"debt_to_equity_normal_max": 2.5},
    },
    "energy_producer": {
        "gics_sectors": ["Energy"],
        "gics_industries": [],
        "primary_metrics": ["revenue", "free_cash_flow", "capex", "debt_to_equity"],
        "not_applicable": ["gross_margin"],
        "cyclicality_note": "Revenue and margins are heavily commodity-price driven — "
                             "single-period trends are less informative than "
                             "multi-year context.",
        "health_thresholds": {},
    },
    "biotech_prerevenue": {
        "gics_sectors": ["Healthcare"],
        "gics_industries": ["Biotechnology", "Drug Manufacturers—Specialty & Generic"],
        # detected dynamically below (near-zero revenue + heavy R&D), not by
        # industry tag alone, since most biotech tickers DO have revenue
        "primary_metrics": ["cash_runway_quarters", "r_and_d", "operating_cf"],
        "not_applicable": ["pe_ratio", "gross_margin", "revenue_cagr_3y"],
        "health_thresholds": {},
    },
    "standard": {
        "gics_sectors": [],  # fallback for everything else — tech, industrials,
        "gics_industries": [],  # retail, healthcare (non-biotech), consumer, etc.
        "primary_metrics": ["revenue", "gross_margin", "operating_margin", "roe",
                            "debt_to_equity", "current_ratio", "free_cash_flow"],
        "not_applicable": [],
        "health_thresholds": {"debt_to_equity_normal_max": 1.5},
    },
}
 
 
def classify_sector_profile(ticker: str, sector: str = "", industry: str = "",
                             fundamentals: dict = None) -> str:
    """
    Returns the key into SECTOR_PROFILES this ticker should use. Checks
    industry first (more specific than sector), falls back to sector,
    then falls back to "standard" for anything unmatched — meaning a
    ticker in an industry we haven't explicitly mapped still gets sane
    general-purpose thresholds instead of crashing or defaulting to bank
    rules by accident.
 
    Biotech pre-revenue detection is dynamic rather than industry-tag-only:
    a company tagged "Biotechnology" that already has real product revenue
    (e.g. Vertex, Regeneron) should use "standard" metrics, not the
    cash-runway profile meant for pre-revenue clinical-stage companies.
    """
    fundamentals = fundamentals or {}
 
    for key, profile in SECTOR_PROFILES.items():
        if industry and industry in profile["gics_industries"]:
            if key == "biotech_prerevenue":
                revenue = fundamentals.get("revenue", 0) or 0
                if revenue > 50_000_000:  # has real commercial revenue — not pre-revenue
                    continue
            return key
 
    for key, profile in SECTOR_PROFILES.items():
        if sector and sector in profile["gics_sectors"] and key != "biotech_prerevenue":
            return key
 
    return "standard"
 
 
def get_sector_context(ticker: str, sector: str = None, industry: str = None,
                        revenue: float = None) -> dict:
    """
    Single entry point — call once per ticker in ingest()/get_metrics(),
    pass the result through to compute_all_metrics(), build_company_health(),
    score_capital_allocation(), and build_risk_profile() so every scoring
    function applies the right thresholds instead of one-size-fits-all.

    Accepts sector/industry/revenue directly when the caller already has
    them (compute_all_metrics passes them from build_financial_history's
    info fetch) — avoids a second yf.Ticker().info round trip for data
    that's already been pulled once.
    """
    if sector is None or industry is None or revenue is None:
        try:
            info = yf.Ticker(ticker).info
            sector = sector if sector is not None else info.get("sector", "")
            industry = industry if industry is not None else info.get("industry", "")
            revenue = revenue if revenue is not None else info.get("totalRevenue", 0)
        except Exception as e:
            log.warning(f"sector context lookup failed for {ticker}: {e}")
            sector, industry, revenue = sector or "", industry or "", revenue or 0

    fundamentals = {"revenue": revenue or 0}
    profile_key = classify_sector_profile(ticker, sector, industry, fundamentals)
    profile = SECTOR_PROFILES[profile_key]
 
    return {
        "profile_key": profile_key,
        "sector": sector,
        "industry": industry,
        "primary_metrics": profile["primary_metrics"],
        "not_applicable": profile["not_applicable"],
        "leverage_note": profile.get("leverage_note", ""),
        "cyclicality_note": profile.get("cyclicality_note", ""),
        "health_thresholds": profile.get("health_thresholds", {}),
    }
 
 
# ══════════════════════════════════════════════════════════════════════════
# build_company_health() — REPLACE existing function with this sector-aware
# version. Signature changes: now takes sector_ctx.
# ══════════════════════════════════════════════════════════════════════════
 
HEALTH_SCORE_WEIGHTS = {
    "revenue_improving"      : 2,
    "gross_margin_strong"    : 2,   # gm > 40
    "gross_margin_ok"        : 1,   # gm > 30
    "gross_margin_strong_thresh": 40,
    "gross_margin_ok_thresh"    : 30,
    "roe_strong"              : 2,
    "roe_ok"                  : 1,
    "debt_conservative"       : 2,  # debt < normal_max * debt_conservative_ratio
    "debt_ok"                 : 1,  # debt < normal_max
    "debt_conservative_ratio" : 0.6,
    "liquidity_healthy"       : 2,  # current >= 1
    "liquidity_acceptable"    : 1,  # current >= 0.8
    "reit_ffo_strong"         : 3,
    "max_score"               : 10,
}
 
CAPITAL_ALLOCATION_WEIGHTS = {
    "buybacks": {"roe_cap": 40, "roe_pts": 35, "revenue_pts": 25,
                 "fcf_pts": 20, "debt_pts": 10, "debt_thresh": 1.5,
                 "fcf_trend_pts": 10},
    "dividend": {"roe_cap": 25, "roe_pts": 30, "revenue_steady_pts": 30,
                 "revenue_positive_pts": 15, "fcf_pts": 20,
                 "debt_low_pts": 20, "debt_low_thresh": 1.0,
                 "debt_mid_pts": 10, "debt_mid_thresh": 2.0},
    "reinvest_growth": {"growth_pts": 40, "margin_cap": 60, "margin_pts": 30,
                        "fcf_pts": 20, "roe_thresh": 15, "roe_pts": 10},
    "reduce_debt": {"debt_cap": 4, "debt_pts": 60, "liquidity_pts": 25,
                    "revenue_neg_pts": 15},
    # confidence bands on winner-vs-runner-up score margin
    "confidence_high_margin": 25,
    "confidence_medium_margin": 10,
}
 
 
def build_company_health(metrics: dict, sector_ctx: dict,
                          weights: dict = HEALTH_SCORE_WEIGHTS) -> dict:
    score = 0
    reasons = []
    profile = sector_ctx["profile_key"]
    thresholds = sector_ctx["health_thresholds"]
    na = sector_ctx["not_applicable"]
 
    revenue = metrics.get("revenue_analysis", {})
    if revenue.get("classification") in ("Steady Growth", "Recovering Growth"):
        score += weights["revenue_improving"]
        reasons.append("Revenue trend is improving.")
 
    if "gross_margin" not in na:
        gm = metrics.get("gross_margin_latest")
        if gm is not None:
            if gm > weights["gross_margin_strong_thresh"]:
                score += weights["gross_margin_strong"]
                reasons.append("Gross margins are strong.")
            elif gm > weights["gross_margin_ok_thresh"]:
                score += weights["gross_margin_ok"]
 
    roe = metrics.get("roe")
    if roe is not None:
        roe_strong = thresholds.get("roe_strong", 20)
        roe_weak = thresholds.get("roe_weak", 10)
        if roe > roe_strong:
            score += weights["roe_strong"]
            reasons.append("Return on equity is excellent.")
        elif roe > roe_weak:
            score += weights["roe_ok"]
 
    if "debt_to_equity" not in na:
        debt = metrics.get("debt_to_equity")
        normal_max = thresholds.get("debt_to_equity_normal_max", 1.0)
        if debt is not None:
            if debt < normal_max * weights["debt_conservative_ratio"]:
                score += weights["debt_conservative"]
                reasons.append("Debt levels are conservative.")
            elif debt < normal_max:
                score += weights["debt_ok"]
                reasons.append(f"Debt is within normal range for a {profile.replace('_',' ')}.")
    elif sector_ctx.get("leverage_note"):
        reasons.append(sector_ctx["leverage_note"])
 
    if "current_ratio" not in na:
        current = metrics.get("current_ratio")
        if current is not None:
            if current >= 1:
                score += weights["liquidity_healthy"]
                reasons.append("Liquidity is healthy.")
            elif current >= 0.8:
                score += weights["liquidity_acceptable"]
                reasons.append("Liquidity is acceptable.")
 
    if profile == "reit":
        ffo_growth = metrics.get("ffo_yoy_latest")
        if ffo_growth is not None:
            if ffo_growth > thresholds.get("ffo_growth_strong", 5):
                score += weights["reit_ffo_strong"]
                reasons.append("FFO growth is strong.")
            elif ffo_growth < thresholds.get("ffo_growth_weak", -2):
                reasons.append("FFO is declining — the primary REIT earnings signal is weakening.")
 
    max_score = weights["max_score"]
    if score >= max_score * 0.9:   overall = "Excellent"
    elif score >= max_score * 0.7: overall = "Strong"
    elif score >= max_score * 0.5: overall = "Average"
    else:                          overall = "Weak"
 
    return {
        "overall_health": overall, "overall_score": score, "max_score": max_score,
        "reasoning": reasons, "sector_profile": profile,
    }
 
 
# ══════════════════════════════════════════════════════════════════════════
# 1. Capital allocation — add dilution and dividend-history signals
# ══════════════════════════════════════════════════════════════════════════
#
# Still Version 1 — WACC, cost of debt, interest rates, and full acquisition
# strategy need external data (rates) or unstructured extraction
# (M&A intent from filings/calls) that isn't wired up yet. What IS available
# cheaply from data you already pull: buyback trend from the cash flow
# statement, and current dividend yield/payout ratio from fundamentals.
# Adding those closes the biggest gap without inventing anything.
 
def score_capital_allocation(metrics: dict, fundamentals: dict = None,
                              weights: dict = CAPITAL_ALLOCATION_WEIGHTS) -> dict:
    fundamentals = fundamentals or {}
    revenue   = metrics.get("revenue_analysis", {})
    roe       = metrics.get("roe") or 0
    debt      = metrics.get("debt_to_equity")
    current   = metrics.get("current_ratio") or 0
    fcf       = metrics.get("fcf_latest") or 0
    fcf_trend = metrics.get("fcf_analysis", {})
    gm        = metrics.get("gross_margin_latest") or 0
 
    # buyback trend now computed upstream in compute_all_metrics() from the
    # actual cash flow statement — positive = repurchases grew YoY
    buyback_yoy = metrics.get("buyback_yoy_latest")          # % change, or None
    dividend_yield = fundamentals.get("dividend_yield") or 0
    payout_ratio   = fundamentals.get("payout_ratio") or 0  # 0-1
 
    rev_positive = revenue.get("direction") == "Positive"
    rev_steady   = revenue.get("classification") in ("Steady Growth", "Stable")
 
    candidates = {}
    w = weights["buybacks"]
    s  = min(roe, w["roe_cap"]) / w["roe_cap"] * w["roe_pts"]
    s += w["revenue_pts"] if rev_positive else 0
    s += w["fcf_pts"] if fcf > 0 else 0
    s += w["debt_pts"] if (debt is not None and debt < w["debt_thresh"]) else 0
    s += w["fcf_trend_pts"] if fcf_trend.get("direction") == "Positive" else 0
    # NEW: reward an existing, increasing buyback program; penalize a
    # company that's actively diluting (negative buyback_yoy = net issuance)
    if buyback_yoy is not None:
        if buyback_yoy > 0: s += 8
        elif buyback_yoy < 0: s -= 8
    candidates["buybacks"] = round(s, 1)
 
    w = weights["dividend"]
    s  = min(roe, w["roe_cap"]) / w["roe_cap"] * w["roe_pts"]
    s += w["revenue_steady_pts"] if rev_steady else (w["revenue_positive_pts"] if rev_positive else 0)
    s += w["fcf_pts"] if fcf > 0 else 0
    s += w["debt_low_pts"] if (debt is not None and debt < w["debt_low_thresh"]) else \
         (w["debt_mid_pts"] if (debt is not None and debt < w["debt_mid_thresh"]) else 0)
    # NEW: an established payout policy is itself evidence for "increase
    # dividend" being the natural next move, distinct from "start a dividend"
    if dividend_yield > 0:
        s += 10
        if payout_ratio and payout_ratio < 0.6:
            s += 5  # room to raise without straining coverage
    candidates["dividend"] = round(s, 1)
 
    w = weights["reinvest_growth"]
    s  = w["growth_pts"] if revenue.get("classification") in ("Steady Growth", "Recovering Growth") else 0
    s += min(gm, w["margin_cap"]) / w["margin_cap"] * w["margin_pts"]
    s += w["fcf_pts"] if fcf > 0 else 0
    s += w["roe_pts"] if roe > w["roe_thresh"] else 0
    candidates["reinvest_growth"] = round(s, 1)
 
    w = weights["reduce_debt"]
    s = 0.0
    if debt is not None:
        s += min(debt, w["debt_cap"]) / w["debt_cap"] * w["debt_pts"]
    if current and current < 1:
        s += w["liquidity_pts"]
    if not rev_positive:
        s += w["revenue_neg_pts"]
    candidates["reduce_debt"] = round(s, 1)
 
    ranked = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    winner, winner_score = ranked[0]
    runner_up, runner_score = ranked[1] if len(ranked) > 1 else (None, 0)
    margin = winner_score - runner_score
 
    if margin >= weights["confidence_high_margin"]:   confidence = "High"
    elif margin >= weights["confidence_medium_margin"]: confidence = "Medium"
    else: confidence = "Low"
 
    labels = {
        "buybacks"       : "Prioritize Share Buybacks",
        "dividend"       : "Increase Dividend",
        "reinvest_growth": "Reinvest Into Growth",
        "reduce_debt"    : "Reduce Debt",
    }
 
    signals_used = ["roe", "revenue_trend", "fcf", "debt_to_equity"]
    if buyback_yoy is not None: signals_used.append("buyback_trend")
    if dividend_yield: signals_used.append("dividend_yield")
 
    return {
        "recommendation": labels[winner],
        "confidence"    : confidence,
        "scores"        : candidates,
        "runner_up"     : labels.get(runner_up, ""),
        "score_margin"  : round(margin, 1),
        "signals_used"  : signals_used,   # NEW — feeds into the confidence
                                           # rework in section 7 below
    }


# ──────────────────────────────────────────────────────────────────────────────
# evidence engine — every recommendation gets a list of supporting signals
# plus a numeric confidence, so the LLM can cite *why*, not just *what*.
# ──────────────────────────────────────────────────────────────────────────────
EXPECTED_METRICS_FOR_COMPLETENESS = [
    "revenue_analysis", "gross_margin_latest", "operating_margin_latest",
    "roe", "roa", "debt_to_equity", "current_ratio", "fcf_latest",
    "eps_analysis", "revenue_cagr_3y",
]

def build_evidence(metrics: dict, sector_ctx: dict = None,
                    contradictions: dict = None) -> dict:
    evidence = []
    claimed  = set()  # signal keys this function claimed — thesis/diagnostics
                       # check this before restating the same fact
 
    revenue = metrics.get("revenue_analysis", {})
    if revenue.get("direction") == "Positive":
        evidence.append(f"Revenue trend is {revenue.get('classification','').lower()}")
        claimed.add("revenue_positive")
 
    roe = metrics.get("roe")
    if roe is not None and roe > 15:
        evidence.append(f"ROE of {roe:.1f}% indicates strong capital efficiency")
        claimed.add("roe_strong")
 
    gm = metrics.get("gross_margin_latest")
    gm_trend = metrics.get("gross_margin_analysis", {})
    if gm is not None and gm_trend.get("direction") == "Positive":
        evidence.append(f"Gross margin improving, latest at {gm:.1f}%")
        claimed.add("gross_margin_improving")
 
    fcf = metrics.get("fcf_latest")
    if fcf is not None and fcf > 0:
        evidence.append("Positive free cash flow generation")
        claimed.add("fcf_positive")
 
    debt = metrics.get("debt_to_equity")
    if debt is not None and debt < 1.5:
        evidence.append(f"Conservative leverage at {debt:.2f}x debt/equity")
        claimed.add("debt_conservative")
 
    current = metrics.get("current_ratio")
    if current is not None and current >= 1:
        evidence.append(f"Healthy liquidity, current ratio {current:.2f}x")
        claimed.add("liquidity_healthy")
 
    # ── data completeness: what fraction of the metrics we'd expect for
    # this sector are actually populated (not None, not excluded by na) ──
    na = set((sector_ctx or {}).get("not_applicable", []))
    expected = [m for m in EXPECTED_METRICS_FOR_COMPLETENESS
                if not any(x in m for x in na)]
    populated = 0
    for m in expected:
        v = metrics.get(m)
        if isinstance(v, dict):
            populated += 1 if v.get("classification") not in (None, "Insufficient Data") else 0
        else:
            populated += 1 if v is not None else 0
    completeness = populated / len(expected) if expected else 0.0
 
    # ── period coverage: trend classifications built on <3 growth points
    # are weaker evidence than ones built on 5+ ──
    period_counts = [len(metrics.get(k, {}).get("growth_rates", []))
                      for k in ("revenue_analysis", "eps_analysis", "fcf_analysis")]
    avg_periods = sum(period_counts) / len(period_counts) if period_counts else 0
    period_factor = min(avg_periods / 4, 1.0)  # 4+ periods = full credit
 
    # ── contradiction penalty ──
    contradiction_penalty = 0.0
    if contradictions and contradictions.get("mixed_signals"):
        contradiction_penalty = 0.25  # knock a quarter off confidence
 
    raw_evidence_score = min(len(evidence) * 1.5, 10) / 10  # 0-1
 
    confidence_score = raw_evidence_score * 0.4 + completeness * 0.3 + period_factor * 0.3
    confidence_score = round(max(confidence_score - contradiction_penalty, 0) * 10, 1)  # back to /10
 
    return {
        "evidence": evidence,
        "evidence_count": len(evidence),
        "claimed_signals": claimed,             # NEW — thesis/diagnostics dedupe against this
        "confidence_score": confidence_score,   # now /10, factors in completeness+periods+contradictions
        "confidence_components": {
            "evidence_strength": round(raw_evidence_score, 2),
            "data_completeness": round(completeness, 2),
            "period_coverage": round(period_factor, 2),
            "contradiction_penalty": contradiction_penalty,
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
# contradiction detector — flags when signals disagree (e.g. revenue up but
# margins down), which should reduce confidence in any single-line verdict
# ──────────────────────────────────────────────────────────────────────────────

def detect_contradictions(metrics: dict) -> dict:
    signals = {
        "revenue"         : metrics.get("revenue_analysis", {}).get("direction"),
        "gross_margin"    : metrics.get("gross_margin_analysis", {}).get("direction"),
        "operating_margin": metrics.get("operating_margin_analysis", {}).get("direction"),
        "fcf"             : metrics.get("fcf_analysis", {}).get("direction"),
        "eps"             : metrics.get("eps_analysis", {}).get("direction"),
    }
    known = {k: v for k, v in signals.items() if v in ("Positive", "Negative")}
    if len(known) < 2:
        return {"mixed_signals": False, "contradictions": [], "note": "insufficient data to compare signals"}

    positives = [k for k, v in known.items() if v == "Positive"]
    negatives = [k for k, v in known.items() if v == "Negative"]

    contradictions = []
    if positives and negatives:
        contradictions.append(
            f"{', '.join(positives)} trending positive while {', '.join(negatives)} trending negative"
        )

    return {
        "mixed_signals" : bool(contradictions),
        "contradictions": contradictions,
        "positive_signals": positives,
        "negative_signals": negatives,
    }


# ──────────────────────────────────────────────────────────────────────────────
# investment thesis — bull/bear/neutral case built from the same signals,
# so the LLM has structured material instead of having to invent a narrative
# ──────────────────────────────────────────────────────────────────────────────

def build_investment_thesis(metrics: dict, health: dict, contradictions: dict,
                             claimed_signals: set = None) -> dict:
    """
    Only adds a bull/bear line when it crosses a MORE EXTREME threshold than
    evidence already claimed (e.g. evidence flags roe>15, thesis only adds
    its own line if roe>20 AND evidence didn't already cover it at a
    comparable tier) — or when it's a genuinely distinct signal (capital
    allocation, buyback trend, contradictions) that evidence never touches.
    This stops "ROE is strong" from being independently restated as if it
    were three separate corroborating facts across the report.
    """
    claimed_signals = claimed_signals or set()
    revenue = metrics.get("revenue_analysis", {})
    roe     = metrics.get("roe")
    gm      = metrics.get("gross_margin_latest")
    debt    = metrics.get("debt_to_equity")
    buyback = metrics.get("buyback_yoy_latest")

    bull = []
    bear = []

    if revenue.get("direction") == "Positive" and "revenue_positive" not in claimed_signals:
        bull.append(f"Revenue is {revenue.get('classification','').lower()}")
    elif revenue.get("direction") == "Negative":
        bear.append(f"Revenue is {revenue.get('classification','').lower()}")

    # only add ROE as a THESIS point if it's extreme enough that evidence's
    # roe>15 threshold doesn't already cover the same ground
    if roe is not None:
        if roe > 20 and "roe_strong" not in claimed_signals:
            bull.append(f"ROE of {roe:.1f}% is well above average")
        elif roe < 8:
            bear.append(f"ROE of {roe:.1f}% is relatively weak")

    if gm is not None:
        if gm > 45 and "gross_margin_improving" not in claimed_signals:
            bull.append(f"Gross margin of {gm:.1f}% reflects strong pricing power")
        elif gm < 25:
            bear.append(f"Gross margin of {gm:.1f}% is thin")

    if debt is not None:
        if debt > 2: bear.append(f"Debt/equity of {debt:.2f}x is elevated")
        elif debt < 0.5 and "debt_conservative" not in claimed_signals:
            bull.append(f"Debt/equity of {debt:.2f}x is conservative")

    # buyback trend — genuinely distinct signal, never covered by evidence
    if buyback is not None:
        if buyback > 10:
            bull.append(f"Share buybacks increased {buyback:.1f}% YoY, signaling management confidence")
        elif buyback < -10:
            bear.append(f"Share buybacks declined {abs(buyback):.1f}% YoY")

    if contradictions.get("mixed_signals"):
        bear.append("Mixed signals across metrics reduce conviction")

    catalyst = bull[0] if bull else "No strong positive catalyst identified in current data"
    risk     = bear[0] if bear else "No major red flag identified in current data"

    if health.get("overall_health") in ("Excellent", "Strong") and not contradictions.get("mixed_signals"):
        neutral_case = "Case for steady continuation of current fundamentals absent new catalysts."
    else:
        neutral_case = "Case for limited near-term change while signals remain mixed or moderate."

    return {
        "bull_case"   : bull or ["No bullish signals identified in current data."],
        "bear_case"   : bear or ["No bearish signals identified in current data."],
        "neutral_case": neutral_case,
        "key_catalyst": catalyst,
        "key_risk"    : risk,
    }


# ──────────────────────────────────────────────────────────────────────────────
# risk engine — separates risk into categories instead of one vague label
# ──────────────────────────────────────────────────────────────────────────────

def build_risk_profile(metrics: dict, sector_ctx: dict) -> dict:
    na = sector_ctx["not_applicable"]
    debt    = metrics.get("debt_to_equity")
    current = metrics.get("current_ratio")
    revenue = metrics.get("revenue_analysis", {})
    fcf     = metrics.get("fcf_latest")
 
    def level(cond_high, cond_med):
        if cond_high: return "High"
        if cond_med:  return "Medium"
        return "Low"
 
    if "debt_to_equity" in na:
        # bank/REIT/utility — leverage isn't scored as generic financial risk
        financial_risk = "Not Applicable"
    else:
        financial_risk = level(debt is not None and debt > 2.5, debt is not None and debt > 1.5)
 
    if "current_ratio" in na:
        liquidity_risk = "Not Applicable"
    else:
        liquidity_risk = level(current is not None and current < 0.8, current is not None and current < 1.0)
 
    growth_risk = level(revenue.get("classification") == "Continuous Decline",
                        revenue.get("classification") in ("Declining", "Volatile"))
    cashflow_risk = level(fcf is not None and fcf < 0, fcf is None)
 
    levels_seen = [x for x in (financial_risk, liquidity_risk, growth_risk, cashflow_risk)
                   if x != "Not Applicable"]
    if "High" in levels_seen:
        overall = "High"
    elif levels_seen.count("Medium") >= 2:
        overall = "Medium"
    else:
        overall = "Low"
 
    return {
        "financial_risk": financial_risk, "liquidity_risk": liquidity_risk,
        "growth_risk": growth_risk, "cashflow_risk": cashflow_risk,
        "overall_risk": overall, "sector_profile": sector_ctx["profile_key"],
    }


# ──────────────────────────────────────────────────────────────────────────────
# scenario engine — names the condition that would flip the current
# capital-allocation recommendation, so the answer isn't a static verdict
# ──────────────────────────────────────────────────────────────────────────────

def build_scenario_analysis(capital: dict) -> dict:
    scores = capital.get("scores", {})
    winner = max(scores, key=scores.get) if scores else None
    if not winner:
        return {"current_recommendation": "Unknown", "flip_condition": "Insufficient data."}

    flip_map = {
        "buybacks"       : "ROE declines sharply or free cash flow turns negative",
        "dividend"       : "Revenue growth accelerates significantly, favoring reinvestment instead",
        "reinvest_growth": "Revenue growth decelerates or margins compress materially",
        "reduce_debt"    : "Leverage falls below 1.5x debt/equity and liquidity normalizes",
    }

    return {
        "current_recommendation": capital.get("recommendation", "Unknown"),
        "flip_condition"        : flip_map.get(winner, "A material change in the underlying signals."),
    }


# ──────────────────────────────────────────────────────────────────────────────
# valuation engine — relative scoring without calling the LLM
# (no DCF / intrinsic value yet — that needs a discount rate assumption
# which should come from the user, not be silently invented)
# ──────────────────────────────────────────────────────────────────────────────

def build_valuation_signals(metrics: dict, fundamentals: dict, history: dict = None) -> dict:
    fundamentals = fundamentals or {}
    history = history or {}
 
    pe       = fundamentals.get("pe_ratio") or 0
    market_cap = fundamentals.get("market_cap") or 0
    rev_cagr = metrics.get("revenue_cagr_3y") or 0
    roe      = metrics.get("roe") or 0
    gm       = metrics.get("gross_margin_latest") or 0
 
    revenue_latest = metrics.get("revenue_latest")
    ebitda_hist    = history.get("ebitda", {})
    ebitda_latest  = max(ebitda_hist.values()) if ebitda_hist else None
    # crude "latest" — use the same latest() helper pattern as compute_all_metrics
    if ebitda_hist:
        ebitda_latest = ebitda_hist[max(ebitda_hist.keys())]
 
    total_debt = metrics.get("total_debt_latest") or 0
    cash       = metrics.get("cash_latest") or 0
    fcf_latest = metrics.get("fcf_latest")
 
    enterprise_value = (market_cap + total_debt - cash) if market_cap else None
 
    ev_ebitda = None
    if enterprise_value and ebitda_latest and ebitda_latest > 0:
        ev_ebitda = round(enterprise_value / ebitda_latest, 2)
 
    price_to_sales = None
    if market_cap and revenue_latest and revenue_latest > 0:
        price_to_sales = round(market_cap / revenue_latest, 2)
 
    fcf_yield = None
    if market_cap and fcf_latest is not None and market_cap > 0:
        fcf_yield = round(fcf_latest / market_cap * 100, 2)  # %
 
    peg = round(pe / rev_cagr, 2) if (pe and rev_cagr and rev_cagr > 0) else None
 
    # relative-positioning scores kept, but relabeled — these describe
    # business QUALITY, not valuation. Don't present them as "valuation
    # score" to the LLM/user; that was the actual bug.
    growth_score = round(min(max(rev_cagr, 0), 25) / 25 * 100, 1)
    margin_score = round(min(max(gm, 0), 60) / 60 * 100, 1)
    roe_score    = round(min(max(roe, 0), 30) / 30 * 100, 1)
    quality_score = round((growth_score + margin_score + roe_score) / 3, 1)
 
    return {
        # actual valuation multiples
        "pe_ratio"           : pe if pe else None,
        "ev_ebitda"          : ev_ebitda,
        "price_to_sales"     : price_to_sales,
        "fcf_yield_pct"      : fcf_yield,
        "peg_approx"         : peg,
        "enterprise_value"   : enterprise_value,
        # business-quality score, explicitly NOT called valuation anymore
        "quality_score"      : quality_score,
        "quality_components" : {"growth": growth_score, "margin": margin_score, "roe": roe_score},
        "note": ("multiples are point-in-time relative measures, not price "
                 "targets. no DCF/intrinsic value is computed — that needs "
                 "a discount rate assumption this function will not invent."),
        "data_gaps": [k for k, v in {
            "ev_ebitda": ev_ebitda, "price_to_sales": price_to_sales,
            "fcf_yield_pct": fcf_yield, "peg_approx": peg,
        }.items() if v is None],
    }


# ──────────────────────────────────────────────────────────────────────────────
# financial diagnostics — strengths / weaknesses / risks / opportunities
# pulled together from every engine above into one summary block
# ──────────────────────────────────────────────────────────────────────────────

def build_financial_diagnostics(metrics: dict) -> dict:
    strengths, weaknesses, risks, opportunities = [], [], [], []

    revenue = metrics.get("revenue_analysis", {})
    health  = metrics.get("company_health", {})
    gm      = metrics.get("gross_margin_latest")
    om      = metrics.get("operating_margin_latest")
    roe     = metrics.get("roe")
    debt    = metrics.get("debt_to_equity")
    current = metrics.get("current_ratio")
    fcf     = metrics.get("fcf_latest")

    if revenue.get("classification") in ("Steady Growth", "Recovering Growth"):
        strengths.append("Revenue growth is improving, indicating strengthening business momentum.")
        opportunities.append("Improving revenue creates room for future earnings expansion.")
    elif revenue.get("classification") in ("Declining", "Continuous Decline"):
        weaknesses.append("Revenue growth is deteriorating.")
        risks.append("Continued revenue weakness could pressure profitability.")

    if gm is not None:
        if gm >= 45:  strengths.append("Gross margins are exceptionally strong.")
        elif gm < 30: weaknesses.append("Gross margins are under pressure.")

    if om is not None:
        if om >= 25:  strengths.append("Operating efficiency remains excellent.")
        elif om < 15: weaknesses.append("Operating profitability is relatively weak.")

    if roe is not None:
        if roe >= 20:  strengths.append("Management is generating excellent shareholder returns.")
        elif roe < 10: weaknesses.append("Return on equity is relatively weak.")

    if debt is not None:
        if debt > 2:   risks.append("Debt levels could reduce financial flexibility.")
        elif debt < 1: strengths.append("Debt remains conservative.")

    if current is not None:
        if current < 1:    risks.append("Liquidity should be monitored closely.")
        elif current >= 1.5: strengths.append("Liquidity position is healthy.")

    if fcf is not None and fcf > 0:
        strengths.append("Business continues generating positive free cash flow.")
        opportunities.append("Strong cash generation supports dividends, buybacks and reinvestment.")

    overall = health.get("overall_health")
    if overall == "Strong" or overall == "Excellent":
        summary = "Overall financial condition is strong with healthy profitability and improving business momentum."
    elif overall == "Average":
        summary = "Business fundamentals remain stable but several areas require monitoring."
    else:
        summary = "Financial performance shows notable weaknesses requiring attention."

    return {
        "strengths"        : strengths,
        "weaknesses"       : weaknesses,
        "risks"            : risks,
        "opportunities"    : opportunities,
        "executive_summary": summary,
    }
def _compute_margin_history(history: dict) -> dict:
    """Stage 1: gross/operating/net margin history from revenue+income lines."""
    rev = history.get("revenue", {})
    gp  = history.get("gross_profit", {})
    oi  = history.get("operating_income", {})
    ni  = history.get("net_income", {})

    gm_hist, om_hist, nm_hist = {}, {}, {}
    for p in rev:
        if p in gp and rev[p]: gm_hist[p] = round(gp[p] / rev[p] * 100, 2)
        if p in oi and rev[p]: om_hist[p] = round(oi[p] / rev[p] * 100, 2)
        if p in ni and rev[p]: nm_hist[p] = round(ni[p] / rev[p] * 100, 2)
    return {"gross_margin_history": gm_hist, "operating_margin_history": om_hist,
            "net_margin_history": nm_hist}


def _compute_trend_layer(history: dict, margins: dict) -> dict:
    """Stage 2: trend classification + insight for every tracked metric."""
    rev = history.get("revenue", {})
    oi  = history.get("operating_income", {})
    fcf = history.get("free_cash_flow", {})

    out = {}
    out["revenue_analysis"]          = build_trend_analysis(rev, "Revenue")
    out["operating_income_analysis"] = build_trend_analysis(oi, "Operating Income")
    out["gross_margin_analysis"]     = build_trend_analysis(margins["gross_margin_history"], "Gross Margin")
    out["operating_margin_analysis"] = build_trend_analysis(margins["operating_margin_history"], "Operating Margin")
    out["eps_analysis"]              = build_trend_analysis(history.get("eps_basic", {}), "EPS")
    out["fcf_analysis"]              = build_trend_analysis(fcf, "Free Cash Flow")

    out["revenue_insight"]          = build_financial_insight(out["revenue_analysis"])
    out["gross_margin_insight"]     = build_financial_insight(out["gross_margin_analysis"])
    out["operating_margin_insight"] = build_financial_insight(out["operating_margin_analysis"])
    out["eps_insight"]              = build_financial_insight(out["eps_analysis"])
    out["fcf_insight"]              = build_financial_insight(out["fcf_analysis"])
    return out


def _compute_growth_metrics(history: dict, yoy_growth_fn, cagr_fn, latest_fn) -> dict:
    """Stage 3: YoY, CAGR, TTM, buyback trend, quarterly gaps for every tracked series."""
    rev = history.get("revenue", {})
    gp  = history.get("gross_profit", {})
    oi  = history.get("operating_income", {})
    ni  = history.get("net_income", {})
    fcf = history.get("free_cash_flow", {})

    out = {}
    out["revenue_yoy"]          = yoy_growth_fn(rev)
    out["gross_profit_yoy"]     = yoy_growth_fn(gp)
    out["operating_income_yoy"] = yoy_growth_fn(oi)
    out["net_income_yoy"]       = yoy_growth_fn(ni)
    out["fcf_yoy"]              = yoy_growth_fn(fcf)
    out["eps_yoy"]              = yoy_growth_fn(history.get("eps_basic", {}))

    # buyback trend — yfinance reports "Repurchase Of Capital Stock" as a
    # negative cash-flow line (cash outflow), so we take abs() before
    # computing YoY: positive buyback_yoy = repurchases grew year over
    # year, negative = repurchases shrank or the company net-issued shares
    buy_raw      = history.get("share_repurchase", {})
    buyback_hist = {p: abs(v) for p, v in buy_raw.items()}
    buyback_yoy_series = yoy_growth_fn(buyback_hist)
    out["buyback_latest"] = latest_fn(buyback_hist)
    out["buyback_yoy_latest"] = (
        buyback_yoy_series[sorted(buyback_yoy_series.keys())[-1]]
        if buyback_yoy_series else None
    )

    out["revenue_gaps"] = find_period_gaps(history.get("revenue_q", {}))
    out["fcf_gaps"]     = find_period_gaps(history.get("free_cash_flow_q", {}))
    out["revenue_ttm"]    = compute_ttm(history.get("revenue_q", {}))
    out["net_income_ttm"] = compute_ttm(history.get("net_income_q", {}))
    out["eps_ttm"]        = compute_ttm(history.get("eps_basic_q", {}))
    out["fcf_ttm"]        = compute_ttm(history.get("free_cash_flow_q", {}))

    out["revenue_cagr_3y"] = cagr_fn(rev, 3)
    out["revenue_cagr_5y"] = cagr_fn(rev, 5)
    out["eps_cagr_3y"]     = cagr_fn(history.get("eps_basic", {}), 3)
    out["fcf_cagr_3y"]     = cagr_fn(fcf, 3)
    return out


def _compute_balance_sheet_ratios(history: dict, latest_fn) -> dict:
    """Stage 4: ROE, ROA, debt/equity, current ratio from latest balance sheet."""
    eq  = history.get("equity", {})
    ta  = history.get("total_assets", {})
    ni  = history.get("net_income", {})
    td  = history.get("total_debt", {})
    ca  = history.get("current_assets", {})
    cl  = history.get("current_liabilities", {})

    latest_eq, latest_ta, latest_ni = latest_fn(eq), latest_fn(ta), latest_fn(ni)
    latest_td, latest_ca, latest_cl = latest_fn(td), latest_fn(ca), latest_fn(cl)

    return {
        "roe": round(latest_ni / latest_eq * 100, 2) if latest_eq and latest_ni else None,
        "roa": round(latest_ni / latest_ta * 100, 2) if latest_ta and latest_ni else None,
        "debt_to_equity": round(latest_td / latest_eq, 2) if latest_eq and latest_td else None,
        "current_ratio": round(latest_ca / latest_cl, 2) if latest_cl and latest_ca else None,
    }


def _compute_sector_specific_metrics(history: dict, out: dict, sector_ctx: dict, latest_fn) -> dict:
    """
    Sector-specific derived metrics — only computed when genuinely
    derivable from standard financial statements. Does NOT compute ARPU,
    subscriber counts, or same-store sales — those require company KPI
    disclosures outside standard statements (filing-text extraction,
    not implemented).
    """
    profile = sector_ctx.get("profile_key")
    metrics = {}
    rev_latest = out.get("revenue_latest")
    fcf_latest = out.get("fcf_latest")

    if profile == "software_saas":
        rd_latest = latest_fn(history.get("r_and_d", {}))
        if rd_latest and rev_latest:
            metrics["rd_to_revenue_ratio"] = round(rd_latest / rev_latest * 100, 2)
        if fcf_latest and rev_latest:
            fcf_margin = round(fcf_latest / rev_latest * 100, 2)
            metrics["fcf_margin"] = fcf_margin
            rev_yoy = out.get("revenue_yoy", {})
            if rev_yoy:
                rev_growth = rev_yoy[sorted(rev_yoy.keys())[-1]]
                metrics["rule_of_40"] = round(rev_growth + fcf_margin, 2)

    elif profile == "consumer_retail":
        inv = history.get("inventory", {})
        gp  = history.get("gross_profit", {})
        rev = history.get("revenue", {})
        periods = sorted(inv.keys())
        if periods and rev_latest:
            latest_p = periods[-1]
            cogs = rev.get(latest_p, 0) - gp.get(latest_p, 0)
            if len(periods) >= 2:
                avg_inv = (inv[latest_p] + inv[periods[-2]]) / 2
            else:
                avg_inv = inv[latest_p]
            if avg_inv:
                metrics["inventory_turnover"] = round(cogs / avg_inv, 2)

    return metrics


def compute_all_metrics(history: dict, ticker: str, company: str, fundamentals: dict = None) -> dict:
    """
    Python computes every financial ratio.
    The LLM never does arithmetic — it only reports what this function produces.

    Orchestrates four computation stages (margins -> trends -> growth ->
    balance sheet ratios), each independently testable via its own
    _compute_*() function, plus sector-specific derived metrics and the
    full reasoning layer (health -> capital allocation -> evidence ->
    contradictions -> thesis -> risk -> scenario -> valuation -> diagnostics)
    built on top of that history. Every number here is computed in Python
    from retrieved data, never invented by the LLM.
    """
    fundamentals = fundamentals or {}

    def yoy_growth(series: dict) -> dict:
        periods = sorted(series.keys())
        result  = {}
        for i in range(1, len(periods)):
            prev, curr = periods[i-1], periods[i]
            pv, cv     = series[prev], series[curr]
            if pv and pv != 0:
                result[f"{prev}_to_{curr}"] = round((cv - pv) / abs(pv) * 100, 2)
        return result

    from typing import Optional

    def cagr(series: dict, n_years: int) -> Optional[float]:
        periods = sorted(series.keys())
        if len(periods) < 2:
            return None
        newest, oldest = periods[-1], periods[0]
        ny = int(newest[:4]) - int(oldest[:4])
        if ny < 1 or series[oldest] <= 0:
            return None
        target_periods = [p for p in periods if int(newest[:4]) - int(p[:4]) <= n_years]
        if len(target_periods) < 2:
            return None
        start = target_periods[0]
        end   = target_periods[-1]
        n     = int(end[:4]) - int(start[:4])
        if n < 1 or series[start] <= 0 or series[end] <= 0:
            return None
        ratio = series[end] / series[start]
        return round(((ratio ** (1/n)) - 1) * 100, 2)

    def latest(series: dict):
        if not series:
            return None
        return series[max(series.keys())]

    out = {"ticker": ticker, "company": company, "history": history}

    # must run BEFORE margin/trend/growth stages read history["revenue"] —
    # patching it after those stages already ran leaves bank tickers with
    # permanently empty revenue trend/margin/CAGR data
    if not history.get("revenue") and history.get("net_interest_income"):
        nii = history.get("net_interest_income", {})
        noninc = history.get("non_interest_income", {})
        history["revenue"] = {p: nii[p] + noninc.get(p, 0) for p in nii}
        out["revenue_is_bank_proxy"] = True

    _rev_hist = history.get("revenue", {})
    _latest_rev = _rev_hist[max(_rev_hist.keys())] if _rev_hist else None
    sector_ctx = get_sector_context(
        ticker,
        sector=history.get("_sector"),
        industry=history.get("_industry"),
        revenue=_latest_rev,
    )
    out["sector_ctx"] = sector_ctx
    fy_month = get_fiscal_year_end_month(ticker)
    out["fiscal_year_end_month"] = fy_month

    # ── stage 1-4: margins, trends, growth, balance sheet ──
    margins = _compute_margin_history(history)
    out.update(margins)
    out["gross_margin_latest"]      = latest(margins["gross_margin_history"])
    out["operating_margin_latest"]  = latest(margins["operating_margin_history"])
    out["net_margin_latest"]        = latest(margins["net_margin_history"])

    out.update(_compute_trend_layer(history, margins))
    out.update(_compute_growth_metrics(history, yoy_growth, cagr, latest))

    rev = history.get("revenue", {})
    rev_q = history.get("revenue_q", {})
    q_yoy = {}
    for p in sorted(rev_q.keys()):
        yr   = int(p[:4]) - 1
        same = p.replace(p[:4], str(yr), 1)
        if same in rev_q and rev_q[same] != 0:
            q_yoy[p] = round((rev_q[p] - rev_q[same]) / abs(rev_q[same]) * 100, 2)
    out["revenue_qoq_yoy"] = q_yoy

    out.update(_compute_balance_sheet_ratios(history, latest))
    out.update(_compute_sector_specific_metrics(history, out, sector_ctx, latest))

    # ── latest-value snapshots ──
    gp  = history.get("gross_profit", {})
    oi  = history.get("operating_income", {})
    ni  = history.get("net_income", {})
    fcf = history.get("free_cash_flow", {})
    ta  = history.get("total_assets", {})
    td  = history.get("total_debt", {})
    csh = history.get("cash", {})

    out["revenue_latest"]          = latest(rev)
    out["gross_profit_latest"]     = latest(gp)
    out["operating_income_latest"] = latest(oi)
    out["net_income_latest"]       = latest(ni)
    out["eps_latest"]              = latest(history.get("eps_basic", {}))
    out["fcf_latest"]              = latest(fcf)
    out["total_assets_latest"]     = latest(ta)
    out["total_debt_latest"]       = latest(td)
    out["cash_latest"]             = latest(csh)

    rev_q_sorted = sorted(rev_q.keys(), reverse=True)
    out["revenue_latest_quarter"]  = rev_q[rev_q_sorted[0]] if rev_q_sorted else None
    out["revenue_latest_q_period"] = rev_q_sorted[0] if rev_q_sorted else None

    gp_q = history.get("gross_profit_q", {})
    if rev_q_sorted and rev_q_sorted[0] in gp_q and rev_q[rev_q_sorted[0]]:
        lq = rev_q_sorted[0]
        out["gross_margin_latest_quarter"]   = round(gp_q[lq] / rev_q[lq] * 100, 2)
        out["revenue_latest_quarter_period"] = lq

    oi_q = history.get("operating_income_q", {})
    if rev_q_sorted and rev_q_sorted[0] in oi_q and rev_q[rev_q_sorted[0]]:
        lq = rev_q_sorted[0]
        out["operating_margin_latest_quarter"] = round(oi_q[lq] / rev_q[lq] * 100, 2)

    # ── reasoning layer — built in dependency order, each reads only what's
    # already in `out` above this point ──
    out["company_health"]         = build_company_health(out, sector_ctx)
    out["capital_allocation"]     = score_capital_allocation(out, fundamentals)
    out["contradictions"]         = detect_contradictions(out)
    out["evidence"]               = build_evidence(out, sector_ctx, out["contradictions"])
    out["investment_thesis"]      = build_investment_thesis(
        out, out["company_health"], out["contradictions"],
        claimed_signals=out["evidence"].get("claimed_signals", set())
    )
    out["risk_profile"]           = build_risk_profile(out, sector_ctx)
    out["scenario"]               = build_scenario_analysis(out["capital_allocation"])
    out["valuation_signals"]      = build_valuation_signals(out, fundamentals, history)
    out["financial_diagnostics"]  = build_financial_diagnostics(out)

    return out


def metrics_to_text(m: dict) -> str:
    """
    Converts the computed metrics dict into a dense text chunk.
    This chunk gets quality=1.0 and is ALWAYS retrieved first for financial queries.
    The LLM reads this text and reports the numbers — no arithmetic needed.

    Surfaces every engine computed above: raw numbers, trend classifications,
    company health, capital allocation scoring, evidence, contradictions,
    investment thesis, risk profile, scenario, and valuation signals.
    """
    lines = []

    company = m["company"]
    ticker = m["ticker"]
    fy_month = m.get("fiscal_year_end_month", 12)
# ==========================================================
# HEADER
# ==========================================================

    lines.append("=" * 70)
    lines.append(f"{company.upper()} ({ticker})")
    lines.append("FINANCIAL INTELLIGENCE REPORT")
    lines.append("=" * 70)

# ==========================================================
# EXECUTIVE SUMMARY
# ==========================================================

    lines.append("")
    lines.append("EXECUTIVE SUMMARY")
    lines.append("-" * 70)

    ch = m.get("company_health", {})
    rp = m.get("risk_profile", {})
    ca = m.get("capital_allocation", {})
    thesis = m.get("investment_thesis", {})
    insight = m.get("revenue_insight", {})

    lines.append(f"Overall Health        : {ch.get('overall_health','Unknown')} ({ch.get('overall_score','?')}/10)")
    lines.append(f"Overall Risk          : {rp.get('overall_risk','Unknown')}")
    lines.append(f"Capital Allocation    : {ca.get('recommendation','Unknown')}")
    lines.append(f"Recommendation Confidence : {ca.get('confidence','Unknown')}")

    if insight.get("executive_summary"):
        lines.append("")
        lines.append("Executive View")
        lines.append(f"• {insight['executive_summary']}")

    if thesis.get("key_catalyst"):
        lines.append(f"Key Catalyst : {thesis['key_catalyst']}")

    if thesis.get("key_risk"):
        lines.append(f"Key Risk     : {thesis['key_risk']}")

# ==========================================================
# BUSINESS PERFORMANCE
# ==========================================================

    lines.append("")
    lines.append("=" * 70)
    lines.append("BUSINESS PERFORMANCE")
    lines.append("=" * 70)

    ra = m.get("revenue_analysis", {})

    if ra:
        lines.append("")
        lines.append("Revenue Trend")
        lines.append("-" * 70)
        lines.append(f"Classification : {ra.get('classification','Unknown')}")
        lines.append(f"Direction      : {ra.get('direction','Unknown')}")
        lines.append(f"Confidence     : {ra.get('confidence','Unknown')}")

        if ra.get("reason"):
            lines.append(f"Reason         : {ra['reason']}")

    ri = m.get("revenue_insight", {})

    if ri:
        lines.append("")
        lines.append("Business Insight")
        lines.append("-" * 70)

        if ri.get("business_insight"):
            lines.append(ri["business_insight"])

        if ri.get("investor_interpretation"):
            lines.append("")
            lines.append("Investor Interpretation")
            lines.append(ri["investor_interpretation"])

    rev = m["history"].get("revenue", {})

    if rev:
        lines.append("")
        lines.append("Annual Revenue")
        lines.append("-" * 70)

        for p in sorted(rev.keys(), reverse=True):
            lines.append(f"{fiscal_label(p, fy_month):<28}{fmt(rev[p])}")

    if m.get("revenue_yoy"):
        lines.append("")
        lines.append("Revenue YoY Growth")
        lines.append("-" * 70)

        for period, growth in sorted(m["revenue_yoy"].items()):
            sign = "+" if growth > 0 else ""
            lines.append(f"{period:<28}{sign}{growth}%")

    if m.get("revenue_cagr_3y") is not None:
        lines.append("")
        lines.append("Revenue CAGR")
        lines.append("-" * 70)
        lines.append(f"3 Year : {pct(m['revenue_cagr_3y'])}")

        if m.get("revenue_cagr_5y") is not None:
            lines.append(f"5 Year : {pct(m['revenue_cagr_5y'])}")

    rev_q = m["history"].get("revenue_q", {})

    if rev_q:
        lines.append("")
        lines.append("Quarterly Revenue")
        lines.append("-" * 70)

        for p in sorted(rev_q.keys(), reverse=True):
            lines.append(f"{p:<12}{fmt(rev_q[p])}")

    if m.get("revenue_gaps"):
        lines.append("")
        lines.append("Data Gaps")
        lines.append("-" * 70)
        for g in m["revenue_gaps"]:
            lines.append(f"⚠ No quarterly revenue reported between {g[0]} and {g[1]}")

    if m.get("revenue_ttm"):
        latest_ttm = max(m["revenue_ttm"].keys())
        lines.append("")
        lines.append("Revenue TTM")
        lines.append("-" * 70)
        lines.append(f"{latest_ttm:<12}{fmt(m['revenue_ttm'][latest_ttm])}")

    if m.get("revenue_qoq_yoy"):
        lines.append("")
        lines.append("Quarterly YoY Growth")
        lines.append("-" * 70)

        for period, growth in sorted(m["revenue_qoq_yoy"].items(), reverse=True):
            sign = "+" if growth > 0 else ""
            lines.append(f"{period:<12}{sign}{growth}%")
# ==========================================================
# OPERATING MARGIN
# ==========================================================

    lines.append("")
    lines.append("=" * 70)
    lines.append("OPERATING MARGIN")
    lines.append("=" * 70)
    
    # Latest margins
    if m.get("operating_margin_latest") is not None:
        lines.append(f"Latest Annual Operating Margin    : {pct(m['operating_margin_latest'])}")
    
    if m.get("operating_margin_latest_quarter") is not None:
        lines.append(f"Latest Quarterly Operating Margin : {pct(m['operating_margin_latest_quarter'])}")
    
    # Trend
    oma = m.get("operating_margin_analysis", {})
    
    if oma.get("classification") and oma["classification"] != "Insufficient Data":
        lines.append("")
        lines.append("Trend Analysis")
        lines.append("-" * 70)
        lines.append(f"Classification : {oma['classification']}")
        lines.append(f"Confidence     : {oma['confidence']}")
        lines.append(f"Reason         : {oma['reason']}")
    
    # History
    if m.get("operating_margin_history"):
        lines.append("")
        lines.append("Operating Margin History")
        lines.append("-" * 70)
    
        for period in sorted(m["operating_margin_history"].keys(), reverse=True):
            lines.append(
                f"{period:<12}{pct(m['operating_margin_history'][period])}"
            )
# ==========================================================
# OPERATING PERFORMANCE
# ==========================================================

    lines.append("")
    lines.append("=" * 70)
    lines.append("OPERATING PERFORMANCE")
    lines.append("=" * 70)

    oi = m["history"].get("operating_income", {})

    if oi:
        lines.append("")
        lines.append("Annual Operating Income")
        lines.append("-" * 70)

        for p in sorted(oi.keys(), reverse=True):
            lines.append(f"{fiscal_label(p, fy_month):<28}{fmt(oi[p])}")

    if m.get("operating_income_yoy"):
        lines.append("")
        lines.append("Operating Income YoY Growth")
        lines.append("-" * 70)

        for period, g in sorted(m["operating_income_yoy"].items()):
            sign = "+" if g > 0 else ""
            lines.append(f"{period:<28}{sign}{g}%")

# ==========================================================
# PROFITABILITY ANALYSIS
# ==========================================================

    lines.append("")
    lines.append("=" * 70)
    lines.append("PROFITABILITY ANALYSIS")
    lines.append("=" * 70)

    gp = m["history"].get("gross_profit", {})

    if gp:
        lines.append("")
        lines.append("Annual Gross Profit")
        lines.append("-" * 70)

        for p in sorted(gp.keys(), reverse=True):
            lines.append(f"{fiscal_label(p, fy_month):<28}{fmt(gp[p])}")

    gma = m.get("gross_margin_analysis", {})

    if gma.get("classification"):

        lines.append("")
        lines.append("Gross Margin Trend")
        lines.append("-" * 70)
        lines.append(f"Classification : {gma['classification']}")
        lines.append(f"Confidence     : {gma['confidence']}")
        lines.append(f"Reason         : {gma['reason']}")

    if m.get("gross_margin_history"):

        lines.append("")
        lines.append("Gross Margin History")
        lines.append("-" * 70)

        for p in sorted(m["gross_margin_history"].keys(), reverse=True):
            lines.append(f"{p:<12}{pct(m['gross_margin_history'][p])}")

# ==========================================================
# NET INCOME
# ==========================================================

    lines.append("")
    lines.append("=" * 70)
    lines.append("NET INCOME")
    lines.append("=" * 70)

    ni = m["history"].get("net_income", {})

    if ni:
        lines.append("")
        lines.append("Annual Net Income")
        lines.append("-" * 70)
        for p in sorted(ni.keys(), reverse=True):
            lines.append(f"{fiscal_label(p, fy_month):<28}{fmt(ni[p])}")

    if m.get("net_margin_latest") is not None:
        lines.append(f"Net Margin (latest)         : {pct(m['net_margin_latest'])}")

    if m.get("net_income_yoy"):
        lines.append("")
        lines.append("Net Income YoY Growth")
        lines.append("-" * 70)
        for period, g in sorted(m["net_income_yoy"].items()):
            sign = "+" if g > 0 else ""
            lines.append(f"{period:<28}{sign}{g}%")

# ==========================================================
# EARNINGS
# ==========================================================

    lines.append("")
    lines.append("=" * 70)
    lines.append("EARNINGS")
    lines.append("=" * 70)

    epa = m.get("eps_analysis", {})

    if epa.get("classification"):
        lines.append("")
        lines.append("EPS Trend")
        lines.append("-" * 70)
        lines.append(f"Classification : {epa['classification']}")
        lines.append(f"Confidence     : {epa['confidence']}")
        lines.append(f"Reason         : {epa['reason']}")
    
        eps= m["history"].get("eps_basic", {})
    
        if eps:
            lines.append("")
            lines.append("Annual EPS")
            lines.append("-" * 70)

        for p in sorted(eps.keys(), reverse=True):
            lines.append(f"{fiscal_label(p, fy_month):<28}${eps[p]:.2f}")

    if m.get("eps_ttm"):
        latest_ttm = max(m["eps_ttm"].keys())
        lines.append("")
        lines.append("EPS TTM")
        lines.append("-" * 70)
        lines.append(f"{latest_ttm:<12}${m['eps_ttm'][latest_ttm]:.2f}")

    if m.get("eps_yoy"):
        lines.append("")
        lines.append("EPS YoY Growth")
        lines.append("-" * 70)

        for period, g in sorted(m["eps_yoy"].items()):
            sign = "+" if g > 0 else ""
            lines.append(f"{period:<28}{sign}{g}%")

# ==========================================================
# CASH FLOW
# ==========================================================

    lines.append("")
    lines.append("=" * 70)
    lines.append("CASH FLOW")
    lines.append("=" * 70)

    fca = m.get("fcf_analysis", {})

    if fca.get("classification"):
        lines.append("")
        lines.append("Free Cash Flow Trend")
        lines.append("-" * 70)
        lines.append(f"Classification : {fca['classification']}")
        lines.append(f"Confidence     : {fca['confidence']}")
        lines.append(f"Reason         : {fca['reason']}")

    fcf = m["history"].get("free_cash_flow", {})

    if fcf:
        lines.append("")
        lines.append("Annual Free Cash Flow")
        lines.append("-" * 70)

        for p in sorted(fcf.keys(), reverse=True):
            lines.append(f"{fiscal_label(p, fy_month):<28}{fmt(fcf[p])}")

    if m.get("fcf_ttm"):
        latest_ttm = max(m["fcf_ttm"].keys())
        lines.append("")
        lines.append("FCF TTM")
        lines.append("-" * 70)
        lines.append(f"{latest_ttm:<12}{fmt(m['fcf_ttm'][latest_ttm])}")

    if m.get("fcf_yoy"):
        lines.append("")
        lines.append("Free Cash Flow YoY")
        lines.append("-" * 70)

        for period, g in sorted(m["fcf_yoy"].items()):
            sign = "+" if g > 0 else ""
            lines.append(f"{period:<28}{sign}{g}%")


    capex = m["history"].get("capex", {})
    if capex:
        lines.append("")
        lines.append("Annual Capital Expenditure")
        lines.append("-" * 70)
        for p in sorted(capex.keys(), reverse=True):
            lines.append(f"{fiscal_label(p, fy_month):<28}{fmt(capex[p])}")

    lines.append("")
    lines.append("=" * 70)
    lines.append("BALANCE SHEET")
    lines.append("=" * 70)

    if m.get("roe") is not None:
        lines.append(f"Return on Equity      : {pct(m['roe'])}")

    if m.get("roa") is not None:
        lines.append(f"Return on Assets      : {pct(m['roa'])}")

    if m.get("debt_to_equity") is not None:
        lines.append(f"Debt to Equity        : {m['debt_to_equity']:.2f}x")

    if m.get("current_ratio") is not None:
        lines.append(f"Current Ratio         : {m['current_ratio']:.2f}x")

    if m.get("total_debt_latest") is not None:
        lines.append(f"Total Debt            : {fmt(m['total_debt_latest'])}")

    if m.get("cash_latest") is not None:
        lines.append(f"Cash & Equivalents    : {fmt(m['cash_latest'])}")

    lines.append("")
    lines.append("=" * 70)
    lines.append("COMPANY HEALTH")
    lines.append("=" * 70)

    lines.append(
        f"Overall Health : {ch.get('overall_health','')} ({ch.get('overall_score','')}/10)"
    )

    lines.append("")

    for r in ch.get("reasoning", []):
        lines.append(f"• {r}")
# ==========================================================
# CAPITAL ALLOCATION
# ==========================================================

    ca_eng = m.get("capital_allocation", {})

    if ca_eng:
        lines.append("")
        lines.append("=" * 70)
        lines.append("CAPITAL ALLOCATION")
        lines.append("=" * 70)

        lines.append(f"Recommendation : {ca_eng.get('recommendation','')}")
        lines.append(f"Confidence     : {ca_eng.get('confidence','')}")
        lines.append(f"Runner-up      : {ca_eng.get('runner_up','')}")

        if ca_eng.get("scores"):
            lines.append("")
            lines.append("Scores")

            for k, v in ca_eng["scores"].items():
                lines.append(f"  {k:<20}: {v}")

# ==========================================================
# KEY EVIDENCE
# ==========================================================

    ev = m.get("evidence", {})

    if ev:
        lines.append("")
        lines.append("=" * 70)
        lines.append("KEY EVIDENCE")
        lines.append("=" * 70)

        lines.append(f"Confidence : {ev.get('confidence_score','')}/10")
        lines.append("")

        for e in ev.get("evidence", []):
            lines.append(f"• {e}")

    # ── contradictions ──
    lines.append("")
    lines.append("=" * 70)
    lines.append("MIXED SIGNALS")
    lines.append("=" * 70)
    co = m.get("contradictions", {})
    for c in co.get("contradictions", []):
        lines.append(f"• {c}")

    # ── investment thesis ──
    lines.append("")
    lines.append("=" * 70)
    lines.append("INVESTMENT THESIS")
    lines.append("=" * 70)
    it = m.get("investment_thesis", {})
    if it:
        lines.append("bull case: " + "; ".join(it.get("bull_case", [])))
        lines.append("bear case: " + "; ".join(it.get("bear_case", [])))
        lines.append(f"key catalyst: {it.get('key_catalyst','')}")
        lines.append(f"key risk: {it.get('key_risk','')}")

    # ── risk profile ──
    lines.append("")
    lines.append("=" * 70)
    lines.append("RISK PROFILE")
    lines.append("=" * 70)

    lines.append(f"Overall Risk     : {rp.get('overall_risk','')}")
    lines.append(f"Financial Risk   : {rp.get('financial_risk','')}")
    lines.append(f"Liquidity Risk   : {rp.get('liquidity_risk','')}")
    lines.append(f"Growth Risk      : {rp.get('growth_risk','')}")
    lines.append(f"Cash Flow Risk   : {rp.get('cashflow_risk','')}")

    # ── scenario ──
    lines.append("")
    lines.append("=" * 70)
    lines.append("SCENARIO")
    lines.append("=" * 70)
    sc = m.get("scenario", {})
    if sc.get("flip_condition"):
        lines.append(f"current recommendation holds unless: {sc['flip_condition']}")

    # ── valuation signals ──
    vs = m.get("valuation_signals", {})
    lines.append("")
    lines.append("=" * 70)
    lines.append("VALUATION")
    lines.append("=" * 70)

    lines.append(f"Quality Score : {vs.get('quality_score','')}/100  (business quality, not a price target)")
    qc = vs.get('quality_components', {})
    lines.append(f"  Growth: {qc.get('growth','')}/100  Margin: {qc.get('margin','')}/100  ROE: {qc.get('roe','')}/100")
    lines.append(f"EV/EBITDA     : {vs.get('ev_ebitda','')}")
    lines.append(f"P/S           : {vs.get('price_to_sales','')}")
    lines.append(f"FCF Yield     : {vs.get('fcf_yield_pct','')}%")
    lines.append(f"PEG (approx)  : {vs.get('peg_approx','')}")
    # Look at the reduction in outstanding shares or the presence of buybacks/dividends

    return "\n".join(lines)
'''
from test3 import get_metrics, metrics_to_text
metrics = get_metrics("AAPL", "Apple Inc.", force_refresh=True)

report = metrics_to_text(metrics)

with open("financial_report.txt", "w", encoding="utf-8") as f:
    f.write(report)

print("Financial report saved successfully.")
'''

def build_metrics_chunk(ticker: str, company: str, metrics: dict):
    """
    Wraps the computed metrics text as a chunk with quality=1.0.
    Always retrieved first for financial queries.
    """
    content = metrics_to_text(metrics)
    return {
        "content": content,
        "source_type": "computed_metrics",
        "ticker": ticker,
        "company": company,
        "date": str(datetime.now().date()),
        "word_count": len(content.split()),
        "language": "en",
        "quality": 1.0,
        "metadata": {
            "section": "computed_metrics",
            "gross_margin": metrics.get("gross_margin_latest"),
            "operating_margin": metrics.get("operating_margin_latest"),
            "net_margin": metrics.get("net_margin_latest"),
            "revenue_cagr_3y": metrics.get("revenue_cagr_3y"),
            "revenue_trend": metrics.get("revenue_analysis", {}).get("classification"),
            "eps_trend": metrics.get("eps_analysis", {}).get("classification"),
            "fcf_trend": metrics.get("fcf_analysis", {}).get("classification"),
            "roe": metrics.get("roe"),
            "roa": metrics.get("roa"),
            "debt_to_equity": metrics.get("debt_to_equity"),
            "current_ratio": metrics.get("current_ratio"),
            "company_health": metrics.get("company_health", {}).get("overall_health"),
            "capital_allocation": metrics.get("capital_allocation", {}).get("recommendation"),
            "overall_risk": metrics.get("risk_profile", {}).get("overall_risk"),
            "valuation_score": metrics.get("valuation_signals", {}).get("overall_score"),
            "priority": "highest"
        }
    }

def build_macro_chunk(series_data: dict, label: str) -> dict:
    """
    Same pattern as build_metrics_chunk — reuses your existing
    build_trend_analysis() so macro series get the same classification/
    momentum/confidence treatment as any company metric.
    """
    trend = build_trend_analysis(series_data, label)
    latest_date = max(series_data.keys()) if series_data else None
    latest_val  = series_data.get(latest_date) if latest_date else None
    content = (f"MACRO INDICATOR: {label}\n"
               f"Latest ({latest_date}): {latest_val}\n"
               f"Trend: {trend['classification']} | {trend['reason']}\n"
               f"Momentum: {trend['momentum']}")
    return {
        "chunk_id"   : f"macro_{label}_{datetime.now().strftime('%Y%m%d')}",
        "content"    : content,
        "source_type": "macro_context",
        "ticker"     : "MACRO",
        "company"    : "MACROECONOMIC",
        "date"       : str(datetime.now().date()),
        "word_count" : len(content.split()),
        "language"   : "en",
        "quality"    : 1.0,
        "metadata"   : {"section": "macro_context", "indicator": label,
                         "trend": trend["classification"]},
    }
# ──────────────────────────────────────────────────────────────────────────────
# query router
#
# decides which engines to activate for each query type.
# financial metric queries go to computed_metrics first.
# risk queries go to sec_content sections.
# news queries go to news chunks.
# capital allocation queries (dividends/buybacks/debt paydown) go to cash flow,
# capital allocation engine output, and management commentary — NOT income
# statement sections, which is what was happening before this class existed.
# the router prevents risk-factor chunks from competing with income-statement
# chunks for a margin query.
# ──────────────────────────────────────────────────────────────────────────────

# keyword tiers: multi-word phrases are far more specific than single words,
# so they're weighted much higher. previously every match scored the same
# 1x/2x regardless of specificity — "margin" alone matched as readily as
# "return on equity", which isn't right.
#
# tier 3 (weight 3): highly specific multi-word financial terms
# tier 2 (weight 2): multi-word phrases, general
# tier 1 (weight 1): single generic words — still useful as weak signal

_QUERY_CLASSES = {
    "financial_metric": [
        # Core profitability & ratios
        "gross margin", "operating margin", "net margin", "profit margin",
        "gross profit", "operating income", "net income", "operating profit",
        "earnings per share", "return on equity", "return on assets",
        "debt to equity", "current ratio", "quick ratio",
        "free cash flow", "capital expenditure", "financial ratio",
        "financial metric", "quarterly result", "latest quarter",
        # Valuation metrics
        "market cap", "valuation", "pe ratio", "price to earnings",
        "ev/ebitda", "enterprise value", "price to book",
        "dividend yield", "payout ratio",
        # Sector‑specific profitability
        "net interest margin", "efficiency ratio", "loan loss provision",
        "combined ratio", "underwriting margin",
        "production volume", "reserves", "r&d spend",
        "subscriber growth", "same-store sales", "arpu",
        # Tier 1 shorthand
        "ebitda", "ebit", "revenue", "eps", "roe", "roa", "roic",
        "leverage", "fcf", "capex", "margin", "profitability",
        "q1", "q2", "q3", "q4",
    ],
    "capital_allocation": [
        "increase dividend", "dividend policy", "dividend history",
        "share buyback", "stock buyback", "share repurchase",
        "capital allocation", "capital return", "buyback authorization",
        "reduce debt", "pay down debt", "debt paydown",
        "cash balance", "cash position", "shareholder return",
        "distribution", "capital return program","dividend", "buyback", "repurchase", "payout",
        "allocating capital","capital efficiency","capital efficiently","capital deployment","capital deployment strategy",
        "capital allocation strategy","reinvest","reinvest into growth","returning capital","return capital","returning capital to shareholders","capital strategy","cash deployment",
        "use of cash","excess cash",
    ],
    "growth": [
        "revenue growth", "earnings growth", "profit growth", "eps growth",
        "year over year", "year-over-year", "compound annual",
        "quarter over quarter", "3 year", "5 year",
        "organic growth", "same-store sales growth", "subscriber growth",
        "cagr", "trend", "historical", "grew", "yoy", "qoq","growing","grown","increasing",
        "improving","accelerating","declining","consistent growth","trend","trajectory","growing","consistently","consistent",
        "decelerating",
    ],
    "trend_analysis": [
        "trend", "trending", "trajectory", "pattern", "historical trend",
        "multi-year", "multi-quarter", "over the years", "over the quarters",
        "past several years", "past several quarters", "long-term trend",
        "how has it trended", "trend over time",
    ],
    "quarterly_comparison": [
        "quarter over quarter", "qoq", "each quarter", "by quarter",
        "quarterly breakdown", "quarterly comparison", "last 4 quarters",
        "last 8 quarters", "quarter by quarter", "sequential quarters",
    ],
    "risk": [
        "risk", "risks", "threat", "danger", "challenge", "uncertainty",
        "regulation", "regulatory", "geopolitical", "decline", "downturn",
        "concern", "exposure", "vulnerability", "lawsuit", "litigation",
        "fine", "penalty", "antitrust", "breach", "cyber", "fraud",
        "credit risk", "liquidity risk", "market risk", "operational risk",
        "compliance risk", "systemic risk",
    ],
    "management_commentary": [
        "conference call", "earnings call", "investor day",
        "management", "ceo", "cfo", "board", "executive", "leadership",
        "said", "stated", "announced", "commentary", "guidance", "outlook",
        "transcript", "prepared remarks", "qa session",
    ],
    "news_sentiment": [
        "news", "headline", "sentiment", "recent", "latest", "developments",
        "happened", "update", "catalyst", "press", "release", "today",
        "this week", "breaking", "market reaction",
    ],
    "segment": [
        "reality labs", "family of apps", "business line", "operating segment",
        "shared cost", "corporate overhead", "division", "unit",
        "reportable", "contribution", "allocation",
        "geographic segment", "regional performance",
    ],
    "investment_thesis":[
              "investment thesis","bull case","bear case","investment case","why invest","why not invest","long thesis",
              "short thesis","investment opportunity","catalyst"
    ],
    "company_health":[
    "financial health","company health","overall health","healthy company","business quality","financial strength",
    "overall strength","company quality"
    ],
    "scenario":[
    "what if","scenario","best case","worst case","flip condition",
    "what changes","what would change","invalidate","future scenario"
    ],
    "evidence":[
    "evidence","support","supporting evidence","why","proof",
    "justify","back it up"
    ],
    "contradictions":[
    "mixed signals","contradiction","conflicting metrics","inconsistency","red flags"
    ],
    "balance_sheet": [
        "deferred revenue", "current liabilities", "balance sheet",
        "working capital", "contract liability", "unearned revenue",
        "long-term", "non-current", "assets", "liabilities", "debt", "equity",
        "goodwill", "intangible", "depreciation", "cash equivalents",
        "total debt", "net debt", "book value","cash","cash balance",
        "cash position","cash holdings","cash reserves","cash equivalents",
    ],
    "price": [
        "stock price", "share price", "closing price", "opening price",
        "52-week", "stock chart", "technical analysis", "moving average",
        "market cap", "target price", "analyst rating", "trading volume",
        "stock volume",
        # bare single words removed: "price", "close", "open", "high", "low",
        # "volume" collide constantly with ordinary financial vocabulary
        # ("close" a deal, "high" margin, "volume" of business, "open" a
        # position) and were causing misclassification on questions that
        # have nothing to do with stock price.
    ],
    "comparison": [
        "compare", "vs", "versus", "better", "worse", "outperform",
        "underperform", "relative", "peer", "than", "between", "difference",
        "benchmark", "ranking", "competitive position",
    ],
    "macro_context": [
        "interest rate", "fed funds", "federal reserve", "inflation", "cpi", "ppi",
        "gdp", "unemployment", "recession", "macro environment", "economic outlook",
        "rate hike", "rate cut", "treasury yield", "macroeconomic",
    ],
    "assumptions": [
        "assumption", "assumptions", "basis", "premise",
        "implicit", "expectation", "underlie", "underlying",
        "needs to be true", "what must hold", "foundation",
        "contingent on", "depends on", "drives this outlook",
    ]
}


# sections that should be EXCLUDED for certain query types
_SECTION_EXCLUSIONS = {
    "financial_metric"     : {"risk_factors", "legal", "board", "general_news", "ma_news", "regulatory_news"},
    "capital_allocation"   : {"risk_factors", "legal", "board", "general_news", "ma_news", "income_statement"},
    "growth"                : {"risk_factors", "legal", "board", "general_news", "ma_news"},
    "trend_analysis"        : {"risk_factors", "legal", "board", "general_news", "ma_news"},
    "quarterly_comparison"  : {"risk_factors", "legal", "board", "general_news", "ma_news"},
    "balance_sheet"         : {"risk_factors", "legal", "board", "general_news"},
    "segment"               : {"risk_factors", "legal", "board", "general_news"},
    "price"                 : {"risk_factors", "legal", "md&a", "earnings"},
    "news_sentiment"        : {"risk_factors", "legal", "financials", "balance_sheet"},
    "management_commentary" : {"risk_factors", "legal", "general_news", "ma_news"},
    "comparison"            : {"risk_factors", "legal", "general_news"},
    "investment_thesis"     : {"legal","board","general_news","risk_factors"},
    "company_health"        : {"legal","board","general_news"},
    "scenario"              : {"legal","board","general_news"},
    "evidence"              : {"legal","board","general_news"},
    "contradictions"        : {"legal","board","general_news"},
}

# sections that should be PREFERRED for certain query types
_SECTION_PREFERENCES = {
    "financial_metric": {"computed_metrics","income_statement","key_ratios",
                         "financials","xbrl_summary","earnings_discussion",
                         "xbrl_data","structured_metrics","fundamentals",
                         "financial_table"},
    "capital_allocation": {"computed_metrics","cash_flow","balance_sheet",
                           "guidance","prepared_remarks","earnings_discussion",
                           "md&a","dividend_history","shareholder_return"},
    "growth": {"computed_metrics","income_statement","revenue_growth",
               "earnings_discussion","guidance","trend_analysis",
               "historical_performance","segment_growth"},
    "trend_analysis": {"computed_metrics","income_statement","revenue_growth",
                        "earnings_discussion","trend_analysis","historical_performance"},
    "quarterly_comparison": {"computed_metrics","income_statement","structured_metrics",
                              "xbrl_data","quarterly"},
    "risk": {"risk_factors","legal","md&a","regulatory_news",
             "credit_risk","liquidity_risk","market_risk"},
    "management_commentary": {"md&a","guidance","prepared_remarks",
                              "qa_session","earnings_discussion",
                              "forward_looking","investor_presentation",
                              "conference_call_transcript"},
    "news_sentiment": {"earnings_news","general_news","management_news",
                       "product_news","analyst_news","market_reaction",
                       "breaking_news"},
    "segment": {"segment","financials","md&a","income_statement",
                "geographic_segment","regional_performance","business_line"},
    "balance_sheet": {"balance_sheet","financials","key_ratios",
                      "computed_metrics","cash_flow_statement","fundamentals",
                      "financial_table"},
    "investment_thesis": {"computed_metrics","earnings_discussion","guidance",
                        "md&a","fundamentals"},
    "company_health": {"computed_metrics","financials","key_ratios","fundamentals"},
    "evidence": {"computed_metrics","financials","earnings_discussion","fundamentals"},
    "contradictions": {"computed_metrics","financials","earnings_discussion"},
    "scenario": {"computed_metrics","guidance","earnings_discussion","md&a"},
    "price": {"price_history","analyst_price_target","valuation_summary"},
    "comparison": {"computed_metrics","income_statement","key_ratios",
                   "peer_analysis","benchmarking","competitive_position"},
    "macro_context": {"macro_context"},
}

_BM25_SYNONYMS: Dict[str, List[str]] = {
    "roe"            : ["return on equity", "return equity"],
    "roa"            : ["return on assets", "return assets"],
    "roic"           : ["return on invested capital", "return invested capital"],
    "eps"            : ["earnings per share", "diluted earnings", "basic earnings"],
    "fcf"            : ["free cash flow", "cash generation"],
    "ebitda"         : ["earnings before interest", "operating cash"],
    "capex"          : ["capital expenditure", "capital spending", "purchases of property"],
    "pe"             : ["price to earnings", "price earnings ratio"],
    "p/e"            : ["price to earnings", "price earnings"],
    "revenue"        : ["net revenue", "total revenue", "net sales", "total sales"],
    "dividend"       : ["cash dividend", "dividend payout", "shareholder distribution",
                        "capital return", "quarterly dividend"],
    "dividend yield" : ["cash dividend", "dividend payout", "shareholder distribution",
                        "annual dividend", "dividend rate"],
    "buyback"        : ["share repurchase", "stock repurchase", "repurchased shares",
                        "buyback program", "repurchase program"],
    "gross margin"   : ["gross profit margin", "cost of goods sold", "cost of revenue"],
    "operating margin": ["operating income margin", "income from operations"],
    "net margin"     : ["net income margin", "profit margin", "net profit"],
    "current ratio"  : ["current assets", "current liabilities", "liquidity"],
    "debt to equity" : ["total debt", "long-term debt", "financial leverage"],
    "yoy"            : ["year over year", "year-over-year", "prior year", "compared to prior"],
    "qoq"            : ["quarter over quarter", "sequential", "prior quarter"],
    "cagr"           : ["compound annual growth", "compounded annual"],
    "r&d"            : ["research and development", "research development expense"],
    "sga"            : ["selling general administrative", "operating expenses"],
    "risk"           : ["risk factors", "item 1a", "uncertainties", "risks"],
    "guidance"       : ["outlook", "expects", "anticipates", "projects", "forecasts"],
    "transcript"     : ["earnings call", "conference call", "prepared remarks",
                        "question and answer", "operator"],
}


def expand_query_for_bm25(query: str) -> List[str]:
    """
    Expands BM25 query tokens using both raw tokens and synonyms
    tied to _QUERY_CLASSES and BM25_SYNONYMS.
    """
    q = query.lower()
    toks = nltk.word_tokenize(q)
    expanded = set(toks)

    for cls, kws in _QUERY_CLASSES.items():
        for kw in kws:
            if kw in q:
                synonyms = _BM25_SYNONYMS.get(kw, [])
                for syn in synonyms:
                    expanded.update(nltk.word_tokenize(syn))

    return list(expanded)


# ══════════════════════════════════════════════════════════════════════════
# classify() — two-stage: keywords first (fast, precise for common
# phrasings), semantic similarity as a backup only when keyword confidence
# is weak. Replaces the earlier classify_query_intent()/get_weighted_sections()
# duplicate path — those are gone; every caller below (dominant_class,
# secondary_classes, get_preferred_sections, get_excluded_sections,
# is_trend_query, hybrid_search, rerank, run_query, _prompt) uses this one
# classify() and automatically benefits from the fallback.
# ══════════════════════════════════════════════════════════════════════════

# 3-5 representative example QUESTIONS per class (full sentences, not
# keywords). Paraphrases of these get caught through embedding similarity
# without needing to enumerate every possible wording.
_CLASS_EXAMPLES = {
    "financial_metric": [
        "what was the company's revenue last quarter",
        "how profitable is this business right now",
        "what numbers explain how the company is performing financially",
        "is the company making good money on each dollar of sales",
        "what's the EBITDA margin",
        "how's the burn rate",
    ],
    "capital_allocation": [
        "what is the company doing with its extra cash",
        "should the company buy back stock or pay a dividend",
        "how is management deploying the cash they generate",
        "is the company better off paying down debt or reinvesting",
    ],
    "growth": [
        "how fast is the company growing",
        "has the business been expanding over the past few years",
        "is the company's growth accelerating or slowing down",
        "how does this year compare to prior years",
        "is user engagement trending up",
        "what's the CAGR over 5 years",
    ],
    "trend_analysis": [
        "how has revenue trended over the past several years",
        "what's the long-term trajectory of margins",
        "has this metric been trending up or down historically",
    ],
    "quarterly_comparison": [
        "how did each of the last 4 quarters compare",
        "show me quarter over quarter performance",
        "what's the quarterly breakdown of revenue",
    ],
    "risk": [
        "what risks does the company face",
        "what could hurt the business going forward",
        "what keeps management worried about the company's future",
        "how does the company ensure it has enough liquidity to operate smoothly",
        "could regulation derail the business",
        "what's the exposure to FX volatility",
    ],
    "management_commentary": [
        "what did the executives say on the earnings call",
        "what is management's outlook for the business",
        "what did leadership say about future plans",
        "what guidance did the company give investors",
        "did Zuckerberg mention AI strategy",
        "what's the tone of leadership on the call",
    ],
    "news_sentiment": [
        "what's the latest news on this company",
        "has anything significant happened recently",
        "what are people saying about the company right now",
        "any recent headlines about this business",
    ],
    "segment": [
        "how is each business division performing",
        "which part of the company makes the most money",
        "how do the different segments compare",
        "what does the geographic breakdown look like",
        "how's Reality Labs vs Family of Apps",
    ],
    "investment_thesis": [
        "why would someone want to invest in this company",
        "what is the bull case here",
        "what would make someone bearish on this stock",
        "is this a good investment right now",
        "is this stock a buy at current multiples",
    ],
    "company_health": [
        "how healthy is this company financially",
        "is this a strong, stable business",
        "how would you rate the overall quality of this company",
        "is the company in good shape",
    ],
    "scenario": [
        "what would need to happen for this view to change",
        "what could flip the current outlook",
        "under what conditions would this thesis break",
        "what's the worst case scenario here",
    ],
    "evidence": [
        "what supports this conclusion",
        "why do you believe that",
        "what's the reasoning behind this view",
        "can you back that up with data",
    ],
    "contradictions": [
        "are there any mixed signals in the data",
        "do any of the metrics conflict with each other",
        "is there anything inconsistent in the company's numbers",
        "are there red flags that contradict the good news",
    ],
    "balance_sheet": [
        "how much debt does the company have",
        "what does the balance sheet look like",
        "how much cash does the company have on hand",
        "what are the company's total assets and liabilities",
        "what's the liquidity position",
        "what's the cash runway",
    ],
    "price": [
        "what is the stock price right now",
        "how has the share price moved recently",
        "what's the 52-week high and low",
        "how is the stock trading today",
    ],
    "comparison": [
        "how does this company compare to its competitors",
        "which company is performing better",
        "how does this stack up against similar businesses",
        "who is winning in this industry",
    ],
    "assumptions": [
        "what assumptions underlie this claim",
        "what's the basis for saying AI + hardware = success",
        "what implicit assumptions are baked into this thesis",
        "what needs to be true for this strategy to work",
        "what expectations drive this outlook",
    ],
}


# lazily computed once, reused across every call
_class_example_embeddings = None


def _get_class_example_embeddings():
    global _class_example_embeddings
    if _class_example_embeddings is None:
        _class_example_embeddings = {}
        for cls, examples in _CLASS_EXAMPLES.items():
            _class_example_embeddings[cls] = EMBEDDER.encode(examples, normalize_embeddings=True)
    return _class_example_embeddings


def classify_semantic(query: str, min_similarity: float = 0.45) -> Dict[str, float]:
    """
    Compares the query's meaning against example questions per class,
    rather than matching literal words. Catches phrasings that share no
    vocabulary with the keyword list (e.g. "how does JPMorgan ensure
    sufficient liquidity" scoring high against the risk class's liquidity
    example, despite containing none of the risk keywords like "risk" or
    "regulatory").
    """
    class_embs = _get_class_example_embeddings()
    query_emb = EMBEDDER.encode([query], normalize_embeddings=True)[0]

    scores = {}
    for cls, embs in class_embs.items():
        sims = util.cos_sim(query_emb, embs)[0]
        best = float(sims.max())
        if best >= min_similarity:
            scores[cls] = best
    return scores


def _classify_keywords(query: str) -> Dict[str, float]:
    """
    Scores each query class by keyword hits, weighted by specificity:
    3-word+ financial terms count for more than bare single words, since a
    bare match on "margin" is a much weaker signal than a match on
    "return on equity". Returns a weight per class that scored > 0.
    """
    q       = query.lower()
    weights = {}
    for cls, kws in _QUERY_CLASSES.items():
        score = 0
        for kw in kws:
            if kw not in q:
                continue
            word_count = kw.count(" ") + 1
            if word_count >= 3:
                score += 3
            elif word_count == 2:
                score += 2
            else:
                score += 1
        if score > 0:
            base = 0.4
            if cls == "assumptions":
                base = 0.5  # slightly higher baseline
            weights[cls] = min(base + score * 0.08, 1.0)
    return weights

def classify(query: str) -> Dict[str, float]:
    """
    Classification with keyword-first, semantic fallback.
    - Fast: keyword scores dominate.
    - Robust: semantic runs only if keyword confidence is weak.
    """
    kw_scores = _classify_keywords(query)
    top_score = max(kw_scores.values()) if kw_scores else 0
    is_weak = top_score < 0.55 or "general" in kw_scores

    if is_weak:
        sem_scores = classify_semantic(query)
        if sem_scores:
            # merge keyword + semantic (semantic only fills gaps)
            for cls, score in sem_scores.items():
                kw_scores[cls] = max(kw_scores.get(cls, 0), score)

    return kw_scores


def analyze_query(query: str) -> dict:
    """
    Classify a user query exactly once and derive all classification-based
    routing information from the same classification result.

    This is the single entry point for query classification in the live
    pipeline. Downstream functions should consume this result instead of
    calling classify(query) again.
    """
    weights = classify(query)

    if not weights:
        return {
            "weights": {},
            "dominant_class": None,
            "secondary_classes": [],
            "preferred_sections": set(),
            "excluded_sections": set(),
            "is_trend": False,
            "needs_computed_metrics": needs_computed_metrics(query),
            "needs_transcript": is_transcript_query(query),
        }

    dom = dominant_class(weights)
    secondary = secondary_classes(weights)

    preferred = set(_SECTION_PREFERENCES.get(dom, set()))
    for cls in secondary:
        preferred |= _SECTION_PREFERENCES.get(cls, set())

    excluded = set(_SECTION_EXCLUSIONS.get(dom, set()))
    for cls in secondary:
        excluded |= _SECTION_EXCLUSIONS.get(cls, set())

    return {
    "weights": weights,
    "dominant_class": dom,
    "secondary_classes": secondary,
    "preferred_sections": preferred,
    "excluded_sections": excluded,
    "is_trend": dom in ("trend_analysis", "quarterly_comparison"),
    "needs_computed_metrics": needs_computed_metrics(query),
    "needs_transcript": is_transcript_query(query),
}

def export_classification(query: str) -> dict:
    """
    Export classification results in structured JSON format.
    Useful for LLMs or downstream analysis.
    """
    weights = classify(query)
    return {
        "dominant_class": dominant_class(weights) if weights else None,
        "secondary_classes": secondary_classes(weights),
        "preferred_sections": list(get_preferred_sections(query)),
        "excluded_sections": list(get_excluded_sections(query)),
        "weights": weights,
    }


def classify_with_confidence(query: str) -> dict:
    """
    Adds confidence labels (High/Medium/Low) to classification.
    """
    weights = classify(query)
    if not weights:
        return {"dominant_class": None, "confidence": "Low", "weights": {}}

    top_score = max(weights.values())
    if top_score >= 0.8:
        conf = "High"
    elif top_score >= 0.6:
        conf = "Medium"
    else:
        conf = "Low"

    return {
        "dominant_class": dominant_class(weights),
        "confidence": conf,
        "weights": weights,
    }


def route_query(query: str, ticker: str, company: str):
    """
    Route query intent to the correct analytic function in the intelligence layer.
    Returns structured JSON with class, confidence, and analysis.
    """
    cls_info = classify_with_confidence(query)
    cls = cls_info["dominant_class"]
    metrics = get_metrics(ticker, company)
    sector_ctx = get_sector_context(ticker)

    result = {"query_class": cls, "confidence": cls_info["confidence"]}

    if cls == "financial_metric":
        result["analysis"] = metrics
    elif cls == "capital_allocation":
        result["analysis"] = score_capital_allocation(metrics)
    elif cls == "company_health":
        result["analysis"] = build_company_health(metrics, sector_ctx)
    elif cls == "growth":
        result["analysis"] = metrics.get("revenue_analysis", {})
    elif cls == "risk":
        result["analysis"] = metrics.get("risk_profile", {})
    else:
        result["analysis"] = {"message": f"No direct handler for class {cls}"}

    return result


def dominant_class(weights: dict) -> str:
    return max(weights, key=weights.get)


def secondary_classes(weights: dict, threshold: float = 0.85) -> List[str]:
    """
    Returns other classes that scored close to the dominant one (within
    `threshold` ratio of the top score), so a query scoring near-equally
    on two classes gets both classes' preferred/excluded sections merged
    instead of only the winner's.
    """
    if not weights:
        return []
    top = max(weights.values())
    dom = dominant_class(weights)
    return [c for c, w in weights.items() if c != dom and w >= top * threshold]


def needs_computed_metrics(query: str) -> bool:
    """
    Dynamically checks if computed_metrics must be injected.
    Uses keywords from _QUERY_CLASSES['financial_metric'] instead of a static list,
    plus broad performance/correlation language — "how has X performed"
    style questions need real grounded numbers even when they don't use
    an explicit financial-metric keyword.
    """
    q = query.lower()
    triggers = _QUERY_CLASSES.get("financial_metric", [])

    for t in triggers:
        if " " in t:
            if t in q:
                return True
        else:
            if re.search(rf"\b{t}\b", q):
                return True

    performance_triggers = ["performance", "correlat", "how has", "track record"]
    return any(t in q for t in performance_triggers)

def is_trend_query(query_analysis: dict) -> bool:
    """
    Return whether the already-classified query is a multi-period trend query.
    Does not classify the query.
    """
    return bool(query_analysis.get("is_trend", False))

def is_transcript_query(query: str) -> bool:
    """
    Returns True when the user explicitly wants earnings call content.
    """
    q = query.lower()
    return any(t in q for t in [
        "transcript", "earnings call", "conference call",
        "what did", "what was said", "earnings transcript",
        "summarize call", "call summary",
    ])

def get_preferred_sections(query_analysis: dict) -> set:
    """
    Return preferred retrieval sections from an already-computed query
    analysis. Does not classify the query.
    """
    return query_analysis.get("preferred_sections", set())

def get_excluded_sections(query_analysis: dict) -> set:
    """
    Return excluded retrieval sections from an already-computed query
    analysis. Does not classify the query.
    """
    return query_analysis.get("excluded_sections", set())

# ──────────────────────────────────────────────────────────────────────────────
# chunk creation utilities
# ──────────────────────────────────────────────────────────────────────────────

def split_sentences(text: str) -> list:
    text = re.sub(r"\b(Mr|Mrs|Ms|Dr|Corp|Inc|Ltd|Co|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.", r"\1<DOT>", text)
    text = re.sub(r"(\d+)\.(\d+)", r"\1<DOT>\2", text)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    return [p.replace("<DOT>", ".").strip() for p in parts if p.strip()]

def strip_boilerplate_sentences(sents: list) -> list:
    """
    Removes only the individual sentences that ARE boilerplate (short
    cover-page/legal noise), keeping the rest of the chunk intact —
    instead of the old approach of discarding an entire chunk because
    it contained 2+ boilerplate substrings anywhere in it.
    """
    out = []
    for s in sents:
        sl = s.lower()
        if sum(1 for b in _BOILERPLATE if b in sl) >= 1 and len(s.split()) < 25:
            continue
        out.append(s)
    return out

def build_chunks(text, company, ticker, form_type, filed_date, url,
                 stype=None, target_words=400, min_quality=0.15):
    sents  = strip_boilerplate_sentences(split_sentences(text))
    out    = []
    buf    = []
    buf_n  = 0
    prefix = f"{company} ({ticker}) {form_type} ({filed_date}): "
    total  = len(text.split())
    done   = 0
    eff_st = stype or "sec_content"

    for i, sent in enumerate(sents):
        sw = sent.split()
        sn = len(sw)
        if buf_n + sn > target_words and buf:
            body = prefix + " ".join(buf)
            q    = chunk_quality(body)
            if not is_junk_chunk(body) and q >= min_quality:
                pct_pos = round(done / total * 100) if total else 0
                pos     = "beginning" if pct_pos < 15 else "end" if pct_pos >= 85 else "middle"
                section = tag_section(body, form_type, eff_st)
                out.append({
                    "content"    : body,
                    "source_type": eff_st,
                    "ticker"     : ticker,
                    "company"    : company,
                    "date"       : filed_date,
                    "word_count" : len(body.split()),
                    "language"   : detect_lang(body),
                    "quality"    : q,
                    "metadata"   : {
                        "form_type" : form_type,
                        "filed_date": filed_date,
                        "url"       : url,
                        "section": section,
                        "position"  : pos,
                        "chunk_num" : len(out) + 1,
                    }
                })
            done  += buf_n
            buf    = " ".join(sents[max(0,i-2):i]).split() + sw
            buf_n  = len(buf)
        else:
            buf.extend(sw)
            buf_n += sn

    if buf:
        body = prefix + " ".join(buf)
        q    = chunk_quality(body)
        if not is_junk_chunk(body) and q >= min_quality:
            out.append({
                "content"    : body,
                "source_type": eff_st,
                "ticker"     : ticker,
                "company"    : company,
                "date"       : filed_date,
                "word_count" : len(body.split()),
                "language"   : detect_lang(body),
                "quality"    : q,
                "metadata"   : {
                    "form_type" : form_type,
                    "filed_date": filed_date,
                    "url"       : url,
                    "section"   : tag_section(body, form_type, eff_st),
                    "position"  : "end",
                    "chunk_num" : len(out) + 1,
                }
            })
    return out


# ──────────────────────────────────────────────────────────────────────────────
# data collection — news
# ──────────────────────────────────────────────────────────────────────────────

_RSS = {
    "yahoo"        : "https://feeds.finance.yahoo.com/rss/2.0/headline?s={t}&region=US&lang=en-US",
    "seeking_alpha": "https://seekingalpha.com/symbol/{t}.xml",
    "marketwatch"  : "https://feeds.marketwatch.com/marketwatch/topstories/",
    "mktpulse"     : "https://feeds.marketwatch.com/marketwatch/marketpulse/",
    "cnbc_finance" : "https://www.cnbc.com/id/10000664/device/rss/rss.html",
    "cnbc_earnings": "https://www.cnbc.com/id/15839135/device/rss/rss.html",
}


def fetch_rss(ticker: str, company: str) -> list:
    t     = re.sub(r"\.(NS|BO)|(-USD)|(=F)", "", ticker)
    w     = company.split()[0] if company.split() else ticker
    out   = []
    seen  = set()

    def _one(name, url, filter_rel=True):
        try:
            entries = feedparser.parse(url).entries[:25]
            if not entries:
                log.warning(f"rss/{name}: empty")
                return []
            result = []
            for e in entries:
                title   = getattr(e, "title", "")
                summary = getattr(e, "summary", getattr(e, "description", ""))
                link    = getattr(e, "link", "")
                pub     = getattr(e, "published", getattr(e, "updated", ""))
                combined = f"{title} {summary}".lower()
                if filter_rel:
                    rel = 0
                    if t.lower() in combined:                  rel += 3
                    if company.lower() in combined:            rel += 5
                    if w.lower() in combined and len(w) > 3:  rel += 2
                    if rel == 0:
                        continue
                else:
                    rel = 1
                ct, cs    = clean(title), clean(summary)
                if not ct:
                    continue
                lbl, conf = finbert(f"{ct}. {cs}")
                sec       = tag_section(f"{ct} {cs}", source_type="news")
                body      = f"{company} news ({lbl}): {ct}. {cs[:300] if cs else ''}. source: {name}. published: {pub}."
                result.append({
                    "content"    : body,
                    "source_type": f"{name}_news",
                    "ticker"     : ticker,
                    "company"    : company,
                    "date"       : pub[:10] if pub else str(datetime.now().date()),
                    "word_count" : len(body.split()),
                    "language"   : "en",
                    "quality"    : chunk_quality(body),
                    "metadata"   : {"sentiment_label":lbl,"confidence":conf,"source":name,"url":link,"relevance":rel,"section":sec},
                })
            log.info(f"rss/{name}: {len(result)} relevant")
            return result
        except Exception as e:
            log.warning(f"rss/{name}: {e}")
            return []

    for name, tmpl in [("yahoo",_RSS["yahoo"]),("seeking_alpha",_RSS["seeking_alpha"])]:
        out.extend(_one(name, tmpl.format(t=t), filter_rel=False))
    for name, url in [("marketwatch",_RSS["marketwatch"]),("mktpulse",_RSS["mktpulse"]),
                      ("cnbc_finance",_RSS["cnbc_finance"]),("cnbc_earnings",_RSS["cnbc_earnings"])]:
        out.extend(_one(name, url, filter_rel=True))

    deduped = []
    for c in out:
        fp = c["content"][:80].lower()
        if fp not in seen:
            seen.add(fp)
            deduped.append(c)

    src_counts = Counter(c["source_type"] for c in deduped)
    print(f"\n  rss news: {sum(src_counts.values())} total")
    for s, n in src_counts.items():
        print(f"    {s}: {n}")
    return deduped


def fetch_newsapi(ticker: str, company: str) -> dict:
    t = re.sub(r"\.(NS|BO)|(-USD)|(=F)", "", ticker)
    w = company.split()[0] if company.split() else ticker

    def _get(kw):
        try:
            url = (f"https://newsapi.org/v2/everything?q={kw}&language=en"
                   f"&sortBy=publishedAt&pageSize=20&apiKey={os.getenv('NEWS_API_KEY','')}")
            return requests.get(url).json().get("articles", [])
        except Exception as e:
            log.warning(f"newsapi failed for '{kw}': {e}")
            return []

    raw, seen = [], set()
    for kw in list(dict.fromkeys([company, w, t])):
        if len(raw) >= 20:
            break
        for a in _get(kw):
            u = a.get("url","")
            if u not in seen:
                seen.add(u)
                raw.append(a)

    scored = []
    for a in raw:
        txt = f"{a.get('title','')} {a.get('description','')}".lower()
        s   = 0
        if t.lower() in txt:                 s += 3
        if company.lower() in txt:           s += 5
        if w.lower() in txt and len(w) > 3:  s += 2
        if s > 0:
            a["rel"] = s
            scored.append(a)
    scored.sort(key=lambda x: x["rel"], reverse=True)

    seen_fp, seen_url, deduped = set(), set(), []
    for a in scored[:20]:
        u  = a.get("url","")
        fp = ((a.get("title") or "").lower(), (a.get("description") or "").lower())
        if u not in seen_url and fp not in seen_fp:
            seen_url.add(u)
            seen_fp.add(fp)
            deduped.append(a)

    articles = []
    for a in deduped[:15]:
        h, d      = a.get("title",""), a.get("description","")
        lbl, conf = finbert(f"{clean(h)}. {clean(d)}")
        articles.append({
            "title"          : clean(h),
            "description"    : clean(d),
            "source"         : a.get("source",{}).get("name",""),
            "published_at"   : a.get("publishedAt",""),
            "url"            : a.get("url",""),
            "sentiment"      : conf,
            "sentiment_label": lbl,
        })

    avg = sum(a["sentiment"] for a in articles) / len(articles) if articles else 0
    return {"articles": articles, "avg_sentiment": round(avg,3), "total": len(articles)}


# ──────────────────────────────────────────────────────────────────────────────
# data collection — price and fundamentals
# ──────────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────
# data collection — macroeconomic (FRED, BLS, BEA, earnings calendar)
# ──────────────────────────────────────────────────────────────────────────────

def fetch_fred_series(series_id: str, start_date: str = "2015-01-01") -> dict:
    try:
        r = retry(lambda: requests.get(
            "https://api.stlouisfed.org/fred/series/observations",
            params={"series_id": series_id, "api_key": FRED_API_KEY,
                    "file_type": "json", "observation_start": start_date},
            timeout=15))
        return {o["date"]: float(o["value"]) for o in r.json().get("observations", [])
                if o["value"] != "."}
    except Exception as e:
        log.error(f"fred {series_id} failed: {e}")
        return {}


def fetch_bls_series(series_ids: list, start_year: str, end_year: str) -> dict:
    try:
        r = retry(lambda: requests.post(
            "https://api.bls.gov/publicAPI/v2/timeseries/data/",
            json={"seriesid": series_ids, "startyear": start_year,
                  "endyear": end_year, "registrationkey": BLS_API_KEY},
            timeout=15))
        out = {}
        for series in r.json().get("Results", {}).get("series", []):
            data_points = {}
            for d in series["data"]:
                key = f"{d['year']}-{d['period'][1:]}"
                val = d["value"]
                try:
                    data_points[key] = float(val)
                except ValueError:
                    data_points[key] = None  # or np.nan if you prefer
            out[series["seriesID"]] = data_points
        return out
    except Exception as e:
        log.error(f"bls fetch failed: {e}")
        return {}



def fetch_bea_data(table_name: str, frequency: str = "Q") -> list:
    try:
        r = retry(lambda: requests.get(
            "https://apps.bea.gov/api/data",
            params={"UserID": BEA_API_KEY, "method": "GetData", "datasetname": "NIPA",
                    "TableName": table_name, "Frequency": frequency,
                    "Year": "ALL", "ResultFormat": "JSON"},
            timeout=15))
        return r.json().get("BEAAPI", {}).get("Results", {}).get("Data", [])
    except Exception as e:
        log.error(f"bea fetch failed: {e}")
        return []


def fetch_bea_series(table_name: str, line_number: str, frequency: str = "Q") -> dict:
    """
    Reshapes BEA's raw NIPA table response into a {date: value} series,
    same shape FRED/BLS already produce — lets it feed straight into
    build_trend_analysis() without a separate code path.
    """
    raw = fetch_bea_data(table_name, frequency=frequency)
    out = {}
    for row in raw:
        if row.get("LineNumber") != line_number:
            continue
        period = row.get("TimePeriod", "")  # e.g. "2024Q2" or "2024"
        val_str = row.get("DataValue", "").replace(",", "")
        try:
            val = float(val_str)
        except (ValueError, TypeError):
            continue
        # normalize to a date-like key so it sorts correctly alongside
        # FRED/BLS's YYYY-MM-DD keys
        if "Q" in period:
            year, q = period.split("Q")
            month = {"1": "01", "2": "04", "3": "07", "4": "10"}.get(q, "01")
            date_key = f"{year}-{month}-01"
        else:
            date_key = f"{period}-01-01"
        out[date_key] = val
    return out


def get_next_earnings_date(ticker: str) -> Optional[dict]:
    try:
        r = retry(lambda: requests.get(
            f"https://api.nasdaq.com/api/company/{ticker}/earnings-surprise",
            headers={"User-Agent": "Mozilla/5.0"}, timeout=15))
        data = r.json().get("data", {})
        return {"next_date": data.get("nextReportDate"),
                "eps_estimate": data.get("epsForecast")}
    except Exception as e:
        log.warning(f"earnings calendar failed for {ticker}: {e}")
        return None
    
def fetch_price_fundamentals(ticker: str, years: int = 3) -> dict:
    log.info(f"yfinance: {ticker} ({years}y)")
    t      = yf.Ticker(ticker)
    period = {1:"1y",2:"2y",3:"3y",5:"5y"}.get(years,"3y")
    hist   = t.history(period=period).dropna(subset=["Close"])

    prices = [
        {"date":str(d.date()),"open":round(float(r["Open"]),2),"high":round(float(r["High"]),2),
         "low":round(float(r["Low"]),2),"close":round(float(r["Close"]),2),"volume":int(r["Volume"])}
        for d, r in hist.iterrows()
    ]
    info = t.info
    fundamentals = {
        "company_name"    : info.get("longName", ticker),
        "short_name"      : info.get("shortName", ticker),
        "market_cap"      : info.get("marketCap", 0),
        "pe_ratio"        : info.get("trailingPE", 0),
        "forward_pe"      : info.get("forwardPE", 0),
        "eps"             : info.get("trailingEps", 0),
        "revenue"         : info.get("totalRevenue", 0),
        "profit_margin"   : info.get("profitMargins", 0),
        "debt_to_equity"  : info.get("debtToEquity", 0),
        "return_on_equity": info.get("returnOnEquity", 0),
        "return_on_assets": info.get("returnOnAssets", 0),
        "dividend_yield"  : info.get("dividendYield", 0),
        "52w_high"        : info.get("fiftyTwoWeekHigh", 0),
        "52w_low"         : info.get("fiftyTwoWeekLow", 0),
        "free_cash_flow"  : info.get("freeCashflow", 0),
        "target_price"    : info.get("targetMeanPrice", 0),
        "sector"          : info.get("sector", ""),
        "industry"        : info.get("industry", ""),
    }

    print(f"\n  {ticker}: {len(prices)} price points | {prices[0]['date'] if prices else 'N/A'} → {prices[-1]['date'] if prices else 'N/A'}")
    print(f"  market cap: {fmt(fundamentals['market_cap'])}")
    return {"price_history": prices, "fundamentals": fundamentals}


# ──────────────────────────────────────────────────────────────────────────────
# data collection — SEC filings
# ──────────────────────────────────────────────────────────────────────────────
def is_valid_ticker(ticker: str) -> bool:
    """
    Confirms yfinance can actually return price data for this symbol.
    Catches delisted/renamed/invalid tickers before wasting time on
    SEC lookups, news fetches, and metrics computation.
    """
    try:
        hist = yf.Ticker(ticker).history(period="5d")
        return not hist.empty
    except Exception as e:
        log.warning(f"ticker validation failed for {ticker}: {e}")
        return False
    
_cik_table_cache: Optional[dict] = None

def get_cik(ticker: str) -> Optional[str]:
    global _cik_table_cache
    if _cik_table_cache is None:
        try:
            _cik_table_cache = retry(lambda: requests.get(
                "https://www.sec.gov/files/company_tickers.json", headers=SEC_HEADER
            ).json())
            log.info(f"cik table cached: {len(_cik_table_cache)} entries")
        except Exception as e:
            log.error(f"cik table fetch failed: {e}")
            return None
    for entry in _cik_table_cache.values():
        if entry["ticker"].upper() == ticker.upper():
            return str(entry["cik_str"]).zfill(10)
    return None


def fetch_sec_filings(ticker: str, cik: str, years: int = 3) -> list:
    if not cik:
        return []
    try:
        data   = retry(lambda: requests.get(f"https://data.sec.gov/submissions/CIK{cik}.json", headers=SEC_HEADER).json())
        recent = data["filings"]["recent"]
        target = {"10-K","10-Q","8-K","DEF 14A","20-F","6-K"}
        cutoff = (datetime.now() - timedelta(days=years*365)).strftime("%Y-%m-%d")
        out    = []

        for i in range(len(recent["form"])):
            if recent["form"][i] in target and recent["filingDate"][i] >= cutoff:
                acc = recent["accessionNumber"][i].replace("-","")
                out.append({
                    "form_type" : recent["form"][i],
                    "filed_date": recent["filingDate"][i],
                    "accession" : recent["accessionNumber"][i],
                    "filing_url": f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{recent['primaryDocument'][i]}",
                })

        if "files" in data["filings"]:
            for ff in data["filings"]["files"][:3]:
                try:
                    old = retry(lambda: requests.get(f"https://data.sec.gov/submissions/{ff['name']}", headers=SEC_HEADER).json())
                    for i in range(len(old.get("form",[]))):
                        if old["form"][i] in target and old["filingDate"][i] >= cutoff:
                            acc = old["accessionNumber"][i].replace("-","")
                            out.append({
                                "form_type" : old["form"][i],
                                "filed_date": old["filingDate"][i],
                                "accession" : old["accessionNumber"][i],
                                "filing_url": f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{old['primaryDocument'][i]}",
                            })
                except Exception as e:
                    log.warning(f"older filings page: {e}")

        print(f"\n  sec filings ({ticker}):")
        for form, n in Counter(f["form_type"] for f in out).items():
            print(f"    {form}: {n}")
        print(f"    total: {len(out)}")
        return out[:40]
    except Exception as e:
        log.error(f"sec filings failed: {e}")
        return []


def _extract_xbrl(soup) -> dict:
    xbrl = {}
    for tag in ["us-gaap:revenues","us-gaap:netincomeloss","us-gaap:earningspersharebasic",
                "us-gaap:assets","us-gaap:liabilities","us-gaap:operatingincomeloss",
                "us-gaap:grossprofit","us-gaap:researchanddevelopmentexpense",
                "us-gaap:cashandcashequivalentsatcarryingvalue"]:
        for el in soup.find_all(tag):
            txt = el.get_text(strip=True)
            if txt and txt.replace(",","").replace(".","").replace("-","").isdigit():
                label = tag.split(":")[-1]
                try:
                    xbrl[label] = float(txt.replace(",",""))
                except ValueError:
                    pass
    return xbrl

REGULATORY_XBRL_TAGS = {
    "tier1_capital_ratio": "Tier1RiskBasedCapitalRatio",
    "tier1_leverage_ratio": "Tier1LeverageRatio",
    "total_capital_ratio": "TotalCapitalRatio",
    "net_interest_margin": "NetInterestMarginPercent",
}


def fetch_xbrl_concept_series(cik: str, tag: str) -> dict:
    if not cik:
        return {}
    url = f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/us-gaap/{tag}.json"
    try:
        r = retry(lambda: requests.get(url, headers=SEC_HEADER, timeout=15))
        if r.status_code != 200:
            return {}
        data = r.json()
        out = {}
        for unit_vals in data.get("units", {}).values():
            for v in unit_vals:
                if v.get("form") in ("10-K", "10-Q") and v.get("end"):
                    out[v["end"]] = v["val"]
        return out
    except Exception as e:
        log.warning(f"xbrl concept {tag} failed for CIK {cik}: {e}")
        return {}


def fetch_regulatory_metrics(cik: str, sector: str) -> dict:
    if not cik or sector not in ("Financial Services", "Financials", "Financial"):
        return {}
    out = {}
    for key, tag in REGULATORY_XBRL_TAGS.items():
        series = fetch_xbrl_concept_series(cik, tag)
        if series:
            out[key] = series
    return out


def find_period_gaps(series: dict, expected_freq_days: int = 90) -> list:
    periods = sorted(series.keys())
    gaps = []
    for i in range(1, len(periods)):
        try:
            d1 = datetime.strptime(periods[i - 1][:10], "%Y-%m-%d")
            d2 = datetime.strptime(periods[i][:10], "%Y-%m-%d")
        except ValueError:
            continue
        if (d2 - d1).days > expected_freq_days * 1.5:
            gaps.append((periods[i - 1], periods[i]))
    return gaps


def compute_ttm(quarterly_series: dict) -> dict:
    periods = sorted(quarterly_series.keys())
    ttm = {}
    for i in range(3, len(periods)):
        window = periods[i - 3:i + 1]
        contiguous = True
        for j in range(3):
            try:
                d1 = datetime.strptime(window[j][:10], "%Y-%m-%d")
                d2 = datetime.strptime(window[j + 1][:10], "%Y-%m-%d")
            except ValueError:
                contiguous = False
                break
            if not (80 <= (d2 - d1).days <= 100):
                contiguous = False
                break
        if contiguous:
            ttm[window[-1]] = round(sum(quarterly_series[p] for p in window), 2)
    return ttm


def get_fiscal_year_end_month(ticker: str) -> int:
    try:
        info = yf.Ticker(ticker).info
        lfye = info.get("lastFiscalYearEnd")
        if lfye:
            return datetime.fromtimestamp(lfye).month
    except Exception as e:
        log.warning(f"fiscal year end lookup failed for {ticker}: {e}")
    return 12


def fiscal_label(period_end: str, fiscal_year_end_month: int) -> str:
    try:
        d = datetime.strptime(period_end[:10], "%Y-%m-%d")
    except ValueError:
        return period_end
    fy = d.year if d.month >= fiscal_year_end_month else d.year - 1
    return f"FY{fy} (period ended {period_end[:10]})"

def _xbrl_text(data: dict, company: str) -> str:
    if not data:
        return ""
    lmap = {
        "revenues":"total revenues","netincomeloss":"net income",
        "earningspersharebasic":"eps basic","assets":"total assets",
        "liabilities":"total liabilities","operatingincomeloss":"operating income",
        "grossprofit":"gross profit","researchanddevelopmentexpense":"r&d",
        "cashandcashequivalentsatcarryingvalue":"cash",
    }
    parts = [f"{lmap.get(k,k)}: {fmt(v)}" for k,v in data.items()]
    return f"{company} financial data from SEC: " + ". ".join(parts) + "." if parts else ""


def _is_transcript(text: str) -> bool:
    markers = ["operator","good morning","good afternoon","ladies and gentlemen","welcome to the",
               "earnings call","question and answer","q&a session","thank you for joining"]
    return sum(1 for m in markers if m in text.lower()) >= 3


def _is_presentation(text: str, form_type: str = None) -> bool:
    # Never classify SEC filings as presentations
    if form_type in ["10-K", "10-Q", "DEF 14A", "8-K"]:
        return False

    # True presentation markers (not boilerplate)
    markers = [
        "investor presentation",
        "investor day",
        "analyst day",
        "slide",
        "agenda",
        "key highlights",
        "financial targets"
    ]

    return sum(1 for m in markers if m in text.lower()) >= 2

def _sort_filings(filings: list) -> list:
    def score(f):
        base = {"10-K":90,"10-Q":85,"20-F":90,"6-K":80,"8-K":80,"DEF 14A":70}.get(f["form_type"],50)
        try:
            age = (datetime.now() - datetime.strptime(f["filed_date"],"%Y-%m-%d")).days
        except Exception:
            age = 999
        return base + (50 if age<=30 else 40 if age<=90 else 30 if age<=180 else 15 if age<=365 else 0) + (15 if f["form_type"]=="8-K" else 0)
    return sorted(filings, key=score, reverse=True)


def _filing_limits(filings: list) -> dict:
    fc = Counter(f["form_type"] for f in filings)
    r8 = sum(1 for f in filings if f["form_type"] in ("8-K","6-K") and
             (datetime.now()-datetime.strptime(f["filed_date"],"%Y-%m-%d")).days<=90)
    if r8 >= 3:
        return {"10-K":3,"10-Q":6,"8-K":6,"DEF 14A":1,"20-F":3,"6-K":6}
    if fc.get("10-K",0) == 0 and fc.get("20-F",0) == 0:
        return {"10-K":0,"10-Q":8,"8-K":7,"DEF 14A":1,"20-F":0,"6-K":8}
    return {"10-K":3,"10-Q":9,"8-K":6,"DEF 14A":1,"20-F":3,"6-K":9}

def extract_tables(soup) -> list:
    """
    Pulls each <table> out as a structured row-by-row text block BEFORE
    the general get_text() flatten destroys row/column meaning. Without
    this, a table like:
        Total net sales | $94,930 | $90,753
    becomes flat text "Total net sales $94,930 $90,753" with no way to
    tell which number is which period — and gets penalized by the
    numeric-density scoring as if it were noise, when it's often the
    highest-value content in the filing.
    """
    tables_out = []
    for table in soup.find_all("table"):
        rows = []
        for tr in table.find_all("tr"):
            cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
            cells = [c for c in cells if c]  # drop empty spacer cells
            if cells:
                rows.append(cells)
        if len(rows) < 2:  # not a real data table (nav/layout table)
            continue

        lines = []
        for row in rows:
            if len(row) == 1:
                lines.append(row[0])
            else:
                label, *vals = row
                lines.append(f"{label}: {' | '.join(vals)}")
        text_repr = "\n".join(lines)

        numeric_cells = sum(1 for row in rows for c in row if re.match(r"^\$?[\d,\.\(\)%-]+$", c))
        total_cells = sum(len(row) for row in rows)
        if total_cells == 0:
            continue

        tables_out.append({
            "text": text_repr,
            "rows": rows,
            "is_financial": (numeric_cells / total_cells) > 0.3,
        })

        table.decompose()  # remove from soup so it's not double-counted in get_text()
    return tables_out


def download_filing(url, ticker, form_type, filed_date, company, counter, thresh=0.15):
    try:
        resp = retry(lambda: requests.get(url, headers=SEC_HEADER, timeout=20))
        if resp.status_code != 200:
            return []

        soup  = BeautifulSoup(resp.content, "html.parser")
        xbrl  = _extract_xbrl(soup)
        xtext = _xbrl_text(xbrl, company)

        for tag in soup(["script","style","noscript","svg","img"]):
            tag.decompose()
        # colon-named tags are XBRL/iXBRL wrappers (us-gaap:, ix:nonfraction, etc).
        # unwrap (keep text, drop tag) instead of decompose (deletes tag AND its
        # text content) — decompose was silently stripping every dollar figure
        # out of every financial table before extract_tables() ever saw them.
        for tag in soup.find_all(lambda t: t.name and ":" in t.name):
            tag.unwrap()

        # NEW: pull tables out with row/column structure intact, before
        # the general text flatten destroys that structure
        tables = extract_tables(soup)

        body = soup.body
        text = body.get_text(separator=" ", strip=True) if body else soup.get_text(separator=" ", strip=True)
        text = clean(text)

        is_txn  = _is_transcript(text)
        is_pres = _is_presentation(text)

        for p in [r"Table of Contents",r"UNITED STATES SECURITIES AND EXCHANGE COMMISSION",
                  r"Washington.*?20549",r"Commission File No\.",r"xbrli:[A-Za-z0-9_]+",
                  r"iso4217:[A-Za-z0-9_]+",r"dei:[A-Za-z0-9_]+",r"us-gaap:[A-Za-z0-9_]+",
                  r"[A-Za-z]+:[A-Za-z0-9_]+Member",r"(?<!\w)Page \d+(?!\w)"]:
            text = re.sub(p, " ", text, flags=re.IGNORECASE)
        text = re.sub(r"\(\s*\)", "", text)
        text = " ".join(text.split())

        if len(text.split()) < 100:
            return []

        stype   = "earnings_transcript" if is_txn else "investor_presentation" if is_pres else "sec_content"
        content = build_chunks(text, company, ticker, form_type, filed_date, url, stype=stype, min_quality=thresh)

        for c in content:
            if is_txn:   c["quality"] = min(c.get("quality",0.5)+0.1, 1.0)
            if is_pres:  c["quality"] = min(c.get("quality",0.5)+0.05, 1.0)
            counter[0] += 1
            c["chunk_id"] = make_cid(ticker, c["source_type"], counter[0])

        # NEW: turn each real financial table into its own dedicated,
        # high-quality chunk — this is often data that has no XBRL tag at
        # all (segment breakdowns, footnote schedules, non-standard line
        # items) and previously only survived as flattened, penalized text
        for tbl in tables:
            if not tbl["is_financial"] or len(tbl["rows"]) < 2:
                continue
            tbl_content = f"{company} ({ticker}) {form_type} ({filed_date}) — financial table:\n{tbl['text']}"
            counter[0] += 1
            content.append({
                "chunk_id"   : make_cid(ticker, "sec_table", counter[0]),
                "content"    : tbl_content,
                "source_type": "sec_content",
                "ticker"     : ticker,
                "company"    : company,
                "date"       : filed_date,
                "word_count" : len(tbl_content.split()),
                "language"   : "en",
                "quality"    : 0.9,
                "metadata"   : {"form_type": form_type, "filed_date": filed_date,
                                 "url": url, "section": "financial_table",
                                 "position": "table", "chunk_num": len(content)+1},
            })

        if xtext:
            counter[0] += 1
            content.insert(0, {
                "chunk_id"   : make_cid(ticker,"xbrl_data",counter[0]),
                "content"    : xtext,
                "source_type": "xbrl_data",
                "ticker"     : ticker,
                "company"    : company,
                "date"       : filed_date,
                "word_count" : len(xtext.split()),
                "language"   : "en",
                "quality"    : 1.0,
                "metadata"   : {"form_type":form_type,"filed_date":filed_date,"url":url,
                                "xbrl_values":xbrl,"section":"xbrl_summary","position":"extracted"},
            })
        return content
    except Exception as e:
        log.error(f"filing download failed: {e}")
        return []


# ──────────────────────────────────────────────────────────────────────────────
# document loader — price/fundamentals/news → chunks
# ──────────────────────────────────────────────────────────────────────────────

def load_base_chunks(doc: dict, counter: list) -> list:
    chunks  = []
    ticker  = doc.get("ticker","")
    company = doc.get("company_name", ticker)

    def mk(content, stype, date, meta=None):
        counter[0] += 1
        q = chunk_quality(content)
        return {
            "chunk_id"   : make_cid(ticker, stype, counter[0]),
            "content"    : content,
            "source_type": stype,
            "ticker"     : ticker,
            "company"    : company,
            "date"       : date,
            "word_count" : len(content.split()),
            "language"   : detect_lang(content),
            "quality"    : q,
            "metadata"   : meta or {},
        }

    f = doc.get("stocks",{}).get("fundamentals",{})
    if f:
        body = (
            f"{company} ({ticker}) fundamentals: market cap {fmt(f.get('market_cap',0))}, "
            f"revenue {fmt(f.get('revenue',0))}, fcf {fmt(f.get('free_cash_flow',0))}, "
            f"pe {f.get('pe_ratio',0)}, forward pe {f.get('forward_pe',0)}, "
            f"profit margin {round(f.get('profit_margin',0)*100,2)}%, eps ${f.get('eps',0)}, "
            f"debt/equity {f.get('debt_to_equity',0)}, roe {round(f.get('return_on_equity',0)*100,2)}%, "
            f"roa {round(f.get('return_on_assets',0)*100,2)}%, "
            f"52w high ${f.get('52w_high',0)}, 52w low ${f.get('52w_low',0)}, "
            f"analyst target {fmt(f.get('target_price',0))}, "
            f"sector {f.get('sector','')}, industry {f.get('industry','')}."
        )
        section = "fundamentals"
        chunks.append(mk(body, "fundamentals", doc.get("collected_at",""), {
            "market_cap":f.get("market_cap",0),"revenue":f.get("revenue",0),
            "pe_ratio":f.get("pe_ratio",0),"profit_margin":f.get("profit_margin",0),
            "sector":f.get("sector",""),"eps":f.get("eps",0),
            "section": section,"tags": tag_section(section,body,"fundamentals"),
        }))

    ph    = doc.get("stocks",{}).get("price_history",[])
    valid = [p for p in ph if p["close"]==p["close"]]
    if valid:
        for days, label in [(22,"1 month"),(63,"3 months"),(126,"6 months"),(252,"1 year")]:
            pd  = valid[-days:] if len(valid)>=days else valid
            if len(pd)<5: continue
            l   = pd[-1]
            o   = pd[0]
            chg = round(l["close"]-o["close"],2)
            pct_chg = round(chg/o["close"]*100,2) if o["close"] else 0
            body = (f"{company} ({ticker}) price ({label}): close ${l['close']} on {l['date']}, "
                    f"{label} ago ${o['close']}, change ${chg} ({pct_chg}%), "
                    f"high ${max(p['high'] for p in pd)}, low ${min(p['low'] for p in pd)}.")
            chunks.append(mk(body, "price_history", l["date"], {
                "latest_close":l["close"],"pct_change":pct_chg,"section":"price_history","period":label,
            }))

    news = doc.get("news",{})
    if isinstance(news, dict):
        for a in news.get("articles",[]):
            if not a.get("title"): continue
            body = (f"{company} news ({a['sentiment_label']}): {a['title']}. "
                    f"{a.get('description','')}. source: {a.get('source','')}.")
            chunks.append(mk(body, "news", a.get("published_at",""), {
                "sentiment_label":a.get("sentiment_label",""),"confidence":a.get("sentiment",0),
                "source":a.get("source",""),"url":a.get("url",""),
                "section":tag_section(body, source_type="news"),
            }))

    for f in doc.get("sec_filings",[]):
        body = (f"{company} SEC filing: {f['form_type']} filed {f['filed_date']}. "
                f"document at {f['filing_url']}.")
        chunks.append(mk(body, "sec_filing", f.get("filed_date",""), {
            "form_type":f.get("form_type",""),"filed_date":f.get("filed_date",""),
            "url":f.get("filing_url",""),"section":tag_section(body, f.get("form_type","")),
        }))

    return chunks


# ──────────────────────────────────────────────────────────────────────────────
# supabase operations — with batch existence check
# ──────────────────────────────────────────────────────────────────────────────

def batch_exists(cids: list) -> set:
    if not cids:
        return set()
    try:
        r = DB.table("finsight_chunks").select("chunk_id").in_("chunk_id", cids).execute()
        return {row["chunk_id"] for row in r.data}
    except Exception as e:
        log.warning(f"batch exists check failed: {e}")
        return set()


def push_chunks(chunks: list, embeddings: np.ndarray) -> int:
    cids    = [c.get("chunk_id","") for c in chunks]
    already = batch_exists(cids)
    rows    = []
    for i, c in enumerate(chunks):
        if c.get("chunk_id","") in already:
            continue
        rows.append({
            "chunk_id"   : c.get("chunk_id",""),
            "ticker"     : c.get("ticker",""),
            "company"    : c.get("company",""),
            "source_type": c.get("source_type",""),
            "content"    : c.get("content",""),
            "word_count" : c.get("word_count",0),
            "language"   : c.get("language","en"),
            "quality"    : float(c.get("quality",0.5)),
            "date"       : c.get("date",""),
            "embedding"  : embeddings[i].tolist(),
            "metadata"   : c.get("metadata",{}),
        })
    if not rows:
        log.info("nothing new to push")
        return 0
    added = 0
    for i in range(0, len(rows), 50):
        batch = rows[i:i+50]
        try:
            DB.table("finsight_chunks").insert(batch).execute()
            added += len(batch)
            log.info(f"db push: batch {i//50+1} — {len(batch)} chunks")
        except Exception as e:
            log.error(f"db insert failed: {e}")
    return added


def vec_search(query: str, ticker: str, k: int = 50, min_q: float = 0.3) -> list:
    qemb = EMBEDDER.encode([query], normalize_embeddings=True)[0].tolist()
    try:
        r = DB.rpc("match_chunks", {"query_embedding":qemb,"match_ticker":ticker,"match_count":k,"min_quality":min_q}).execute()
        return [{
            "chunk_index" : i,
            "chunk_id"    : row["chunk_id"],
            "content"     : row["content"],
            "source_type" : row["source_type"],
            "ticker"      : row["ticker"],
            "company"     : row["company"],
            "date"        : row["date"],
            "hybrid_score": round(float(row["similarity"]),4),
            "quality"     : float(row.get("quality",0.5)),
            "metadata"    : row.get("metadata",{}),
        } for i, row in enumerate(r.data)]
    except Exception as e:
        log.error(f"vector search failed: {e}")
        return []


def load_chunks(ticker: str) -> list:
    try:
        r = DB.table("finsight_chunks").select(
            "chunk_id,ticker,company,source_type,content,date,quality,metadata"
        ).eq("ticker", ticker).execute()
        out = [{
            "chunk_id"   : row["chunk_id"],
            "ticker"     : row["ticker"],
            "company"    : row["company"],
            "source_type": row["source_type"],
            "content"    : row["content"],
            "date"       : row["date"],
            "quality"    : float(row.get("quality",0.5)),
            "metadata"   : row.get("metadata",{}),
        } for row in r.data]
        log.info(f"loaded {len(out)} chunks from db for {ticker}")
        return out
    except Exception as e:
        log.error(f"db load failed: {e}")
        return []


def mark_fresh(ticker: str, stype: str):
    try:
        field = {"news":"last_news_fetch","sec_content":"last_sec_fetch","price_history":"last_price_fetch"}.get(stype,"updated_at")
        DB.table("finsight_data_freshness").upsert({"ticker":ticker,field:datetime.now().isoformat(),"updated_at":datetime.now().isoformat()}).execute()
    except Exception as e:
        log.warning(f"freshness update failed: {e}")


def log_ingest(ticker, stype, status, added=0, error="", dur=0):
    try:
        DB.table("finsight_ingestion_log").insert({"ticker":ticker,"source_type":stype,"status":status,"chunks_added":added,"error_msg":error,"duration_sec":round(dur,2)}).execute()
    except Exception as e:
        log.warning(f"ingest log failed: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# bm25 index — built once per ticker, cached
# ──────────────────────────────────────────────────────────────────────────────

def build_bm25(ticker: str, chunks: list) -> BM25Okapi:
    if ticker in _bm25_cache:
        cached_bm25, cached_chunks = _bm25_cache[ticker]
        cur_ids  = {c.get("chunk_id","") for c in chunks}
        cach_ids = {c.get("chunk_id","") for c in cached_chunks}
        if cur_ids == cach_ids:
            return cached_bm25
    log.info(f"building bm25 index: {ticker} ({len(chunks)} chunks)")
    tok = [nltk.word_tokenize(c["content"].lower()) for c in chunks]
    b   = BM25Okapi(tok)
    _bm25_cache[ticker] = (b, chunks)
    return b


# ──────────────────────────────────────────────────────────────────────────────
# master ingestion
# ──────────────────────────────────────────────────────────────────────────────

def ingest(ticker: str, years: int = 3) -> Tuple[dict, list]:
    start = time.time()
    log.info(f"ingesting {ticker} ({years}y)")

    if not is_valid_ticker(ticker):
        log.error(f"{ticker} appears delisted, renamed, or invalid — aborting ingestion")
        print(f"\n⚠ {ticker} may be delisted or an invalid symbol — no data available.")
        return {}, []

    cid        = get_cik(ticker)
    price_data = fetch_price_fundamentals(ticker, years=years)
    company    = price_data["fundamentals"].get("company_name", ticker)
    short      = price_data["fundamentals"].get("short_name", company)

    news_data = fetch_newsapi(ticker, strip_corp(short))
    sec_data  = fetch_sec_filings(ticker, cid, years=years)

    doc = {
        "ticker"      : ticker,
        "company_name": company,
        "cik"         : cid,
        "collected_at": str(datetime.now()),
        "stocks"      : price_data,
        "news"        : news_data,
        "sec_filings" : sec_data,
    }

    counter = [0]
    chunks  = load_base_chunks(doc, counter)

    # build financial intelligence layer — this is the anti-hallucination core
    log.info(f"building financial intelligence layer for {ticker}...")
    metrics = get_metrics(ticker, company, fundamentals=price_data.get("fundamentals", {}), force_refresh=True)
    mc      = build_metrics_chunk(ticker, company, metrics)
    counter[0] += 1
    mc["chunk_id"] = make_cid(ticker, "computed_metrics", counter[0])
    chunks.append(mc)

    # old-style structured metrics chunks (for embedding diversity)
    t_yf = yf.Ticker(ticker)
    sm   = []
    for label, fn, max_cols, rows in [
        ("annual income",    lambda: t_yf.income_stmt,           4, ["Total Revenue","Gross Profit","Operating Income","Net Income","EBITDA","Basic EPS"]),
        ("quarterly income", lambda: t_yf.quarterly_income_stmt, 8, ["Total Revenue","Gross Profit","Operating Income","Net Income","Basic EPS"]),
        ("balance sheet",    lambda: t_yf.balance_sheet,         4, ["Total Assets","Stockholders Equity","Total Debt","Cash And Cash Equivalents","Current Assets","Current Liabilities"]),
        ("cash flow",        lambda: t_yf.cashflow,              4, ["Operating Cash Flow","Free Cash Flow","Capital Expenditure"]),
    ]:
        try:
            stmt = fn()
            if stmt is None or stmt.empty:
                continue
            for col in stmt.columns[:max_cols]:
                period = str(col.date()) if hasattr(col,"date") else str(col)
                met    = {}
                for row in rows:
                    if row in stmt.index:
                        v = stmt.loc[row,col]
                        if v is not None and not (isinstance(v,float) and np.isnan(v)):
                            met[row] = float(v)
                if not met:
                    continue
                lines   = [f"{k}: {fmt(v)}" for k,v in met.items()]
                content = f"{company} ({ticker}) {label} ({period}): " + ". ".join(lines) + "."
                counter[0] += 1
                sm.append({
                    "chunk_id"   : make_cid(ticker,"structured_metrics",counter[0]),
                    "content"    : content,
                    "source_type": "structured_metrics",
                    "ticker"     : ticker,
                    "company"    : company,
                    "date"       : period,
                    "word_count" : len(content.split()),
                    "language"   : "en",
                    "quality"    : 0.95,
                    "metadata"   : {
                        "statement_type": label.replace(" ","_"),
                        "period"        : period,
                        "frequency"     : "annual" if "annual" in label else "quarterly",
                        "section"       : tag_section(content, source_type="structured_metrics"),
                        "raw_metrics"   : met,
                    }
                })
        except Exception as e:
            log.warning(f"structured metric {label}: {e}")
    chunks.extend(sm)
    print(f"\n  computed metrics: {len(sm)+1} chunks (1 computed + {len(sm)} raw statements)")

    # rss news
    rss = fetch_rss(ticker, company)
    for c in rss:
        counter[0] += 1
        c["chunk_id"] = make_cid(ticker, c.get("source_type","rss_news"), counter[0])
    chunks.extend(rss)

    # sec filing content
    print(f"\n  sec content download — {ticker}")
    sorted_fl = _sort_filings(sec_data)
    limits    = _filing_limits(sec_data)
    fc        = Counter()
    n_dl      = 0

    for fl in sorted_fl:
        if n_dl >= 18: break
        form  = fl["form_type"]
        limit = limits.get(form, 2)
        if fc[form] >= limit: continue
        time.sleep(0.5)
        content = download_filing(fl["filing_url"], ticker, form, fl["filed_date"], company, counter)
        if content:
            chunks.extend(content)
            n_dl += 1
            fc[form] += 1
            types = Counter(c["source_type"] for c in content)
            note  = " (transcript)" if "earnings_transcript" in types else " (presentation)" if "investor_presentation" in types else ""
            print(f"    {form} ({fl['filed_date']}): {len(content)} chunks{note}")
        else:
            print(f"    {form} ({fl['filed_date']}): skipped")

    chunks = dedup(chunks)

    print(f"\n  chunk summary — {ticker}")
    for stype, n in sorted(Counter(c["source_type"] for c in chunks).items()):
        print(f"    {stype}: {n}")
    print(f"    total: {len(chunks)}")

    log.info(f"embedding {len(chunks)} chunks...")
    emb = EMBEDDER.encode([c["content"] for c in chunks], normalize_embeddings=True, show_progress_bar=True)
    emb = np.array(emb, dtype="float32")

    log.info("pushing to supabase...")
    added = push_chunks(chunks, emb)

    mark_fresh(ticker,"sec_content")
    mark_fresh(ticker,"news")
    mark_fresh(ticker,"price_history")

    dur = round(time.time()-start,2)
    log_ingest(ticker,"full_ingestion","success",added,dur=dur)
    build_bm25(ticker, chunks)

    if len(chunks) < 5:
        log.error(f"{ticker}: only {len(chunks)} chunks produced — likely a data source "
                  f"failure (SEC/yfinance/news), not just a data-poor ticker. Investigate.")

    log.info(f"ingestion done — {len(chunks)} total, {added} new, {dur}s")
    return doc, chunks

def ingest_macro_data(years: int = 10):
    """
    Ticker-agnostic — runs once for a full historic backfill, or with a
    short `years` window as a cheap periodic refresh. push_chunks() already
    upserts on chunk_id, so re-running this is safe and just updates values.
    """
    start_date = (datetime.now() - timedelta(days=years*365)).strftime("%Y-%m-%d")
    chunks = []

    for label, sid in FRED_SERIES.items():
        series = fetch_fred_series(sid, start_date=start_date)
        if series:
            chunks.append(build_macro_chunk(series, label))

    bls_ids = list(BLS_SERIES.values())
    bls_data = fetch_bls_series(bls_ids, str(datetime.now().year - years), str(datetime.now().year))
    for label, sid in BLS_SERIES.items():
        series = bls_data.get(sid, {})
        if series:
            chunks.append(build_macro_chunk(series, label))

    for label, cfg in BEA_SERIES.items():
        series = fetch_bea_series(cfg["table"], cfg["line"])
        if series:
            chunks.append(build_macro_chunk(series, label))

    if not chunks:
        log.warning("macro ingestion produced 0 chunks — check API keys")
        return

    emb = EMBEDDER.encode([c["content"] for c in chunks], normalize_embeddings=True)
    emb = np.array(emb, dtype="float32")
    added = push_chunks(chunks, emb)
    log.info(f"macro ingestion: {len(chunks)} indicators, {added} new/updated")


def refresh_macro_data_live():
    """Scheduled daily — FRED/BLS/BEA update monthly at most, so daily is generous headroom."""
    ingest_macro_data(years=1)

# ──────────────────────────────────────────────────────────────────────────────
# background refresh
# ──────────────────────────────────────────────────────────────────────────────

WATCHED: List[str] = []


def refresh_news(ticker: str):
    log.info(f"news refresh: {ticker}")
    start = time.time()
    try:
        pd      = fetch_price_fundamentals(ticker, years=1)
        company = pd["fundamentals"].get("company_name", ticker)
        short   = pd["fundamentals"].get("short_name", company)
        counter = [9000]
        all_news = []

        nd = fetch_newsapi(ticker, strip_corp(short))
        for a in nd.get("articles",[]):
            if not a.get("title"): continue
            body = (f"{company} news ({a['sentiment_label']}): {a['title']}. "
                    f"{a.get('description','')}. source: {a.get('source','')}.")
            counter[0] += 1
            all_news.append({
                "chunk_id"   : make_cid(ticker,"news",counter[0]),
                "content"    : body,"source_type":"news","ticker":ticker,
                "company"    : company,"date":a.get("published_at",""),
                "word_count" : len(body.split()),"language":"en",
                "quality"    : chunk_quality(body),
                "metadata"   : {"sentiment_label":a.get("sentiment_label",""),"section":tag_section(body,source_type="news")},
            })

        rss = fetch_rss(ticker, company)
        for c in rss:
            counter[0] += 1
            c["chunk_id"] = make_cid(ticker, c.get("source_type","rss_news"), counter[0])
        all_news.extend(rss)

        if all_news:
            emb   = EMBEDDER.encode([c["content"] for c in all_news], normalize_embeddings=True)
            emb   = np.array(emb, dtype="float32")
            added = push_chunks(all_news, emb)
            mark_fresh(ticker, "news")
            log.info(f"news refresh: {added} new for {ticker}")

        log_ingest(ticker,"news_refresh","success",len(all_news),dur=round(time.time()-start,2))
    except Exception as e:
        log.error(f"news refresh failed: {e}")
        log_ingest(ticker,"news_refresh","failed",error=str(e))


def refresh_sec(ticker: str):
    log.info(f"sec refresh: {ticker}")
    start = time.time()
    try:
        cid     = get_cik(ticker)
        sec     = fetch_sec_filings(ticker, cid, years=1)
        if not sec: return
        pd      = fetch_price_fundamentals(ticker, years=1)
        company = pd["fundamentals"].get("company_name", ticker)
        limits  = {"10-K":1,"10-Q":2,"8-K":3}
        fc, counter, new = Counter(), [8000], []
        for fl in _sort_filings(sec):
            if sum(fc.values()) >= 6: break
            form = fl["form_type"]
            if fc[form] >= limits.get(form,1): continue
            time.sleep(0.5)
            content = download_filing(fl["filing_url"],ticker,form,fl["filed_date"],company,counter)
            if content:
                new.extend(content)
                fc[form] += 1
        if new:
            emb   = EMBEDDER.encode([c["content"] for c in new], normalize_embeddings=True)
            emb   = np.array(emb, dtype="float32")
            added = push_chunks(new, emb)
            mark_fresh(ticker,"sec_content")
            log.info(f"sec refresh: {added} new for {ticker}")
        log_ingest(ticker,"sec_refresh","success",len(new),dur=round(time.time()-start,2))
    except Exception as e:
        log.error(f"sec refresh failed: {e}")
        log_ingest(ticker,"sec_refresh","failed",error=str(e))


def refresh_price_and_metrics(ticker: str):
    """refreshes price chunk AND recomputes all financial metrics"""
    log.info(f"price+metrics refresh: {ticker}")
    try:
        pd      = fetch_price_fundamentals(ticker, years=1)
        company = pd["fundamentals"].get("company_name", ticker)
        valid   = [p for p in pd.get("price_history",[]) if p["close"]==p["close"]]
        if valid:
            last  = valid[-1]
            old   = valid[-22] if len(valid)>=22 else valid[0]
            chg   = round(last["close"]-old["close"],2)
            pct_v = round(chg/old["close"]*100,2) if old["close"] else 0
            body  = (f"{company} ({ticker}) price update: close ${last['close']} on {last['date']}, "
                     f"month change ${chg} ({pct_v}%), vol {last['volume']:,}.")
            chunk = {
                "chunk_id"   : make_cid(ticker,"price_history",int(time.time())),
                "content"    : body,"source_type":"price_history","ticker":ticker,
                "company"    : company,"date":last["date"],"word_count":len(body.split()),
                "language"   : "en","quality":chunk_quality(body),
                "metadata"   : {"latest_close":last["close"],"pct_change":pct_v,"section":"price_history"},
            }
            emb = EMBEDDER.encode([body], normalize_embeddings=True)
            emb = np.array(emb, dtype="float32")
            push_chunks([chunk], emb)
            mark_fresh(ticker,"price_history")

        # refresh computed metrics chunk
        metrics = get_metrics(ticker, company, fundamentals=pd.get("fundamentals", {}), force_refresh=True)
        mc      = build_metrics_chunk(ticker, company, metrics)
        mc["chunk_id"] = make_cid(ticker,"computed_metrics",int(time.time())+1)
        emb = EMBEDDER.encode([mc["content"]], normalize_embeddings=True)
        emb = np.array(emb, dtype="float32")
        push_chunks([mc], emb)
        log.info(f"metrics refreshed for {ticker}")

    except Exception as e:
        log.error(f"price+metrics refresh failed: {e}")


_macro_scheduled = False

def start_scheduler(tickers: List[str]):
    global WATCHED, _macro_scheduled
    WATCHED = list(set(WATCHED + tickers))
    for t in tickers:
        schedule.every(4).hours.do(refresh_news, ticker=t)
        schedule.every(24).hours.do(refresh_sec, ticker=t)
        schedule.every(1).hours.do(refresh_price_and_metrics, ticker=t)

    if not _macro_scheduled:
        schedule.every(24).hours.do(refresh_macro_data_live)
        _macro_scheduled = True
        log.info("macro scheduler: refresh=24h")

    log.info("scheduler: news=4h | sec=24h | price+metrics=1h")


# ──────────────────────────────────────────────────────────────────────────────
# retrieval layer
#
# key changes vs previous version:
#   1. computed_metrics chunk is always injected first for financial queries
#   2. section exclusions prevent risk_factors from competing with financials
#   3. section preferences boost the right chunks before reranking
#   4. bm25 uses cache (not rebuilt every query)
#   5. diversity filter batches embeddings (not one encode per chunk)
# ──────────────────────────────────────────────────────────────────────────────
def _pre_rerank_diversity(candidates: list, chunks: list,
                          min_sim: float = 0.82, keep: int = 40) -> list:
    """
    Runs diversity filtering on hybrid candidates BEFORE they reach the
    reranker (change 4). Without this, the reranker scores five
    near-identical income-statement chunks and the LLM sees five copies
    of the same revenue line instead of five different evidence types.
 
    This is separate from the post-rerank diversity_filter() call:
    - pre-rerank (this): looser threshold (0.82), runs on the raw
      candidate pool to give reranker a diverse input
    - post-rerank: tighter threshold (0.85), final cleanup pass
 
    Uses the same batch-encode approach as diversity_filter() to avoid
    the slow per-chunk EMBEDDER.encode() call.
    """
    if not candidates:
        return []
 
    # batch encode all candidates at once
    texts = [c["content"] for c in candidates]
    embs  = EMBEDDER.encode(texts, normalize_embeddings=True)
 
    out      = []
    kept_emb = []
    for j, r in enumerate(candidates):
        emb = embs[j]
        if not any(util.cos_sim(emb, prev).item() > min_sim for prev in kept_emb):
            out.append(r)
            kept_emb.append(emb)
        if len(out) >= keep:
            break
    return out

def hybrid_search(query: str, ticker: str, chunks: list,query_analysis: dict,k: int = 40, known_tickers=None) -> list:

    weights   = query_analysis["weights"]
    dom       = query_analysis["dominant_class"]
    excluded  = query_analysis["excluded_sections"]
    preferred = query_analysis["preferred_sections"]
    ncm       = query_analysis["needs_computed_metrics"]
    trend_q   = query_analysis["is_trend"]
    txn_q     = query_analysis["needs_transcript"]

    print(f"  retrieval mode  : {dom}")
    print(f"  preferred secs  : {preferred}")
    print(f"  excluded secs   : {excluded}")
    print(f"  chunk pool      : {len(chunks)}")
    print(f"  flags           : ncm={ncm} trend={trend_q} transcript={txn_q}")

    # BM25_K raised to 250 for financial queries (was 150) — previous size
    # caused the "1 candidate" problem for trend queries because the pool
    # was too small to find multiple quarterly periods. trend_analysis and
    # quarterly_comparison use 350 to ensure all 8 quarters are candidates.
    params = {
        "financial_metric"    : (250, 0.70, min(120, len(chunks))),
        "trend_analysis"      : (350, 0.60, min(150, len(chunks))),
        "quarterly_comparison": (350, 0.55, min(150, len(chunks))),
        "capital_allocation"  : (250, 0.55, min(150, len(chunks))),
        "growth"              : (250, 0.65, min(120, len(chunks))),
        "risk"                : (300, 0.40, min(150, len(chunks))),
        "management_commentary": (200, 0.60, min(120, len(chunks))),
        "news_sentiment"      : (200, 0.35, min(50,  len(chunks))),
        "segment"             : (250, 0.50, min(150, len(chunks))),
        "balance_sheet"       : (250, 0.55, min(150, len(chunks))),
        "price"               : (50,  0.80, min(30,  len(chunks))),
        "comparison"          : (250, 0.55, min(100, len(chunks))),
    }
    bm25_k, alpha, pv_k = params.get(dom, (200, 0.65, min(100, len(chunks))))

    mentioned = [w for w in re.findall(r"[A-Z]{1,5}", query) if known_tickers and w in known_tickers]
    if mentioned:
        alpha = max(0.35, alpha - 0.1)

    # ── step 1: chunk_id → index map (never use content[:120]) ───────────────
    cid_to_idx = {c.get("chunk_id",""): i for i, c in enumerate(chunks)}

    # ── step 2: vector search ────────────────────────────────────────────────
    pv    = vec_search(query, ticker, k=pv_k)
    pv_sc = {}
    for r in pv:
        idx = cid_to_idx.get(r.get("chunk_id",""), -1)
        if idx >= 0:
            pv_sc[idx] = r["hybrid_score"]

    if pv_sc:
        lo, hi  = min(pv_sc.values()), max(pv_sc.values())
        norm_pv = {i: (s-lo)/(hi-lo+1e-9) for i,s in pv_sc.items()}
    else:
        norm_pv = {}

    print(f"  pgvector hits   : {len(pv)}")

    # ── step 3: BM25 with synonym-expanded query ──────────────────────────────
    # previously BM25 tokenized the raw query. "ROE" never matched "return on
    # equity" in the filings. expand_query_for_bm25() appends synonym tokens
    # so "dividend yield" also finds "cash dividend", "dividend payout", etc.
    bm25     = build_bm25(ticker, chunks)
    qtok_exp = expand_query_for_bm25(query)  # synonym-expanded tokens
    bs       = bm25.get_scores(qtok_exp)
    top_bi   = np.argsort(bs)[::-1][:bm25_k * 2]  # oversample then filter

    eligible = []
    for idx in top_bi:
        idx = int(idx)
        if not (0 <= idx < len(chunks)):
            continue
        sec = chunks[idx].get("metadata",{}).get("section","")
        if sec in excluded:
            continue  # hard remove before scores are computed
        eligible.append(idx)
        if len(eligible) >= bm25_k:
            break

    maxbs = max((bs[i] for i in eligible), default=1e-9) + 1e-9
    nb    = {i: float(bs[i])/maxbs for i in eligible}

    print(f"  bm25 candidates : {len(nb)} (expanded query: {len(qtok_exp)} tokens)")

    # ── step 4: hybrid combination ───────────────────────────────────────────
    all_ids = set(norm_pv.keys()) | set(nb.keys())
    scores  = {i: alpha*norm_pv.get(i,0) + (1-alpha)*nb.get(i,0) for i in all_ids}

    # quality boost
    for i in scores:
        if 0 <= i < len(chunks):
            scores[i] *= 0.85 + chunks[i].get("quality",0.5) * 0.15

    # ── step 5: force computed_metrics into pool before scoring (fix 4) ──────
    # previously computed_metrics was only injected AFTER reranking (in
    # ensure_metrics_chunk). if the embedding similarity was low it never
    # entered the candidate pool at all — so ensure_metrics_chunk never ran.
    # now for any ncm=True query we explicitly add the computed_metrics chunk
    # to scores with a baseline score before the boost pass below.
    if ncm:
        for i, c in enumerate(chunks):
            if c.get("source_type","") == "computed_metrics" and i not in scores:
                scores[i] = 0.5  # guaranteed entry; boost below raises it further

    # ── step 6: controlled relevance adjustments ─────────────────────────────
    to_remove = []

    for i in list(scores.keys()):
        if not (0 <= i < len(chunks)):
            to_remove.append(i)
            continue

        c = chunks[i]
        stype = c.get("source_type", "")
        sec = c.get("metadata", {}).get("section", "")
        ct = c.get("content", "").lower()

        # Hard exclusion
        if sec in excluded:
            to_remove.append(i)
            continue

        # ------------------------------------------------------------
        # IMPORTANT:
        # Do NOT multiply the score repeatedly for every matching rule.
        # We use small additive adjustments instead.
        # ------------------------------------------------------------

        adjustment = 0.0

        # Preferred section: small bonus only
        if sec in preferred:
            adjustment += 0.08

        # Computed metrics are useful when explicitly requested
        if stype == "computed_metrics":
            if ncm:
                adjustment += 0.15

        # Query-class specific adjustments
        if dom in ("financial_metric", "growth", "balance_sheet"):
            if stype in ("structured_metrics", "xbrl_data", "computed_metrics"):
                adjustment += 0.10

        elif dom in ("trend_analysis", "quarterly_comparison"):
            if stype in ("structured_metrics", "xbrl_data", "computed_metrics"):
                adjustment += 0.10

            if any(
                term in ct
                for term in (
                    "q1", "q2", "q3", "q4",
                    "quarter", "quarterly"
                )
            ):
                adjustment += 0.05

        elif dom == "capital_allocation":
            if any(
                term in ct
                for term in (
                    "dividend",
                    "buyback",
                    "repurchase",
                    "capital allocation",
                    "payout"
                )
            ):
                adjustment += 0.08

        elif dom == "risk":
            if sec == "risk_factors":
                adjustment += 0.12

            if sec == "legal":
                adjustment += 0.08

        elif dom == "management_commentary":
            if stype == "earnings_transcript":
                adjustment += 0.12

            if sec in (
                "guidance",
                "prepared_remarks",
                "earnings_discussion",
                "qa_session"
            ):
                adjustment += 0.10

        elif dom == "segment":
            if any(
                term in ct
                for term in (
                    "reportable segment",
                    "operating segment"
                )
            ):
                adjustment += 0.08

        elif dom == "news_sentiment":
            if "news" in stype:
                adjustment += 0.10

            scores[i] = age_boost(c.get("date", ""), scores[i])

        # ------------------------------------------------------------
        # Apply ONE bounded adjustment.
        # This prevents score explosion.
        # ------------------------------------------------------------
        scores[i] = scores[i] + adjustment

    for i in to_remove:
        scores.pop(i, None)

    # ── step 7: sort → build candidate dicts ─────────────────────────────────
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    candidates = []
    for i, s in ranked[:k * 2]:
        if not (0 <= i < len(chunks)):
            continue
        c = chunks[i]
        candidates.append({
            "chunk_index" : i,
            "chunk_id"    : c.get("chunk_id",""),
            "content"     : c["content"],
            "source_type" : c.get("source_type",""),
            "ticker"      : c.get("ticker",""),
            "company"     : c.get("company",""),
            "date"        : c.get("date",""),
            "hybrid_score": round(float(s),4),
            "quality"     : c.get("quality",0.5),
            "metadata"    : c.get("metadata",{}),
        })

    # ── step 8: pre-rerank diversity ──────────────────────────────────────────
    # threshold raised from 0.82 → 0.88 (fix 8).
    # 0.82 was too aggressive — quarterly chunks that legitimately start
    # similarly ("Salesforce Q1 quarterly income...", "Salesforce Q2 quarterly
    # income...") were being collapsed into one, giving the reranker only
    # 1-2 candidates for trend queries.
    #
    # for trend_analysis and quarterly_comparison, diversity is DISABLED
    # entirely — we WANT multiple quarterly period chunks and similarity
    # filtering is the exact wrong thing to do for those queries.
    if trend_q:
        # no diversity filtering — keep all quarterly period chunks
        pass
    elif len(candidates) > k:
        # threshold 0.88 (was 0.82), also enforce minimum floor of k//2
        # so we never collapse below half the requested candidate count
        candidates = _pre_rerank_diversity(candidates, chunks, min_sim=0.88, keep=k)

    print(f"  hybrid ranked   : {len(candidates)} candidates (post-diversity)")
    return candidates

_RANGE_NUMBER_PATTERN = re.compile(
    r"(~|approximately|around|roughly)?\s*\d+(\.\d+)?\s*[-–]\s*\d+(\.\d+)?\s*%"
    r"|~\s*\d+(\.\d+)?\s*%"
)

def rerank(query: str, ranked: list, query_analysis: dict, top_n: int = 10,
           batch_size: int = 16, known_tickers=None) -> list:
    if not ranked:
        return []

    # Same classification approach hybrid_search() already uses — classify()
    # now has the semantic fallback built in, so both functions automatically
    # benefit from it without needing a second, separate intent classifier.
    weights   = query_analysis["weights"]
    dom       = query_analysis["dominant_class"]
    excluded  = query_analysis["excluded_sections"]
    preferred = query_analysis["preferred_sections"]

    tickers = [w for w in re.findall(r"[A-Z]{1,5}", query) if known_tickers and w in known_tickers]


    pairs = [(query, r["content"]) for r in ranked]

    rsc    = []
    for i in range(0, len(pairs), batch_size):
        rsc.extend(RERANKER.predict(pairs[i:i+batch_size]))

    # bge-reranker-large's CrossEncoder.predict() already applies sigmoid
    # internally (num_labels=1 config) — rsc is already a bounded [0,1]
    # relevance probability. Do NOT transform it again; that's what was
    # crushing all scores into a narrow band near 0.5.
    norm = list(rsc)

    boosted = []
    for r, ns in zip(ranked, norm):
        sec   = r.get("metadata",{}).get("section","general")
        ftype = r.get("metadata",{}).get("form_type","")
        stype = r.get("source_type","")
        ct    = r["content"].lower()
        qual  = r.get("quality",0.5)
        score = ns

        # hard exclusion after reranking too — original 0.01, not the
        # flattened 0.2 from get_weighted_sections
        if sec in excluded:
            score *= 0.01

        # computed_metrics always wins for financial queries
        if stype == "computed_metrics":
            score *= 1.0 + weights.get("financial_metric",0) * 1.5
            score *= 1.0 + weights.get("growth",0) * 1.2
            score *= 1.0 + weights.get("balance_sheet",0) * 1.2

        # preferred section boost — original 1.8, not the flattened 1.5
        if sec in preferred:
            score *= 1.8

        # class-specific section boosts
        fw = weights.get("financial_metric",0)
        if fw > 0:
            if stype in ("structured_metrics","xbrl_data"):
                score *= 1.0 + fw*0.6
            if sec in ("income_statement","key_ratios","computed_metrics","financials","xbrl_summary"):
                score *= 1.0 + fw*0.5
            if any(k in ct for k in ["revenue","profit","margin","eps","earnings per share","gross profit"]):
                score *= 1.0 + fw*0.2

        gw = weights.get("growth",0)
        if gw > 0:
            if sec in ("computed_metrics","income_statement","revenue_growth","guidance","earnings_discussion"):
                score *= 1.0 + gw*0.5
            if any(k in ct for k in ["year over year","yoy","cagr","growth"]):
                score *= 1.0 + gw*0.3

        rw = weights.get("risk",0)
        if rw > 0:
            if sec == "risk_factors":         score *= 1.0 + rw*0.4
            elif sec in ("md&a","legal"):     score *= 1.0 + rw*0.3
            elif sec == "regulatory_news":    score *= 1.0 + rw*0.2

        mw = weights.get("management_commentary",0)
        if mw > 0:
            if sec in ("md&a","guidance","prepared_remarks","qa_session","earnings_discussion","forward_looking"):
                score *= 1.0 + mw*0.4
            if stype == "investor_presentation":
                score *= 1.0 + mw*0.25

        nw = weights.get("news_sentiment",0)
        news_stypes = ("news","yahoo_news","seeking_alpha_news","marketwatch_news",
                       "mktpulse_news","cnbc_finance_news","cnbc_earnings_news")
        if nw > 0 and stype in news_stypes:
            score *= 1.0 + nw*0.35
            score  = age_boost(r.get("date",""), score)

        pw = weights.get("price",0)
        if pw > 0 and stype == "price_history":
            score *= 1.0 + pw*0.5

        # numeric vs narrative handling — based on dom (dominant_class of
        # classify()'s output), same source of truth as everywhere else
        # in the file, instead of a separately-computed intent variable
        numeric_classes = ("financial_metric", "growth", "balance_sheet")
        if dom in numeric_classes:
            if stype == "investor_presentation":
                score *= 0.15  # penalize decks on numeric queries
            if stype in ("sec_content", "structured_metrics", "xbrl_data", "computed_metrics"):
                score *= 1.1   # prefer filed/computed sources
            if _RANGE_NUMBER_PATTERN.search(ct):
                score *= 0.6   # narrative ranges, not a filed figure
        else:
            if stype == "earnings_transcript":
                score *= 1.25
            if stype == "investor_presentation":
                score *= 1.15

        if ftype == "10-K":   score *= 1.20
        elif ftype == "10-Q": score *= 1.15
        if tickers and r.get("ticker") in tickers:
            score *= 1.2

        multiplier = 0.7 + 0.6 * qual   # maps [0,1] → [0.7,1.3]
        score *= multiplier

        # clamp the boosted score before blending so a long multiplier chain
        # can't let final_score be dominated by stacked boosts rather than
        # actual rerank relevance
        score = min(score, 2.0)
        boosted.append((r, 0.75*score + 0.25*r["hybrid_score"], ns))

    MIN_ABS_RELEVANCE = 0.5
    passing = [b for b in boosted if b[2] >= MIN_ABS_RELEVANCE]
    if not passing:
        # nothing cleared the bar — don't starve the LLM down to a single
        # chunk; keep the best few so it has enough to say "data not found"
        # accurately, or partially answer, instead of guessing from one chunk
        boosted.sort(key=lambda x: x[2], reverse=True)
        passing = boosted[:3]
    boosted = passing

    boosted.sort(key=lambda x: x[1], reverse=True)
    print("\n--- RERANK DEBUG ---")
    for r, final, ns in boosted:
        print(
            f"rerank={ns:.4f} | final={final:.4f} | "
            f"section={r.get('metadata', {}).get('section', 'general')} | "
            f"type={r.get('source_type', '')} | "
            f"hybrid={r.get('hybrid_score', 0):.4f}"
        )
    print("--------------------")
    out = []
    for chunk, final, ns in boosted[:top_n]:
        out.append({
            "rank"        : len(out)+1,
            "chunk_index" : chunk.get("chunk_index",-1),
            "final_score" : round(float(final),4),
            "rerank_score": round(float(ns),4),
            "hybrid_score": round(float(chunk["hybrid_score"]),4),
            "quality"     : round(float(chunk.get("quality",0.5)),3),
            "chunk_id"    : chunk.get("chunk_id",""),
            "content"     : chunk["content"],
            "source_type" : chunk.get("source_type",""),
            "ticker"      : chunk.get("ticker",""),
            "company"     : chunk.get("company",""),
            "date"        : chunk.get("date",""),
            "metadata"    : chunk.get("metadata",{}),
        })
    return out


def diversity_filter(results: list, chunks: list, min_sim: float = 0.85) -> list:
    if not results:
        return []
    valid = [(r.get("chunk_index",-1), r) for r in results]
    valid = [(i, r) for i, r in valid if 0 <= i < len(chunks)]
    if not valid:
        return []
    texts    = [chunks[i]["content"] for i, _ in valid]
    embs     = EMBEDDER.encode(texts, normalize_embeddings=True)
    out      = []
    kept_emb = []
    for j, (_, r) in enumerate(valid):
        emb = embs[j]
        if not any(util.cos_sim(emb, prev).item() > min_sim for prev in kept_emb):
            out.append(r)
            kept_emb.append(emb)
    return out


def filter_boilerplate(results: list, chunks: list) -> list:
    bad = ["commission file number","united states securities and exchange commission",
           "sec.gov/cgi-bin","pursuant to the requirements of the securities exchange act"]
    out = []
    for r in results:
        idx = r.get("chunk_index",-1)
        if idx < 0 or idx >= len(chunks): continue
        txt   = chunks[idx]["content"].lower()
        words = txt.split()
        if len(words) < 30: continue
        if any(b in txt for b in bad): continue
        nums = sum(1 for w in words if re.match(r"^\$?[\d,\.%]+$",w))
        if nums/len(words) > 0.80: continue
        if r.get("quality",1.0) < 0.2: continue
        out.append(r)
    return out


def _normalize_injected(raw: list) -> list:
    """
    vec_search() returns a different shape than rerank() output — no
    'rank' or 'final_score' keys, which run_query()'s print loop and
    build_context() both expect on every result. This reshapes injected
    chunks (from vec_search-based injectors like macro/price) into the
    same schema rerank() produces, same pattern ensure_metrics_chunk()
    already uses for its own injected chunk.
    """
    out = []
    for r in raw:
        out.append({
            "rank"        : 0,
            "chunk_index" : r.get("chunk_index", -1),
            "final_score" : 1.5,   # injected, ranks above normal retrieval
            "rerank_score": 1.0,
            "hybrid_score": r.get("hybrid_score", 1.0),
            "quality"     : r.get("quality", 0.9),
            "chunk_id"    : r.get("chunk_id", ""),
            "content"     : r.get("content", ""),
            "source_type" : r.get("source_type", ""),
            "ticker"      : r.get("ticker", ""),
            "company"     : r.get("company", ""),
            "date"        : r.get("date", ""),
            "metadata"    : r.get("metadata", {}),
        })
    return out


def maybe_inject_macro(results: list,sector_ctx: dict,query: str,query_analysis: dict) -> list:    
    """
    Injects macro context in two cases:
    1. Sector is structurally rate-sensitive (bank/reit/utility/insurance) —
       inject even if the user didn't ask.
    2. The query is explicitly classified as macro_context — inject
       regardless of sector, since macro chunks live under ticker="MACRO"
       and are never in-scope for a normal ticker-scoped hybrid_search()
       call otherwise, no matter how the query classifies.
    """
    weights = query_analysis.get("weights", {})
    is_macro_query = weights.get("macro_context", 0) > 0.4

    if sector_ctx.get("profile_key") in ("bank", "reit", "utility", "insurance") or is_macro_query:
        macro = vec_search(query or "interest rates inflation gdp", "MACRO", k=4)
        return _normalize_injected(macro) + results
    return results

def maybe_inject_price(results: list,ticker: str,query: str,query_analysis: dict) -> list:    
    """
    Forces price_history chunks into the pool for stock-performance-style
    queries — same gap as macro: classification correctly identifies intent,
    but nothing previously guaranteed the right chunk type reached the LLM.
    """
    weights = query_analysis.get("weights", {})
    ql = query.lower()

    if weights.get("price", 0) > 0.3 or "stock performance" in ql or "share price" in ql:
        price = vec_search(query, ticker, k=2)
        price = [p for p in price if p.get("source_type") == "price_history"]
        return _normalize_injected(price) + results
    return results

def ensure_metrics_chunk(results: list, chunks: list, ticker: str, company: str) -> list:
    """
    If the query needs financial metrics but no computed_metrics chunk was retrieved,
    force-inject it. This prevents the llm from ever seeing a margin query without
    the pre-computed answer.
    """
    has_metrics = any(r.get("source_type","") in ("computed_metrics","structured_metrics","xbrl_data")
                      for r in results)
    if has_metrics:
        return results

    # find the computed_metrics chunk in the full chunk pool
    for i, c in enumerate(chunks):
        if c.get("source_type","") == "computed_metrics":
            injected = {
                "rank"        : 0,
                "chunk_index" : i,
                "final_score" : 2.0,  # forced to top
                "rerank_score": 1.0,
                "hybrid_score": 1.0,
                "quality"     : 1.0,
                "chunk_id"    : c.get("chunk_id",""),
                "content"     : c["content"],
                "source_type" : "computed_metrics",
                "ticker"      : ticker,
                "company"     : company,
                "date"        : c.get("date",""),
                "metadata"    : c.get("metadata",{}),
            }
            log.info("force-injected computed_metrics chunk")
            return [injected] + results

    # if not in chunk pool, build it fresh
    metrics = get_metrics(ticker, company)
    mc      = build_metrics_chunk(ticker, company, metrics)
    injected = {
        "rank":0,"chunk_index":-1,"final_score":2.0,"rerank_score":1.0,
        "hybrid_score":1.0,"quality":1.0,"chunk_id":"injected_metrics",
        "content":mc["content"],"source_type":"computed_metrics",
        "ticker":ticker,"company":company,"date":str(datetime.now().date()),
        "metadata":mc.get("metadata",{}),
    }
    log.info("force-injected fresh computed_metrics chunk (not in db)")
    return [injected] + results


# ──────────────────────────────────────────────────────────────────────────────
# context builder
# ──────────────────────────────────────────────────────────────────────────────

_SRC_PRIORITY = {
    "computed_metrics"   : 1,
    "xbrl_data"          : 2,
    "structured_metrics" : 3,
    "fundamentals"       : 4,
    "earnings_transcript": 5,
    "investor_presentation": 6,
    "sec_content"        : 7,
    "sec_filing"         : 8,
    "news"               : 9,
    "price_history"      : 10,
}

_SRC_LABELS = {
    "computed_metrics"   : "COMPUTED FINANCIAL METRICS (python-calculated, use these for all numbers)",
    "structured_metrics" : "FINANCIAL STATEMENTS",
    "xbrl_data"          : "XBRL DATA",
    "fundamentals"       : "FUNDAMENTALS",
    "earnings_transcript": "EARNINGS CALL",
    "investor_presentation": "INVESTOR PRESENTATION",
    "sec_content"        : "SEC FILING",
    "sec_filing"         : "SEC METADATA",
    "price_history"      : "PRICE DATA",
}


def build_context(results: list, query: str) -> str:
    if not results:
        return f"question: {query}\n\nno evidence retrieved."

    sorted_r = sorted(results, key=lambda x: (_SRC_PRIORITY.get(x.get("source_type",""),99), -x.get("final_score",0)))

    grouped = defaultdict(list)
    seen    = set()

    for i, r in enumerate(sorted_r):
        content = r.get("content","").strip()
        if len(content.split()) > 500:
            content = " ".join(content.split()[:500]) + "..."
        fp = content[:200]
        if fp in seen: continue
        seen.add(fp)

        stype = r.get("source_type","")
        meta  = r.get("metadata",{})
        date  = r.get("date","")
        sec   = meta.get("section","")

        if stype == "computed_metrics":
            cite = f"[SOURCE {i+1} | COMPUTED METRICS (python-calculated) | {date}]"
        elif stype == "structured_metrics":
            cite = f"[SOURCE {i+1} | {meta.get('statement_type','').replace('_',' ')} | {meta.get('period','')} {meta.get('frequency','')}]"
        elif stype == "xbrl_data":
            cite = f"[SOURCE {i+1} | xbrl | {date}]"
        elif stype == "fundamentals":
            cite = f"[SOURCE {i+1} | fundamentals | {date[:10]}]"
        elif stype in ("sec_content","earnings_transcript","investor_presentation"):
            cite = f"[SOURCE {i+1} | SEC {meta.get('form_type','')} {sec} | {date}]"
        elif "news" in stype:
            cite = f"[SOURCE {i+1} | {stype.replace('_news','').replace('_',' ')} | {sec} | {date[:10]}]"
        elif stype == "price_history":
            cite = f"[SOURCE {i+1} | price {meta.get('period','')} | {date}]"
        else:
            cite = f"[SOURCE {i+1} | {stype} | {date[:10]}]"

        grouped[stype].append(f"{cite}\n{content}")

    ctx   = f"question: {query}\n\nsource documents:\n" + "="*45
    order = ["computed_metrics","xbrl_data","structured_metrics","fundamentals",
             "earnings_transcript","investor_presentation","sec_content","sec_filing"]
    for st in grouped:
        if "news" in st and st not in order:
            order.append(st)
    order.append("price_history")

    for stype in order:
        if stype not in grouped: continue
        label = _SRC_LABELS.get(stype, stype.upper())
        if "news" in stype and stype not in _SRC_LABELS:
            label = stype.replace("_news","").replace("_"," ").upper() + " NEWS"
        ctx += f"\n\n{label}\n" + "-"*30 + "\n"
        ctx += "\n\n".join(grouped[stype])

    return ctx


# ──────────────────────────────────────────────────────────────────────────────
# qwen via ollama
# ──────────────────────────────────────────────────────────────────────────────
def detect_available_data(results: list) -> dict:
    """
    Scans retrieved chunks to know what's actually available before
    building the prompt — prevents asking the LLM for P/E, EV/EBITDA,
    price performance, etc. when none of that data was retrieved,
    which is exactly what caused it to fall back on memorized training
    data instead of saying "not available."
    """
    stypes = set(r.get("source_type", "") for r in results)
    secs   = set(r.get("metadata", {}).get("section", "") for r in results)
    return {
        "has_computed_metrics" : "computed_metrics" in stypes,
        "has_structured_metrics": "structured_metrics" in stypes or "xbrl_data" in stypes,
        "has_price"             : "price_history" in stypes,
        "has_macro"             : "macro_context" in stypes,
        "has_news"              : any("news" in s for s in stypes),
        "has_transcript"        : "earnings_transcript" in stypes,
        "has_risk_content"      : "risk_factors" in secs or "legal" in secs,
        "has_segment"           : "segment" in secs,
        "has_fundamentals"      : "fundamentals" in stypes,
    }


_SECTION_LIBRARY = {
    "executive_summary": (
        "### Executive Summary\n"
        "Overall investment view, confidence, one-sentence thesis, top catalysts, key risks."
    ),
    "business_health": (
        "### Business Health\nOverall health, financial quality, profitability, cash generation, liquidity, leverage."
    ),
    "financial_performance": (
        "### Financial Performance\n"
        "Report ONLY the specific figures present in the COMPUTED METRICS or FINANCIAL STATEMENTS "
        "sources above — revenue, margins, EPS, cash flow, ROE, ROA, debt, liquidity. "
        "If a figure isn't in the sources, write 'not available in retrieved data' instead of estimating it."
    ),
    "trend_intelligence": (
        "### Trend Intelligence\nRevenue/margin/EPS/cash flow trend classifications, confidence, reasoning — "
        "use ONLY the trend labels and reasons given in COMPUTED METRICS."
    ),
    "financial_interpretation": (
        "### Financial Interpretation\nTranslate the reported metrics into business implications — "
        "operating leverage, profitability quality, capital efficiency."
    ),
    "investment_thesis": (
        "### Investment Thesis\nBull case, bear case, base case, key catalyst, key risk — "
        "grounded in the evidence and thesis fields from COMPUTED METRICS if present."
    ),
    "evidence_summary": (
        "### Evidence Summary\nSummarize the strongest supporting evidence from COMPUTED METRICS' evidence field."
    ),
    "contradictions": (
        "### Contradictions\nIdentify conflicting signals using the MIXED SIGNALS data if present. "
        "If none were flagged, say so plainly rather than inventing one."
    ),
    "capital_allocation": (
        "### Capital Allocation\nUse ONLY the capital allocation recommendation and scores from COMPUTED METRICS. "
        "Do not guess management's priorities without that data."
    ),
    "risk_assessment": (
        "### Risk Assessment\nFinancial, liquidity, growth, execution risk — cite the RISK PROFILE fields if present, "
        "and pull qualitative risk language from any risk_factors/legal sources."
    ),
    "valuation": (
        "### Valuation\nReport ONLY valuation figures (P/E, PEG, EV/EBITDA, FCF yield) that appear in the "
        "VALUATION section of COMPUTED METRICS. Never invent a price target or a multiple not in the sources."
    ),
    "peer_comparison": (
        "### Peer Comparison\nIf peer data exists in sources, compare. Otherwise state it was not retrieved — "
        "do not fabricate competitor figures from memory."
    ),
    "earnings_call": (
        "### Earnings Call Intelligence\nSummarize management tone, guidance, priorities from the EARNINGS CALL source. "
        "Paraphrase closely — never invent a management quote."
    ),
    "macro_context": (
        "### Macroeconomic Context\nUse the MACRO INDICATOR sources to discuss how rates/inflation/GDP relate to "
        "this company's business, sector, and valuation. Cite the specific indicator values and trends given — "
        "never state a GDP, CPI, or rate figure not present in the MACRO sources."
    ),
    "price_performance": (
        "### Price Performance\nUse the PRICE DATA sources for any statement about stock/share price movement. "
        "If price data isn't present, say price performance data was not retrieved rather than estimating it."
    ),
    "scenario_analysis": (
        "### Scenario Analysis\nBull/base/bear scenario framing, and what would flip the current view — "
        "use the SCENARIO field from COMPUTED METRICS if present."
    ),
    "final_opinion": (
        "### Final Analyst Opinion\nOverall recommendation, confidence, suitable investor profile, "
        "primary strengths/risks, key metrics to monitor. Do not state a numeric recommendation with more "
        "confidence than the underlying evidence supports."
    ),
    "sources": (
        "### Sources\nList the evidence types actually used (SEC filings, computed metrics, earnings call, "
        "news, macro data, market data) — only list what was actually present in the source documents above."
    ),
}


def determine_required_sections(dom: str, available: dict) -> list:
    """
    Picks which sections to request based on query class AND what data
    is actually available — this is the core fix: the prompt never asks
    for a section the retrieved context can't support.
    """
    always = ["executive_summary", "final_opinion", "sources"]

    if dom in ("financial_metric", "growth", "balance_sheet"):
        core = ["financial_performance", "trend_intelligence", "financial_interpretation"]
    elif dom == "capital_allocation":
        core = ["capital_allocation", "evidence_summary", "scenario_analysis"]
    elif dom == "risk":
        core = ["risk_assessment"]
    elif dom == "investment_thesis":
        core = ["investment_thesis", "evidence_summary", "contradictions", "scenario_analysis"]
    elif dom == "management_commentary":
        core = ["earnings_call"]
    elif dom == "macro_context":
        core = ["macro_context", "financial_interpretation"]
    elif dom == "price":
        core = ["price_performance"]
    elif dom == "comparison":
        core = ["peer_comparison", "financial_performance"]
    elif dom == "segment":
        core = ["financial_performance"]
    else:
        # broad/company-overview style questions get the full report,
        # but still gated by availability below
        core = ["business_health", "financial_performance", "trend_intelligence",
                "investment_thesis", "risk_assessment", "valuation"]

    sections = always[:1] + core + always[1:]

    # gate by actual data availability — drop sections with nothing behind them
    if not available["has_computed_metrics"] and not available["has_structured_metrics"]:
        sections = [s for s in sections if s not in
                    ("financial_performance", "trend_intelligence", "valuation", "capital_allocation")]
    if not available["has_price"]:
        sections = [s for s in sections if s != "price_performance"]
    if not available["has_macro"]:
        sections = [s for s in sections if s != "macro_context"]
    if not available["has_transcript"]:
        sections = [s for s in sections if s != "earnings_call"]

    # dedupe, preserve order
    seen, out = set(), []
    for s in sections:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out

def build_sector_framing(sector_ctx: dict) -> str:
    """
    Injects the sector-specific context you already built in SECTOR_PROFILES
    directly into the prompt — so the LLM knows a bank's high debt/equity
    isn't a red flag, or that a REIT's leverage is structural, instead of
    applying generic-company judgment to every sector uniformly.
    """
    if not sector_ctx:
        return ""
    lines = [f"SECTOR CONTEXT: {sector_ctx.get('sector','Unknown')} / {sector_ctx.get('industry','Unknown')} "
             f"(profile: {sector_ctx.get('profile_key','standard')})"]
    if sector_ctx.get("not_applicable"):
        lines.append(f"NOT MEANINGFUL FOR THIS SECTOR: {', '.join(sector_ctx['not_applicable'])} — "
                      f"do not penalize the company for these or treat them as red flags.")
    if sector_ctx.get("leverage_note"):
        lines.append(f"LEVERAGE NOTE: {sector_ctx['leverage_note']}")
    if sector_ctx.get("cyclicality_note"):
        lines.append(f"CYCLICALITY NOTE: {sector_ctx['cyclicality_note']}")
    return "\n".join(lines)


_SYSTEM_HEADER = '''You are an institutional-grade AI Equity Research Analyst. Synthesize
ONLY verified evidence from the source documents provided. Never fabricate
numbers, metrics, management quotes, or conclusions. If a figure the report
format would normally include is not present in the sources, explicitly
state it was not available in retrieved data — do not estimate, recall
from general knowledge, or infer a plausible number. A missing figure
stated honestly is always better than an invented one.

Write like an institutional analyst: objective, concise, no marketing
language, no filler. Separate facts from interpretation. Cite every
factual claim with [SOURCE N].'''


def build_dynamic_system_prompt(dom: str, sector_ctx: dict, available: dict) -> str:
    sections = determine_required_sections(dom, available)
    section_text = "\n\n".join(_SECTION_LIBRARY[s] for s in sections)
    sector_text  = build_sector_framing(sector_ctx)

    parts = [_SYSTEM_HEADER]
    if sector_text:
        parts.append(sector_text)
    parts.append("Produce a report using exactly these sections, in this order:\n\n" + section_text)
    return "\n\n---\n\n".join(parts)



def _prompt(query: str,context: str,company: str,ticker: str,query_analysis: dict) -> str:
    dom = query_analysis["dominant_class"]
    ncm = query_analysis["needs_computed_metrics"]
    p       = f"company: {company} ({ticker})\nquestion: {query}\n\n{context}\n\n---\n"

    if dom == "capital_allocation":
        p += ("Answer using the capital allocation recommendation, evidence, bull/bear case, "
              "and scenario fields in the COMPUTED METRICS section if present, plus any "
              "dividend/buyback/cash flow figures in the other sources. "
              "Don't list unrelated metrics like gross margin or EPS unless they directly "
              "support the capital allocation answer. State the recommendation, the "
              "supporting evidence, and what would change it. [SOURCE N] after each claim.")
    elif ncm or dom in ("financial_metric","growth","balance_sheet"):
        p += ("The COMPUTED METRICS section has pre-calculated values. "
              "Report those numbers directly — don't recalculate. "
              "For margins, use the values labeled 'gross margin', 'operating margin', etc. "
              "For growth, use the values labeled 'YoY' or 'CAGR'. "
              "Only report the specific metric the question asked about — don't list "
              "every available metric. "
              "[SOURCE N] after each number.")
    elif dom == "segment":
        p += ("Pull segment revenue and operating income. "
              "If shared costs are disclosed, include them. Otherwise say so. "
              "Cite with [SOURCE N].")
    elif dom == "risk":
        p += ("Summarize key risks in prose. For each, explain the exposure and mitigation. "
              "Cite with [SOURCE N].")
    elif dom == "management_commentary":
        p += ("Summarize what management said. Paraphrase closely. "
              "Note date and context. Cite with [SOURCE N].")
    elif dom == "news_sentiment":
        p += ("Summarize recent news. Give dates. "
              "Factual — what happened, when, what was said. Cite with [SOURCE N].")
    elif dom == "price":
        p += "Describe price performance using actual figures. Cite with [SOURCE N]."
    elif dom == "comparison":
        p += ("Compare using the data. Same metrics, same periods where possible. "
              "Note data gaps. Cite with [TICKER-SOURCE N].")
    else:
        p += "Answer using only the sources. Cite every claim with [SOURCE N]."
    return p


def compute_llm_perf_metrics(resp_json: dict) -> dict:
    """
    Extracts Ollama's native timing fields and derives prefill/decode
    throughput. All fields below come straight from Ollama's response —
    nothing here is estimated or interpolated.

    ollama reports durations in nanoseconds:
      load_duration         - model load time (0 if already warm)
      prompt_eval_count     - # tokens in the prompt (prefill)
      prompt_eval_duration  - time spent on prefill
      eval_count            - # tokens generated (decode)
      eval_duration         - time spent on decode
      total_duration        - load + prefill + decode
    """
    pe_count  = resp_json.get("prompt_eval_count", 0)
    pe_dur    = resp_json.get("prompt_eval_duration", 0)  # ns
    ev_count  = resp_json.get("eval_count", 0)
    ev_dur    = resp_json.get("eval_duration", 0)          # ns
    load_dur  = resp_json.get("load_duration", 0)
    total_dur = resp_json.get("total_duration", 0)

    prefill_tok_s = round(pe_count / (pe_dur / 1e9), 2) if pe_dur else None
    decode_tok_s  = round(ev_count / (ev_dur / 1e9), 2) if ev_dur else None

    return {
        "prompt_tokens"     : pe_count,
        "completion_tokens" : ev_count,
        "load_ms"           : round(load_dur / 1e6, 1),
        "prefill_ms"        : round(pe_dur / 1e6, 1),
        "decode_ms"         : round(ev_dur / 1e6, 1),
        "total_ms"          : round(total_dur / 1e6, 1),
        "prefill_tok_per_s" : prefill_tok_s,
        "decode_tok_per_s"  : decode_tok_s,
    }


def call_llm(query: str, results: list, company: str, ticker: str, query_analysis: dict) -> dict:

    """
        print("\n" + "=" * 100)
        print("EXACT CHUNKS SENT TO LLM")
        print("=" * 100)

        for i, r in enumerate(results, 1):
            print("\n" + "-" * 100)
            print(f"SOURCE {i}")
            print(f"final_score : {r.get('final_score')}")
            print(f"rerank_score: {r.get('rerank_score')}")
            print(f"hybrid_score: {r.get('hybrid_score')}")
            print(f"section     : {r.get('metadata', {}).get('section')}")
            print(f"source_type : {r.get('source_type')}")
            print(f"form_type   : {r.get('metadata', {}).get('form_type')}")
            print(f"date        : {r.get('date')}")
            print("\nCONTENT:")
            print(r.get("content", ""))

        print("=" * 100)
    """
    context = build_context(results, query)
    prompt = _prompt(query, context, company, ticker, query_analysis)
    dom = query_analysis["dominant_class"]
    available  = detect_available_data(results)
    sector_ctx = get_sector_context(ticker)
    system     = build_dynamic_system_prompt(dom, sector_ctx, available)

    payload = {
        "model"  : OLLAMA_MODEL,
        "system" : system,
        "prompt" : prompt,
        "stream" : False,
        "options": {"temperature":0.15,"top_p":0.92,"repeat_penalty":1.15,"num_predict":1200,"num_ctx":8192},
    }

    start = time.time()
    perf  = {}
    try:
        resp   = requests.post(f"{OLLAMA_BASE}/api/generate", json=payload, timeout=300)
        resp.raise_for_status()
        resp_json = resp.json()
        answer    = resp_json.get("response","").strip()
        perf      = compute_llm_perf_metrics(resp_json)
    except requests.exceptions.ConnectionError:
        answer = "ollama not running — run: ollama serve"
        log.error("ollama connection refused")
    except Exception as e:
        answer = f"llm call failed: {e}"
        log.error(f"llm call failed: {e}")

    dur = round(time.time()-start, 2)

    # numeric validator — flags percentages not found verbatim in context
    if "[SOURCE" not in answer and "error" not in answer.lower():
        log.warning("no citations in answer")
        answer += "\n\nnote: verify figures against source documents."

    num_pattern = r"\$\d+\.?\d*[BMK]?|\d+\.?\d*%"
    ctx_nums    = set(re.findall(num_pattern, context))
    answer_nums = set(re.findall(num_pattern, answer))
    invented    = answer_nums - ctx_nums
    if invented:
        log.warning(f"unverified figures: {invented}")
        answer += (f"\n\nnote: these figures weren't found verbatim in source documents "
                   f"— verify: {', '.join(sorted(invented))}")

    if perf:
        log.info(
            f"llm perf | prefill {perf['prefill_tok_per_s']} tok/s "
            f"({perf['prompt_tokens']} tok, {perf['prefill_ms']}ms) | "
            f"decode {perf['decode_tok_per_s']} tok/s "
            f"({perf['completion_tokens']} tok, {perf['decode_ms']}ms) | "
            f"load {perf['load_ms']}ms"
        )

    log.info(f"llm done | {OLLAMA_MODEL} | {dur}s")
    return {
        "query"       : query,
        "company"     : company,
        "ticker"      : ticker,
        "answer"      : answer,
        "sources_used": len(results),
        "source_types": list(set(r["source_type"] for r in results)),
        "generated_at": str(datetime.now()),
        "duration_sec": dur,
        "model"       : OLLAMA_MODEL,
        "perf_metrics": perf,
    }


# ──────────────────────────────────────────────────────────────────────────────
# comparison
# ──────────────────────────────────────────────────────────────────────────────

def compare(query: str, tickers: List[str], chunks_map: Dict[str,list],
            metrics_map: Dict[str,dict], known_tickers=None, top_k: int = 5) -> dict:
    log.info(f"compare: '{query}' | {tickers}")
    ticker_data = {}

    query_analysis = analyze_query(query)

    for t in tickers:
        ch = chunks_map.get(t,[])
        if not ch:
            log.warning(f"no chunks for {t}")
            continue
        ranked = hybrid_search(query,t,ch,query_analysis,k=40,known_tickers=known_tickers)
        results = rerank(query,ranked,query_analysis,top_n=top_k,known_tickers=known_tickers)
        results = diversity_filter(results, ch)
        results = filter_boilerplate(results, ch)
        co      = ch[0]["company"] if ch else t
        if query_analysis["needs_computed_metrics"]:
            results = ensure_metrics_chunk(results, ch, t, co)
        ticker_data[t] = {"results":results,"company":co}
        print(f"  {t}: {len(results)} chunks")

    if not ticker_data:
        return {"query":query,"answer":"no data for comparison.","tickers":tickers}

    ctx = f"comparison: {query}\n\n" + "="*40
    for t, d in ticker_data.items():
        ctx += f"\n\n{d['company']} ({t})\n" + "-"*30 + "\n"
        for i, r in enumerate(d["results"]):
            sec  = r.get("metadata",{}).get("section","")
            date = r.get("date","")[:10]
            cite = f"[{t}-SOURCE {i+1} | {r.get('source_type','')} | {sec} | {date}]"
            body = r.get("content","")
            if len(body.split()) > 400:
                body = " ".join(body.split()[:400]) + "..."
            ctx += f"{cite}\n{body}\n\n"

    companies = " vs ".join([f"{d['company']} ({t})" for t,d in ticker_data.items()])
    prompt    = (f"comparison: {companies}\nquestion: {query}\n\n{ctx}\n\n---\n"
                 "compare using the data above — same metrics, same periods where possible. "
                 "note data gaps. cite with [TICKER-SOURCE N].")

    # build the dynamic system prompt — checks what data is actually
    # available across ALL compared tickers combined, and frames using
    # the first ticker's sector (cross-sector comparisons only get one
    # sector's framing right now, a known limitation)
    all_results  = [r for d in ticker_data.values() for r in d["results"]]
    available    = detect_available_data(all_results)
    first_ticker = list(ticker_data.keys())[0]
    sector_ctx   = get_sector_context(first_ticker)
    system       = build_dynamic_system_prompt("comparison", sector_ctx, available)

    payload = {
        "model"  : OLLAMA_MODEL,
        "system" : system,
        "prompt" : prompt,
        "stream" : False,
        "options": {"temperature":0.1,"top_p":0.92,"num_predict":1500,"num_ctx":8192},
    }
    start = time.time()
    try:
        resp   = requests.post(f"{OLLAMA_BASE}/api/generate", json=payload, timeout=300)
        answer = resp.json().get("response","").strip()
    except Exception as e:
        answer = f"comparison failed: {e}"
        log.error(f"compare failed: {e}")

    return {
        "query"       : query,
        "tickers"     : tickers,
        "answer"      : answer,
        "chunks_used" : {t:len(d["results"]) for t,d in ticker_data.items()},
        "duration_sec": round(time.time()-start,2),
        "model"       : OLLAMA_MODEL,
    }


# ──────────────────────────────────────────────────────────────────────────────
# full query pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_query(query: str, ticker: str, company: str, chunks: list,
              known_tickers=None, top_k: int = 10) -> dict:
    log.info(f"query: '{query}' | {ticker}")

    query_analysis = analyze_query(query)
    weights = query_analysis["weights"]
    dom     = query_analysis["dominant_class"]
    ncm     = query_analysis["needs_computed_metrics"]

    print(f"\nquery: {query}")
    print(f"  class: {dom} | needs computed metrics: {ncm}")
    print(f"  weights: { {k:round(v,2) for k,v in weights.items()} }")

    ranked  = hybrid_search(query, ticker, chunks,query_analysis, k=40, known_tickers=known_tickers)
    results = rerank(query,ranked,query_analysis,top_n=top_k,known_tickers=known_tickers)
    """
    # ============================================================
    # RETRIEVAL DIAGNOSTICS — DEBUG ONLY
    # Shows exactly what will be sent to the LLM.
    # Does NOT modify retrieval results.
    # ============================================================
    print("\n" + "=" * 100)
    print("FINAL RETRIEVAL DIAGNOSTICS")
    print("=" * 100)

    print(f"query            : {query}")
    print(f"dominant class   : {query_analysis['dominant_class']}")
    print(f"preferred        : {query_analysis['preferred_sections']}")
    print(f"excluded         : {query_analysis['excluded_sections']}")
    print(f"chunks selected  : {len(results)}")

    print("-" * 100)

    for i, r in enumerate(results, 1):
        metadata = r.get("metadata", {})

        section = metadata.get("section", "unknown")
        form_type = metadata.get("form_type", "")
        source_type = r.get("source_type", "")
        date = r.get("date", "")[:10]
        quality = r.get("quality", 0)
        final_score = r.get("final_score", 0)
        rerank_score = r.get("rerank_score", 0)
        hybrid_score = r.get("hybrid_score", 0)

        content = " ".join(r.get("content", "").split())
        preview = content[:300]

        print(
            f"[{i:02d}] "
            f"final={final_score:.4f} | "
            f"rerank={rerank_score:.4f} | "
            f"hybrid={hybrid_score:.4f} | "
            f"quality={quality:.3f}"
        )

        print(
            f"     section={section} | "
            f"source={source_type} | "
            f"form={form_type} | "
            f"date={date}"
        )

        print(f"     preview={preview}...")
        print()

    print("=" * 100)
"""

    results = diversity_filter(results, chunks)
    results = filter_boilerplate(results, chunks)

    # force-inject computed metrics if the query needs it and retrieval missed it
    if ncm:
        results = ensure_metrics_chunk(results, chunks, ticker, company)

    sector_ctx = get_sector_context(ticker)
    results = maybe_inject_macro(results,sector_ctx,query,query_analysis)
    #results = maybe_inject_macro(results, sector_ctx, query)
    results = maybe_inject_price(results,ticker,query,query_analysis)
    #results = maybe_inject_price(results, ticker, query)

    print(f"\n  chunks to llm: {len(results)}")
    for r in results:
        sec  = r["metadata"].get("section","")
        date = r.get("date","")[:10]
        print(f"  [{r['rank']:>2}] {r['source_type']:<28} sec={sec:<20} score={r['final_score']:.4f}  {date}")

    if not results:
        log.warning(f"no results for: {query}")
        return {"query":query,"answer":"not enough indexed data.","sources":[]}

    resp = call_llm(query, results, company, ticker, query_analysis)
    resp["retrieved_chunks"] = results
    log.info(f"done | {len(results)} sources | {resp['source_types']} | {resp['duration_sec']}s")
    return resp


# ──────────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\nFinSight — Bloomberg Terminal (free)")
    print(f"LLM: {OLLAMA_MODEL} via ollama")
    print("="*45)

    if not check_ollama():
        print(f"\nrun 'ollama serve' in a separate terminal, then restart.")
        print(f"if model not downloaded: ollama pull {OLLAMA_MODEL}")
        exit(1)

    print("\nchecking macro data...")
    ingest_macro_data(years=10)

    raw     = input("\nticker(s) — comma-separated for comparison (e.g. AAPL,META): ").strip().upper()
    tickers = [t.strip() for t in raw.split(",")]
    primary = tickers[0]

    chunks_map  : Dict[str,list] = {}
    company_map : Dict[str,str]  = {}
    metrics_map : Dict[str,dict] = {}

    for t in tickers:
        existing = load_chunks(t)
        if existing:
            print(f"\n{t}: {len(existing)} chunks in db — refreshing news + metrics...")
            refresh_news(t)
            chunks = load_chunks(t)
        else:
            print(f"\n{t}: no data — running full 3-year ingestion...")
            _, chunks = ingest(t, years=3)

        chunks_map[t]  = chunks
        company_map[t] = chunks[0]["company"] if chunks else t
        metrics_map[t] = get_metrics(t, company_map[t])

        build_bm25(t, chunks)

        tc = Counter(c["source_type"] for c in chunks)
        print(f"\n  {t} breakdown:")
        for stype, n in sorted(tc.items()):
            print(f"    {stype}: {n}")

        # show what the metrics engine computed
        m = metrics_map[t]
        print(f"\n  computed metrics for {t}:")
        if m.get("gross_margin_latest") is not None:         print(f"    gross margin (annual):    {pct(m['gross_margin_latest'])}")
        if m.get("gross_margin_latest_quarter") is not None: print(f"    gross margin (latest Q):  {pct(m['gross_margin_latest_quarter'])} ({m.get('revenue_latest_quarter_period','')})")
        if m.get("operating_margin_latest") is not None:     print(f"    operating margin:         {pct(m['operating_margin_latest'])}")
        if m.get("net_margin_latest") is not None:           print(f"    net margin:               {pct(m['net_margin_latest'])}")
        if m.get("revenue_cagr_3y") is not None:             print(f"    revenue CAGR 3y:          {pct(m['revenue_cagr_3y'])}")
        if m.get("roe") is not None:                         print(f"    ROE:                      {pct(m['roe'])}")
        if m.get("debt_to_equity") is not None:              print(f"    debt/equity:              {m['debt_to_equity']:.2f}x")

    start_scheduler(tickers)

    print(f"\n{'='*45}")
    print(f"ready | {OLLAMA_MODEL}")
    print(f"loaded: {', '.join([f'{t} ({company_map.get(t,t)})' for t in tickers])}")
    print("commands: exit | status | switch TICKER | compare QUERY | add TICKER | metrics TICKER")
    print("="*45)

    active    = primary
    known_set = set(tickers)

    while True:
        try:
            label = "/".join(tickers) if len(tickers)>1 else active
            query = input(f"\n{label} > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nshutting down.")
            break

        if not query: continue
        if query.lower() in ("exit","quit","q"):
            print("bye.")
            break

        if query.lower() == "status":
            print(f"\nactive: {active} | loaded: {list(chunks_map.keys())} | watched: {WATCHED}")
            for t, ch in chunks_map.items():
                print(f"  {t}: {len(ch)} chunks")
            continue

        if query.lower().startswith("metrics "):
            t = query.split()[-1].upper()
            co = company_map.get(t, t)
            m  = get_metrics(t, co, force_refresh=True)
            print(f"\ncomputed metrics — {co} ({t}):")
            print(metrics_to_text(m))
            continue

        if query.lower().startswith("switch "):
            nt = query.split()[-1].upper()
            if nt in chunks_map:
                active = nt
                print(f"switched to {active}")
            else:
                print(f"{nt} not loaded")
            continue

        if query.lower().startswith("add "):
            nt = query.split()[-1].upper()
            print(f"\nadding {nt}...")
            _, nc = ingest(nt, years=3)
            chunks_map[nt]  = nc
            company_map[nt] = nc[0]["company"] if nc else nt
            metrics_map[nt] = get_metrics(nt, company_map[nt])
            known_set.add(nt)
            build_bm25(nt, nc)
            start_scheduler([nt])
            print(f"added {nt} — {len(nc)} chunks")
            continue

        if query.lower().startswith("compare "):
            cq = query[8:].strip()
            if len(tickers) < 2:
                print("need at least 2 tickers")
                continue
            result = compare(cq, tickers, chunks_map, metrics_map, known_set, top_k=5)
            print(f"\n{'='*45}")
            print(f"comparison: {' vs '.join([company_map.get(t,t) for t in tickers])}")
            print("="*45)
            print(result["answer"])
            print(f"\n{result.get('duration_sec','?')}s | {result.get('model','?')}")
            continue

        ch = chunks_map.get(active,[])
        co = company_map.get(active, active)

        resp = run_query(query, active, co, ch, known_set, top_k=10)

        print(f"\n{'='*45}")
        print(f"{co} ({active})")
        print("="*45)
        print(resp["answer"])
        print(f"\nsources: {resp['sources_used']} | types: {resp['source_types']}")
        print(f"time: {resp.get('duration_sec','?')}s | model: {resp.get('model','?')}")