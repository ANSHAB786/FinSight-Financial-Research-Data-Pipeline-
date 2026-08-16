TEST_MATRIX = {
    "standard": {
        "ticker": "AAPL",
        "queries": {
            "financial_metric"    : "what was the gross margin last quarter",
            "capital_allocation"  : "should the company increase its dividend or buy back more stock",
            "growth"              : "how has revenue grown over the past 3 years",
            "risk"                : "what are the biggest risks the company faces",
            "investment_thesis"   : "what is the bull case for this stock",
            "company_health"      : "how healthy is this company financially",
            "balance_sheet"       : "how much debt does the company have",
            "management_commentary": "what did management say about future guidance",
            "news_sentiment"      : "what's the latest news on this company",
            "segment"             : "how is each business segment performing",
            "scenario"            : "what would need to happen for the outlook to change",
            "evidence"            : "what evidence supports the current investment view",
            "contradictions"      : "are there any mixed signals in the data",
            "price"               : "how has the stock price moved over the past year",
            "assumptions"         : "what assumptions underlie the current growth outlook",
        },
    },
    "bank": {
        "ticker": "JPM",
        "queries": {
            "financial_metric"  : "what is the net interest margin",
            "risk"              : "what credit and liquidity risks does the bank face",
            "company_health"    : "how healthy is this bank financially",
            "capital_allocation": "how is the bank returning capital to shareholders",
            "growth"            : "how has net interest income trended over the past few years",
            "balance_sheet"     : "what does the bank's balance sheet leverage look like",
            "management_commentary": "what did management say about loan growth on the earnings call",
        },
    },
    "reit": {
        "ticker": "O",
        "queries": {
            "financial_metric"  : "what is the FFO per share",
            "capital_allocation": "how is the company using its cash",
            "growth"            : "how has FFO grown over the past few years",
            "risk"              : "what risks does the company face from interest rates",
            "balance_sheet"     : "how much leverage does the REIT carry",
        },
    },
    "utility": {
        "ticker": "DUK",
        "queries": {
            "financial_metric"  : "what is the operating margin",
            "risk"              : "what regulatory risks does the company face",
            "capital_allocation": "how is the company funding capital expenditures",
            "growth"            : "how has revenue trended over the past few years",
            "balance_sheet"     : "is the company's debt level normal for a utility",
        },
    },
    "energy_producer": {
        "ticker": "XOM",
        "queries": {
            "financial_metric"  : "what is the free cash flow",
            "growth"            : "how has revenue trended given commodity price swings",
            "risk"              : "what risks does commodity price volatility pose",
            "capital_allocation": "how is the company allocating capital between buybacks and reinvestment",
            "balance_sheet"     : "how much debt does the company carry",
        },
    },
    "insurance": {
        "ticker": "PGR",
        "queries": {
            "financial_metric"  : "what is the combined ratio",
            "risk"              : "what underwriting risks does the company face",
            "growth"            : "how has premium growth trended over the past few years",
            "company_health"    : "how healthy is this insurer financially",
        },
    },
    "biotech_prerevenue": {
        "ticker": "MRNA",
        "queries": {
            "financial_metric"  : "what is the company's cash runway",
            "risk"              : "what clinical and regulatory risks does the company face",
            "balance_sheet"     : "how much cash does the company have relative to its burn rate",
        },
    },
}

# eval_finsight.py
"""
FinSight eval harness — batch-scores answer quality across sectors and
query classes using DeepEval's Faithfulness/AnswerRelevancy metrics.

Not called from finsight.py's live query path. Run standalone:
    python eval_finsight.py                  # full matrix
    python eval_finsight.py --sector bank     # one sector only
    python eval_finsight.py --ticker JPM      # one ticker only
"""

import argparse
import csv
import json
import time
from collections import defaultdict
from datetime import datetime
from statistics import mean

from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

from Finsight import run_query, load_chunks, ingest, OLLAMA_BASE, OLLAMA_MODEL,log


# ──────────────────────────────────────────────────────────────────────────
# test matrix — one representative ticker per SECTOR_PROFILES entry, with
# queries chosen to hit different _QUERY_CLASSES per ticker. tickers are
# real, well-covered symbols for their sector — not stand-ins for fabricated
# data; the actual filings/metrics still come from your live pipeline.
# ──────────────────────────────────────────────────────────────────────────




# ──────────────────────────────────────────────────────────────────────────
# scoring
# ──────────────────────────────────────────────────────────────────────────
from deepeval.models import OllamaModel

JUDGE_MODEL = OllamaModel(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_BASE,
)

def score_answer_quality(query: str, answer: str, results: list) -> dict:
    test_case = LLMTestCase(
        input=query,
        actual_output=answer,
        retrieval_context=[r["content"] for r in results],
    )
    faithfulness = FaithfulnessMetric(threshold=0.7, model=JUDGE_MODEL)
    relevancy    = AnswerRelevancyMetric(threshold=0.7, model=JUDGE_MODEL)
    faithfulness.measure(test_case)
    relevancy.measure(test_case)
    return {
        "faithfulness_score" : round(faithfulness.score, 3),
        "faithfulness_reason": faithfulness.reason,
        "relevancy_score"    : round(relevancy.score, 3),
        "relevancy_reason"   : relevancy.reason,
    }


def run_case(sector: str, ticker: str, qclass: str, query: str, chunks: list, company: str) -> dict:
    log.info(f"eval: [{sector}/{qclass}] {ticker} — {query}")
    t0   = time.time()
    resp = run_query(query, ticker, company, chunks)
    quality = score_answer_quality(query, resp["answer"], resp["retrieved_chunks"])

    return {
        "sector"             : sector,
        "ticker"             : ticker,
        "query_class"        : qclass,
        "query"              : query,
        "answer"             : resp["answer"],
        "sources_used"       : resp["sources_used"],
        "citation_count"     : resp.get("answer_metrics", {}).get("citation_count"),
        "source_coverage_pct": resp.get("answer_metrics", {}).get("source_coverage_pct"),
        "invented_pcts"      : resp.get("answer_metrics", {}).get("invented_pcts"),
        "prefill_tok_per_s"  : resp.get("perf_metrics", {}).get("prefill_tok_per_s"),
        "decode_tok_per_s"   : resp.get("perf_metrics", {}).get("decode_tok_per_s"),
        "duration_sec"       : resp["duration_sec"],
        "eval_duration_sec"  : round(time.time() - t0, 2),
        **quality,
    }


# ──────────────────────────────────────────────────────────────────────────
# batch runner
# ──────────────────────────────────────────────────────────────────────────

def run_eval_batch(matrix: dict) -> list:
    results = []
    for sector, cfg in matrix.items():
        ticker = cfg["ticker"]
        chunks = load_chunks(ticker)
        if not chunks:
            log.warning(f"no chunks for {ticker} — ingesting fresh (3y)")
            _, chunks = ingest(ticker, years=3)
        if not chunks:
            log.error(f"{ticker}: ingestion produced nothing, skipping sector {sector}")
            continue
        company = chunks[0]["company"]

        for qclass, query in cfg["queries"].items():
            try:
                results.append(run_case(sector, ticker, qclass, query, chunks, company))
            except Exception as e:
                log.error(f"eval case failed [{sector}/{qclass}]: {e}")
                results.append({
                    "sector": sector, "ticker": ticker, "query_class": qclass,
                    "query": query, "error": str(e),
                })
    return results


# ──────────────────────────────────────────────────────────────────────────
# aggregation + reporting
# ──────────────────────────────────────────────────────────────────────────

def summarize(results: list) -> dict:
    ok = [r for r in results if "error" not in r]
    by_sector = defaultdict(list)
    by_class  = defaultdict(list)
    for r in ok:
        by_sector[r["sector"]].append(r)
        by_class[r["query_class"]].append(r)

    def agg(rows):
        return {
            "n"                 : len(rows),
            "avg_faithfulness"  : round(mean(r["faithfulness_score"] for r in rows), 3),
            "avg_relevancy"     : round(mean(r["relevancy_score"] for r in rows), 3),
            "avg_source_coverage_pct": round(mean(r["source_coverage_pct"] or 0 for r in rows), 1),
            "any_invented_pcts" : sum(1 for r in rows if r["invented_pcts"]),
        }

    return {
        "total_cases" : len(results),
        "failed_cases": len(results) - len(ok),
        "overall"     : agg(ok) if ok else {},
        "by_sector"   : {s: agg(rows) for s, rows in by_sector.items()},
        "by_class"    : {c: agg(rows) for c, rows in by_class.items()},
    }


def save_results(results: list, summary: dict):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    with open(f"eval_results_{stamp}.json", "w") as f:
        json.dump({"summary": summary, "cases": results}, f, indent=2)

    fieldnames = ["sector","ticker","query_class","query","faithfulness_score",
                  "relevancy_score","source_coverage_pct","citation_count",
                  "invented_pcts","duration_sec","eval_duration_sec"]
    with open(f"eval_results_{stamp}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print(f"\nsaved: eval_results_{stamp}.json / .csv")


# ──────────────────────────────────────────────────────────────────────────
# cli
# ──────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sector", help="run only this sector (e.g. bank)")
    parser.add_argument("--ticker", help="run only this ticker, across its sector's queries")
    args = parser.parse_args()

    matrix = TEST_MATRIX
    if args.sector:
        matrix = {args.sector: TEST_MATRIX[args.sector]}
    elif args.ticker:
        matrix = {s: c for s, c in TEST_MATRIX.items() if c["ticker"] == args.ticker.upper()}

    if not matrix:
        print("no matching sector/ticker in TEST_MATRIX")
        exit(1)

    n_cases = sum(len(c["queries"]) for c in matrix.values())
    print(f"running {n_cases} eval cases across {len(matrix)} sector(s)")
    print("each case makes 2 judge-LLM API calls — this will take a while and cost real API spend.\n")

    results = run_eval_batch(matrix)
    summary = summarize(results)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(json.dumps(summary, indent=2))

    save_results(results, summary) 