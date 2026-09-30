"""
QuickMart ticket cleaner: clean messy tickets, classify them with a local
LLM (Ollama), evaluate against ground truth, and write a report.

Usage:
    python pipeline.py
    python pipeline.py --model qwen2.5:3b --tag qwen_tuning
    python pipeline.py --input data/holdout_tickets.csv --truth data/holdout_truth.csv --tag holdout
"""
import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime

import pandas as pd
import requests

OLLAMA_BASE = "http://localhost:11434"
OUT_DIR = "outputs"
CACHE_FILE = os.path.join(OUT_DIR, "llm_cache.json")
CATEGORIES = ["delivery_delay", "refund", "product_quality",
              "payment_issue", "praise", "other"]

PROMPT_VERSION = "v4"
SYSTEM_PROMPT = f"""You are a support-ticket analyst for a retail company.
Messages may be English or Hinglish, with typos and shouting.
Words like "pls hurry" or "!!!" are just tone, not a category signal.
Return ONLY a JSON object with these keys:
"category": one of {CATEGORIES}
"sentiment": one of ["positive","neutral","negative"]
"urgency": one of ["low","medium","high"]
"summary": max 12 words, in English

Category definitions:
- delivery_delay: order is late, not arrived, or courier not responding
- refund: customer asks for money back, a refund, a return, or refund status
- product_quality: item is defective, damaged, faded, or stopped working,
  and the customer does NOT ask for money back
- payment_issue: payment failed, stuck, or money deducted wrongly
- praise: customer is thanking or complimenting the service
- other: general questions or requests that are NOT a complaint. Examples:
  store timings, product availability, cash on delivery or payment options,
  changing address or phone number, invoices, gift wrap, franchise enquiries.
  Asking about an order (e.g. changing its address) is still "other" unless
  the customer says something went wrong.

Priority rules (apply in this order):
1. If the customer asks for money back, a refund, or a return, the category is
   "refund", even if the product is also damaged.
2. If a message contains BOTH a complaint and an unrelated question, classify
   by the complaint and ignore the question.
3. If there is no complaint at all, use "other" (or "praise" if thankful).

Hinglish hints:
- "refund", "paisa wapas", "paise wapas" always signal "refund", even when the
  message also says how many days have passed ("10 din ho gaye").
- Mentioning waiting time does NOT make something a delivery_delay unless the
  ORDER itself has not arrived.
- A question about changing an address or details BEFORE shipping is "other",
  not a delay.

Urgency rules: high = money lost or order badly overdue, medium = problem but
no money at risk, low = questions and praise."""


# ---------- Step 1: rule-based cleaning ----------
def clean_amount(s):
    m = re.search(r"\d[\d,]*", str(s))
    return int(m.group().replace(",", "")) if m else None


def clean_date(s):
    s = str(s).strip()
    for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d-%b-%y", "%b %d, %Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            pass
    return pd.NaT


def clean_city(s):
    s = str(s).strip().lower()
    if s in ("", "nan"):
        return "Unknown"
    if s == "jbp":
        return "Jabalpur"
    return s.title()


def clean_data(df):
    df = df.copy()
    df["ticket_id"] = df["ticket_id"].str.strip().str.upper()
    before = len(df)
    df = df.drop_duplicates(subset="ticket_id", keep="first")
    removed = before - len(df)
    print(f"Removed {removed} duplicate tickets")
    df["date"] = df["date"].apply(clean_date)
    df["customer"] = df["customer"].str.strip().str.title().replace("", "Unknown")
    df["city"] = df["city"].apply(clean_city)
    df["amount"] = df["amount"].apply(clean_amount)
    df["message"] = df["message"].str.strip().str.replace(r"\s+", " ", regex=True)
    return df, removed


# ---------- Step 2: LLM extraction ----------
def check_ollama(model):
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        r.raise_for_status()
        names = [m["name"] for m in r.json().get("models", [])]
    except Exception:
        sys.exit("Ollama is not reachable at localhost:11434. "
                 "Windows: open the Ollama app. Linux: run `ollama serve`.")
    if model not in names and f"{model}:latest" not in names:
        sys.exit(f"Model '{model}' not found. Installed: {names}. "
                 f"Run: ollama pull {model}")


def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def cache_key(model, text):
    raw = f"{model}|{SYSTEM_PROMPT}|{text}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def ask_llm(model, text, retries=3):
    payload = {
        "model": model,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
    }
    err = None
    for _ in range(retries):
        try:
            r = requests.post(f"{OLLAMA_BASE}/api/chat", json=payload, timeout=180)
            r.raise_for_status()
            data = json.loads(r.json()["message"]["content"])
            if data.get("category") not in CATEGORIES:
                data["category"] = "other"
            if data.get("sentiment") not in ("positive", "neutral", "negative"):
                data["sentiment"] = "neutral"
            if data.get("urgency") not in ("low", "medium", "high"):
                data["urgency"] = "low"
            data["summary"] = str(data.get("summary", ""))[:100]
            return data, True
        except Exception as e:
            err = e
    print(f"  LLM failed after {retries} tries: {err}")
    return {"category": "other", "sentiment": "neutral",
            "urgency": "low", "summary": "LLM_FAILED"}, False


def enrich(df, model):
    os.makedirs(OUT_DIR, exist_ok=True)
    cache = load_cache()
    results = []
    for i, text in enumerate(df["message"], 1):
        key = cache_key(model, text)
        if key not in cache:
            print(f"[{i}/{len(df)}] asking {model}...")
            data, ok = ask_llm(model, text)
            if ok:  # never cache failures
                cache[key] = data
                with open(CACHE_FILE, "w", encoding="utf-8") as f:
                    json.dump(cache, f, ensure_ascii=False)
            results.append(data)
        else:
            results.append(cache[key])
    out = pd.DataFrame(results, index=df.index)
    return pd.concat([df, out], axis=1)


# ---------- Step 3: evaluation + report ----------
def evaluate(df, truth_path, tag):
    if not os.path.exists(truth_path):
        return None
    truth = pd.read_csv(truth_path).drop_duplicates("ticket_id")
    m = df.merge(truth, on="ticket_id")
    m["correct"] = m["category"] == m["true_category"]
    acc = m["correct"].mean() * 100
    wrong = m[~m["correct"]][["ticket_id", "group", "message", "category", "true_category"]]
    wrong.to_csv(os.path.join(OUT_DIR, f"misclassified_{tag}.csv"), index=False)
    by_group = m.groupby("group")["correct"].agg(["sum", "count"])
    by_group["acc_%"] = (by_group["sum"] / by_group["count"] * 100).round(1)
    confusion = pd.crosstab(m["true_category"], m["category"],
                            rownames=["true"], colnames=["predicted"])
    return {"acc": acc, "wrong": len(wrong), "total": len(m),
            "by_group": by_group, "confusion": confusion}


def write_report(df, ev, model, tag, removed):
    L = [f"# QuickMart Ticket Report ({tag})\n",
         f"- Model: `{model}`",
         f"- Prompt version: `{PROMPT_VERSION}`",
         f"- Tickets analysed: {len(df)} (duplicates removed: {removed})\n"]
    if ev:
        L += [f"## Accuracy\n",
              f"**{ev['acc']:.1f}%** ({ev['total'] - ev['wrong']}/{ev['total']} correct)\n",
              "```", ev["by_group"].to_string(), "```\n",
              "## Confusion matrix\n", "```", ev["confusion"].to_string(), "```\n"]
    L += ["## Tickets by category\n```", df["category"].value_counts().to_string(), "```\n",
          "## Sentiment\n```", df["sentiment"].value_counts().to_string(), "```\n",
          "## Amount at stake by category (Rs.)\n```",
          df.groupby("category")["amount"].agg(["count", "sum"]).to_string(), "```\n",
          "## Tickets by city\n```", df["city"].value_counts().to_string(), "```\n",
          "## High-urgency tickets\n"]
    urgent = df[df["urgency"] == "high"]
    for _, r in urgent.iterrows():
        L.append(f"- {r['ticket_id']} ({r['city']}): {r['summary']}")
    if urgent.empty:
        L.append("- none")
    L.append(f"\n## Data quality notes\n- {df['amount'].isna().sum()} tickets have no usable amount"
             f"\n- {df['date'].isna().sum()} tickets have no usable date")
    path = os.path.join(OUT_DIR, f"report_{tag}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    return path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/messy_tickets.csv")
    p.add_argument("--truth", default="data/truth.csv")
    p.add_argument("--model", default=os.environ.get("MODEL", "llama3.2:latest"))
    p.add_argument("--tag", default="tuning")
    a = p.parse_args()

    check_ollama(a.model)
    os.makedirs(OUT_DIR, exist_ok=True)

    raw = pd.read_csv(a.input, dtype=str, keep_default_na=False)
    print(f"Loaded {len(raw)} raw rows | model={a.model} | prompt={PROMPT_VERSION}")
    clean, removed = clean_data(raw)
    final = enrich(clean, a.model)
    final.to_csv(os.path.join(OUT_DIR, f"cleaned_{a.tag}.csv"), index=False)

    ev = evaluate(final, a.truth, a.tag)
    if ev:
        print(ev["by_group"].to_string())
        print(f"Category accuracy: {ev['acc']:.1f}% ({ev['wrong']} wrong of {ev['total']})")
    print("Report:", write_report(final, ev, a.model, a.tag, removed))
