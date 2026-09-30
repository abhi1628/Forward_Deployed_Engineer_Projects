# QuickMart Ticket Cleaner: a Forward Deployed Engineering project

**Turning a client's messy support tickets into clean, structured, measured data using a local LLM (Ollama). No cloud APIs, no data leaving the machine.**

![python](https://img.shields.io/badge/python-3.10+-blue) ![ollama](https://img.shields.io/badge/LLM-Ollama%20(local)-black) ![license](https://img.shields.io/badge/license-MIT-green)

---

## TL;DR

- **Client (fictional):** QuickMart, a retail chain with support tickets arriving in mixed formats, languages (English + Hinglish), and quality.
- **Problem:** Tickets can't be analysed because dates, amounts, cities and duplicates are inconsistent, and nobody tags the complaint type.
- **Solution:** A reproducible pipeline: rule-based cleaning, then local-LLM classification (category / sentiment / urgency / summary), then evaluation against ground truth, then a management report.
- **Key lesson:** Most accuracy gains came from **clarifying label definitions with the client**, not from changing the model.

> This is a **simulation** built on synthetic data to practise the FDE workflow. All numbers below are from synthetic tickets and should be read as baselines, not production claims.
>
> **Status:** tuning-set results are complete (up to 97.9%, see Results). The untouched holdout evaluation and the multi-model comparison are the next steps.

---

## What is a Forward Deployed Engineer (FDE)?

An FDE embeds with a customer, understands their messy real-world problem, and ships a working solution inside the customer's environment, fast. The work is roughly 30% code and 70% scoping, asking the right questions, handling constraints, and proving results. This project practises that loop: **understand, clean, build, measure, find failures, ask the client, iterate.**

---

## The client brief (simulated)

> *"We get ~100 tickets a day from WhatsApp, email and the app. Dates are all over the place, amounts are typed however people like, the same ticket sometimes arrives twice, and half the messages are in Hinglish. We want to know: what are customers complaining about, where, how much money is at stake, and which tickets are urgent? Our data can't leave our servers."*

**Constraints derived from the brief**

| Constraint | Design decision |
|---|---|
| Data cannot leave the premises | Local LLM via Ollama, no external API calls |
| Hinglish + typos + shouting | LLM-based extraction instead of keyword rules |
| Needs to be trustworthy | Ground-truth evaluation + per-segment accuracy |
| Runs on modest hardware | 3B-parameter model, response caching |

---

## Architecture

```mermaid
flowchart LR
    A[Messy CSV] --> B[Rule-based cleaning]
    B --> C[Local LLM via Ollama]
    C --> D[Validation and fallback]
    D --> E[Evaluation vs ground truth]
    E --> F[Report and cleaned CSV]
```

1. **Clean (pandas + regex):** deduplicate tickets, parse 4 date formats, extract amounts from `Rs. 1,200` / `1200/-` / `₹1200` / `1200 rupees`, normalise city names.
2. **Classify (Ollama, JSON mode, temperature 0):** category, sentiment, urgency, one-line English summary.
3. **Validate:** any out-of-schema answer falls back to a safe default; failed calls are retried and never cached.
4. **Evaluate:** accuracy overall, per segment, confusion matrix, list of misclassified tickets.
5. **Report:** category counts, money at stake per category, city breakdown, urgent tickets, data-quality notes.

---

## The mess (what the dataset contains)

| Field | Problems injected |
|---|---|
| `date` | 4 formats, plus "yesterday", "last week", blanks |
| `amount` | `Rs. 1,200`, `1200/-`, `₹1200`, `1200 rupees`, `N/A`, blank |
| `city` | `Jabalpur` / `jabalpur` / `JABALPUR` / `Jbp` / trailing spaces / blank |
| `customer` | inconsistent casing, padding, blanks |
| `message` | English + Hinglish, typos, ALL CAPS, `!!!`, filler like "pls hurry" |
| rows | duplicate tickets with differently-cased IDs |

The generator also writes a hidden `truth.csv` so accuracy can be measured.

---

## Repo structure

```
quickmart-ticket-cleaner/
├── make_data.py        # synthetic messy dataset + ground truth (tuning and holdout)
├── pipeline.py         # clean, classify, evaluate, report
├── requirements.txt
├── data/               # generated CSVs
└── outputs/            # cleaned data, reports, misclassified tickets
```

---

## Quickstart

**Prerequisites:** Python 3.10+, [Ollama](https://ollama.com/download) installed and running.

```bash
git clone https://github.com/YOUR_USERNAME/quickmart-ticket-cleaner.git
cd quickmart-ticket-cleaner
python -m venv venv
# Windows: venv\Scripts\Activate.ps1    |    Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
ollama pull llama3.2
```

```bash
# 1. Generate the tuning dataset
python make_data.py --seed 42

# 2. Run the pipeline
python pipeline.py --tag tuning_seed42

# 3. Generate and run the untouched holdout set (phrasings NOT used for tuning)
python make_data.py --holdout --seed 99
python pipeline.py --input data/holdout_tickets.csv --truth data/holdout_truth.csv --tag holdout

# 4. Compare another model
ollama pull qwen2.5:3b
python pipeline.py --model qwen2.5:3b --tag qwen_tuning_seed42
```

Windows note: if you see encoding errors, run `$env:PYTHONUTF8 = "1"` first.

---

## Results

### Iteration log (same model: `llama3.2:latest`)

| Run | Data | Prompt change | Accuracy |
|---|---|---|---|
| 1 | seed 42, 70 tickets | v1: category names only | 91.4% (6 errors, **all in `other`**) |
| 2 | seed 42, 96 tickets (harder cases added) | v2: written category definitions | 88.5% (11 errors) |
| 3 | seed 7, 96 tickets | v3: priority rules (refund beats quality; complaint beats question) | 96.9% (3 errors) |
| 4 | seed 42, 96 tickets | v4: Hinglish hints | **97.9%** (2 errors) |
| 5 | **holdout**, 34 tickets, prompt frozen | v4 | *not yet run* |

**Run 4 per-segment result (seed 42, prompt v4):**

| Segment | Correct | Total | Accuracy |
|---|---|---|---|
| base | 68 | 70 | 97.1% |
| hard_mixed (complaint + unrelated question) | 10 | 10 | 100.0% |
| hard_other (questions, many mention an order ID) | 16 | 16 | 100.0% |

**Controlled comparison:** runs 2 and 4 use the *same data* (seed 42, 96 tickets) and the same model; only the prompt changed (v2 to v4). That isolates the prompt effect: **88.5% to 97.9%**.

**Caveat on runs 2 to 3:** they differ in both prompt *and* random seed, so that step alone cannot be attributed to the prompt. Run 4 exists to separate these effects.

**Caveat on run 4:** prompt v4 was written by looking at this set's failures, so 97.9% is an optimistic, tuning-set number. The untouched holdout set (run 5) is the number to trust; it has not been run yet.

### Model comparison

Not yet run. Only `llama3.2` (3B) has been evaluated so far. The pipeline takes a `--model` flag, so comparing another model (e.g. `qwen2.5:3b`) is one command; see Quickstart.

---

## Key findings (the FDE story)

1. **The model wasn't the problem, the labels were.** In run 1 every error was in the `other` class. The prompt never said what "other" meant, so the model forced general questions ("do you sell laptop bags?") into complaint buckets. Writing definitions fixed the class.
2. **Two labels overlapped by nature.** "Product damaged and I want my money back" is both *product_quality* and *refund*. No model can guess the client's business rule. A written priority rule resolved it. In production this is a question to ask the client, not something to engineer around.
3. **Tone words poisoned the signal.** "pls hurry" and "!!!" pushed questions into the *delivery_delay* bucket. The prompt now states that tone is not a category signal.
4. **Hinglish needs explicit hints.** "refund nahi mila... 10 din ho gaye" was read as a delay because of "days have passed". Prompt v4 added explicit Hinglish hints alongside the priority rules, and on the same seed-42 data accuracy rose from 88.5% (v2) to 97.9% (v4). I did not ablate each rule separately, so I can't say which one mattered most.
5. **Measure by segment, not just overall.** Overall accuracy hid that the harder segments (mixed complaint + question) behaved very differently from the base set.

---

## Limitations (honest section)

- **Synthetic data.** Tickets are generated from templates. Real customers are messier; expect lower accuracy on real data.
- **Small test sets.** With 96 (tuning) and 34 (holdout) tickets, a single error moves accuracy by about 1 to 3 points. Several tickets are the same template with different noise, so effective diversity is lower than the row count.
- **Prompt tuning on the tuning set.** Prompts v2 to v4 were refined by looking at that set's failures, so tuning-set scores are optimistic. The holdout set exists to check this, but **the holdout result is not yet included here**.
- **Some holdout overlap.** The v4 prompt's examples (cash on delivery, invoices, gift wrap, franchise) were derived from tuning failures; the holdout uses different `other` topics, but the wording style is still mine.
- **Ambiguous cases deliberately excluded.** Order-status questions ("where is my order?") and praise-plus-question are omitted because they need a client decision.
- **Single-label output.** Real tickets can legitimately belong to two categories.

---

## Questions I would ask the client before production

1. Is "where is my order?" a delay complaint or a general query?
2. Should a message with praise *and* a complaint be routed as a complaint?
3. What counts as **high urgency**: amount threshold, days overdue, or repeat contact?
4. Which languages and scripts appear in real tickets (Devanagari Hinglish, regional languages)?
5. Who reviews low-confidence tickets, and what volume is acceptable?

---

## What I would do next in production

- Add a **confidence score / human-review queue** for uncertain tickets.
- Build a **labelled golden set from real tickets** (200+), reviewed by the client's support team.
- Add **few-shot examples** and compare against fine-tuning a small model.
- Wrap the pipeline in an API/dashboard (FastAPI or Streamlit) and schedule it on the client's server.
- Monitor drift: re-evaluate monthly on fresh labelled samples.

---

## Tech stack

Python, pandas, requests, Ollama (local LLM serving), `llama3.2` and `qwen2.5` (3B class models).

## Author

**Abhishek Singh**: [LinkedIn](https://linkedin.com/in/abhishek-singh-170726123) | [GitHub](https://github.com/abhi1628)

## License

MIT
