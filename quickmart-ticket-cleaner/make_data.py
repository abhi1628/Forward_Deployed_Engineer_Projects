"""
Generate a synthetic, deliberately messy support-ticket dataset for the
fictional client "QuickMart".

Usage:
    python make_data.py --seed 42             # tuning set  -> data/messy_tickets.csv, data/truth.csv
    python make_data.py --holdout --seed 99   # holdout set -> data/holdout_tickets.csv, data/holdout_truth.csv

The holdout set uses phrasings that were NOT used while tuning the prompt.
"""
import argparse
import csv
import os
import random
from datetime import date

TEMPLATES = {
    "delivery_delay": [
        "my order {oid} has not arrived yet, it was supposed to come 5 days ago",
        "delivery bahut late hai order {oid}, kab aayega??",
        "still waiting for order {oid}. courier not responding",
    ],
    "refund": [
        "i want my money back for order {oid}, product was damaged",
        "refund nahi mila abhi tak order {oid} ka, 10 din ho gaye",
        "please refund order {oid}. wrong item delivered",
    ],
    "product_quality": [
        "the phone charger from order {oid} stopped working in 2 days",
        "shirt ka color fade ho gaya after first wash. order {oid}",
        "packaging was torn and item scratched, {oid}",
    ],
    "payment_issue": [
        "money deducted twice for order {oid} but only one order showing",
        "payment failed but amount cut from my account, {oid}",
        "UPI payment stuck for {oid}, please check",
    ],
    "praise": [
        "great service! order {oid} came early, thank you team",
        "very happy with the product, will buy again ({oid})",
        "delivery boy was very polite. order {oid} perfect",
    ],
    "other": [
        "what are your store timings on sunday?",
        "do you sell laptop bags in the Indore store",
        "how do i change my registered phone number",
    ],
}

# Questions only (no problem reported); many mention an order ID
HARD_OTHER = [
    "can i change the delivery address for order {oid} before it ships?",
    "how do i download the invoice for order {oid}",
    "is order {oid} eligible for gift wrapping, or is it too late to add?",
    "kya order {oid} ka delivery address badal sakte hain?",
    "i placed order {oid} yesterday. can i also add one more item to it?",
    "do you have cash on delivery for orders above 10000?",
    "store timings for Bhopal branch during Diwali week?",
    "i want to become a franchise partner, who should i contact",
]

# Complaint + unrelated question. Label = the complaint.
HARD_MIXED = [
    ("delivery_delay", "order {oid} is 6 days late. also, do you have a store in Nagpur?"),
    ("refund", "how do i return order {oid}? item is damaged and i need a refund"),
    ("payment_issue", "amount was deducted twice for {oid}. btw what are your sunday timings?"),
    ("product_quality", "charger from {oid} stopped working. do you sell extra cables too?"),
    ("delivery_delay", "{oid} abhi tak nahi aaya, bahut late. and can i change my phone number?"),
]

# Holdout phrasings: different style, longer sentences, more Hinglish
HOLDOUT = {
    "delivery_delay": [
        "bhai order {oid} ko 8 din ho gaye, courier wala phone hi nahi uthata",
        "Ordered on the 3rd, tracking shows no movement since last week. Order {oid}. Very frustrating.",
        "my parcel {oid} is stuck at the hub and nobody is telling me anything",
    ],
    "refund": [
        "bhai mera paisa kab lautega, order {oid} 8 din pehle cancel kiya tha",
        "I returned order {oid} last week and the refund still hasn't hit my account",
        "cancelled {oid} but no money back yet, please process the refund",
    ],
    "product_quality": [
        "received {oid} and the zip on the jacket broke on day one",
        "headphones from {oid} have no sound in the left ear",
        "{oid} ka mixer grinder chalte hi garam ho jata hai, quality bekar hai",
    ],
    "payment_issue": [
        "card charged 2499 for {oid} but the app says payment pending",
        "{oid} ke liye paisa kat gaya lekin order confirm nahi hua",
        "tried net banking for {oid}, got an error twice and my balance dropped",
    ],
    "praise": [
        "Just wanted to say the packaging for {oid} was excellent, well done",
        "shukriya team, {oid} time se pehle pahunch gaya",
        "support agent solved my issue in minutes, really impressed",
    ],
    "other": [
        "what documents do i need to open a corporate account with you",
        "can i pick up an online order from the Indore store instead of home delivery",
        "is there a loyalty program for frequent buyers",
        "do you offer bulk discounts for schools",
    ],
}

HOLDOUT_MIXED = [
    ("delivery_delay", "order {oid} still not here after a week. by the way, do you ship to Nagpur?"),
    ("refund", "mujhe {oid} ka refund chahiye, item damaged tha. aur kya wahi model stock mein hai?"),
    ("payment_issue", "money got deducted twice for {oid}. also what is the warranty on this phone?"),
]

CITIES = ["Jabalpur", "jabalpur", "JABALPUR", "Jbp", "Bhopal", "bhopal ",
          "Indore", "indore", "Nagpur", "nagpur", ""]
NAMES = ["Ravi Sharma", "priya verma", "AMIT PATEL", "Sneha Rao", "  Karan Singh ",
         "Neha Gupta", "rahul jain", "", "Pooja Mishra", "Vikram Thakur"]
DATE_FMTS = ["%d/%m/%Y", "%Y-%m-%d", "%d-%b-%y", "%b %d, %Y"]


def messy_amount(n):
    return random.choice([
        f"Rs. {n:,}", f"{n}/-", f"₹{n}", str(n), f"{n} rupees", "", "N/A",
    ])


def messy_date():
    day = random.randint(1, 28)
    month = random.randint(1, 9)
    d = date(2026, month, day)
    if random.random() < 0.08:
        return random.choice(["yesterday", "", "last week"])
    return d.strftime(random.choice(DATE_FMTS))


def add_noise(t):
    r = random.random()
    if r < 0.15:
        return t.upper()
    if r < 0.30:
        return t + "!!!"
    if r < 0.45:
        words = t.split()
        i = random.randrange(len(words))
        w = words[i]
        if len(w) > 3:
            j = random.randrange(len(w) - 1)
            w = w[:j] + w[j + 1] + w[j] + w[j + 2:]
        words[i] = w
        return " ".join(words)
    if r < 0.55:
        return "  " + t + "   pls hurry"
    return t


def new_oid():
    return f"QM{random.randint(10000, 99999)}"


def build_tuning():
    rows, truth = [], []
    for i in range(70):
        cat = random.choice(list(TEMPLATES))
        oid = new_oid()
        text = add_noise(random.choice(TEMPLATES[cat]).format(oid=oid))
        tid = f"T{1001 + i}"
        rows.append({
            "ticket_id": tid,
            "date": messy_date(),
            "customer": random.choice(NAMES),
            "city": random.choice(CITIES),
            "amount": messy_amount(random.choice([499, 1200, 2599, 850, 15000, 320])),
            "message": text,
        })
        truth.append({"ticket_id": tid, "true_category": cat, "group": "base"})

    # duplicates with slightly different casing
    for r in random.sample(rows, 8):
        dup = dict(r)
        dup["ticket_id"] = dup["ticket_id"].lower()
        rows.append(dup)

    random.shuffle(rows)

    next_id = 1071

    def add_hard(text, cat, group):
        nonlocal next_id
        tid = f"T{next_id}"
        next_id += 1
        rows.append({
            "ticket_id": tid,
            "date": messy_date(),
            "customer": random.choice(NAMES),
            "city": random.choice(CITIES),
            "amount": messy_amount(random.choice([499, 1200, 2599, 850, 15000, 320])),
            "message": add_noise(text),
        })
        truth.append({"ticket_id": tid, "true_category": cat, "group": group})

    for i in range(16):
        t = HARD_OTHER[i % len(HARD_OTHER)]
        add_hard(t.format(oid=new_oid()), "other", "hard_other")
    for i in range(10):
        cat, t = HARD_MIXED[i % len(HARD_MIXED)]
        add_hard(t.format(oid=new_oid()), cat, "hard_mixed")

    random.shuffle(rows)
    return rows, truth


def build_holdout():
    rows, truth = [], []
    n = 1

    def add(text, cat, group):
        nonlocal n
        tid = f"H{n:03d}"
        n += 1
        rows.append({
            "ticket_id": tid,
            "date": messy_date(),
            "customer": random.choice(NAMES),
            "city": random.choice(CITIES),
            "amount": messy_amount(random.choice([499, 1200, 2599, 850, 15000, 320])),
            "message": add_noise(text),
        })
        truth.append({"ticket_id": tid, "true_category": cat, "group": group})

    for _ in range(28):
        cat = random.choice(list(HOLDOUT))
        add(random.choice(HOLDOUT[cat]).format(oid=new_oid()), cat, "holdout_single")
    for i in range(6):
        cat, t = HOLDOUT_MIXED[i % len(HOLDOUT_MIXED)]
        add(t.format(oid=new_oid()), cat, "holdout_mixed")

    random.shuffle(rows)
    return rows, truth


def write_csv(path, fieldnames, records):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--holdout", action="store_true",
                        help="generate the untouched holdout set instead")
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs("data", exist_ok=True)

    if args.holdout:
        rows, truth = build_holdout()
        tickets_path, truth_path = "data/holdout_tickets.csv", "data/holdout_truth.csv"
    else:
        rows, truth = build_tuning()
        tickets_path, truth_path = "data/messy_tickets.csv", "data/truth.csv"

    write_csv(tickets_path, list(rows[0].keys()), rows)
    write_csv(truth_path, ["ticket_id", "true_category", "group"], truth)
    print(f"Wrote {tickets_path} ({len(rows)} rows) and {truth_path} (seed={args.seed})")
