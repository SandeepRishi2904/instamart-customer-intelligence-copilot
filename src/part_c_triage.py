"""Part C: support-ticket triage with guardrails around the offline mock_llm.
Run from the repository root (after src/part_b_segments.py): python src/part_c_triage.py"""
import json
import os
import random
import re
import sys
import time

import pandas as pd

sys.path.insert(0, os.getcwd())
import mock_llm  # noqa: E402  (supplied file, used unmodified)
from mock_llm import TransientLLMError, est_tokens  # noqa: E402

MAX_INPUT_TOKENS = 400
MAX_RETRIES = 3
BASE_DELAY = 0.05
CATEGORIES = ["billing", "delivery", "technical", "fraud"]

# redaction patterns, applied in exactly this order
PATTERNS = [("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
            ("phone", re.compile(r"\b[6-9]\d{9}\b"), "[PHONE]"),
            ("order_id", re.compile(r"ORD-\d{6}"), "[ORDER_ID]")]

ZERO_SHOT = ("You CLASSIFY customer support tickets into exactly one category: billing, delivery, technical or fraud. "
             'Reply with JSON only, in the form {"category": "...", "reason": "..."}.')
FEW_SHOT = ZERO_SHOT + """
EXAMPLES:
Ticket: I was charged twice for [ORDER_ID]. Please refund the extra payment. -> {"category": "billing", "reason": "double charge"}
Ticket: The rider marked [ORDER_ID] as delivered but I never received the package. -> {"category": "delivery", "reason": "package not received"}
Ticket: The app crashes every time I open the cart. -> {"category": "technical", "reason": "app crash"}
Ticket: There is an unauthorized transaction on my card linked to [ORDER_ID]. I did not place it. -> {"category": "fraud", "reason": "unauthorized transaction"}"""

RETRY_LOG = []
STATS = {"transient_errors": 0}


def redact(text, counts=None):
    for name, pat, placeholder in PATTERNS:
        text, n = pat.subn(placeholder, text)
        if counts is not None:
            counts[name] += n
    return text


def call_with_retry(ticket_id, messages):
    """Every model call goes through here: up to 3 retries with exponential backoff and jitter."""
    for n in range(MAX_RETRIES + 1):
        try:
            return mock_llm.chat(messages)
        except TransientLLMError:
            STATS["transient_errors"] += 1
            if n == MAX_RETRIES:
                raise
            wait = BASE_DELAY * 2 ** n * random.uniform(0.9, 1.15)
            RETRY_LOG.append({"ticket_id": ticket_id, "retry": n, "wait": wait})
            time.sleep(wait)


def validate(text):
    """Returns (status, category). Never guesses."""
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        words = re.findall(r"[a-z]+", (text or "").lower())
        found = {w for w in words if w in CATEGORIES}
        if len(found) == 1:
            return "classified_keyword", found.pop()
        return "flagged_invalid_output", ""
    cat = data.get("category") if isinstance(data, dict) else None
    if cat in CATEGORIES:
        return "classified_json", cat
    return "flagged_invalid_output", ""


def run_pipeline(tickets, system_prompt, token_budget=None):
    mock_llm.reset()
    RETRY_LOG.clear()
    STATS["transient_errors"] = 0
    total_tokens, rows = 0, []
    for t in tickets.itertuples(index=False):
        row = {"ticket_id": t.ticket_id, "customer_id": t.customer_id, "status": "", "category": "", "tokens_used": 0}
        text = t.text
        if not text.strip():
            row["status"] = "skipped_empty"
        elif est_tokens(text) > MAX_INPUT_TOKENS:
            row["status"] = "flagged_long"
        elif token_budget is not None and total_tokens >= token_budget:
            row["status"] = "budget_exhausted"
        else:
            user = redact(text)
            assert not any(p.search(user) for _, p, _ in PATTERNS), "personal data reached the prompt"
            resp = call_with_retry(t.ticket_id, [{"role": "system", "content": system_prompt},
                                                 {"role": "user", "content": user}])
            used = resp.usage.total_tokens
            total_tokens += used
            row["tokens_used"] = used
            row["status"], row["category"] = validate(resp.choices[0].message.content)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(name, res, tickets):
    truth = tickets.set_index("ticket_id")["true_category"]
    ok = res[res["status"].isin(["classified_json", "classified_keyword"])]
    correct = int((ok["category"] == ok["ticket_id"].map(truth)).sum())
    tokens = int(res["tokens_used"].sum())
    print(f"\n[{name}] status counts: {res['status'].value_counts().to_dict()}")
    print(f"[{name}] classified {len(ok)}; accuracy {correct / len(ok) * 100:.1f}% ({correct}/{len(ok)}); "
          f"total tokens {tokens:,}; tokens per classified ticket {tokens / len(ok):.1f}")
    print(f"[{name}] transient errors {STATS['transient_errors']}, retries logged {len(RETRY_LOG)}")
    return ok


def route(res, segments):
    seg = segments.set_index("customer_id")
    out = res.copy()
    out["segment_name"] = out["customer_id"].map(seg["segment_name"]).fillna("")
    premium = out["customer_id"].map(seg["is_premium"]).fillna(False).astype(bool)
    human = out["status"].isin(["skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted"])
    out["route"] = "support_queue"
    out["priority"] = "NORMAL"
    fraud = ~human & (out["category"] == "fraud")
    prem = ~human & ~fraud & premium
    out.loc[prem, ["route", "priority"]] = ["premium_desk", "HIGH"]
    out.loc[fraud, ["route", "priority"]] = ["security_team", "HIGH"]
    out.loc[human, ["route", "priority"]] = ["human_review", "NORMAL"]
    return out[["ticket_id", "customer_id", "status", "category", "segment_name", "priority", "route", "tokens_used"]]


def main():
    random.seed(42)
    tickets = pd.read_csv("data/raw/tickets.csv", dtype=str, keep_default_na=False)
    segments = pd.read_csv("outputs/customer_segments.csv")
    os.makedirs("outputs", exist_ok=True)

    # guardrail and redaction figures over all 164 tickets
    empty = int((~tickets["text"].str.strip().astype(bool)).sum())
    long_ = tickets[tickets["text"].map(est_tokens) > MAX_INPUT_TOKENS]
    print(f"tickets {len(tickets)}; empty {empty}; long {len(long_)} (estimated tokens {sorted(set(long_['text'].map(est_tokens)))})")
    counts = {"email": 0, "phone": 0, "order_id": 0}
    redacted = [redact(t, counts) for t in tickets["text"]]
    left = sum(len(p.findall(t)) for t in redacted for _, p, _ in PATTERNS)
    print(f"redactions: {counts}, total {sum(counts.values())}; patterns still found afterwards: {left}")
    print(f"prompt token estimates: zero-shot {est_tokens(ZERO_SHOT)}, few-shot {est_tokens(FEW_SHOT)}")
    assert "EXAMPLES:" not in ZERO_SHOT and "EXAMPLES:" in FEW_SHOT

    zero = run_pipeline(tickets, ZERO_SHOT)
    ok0 = summarize("zero-shot", zero, tickets)
    waits = [r["wait"] for r in RETRY_LOG]
    print(f"retry waits: min {min(waits):.4f}s, max {max(waits):.4f}s; retry numbers {sorted({r['retry'] for r in RETRY_LOG})}")
    with open("outputs/triage_retry_log.txt", "w") as f:
        for r in RETRY_LOG:
            f.write(f"ticket {r['ticket_id']} retry {r['retry']} wait {r['wait']:.4f}s\n")
    truth = tickets.set_index("ticket_id")["true_category"]
    print("\nZero-shot confusion table (rows true, columns predicted):")
    print(pd.crosstab(ok0["ticket_id"].map(truth), ok0["category"]).to_string())

    few = run_pipeline(tickets, FEW_SHOT)
    summarize("few-shot", few, tickets)

    budget = 6000
    demo = run_pipeline(tickets, ZERO_SHOT, token_budget=budget)
    print(f"\n[budget demo] TOKEN_BUDGET={budget}: budget_exhausted {int((demo['status'] == 'budget_exhausted').sum())}, "
          f"tokens used {int(demo['tokens_used'].sum())}")
    route(demo, segments).to_csv("outputs/triage_budget_demo.csv", index=False)

    final = run_pipeline(tickets, FEW_SHOT, token_budget=100000)
    routed = route(final, segments)
    routed.to_csv("outputs/triage_results.csv", index=False)
    print(f"\nfinal run: {len(routed)} rows; routes {routed['route'].value_counts().to_dict()}; "
          f"priorities {routed['priority'].value_counts().to_dict()}")
    print("status counts:", routed["status"].value_counts().to_dict())


if __name__ == "__main__":
    main()
