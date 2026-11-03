# Instamart Customer Intelligence & Support Copilot

## Overview

One repository, four connected Parts, all on synthetic data. Part A cleans the raw exports and stores them in SQLite. Part B segments customers and predicts lapse. Part C triages support tickets. Part D adds a policy RAG assistant and a refund agent. All amounts are in INR.

## Reproduce

```bash
python -m venv .venv
source .venv/bin/activate
pip install pandas numpy scikit-learn joblib
pip freeze > requirements.txt
python generate_data.py            # last line: generated: customers=1200 products=40 orders_raw=13084 tickets=164
python src/part_a_audit.py         # Part A audit figures
python src/part_a_clean.py         # cleaning, data/clean/*.csv, data/instamart.db
python src/validate.py             # must print ALL CHECKPOINTS PASSED
python src/part_a_sql.py           # runs the seven queries in sql/queries.sql
python src/part_a_pandas.py        # pivot cross-check and exploratory statistics
python src/part_b_segments.py      # RFM, K-means, PCA, outputs/customer_segments.csv
python src/part_b_lapse.py         # lapse model, outputs/lapse_model.joblib, outputs/retention_list.csv
python src/part_c_triage.py        # ticket triage, outputs/triage_results.csv, outputs/triage_budget_demo.csv
python src/part_d_run.py           # RAG evaluation, refund agent runs, demos; writes outputs/agent_runs.jsonl, refund_ledger.jsonl, state/
```

To run all the script steps above (from `generate_data.py` onwards) in one command instead, after installing the libraries: `python run_all.py`. It stops at the first failing script.

## Part A

### Audit of the raw data

| Table | Shape | Missing values (count, %) |
|---|---|---|
| customers | 1200 x 5 | age 61 (5.08%), gender 30 (2.50%) |
| products | 40 x 4 | none |
| orders | 13084 x 9 | order_amount 53 (0.41%, the literal "N/A" is read as missing by pandas; 120 entries are non-numeric in all), delivery_minutes 2541 (19.42%) |

| Audit figure | Value |
|---|---|
| raw orders rows | 13,084 |
| exact duplicate rows | 119 |
| rows with status pending | 656 |
| distinct order_id | 12,309 |
| non-numeric order_amount entries | 120 |
| negative delivery_minutes | 106 |
| blank delivery_minutes | 2,541 |
| blank age | 61 |
| age outside 10-100 | 6 |
| distinct raw city spellings | 24 |
| distinct non-blank raw gender spellings | 10 |
| blank gender | 30 |

### Cleaning rules and rows changed

Customers: (a) city stripped and title-cased, leaving 6 cities; (b) gender mapped to male / female / unknown (30 blanks become unknown); (c) 6 ages outside 10-100 set to missing, then all 67 missing ages filled with the city median via groupby/transform; (d) signup_date converted to datetime.

Orders, in this order:
1. order_ts converted to datetime.
2. 119 exact duplicate rows removed.
3. 656 earlier pending records removed by keeping, per order_id, the record with the latest order_ts.
4. 112 unparseable order_amount values (after steps 2-3) set to missing and recomputed as round(price_inr x quantity x (1 - discount_pct/100), 2).
5. 105 negative delivery_minutes set to missing; 774 missing delivery_minutes of completed orders filled with the median of the order's product category among completed orders; the 1,199 cancelled orders keep a missing delivery time because they were never delivered.

Result: 12,309 orders (11,110 completed, 1,199 cancelled), city counts Pune 227, Delhi 210, Bengaluru 197, Chennai 191, Hyderabad 191, Mumbai 184, genders female 610 / male 560 / unknown 30, completed revenue INR 2,998,668.80. `src/validate.py` asserts 14 of these checkpoints and prints `ALL CHECKPOINTS PASSED`.

### SQL results (sql/queries.sql, run against data/instamart.db)

| Query | Result |
|---|---|
| Q1 WHERE then HAVING | 30 customers |
| Q2 city x category revenue | 30 rows, total INR 2,998,668.80, largest Bengaluru / Electronics INR 233,909.45 |
| Q3 above-average price | 11 products |
| Q4 no completed order | 25 customers |
| Q5 CTE above-average spend | 550 customers, average INR 2,552.06 |
| Q6 ranking | top per city: Bengaluru C0082 (12,856.20), Chennai C0930 (10,151.40), Delhi C0291 (13,742.60), Hyderabad C0743 (12,730.05), Mumbai C0693 (14,766.55), Pune C0426 (10,377.10); 8 rows have RANK() different from ROW_NUMBER() |
| Q7 running total | 183 daily rows, last cumulative INR 2,998,668.80 |

RANK() and ROW_NUMBER() differ when customers in a city tie on total spend: tied customers share one rank but ROW_NUMBER() still gives each a distinct number, so every row after the first tie diverges.

### pandas cross-check and exploration

The `pd.pivot_table(..., margins=True)` grand total, INR 2,998,668.80, equals the SQL total.

| City | Sum (INR) | Mean (INR) | Count |
|---|---|---|---|
| Bengaluru | 523,643.40 | 276.47 | 1,894 |
| Chennai | 463,629.15 | 272.72 | 1,700 |
| Delhi | 502,829.80 | 243.62 | 2,064 |
| Hyderabad | 528,589.90 | 284.95 | 1,855 |
| Mumbai | 479,058.50 | 297.00 | 1,613 |
| Pune | 500,918.05 | 252.48 | 1,984 |

Completed order_amount: mean 269.91, median 135.00, skewness 4.61, Q1 64.80, Q3 297.00, IQR 232.20, fences -283.50 and 645.30, 0 outliers below and 897 above.

Correlations over completed orders (order_amount / discount_pct / quantity / delivery_minutes): order_amount vs discount_pct -0.095, order_amount vs quantity +0.087, all others within 0.013 of zero.

**What is this data about?** Six months (April to September 2025) of quick-commerce orders from 1,200 customers in six cities. **Distributions?** Order values are strongly right-skewed: the mean (269.91) is twice the median (135.00). **Outliers?** 897 high-value orders lie above the 645.30 fence, mostly electronics; none are errors, and none lie below. **Relationships?** The strongest correlation, order_amount with discount_pct at -0.095, is very weak, and a correlation of this size does not establish causation.

## Part B

### RFM table (snapshot 2025-10-01 00:00:00, completed orders only)

1,175 customers. Recency median 19 days (min 0, max 180); frequency mean 9.46 (max 33); monetary mean INR 2,552.06 (max INR 14,766.55).

### K-means (StandardScaler on R, F, M; KMeans n_init=10, random_state=42)

| K | Inertia | Silhouette |
|---|---|---|
| 2 | 1820.8 | 0.465 |
| 3 | 1268.2 | 0.445 |
| 4 | 743.1 | 0.526 |
| 5 | 617.4 | 0.480 |
| 6 | 503.2 | 0.482 |
| 7 | 408.5 | 0.416 |
| 8 | 366.5 | 0.412 |

The elbow is at K=4: inertia drops by about 525 from K=3 to K=4 and only by about 125 from K=4 to K=5, and K=4 also has the highest silhouette (0.526). **Final K = 4**, which also gives four campaigns the business can run. Inertia is the sum of squared distances to the assigned centroid, so adding clusters can only shrink it: K=8 is always below K=2, and the extreme case of one cluster per customer gives zero. That fact alone therefore cannot choose K; the elbow, the silhouette and interpretability must.

### Segment profile (cluster numbers are only labels and have no order)

| segment_id | Name | Size | Mean recency (days) | Mean frequency | Mean monetary (INR) | Recommended action |
|---|---|---|---|---|---|---|
| 0 | Occasional Shoppers | 436 | 28.86 | 3.77 | 1,057.62 | Light-touch nudges and first-order-of-the-week discounts |
| 1 | **Premium Big Spenders** | 138 | 24.12 | 6.77 | 7,129.70 | Early access to electronics launches and a dedicated premium desk |
| 2 | Loyal Regulars | 419 | 8.77 | 19.40 | 3,410.02 | Subscription / bundle offers to protect the high frequency |
| 3 | Drifting / At Risk | 182 | 109.04 | 2.20 | 685.99 | Win-back coupon before the next expected order |

Premium is the segment with the highest mean monetary value. PCA with 2 components on the scaled RFM matrix explains 0.634 and 0.218 of the variance, 0.852 in total. `outputs/customer_segments.csv` has columns customer_id, recency, frequency, monetary, segment_id, segment_name, is_premium (138 customers are premium).

### Lapse problem

Cutoff 2025-09-01. Cohort: customers with at least one completed order before the cutoff, 1,156 customers. Label lapsed = 1 when there is no completed order from 2025-09-01 to 2025-09-30: 443 lapsed (38.32%). Features use only orders dated before the cutoff plus the clean customer table: recency, frequency, monetary, average discount_pct, cancellation rate, average delivery_minutes, favourite category, city, age, gender, tenure in days. **No order dated on or after 2025-09-01 entered the features, and the segment labels from the K-means step were not used as features.** Split: `train_test_split(test_size=0.2, stratify=y, random_state=42)` gives 924 training and 232 test rows. Baseline (always "not lapsed"): accuracy 61.6%, lapsed-class F1 0.

### Preprocessing without leakage

The numeric columns are standardised and city, gender and favourite category are one-hot encoded with `drop="first"` (k-1 columns) inside a scikit-learn Pipeline. **No encoder, scaler or resampler saw test rows:** the pipeline is fitted on training rows only, and inside GridSearchCV it is refitted on each fold's training part. No feature column has a missing value (asserted in code). The scaler's learned mean of frequency is 8.115 while the full-cohort mean is 8.029; they differ because the scaler only saw the 924 training rows, and a different subset of customers has a slightly different mean.

### Tuning and test evaluation (lapsed class)

| Model | Best parameters | Best CV F1 |
|---|---|---|
| Logistic regression (C in 0.01, 0.1, 1, 10, 100) | C = 100 | 0.651 |
| Random forest (max_depth in 3, 5, 8; min_samples_leaf in 5, 10, 20) | max_depth 8, min_samples_leaf 10 | 0.681 |

| Test set | Confusion matrix [[TN, FP], [FN, TP]] | Precision | Recall | F1 |
|---|---|---|---|---|
| Baseline | - | 0 | 0 | 0 |
| Logistic regression | [[111, 32], [31, 58]] | 0.644 | 0.652 | 0.648 |
| Random forest | [[113, 30], [31, 58]] | 0.659 | 0.652 | 0.655 |

**Selected: random forest** (test F1 0.655, above the 0.55 requirement). It has the highest cross-validated and test F1 and catches the same 58 lapsing customers with two fewer false alarms. Missing a lapsing customer loses the revenue of that customer's future orders, while wrongly targeting a loyal customer costs only a coupon or a message, so recall matters more than precision here; the two models tie on recall, so the better precision decides. The test set was used only after tuning was finished.

### Class-balance experiment (training set only)

The lapsed training rows were randomly duplicated with replacement (random_state=42) until both classes had 570 rows, and the best logistic regression was refitted. On the unchanged test set:

| Logistic regression | Precision | Recall | F1 |
|---|---|---|---|
| Un-balanced | 0.644 | 0.652 | 0.648 |
| Over-sampled | 0.589 | 0.854 | 0.697 |

Recall moved most (+0.20): the model flags many more lapsing customers (76 of 89 instead of 58), while precision fell by 0.055 because it also flags 53 loyal customers instead of 32. For the retention team this trades more wasted offers for far fewer missed lapsers, which is the right direction when a missed lapser costs more than a coupon.

### Interpretation, persistence and retention list

Top five random-forest importances: frequency 0.280, monetary 0.271, recency 0.115, average delivery minutes 0.061, cancellation rate 0.052. Business reading: how often and how much customers already buy is what separates them, and recency and slow deliveries add warning signals.

Logistic regression coefficients (standardised features): frequency -1.698 (customers who order often rarely lapse), cancellation rate -0.124 (a weak effect in this data), monetary +0.285 (at the same frequency, bigger-ticket customers lapse a little more, since they order rarely).

The selected model was saved with `joblib.dump` to `outputs/lapse_model.joblib`; the reloaded model reproduces the test predictions exactly.

`outputs/retention_list.csv` (customer_id, lapse_probability, actually_lapsed) lists the 10 Premium customers of the test set with the highest predicted lapse probability; 1 of them truly lapsed. The retention list is illustrative and not a leakage-free evaluation, because the Premium flag comes from the whole-window RFM, which includes September, the window that defines the lapse label.

## Part C

`src/part_c_triage.py` processes `data/raw/tickets.csv` in file order (read with `keep_default_na=False`) around the unmodified `mock_llm.py`; `mock_llm.reset()` is called before every full run.

### Input guardrails and redaction

- Empty or whitespace-only text gets `skipped_empty` and is never sent to the model: 4 tickets.
- A text above MAX_INPUT_TOKENS = 400 estimated tokens gets `flagged_long` and is never sent: 4 tickets (each estimates to 734 tokens).
- Redaction, applied in this order before any prompt: email `[\w.+-]+@[\w-]+\.[\w.]+` to `[EMAIL]`, phone `\b[6-9]\d{9}\b` to `[PHONE]`, order id `ORD-\d{6}` to `[ORDER_ID]`. The user message is exactly the redacted text.

| Type | Substitutions over 164 tickets |
|---|---|
| email | 55 |
| phone | 91 |
| order id | 132 |
| total | 278 |

Re-applying the three patterns to the redacted texts finds 0. The pipeline also asserts, before every call, that none of the patterns matches the user message.

### Prompts

Both contain CLASSIFY, the four categories (billing, delivery, technical, fraud) and a JSON-only reply format `{"category": "...", "reason": "..."}`. The zero-shot prompt (33 estimated tokens) has no `EXAMPLES:`; the few-shot prompt (134 estimated tokens) has an `EXAMPLES:` line followed by four labelled examples, including a fraud one.

### Retries and output validation

Every call goes through one retry function: at most 3 retries per ticket, wait before retry n = 0.05 x 2^n x uniform(0.9, 1.15) seconds, one log line per retry (written to `outputs/triage_retry_log.txt`). A full run over the 156 eligible tickets hits 19 TransientLLMErrors, all recovered at retry number 0 with waits between 0.0453 s and 0.0562 s (inside the stated 0.045 to 0.0575 s range); no ticket ends with a permanent error.

Replies are validated, never guessed: valid JSON with an allowed category is `classified_json`; a non-JSON reply is accepted as `classified_keyword` only if `re.findall(r"[a-z]+", text.lower())` contains exactly one allowed category word (so `billing-related` yields `billing`); everything else, including valid JSON with a category outside the list, is `flagged_invalid_output` and goes to human review.

| Status (one full run, either prompt) | Tickets |
|---|---|
| classified_json | 135 |
| classified_keyword | 12 |
| flagged_invalid_output | 9 |
| skipped_empty | 4 |
| flagged_long | 4 |

### Zero-shot versus few-shot (no token budget)

| Run | Classified | Accuracy | Total tokens | Tokens per classified ticket |
|---|---|---|---|---|
| Zero-shot | 147 | 85.7% (126/147) | 9,556 | 65.0 |
| Few-shot | 147 | 100.0% (147/147) | 25,312 | 172.2 |

Zero-shot confusion table (rows true, columns predicted):

| true \ predicted | billing | delivery | fraud | technical |
|---|---|---|---|---|
| billing | 38 | 0 | 0 | 0 |
| delivery | 0 | 36 | 0 | 0 |
| fraud | 11 | 0 | 23 | 0 |
| technical | 10 | 0 | 0 | 29 |

The few-shot prompt costs 107 more tokens per ticket (about 2.6 times as many) and 15,756 more tokens over the run, but it removes all 21 errors. Every zero-shot error is a fraud ticket sent to billing (11) or a technical ticket sent to billing (10); the fraud ones matter most, because they would miss the security team and its 2-hour target. The extra tokens are worth paying when a wrong route is costly, as with fraud, and when ticket volume is moderate; for a high-volume stream of low-risk categories the cheaper prompt may be enough.

### Budget, routing and output

- Budget demo: zero-shot with TOKEN_BUDGET = 6000 stops after 6,017 tokens; **58 tickets end `budget_exhausted`** and go to human review. Saved as `outputs/triage_budget_demo.csv`.
- Final run: few-shot with TOKEN_BUDGET = 100000 (not reached), routed with the segments from Part B: skipped_empty, flagged_long, flagged_invalid_output or budget_exhausted go to `human_review` (NORMAL); fraud goes to `security_team` (HIGH); otherwise a premium customer goes to `premium_desk` (HIGH); everything else goes to `support_queue` (NORMAL). A customer with no segment row is treated as not premium. `outputs/triage_results.csv` has 164 rows with columns ticket_id, customer_id, status, category, segment_name, priority, route, tokens_used.

| Route | Tickets | Priority |
|---|---|---|
| support_queue | 102 | NORMAL |
| security_team | 34 | HIGH |
| premium_desk | 11 | HIGH |
| human_review | 17 (4 empty, 4 long, 9 invalid output) | NORMAL |

Rule-validated: the 147 tickets with status `classified_json` or `classified_keyword` were routed automatically, and in the few-shot run all 147 match the true category. Sent to humans: the 17 tickets above, because they were empty, too long or had an unusable model reply.

## Part D

`src/rag.py` holds chunking, retrieval, answering and evaluation; `src/agent.py` holds the tools, the ReAct loop and resume; `src/part_d_run.py` runs everything. All model calls use `mock_llm.py`; no network call or API key is used.

### Chunking, retrieval and grounded answers

Each policy file is split into 40-word chunks with an overlap of 10 words (start 0, then every 30 words, stopping after the first chunk that reaches the last word). 15 chunks in total: refund 4, delivery 3, escalation 3, payment 3, privacy 2, each stored with its source file name. A `TfidfVectorizer` is fitted on the chunk texts, a query is transformed with the same fitted vectorizer, and chunks are ranked by cosine similarity (`retrieve(query, k=3)` returns dictionaries with source, score, text). For `refund policy late delivery` the top three chunks all come from `refund_policy.txt` (scores 0.272, 0.251, 0.245).

TF (term frequency) measures how often a word occurs in a chunk; IDF (inverse document frequency) down-weights words that occur in many chunks, so rare, informative words count more. A limitation is that TF-IDF only matches the exact words used: two phrasings with the same meaning but different words (for example "money back" and "refund") score as unrelated.

`answer(question)` retrieves three chunks and sends two messages to the mock: a system message containing POLICY ASSISTANT (answer only from the context, otherwise say the information could not be found) and a user message `CONTEXT:`, the three chunks each prefixed with `[<file name>]` on its own line, `QUESTION:`, the question. It returns the model's text and the sources used.

| Evaluation of policy_eval.csv (12 answerable, 2 not covered) | Overlap 10 (15 chunks) | Overlap 0 (13 chunks) |
|---|---|---|
| expected_doc among the top 3 sources | 12/12 | 12/12 |
| answer contains expected_fact | 12/12 | 10/12 |
| fallback "I could not find..." on unanswerable questions | 2/2 | 2/2 |

With zero overlap two questions break. "What delivery time makes a completed order eligible for a full refund?" returns "receives a full refund of the order amount.", because the sentence that holds "more than 45 minutes" is cut across a chunk boundary and its second half sits in a different chunk. "How soon is money returned when a payment fails but is deducted?" returns only "...the amount is returned automatically", because "within 48 hours" falls into the next chunk. The overlap keeps such sentences whole in at least one chunk.

### Tools and the ReAct loop

Five tools with a name, one-sentence description and parameter specification in `TOOLS_SPEC`, mapped by `TOOL_MAP`: `get_order` (reads `data/instamart.db`), `search_policy` (the retrieval above), `calculate_refund` (completed orders delivered in more than 45 minutes get 100% of the order amount, otherwise 0), `issue_refund` and `escalate_to_human` (each appends one JSON line with timestamp and tool name to `outputs/refund_ledger.jsonl`). Refunds above INR 1000 are escalated, not issued. `run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000, time_limit_seconds=10)` starts from a system message containing AGENT, calls the model, executes the requested tool by looking it up in `TOOL_MAP`, appends the assistant and tool messages and loops; the status is the word before the colon in the final answer. One JSON line per scenario run is appended to `outputs/agent_runs.jsonl` (the demo runs go to `outputs/agent_demo_runs.jsonl`).

| Scenario | Status | Refund (INR) | Model calls | Tokens used |
|---|---|---|---|---|
| S1 | REFUND | 128.25 | 5 | 992 |
| S2 | REFUND | 135.00 | 5 | 992 |
| S3 | ESCALATE | 1999.00 | 5 | 979 |
| S4 | DENIED | 0.00 | 4 | 681 |
| S5 | DENIED | 0.00 | 3 | 394 |
| S6 | NOT_FOUND | 0.00 | 2 | 110 |
| S7 | FALLBACK_MAX_ITERATIONS | 0.00 | 6 | 3,114 |

All refund amounts match `expected_refund_inr` within 0.01, and S7 used all six model calls under the default token_limit of 6000.

### Stop safely

- Iterations: S7 stops after 6 model calls without a final answer with `FALLBACK_MAX_ITERATIONS`.
- Tokens: S1 with `token_limit=200` ends with **`FALLBACK_TOKEN_LIMIT`** (3 model calls, 411 tokens, the tool requested by the third call was not executed).
- Time: S1 with `time_limit_seconds=0` ends with **`FALLBACK_TIMEOUT`** (0 model calls, stopped before the first call).

Every fallback returns the steps completed so far and a human-handoff summary in `final_answer`.

### Crash and resume

After every tool observation a checkpoint is written to `state/<run_id>.json` (run id, scenario id, full message list, tokens_used, step number, tools executed, plus the steps, call count and limits). For S1 a simulated crash was raised right after the second tool execution; `resume_agent("S1-crash")` loaded the checkpoint and continued. Tools executed across the crash and the resume, in order: **get_order, search_policy** (before the crash), then **calculate_refund, issue_refund** (after the resume): four executions, none repeated. Final status REFUND with INR 128.25 and 5 model calls in total. `state/` holds 9 checkpoints and the ledger has 4 lines (S1 and S2 issued, S3 escalated, the resumed S1 issued).

### ReAct versus plan-and-execute

ReAct asked the model for a decision after every observation: S1 took 5 model calls and 992 estimated tokens, S4 took 4, S5 took 3, S6 took 2. Most of those tokens are the growing history being re-sent, including the policy chunks. Plan-and-execute would ask once for the whole step list (get_order, search_policy, calculate_refund, issue_refund), so it needs about 1 to 2 model calls and a fraction of the tokens. The difference is what happens at an unexpected observation. With ReAct, NOT_FOUND (S6) ended after 2 calls and the cancelled order (S5) ended after 3, because the model read each observation before choosing the next step. A fixed plan would carry on towards calculate_refund and issue_refund with an error or a cancelled order unless every step has a guard and a replanning call, which costs extra calls exactly in the cases that matter. I would choose ReAct for this refund workflow: money moves at the end, three of the seven scenarios branch or stop early on what an observation says, and the cost of ReAct is about 1,000 tokens per request, which is small next to a wrongly issued refund. The iteration, token and time limits stop it from looping, as S7 shows. If volume grew and the steps stayed fixed, a hybrid (planned happy path, ReAct only on errors) would be the next step.

## Limitations and Next Steps

All four Parts are implemented. The Part C classifier is the keyword-based offline mock, so its 100% few-shot accuracy says the pipeline plumbing works, not that a real model would be perfect. The Part B retention list is illustrative and not a leakage-free evaluation, because the Premium flag is a post-hoc whole-window label computed from RFM that includes September, the month that defines the lapse label. The lapse model is trained on 924 customers from one six-month window of synthetic data, so its scores should not be read as production accuracy.

The Part D assistant answers with the single best-matching sentence of the retrieved chunks (the mock's behaviour), so it cannot combine facts across chunks, and TF-IDF retrieval only matches exact words. The refund agent assumes every request is inside the 7-day window and never checks who is asking, so a production version would verify the customer against the order. Next steps: replace the mock with a hosted model, add embedding retrieval, and run the lapse model on later months to check it holds up.
