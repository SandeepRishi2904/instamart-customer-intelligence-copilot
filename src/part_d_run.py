"""Part D runner: RAG checks, the seven agent scenarios, the stop-safely demos and the crash-and-resume demo.
Run from the repository root: python src/part_d_run.py"""
import glob
import json
import os
import shutil

import pandas as pd

import agent
import rag


def rag_report():
    chunks = rag.chunk_documents(10)
    counts = pd.Series([c["source"] for c in chunks]).value_counts().to_dict()
    print(f"chunks (size 40, overlap 10): {len(chunks)} {counts}")
    top = rag.retrieve("refund policy late delivery")
    print("top 3 for 'refund policy late delivery':", [(h["source"], round(h["score"], 3)) for h in top])
    for ov in (10, 0):
        r = rag.evaluate(ov)
        print(f"\noverlap {ov}: chunks {r['chunks']}; expected doc in top 3: {r['doc_hit']}/12; "
              f"expected fact in answer: {r['fact_hit']}/12; fallback on unanswerable: {r['fallback']}/2")
        for q, why, text in r["failures"]:
            print(f"   FAIL: {q} -> {why}\n         answer: {text}")


def agent_report():
    for p in ("outputs/agent_runs.jsonl", "outputs/agent_demo_runs.jsonl", agent.LEDGER):
        if os.path.exists(p):
            os.remove(p)
    shutil.rmtree(agent.STATE_DIR, ignore_errors=True)

    sc = pd.read_csv("data/raw/agent_scenarios.csv")
    print("\nScenario | status | refund | model calls | tokens | expected")
    with open("outputs/agent_runs.jsonl", "a", encoding="utf-8") as f:
        for r in sc.itertuples(index=False):
            res = agent.run_agent(r.message, r.scenario_id)
            f.write(json.dumps(res) + "\n")
            exp_ok = res["status"].startswith("FALLBACK") if r.expected_outcome == "FALLBACK" else res["status"] == r.expected_outcome
            amt_ok = abs(res["refund_inr"] - r.expected_refund_inr) <= 0.01 if r.expected_outcome in ("REFUND", "ESCALATE") else True
            print(f"{r.scenario_id} | {res['status']} | {res['refund_inr']:.2f} | {res['llm_calls']} | {res['tokens_used']} | "
                  f"{r.expected_outcome} {r.expected_refund_inr:.2f} -> {'OK' if exp_ok and amt_ok else 'MISMATCH'}")

    s1 = sc.iloc[0]
    demos = [("token_limit=200", agent.run_agent(s1.message, "S1", token_limit=200, run_id="S1-tokenlimit")),
             ("time_limit_seconds=0", agent.run_agent(s1.message, "S1", time_limit_seconds=0, run_id="S1-timeout"))]
    print()
    with open("outputs/agent_demo_runs.jsonl", "a", encoding="utf-8") as f:
        for name, res in demos:
            f.write(json.dumps(res) + "\n")
            print(f"S1 with {name}: {res['status']} (model calls {res['llm_calls']}, steps {len(res['steps'])}, "
                  f"tokens {res['tokens_used']})\n   handoff: {res['final_answer']}")

        try:
            agent.run_agent(s1.message, "S1", run_id="S1-crash", crash_after_tools=2)
        except agent.SimulatedCrash as e:
            print(f"\ncrash: {e}")
        ckpt = json.load(open("state/S1-crash.json"))
        print(f"checkpoint tools executed before resume: {ckpt['tools_executed']}")
        res = agent.resume_agent("S1-crash")
        f.write(json.dumps(res) + "\n")
    print(f"resumed run: status {res['status']}, refund {res['refund_inr']:.2f}, tools executed in order: {res['tools_executed']}, "
          f"model calls {res['llm_calls']}")
    print(f"ledger lines: {len(open(agent.LEDGER).readlines())}; checkpoints in state/: {len(glob.glob('state/*.json'))}")


if __name__ == "__main__":
    rag_report()
    agent_report()
