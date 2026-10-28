"""Part D, tasks 5-8: tools, ReAct loop, stop-safely limits, checkpoints and resume. All model calls use mock_llm."""
import json
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.getcwd())
import mock_llm  # noqa: E402  (supplied file, used unmodified)
import rag  # noqa: E402

DB = "data/instamart.db"
LEDGER = "outputs/refund_ledger.jsonl"
STATE_DIR = "state"

SYSTEM_PROMPT = ("You are a refund AGENT for customer support. Use the tools get_order, search_policy, calculate_refund, "
                 "issue_refund and escalate_to_human. When finished, reply with STATUS: short explanation.")

_CURRENT_RUN = {"id": None}


class SimulatedCrash(Exception):
    """Raised on purpose to demonstrate resuming from a checkpoint."""


# ------------------------------------------------------------------ tools
def get_order(order_id):
    con = sqlite3.connect(DB)
    row = con.execute("SELECT order_id, status, order_amount, delivery_minutes FROM orders WHERE order_id = ?",
                      (order_id,)).fetchone()
    con.close()
    if row is None:
        return {"error": "order not found"}
    minutes = None if row[3] is None or row[3] != row[3] else row[3]
    return {"order_id": row[0], "status": row[1], "order_amount": row[2], "delivery_minutes": minutes}


def search_policy(query):
    return {"results": [{"source": r["source"], "score": round(r["score"], 3), "text": r["text"]}
                        for r in rag.retrieve(query, k=3)]}


def calculate_refund(order_amount, delivery_minutes):
    late = delivery_minutes is not None and delivery_minutes > 45
    return {"refund_inr": round(float(order_amount), 2) if late else 0}


def _ledger(tool, order_id, refund_inr):
    os.makedirs(os.path.dirname(LEDGER), exist_ok=True)
    line = {"timestamp": datetime.now(timezone.utc).isoformat(), "tool": tool, "run_id": _CURRENT_RUN["id"],
            "order_id": order_id, "refund_inr": refund_inr}
    with open(LEDGER, "a", encoding="utf-8") as f:
        f.write(json.dumps(line) + "\n")
    return {"refund_inr": refund_inr}


def issue_refund(order_id, refund_inr):
    return _ledger("issue_refund", order_id, refund_inr)


def escalate_to_human(order_id, refund_inr):
    return _ledger("escalate_to_human", order_id, refund_inr)


TOOLS_SPEC = [
    {"name": "get_order", "description": "Look up an order in the database by its order id.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}},
    {"name": "search_policy", "description": "Search the policy documents and return the three best matching chunks.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    {"name": "calculate_refund", "description": "Compute the refund in INR from the order amount and delivery minutes.",
     "parameters": {"type": "object", "properties": {"order_amount": {"type": "number"},
                                                      "delivery_minutes": {"type": "number"}},
                    "required": ["order_amount", "delivery_minutes"]}},
    {"name": "issue_refund", "description": "Issue an automatic refund of up to INR 1000 and record it in the ledger.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}},
                    "required": ["order_id", "refund_inr"]}},
    {"name": "escalate_to_human", "description": "Send a refund above INR 1000 to a human lead and record it.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}},
                    "required": ["order_id", "refund_inr"]}},
]
TOOL_MAP = {"get_order": get_order, "search_policy": search_policy, "calculate_refund": calculate_refund,
            "issue_refund": issue_refund, "escalate_to_human": escalate_to_human}


# ------------------------------------------------------------------ loop
def _checkpoint(state):
    os.makedirs(STATE_DIR, exist_ok=True)
    with open(os.path.join(STATE_DIR, f"{state['run_id']}.json"), "w", encoding="utf-8") as f:
        json.dump(state, f, indent=1)


def _result(state, status, final_answer):
    refund = 0.0
    for s in state["steps"]:
        if s["tool"] in ("issue_refund", "escalate_to_human"):
            refund = s["observation"]["refund_inr"]
    return {"run_id": state["run_id"], "scenario_id": state["scenario_id"], "status": status, "final_answer": final_answer,
            "steps": state["steps"], "llm_calls": state["llm_calls"], "tokens_used": state["tokens_used"],
            "refund_inr": refund, "tools_executed": list(state["tools_executed"])}


def _fallback(state, status, reason):
    done = ", ".join(state["tools_executed"]) or "none"
    summary = (f"Handing over to a human agent: {reason}. Scenario {state['scenario_id']}; tools run so far: {done}; "
               f"request: {state['user_message']}")
    return _result(state, status, summary)


def _loop(state, crash_after_tools=None):
    _CURRENT_RUN["id"] = state["run_id"]
    start = time.monotonic()
    while state["llm_calls"] < state["max_iterations"]:
        if time.monotonic() - start >= state["time_limit_seconds"]:
            return _fallback(state, "FALLBACK_TIMEOUT", "time limit reached")
        resp = mock_llm.chat(state["messages"], tools=TOOLS_SPEC)
        state["llm_calls"] += 1
        state["tokens_used"] += resp.usage.total_tokens
        if state["tokens_used"] > state["token_limit"]:
            return _fallback(state, "FALLBACK_TOKEN_LIMIT", "token limit exceeded")
        msg = resp.choices[0].message
        if not msg.tool_calls:
            answer = msg.content or ""
            return _result(state, answer.split(":")[0].strip(), answer)
        call = msg.tool_calls[0]
        state["messages"].append({"role": "assistant", "content": None,
                                  "tool_calls": [{"id": call.id, "name": call.name, "arguments": call.arguments}]})
        observation = TOOL_MAP[call.name](**call.arguments)
        state["tools_executed"].append(call.name)
        state["steps"].append({"tool": call.name, "arguments": call.arguments, "observation": observation})
        state["messages"].append({"role": "tool", "tool_call_id": call.id, "name": call.name,
                                  "content": json.dumps(observation)})
        state["step"] = len(state["steps"])
        _checkpoint(state)
        if crash_after_tools is not None and len(state["tools_executed"]) == crash_after_tools:
            raise SimulatedCrash(f"simulated crash after tool {crash_after_tools}")
    return _fallback(state, "FALLBACK_MAX_ITERATIONS", f"no final answer after {state['max_iterations']} model calls")


def run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000, time_limit_seconds=10,
              run_id=None, crash_after_tools=None):
    state = {"run_id": run_id or f"{scenario_id}-run", "scenario_id": scenario_id, "user_message": user_message,
             "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_message}],
             "tokens_used": 0, "step": 0, "llm_calls": 0, "steps": [], "tools_executed": [],
             "max_iterations": max_iterations, "token_limit": token_limit, "time_limit_seconds": time_limit_seconds}
    return _loop(state, crash_after_tools)


def resume_agent(run_id):
    with open(os.path.join(STATE_DIR, f"{run_id}.json"), encoding="utf-8") as f:
        state = json.load(f)
    return _loop(state)
