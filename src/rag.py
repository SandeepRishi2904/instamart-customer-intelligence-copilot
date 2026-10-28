"""Part D, tasks 1-4: chunking, TF-IDF retrieval, grounded answers and evaluation over data/policies/."""
import glob
import os
import sys

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.getcwd())
import mock_llm  # noqa: E402  (supplied file, used unmodified)

POLICY_DIR = "data/policies"
CHUNK_WORDS = 40

SYSTEM_PROMPT = ("You are a POLICY ASSISTANT for customer support. Answer only from the supplied context. "
                 "If the answer is not in the context, say that you could not find the information.")


def chunk_documents(overlap=10, size=CHUNK_WORDS, folder=POLICY_DIR):
    """Chunks of `size` words, each starting size-overlap words after the previous one; stops after the first
    chunk whose end reaches or passes the last word."""
    chunks = []
    for path in sorted(glob.glob(os.path.join(folder, "*.txt"))):
        words = open(path, encoding="utf-8").read().split()
        start = 0
        while True:
            chunks.append({"source": os.path.basename(path), "text": " ".join(words[start:start + size])})
            if start + size >= len(words):
                break
            start += size - overlap
    return chunks


class PolicyIndex:
    def __init__(self, overlap=10):
        self.chunks = chunk_documents(overlap)
        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform([c["text"] for c in self.chunks])

    def retrieve(self, query, k=3):
        sims = cosine_similarity(self.vectorizer.transform([query]), self.matrix)[0]
        top = sims.argsort()[::-1][:k]
        return [{"source": self.chunks[i]["source"], "score": float(sims[i]), "text": self.chunks[i]["text"]} for i in top]

    def answer(self, question):
        hits = self.retrieve(question, k=3)
        context = "\n".join(f"[{h['source']}]\n{h['text']}" for h in hits)
        user = f"CONTEXT:\n{context}\nQUESTION:\n{question}"
        resp = mock_llm.chat([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}])
        return resp.choices[0].message.content, [h["source"] for h in hits]


_DEFAULT = None


def default_index():
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = PolicyIndex(overlap=10)
    return _DEFAULT


def retrieve(query, k=3):
    return default_index().retrieve(query, k)


def answer(question):
    return default_index().answer(question)


def evaluate(overlap):
    index = PolicyIndex(overlap)
    qa = pd.read_csv("data/raw/policy_eval.csv", keep_default_na=False)
    doc_hit = fact_hit = fallback = 0
    failures = []
    for r in qa.itertuples(index=False):
        text, sources = index.answer(r.question)
        if r.expected_doc == "NONE":
            fallback += text.startswith("I could not find")
            if not text.startswith("I could not find"):
                failures.append((r.question, "unanswerable question got an answer", text))
        else:
            d, f = r.expected_doc in sources, r.expected_fact in text
            doc_hit += d
            fact_hit += f
            if not (d and f):
                failures.append((r.question, f"doc in top-3: {d}; fact in answer: {f}", text))
    return {"chunks": len(index.chunks), "doc_hit": doc_hit, "fact_hit": fact_hit, "fallback": fallback,
            "failures": failures}
