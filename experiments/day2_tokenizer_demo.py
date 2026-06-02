"""
Day 2 Experiment: Tokenizer & JD Parser Demo

Demonstrates:
  1. Tokenizer — how text is split into tokens (subword units)
  2. JD Parser — extracting skills, responsibilities, nice_to_haves
  3. Token-level comparison — JD skills vs. resume tokens

Key concept: Tokenizer is the bridge between raw text and model input.
Every word the model "reads" goes through tokenization first.

Usage:
    docker compose exec backend python experiments/day2_tokenizer_demo.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sentence_transformers import SentenceTransformer
from app.services.jd_parser import parse_jd


# ── Sample Data ─────────────────────────────────────────────────────────

JD_TEXT = """
Senior Machine Learning Engineer

at AcmeCorp

We are building next-generation AI products and we need someone who
can bridge the gap between research and production.

Requirements:
- 5+ years experience in machine learning and deep learning
- Proficient in Python, PyTorch, TensorFlow
- Experience with NLP and transformer models
- Strong understanding of RAG systems and vector databases
- Familiar with Docker, Kubernetes, AWS or GCP

Responsibilities:
- Design and implement end-to-end ML pipelines
- Build and deploy RAG-based applications
- Optimize model inference for production
- Mentor junior engineers on ML best practices

Nice to have:
- Experience with MLflow or Kubeflow
- Knowledge of Go or Rust
- Published research papers in top-tier conferences
"""

RESUME_TEXT = """
Machine Learning Engineer with 6 years of experience.

Skills: Python, PyTorch, Docker, Kubernetes, AWS
Experience: Built RAG systems using LangChain and vector databases.
Deployed ML models to production using Docker and Kubernetes.
Worked on NLP projects including text classification and NER.
Familiar with Go and MLflow.
"""


def main():
    print("=" * 65)
    print("Day 2 Experiment: Tokenizer & JD Parser Demo")
    print("=" * 65)

    # ── Part 1: Tokenizer ──────────────────────────────────────────────

    print("\n" + "─" * 65)
    print("[Part 1] Tokenizer — How text becomes tokens")
    print("─" * 65)

    # Load the same model used for embeddings — it includes a tokenizer
    model_name = "all-MiniLM-L6-v2"
    print(f"\nLoading model: {model_name}")
    model = SentenceTransformer(model_name)
    tokenizer = model.tokenizer

    # Tokenize a sample sentence
    sample = "Machine Learning Engineer with 5+ years experience in PyTorch."
    tokens = tokenizer.tokenize(sample)
    token_ids = tokenizer.encode(sample)

    print(f"\n  Input text:  {sample}")
    print(f"  Tokens ({len(tokens)}):")
    # Display tokens in a compact grid
    for i, tok in enumerate(tokens):
        if i > 0 and i % 10 == 0:
            print()
        print(f"    {tok}", end="")
    print()

    print(f"\n  Token IDs ({len(token_ids)}):")
    print(f"    {token_ids}")

    # Show special tokens
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    cls_tok = tokenizer.cls_token
    sep_tok = tokenizer.sep_token
    print(f"\n  Special tokens: [{cls_tok}]={cls_id}, [{sep_tok}]={sep_id}")

    # Show subword breakdown
    print("\n  Subword breakdown examples:")
    examples = ["transformer", "tokenization", "unsupervised", "PyTorch"]
    for word in examples:
        subtokens = tokenizer.tokenize(word)
        print(f"    {word:20s} → {' + '.join(subtokens)}")

    # ── Part 2: JD Parser ──────────────────────────────────────────────

    print("\n" + "─" * 65)
    print("[Part 2] JD Parser — Extracting structured info")
    print("─" * 65)

    jd = parse_jd(JD_TEXT)

    print(f"\n  Title:          {jd.title}")
    print(f"  Company:        {jd.company}")
    print(f"  Skills ({len(jd.skills)}):       {', '.join(jd.skills)}")
    print(f"  Responsibilities ({len(jd.responsibilities)}):")
    for r in jd.responsibilities:
        print(f"    - {r[:80]}")
    print(f"  Nice-to-haves ({len(jd.nice_to_haves)}):")
    for n in jd.nice_to_haves:
        print(f"    - {n[:80]}")

    # ── Part 3: Token coverage — JD skills in resume ────────────────────

    print("\n" + "─" * 65)
    print("[Part 3] Token-level comparison: JD skills vs. Resume")
    print("─" * 65)

    # Tokenize all extracted JD skills
    print("\n  JD skills as tokens:")
    jd_skill_tokens: set[str] = set()
    for skill in jd.skills:
        skill_tokens = set(tokenizer.tokenize(skill))
        jd_skill_tokens.update(skill_tokens)
        # Remove special chars (## prefix for subword continuation)
        clean = {t.replace("##", "") for t in skill_tokens}
        print(f"    {skill:25s} → {', '.join(sorted(clean))}")

    # Tokenize the resume text
    resume_tokens_list = tokenizer.tokenize(RESUME_TEXT)
    resume_tokens_set = set(resume_tokens_list)
    resume_clean = {t.replace("##", "") for t in resume_tokens_set}

    # How many JD skill tokens appear in the resume?
    jd_clean = {t.replace("##", "") for t in jd_skill_tokens}
    overlap = jd_clean & resume_clean

    print(f"\n  Total unique JD skill sub-tokens:   {len(jd_clean)}")
    print(f"  Total unique resume sub-tokens:     {len(resume_clean)}")
    print(f"  Overlapping tokens:                 {len(overlap)}")
    print(f"  Token recall (resume has JD token): {len(overlap)/len(jd_clean)*100:.1f}%")

    if overlap:
        print(f"\n  Overlapping tokens: {sorted(overlap)}")
    missing = jd_clean - resume_clean
    if missing:
        print(f"  Missing tokens:     {sorted(missing)}")

    # ── Summary ────────────────────────────────────────────────────────

    print("\n" + "=" * 65)
    print("Experiment complete!")
    print("=" * 65)

    print("""
  Key takeaways:
  ┌────────────────────────────────────────────────────────────────┐
  │ 1. Tokenizer splits text into subword units (tokens).          │
  │    Example: "transformer" → "transform" + "##er"               │
  │                                                                │
  │ 2. Special tokens ([CLS]=101, [SEP]=102) mark sequence         │
  │    boundaries — the model uses them to understand structure.    │
  │                                                                │
  │ 3. JD Parser extracts 3 types of info:                         │
  │    - Skills (matched against a tech vocabulary)                │
  │    - Responsibilities (from bullet lists)                      │
  │    - Nice-to-haves (from bonus sections)                       │
  │                                                                │
  │ 4. Token overlap between JD and resume gives a first signal    │
  │    of match quality — but it's only lexical, not semantic.     │
  │    That's why we need embeddings (Day 3+).                     │
  └────────────────────────────────────────────────────────────────┘
    """)


if __name__ == "__main__":
    main()
