"""
Day 3 Experiment: Embedding Semantic Matching Demo

Demonstrates:
  1. Keyword matching — exact word overlap between JD requirements and resume
  2. Semantic matching — embedding similarity captures meaning beyond keywords
  3. Why embedding matters for CareerAgent matching

Key concept: Two texts can describe the same skill using different words.
Keyword matching misses these; embedding similarity captures them.

Usage:
    docker compose exec backend python experiments/day3_embedding_demo.py
"""

import sys
import os

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services.embedding import EmbeddingService


# ── Sample Data ─────────────────────────────────────────────────────────

# Job requirement snippets
JD_REQUIREMENTS = [
    "Experience building real-time data processing systems",
    "Strong knowledge of container orchestration and deployment",
    "Ability to design fault-tolerant distributed architectures",
    "Proficiency in a systems programming language",
    "Experience with infrastructure as code",
]

# Resume skill descriptions (semantically equivalent but different wording)
RESUME_SKILLS = [
    "Built streaming data pipelines with Apache Kafka and Flink",
    "Managed production Kubernetes clusters with Helm and Istio",
    "Designed highly available microservices with circuit breakers",
    "Wrote performance-critical services in Rust and Go",
    "Automated cloud provisioning using Terraform and Ansible",
]

# Keyword overlap pairs (for direct comparison)
KEYWORD_PAIRS = [
    ("Python", "Python"),           # exact match
    ("Docker", "containers"),       # related but different words
    ("Kubernetes", "K8s"),          # abbreviation
    ("machine learning", "deep learning"),  # related field
    ("REST API", "GraphQL"),        # different API paradigms
    ("PostgreSQL", "MongoDB"),      # different database types
]


def main():
    print("=" * 65)
    print("Day 3 Experiment: Embedding Semantic Matching Demo")
    print("=" * 65)

    svc = EmbeddingService()
    print(f"\nModel: {svc.model_name}")
    print(f"Embedding dimension: {svc.encode('test').shape[1]}")

    # ── Part 1: Keyword vs Semantic Matching ──────────────────────────

    print("\n" + "─" * 65)
    print("[Part 1] JD Requirements vs Resume Skills — Semantic Matching")
    print("─" * 65)

    print(f"\n  {'JD Requirement':55s} {'Best Resume Match':40s} {'Score':>6}")
    print(f"  {'-'*55} {'-'*40} {'-'*6}")

    for jd_req in JD_REQUIREMENTS:
        similarities = svc.batch_similarity(jd_req, RESUME_SKILLS)
        best_idx = max(range(len(similarities)), key=lambda i: similarities[i])
        best_score = similarities[best_idx]
        best_match = RESUME_SKILLS[best_idx]
        print(f"  {jd_req[:54]:54s} → {best_match[:38]:38s}  {best_score:.3f}")

    # ── Part 2: Direct similarity comparison ──────────────────────────

    print("\n" + "─" * 65)
    print("[Part 2] Keyword Pairs — Cosine Similarity")
    print("─" * 65)

    print(f"\n  {'Term A':25s} {'Term B':25s} {'Similarity':>10}")
    print(f"  {'-'*25} {'-'*25} {'-'*10}")
    for term_a, term_b in KEYWORD_PAIRS:
        sim = svc.similarity(term_a, term_b)
        bar = "█" * int(sim * 20)
        print(f"  {term_a:25s} {term_b:25s} {sim:>8.3f}  {bar}")

    # ── Part 3: Keyword overlap limitation ────────────────────────────

    print("\n" + "─" * 65)
    print("[Part 3] Where keyword matching fails")
    print("─" * 65)

    jd_phrase = "deploy containerized microservices"
    resume_phrase = "ship Docker-based distributed apps with K8s"

    # Keyword overlap
    jd_words = set(jd_phrase.lower().split())
    resume_words = set(resume_phrase.lower().split())
    overlap = jd_words & resume_words

    # Semantic similarity
    semantic_sim = svc.similarity(jd_phrase, resume_phrase)

    print(f"\n  JD text:      {jd_phrase}")
    print(f"  Resume text:  {resume_phrase}")
    print(f"\n  Keyword overlap:      {overlap} ({len(overlap)} tokens)")
    print(f"  Semantic similarity:  {semantic_sim:.3f}")
    print(f"\n  The keyword overlap is near zero, but semantically these")
    print(f"  phrases describe the same thing. Embedding catches this.")

    # ── Part 4: Embedding vector inspection ───────────────────────────

    print("\n" + "─" * 65)
    print("[Part 4] Embedding vector sample")
    print("─" * 65)

    emb = svc.encode("machine learning engineer")
    print(f"\n  Text: 'machine learning engineer'")
    print(f"  Embedding shape: {emb.shape}")
    print(f"  First 10 dimensions: {emb[0, :10]}")
    print(f"  L2 norm: {np.linalg.norm(emb[0]):.4f}")
    print(f"  Mean: {emb.mean():.6f}  Std: {emb.std():.6f}")
    print(f"\n  (The embedding is a dense 384-dimensional vector.")
    print(f"   Every dimension captures some aspect of meaning.)")

    # ── Summary ────────────────────────────────────────────────────────

    print("\n" + "=" * 65)
    print("Experiment complete!")
    print("=" * 65)

    print("""
  Key takeaways:
  ┌────────────────────────────────────────────────────────────────┐
  │ 1. Keyword matching only finds exact word overlap.             │
  │    "Docker" != "containers" — keyword fails, embedding works.  │
  │                                                                │
  │ 2. Embedding captures semantic similarity.                     │
  │    "deploy containerized microservices" and                     │
  │    "ship Docker-based distributed apps" mean the same thing.   │
  │                                                                │
  │ 3. Cosine similarity ranges from 0 (unrelated) to 1 (same).   │
  │    - Exact match (Python/Python):     ~0.90+                   │
  │    - Related (K8s/Kubernetes):        ~0.70+                   │
  │    - Different paradigms (REST/GraphQL): ~0.50+                │
  │                                                                │
  │ 4. This is why CareerAgent needs embeddings (Day 5 vector      │
  │    matching) — to find relevant resume entries even when       │
  │    the candidate uses different words than the JD.             │
  └────────────────────────────────────────────────────────────────┘
    """)


if __name__ == "__main__":
    main()
