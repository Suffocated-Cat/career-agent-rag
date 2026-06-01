"""
Day 1 Experiment: Embedding Similarity Demo

Generates embeddings for sample JD and resume texts, computes cosine similarity
to verify that semantically similar texts score higher than dissimilar ones.

Usage:
    docker compose exec app python experiments/day1_embedding_demo.py
"""

import sys
import os

# Ensure app/ is importable when running from experiments/ directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services.embedding import EmbeddingService


def main():
    print("=" * 60)
    print("Day 1 Experiment: Embedding Similarity Demo")
    print("=" * 60)

    # Sample job description
    jd_text = """
    Senior Machine Learning Engineer

    Requirements:
    - 5+ years experience in ML/DL
    - Proficient in Python, PyTorch, TensorFlow
    - Experience with NLP and transformer models
    - Familiar with MLOps tools (Docker, Kubernetes, MLflow)
    - Strong understanding of RAG systems and vector databases

    Responsibilities:
    - Design and implement ML pipelines
    - Build and deploy RAG-based applications
    - Optimize model inference performance
    """

    # Sample resume (strong match — ML Engineer)
    resume_ml = """
    Machine Learning Engineer with 6 years of experience.

    Skills: Python, PyTorch, TensorFlow, Docker, Kubernetes, MLflow
    Experience: Built RAG systems using LangChain and vector databases.
    Deployed ML models to production using Docker and Kubernetes.
    Worked on NLP projects including text classification and NER.
    """

    # Sample resume (weak match — Frontend Developer)
    resume_fe = """
    Frontend Developer with 3 years of experience.

    Skills: JavaScript, React, CSS, HTML, TypeScript
    Experience: Built responsive web applications using React.
    Worked with REST APIs and GraphQL.
    """

    print("\n[1] Initializing EmbeddingService (model: all-MiniLM-L6-v2)...")
    service = EmbeddingService()

    print("\n[2] JD <-> ML Engineer Resume similarity...")
    sim_ml = service.similarity(jd_text, resume_ml)
    print(f"    Cosine similarity: {sim_ml:.4f}")

    print("\n[3] JD <-> Frontend Developer Resume similarity...")
    sim_fe = service.similarity(jd_text, resume_fe)
    print(f"    Cosine similarity: {sim_fe:.4f}")

    print("\n[4] Batch similarity (JD vs both resumes)...")
    similarities = service.batch_similarity(jd_text, [resume_ml, resume_fe])
    for i, sim in enumerate(similarities):
        label = "ML Engineer" if i == 0 else "Frontend Dev"
        print(f"    Resume {i+1} ({label}): {sim:.4f}")

    print("\n[5] Embedding dimension check...")
    embedding = service.encode(jd_text)
    print(f"    JD embedding shape: {embedding.shape}")

    # Assertion: matching resume must score higher than non-matching
    assert sim_ml > sim_fe, (
        f"FAIL: Expected ML Engineer similarity ({sim_ml:.4f}) "
        f"> Frontend Dev similarity ({sim_fe:.4f})"
    )

    print("\n" + "=" * 60)
    print("Experiment complete!")
    print(f"ML Engineer similarity:   {sim_ml:.4f}")
    print(f"Frontend Dev similarity:  {sim_fe:.4f}")
    print(f"Score gap:                {sim_ml - sim_fe:.4f}")
    print("[PASS] Semantic similarity verified ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
