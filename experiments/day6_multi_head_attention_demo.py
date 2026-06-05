"""
Day 6 Experiment: Multi-Head Attention Demo

Demonstrates:
  1. Why multi-head attention — one head can't capture all relationships
  2. Multiple Q, K, V projections per head
  3. How different heads learn different attention patterns
  4. Concatenation and output projection
  5. Connection to CareerAgent — multi-perspective matching

Key concept: Instead of one attention pattern, multi-head attention runs
several in parallel. Each head can specialize — one for syntax, one for
semantics, one for long-range dependencies. The outputs are concatenated
and projected back to the model dimension.

Usage:
    docker compose exec backend python experiments/day6_multi_head_attention_demo.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute softmax along the given axis (numerically stable)."""
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def multi_head_attention(
    X: np.ndarray,
    W_Q: np.ndarray,
    W_K: np.ndarray,
    W_V: np.ndarray,
    W_O: np.ndarray,
    num_heads: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Compute multi-head attention.

    Args:
        X: Input embeddings of shape (seq_len, d_model).
        W_Q, W_K, W_V: Weight matrices of shape (num_heads, d_model, d_k).
        W_O: Output projection of shape (num_heads * d_v, d_model).
        num_heads: Number of attention heads.

    Returns:
        output: Multi-head attention output of shape (seq_len, d_model).
        head_weights: List of attention weight matrices per head.
    """
    seq_len, d_model = X.shape
    d_k = W_Q.shape[-1]
    d_v = W_V.shape[-1]

    head_outputs = []
    head_weights = []

    for h in range(num_heads):
        Q = np.dot(X, W_Q[h])  # (seq_len, d_k)
        K = np.dot(X, W_K[h])  # (seq_len, d_k)
        V = np.dot(X, W_V[h])  # (seq_len, d_v)

        scores = np.dot(Q, K.T) / np.sqrt(d_k)
        weights = softmax(scores, axis=-1)
        head_out = np.dot(weights, V)

        head_outputs.append(head_out)
        head_weights.append(weights)

    # Concatenate all heads: (seq_len, num_heads * d_v)
    concatenated = np.concatenate(head_outputs, axis=-1)

    # Final linear projection: (seq_len, d_model)
    output = np.dot(concatenated, W_O)

    return output, head_weights


def main():
    print("=" * 65)
    print("Day 6 Experiment: Multi-Head Attention Demo")
    print("=" * 65)

    # ── Setup ───────────────────────────────────────────────────────
    sentence = "I love machine learning"
    tokens = sentence.split()
    seq_len = len(tokens)

    d_model = 12
    num_heads = 3
    d_k = 4  # per-head key/query dim
    d_v = 4  # per-head value dim

    rng = np.random.default_rng(42)
    X = rng.normal(size=(seq_len, d_model)).astype(np.float64)

    # Each head gets its own W_Q, W_K, W_V
    W_Q = rng.normal(size=(num_heads, d_model, d_k)).astype(np.float64)
    W_K = rng.normal(size=(num_heads, d_model, d_k)).astype(np.float64)
    W_V = rng.normal(size=(num_heads, d_model, d_v)).astype(np.float64)
    W_O = rng.normal(size=(num_heads * d_v, d_model)).astype(np.float64)

    # ── Part 1: Why Multi-Head? ─────────────────────────────────────

    print("\n" + "─" * 65)
    print("[Part 1] Why Multi-Head Attention?")
    print("─" * 65)

    print(f"""
  Sentence: "{sentence}"
  Tokens: {seq_len}
  Model dimension: {d_model}
  Number of heads: {num_heads}
  Per-head dimension (d_k = d_v): {d_k}

  Single-head attention (Day 5):
    - One attention pattern for the whole sequence
    - "learning" attends mostly to its nearest neighbor "machine"
    - All relationships are captured by ONE set of Q/K/V weights

  Problem: One attention pattern can't capture everything.
    - Syntactic relationships (verb → subject)
    - Semantic relationships (synonyms, related concepts)
    - Long-range dependencies (pronouns → entities)

  Multi-head attention solves this by running N independent
  attention heads in parallel — each learns different patterns.

  Total Q/K/V parameters: {num_heads} heads × {d_model}×{d_k} each
  = {num_heads * d_model * d_k} params per projection
    """)

    # ── Part 2: Per-Head Projections ────────────────────────────────

    print("─" * 65)
    print("[Part 2] Per-Head Q, K, V Projections")
    print("─" * 65)

    output, head_weights = multi_head_attention(X, W_Q, W_K, W_V, W_O, num_heads)

    for h in range(num_heads):
        Q_h = np.dot(X, W_Q[h])
        print(f"\n  Head {h + 1} — Q matrix (first 3 of {d_k} dims):")
        print(f"  {'Token':>12s} | {'dim 0':>8s} {'dim 1':>8s} {'dim 2':>8s}")
        print(f"  {'-'*12}-+-{'-'*26}")
        for i, token in enumerate(tokens):
            vals = " ".join(f"{Q_h[i, d]:8.4f}" for d in range(min(3, d_k)))
            print(f"  {token:>12s} | {vals}")

    # ── Part 3: Different Heads, Different Patterns ─────────────────

    print("\n" + "─" * 65)
    print("[Part 3] Attention Patterns Per Head")
    print("─" * 65)

    print(f"""
  Each head learns a distinct attention pattern.  Below we show
  which token each query attends to most strongly per head.

  The same sentence, {num_heads} different "views":
    """)

    for h in range(num_heads):
        weights = head_weights[h]
        print(f"\n  Head {h + 1} attention matrix:")
        header = " " * 12 + "".join(f"{t:>10s}" for t in tokens)
        print(f"  {header}")
        print(f"  {'-'*12}+{'-'*40}")
        for i, token in enumerate(tokens):
            bar_str = " ".join(
                f"{'█' * max(1, int(weights[i, j] * 20)):>10s}"
                for j in range(seq_len)
            )
            print(f"  {token:>12s}| {bar_str}")

    print(f"\n  Each token's strongest attention per head:")
    print(f"  {'Token':>12s} | {'Head 1':>10s} {'Head 2':>10s} {'Head 3':>10s}")
    print(f"  {'-'*12}-+-{'-'*32}")
    for i, token in enumerate(tokens):
        best = [
            tokens[int(np.argmax(head_weights[h][i]))] for h in range(num_heads)
        ]
        print(f"  {token:>12s} | {best[0]:>10s} {best[1]:>10s} {best[2]:>10s}")

    # ── Part 4: Concatenation and Output ────────────────────────────

    print("\n" + "─" * 65)
    print("[Part 4] Concatenation and Output Projection")
    print("─" * 65)

    print(f"""
  After computing attention for each head:

  Step 1: Concatenate head outputs
    head_1_out: (seq_len, {d_v})
    head_2_out: (seq_len, {d_v})
    head_3_out: (seq_len, {d_v})
    → concatenated: (seq_len, {num_heads * d_v})

  Step 2: Linear projection back to d_model
    output = concatenated @ W_O
    W_O shape: ({num_heads * d_v}, {d_model})
    → final output: (seq_len, {d_model})

  This projection lets the model mix information from all heads.
    """)

    print(f"  Final multi-head output (first 4 of {d_model} dims):")
    print(f"  {'Token':>12s} | {'dim 0':>8s} {'dim 1':>8s} {'dim 2':>8s} {'dim 3':>8s}")
    print(f"  {'-'*12}-+-{'-'*35}")
    for i, token in enumerate(tokens):
        vals = " ".join(f"{output[i, d]:8.4f}" for d in range(4))
        print(f"  {token:>12s} | {vals}")

    # ── Part 5: Multi-Head in CareerAgent ───────────────────────────

    print("\n" + "─" * 65)
    print("[Part 5] Why Multi-Head Matters for CareerAgent")
    print("─" * 65)

    print("""
  Multi-head attention enables multiple matching perspectives:

  Perspective 1 — Exact skill matching:
    "Python" in JD ←→ "Python" in resume  (exact match)

  Perspective 2 — Semantic skill matching:
    "deep learning" in JD ←→ "neural networks" in resume

  Perspective 3 — Experience alignment:
    "Build scalable APIs" in JD ←→ "Designed REST services" in resume

  Each "head" in our matching system can be thought of as a different
  similarity lens.  Day 5's VectorMatcher with its two thresholds
  (skill_threshold=0.55, experience_threshold=0.50) is a simplified
  version of this — different criteria for different match types.

  In Week 3, when we add an LLM, multi-head attention will be the
  mechanism that lets the model simultaneously consider:
    - Skill overlap
    - Experience relevance
    - Seniority level match
    - Industry fit
    ...all in one forward pass.
    """)

    # ── Summary ────────────────────────────────────────────────────

    print("─" * 65)
    print("Experiment complete!")
    print("─" * 65)

    print("""
  Key takeaways:
  ┌────────────────────────────────────────────────────────────────┐
  │ 1. Single-head attention has ONE pattern — it can't capture    │
  │    syntax, semantics, AND long-range dependencies at once.     │
  │                                                                │
  │ 2. Multi-head attention runs N independent attention heads     │
  │    in parallel, each with its own W_Q, W_K, W_V weights.       │
  │                                                                │
  │ 3. Different heads learn different patterns: some focus on     │
  │    local context, some on semantic relationships, some on      │
  │    distant token connections.                                   │
  │                                                                │
  │ 4. Head outputs are concatenated and projected back to the     │
  │    model dimension — the model learns to mix perspectives.     │
  │                                                                │
  │ 5. In CareerAgent, this maps to multi-perspective matching:    │
  │    exact skills, semantic skills, experience, seniority —      │
  │    each "head" provides a different matching lens.             │
  └────────────────────────────────────────────────────────────────┘
    """)


if __name__ == "__main__":
    main()
