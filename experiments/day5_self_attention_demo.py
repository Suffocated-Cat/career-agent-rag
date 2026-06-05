"""
Day 5 Experiment: Self-Attention Mechanism Demo

Demonstrates:
  1. The concept of self-attention — how tokens "look at" each other
  2. Q, K, V projection — transforming embeddings into query/key/value spaces
  3. Scaled dot-product attention — computing and interpreting attention weights
  4. Attention output — how each token's representation is a weighted blend
  5. Connection to CareerAgent — why attention matters for semantic matching

Key concept: Self-Attention is the core mechanism that lets each word in a
sequence attend to every other word, building context-aware representations.
This is what powers modern semantic matching — instead of exact string
comparison, attention learns which concepts are related.

Usage:
    docker compose exec backend python experiments/day5_self_attention_demo.py
"""

import sys
import os

# Make app/ importable when run directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute softmax along the given axis (numerically stable)."""
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def scaled_dot_product_attention(
    Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Compute scaled dot-product attention.

    Args:
        Q: Query matrix of shape (seq_len, d_k).
        K: Key matrix of shape (seq_len, d_k).
        V: Value matrix of shape (seq_len, d_v).
        mask: Optional mask of shape (seq_len, seq_len), with -inf for masked positions.

    Returns:
        output: Attention output of shape (seq_len, d_v).
        weights: Attention weights of shape (seq_len, seq_len).
    """
    d_k = Q.shape[-1]
    scores = np.dot(Q, K.T) / np.sqrt(d_k)  # (seq_len, seq_len)
    if mask is not None:
        scores = scores + mask
    weights = softmax(scores, axis=-1)
    output = np.dot(weights, V)  # (seq_len, d_v)
    return output, weights


def main():
    print("=" * 65)
    print("Day 5 Experiment: Self-Attention Mechanism Demo")
    print("=" * 65)

    # ── Example setup ────────────────────────────────────────────────
    # Simulated token embeddings for a simple sentence.
    # In a real model these come from an Embedding layer.
    sentence = "I love machine learning"
    tokens = sentence.split()

    # Each token gets a random 8-dim embedding (for visualization clarity)
    d_model = 8
    d_k = 4  # key/query dimension (typically d_model // num_heads)
    d_v = 4  # value dimension

    rng = np.random.default_rng(42)
    embeddings = rng.normal(size=(len(tokens), d_model)).astype(np.float64)

    seq_len = len(tokens)

    # ── Part 1: What is Self-Attention? ──────────────────────────────

    print("\n" + "─" * 65)
    print("[Part 1] What is Self-Attention?")
    print("─" * 65)

    print(f"""
  Sentence: "{sentence}"
  Token count: {seq_len}
  Model dimension (d_model): {d_model}
  Key/Query dimension (d_k): {d_k}

  Self-Attention lets each token "look at" every other token in the
  sequence.  The token for "learning" can attend to "machine" to
  understand it's about AI, not education.

  Without attention, each token embedding is isolated — the model
  has no way to connect "machine" with "learning".

  The three matrices:
    Q (Query): "What am I looking for?"
    K (Key):   "What do I contain?"
    V (Value): "What information do I contribute?"

  Attention = softmax(Q @ K^T / sqrt(d_k)) @ V
    """)

    # Show raw embeddings
    print(f"  Token embeddings (first 4 of {d_model} dims):")
    print(f"  {'Token':>12s} | {'dim 0':>8s} {'dim 1':>8s} {'dim 2':>8s} {'dim 3':>8s}")
    print(f"  {'-'*12}-+-{'-'*35}")
    for i, token in enumerate(tokens):
        vals = " ".join(f"{embeddings[i, d]:8.4f}" for d in range(4))
        print(f"  {token:>12s} | {vals}")

    # ── Part 2: Q, K, V Computation ─────────────────────────────────

    print("\n" + "─" * 65)
    print("[Part 2] Q, K, V Projection")
    print("─" * 65)

    # In practice these are learned weight matrices.
    # Here we use fixed random weights for demonstration.
    W_Q = rng.normal(size=(d_model, d_k)).astype(np.float64)
    W_K = rng.normal(size=(d_model, d_k)).astype(np.float64)
    W_V = rng.normal(size=(d_model, d_v)).astype(np.float64)

    Q = np.dot(embeddings, W_Q)  # (seq_len, d_k)
    K = np.dot(embeddings, W_K)  # (seq_len, d_k)
    V = np.dot(embeddings, W_V)  # (seq_len, d_v)

    print(f"""
  Step 1: Multiply embeddings by weight matrices W_Q, W_K, W_V.

  W_Q shape: ({d_model}, {d_k})    Q = Embeddings @ W_Q → ({seq_len}, {d_k})
  W_K shape: ({d_model}, {d_k})    K = Embeddings @ W_K → ({seq_len}, {d_k})
  W_V shape: ({d_model}, {d_v})    V = Embeddings @ W_V → ({seq_len}, {d_v})

  Q matrix (what each token queries for):
    """)
    print(f"  {'Token':>12s} | {'dim 0':>8s} {'dim 1':>8s} {'dim 2':>8s} {'dim 3':>8s}")
    print(f"  {'-'*12}-+-{'-'*35}")
    for i, token in enumerate(tokens):
        vals = " ".join(f"{Q[i, d]:8.4f}" for d in range(d_k))
        print(f"  {token:>12s} | {vals}")

    # ── Part 3: Attention Weights ────────────────────────────────────

    print("\n" + "─" * 65)
    print("[Part 3] Scaled Dot-Product Attention Weights")
    print("─" * 65)

    # Compute raw scores
    raw_scores = np.dot(Q, K.T) / np.sqrt(d_k)  # (seq_len, seq_len)
    attention_weights = softmax(raw_scores, axis=-1)

    print(f"""
  Step 2: scores = Q @ K^T / sqrt(d_k)
          → Raw similarity between each query and each key.

  Step 3: weights = softmax(scores, axis=-1)
          → Each row sums to 1.0 — a probability distribution over
            which tokens to attend to.

  Attention weight matrix (rows=query, cols=key):
    """)

    # Text-based heatmap
    header = " " * 12 + "".join(f"{t:>10s}" for t in tokens)
    print(f"  {header}")
    print(f"  {'-'*12}+{'-'*40}")
    for i, token in enumerate(tokens):
        weight_str = " ".join(f"{attention_weights[i, j]:10.4f}" for j in range(seq_len))
        bar_str = " ".join(f"{'█' * int(attention_weights[i, j] * 20):>10s}" for j in range(seq_len))
        print(f"  {token:>12s}| {weight_str}")
        print(f"  {'':>12s}| {bar_str}")

    # Highlight the max attention for each token
    print(f"\n  Each token's strongest attention:")
    print(f"  {'Token':>12s} → {'Attends to':>12s}  {'Weight':>8s}")
    print(f"  {'-'*12}-+-{'-'*12}--{'-'*8}")
    for i, token in enumerate(tokens):
        best_j = int(np.argmax(attention_weights[i]))
        print(f"  {token:>12s} → {tokens[best_j]:>12s}  {attention_weights[i, best_j]:8.4f}")

    # ── Part 4: Attention Output ─────────────────────────────────────

    print("\n" + "─" * 65)
    print("[Part 4] Attention Output")
    print("─" * 65)

    output, _ = scaled_dot_product_attention(Q, K, V)

    print(f"""
  Step 4: output = attention_weights @ V

  Each token's output is a weighted blend of ALL value vectors.
  "learning" at position 2 gets:
    - its own value (weight: {attention_weights[2, 2]:.4f})
    - "machine"'s value (weight: {attention_weights[2, 1]:.4f})
    - "I"'s value (weight: {attention_weights[2, 0]:.4f})
    - "love"'s value (weight: {attention_weights[2, 3]:.4f})

  This means the representation for "learning" now incorporates
  context from the entire sentence — it's no longer an isolated word.
    """)

    print(f"  Attention output (first 4 dims):")
    print(f"  {'Token':>12s} | {'dim 0':>8s} {'dim 1':>8s} {'dim 2':>8s} {'dim 3':>8s}")
    print(f"  {'-'*12}-+-{'-'*35}")
    for i, token in enumerate(tokens):
        vals = " ".join(f"{output[i, d]:8.4f}" for d in range(4))
        print(f"  {token:>12s} | {vals}")

    # ── Part 5: Why Self-Attention Matters for CareerAgent ────────────

    print("\n" + "─" * 65)
    print("[Part 5] Why Self-Attention Matters for CareerAgent")
    print("─" * 65)

    print("""
  Self-Attention is the foundation of Transformer models (BERT, GPT)
  and the mechanism behind modern semantic understanding.

  In CareerAgent, there are three levels of matching:

  Level 1 — Keyword matching (Day 4):
    "python" == "python"  ✓  (exact string match)
    "k8s" == "kubernetes" ✗  (missed unless aliased)

  Level 2 — Vector matching (Day 5 — today!):
    Each skill is embedded as a vector. Cosine similarity finds
    that "deep learning" and "neural networks" are close in
    embedding space.  This is a simplified form of attention:
    instead of full Q/K/V, we compare embedding vectors directly.

  Level 3 — Attention-based matching (Week 3+):
    An LLM uses multi-head self-attention to understand the full
    context of each skill, responsibility, and experience entry.
    "Python" in a JD about data engineering means something
    different from "Python" in a JD about web development — and
    attention captures that context.

  The progression:
    String equality → cosine similarity → multi-head attention
    (Day 4)         (Day 5)             (Week 3+)
    """)

    # ── Bonus: Multi-Head Attention concept ──────────────────────────

    print("─" * 65)
    print("[Bonus] Multi-Head Attention Preview")
    print("─" * 65)

    print("""
  Single-head attention (what we computed above):
    - One set of Q, K, V projections
    - One attention pattern
    - May miss some relationships

  Multi-head attention (what Transformers actually use):
    - N independent attention heads, each with its own W_Q, W_K, W_V
    - Each head can learn a different relationship pattern:
      • Head 1: attend to nearby words (syntax)
      • Head 2: attend to semantically related words (meaning)
      • Head 3: attend to the same entity across mentions (coreference)
    - Outputs are concatenated: [head_1 | head_2 | ... | head_N]

  This is why BERT/GPT can understand that "it" refers to "the model"
  three sentences ago — different heads specialize in different kinds
  of relationships.

  Multi-head attention is covered in Day 6.
    """)

    # ── Summary ──────────────────────────────────────────────────────

    print("─" * 65)
    print("Experiment complete!")
    print("─" * 65)

    print("""
  Key takeaways:
  ┌────────────────────────────────────────────────────────────────┐
  │ 1. Self-Attention computes a weighted blend of all tokens'     │
  │    values, where weights come from query-key similarity.       │
  │                                                                │
  │ 2. Three learnable projections: Q (what I want), K (what I     │
  │    have), V (what I contribute).  These are the core of how    │
  │    Transformers learn language patterns.                       │
  │                                                                │
  │ 3. Scaled dot-product (÷√d_k) prevents large dot products      │
  │    from pushing softmax into extreme (0 or 1) values.          │
  │                                                                │
  │ 4. Today's vector matching (cosine similarity) is a simplified │
  │    analog: each JD skill "attends" to all resume skills,       │
  │    picking the best match based on embedding similarity.       │
  │                                                                │
  │ 5. Multi-head attention (Day 6) extends this by having         │
  │    multiple attention patterns in parallel — capturing         │
  │    syntax, semantics, and coreference simultaneously.          │
  └────────────────────────────────────────────────────────────────┘
    """)


if __name__ == "__main__":
    main()
