"""
Day 4 Experiment: Position Encoding Demo

Demonstrates:
  1. Sinusoidal position encoding — the math behind Transformer positions
  2. How different positions get different vector representations
  3. Similarity patterns between nearby vs. distant positions
  4. Why position information matters for sequence understanding

Key concept: Self-attention is permutation-invariant — without position
encoding, "I love AI" and "AI love I" look identical to the model.
Position encoding injects order information so the model knows which
word comes first.

Usage:
    docker compose exec backend python experiments/day4_position_encoding_demo.py
"""

import numpy as np


def sinusoidal_position_encoding(seq_len: int, d_model: int) -> np.ndarray:
    """Compute sinusoidal position encoding matrix.

    Args:
        seq_len: Number of positions (sequence length).
        d_model: Model dimension (must be even).

    Returns:
        Array of shape (seq_len, d_model).
    """
    pos = np.arange(seq_len)[:, np.newaxis]  # (seq_len, 1)
    i = np.arange(d_model)[np.newaxis, :]     # (1, d_model)

    angle_rates = 1.0 / np.power(10000, (2 * (i // 2)) / d_model)
    angle = pos * angle_rates  # (seq_len, d_model)

    pe = np.zeros_like(angle)
    pe[:, 0::2] = np.sin(angle[:, 0::2])  # even indices: sin
    pe[:, 1::2] = np.cos(angle[:, 1::2])  # odd indices: cos

    return pe


def main():
    print("=" * 65)
    print("Day 4 Experiment: Position Encoding Demo")
    print("=" * 65)

    d_model = 16   # small for visualization
    seq_len = 10

    pe = sinusoidal_position_encoding(seq_len, d_model)

    # ── Part 1: The encoding matrix ─────────────────────────────────

    print("\n" + "─" * 65)
    print("[Part 1] Sinusoidal Position Encoding Matrix")
    print("─" * 65)

    print(f"\n  Shape: ({seq_len}, {d_model})  — {seq_len} positions, {d_model} dims")
    print(f"\n  Position encoding matrix (first 8 dimensions):")
    print(f"  {'Pos':>4s} | {'dim 0':>8s} {'dim 1':>8s} {'dim 2':>8s} {'dim 3':>8s} "
          f"{'dim 4':>8s} {'dim 5':>8s} {'dim 6':>8s} {'dim 7':>8s}")
    print(f"  {'-'*4}-+-{'-'*59}")
    for p in range(min(seq_len, 10)):
        vals = " ".join(f"{pe[p, d]:8.4f}" for d in range(8))
        print(f"  {p:4d} | {vals}")

    # ── Part 2: Wavelength patterns ─────────────────────────────────

    print("\n" + "─" * 65)
    print("[Part 2] Wavelengths across dimensions")
    print("─" * 65)

    print(f"\n  Lower dimensions → shorter wavelengths (fine position detail)")
    print(f"  Higher dimensions  → longer wavelengths (coarse position)")
    print(f"\n  {'Dimension':12s} {'Wavelength':>12s}")
    print(f"  {'-'*12} {'-'*12}")
    for d in [0, 2, 4, 8, d_model - 2]:
        wavelength = 2 * np.pi * (10000 ** (2 * (d // 2) / d_model))
        if wavelength > 100000:
            ws = f"{wavelength:.0f}"
        else:
            ws = f"{wavelength:.1f}"
        print(f"  dim {d:<9d} {ws:>12s}")

    # ── Part 3: Similarity between positions ────────────────────────

    print("\n" + "─" * 65)
    print("[Part 3] Cosine similarity between position vectors")
    print("─" * 65)

    print(f"\n  Nearby positions should be more similar than distant ones.")
    print(f"\n  {'Pos A':>6s} {'Pos B':>6s} {'Distance':>9s} {'Similarity':>11s}")
    print(f"  {'-'*6} {'-'*6} {'-'*9} {'-'*11}")
    for a, b in [(0, 1), (0, 2), (0, 5), (0, 9), (4, 5), (4, 9)]:
        sim = np.dot(pe[a], pe[b]) / (np.linalg.norm(pe[a]) * np.linalg.norm(pe[b]))
        print(f"  {a:6d} {b:6d} {b-a:9d} {sim:11.4f}")

    # ── Part 4: Why this matters ────────────────────────────────────

    print("\n" + "─" * 65)
    print("[Part 4] Why position encoding matters")
    print("─" * 65)

    print("""
  Without position encoding (PE):
    "I love AI"  and  "AI love I"
    → Both produce the same attention output!
    → The model can't tell which word comes first.

  With PE:
    Each token gets a unique position vector added to its embedding.
    "I" at position 0 ≠ "I" at position 2
    → The model learns to attend based on both content AND order.

  In CareerAgent:
    - JD text: skills listed first vs. buried in a paragraph
    - Resume: most recent experience vs. oldest
    - PE helps models distinguish these — the order carries meaning.
    """)

    # ── Summary ────────────────────────────────────────────────────

    print("─" * 65)
    print("Experiment complete!")
    print("─" * 65)

    print("""
  Key takeaways:
  ┌────────────────────────────────────────────────────────────────┐
  │ 1. Transformers have no built-in sense of order.               │
  │    Position encoding is how we inject sequence information.    │
  │                                                                │
  │ 2. Sinusoidal PE uses sin/cos at different frequencies:        │
  │    - Low dims = high frequency (nearby positions differ a lot) │
  │    - High dims = low frequency (position changes slowly)       │
  │                                                                │
  │ 3. Nearby positions have higher cosine similarity than         │
  │    distant ones — this helps attention weigh local context.    │
  │                                                                │
  │ 4. Modern models (Llama, GPT-4) use RoPE instead of            │
  │    sinusoidal PE — we'll learn that in Week 2.                 │
  └────────────────────────────────────────────────────────────────┘
    """)


if __name__ == "__main__":
    main()
