import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.config import settings


class EmbeddingService:
    """Service for generating text embeddings and computing similarities."""

    def __init__(self, model_name: str | None = None, device: str | None = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL_NAME
        self.device = device or settings.EMBEDDING_DEVICE
        self.model = SentenceTransformer(self.model_name, device=self.device)

    def encode(self, texts: str | list[str]) -> np.ndarray:
        """Generate embedding vectors for one or more texts.

        Args:
            texts: A single text string or a list of text strings.

        Returns:
            A numpy array of shape (n_texts, embedding_dim).
        """
        if isinstance(texts, str):
            texts = [texts]
        return self.model.encode(texts, convert_to_numpy=True)

    def similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts.

        Args:
            text1: First text.
            text2: Second text.

        Returns:
            Cosine similarity score between 0 and 1.
        """
        embeddings = self.encode([text1, text2])
        vec1, vec2 = embeddings[0], embeddings[1]
        cos_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        return float(cos_sim)

    def batch_similarity(self, query: str, candidates: list[str]) -> list[float]:
        """Compute similarity between a query and multiple candidate texts.

        Args:
            query: The query text.
            candidates: A list of candidate texts to compare against.

        Returns:
            A list of cosine similarity scores, one per candidate.
        """
        query_emb = self.encode(query).squeeze()  # shape: (emb_dim,)
        candidate_embs = self.encode(candidates)   # shape: (n, emb_dim)

        query_emb = query_emb / np.linalg.norm(query_emb)
        candidate_embs = candidate_embs / np.linalg.norm(
            candidate_embs, axis=1, keepdims=True
        )
        similarities = np.dot(candidate_embs, query_emb)  # shape: (n,)
        return similarities.tolist()
