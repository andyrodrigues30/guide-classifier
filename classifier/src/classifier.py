from sentence_transformers import SentenceTransformer
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple


model = SentenceTransformer("all-MiniLM-L6-v2")


# data structures
@dataclass
class Guide:
    id: str
    title: str
    content: str
    embedding: np.ndarray = None


@dataclass
class ClassificationResult:
    label: str  # Base | Variant | Amendment | Duplicate
    confidence: float
    related_guide_id: Optional[str]
    reason: Dict


# embedding
def embed(text: str) -> np.ndarray:
    return model.encode(text)


def build_text(g: Guide) -> str:
    return f"""
TITLE: {g.title}
CONTENT: {g.content}
"""


# similarity
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def compute_similarity(new_emb: np.ndarray, guides: List[Guide]) -> List[Tuple[Guide, float]]:
    sims = []
    for g in guides:
        sims.append((g, cosine_similarity(new_emb, g.embedding)))
    sims.sort(key=lambda x: x[1], reverse=True)
    return sims


# structure similarity
def structure_similarity(title_a: str, title_b: str) -> float:
    a = set(title_a.lower().split())
    b = set(title_b.lower().split())
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# domain boost
DOMAIN_BOOSTS = {
    "docker": 0.04,
    "kubernetes": 0.05,
    "rust": 0.03,
    "api": 0.02,
    "ubuntu": 0.02,
    "install": 0.01
}


def apply_domain_boost(text: str, score: float) -> float:
    text_lower = text.lower()

    for keyword, boost in DOMAIN_BOOSTS.items():
        if keyword in text_lower:
            score += boost

    return min(score, 1.0)


# main classifier
def classify_guide(new_guide: Guide, existing_guides: List[Guide]) -> ClassificationResult:

    new_emb = embed(build_text(new_guide))

    if not existing_guides:
        return ClassificationResult(
            label="Base",
            confidence=1.0,
            related_guide_id=None,
            reason={"rule": "no_existing_guides"}
        )

    similarities = compute_similarity(new_emb, existing_guides)

    top_k = similarities[:5]

    best_guide, best_sim = top_k[0]
    second_sim = top_k[1][1] if len(top_k) > 1 else 0.0

    raw_text = f"{new_guide.title} {new_guide.content}"

    best_sim = apply_domain_boost(raw_text, best_sim)
    second_sim = apply_domain_boost(raw_text, second_sim)

    title_sim = structure_similarity(new_guide.title, best_guide.title)

    delta = best_sim - second_sim
    identity_score = (0.75 * best_sim) + (0.25 * title_sim)

    # DUPLICATE DETECTION
    if best_sim >= 0.95 and title_sim >= 0.9:
        return ClassificationResult(
            label="Duplicate",
            confidence=best_sim,
            related_guide_id=best_guide.id,
            reason={
                "rule": "near_exact_duplicate",
                "best_similarity": best_sim,
                "title_similarity": title_sim,
                "identity_score": identity_score
            }
        )

    # AMENDMENT (now stricter)
    if (
        best_sim >= 0.78 and
        title_sim >= 0.25 and
        delta >= 0.05 and
        best_sim < 0.95   # prevents duplicates slipping through
    ):
        return ClassificationResult(
            label="Amendment",
            confidence=best_sim,
            related_guide_id=best_guide.id,
            reason={
                "rule": "amendment_relative_match",
                "best_similarity": best_sim,
                "second_best_similarity": second_sim,
                "delta": delta,
                "title_similarity": title_sim,
                "identity_score": identity_score
            }
        )

    # VARIANT
    if 0.75 <= best_sim < 0.82:

        return ClassificationResult(
            label="Variant",
            confidence=best_sim,
            related_guide_id=best_guide.id,
            reason={
                "rule": "semantic_variant",
                "best_similarity": best_sim,
                "title_similarity": title_sim
            }
        )

    # BASE
    return ClassificationResult(
        label="Base",
        confidence=1.0 - best_sim,
        related_guide_id=None,
        reason={
            "rule": "new_concept_or_low_similarity",
            "best_similarity": best_sim,
            "title_similarity": title_sim
        }
    )


# indexing helper
def index_guides(guides: List[Guide]) -> None:
    for g in guides:
        g.embedding = embed(build_text(g))