"""
archetypes.py
-------------
Loads the pre-computed archetypes.json from the clustering pipeline
and provides a fast matcher for new users at runtime.

Given a user's OCEAN scores, returns their archetype assignment
as a full probability distribution (soft assignment, matching the
GMM approach used in cluster.py).

Usage:
    from core.ocean.archetypes import ArchetypeMatcher

    matcher = ArchetypeMatcher()  # loads archetypes.json once

    ocean = {"O": 0.72, "C": 0.45, "E": 0.30, "A": 0.68, "N": 0.55}
    result = matcher.match(ocean)

    result = {
        "dominant":   "loyalist",
        "secondary":  "advocate",
        "confidence": 0.74,
        "probabilities": {
            "enthusiast": 0.03,
            "critic":     0.08,
            ...
            "loyalist":   0.74,
        },
        "archetype": { ...full archetype definition... }
    }
"""

from __future__ import annotations

import json
import os
import math
import numpy as np
from pathlib import Path
import random


# ── default path to archetypes.json ──────────────────────────────────────────
# Resolved relative to this file so imports work regardless of CWD.
_HERE = Path(__file__).resolve().parent
_DEFAULT_ARCHETYPES_PATH = _HERE.parent.parent / "data" / "processed" / "archetypes.json"


OCEAN_KEYS = ["O", "C", "E", "A", "N"]


# ── matcher ───────────────────────────────────────────────────────────────────

class ArchetypeMatcher:
    """
    Loads archetypes.json once and caches it in memory.
    Matches new users to archetypes via soft cosine similarity
    across all 8 archetype centroids — approximating the GMM
    soft assignment without needing the fitted GMM object at runtime.
    """

    def __init__(self, archetypes_path: str | Path | None = None):
        path = Path(archetypes_path) if archetypes_path else _DEFAULT_ARCHETYPES_PATH

        if not path.exists():
            raise FileNotFoundError(
                f"archetypes.json not found at {path}. "
                "Run data/pipeline/cluster.py first."
            )

        with open(path) as f:
            self._archetypes: list[dict] = json.load(f)

        # Load hardcoded scenarios if they exist
        scenarios_path = path.parent / "hardcoded_scenarios.json"
        self._scenarios = []
        if scenarios_path.exists():
            with open(scenarios_path) as f:
                self._scenarios = json.load(f)

        # pre-build centroid matrix for fast similarity computation
        # use empirical_centroid (data-derived) if available, else gmm_centroid
        self._ids      = []
        self._centroids = []

        for a in self._archetypes:
            self._ids.append(a["id"])
            centroid = a.get("target_centroid")
            self._centroids.append([centroid[k] for k in OCEAN_KEYS])

        self._centroid_matrix = np.array(self._centroids)  # (8, 5)
        self._archetype_map   = {a["id"]: a for a in self._archetypes}

    # ── public api ────────────────────────────────────────────────────────────

    def match(self, ocean: dict[str, float]) -> dict:
        """
        Match a user's OCEAN scores to archetypes.

        Args:
            ocean: {"O": float, "C": float, "E": float, "A": float, "N": float}
                   All values should be in [0, 1].

        Returns:
            {
                "dominant":      str,    # archetype id with highest probability
                "secondary":     str,    # archetype id with second highest
                "confidence":    float,  # probability of dominant archetype
                "probabilities": dict,   # {archetype_id: probability} sums to 1
                "archetype":     dict,   # full dominant archetype definition
                "voice_summary": str,    # plain-English voice description
            }
        """
        user_vec = np.array([[ocean.get(k, 0.5) for k in OCEAN_KEYS]])  # (1, 5)

        # soft cosine similarities → convert to probabilities via softmax
        sims     = self._cosine_similarity(user_vec, self._centroid_matrix)[0]  # (8,)
        probs    = self._softmax(sims)

        sorted_idx = np.argsort(probs)[::-1]
        dominant_id  = self._ids[sorted_idx[0]]
        secondary_id = self._ids[sorted_idx[1]]
        confidence   = float(round(probs[sorted_idx[0]], 4))

        probabilities = {
            self._ids[i]: round(float(probs[i]), 4)
            for i in range(len(self._ids))
        }

        dominant_archetype = self._archetype_map[dominant_id]

        return {
            "dominant":      dominant_id,
            "secondary":     secondary_id,
            "confidence":    confidence,
            "probabilities": probabilities,
            "archetype":     dominant_archetype,
            "voice_summary": self._voice_summary(
                dominant_archetype,
                self._archetype_map[secondary_id],
                confidence,
            ),
        }

    def get_archetype(self, archetype_id: str) -> dict | None:
        """Returns a single archetype definition by id."""
        return self._archetype_map.get(archetype_id)

    def get_all_archetypes(self) -> list[dict]:
        """Returns all archetype definitions."""
        return self._archetypes

    def blend_description(self,
                           dominant_id: str,
                           secondary_id: str,
                           confidence: float) -> str:
        """
        Returns a blended description when confidence is low
        (user sits between two archetypes).
        """
        dominant  = self._archetype_map.get(dominant_id, {})
        secondary = self._archetype_map.get(secondary_id, {})

        if confidence >= 0.75:
            # clearly one type
            return dominant.get("description", "")

        # blend the two descriptions
        d_name = dominant.get("name", dominant_id)
        s_name = secondary.get("name", secondary_id)
        return (
            f"You sit between {d_name} and {s_name}. "
            f"{dominant.get('description', '')} "
            f"But you also show traits of {s_name.lower()}: "
            f"{secondary.get('description', '')}"
        )

    def get_hardcoded_scenario(self, seed: int | None = None) -> dict:
        """Returns a random scenario from the hardcoded list."""
        if not self._scenarios:
            return {}
        rng = random.Random(seed)
        scenario = rng.choice(self._scenarios).copy()
        
        # Hydrate with archetype centroids for scoring
        for key in ["review_a", "review_b"]:
            arch_id = scenario[key]["archetype_id"]
            arch = self.get_archetype(arch_id)
            if arch:
                scenario[key]["ocean_profile"] = arch["target_centroid"]
        
        return scenario

    # ── private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _cosine_similarity(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between rows of A and rows of B."""
        A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
        B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
        return A_norm @ B_norm.T

    @staticmethod
    def _softmax(x: np.ndarray, temperature: float = 15.0) -> np.ndarray:
        """
        Softmax with temperature scaling.
        Higher temperature → softer probabilities (more spread out).
        Lower temperature → sharper probabilities (more peaked).
        Temperature=5 gives a reasonable spread for our 0-1 OCEAN space.
        """
        x_scaled = x * temperature
        x_shifted = x_scaled - x_scaled.max()   # numerical stability
        exp_x = np.exp(x_shifted)
        return exp_x / exp_x.sum()

    def _voice_summary(self,
                        dominant: dict,
                        secondary: dict,
                        confidence: float) -> str:
        """
        Build a one-line voice summary for the user's profile page.
        """
        d_traits = dominant.get("voice_traits", {})
        s_traits = secondary.get("voice_traits", {})

        length   = d_traits.get("length", "medium-length")
        tone     = d_traits.get("tone", "balanced")
        bias     = d_traits.get("rating_bias", "balanced")
        focus    = d_traits.get("focus", "overall experience")

        if confidence >= 0.75:
            return (
                f"You write {length} reviews in a {tone} voice. "
                f"You tend to be {bias} with ratings and focus on {focus}."
            )
        else:
            s_tone = s_traits.get("tone", "")
            return (
                f"You write {length} reviews, primarily in a {tone} voice "
                f"with {s_tone} tendencies. "
                f"You tend to be {bias} with ratings and focus on {focus}."
            )
    def get_pairwise_reviews(
        self,
        category: str,
        n_pairs: int = 3,
        review_bank_path: str | None = None,
        seed: int | None = None,
    ) -> list[dict]:
        """
        Returns n_pairs of review pairs for the pairwise selection UI.
        Each pair contains two reviews from DIFFERENT OCEAN profiles
        so the user is always choosing between meaningfully distinct styles.

        Args:
            category:         "books" | "electronics" | "movies" | "apps"
            n_pairs:          how many pairs to return (default 3)
            review_bank_path: optional override path to review_bank.json
            seed:             for reproducible selection

        Returns:
            [
                {
                    "pair_index": 0,
                    "review_a": {
                        "text": "...",
                        "title": "...",
                        "rating": 4.0,
                        "ocean_profile": {"O": 0.7, "C": 0.4, ...}
                    },
                    "review_b": {
                        "text": "...",
                        "title": "...",
                        "rating": 2.0,
                        "ocean_profile": {"O": 0.3, "C": 0.8, ...}
                    }
                },
                ...
            ]
        """
        import json
        import random
        from pathlib import Path

        # load review bank
        if review_bank_path:
            bank_path = Path(review_bank_path)
        else:
            bank_path = _HERE.parent.parent / "data" / "processed" / "review_bank.json"
            apps_path = _HERE.parent.parent / "data" / "processed" / "review_bank_apps.json"

        reviews = []
        try:
            with open(bank_path) as f:
                reviews.extend(json.load(f))
        except FileNotFoundError:
            pass

        # load apps bank separately if category is apps
        if category == "apps":
            try:
                with open(apps_path) as f:
                    reviews = json.load(f)
            except FileNotFoundError:
                pass

        # filter to requested category
        pool = [r for r in reviews if r.get("category") == category]

        if len(pool) < 2:
            return []

        rng   = random.Random(seed)
        pairs = []

        for i in range(n_pairs):
            # pick two reviews that are maximally different in OCEAN space
            # sample a candidate pool then pick the most distant pair
            candidates = rng.sample(pool, min(20, len(pool)))

            best_a, best_b, best_dist = candidates[0], candidates[1], 0.0

            for j in range(len(candidates)):
                for k in range(j + 1, len(candidates)):
                    a = candidates[j]
                    b = candidates[k]
                    # euclidean distance in OCEAN space
                    dist = sum(
                        (a.get(f"ocean_{d}", 0.5) - b.get(f"ocean_{d}", 0.5)) ** 2
                        for d in ["O", "C", "E", "A", "N"]
                    ) ** 0.5
                    if dist > best_dist:
                        best_dist = dist
                        best_a, best_b = a, b

            pairs.append({
                "pair_index": i,
                "review_a": {
                    "text":  best_a["text"],
                    "title": best_a.get("title", ""),
                    "rating": best_a.get("rating", 0),
                    "ocean_profile": {
                        d: best_a.get(f"ocean_{d}", 0.5)
                        for d in ["O", "C", "E", "A", "N"]
                    },
                },
                "review_b": {
                    "text":  best_b["text"],
                    "title": best_b.get("title", ""),
                    "rating": best_b.get("rating", 0),
                    "ocean_profile": {
                        d: best_b.get(f"ocean_{d}", 0.5)
                        for d in ["O", "C", "E", "A", "N"]
                    },
                },
            })

            # remove chosen reviews from pool to avoid repeats
            pool = [r for r in pool if r is not best_a and r is not best_b]
            if len(pool) < 2:
                break

        return pairs


# ── self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("archetypes.py — self test")
    print("=" * 50)

    try:
        matcher = ArchetypeMatcher()
        print(f"Loaded {len(matcher.get_all_archetypes())} archetypes\n")

        test_cases = [
            ("High Enthusiast",  {"O": 0.75, "C": 0.45, "E": 0.80, "A": 0.75, "N": 0.20}),
            ("High Critic",      {"O": 0.75, "C": 0.65, "E": 0.35, "A": 0.30, "N": 0.75}),
            ("High Minimalist",  {"O": 0.25, "C": 0.40, "E": 0.25, "A": 0.55, "N": 0.35}),
            ("Mixed/Ambiguous",  {"O": 0.50, "C": 0.50, "E": 0.50, "A": 0.50, "N": 0.50}),
        ]

        for label, ocean in test_cases:
            result = matcher.match(ocean)
            print(f"{label}")
            print(f"  OCEAN: {ocean}")
            print(f"  → dominant:  {result['dominant']} (confidence: {result['confidence']:.3f})")
            print(f"  → secondary: {result['secondary']}")
            print(f"  → voice:     {result['voice_summary']}")
            top3 = sorted(result["probabilities"].items(),
                          key=lambda x: -x[1])[:3]
            print(f"  → top-3 probs: {top3}")
            print()

    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Run data/pipeline/cluster.py first to generate archetypes.json")