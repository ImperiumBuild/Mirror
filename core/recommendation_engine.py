"""
core/recommendation_engine.py
------------------------------
Orchestrates personalised recommendations.

Categories: apps (with sub-categories), books, products (electronics), movies
Movies replaced with products as primary non-app category.
Apps split into sub-categories: productivity, social, entertainment, utilities, gaming
"""

from __future__ import annotations

import json
import sys
import random
from pathlib import Path
from collections import defaultdict
from core.llm.prompt_builder import _parse_recommendations



from core.llm.service import LLMService
from core.ocean.archetypes import ArchetypeMatcher

_HERE      = Path(__file__).resolve().parent
_DATA_DIR  = _HERE.parent / "data" / "processed"

BANK_PATH      = _DATA_DIR / "review_bank.json"
BANK_APPS_PATH = _DATA_DIR / "review_bank_apps.json"

CATEGORY_ALIASES = {
    "apps":            "apps",
    "mobile apps":     "apps",
    "electronics":     "products",
    "products":        "products",
    "online shopping": "products",
    "books":           "books",
    "movies":          "movies",
    "movies & tv":     "movies",
    "tv":              "movies",
    "restaurants":     "products",
    "food":            "products",
}

APP_SUBCATEGORIES = [
    "productivity",
    "social",
    "entertainment",
    "utilities",
    "gaming",
]


class RecommendationEngine:

    def __init__(self, provider: str = "gemini", api_key: str | None = None):
        self._llm     = LLMService(provider_name=provider, api_key=api_key)
        self._matcher = ArchetypeMatcher()
        self._bank    = self._load_bank()
        self._index   = self._build_item_index()

    # ── public api ────────────────────────────────────────────────────────────

    def recommend(
        self,
        profile:  dict,
        category: str,
        n:        int = 5,
        seed:     int | None = None,
    ) -> list[dict]:
        from core.llm.prompt_builder import (
            build_freeform_recommendation_prompt,
            build_product_recommendation_prompt,
            build_affinity_recommendation_prompt,
        )
        from core.jumia_scraper import search_jumia
        from recommendations.models import ArchetypeAffinity

        arch_id = profile.get("dominant_archetype")
        
        # 1. Try Data-Driven Retrieval (GMM Archetype Segmentation)
        # Pull top items that people with this personality ACTUALLY reviewed highly
        affinities = ArchetypeAffinity.objects.filter(
            archetype_id=arch_id,
            category=category
        ).order_by("-affinity_score")[:40]

        if affinities.exists():
            candidate_list = [
                {"title": a.item_title, "score": a.affinity_score, "avg_rating": a.avg_rating}
                for a in affinities
            ]
            
            # Use LLM to pick the absolute best from the cluster and personalize the reasoning
            prompt   = build_affinity_recommendation_prompt(
                profile=profile, category=category, candidates=candidate_list, n=n)
            response = self._llm._provider.generate(prompt)
            ranked   = _parse_recommendations(response)
            
            for rec in ranked:
                rec["category"] = category
                rec["is_cold_start"] = False
            return ranked[:n]

        # 2. Fallback to Jumia for electronics/products if no data exists
        if category.lower() in ("products", "electronics"):
            # Step 1 — LLM generates search terms + reasoning based on persona
            prompt       = build_product_recommendation_prompt(profile, n=n)
            response     = self._llm._provider.generate(prompt)

            # parse items
            import json, re
            cleaned = re.sub(r"```(?:json)?", "", response).strip().rstrip("`")
            try:
                items = json.loads(cleaned)
                if not isinstance(items, list):
                    items = []
            except Exception:
                items = []

            # Step 2 — search Jumia for each term
            results = []
            for item in items[:n]:
                term      = item.get("term", "")
                reasoning = item.get("reasoning", f"Matched to your profile — {term}.")
                
                if not term: continue

                products = search_jumia(term, n=1)
                if products:
                    p = products[0]
                    results.append({
                        "rank":        len(results) + 1,
                        "title":       p["title"],
                        "price":       p["price"],
                        "image_url":   p["image_url"],
                        "url":         p["url"],
                        "rating":      p["rating"],
                        "confidence":  0.80,
                        "reasoning":   reasoning,
                        "source":      "jumia",
                        "category":    category,
                        "is_cold_start": False,
                    })

            # fallback if no Jumia results found
            if not results:
                for i, item in enumerate(items[:n]):
                    results.append({
                        "rank":        i + 1,
                        "title":       item.get("term", "Product"),
                        "price":       "N/A",
                        "image_url":   "",
                        "url":         "#",
                        "rating":      0.0,
                        "confidence":  0.50,
                        "reasoning":   item.get("reasoning", "Recommended based on your profile."),
                        "source":      "jumia",
                        "category":    category,
                        "is_cold_start": True,
                    })

            return results[:n]

        else:
            # books, movies — freeform LLM
            prompt   = build_freeform_recommendation_prompt(
                profile=profile, category=category, n=n)
            response = self._llm._provider.generate(prompt)
            ranked   = _parse_recommendations(response)

            # Enrich movies with poster paths (TMDB full URL)
            if category.lower() == "movies":
                from core.llm.prompt_builder import load_movies_from_json
                try:
                    movie_db  = load_movies_from_json()
                    movie_map = {m["title"].lower().strip(): m for m in movie_db}
                    for rec in ranked:
                        title_key = rec.get("title", "").lower().strip()
                        if title_key in movie_map:
                            movie_data = movie_map[title_key]
                            path = movie_data.get("poster_path")
                            if path:
                                # Transform relative TMDB path to full URL
                                full_url = path if path.startswith("http") else f"https://image.tmdb.org/t/p/w500{path}"
                                rec["poster_path"] = full_url
                                rec["image_url"]   = full_url # match product schema
                except Exception:
                    pass

            for rec in ranked:
                rec["category"]      = category
                rec["is_cold_start"] = False
            return ranked[:n]


    def recommend_apps_by_subcategory(
        self,
        profile:           dict,
        n_per_subcategory: int = 3,
        seed:              int | None = None,
    ) -> dict[str, list[dict]]:
        from core.llm.prompt_builder import build_freeform_recommendation_prompt
        import concurrent.futures

        def fetch_sub(sub):
            try:
                prompt  = build_freeform_recommendation_prompt(
                    profile=profile, category="apps",
                    sub_category=sub, n=n_per_subcategory)
                response = self._llm._provider.generate(prompt)
                ranked   = _parse_recommendations(response)

                # Fetch icons from Play Store
                try:
                    from google_play_scraper import search
                    for rec in ranked:
                        title = rec.get("title", "")
                        if title:
                            # Search for the app to get the icon
                            search_results = search(title, lang="en", country="ng")
                            if search_results:
                                icon_url = search_results[0].get("icon", "")
                                rec["image_url"] = icon_url
                                rec["app_icon"]  = icon_url # extra field for clarity
                except Exception:
                    pass

                for rec in ranked:
                    rec["category"]      = "apps"
                    rec["sub_category"]  = sub
                    rec["is_cold_start"] = False
                return sub, ranked[:n_per_subcategory]
            except Exception as e:
                print(f"  WARNING: '{sub}' failed — {e}")
                return sub, []

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = {executor.submit(fetch_sub, sub): sub
                    for sub in APP_SUBCATEGORIES}
            result  = {}
            for future in concurrent.futures.as_completed(futures):
                sub, recs = future.result()
                result[sub] = recs

        return result

    def recommend_dashboard(
        self,
        profile:        dict,
        n_per_category: int = 3,
    ) -> dict:
        """
        Returns full dashboard recommendations:
        - apps: grouped by sub-category (2 per sub)
        - books: top n
        - products: top n
        - movies: top n

        Returns:
            {
                "apps": {
                    "productivity": [...],
                    "social": [...],
                    ...
                },
                "books":    [...],
                "products": [...],
                "movies":   [...],
            }
        """
        result = {}

        # apps by sub-category
        result["apps"] = self.recommend_apps_by_subcategory(
            profile=profile,
            n_per_subcategory=2,
        )

        # other categories
        for category in ["books", "products", "movies"]:
            try:
                result[category] = self.recommend(
                    profile=profile,
                    category=category,
                    n=n_per_category,
                )
            except Exception as e:
                print(f"  WARNING: '{category}' failed — {e}")
                result[category] = []

        return result

    # ── private ───────────────────────────────────────────────────────────────

    def _load_bank(self) -> list[dict]:

        from recommendations.models import ReviewBank

        bank = list(
            ReviewBank.objects.values(
                "category",
                "sub_category",
                "item_id",
                "title",
                "text",
                "rating",
                "ocean_o",
                "ocean_c",
                "ocean_e",
                "ocean_a",
                "ocean_n",
                "generosity_score",
                "avg_review_length",
                "source",
            )
        )

        renamed = []

        for r in bank:
            renamed.append({
                "category":          r["category"],
                "sub_category":      r["sub_category"],
                "item_id":           r["item_id"],
                "title":             r["title"],
                "text":              r["text"],
                "rating":            r["rating"],
                "ocean_O":           r["ocean_o"],
                "ocean_C":           r["ocean_c"],
                "ocean_E":           r["ocean_e"],
                "ocean_A":           r["ocean_a"],
                "ocean_N":           r["ocean_n"],
                "generosity_score":  r["generosity_score"],
                "avg_review_length": r["avg_review_length"],
                "source":            r["source"],
            })

        return renamed

    def _build_item_index(self) -> dict[str, dict]:
        """
        Builds per-item index from review bank.
        Aggregates all reviews for the same title into one record.
        Normalises category: electronics → products.
        """
        items = defaultdict(lambda: {
            "title":        "",
            "category":     "",
            "sub_category": "",
            "ratings":      [],
            "ocean_O":      [],
            "ocean_C":      [],
            "ocean_E":      [],
            "ocean_A":      [],
            "ocean_N":      [],
        })

        for review in self._bank:
            title = review.get("title", "").strip()
            if not title:
                continue

            item = items[title]
            item["title"] = title

            # normalise category
            raw_cat = review.get("category", "")
            item["category"]     = CATEGORY_ALIASES.get(raw_cat, raw_cat)
            item["sub_category"] = review.get("sub_category", "")

            item["ratings"].append(review.get("rating", 3))
            for dim in ["O", "C", "E", "A", "N"]:
                item[f"ocean_{dim}"].append(review.get(f"ocean_{dim}", 0.5))

        index = {}
        for title, item in items.items():
            if not item["ratings"]:
                continue
            n = len(item["ratings"])
            index[title] = {
                "title":        title,
                "category":     item["category"],
                "sub_category": item["sub_category"],
                "avg_rating":   round(sum(item["ratings"]) / n, 2),
                "review_count": n,
                "ocean_O":      round(sum(item["ocean_O"]) / n, 3),
                "ocean_C":      round(sum(item["ocean_C"]) / n, 3),
                "ocean_E":      round(sum(item["ocean_E"]) / n, 3),
                "ocean_A":      round(sum(item["ocean_A"]) / n, 3),
                "ocean_N":      round(sum(item["ocean_N"]) / n, 3),
            }

        return index

    def _get_candidates(
        self,
        profile:   dict,
        category:  str,
        pool_size: int = 20,
        seed:      int | None = None,
    ) -> tuple[list[dict], bool]:
        rng           = random.Random(seed)
        ocean         = profile.get("ocean", {})
        is_cold_start = False

        category_items = [
            item for item in self._index.values()
            if item["category"] == category
        ]

        if len(category_items) < 5:
            category_items = list(self._index.values())
            is_cold_start  = True

        scored = [
            (self._ocean_similarity(ocean, item), item)
            for item in category_items
        ]
        scored.sort(key=lambda x: -x[0])

        top_pool = scored[:pool_size * 2]
        rng.shuffle(top_pool)
        selected = top_pool[:pool_size]

        candidates = []
        for _, item in selected:
            candidates.append({
                "title":       item["title"],
                "avg_rating":  item["avg_rating"],
                "review_count": item["review_count"],
                "description": (
                    f"Avg rating {item['avg_rating']}/5 from "
                    f"{item['review_count']} reviewer(s)."
                ),
            })

        return candidates, is_cold_start

    def _get_app_subcategory_candidates(
        self,
        profile:   dict,
        sub:       str,
        pool_size: int = 10,
        seed:      int | None = None,
    ) -> tuple[list[dict], bool]:
        """Retrieves candidates filtered to a specific app sub-category."""
        rng           = random.Random(seed)
        ocean         = profile.get("ocean", {})
        is_cold_start = False

        sub_items = [
            item for item in self._index.values()
            if item["category"] == "apps" and item["sub_category"] == sub
        ]

        if len(sub_items) < 3:
            # fall back to all apps
            sub_items     = [
                item for item in self._index.values()
                if item["category"] == "apps"
            ]
            is_cold_start = True

        scored = [
            (self._ocean_similarity(ocean, item), item)
            for item in sub_items
        ]
        scored.sort(key=lambda x: -x[0])

        top_pool = scored[:pool_size * 2]
        rng.shuffle(top_pool)
        selected = top_pool[:pool_size]

        candidates = []
        for _, item in selected:
            candidates.append({
                "title":       item["title"],
                "avg_rating":  item["avg_rating"],
                "review_count": item["review_count"],
                "description": (
                    f"Avg rating {item['avg_rating']}/5 from "
                    f"{item['review_count']} reviewer(s)."
                ),
            })

        return candidates, is_cold_start

    def _ocean_similarity(
        self,
        user_ocean: dict[str, float],
        item:       dict,
    ) -> float:
        dims   = ["O", "C", "E", "A", "N"]
        u      = [user_ocean.get(d, 0.5) for d in dims]
        v      = [item.get(f"ocean_{d}", 0.5) for d in dims]
        dot    = sum(a * b for a, b in zip(u, v))
        norm_u = sum(a ** 2 for a in u) ** 0.5
        norm_v = sum(b ** 2 for b in v) ** 0.5
        if norm_u == 0 or norm_v == 0:
            return 0.0
        return dot / (norm_u * norm_v)


# ── self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("recommendation_engine.py — self test")
    print("=" * 55)

    enthusiast = {
        "ocean": {"O": 0.75, "C": 0.45, "E": 0.80, "A": 0.85, "N": 0.15},
        "voice_profile": {
            "length": "verbose", "tone": "warm and expressive",
            "rating_bias": "generous", "focus": "emotional experience",
            "negatives": "rarely mentioned",
        },
        "dominant_archetype":  "enthusiast",
        "secondary_archetype": "advocate",
        "rating_calibration":  {"harsh_or_generous": "generous", "avg_given_rating": 4.7},
        "style_tags":          ["focuses on the positive"],
        "profile_summary":     "You are an Enthusiast — warm, generous, expressive.",
    }

    critic = {
        "ocean": {"O": 0.70, "C": 0.75, "E": 0.35, "A": 0.22, "N": 0.85},
        "voice_profile": {
            "length": "long and structured", "tone": "analytical and direct",
            "rating_bias": "harsh", "focus": "defects, unmet expectations",
            "negatives": "front and centre",
        },
        "dominant_archetype":  "critic",
        "secondary_archetype": "skeptic",
        "rating_calibration":  {"harsh_or_generous": "harsh", "avg_given_rating": 2.0},
        "style_tags":          ["doesn't shy away from calling out problems"],
        "profile_summary":     "You are a Critic — demanding, analytical, harsh rater.",
    }

    engine = RecommendationEngine(provider="gemini")
    print(f"\n  Item index: {len(engine._index)} unique items")

    cats = {}
    for item in engine._index.values():
        c = item["category"]
        cats[c] = cats.get(c, 0) + 1
    print(f"  By category: {cats}")

    print("\n── Enthusiast — Products ────────────────────────────")
    recs = engine.recommend(profile=enthusiast, category="products", n=3, seed=42)
    for r in recs:
        print(f"  {r.get('rank','?')}. {r.get('title','?')}")
        print(f"     {r.get('reasoning','')}")

    print("\n── Critic — Books ───────────────────────────────────")
    recs2 = engine.recommend(profile=critic, category="books", n=3, seed=42)
    for r in recs2:
        print(f"  {r.get('rank','?')}. {r.get('title','?')}")
        print(f"     {r.get('reasoning','')}")

    print("\n── Apps by sub-category (Enthusiast) ────────────────")
    app_recs = engine.recommend_apps_by_subcategory(
        profile=enthusiast, n_per_subcategory=2, seed=42)
    for sub, recs in app_recs.items():
        print(f"\n  {sub.upper()}:")
        for r in recs:
            print(f"    - {r.get('title','?')}: {r.get('reasoning','')[:80]}...")