"""
core/product_context.py
-----------------------
Fetches real product/app context before review generation.

For apps:    Google Play Scraper — description + top reviews
For products: Amazon dataset lookup — description + similar reviews

This context feeds into the LLM prompt so the generated review
focuses on what actually matters for that specific product,
not just the user's general persona.

Usage:
    from core.product_context import get_product_context

    context = get_product_context(
        product_name="Kuda",
        category="apps",
    )
    # {
    #   "name": "Kuda",
    #   "description": "Kuda is a digital bank...",
    #   "common_focus_areas": ["customer service", "transactions", "debit card"],
    #   "sample_reviews": ["Great app but...", "Transaction failed..."],
    #   "avg_public_rating": 4.1,
    #   "source": "playstore",
    # }
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

_HERE     = Path(__file__).resolve().parent
_DATA_DIR = _HERE.parent / "data" / "processed"

# ── Play Store app search ─────────────────────────────────────────────────────

# Add this at the top of product_context.py
KNOWN_APP_IDS = {
    "kuda":       "com.kudabank.app",
    "opay":       "team.opay.pay",
    "palmpay":    "com.palmpay.merchant",
    "moniepoint": "com.teamapt.moniepointmfb",
    "alat":       "com.wemabank.alat",
    "spotify":    "com.spotify.music",
    "whatsapp":   "com.whatsapp",
    "telegram":   "org.telegram.messenger",
    "instagram":  "com.instagram.android",
    "youtube":    "com.google.android.youtube",
    "twitter":    "com.twitter.android",
    "netflix":    "com.netflix.mediaclient",
    "uber":       "com.ubercab",
    "bolt":       "ee.mtakso.client",
    "jumia":      "com.jumia.android",
    "flutterwave":"com.flutterwave.raveandroid",
    "paystack":   "co.paystack.android",
    "cowrywise":  "com.cowrywise.android",
    "piggyvest":  "com.piggyvest.app",
    "carbon":     "com.accesscarbon",
}

def _fetch_playstore_context(product_name: str) -> dict:
    try:
        from google_play_scraper import reviews, Sort, app as get_app_info

        # look up known app ID
        name_lower = product_name.lower().strip()
        app_id = KNOWN_APP_IDS.get(name_lower)

        if not app_id:
            # try search as fallback
            from google_play_scraper import search
            results = search(product_name, lang="en", country="ng", n_hits=5)
            for r in results:
                if r.get("appId"):
                    app_id = r["appId"]
                    break

        if not app_id:
            return {}

        # fetch app info
        try:
            app_info = get_app_info(app_id, lang="en", country="ng")
        except Exception:
            app_info = {"title": product_name, "description": "", "score": 0}

        # fetch reviews
        review_results, _ = reviews(
            app_id,
            lang="en",
            country="ng",
            sort=Sort.NEWEST,
            count=15,
        )

        review_texts = [
            r["content"] for r in review_results
            if r.get("content") and len(r["content"]) > 20
        ][:10]

        focus_areas = _extract_focus_areas(review_texts, category="apps")

        return {
            "name":              app_info.get("title", product_name),
            "description":       app_info.get("description", "")[:500],
            "avg_public_rating": app_info.get("score", 0),
            "installs":          app_info.get("installs", ""),
            "common_focus_areas": focus_areas,
            "sample_reviews":    review_texts[:5],
            "source":            "playstore",
            "app_id":            app_id,
        }

    except Exception as e:
        print(f"  [ProductContext] Play Store fetch failed: {e}")
        return {}


# ── Amazon dataset lookup ─────────────────────────────────────────────────────

def _fetch_product_context(product_name: str) -> dict:
    """
    Looks up the product in the Amazon review bank.
    Falls back to fuzzy matching if exact match not found.
    """
    try:
        bank_path = _DATA_DIR / "review_bank.json"
        if not bank_path.exists():
            return {}

        with open(bank_path) as f:
            bank = json.load(f)

        # exact match first
        name_lower = product_name.lower()
        matches = [
            r for r in bank
            if r.get("title", "").lower() == name_lower
            and r.get("category") in ("electronics", "products")
        ]

        # fuzzy match — product name appears in title
        if not matches:
            matches = [
                r for r in bank
                if name_lower in r.get("title", "").lower()
                and r.get("category") in ("electronics", "products")
            ]

        # broader fuzzy — any word matches
        if not matches:
            words = [w for w in name_lower.split() if len(w) > 3]
            matches = [
                r for r in bank
                if any(w in r.get("title", "").lower() for w in words)
                and r.get("category") in ("electronics", "products")
            ]

        if not matches:
            return {}

        # aggregate
        review_texts = [r.get("text", "") for r in matches if r.get("text")][:10]
        ratings      = [r.get("rating", 3) for r in matches if r.get("rating")]
        title        = matches[0].get("title", product_name)
        description  = matches[0].get("description", "")

        # clean description
        if isinstance(description, str) and description.startswith("["):
            try:
                desc_list   = json.loads(description.replace("'", '"'))
                description = " ".join(desc_list)[:500]
            except Exception:
                description = description[:500]

        focus_areas = _extract_focus_areas(review_texts, category="products")

        return {
            "name":              title,
            "description":       description[:500],
            "avg_public_rating": round(sum(ratings) / len(ratings), 1) if ratings else 0,
            "common_focus_areas": focus_areas,
            "sample_reviews":    review_texts[:5],
            "source":            "amazon_dataset",
        }

    except Exception as e:
        print(f"  [ProductContext] Product lookup failed: {e}")
        return {}


# ── focus area extractor ──────────────────────────────────────────────────────

# keyword groups that map to focus areas reviewers care about
FOCUS_KEYWORD_MAP = {
    "apps": {
        "transactions / payments":  ["transaction", "transfer", "payment", "send money", "debit", "credit", "refund"],
        "customer service":         ["customer service", "support", "helpline", "response", "contact"],
        "bugs / crashes":           ["bug", "crash", "error", "freeze", "glitch", "issue", "problem"],
        "speed / performance":      ["slow", "fast", "speed", "quick", "lag", "load"],
        "user interface":           ["ui", "interface", "design", "easy to use", "navigation", "simple"],
        "notifications":            ["notification", "alert", "push", "sms"],
        "security":                 ["security", "safe", "hack", "fraud", "otp", "verification"],
        "updates":                  ["update", "version", "upgrade", "new feature"],
    },
    "products": {
        "build quality":            ["quality", "build", "material", "sturdy", "flimsy", "durable"],
        "value for money":          ["price", "worth", "value", "expensive", "cheap", "cost"],
        "packaging / delivery":     ["packaging", "delivery", "arrived", "shipping", "box"],
        "ease of use":              ["easy", "simple", "setup", "instructions", "manual"],
        "performance":              ["works", "performance", "function", "reliable", "battery"],
        "size / fit":               ["size", "fit", "small", "large", "dimensions"],
        "customer service":         ["return", "refund", "support", "warranty", "replacement"],
    },
    "books": {
        "plot / story":             ["plot", "story", "storyline", "ending", "twist", "narrative"],
        "writing style":            ["writing", "prose", "style", "author", "language"],
        "characters":               ["character", "protagonist", "villain", "cast"],
        "pacing":                   ["pacing", "slow", "fast", "dragged", "page-turner"],
        "value":                    ["worth", "price", "value", "recommend"],
    },
    "movies": {
        "acting":                   ["acting", "performance", "actor", "actress", "cast"],
        "plot":                     ["plot", "story", "ending", "twist", "script"],
        "visuals":                  ["visual", "effects", "cinematography", "graphics"],
        "pacing":                   ["slow", "boring", "fast", "engaging", "dragged"],
        "overall experience":       ["worth", "recommend", "cinema", "watch"],
    },
}


def _extract_focus_areas(
    review_texts: list[str],
    category:     str,
) -> list[str]:
    """
    Analyses review texts to find what aspects reviewers focus on most.
    Returns top 3-5 focus areas by mention frequency.
    """
    keyword_map = FOCUS_KEYWORD_MAP.get(category, FOCUS_KEYWORD_MAP["apps"])
    counts      = {area: 0 for area in keyword_map}
    combined    = " ".join(review_texts).lower()

    for area, keywords in keyword_map.items():
        for kw in keywords:
            counts[area] += combined.count(kw)

    # sort by count, return top areas that have at least 1 mention
    sorted_areas = sorted(counts.items(), key=lambda x: -x[1])
    return [area for area, count in sorted_areas if count > 0][:5]


# ── main public function ──────────────────────────────────────────────────────

def get_product_context(
    product_name: str,
    category:     str,
) -> dict:
    """
    Main entry point. Fetches product context based on category.

    Args:
        product_name: e.g. "Kuda", "AirPods Pro"
        category:     "apps" | "products" | "books" | "movies" | "restaurants"

    Returns:
        {
            "name":               str,
            "description":        str,
            "avg_public_rating":  float,
            "common_focus_areas": list[str],
            "sample_reviews":     list[str],
            "source":             str,
        }
        Returns empty dict if nothing found — caller handles gracefully.
    """
    category = category.lower().strip()

    if category in ("apps", "mobile apps"):
        context = _fetch_playstore_context(product_name)
    elif category in ("products", "electronics", "online shopping"):
        context = _fetch_product_context(product_name)
    elif category == "books":
        context = _fetch_product_context(product_name)  # books are in Amazon dataset
    else:
        context = {}

    if context:
        print(f"  [ProductContext] ✓ {product_name} — "
              f"source: {context.get('source', 'unknown')}, "
              f"focus areas: {context.get('common_focus_areas', [])}")
    else:
        print(f"  [ProductContext] No context found for '{product_name}' "
              f"in '{category}' — generating from persona only")

    return context


# ── self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("product_context.py — self test")
    print("=" * 50)

    tests = [
        ("Kuda",     "apps"),
        ("Spotify",  "apps"),
        ("AirPods",  "products"),
    ]

    for name, cat in tests:
        print(f"\n── {name} ({cat}) ──────────────────────────")
        ctx = get_product_context(name, cat)
        if ctx:
            print(f"  Name:         {ctx.get('name', '')[:50]}")
            print(f"  Avg rating:   {ctx.get('avg_public_rating', 'N/A')}")
            print(f"  Focus areas:  {ctx.get('common_focus_areas', [])}")
            print(f"  Description:  {ctx.get('description', '')[:100]}...")
            print(f"  Sample reviews ({len(ctx.get('sample_reviews', []))}):")
            for r in ctx.get("sample_reviews", [])[:2]:
                print(f"    - {r[:80]}...")
        else:
            print("  No context found")