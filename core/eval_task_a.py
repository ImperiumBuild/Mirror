"""
eval_task_a.py
--------------
Evaluates Task A (Review Generation & Rating Prediction) by simulating 
human reviews using the Amazon dataset. 
Compares AI-generated reviews against real Amazon reviews (Ground Truth).
"""

import os
import sys
import django
import pandas as pd
import numpy as np
from pathlib import Path
from rouge_score import rouge_scorer
import math

# Setup Django
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "mirror_app"))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "mirror_app.settings")
django.setup()

from core.review_engine import ReviewEngine
from recommendations.models import ReviewBank

def calculate_rmse(errors):
    if not errors: return 0.0
    return math.sqrt(sum(e**2 for e in errors) / len(errors))

def run_evaluation(n_samples=10):
    print("=" * 60)
    print("TASK A EVALUATION: LINGUISTIC FIDELITY & RATING ACCURACY")
    print("=" * 60)
    print(f"Direct comparison using {n_samples} samples from ReviewBank...\n")

    engine = ReviewEngine()
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    # Get random samples from ReviewBank
    reviews = list(ReviewBank.objects.all().order_by('?')[:n_samples])
    
    results = []
    rating_errors = []

    for rev in reviews:
        print(f"Testing Item: {rev.title[:50]}... ({rev.category})")

        profile = {
            "ocean": {
                "O": rev.ocean_o,
                "C": rev.ocean_c,
                "E": rev.ocean_e,
                "A": rev.ocean_a,
                "N": rev.ocean_n
            },
            "dominant_archetype": "pragmatist", # Fallback
            "voice_profile": {
                "length": "medium", "tone": "casual", "rating_bias": "balanced", "focus": "quality"
            }
        }

        # Try to match the archetype if possible for better prompt
        from core.ocean.archetypes import ArchetypeMatcher
        matcher = ArchetypeMatcher()
        match = matcher.match(profile["ocean"])
        profile["dominant_archetype"] = match["dominant"]
        profile["voice_profile"] = match["archetype"]["voice_traits"]

        try:
            # 1. Generate AI Review
            ai_result = engine.generate_review(
                profile=profile,
                product_name=rev.title,
                category=rev.category
            )

            # 2. Compare with Ground Truth (The original review in the bank)
            scores = scorer.score(rev.text, ai_result["review"])
            rouge_l = scores['rougeL'].fmeasure
            
            # 3. Calculate Rating Error
            error = abs(ai_result["predicted_rating"] - rev.rating)
            rating_errors.append(error)

            results.append({
                "item": rev.title[:20],
                "rouge_l": rouge_l,
                "rating_error": error
            })
            print(f"  Result: ROUGE-L={rouge_l:.4f}, Rating Err={error:.1f}")

        except Exception as e:
            print(f"  Error testing item {rev.title}: {e}")

    # Aggregate
    if not results:
        print("No results to display.")
        return

    res_df = pd.DataFrame(results)
    rmse = calculate_rmse(rating_errors)

    print("\n" + "=" * 60)
    print("FINAL TASK A PERFORMANCE REPORT")
    print("-" * 60)
    print(f"Mean ROUGE-L (Linguistic Fidelity): {res_df['rouge_l'].mean():.4f}")
    print(f"Mean Absolute Error (Rating):       {res_df['rating_error'].mean():.4f}")
    print(f"RMSE (Rating Accuracy):            {rmse:.4f}")
    print("=" * 60)

if __name__ == "__main__":
    run_evaluation(n_samples=5) # Keeping it small for speed as requested
