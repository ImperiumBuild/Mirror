"""
eval_task_b.py
--------------
Evaluates the Recommendation Engine using the 'Amazon Cluster' strategy.
Calculates NDCG@10 and Hit Rate by comparing cluster-based recommendations
against actual user history.
"""

import os
import sys
import django
import pandas as pd
import numpy as np
from pathlib import Path
import math

# Setup Django
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "mirror_app"))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "mirror_app.settings")
django.setup()

from recommendations.models import ArchetypeAffinity
from core.recommendation_engine import RecommendationEngine

def calculate_ndcg(recommended_titles, hidden_titles, k=10):
    """Calculates Normalized Discounted Cumulative Gain at k."""
    dcg = 0.0
    for i, title in enumerate(recommended_titles[:k]):
        if title in hidden_titles:
            dcg += 1.0 / math.log2(i + 2)
            
    # IDCG (Ideal DCG - if all hidden items were at the top)
    idcg = 0.0
    for i in range(min(len(hidden_titles), k)):
        idcg += 1.0 / math.log2(i + 2)
        
    return dcg / idcg if idcg > 0 else 0.0

def run_evaluation():
    print("=" * 60)
    print("TASK B EVALUATION: CLUSTER-BASED RECOMMENDATIONS")
    print("=" * 60)

    # Load users from archetypes CSV
    users_csv = BASE_DIR / "data" / "processed" / "user_archetypes.csv"
    if not users_csv.exists():
        print(f"ERROR: {users_csv} not found.")
        return
        
    df = pd.read_csv(users_csv)
    
    # We need the product mapping. We'll simulate 'hidden' reviews by 
    # taking a subset of items associated with that archetype in ReviewBank.
    # (Since we don't have the full raw user-item matrix for 30k users in memory)
    
    archetypes = df["dominant_archetype"].unique()
    results = []

    engine = RecommendationEngine()

    print(f"Testing across {len(archetypes)} archetypes...\n")

    for arch_id in archetypes:
        # Get users of this type
        arch_users = df[df["dominant_archetype"] == arch_id]
        if arch_users.empty: continue
        
        # Take a representative user profile
        sample_user = arch_users.iloc[0]
        profile = {
            "ocean": {
                "O": sample_user["ocean_O"],
                "C": sample_user["ocean_C"],
                "E": sample_user["ocean_E"],
                "A": sample_user["ocean_A"],
                "N": sample_user["ocean_N"]
            },
            "dominant_archetype": arch_id,
            "voice_profile": {"focus": "reliability and quality"} # Mock
        }

        # Categories to test
        for category in ["books", "electronics", "movies"]:
            # 1. Get 'Hidden Truth' (Items real users in this cluster reviewed highly)
            # We take items from the database that HAVE reviews from this archetype
            truth_items = list(ArchetypeAffinity.objects.filter(
                archetype_id=arch_id, 
                category=category,
                avg_rating__gte=4.0
            ).order_by("-review_count").values_list("item_title", flat=True)[:20])

            if not truth_items: continue

            # Split Truth: 60% for 'training' (not really used in this zero-shot eval), 
            # 40% for 'testing' (the items we want to hit)
            split_idx = int(len(truth_items) * 0.6)
            hidden_truth = set(truth_items[split_idx:])

            # 2. Generate Recommendations
            try:
                recs = engine.recommend(profile=profile, category=category, n=10)
                rec_titles = [r["title"] for r in recs]
                
                # 3. Calculate Metrics
                hits = len(set(rec_titles) & hidden_truth)
                hr   = hits / len(hidden_truth) if hidden_truth else 0
                ndcg = calculate_ndcg(rec_titles, hidden_truth, k=10)
                
                results.append({
                    "archetype": arch_id,
                    "category": category,
                    "ndcg_10": ndcg,
                    "hit_rate": hr,
                    "hits": hits
                })
            except Exception as e:
                print(f"  Error testing {arch_id}/{category}: {e}")

    # Aggregate Results
    if not results:
        print("No results to display.")
        return

    res_df = pd.DataFrame(results)
    print(f"{'Archetype':<15} | {'Category':<12} | {'NDCG@10':<10} | {'Hit Rate'}")
    print("-" * 60)
    for _, row in res_df.iterrows():
        print(f"{row['archetype']:<15} | {row['category']:<12} | {row['ndcg_10']:<10.4f} | {row['hit_rate']*100:>5.1f}%")

    print("\n" + "=" * 60)
    print(f"OVERALL MEAN NDCG@10: {res_df['ndcg_10'].mean():.4f}")
    print(f"OVERALL MEAN HIT RATE: {res_df['hit_rate'].mean()*100:.1f}%")
    print("=" * 60)

if __name__ == "__main__":
    run_evaluation()
