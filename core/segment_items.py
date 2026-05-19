"""
core/segment_items.py
---------------------
Performs the 'Mixture Model' segmentation on our review database.
1. Extracts linguistic fingerprints from item reviews.
2. Uses GMM to cluster items into personality-aligned buckets.
3. Maps these buckets to our 8 archetypes.
"""

import os
import sys
import django
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# Setup Django
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "mirror_app"))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "mirror_app.settings")
django.setup()

from recommendations.models import ReviewBank, ArchetypeAffinity
from core.ocean.Liwc import analyse_text

def run_segmentation():
    print("--- Starting Item-Persona Segmentation (GMM) ---")
    
    # 1. Fetch items and their reviews
    # We group by title/category to see 'who' likes what
    items = ReviewBank.objects.all()
    print(f"Analyzing {items.count()} reviews...")
    
    data = []
    for item in items:
        # We use the OCEAN scores already associated with the reviews in the bank
        # These represent the 'author' of that review.
        data.append({
            "id": item.id,
            "title": item.title,
            "category": item.category,
            "O": item.ocean_o,
            "C": item.ocean_c,
            "E": item.ocean_e,
            "A": item.ocean_a,
            "N": item.ocean_n,
            "rating": item.rating
        })
    
    df = pd.DataFrame(data)
    
    # 2. Group by item to find the 'Ideal Persona' for each product
    # An item is good for a cluster if people in that cluster rate it highly
    item_profiles = df.groupby(['title', 'category']).agg({
        'O': 'mean', 'C': 'mean', 'E': 'mean', 'A': 'mean', 'N': 'mean',
        'rating': 'mean',
        'id': 'count'
    }).rename(columns={'id': 'review_count'}).reset_index()
    
    # 3. Perform GMM Clustering on the Items
    X = item_profiles[['O', 'C', 'E', 'A', 'N']].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"Clustering {len(item_profiles)} products into 8 personality buckets...")
    gmm = GaussianMixture(n_components=8, random_state=42)
    item_profiles['cluster'] = gmm.fit_predict(X_scaled)
    
    # 4. Map Clusters to our 8 Archetypes
    # (Simplified: find which archetype centroid is closest to the cluster mean)
    from core.ocean.archetypes import ArchetypeMatcher
    matcher = ArchetypeMatcher()
    archetypes = matcher.get_all_archetypes()
    
    cluster_to_arch = {}
    for c in range(8):
        cluster_mean = item_profiles[item_profiles['cluster'] == c][['O', 'C', 'E', 'A', 'N']].mean().to_dict()
        match = matcher.match(cluster_mean)
        cluster_to_arch[c] = match['dominant']
    
    # 5. Populate ArchetypeAffinity
    print("Populating Affinity Database...")
    ArchetypeAffinity.objects.all().delete()
    
    to_create = []
    for _, row in item_profiles.iterrows():
        arch_id = cluster_to_arch[row['cluster']]
        
        # Affinity = (Rating weight) * (Popularity weight)
        affinity = (row['rating'] / 5.0) * (np.log1p(row['review_count']))
        
        to_create.append(ArchetypeAffinity(
            archetype_id = arch_id,
            item_title   = row['title'],
            category     = row['category'],
            affinity_score = round(affinity, 4),
            review_count = row['review_count'],
            avg_rating   = round(row['rating'], 2)
        ))
    
    ArchetypeAffinity.objects.bulk_create(to_create)
    print(f"SUCCESS: Segmented items into {len(cluster_to_arch)} archetype-mapped clusters.")

if __name__ == "__main__":
    run_segmentation()
