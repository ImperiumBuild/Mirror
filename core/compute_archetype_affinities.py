"""
compute_archetype_affinities.py
------------------------------
Analyses the ReviewBank and segments products by the 8 personality archetypes.
Populates the ArchetypeAffinity table with 'Cluster Weights'.
"""

import os
import sys
import django
from pathlib import Path
from collections import defaultdict
import math

# Setup Django
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "mirror_app"))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "mirror_app.settings")
django.setup()

from recommendations.models import ReviewBank, ArchetypeAffinity
from core.ocean.archetypes import ArchetypeMatcher

def compute():
    print("--- Computing Archetype-Product Affinities ---")
    
    # 1. Clear existing affinities
    ArchetypeAffinity.objects.all().delete()
    
    # 2. Initialize Matcher
    matcher = ArchetypeMatcher()
    
    # 3. Aggregate data
    # Structure: affinities[archetype_id][(category, title)] = [ratings]
    affinities = defaultdict(lambda: defaultdict(list))
    
    reviews = ReviewBank.objects.all()
    print(f"Processing {reviews.count()} reviews...")
    
    for rev in reviews:
        # Determine archetype of this review's author style
        ocean = {
            "O": rev.ocean_o,
            "C": rev.ocean_c,
            "E": rev.ocean_e,
            "A": rev.ocean_a,
            "N": rev.ocean_n
        }
        match = matcher.match(ocean)
        arch_id = match["dominant"]
        
        item_key = (rev.category, rev.title)
        affinities[arch_id][item_key].append(rev.rating)
        
    # 4. Calculate scores and save
    to_create = []
    for arch_id, items in affinities.items():
        print(f"  Archetype: {arch_id} ({len(items)} unique items)")
        for (category, title), ratings in items.items():
            count = len(ratings)
            avg_rating = sum(ratings) / count
            
            # Affinity Formula: Combination of popularity and sentiment
            # We use log(count + 1) to dampen high-volume outliers
            # but still give them more weight than single-review items.
            score = (math.log(count + 1) * 2.0) * (avg_rating / 5.0)
            
            to_create.append(ArchetypeAffinity(
                archetype_id = arch_id,
                item_title   = title,
                category     = category,
                affinity_score = round(score, 4),
                review_count = count,
                avg_rating   = round(avg_rating, 2)
            ))
            
            # Batch save every 500 items
            if len(to_create) >= 500:
                ArchetypeAffinity.objects.bulk_create(to_create)
                to_create = []
                
    if to_create:
        ArchetypeAffinity.objects.bulk_create(to_create)
        
    print(f"\nSUCCESS: Populated {ArchetypeAffinity.objects.count()} affinity records.")

if __name__ == "__main__":
    compute()
