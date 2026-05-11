from datasets import load_dataset
import pandas as pd
import os

########################################
# CONFIG
########################################

REVIEW_LIMIT = 200000
MIN_REVIEWS_PER_ITEM = 3
TARGET_META_COVERAGE = 0.7   # 70%
MAX_META_SCAN = 800000       # safety cutoff


########################################
# STEP 1: LOAD OR CREATE REVIEWS
########################################

def load_or_create_reviews(config_name, output_file):
    if os.path.exists(output_file):
        print(f"Loading existing {output_file}")
        return pd.read_csv(output_file)

    print(f"Fetching {config_name}...")

    ds = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        config_name,
        streaming=True,
        trust_remote_code=True
    )

    stream = ds["full"]

    rows = []
    for i, r in enumerate(stream):
        rows.append({
            "user_id": r.get("user_id"),
            "item_id": r.get("parent_asin"),
            "rating": r.get("rating"),
            "text": r.get("text"),
            "timestamp": r.get("timestamp"),
            "verified": r.get("verified_purchase")
        })

        if i >= REVIEW_LIMIT:
            break

    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)

    print(f"Saved {output_file}: {df.shape}")
    return df


########################################
# STEP 2: FILTER IMPORTANT ITEMS
########################################

def get_high_signal_items(df, min_reviews):
    counts = df["item_id"].value_counts()
    filtered = counts[counts >= min_reviews].index
    print(f"Items reduced: {len(counts)} → {len(filtered)}")
    return set(filtered)


########################################
# STEP 3: LOAD OR CREATE METADATA
########################################

def load_or_create_meta(meta_config, item_set, output_file):
    if os.path.exists(output_file):
        print(f"Loading existing {output_file}")
        return pd.read_csv(output_file)

    print(f"Fetching {meta_config}...")

    ds = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        meta_config,
        streaming=True,
        trust_remote_code=True
    )

    stream = ds["full"]

    rows = []
    seen = set()

    for i, r in enumerate(stream):
        item_id = r.get("parent_asin") or r.get("asin")

        if item_id in item_set and item_id not in seen:
            rows.append({
                "item_id": item_id,
                "title": r.get("title"),
                "category": r.get("category"),
                "description": r.get("description"),
                "price": r.get("price")
            })
            seen.add(item_id)

        # ✅ EARLY STOP (coverage-based)
        if len(seen) >= TARGET_META_COVERAGE * len(item_set):
            print(f"Stopped early at {len(seen)} items ({round(len(seen)/len(item_set),2)*100}%)")
            break

        # ✅ HARD STOP (safety)
        if i >= MAX_META_SCAN:
            print(f"Stopped due to scan limit ({MAX_META_SCAN})")
            break

    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)

    print(f"Saved {output_file}: {df.shape}")
    return df


########################################
# STEP 4: MERGE
########################################

def merge_and_save(reviews_df, meta_df, output_file):
    if os.path.exists(output_file):
        print(f"Skipping merge, {output_file} already exists")
        return

    merged = reviews_df.merge(meta_df, on="item_id", how="left")
    merged.to_csv(output_file, index=False)

    print(f"Saved merged file: {output_file}")


########################################
# RUN PIPELINE
########################################

# 1. Reviews
movies_df = load_or_create_reviews("raw_review_Movies_and_TV", "movies_reviews.csv")
books_df = load_or_create_reviews("raw_review_Books", "books_reviews.csv")
electronics_df = load_or_create_reviews("raw_review_Electronics", "electronics_reviews.csv")

# 2. Filter items (IMPORTANT)
movies_items = get_high_signal_items(movies_df, MIN_REVIEWS_PER_ITEM)
books_items = get_high_signal_items(books_df, MIN_REVIEWS_PER_ITEM)
electronics_items = get_high_signal_items(electronics_df, MIN_REVIEWS_PER_ITEM)

# 3. Metadata (optimized)
movies_meta = load_or_create_meta("raw_meta_Movies_and_TV", movies_items, "movies_meta.csv")
books_meta = load_or_create_meta("raw_meta_Books", books_items, "books_meta.csv")
electronics_meta = load_or_create_meta("raw_meta_Electronics", electronics_items, "electronics_meta.csv")

# 4. Merge
merge_and_save(movies_df, movies_meta, "movies_final.csv")
merge_and_save(books_df, books_meta, "books_final.csv")
merge_and_save(electronics_df, electronics_meta, "electronics_final.csv")

print("✅ Pipeline complete")