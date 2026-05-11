"""
amazon.py
---------
Day 1 pipeline. Loads the three Amazon review CSVs, cleans them,
extracts per-user behavioural fingerprints from text + rating signals,
and outputs two files:

    data/processed/user_profiles.csv   - one row per user (3+ reviews)
    data/processed/review_bank.json    - real reviews tagged by category
                                         with title, for pairwise UI
"""

import os
import re
import json
import string
import warnings
import pandas as pd
import numpy as np
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── paths ────────────────────────────────────────────────────────────────────
RAW_DIR = "data/raw"
OUT_DIR = "data/processed"
os.makedirs(OUT_DIR, exist_ok=True)

# ── word lists for LIWC-style analysis ───────────────────────────────────────
# These are lightweight proxy lists, not the full LIWC dictionary
POS_WORDS = {
    "love", "great", "excellent", "amazing", "wonderful", "fantastic",
    "perfect", "awesome", "best", "good", "nice", "happy", "pleased",
    "recommend", "beautiful", "enjoy", "enjoyed", "impressed", "solid",
    "quality", "worth", "brilliant", "superb", "outstanding", "delightful"
}

NEG_WORDS = {
    "bad", "terrible", "awful", "horrible", "worst", "poor", "useless",
    "broken", "waste", "disappointed", "disappointing", "defective", "cheap",
    "flimsy", "returned", "return", "refund", "fail", "failed", "garbage",
    "junk", "scam", "fraud", "misleading", "damaged", "missing", "wrong",
    "never", "avoid", "regret", "annoying", "frustrated", "rubbish"
}

CERTAINTY_WORDS = {
    "definitely", "absolutely", "certainly", "clearly", "obviously",
    "always", "never", "completely", "totally", "exactly", "must",
    "will", "without", "undoubtedly", "guaranteed", "every", "all"
}

HEDGE_WORDS = {
    "maybe", "perhaps", "possibly", "might", "could", "seems", "appears",
    "somewhat", "kind", "sort", "little", "fairly", "rather", "quite",
    "almost", "guess", "think", "believe", "probably", "generally"
}


# ── helpers ───────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Lowercase, strip HTML tags and extra whitespace."""
    text = str(text)
    text = re.sub(r"<[^>]+>", " ", text)       # remove HTML
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer."""
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.split()


def extract_text_features(texts: list[str]) -> dict:
    """
    Given a list of review texts for a single user, compute
    LIWC-style linguistic features aggregated across all their reviews.
    """
    all_tokens = []
    all_sentences = []
    exclamation_count = 0
    question_count = 0
    total_chars = 0

    for raw in texts:
        cleaned = clean_text(raw)
        tokens = tokenize(cleaned)
        all_tokens.extend(tokens)

        # sentence-level
        sentences = re.split(r"[.!?]+", raw.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        all_sentences.extend(sentences)

        exclamation_count += raw.count("!")
        question_count += raw.count("?")
        total_chars += len(raw)

    total_tokens = len(all_tokens)
    if total_tokens == 0:
        return {}

    token_set = set(all_tokens)

    # pronoun ratios
    first_person = sum(1 for t in all_tokens if t in {
        "i", "me", "my", "mine", "myself"})
    second_person = sum(1 for t in all_tokens if t in {
        "you", "your", "yours", "yourself"})
    third_person = sum(1 for t in all_tokens if t in {
        "he", "she", "they", "them", "his", "her", "their"})

    # sentiment word ratios
    pos_count = sum(1 for t in all_tokens if t in POS_WORDS)
    neg_count = sum(1 for t in all_tokens if t in NEG_WORDS)
    certainty_count = sum(1 for t in all_tokens if t in CERTAINTY_WORDS)
    hedge_count = sum(1 for t in all_tokens if t in HEDGE_WORDS)

    # vocabulary richness (type-token ratio, capped at 500 tokens)
    sample = all_tokens[:500]
    ttr = len(set(sample)) / len(sample) if sample else 0

    avg_sentence_len = (
        total_tokens / len(all_sentences) if all_sentences else 0
    )

    return {
        "first_person_ratio":  first_person / total_tokens,
        "second_person_ratio": second_person / total_tokens,
        "third_person_ratio":  third_person / total_tokens,
        "pos_word_ratio":      pos_count / total_tokens,
        "neg_word_ratio":      neg_count / total_tokens,
        "certainty_ratio":     certainty_count / total_tokens,
        "hedge_ratio":         hedge_count / total_tokens,
        "exclamation_ratio":   exclamation_count / total_tokens,
        "question_ratio":      question_count / total_tokens,
        "vocab_richness":      ttr,
        "avg_sentence_length": avg_sentence_len,
        "avg_review_length":   total_chars / len(texts),
        "avg_token_count":     total_tokens / len(texts),
    }


def extract_rating_features(ratings: pd.Series) -> dict:
    """
    Compute rating behaviour features from a user's rating history.
    These are the primary signals for the rating prediction model.
    """
    counts = ratings.value_counts()
    total = len(ratings)

    return {
        "avg_rating":       round(ratings.mean(), 4),
        "rating_std":       round(ratings.std(), 4) if len(ratings) > 1 else 0,
        "rating_min":       int(ratings.min()),
        "rating_max":       int(ratings.max()),
        "pct_1_star":       round(counts.get(1.0, 0) / total, 4),
        "pct_2_star":       round(counts.get(2.0, 0) / total, 4),
        "pct_3_star":       round(counts.get(3.0, 0) / total, 4),
        "pct_4_star":       round(counts.get(4.0, 0) / total, 4),
        "pct_5_star":       round(counts.get(5.0, 0) / total, 4),
        "review_count":     total,
        # harsh/generous score: 0 = very harsh, 1 = very generous
        "generosity_score": round((ratings.mean() - 1) / 4, 4),
    }


def derive_ocean_proxies(text_feats: dict, rating_feats: dict) -> dict:
    """
    Map extracted features to OCEAN dimension proxy scores (0–1).
    These are approximations based on published personality-language research:

      O (Openness)         → vocabulary richness + long reviews + hedging
      C (Conscientiousness)→ consistent ratings (low std) + specific language
      E (Extraversion)     → exclamation marks + 2nd person + positive words
      A (Agreeableness)    → generosity score + positive ratio + low negativity
      N (Neuroticism)      → neg word ratio + high rating std + low avg rating
    """
    o = np.mean([
        text_feats.get("vocab_richness", 0),
        min(text_feats.get("avg_token_count", 0) / 200, 1),
        text_feats.get("hedge_ratio", 0) * 5,
    ])

    c = np.mean([
        1 - min(rating_feats.get("rating_std", 0) / 2, 1),
        text_feats.get("certainty_ratio", 0) * 5,
        min(text_feats.get("avg_token_count", 0) / 100, 1),
    ])

    e = np.mean([
        min(text_feats.get("exclamation_ratio", 0) * 20, 1),
        text_feats.get("second_person_ratio", 0) * 5,
        text_feats.get("pos_word_ratio", 0) * 5,
    ])

    a = np.mean([
        rating_feats.get("generosity_score", 0),
        text_feats.get("pos_word_ratio", 0) * 4,
        1 - text_feats.get("neg_word_ratio", 0) * 5,
    ])

    n = np.mean([
        text_feats.get("neg_word_ratio", 0) * 5,
        min(rating_feats.get("rating_std", 0) / 2, 1),
        1 - rating_feats.get("generosity_score", 0),
    ])

    # clip all to [0, 1]
    return {
        "ocean_O": round(float(np.clip(o, 0, 1)), 4),
        "ocean_C": round(float(np.clip(c, 0, 1)), 4),
        "ocean_E": round(float(np.clip(e, 0, 1)), 4),
        "ocean_A": round(float(np.clip(a, 0, 1)), 4),
        "ocean_N": round(float(np.clip(n, 0, 1)), 4),
    }


# ── load & clean ──────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    print("Loading CSVs...")
    books = pd.read_csv(
        os.path.join(RAW_DIR, "books_final.csv"), low_memory=False)
    electronics = pd.read_csv(
        os.path.join(RAW_DIR, "electronics_final.csv"), low_memory=False)
    movies = pd.read_csv(
        os.path.join(RAW_DIR, "movies_final.csv"), low_memory=False)

    books["category"] = "books"
    electronics["category"] = "electronics"
    movies["category"] = "movies"

    df = pd.concat([books, electronics, movies], ignore_index=True)

    # drop rows with no review text
    df = df[df["text"].notna()].copy()
    df["text"] = df["text"].astype(str)

    # clean price column
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    print(f"  Total reviews after cleaning: {len(df):,}")
    print(f"  Unique users: {df['user_id'].nunique():,}")
    return df


# ── build user profiles ───────────────────────────────────────────────────────

def build_user_profiles(df: pd.DataFrame, min_reviews: int = 3) -> pd.DataFrame:
    """
    For each user with >= min_reviews, compute their full behavioural
    fingerprint: rating features + text features + OCEAN proxies.
    """
    # filter to users with enough reviews
    counts = df.groupby("user_id").size()
    eligible = counts[counts >= min_reviews].index
    df_eligible = df[df["user_id"].isin(eligible)].copy()

    print(f"\nBuilding profiles for {len(eligible):,} users "
          f"({min_reviews}+ reviews)...")

    profiles = []

    for user_id, group in tqdm(df_eligible.groupby("user_id"),
                                total=len(eligible),
                                desc="  Processing users"):
        texts = group["text"].tolist()
        ratings = group["rating"]

        text_feats = extract_text_features(texts)
        if not text_feats:
            continue

        rating_feats = extract_rating_features(ratings)
        ocean = derive_ocean_proxies(text_feats, rating_feats)

        # category distribution
        cat_counts = group["category"].value_counts(normalize=True)

        profile = {
            "user_id": user_id,
            "category_books":       round(cat_counts.get("books", 0), 4),
            "category_electronics": round(cat_counts.get("electronics", 0), 4),
            "category_movies":      round(cat_counts.get("movies", 0), 4),
            **rating_feats,
            **text_feats,
            **ocean,
        }
        profiles.append(profile)

    return pd.DataFrame(profiles)


# ── build review bank ─────────────────────────────────────────────────────────

def build_review_bank(df: pd.DataFrame,
                       profiles_df: pd.DataFrame,
                       reviews_per_category: int = 300) -> list[dict]:
    print("\nBuilding review bank...")

    titled = df[df["title"].notna()].copy()

    titled = titled.merge(
        profiles_df[["user_id", "ocean_O", "ocean_C", "ocean_E",
                      "ocean_A", "ocean_N", "generosity_score",
                      "avg_review_length"]],
        on="user_id",
        how="inner"
    )

    titled = titled[titled["text"].str.len() >= 50].reset_index(drop=True)

    bank = []

    for category in ["books", "electronics", "movies"]:
        cat_df = titled[titled["category"] == category].reset_index(drop=True)
        if cat_df.empty:
            print(f"  WARNING: no titled reviews for {category}")
            continue

        # sample per rating bucket without groupby
        frames = []
        for star in [1.0, 2.0, 3.0, 4.0, 5.0]:
            bucket = cat_df[cat_df["rating"] == star]
            if bucket.empty:
                continue
            frames.append(bucket.sample(
                min(len(bucket), reviews_per_category // 5),
                random_state=42
            ))

        sampled = pd.concat(frames, ignore_index=True)

        # build bank as list of dicts directly from DataFrame
        records = sampled[[
            "category", "item_id", "title", "rating", "text",
            "ocean_O", "ocean_C", "ocean_E", "ocean_A", "ocean_N",
            "generosity_score", "avg_review_length"
        ]].to_dict(orient="records")

        bank.extend(records)
        print(f"  {category}: {len(records)} reviews added to bank")

    return bank

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    df = load_data()

    # ── user profiles ──
    profiles_df = build_user_profiles(df, min_reviews=3)
    out_profiles = os.path.join(OUT_DIR, "user_profiles.csv")
    profiles_df.to_csv(out_profiles, index=False)
    print(f"\n✓ user_profiles.csv saved → {len(profiles_df):,} users")
    print(profiles_df.describe().round(3).to_string())

    # ── review bank ──
    bank = build_review_bank(df, profiles_df, reviews_per_category=300)
    out_bank = os.path.join(OUT_DIR, "review_bank.json")
    with open(out_bank, "w") as f:
        json.dump(bank, f, indent=2)
    print(f"\n✓ review_bank.json saved → {len(bank):,} reviews")

    print("\n── Day 1 complete ──────────────────────────────────────────")
    print(f"   user_profiles.csv : {out_profiles}")
    print(f"   review_bank.json  : {out_bank}")


if __name__ == "__main__":
    main()