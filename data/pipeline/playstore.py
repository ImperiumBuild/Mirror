"""
playstore.py
------------
Scrapes real user reviews from Google Play Store across app categories.
No API key needed — uses google-play-scraper.

Output:
    data/processed/apps_user_profiles.csv   - per-user behavioural fingerprints
    data/processed/review_bank_apps.json    - reviews tagged with OCEAN scores
                                              for the pairwise UI
"""

import os
import json
import time
import random
import warnings
import pandas as pd
import numpy as np
from tqdm import tqdm
from google_play_scraper import reviews, Sort, app as get_app_info

warnings.filterwarnings("ignore")

OUT_DIR = "data/processed"
os.makedirs(OUT_DIR, exist_ok=True)

# ── app list ──────────────────────────────────────────────────────────────────
# Curated list across 5 sub-categories to ensure diverse review styles.
# Mix of popular and niche apps to get varied user types.

APPS = {
    "productivity": [
        "com.todoist",                    # Todoist
        "com.notion.id",                  # Notion
        "com.microsoft.todos",            # Microsoft To Do
        "com.evernote",                   # Evernote
        "com.google.android.keep",        # Google Keep
    ],
    "social": [
        "com.instagram.android",          # Instagram
        "com.twitter.android",            # Twitter/X
        "com.whatsapp",                   # WhatsApp
        "com.snapchat.android",           # Snapchat
        "com.reddit.frontpage",           # Reddit
    ],
    "entertainment": [
        "com.spotify.music",              # Spotify
        "com.netflix.mediaclient",        # Netflix
        "com.google.android.youtube",     # YouTube
        "com.amazon.avod.thirdpartyclient", # Prime Video
        "com.disney.disneyplus",          # Disney+
    ],
    "utilities": [
        "com.google.android.apps.maps",   # Google Maps
        "com.duolingo",                   # Duolingo
        "com.ubercab",                    # Uber
        "com.weather.Weather",            # The Weather Channel
        "com.adobe.reader",               # Adobe Acrobat
    ],
    "gaming": [
        "com.kiloo.subwaysurf",           # Subway Surfers
        "com.mojang.minecraftpe",         # Minecraft
        "com.supercell.clashofclans",     # Clash of Clans
        "com.roblox.client",              # Roblox
        "com.nintendo.zara",              # Mario Kart Tour
    ],
    "banking": ["com.kudabank.app",   # Kuda Bank - digital banking app
                "com.moniepoint.personal",  # Moniepoint - digital banking app
                "com.opay.android",         # OPay - digital banking app
                "com.gtbank.gtworldv1",    # GTBank Mobile - digital banking app
                "com.wemabank.alat.prod",  # ALAT by Wema - digital banking app
                "com.firstbank.firstmobile",  # FirstBank Mobile - digital banking app
                "com.accessbank.nextgen",  # Access Bank Mobile - digital banking app
                "com.uba.vericash",      # UBA Mobile Banking - digital banking app
                "com.transsnet.palmpay",  # PalmPay - digital banking app
                "com.zenithBank.eazymoney"  # Zenith Bank Mobile - digital banking app
    ],
}

# reviews to fetch per app
REVIEWS_PER_APP = 200


# ── reuse text/rating feature extractors from amazon.py ──────────────────────
# copied here so playstore.py is self-contained

import re
import string

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


def clean_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def tokenize(text: str) -> list:
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.split()


def extract_text_features(texts: list) -> dict:
    all_tokens = []
    all_sentences = []
    exclamation_count = 0
    question_count = 0
    total_chars = 0

    for raw in texts:
        cleaned = clean_text(raw)
        tokens = tokenize(cleaned)
        all_tokens.extend(tokens)
        sentences = re.split(r"[.!?]+", raw.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        all_sentences.extend(sentences)
        exclamation_count += raw.count("!")
        question_count += raw.count("?")
        total_chars += len(raw)

    total_tokens = len(all_tokens)
    if total_tokens == 0:
        return {}

    first_person = sum(1 for t in all_tokens if t in {
        "i", "me", "my", "mine", "myself"})
    second_person = sum(1 for t in all_tokens if t in {
        "you", "your", "yours", "yourself"})
    third_person = sum(1 for t in all_tokens if t in {
        "he", "she", "they", "them", "his", "her", "their"})

    pos_count = sum(1 for t in all_tokens if t in POS_WORDS)
    neg_count = sum(1 for t in all_tokens if t in NEG_WORDS)
    certainty_count = sum(1 for t in all_tokens if t in CERTAINTY_WORDS)
    hedge_count = sum(1 for t in all_tokens if t in HEDGE_WORDS)

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
        "generosity_score": round((ratings.mean() - 1) / 4, 4),
    }


def derive_ocean_proxies(text_feats: dict, rating_feats: dict) -> dict:
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
    return {
        "ocean_O": round(float(np.clip(o, 0, 1)), 4),
        "ocean_C": round(float(np.clip(c, 0, 1)), 4),
        "ocean_E": round(float(np.clip(e, 0, 1)), 4),
        "ocean_A": round(float(np.clip(a, 0, 1)), 4),
        "ocean_N": round(float(np.clip(n, 0, 1)), 4),
    }


# ── scraper ───────────────────────────────────────────────────────────────────

def scrape_app_reviews(app_id: str, app_name: str,
                        sub_category: str, count: int = 200) -> list:
    """
    Fetch reviews for a single app. Returns list of dicts.
    Fetches both NEWEST and MOST_RELEVANT to get review style diversity.
    """
    all_reviews = []

    for sort_order in [Sort.NEWEST, Sort.MOST_RELEVANT]:
        try:
            result, _ = reviews(
                app_id,
                lang="en",
                country="us",
                sort=sort_order,
                count=count // 2,
            )
            for r in result:
                # skip empty or very short reviews
                if not r.get("content") or len(r["content"]) < 20:
                    continue
                all_reviews.append({
                    "user_id":      r.get("userName", "unknown"),
                    "app_id":       app_id,
                    "app_name":     app_name,
                    "sub_category": sub_category,
                    "category":     "apps",
                    "rating":       float(r["score"]),
                    "text":         r["content"],
                    "thumbs_up":    r.get("thumbsUpCount", 0),
                    "at":           str(r.get("at", "")),
                })
            # be polite to Google's servers
            time.sleep(random.uniform(0.5, 1.5))

        except Exception as e:
            print(f"    WARNING: failed {app_id} ({sort_order}) — {e}")
            continue

    return all_reviews


def scrape_all_apps() -> pd.DataFrame:
    """Scrape all apps across all sub-categories."""
    all_reviews = []

    for sub_category, app_ids in APPS.items():
        print(f"\n  Scraping {sub_category} apps...")

        for app_id in tqdm(app_ids, desc=f"    {sub_category}"):
            # get app name first
            try:
                info = get_app_info(app_id, lang="en", country="us")
                app_name = info.get("title", app_id)
            except Exception:
                app_name = app_id

            app_reviews = scrape_app_reviews(
                app_id, app_name, sub_category, count=REVIEWS_PER_APP)
            all_reviews.extend(app_reviews)

            # pause between apps
            time.sleep(random.uniform(1.0, 2.5))

    df = pd.DataFrame(all_reviews)
    print(f"\n  Total app reviews scraped: {len(df):,}")
    print(f"  Unique users: {df['user_id'].nunique():,}")
    print(f"  Apps covered: {df['app_id'].nunique():,}")
    return df


# ── build app user profiles ───────────────────────────────────────────────────

def build_app_user_profiles(df: pd.DataFrame, min_reviews: int = 2) -> pd.DataFrame:
    """
    App store users rarely review more than 2-3 apps so we lower
    the min_reviews threshold to 2 (vs 3 for Amazon).
    """
    counts = df.groupby("user_id").size()
    eligible = counts[counts >= min_reviews].index
    df_eligible = df[df["user_id"].isin(eligible)].copy()

    print(f"\nBuilding app user profiles for "
          f"{len(eligible):,} users ({min_reviews}+ reviews)...")

    profiles = []

    for user_id, group in tqdm(df_eligible.groupby("user_id"),
                                total=len(eligible),
                                desc="  Processing"):
        texts = group["text"].tolist()
        ratings = group["rating"]

        text_feats = extract_text_features(texts)
        if not text_feats:
            continue

        rating_feats = extract_rating_features(ratings)
        ocean = derive_ocean_proxies(text_feats, rating_feats)

        profile = {
            "user_id":  user_id,
            "category": "apps",
            **rating_feats,
            **text_feats,
            **ocean,
        }
        profiles.append(profile)

    return pd.DataFrame(profiles)


# ── build app review bank ─────────────────────────────────────────────────────

def build_app_review_bank(df: pd.DataFrame,
                           profiles_df: pd.DataFrame,
                           target: int = 300) -> list:
    """
    Select diverse app reviews for the pairwise UI.
    Samples across star ratings and sub-categories.
    """
    print("\nBuilding app review bank...")

    # only reviews long enough to be useful
    df_long = df[df["text"].str.len() >= 50].copy()

    # join with OCEAN profiles
    if len(profiles_df) > 0:
        df_long = df_long.merge(
            profiles_df[["user_id", "ocean_O", "ocean_C", "ocean_E",
                          "ocean_A", "ocean_N", "generosity_score",
                          "avg_review_length"]],
            on="user_id",
            how="left"
        )
        # fill NaN OCEAN scores with neutral 0.5 for unprofileable users
        ocean_cols = ["ocean_O", "ocean_C", "ocean_E", "ocean_A", "ocean_N",
                      "generosity_score", "avg_review_length"]
        df_long[ocean_cols] = df_long[ocean_cols].fillna(0.5)
    else:
        # no profiles built — add neutral scores
        for col in ["ocean_O", "ocean_C", "ocean_E", "ocean_A", "ocean_N"]:
            df_long[col] = 0.5
        df_long["generosity_score"] = 0.5
        df_long["avg_review_length"] = df_long["text"].str.len()

    df_long = df_long.reset_index(drop=True)

    bank = []
    per_star = target // 5

    for star in [1.0, 2.0, 3.0, 4.0, 5.0]:
        bucket = df_long[df_long["rating"] == star]
        if bucket.empty:
            continue
        sampled = bucket.sample(min(len(bucket), per_star), random_state=42)
        records = sampled[[
            "category", "app_id", "app_name", "sub_category",
            "rating", "text",
            "ocean_O", "ocean_C", "ocean_E", "ocean_A", "ocean_N",
            "generosity_score", "avg_review_length"
        ]].rename(columns={"app_name": "title", "app_id": "item_id"}).to_dict(orient="records")
        bank.extend(records)

    print(f"  apps: {len(bank)} reviews added to bank")
    return bank


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Play Store Pipeline")
    print("=" * 60)

    # ── scrape ──
    df = scrape_all_apps()

    if df.empty:
        print("ERROR: no reviews scraped. Check your internet connection.")
        return

    # save raw scrape
    raw_out = os.path.join("data/raw", "playstore_raw.csv")
    df.to_csv(raw_out, index=False)
    print(f"\n✓ Raw scrape saved → {raw_out}")

    # ── profiles ──
    profiles_df = build_app_user_profiles(df, min_reviews=2)

    if not profiles_df.empty:
        out_profiles = os.path.join(OUT_DIR, "apps_user_profiles.csv")
        profiles_df.to_csv(out_profiles, index=False)
        print(f"✓ apps_user_profiles.csv saved → {len(profiles_df):,} users")
    else:
        print("  NOTE: not enough multi-review users for profiles. "
              "Review bank will use neutral OCEAN scores.")
        profiles_df = pd.DataFrame()

    # ── review bank ──
    bank = build_app_review_bank(df, profiles_df, target=300)
    out_bank = os.path.join(OUT_DIR, "review_bank_apps.json")
    with open(out_bank, "w") as f:
        json.dump(bank, f, indent=2)
    print(f"✓ review_bank_apps.json saved → {len(bank):,} reviews")

    # ── summary ──
    print("\n── Day 2a complete ─────────────────────────────────────")
    print(f"   Raw reviews     : {len(df):,}")
    print(f"   Apps covered    : {df['app_id'].nunique():,}")
    print(f"   Sub-categories  : {', '.join(APPS.keys())}")
    print(f"   User profiles   : {len(profiles_df):,}")
    print(f"   Review bank     : {len(bank):,} reviews")
    print(f"   Output dir      : {OUT_DIR}/")


if __name__ == "__main__":
    main()