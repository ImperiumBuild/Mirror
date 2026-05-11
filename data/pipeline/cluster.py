"""
cluster.py
----------
Takes user_profiles.csv + apps_user_profiles.csv, clusters users
by their OCEAN vectors using a Gaussian Mixture Model (GMM).

GMM is used over KMeans because:
- Personality is not a hard category — it's a blend
- GMM gives soft assignments: every user gets a probability across
  all 8 archetypes, not just a single hard label
- GMM handles elliptical clusters of different sizes and densities
- The resulting probability vector is stored per user and used at
  runtime to enrich persona prompts

Outputs:
    data/processed/archetypes.json       - 8 archetype definitions with
                                           GMM-derived centroids, covariances,
                                           descriptions, and voice traits
    data/processed/user_archetypes.csv   - every user with full archetype
                                           probability vector + dominant type
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

OUT_DIR  = "data/processed"
os.makedirs(OUT_DIR, exist_ok=True)

OCEAN_COLS   = ["ocean_O", "ocean_C", "ocean_E", "ocean_A", "ocean_N"]
N_ARCHETYPES = 8
RANDOM_STATE = 42


# ── archetype definitions ─────────────────────────────────────────────────────
# Used ONLY for naming GMM components after fitting.
# Actual centroids come from the data via GMM.

ARCHETYPE_DEFINITIONS = [
    {
        "id":          "enthusiast",
        "name":        "The Enthusiast",
        "centroid":    [0.75, 0.45, 0.80, 0.75, 0.20],
        "description": "Expressive and generous. Writes long, warm reviews full of exclamation marks. Rates highly and focuses on the positives. Their negativity is rare but when it appears, it's dramatic.",
        "voice_traits": {
            "length":      "verbose",
            "tone":        "warm and expressive",
            "rating_bias": "generous",
            "focus":       "emotional experience and highlights",
            "negatives":   "rarely mentioned, softened when they are",
        },
    },
    {
        "id":          "critic",
        "name":        "The Critic",
        "centroid":    [0.75, 0.65, 0.35, 0.30, 0.75],
        "description": "Detailed and demanding. Notices flaws others miss. Rates 1-3 stars often. Their reviews are analytical and structured, always weighing pros against cons — but cons usually win.",
        "voice_traits": {
            "length":      "long and structured",
            "tone":        "analytical and direct",
            "rating_bias": "harsh",
            "focus":       "defects, unmet expectations, value for money",
            "negatives":   "front and centre, specific and detailed",
        },
    },
    {
        "id":          "pragmatist",
        "name":        "The Pragmatist",
        "centroid":    [0.40, 0.85, 0.35, 0.55, 0.30],
        "description": "Brief, factual, consistent. Rates exactly what was delivered — no more, no less. Their reviews read like a checklist. They say what worked and what didn't in the fewest words possible.",
        "voice_traits": {
            "length":      "short to medium",
            "tone":        "neutral and factual",
            "rating_bias": "balanced",
            "focus":       "specific features, does it do what it says",
            "negatives":   "stated plainly without emotion",
        },
    },
    {
        "id":          "advocate",
        "name":        "The Advocate",
        "centroid":    [0.55, 0.60, 0.70, 0.85, 0.15],
        "description": "Community-minded and helpful. Writes reviews to help others decide. Generous but honest. Uses 'you' a lot — they're talking directly to the next buyer.",
        "voice_traits": {
            "length":      "medium to long",
            "tone":        "helpful and direct",
            "rating_bias": "generous but fair",
            "focus":       "what the buyer needs to know",
            "negatives":   "mentioned as warnings to the reader, not complaints",
        },
    },
    {
        "id":          "minimalist",
        "name":        "The Minimalist",
        "centroid":    [0.25, 0.40, 0.25, 0.55, 0.35],
        "description": "Short and to the point. One or two sentences max. The rating does most of the work. They don't elaborate unless something really surprised them.",
        "voice_traits": {
            "length":      "very short",
            "tone":        "blunt",
            "rating_bias": "balanced",
            "focus":       "single most important thing",
            "negatives":   "one word or one sentence if mentioned",
        },
    },
    {
        "id":          "storyteller",
        "name":        "The Storyteller",
        "centroid":    [0.85, 0.40, 0.70, 0.60, 0.30],
        "description": "Narrative and contextual. Gives you their whole journey — why they bought it, what happened, how it ended. Reviews read like short essays. High vocabulary, lots of colour.",
        "voice_traits": {
            "length":      "long and narrative",
            "tone":        "personal and vivid",
            "rating_bias": "moderate to generous",
            "focus":       "the full experience from purchase to use",
            "negatives":   "woven into the story, not a list",
        },
    },
    {
        "id":          "skeptic",
        "name":        "The Skeptic",
        "centroid":    [0.65, 0.75, 0.30, 0.25, 0.65],
        "description": "Questioning and cautious. References other reviews or products. Rarely impressed. Uses hedging language but their conclusions are firm. Suspicious of marketing claims.",
        "voice_traits": {
            "length":      "medium",
            "tone":        "cautious and questioning",
            "rating_bias": "harsh to balanced",
            "focus":       "claims vs reality, comparison to alternatives",
            "negatives":   "framed as warnings, references expectations",
        },
    },
    {
        "id":          "loyalist",
        "name":        "The Loyalist",
        "centroid":    [0.45, 0.70, 0.50, 0.80, 0.20],
        "description": "Fair and forgiving of trusted brands. Gives context — this is their third purchase, they've used it for years. Consistent rater. Negatives are noted but not dwelt on.",
        "voice_traits": {
            "length":      "medium",
            "tone":        "measured and trustworthy",
            "rating_bias": "generous for known brands",
            "focus":       "reliability over time, repeat experience",
            "negatives":   "acknowledged briefly, usually forgiven",
        },
    },
]


# ── load profiles ─────────────────────────────────────────────────────────────

def load_profiles() -> pd.DataFrame:
    amazon = pd.read_csv(os.path.join(OUT_DIR, "user_profiles.csv"))
    amazon["source"] = "amazon"

    apps_path = os.path.join(OUT_DIR, "apps_user_profiles.csv")
    if os.path.exists(apps_path):
        apps = pd.read_csv(apps_path)
        apps["source"] = "playstore"
        for col in ["category_books", "category_electronics", "category_movies"]:
            if col not in apps.columns:
                apps[col] = 0.0
        df = pd.concat([amazon, apps], ignore_index=True)
    else:
        print("  WARNING: apps_user_profiles.csv not found, using Amazon only")
        df = amazon

    df = df.dropna(subset=OCEAN_COLS).reset_index(drop=True)
    print(f"  Total users loaded: {len(df):,}")
    return df


# ── fit gmm ───────────────────────────────────────────────────────────────────

def fit_gmm(df: pd.DataFrame):
    """
    Fit a Gaussian Mixture Model on OCEAN vectors.
    Returns gmm, scaler, soft probability matrix, scaled input.
    """
    X = df[OCEAN_COLS].values

    scaler   = StandardScaler()
    scaled_X = scaler.fit_transform(X)

    print(f"  Fitting GMM ({N_ARCHETYPES} components, "
          f"covariance_type=full, n_init=10)...")
    print(f"  Input: {len(df):,} users × 5 OCEAN dimensions")

    gmm = GaussianMixture(
        n_components=N_ARCHETYPES,
        covariance_type="full",
        max_iter=300,
        n_init=10,
        random_state=RANDOM_STATE,
        verbose=0,
    )
    gmm.fit(scaled_X)

    if not gmm.converged_:
        print("  WARNING: GMM did not fully converge. "
              "Consider increasing max_iter.")
    else:
        print(f"  Converged in {gmm.n_iter_} iterations. "
              f"Log-likelihood: {gmm.lower_bound_:.4f}")

    probs = gmm.predict_proba(scaled_X)   # shape: (n_users, 8)
    return gmm, scaler, probs, scaled_X


# ── name gmm components ───────────────────────────────────────────────────────

def name_components(gmm, scaler):
    """
    Match each GMM component to the nearest archetype definition
    using cosine similarity. Greedy one-to-one assignment.
    Returns component_map and GMM means in original OCEAN space.
    """
    means_original = np.clip(
        scaler.inverse_transform(gmm.means_), 0, 1)        # (8, 5)

    target_centroids = np.array(
        [a["centroid"] for a in ARCHETYPE_DEFINITIONS])    # (8, 5)

    sims = cosine_similarity(means_original, target_centroids)  # (8, 8)

    component_map   = {}
    used_archetypes = set()

    for flat_idx in np.argsort(sims.ravel())[::-1]:
        comp_idx = flat_idx // N_ARCHETYPES
        arch_idx = flat_idx  % N_ARCHETYPES
        if comp_idx in component_map or arch_idx in used_archetypes:
            continue
        component_map[comp_idx] = ARCHETYPE_DEFINITIONS[arch_idx]
        used_archetypes.add(arch_idx)
        if len(component_map) == N_ARCHETYPES:
            break

    return component_map, means_original


# ── assign users ──────────────────────────────────────────────────────────────

def assign_users(df: pd.DataFrame,
                 probs: np.ndarray,
                 component_map: dict) -> pd.DataFrame:
    """
    Add soft probability columns + dominant/secondary archetype per user.
    """
    archetype_ids = [component_map[i]["id"] for i in range(N_ARCHETYPES)]

    prob_df = pd.DataFrame(
        probs,
        columns=[f"prob_{aid}" for aid in archetype_ids]
    )
    df = pd.concat([df.reset_index(drop=True), prob_df], axis=1)

    sorted_idx = np.argsort(probs, axis=1)[:, ::-1]
    df["dominant_archetype"]  = [archetype_ids[i] for i in sorted_idx[:, 0]]
    df["secondary_archetype"] = [archetype_ids[i] for i in sorted_idx[:, 1]]
    df["archetype_confidence"] = probs.max(axis=1).round(4)

    # alias for backwards compatibility
    df["archetype_id"]   = df["dominant_archetype"]
    df["archetype_name"] = df["dominant_archetype"].map(
        {a["id"]: a["name"] for a in ARCHETYPE_DEFINITIONS})

    return df


# ── build archetype output ────────────────────────────────────────────────────

def build_archetype_output(df: pd.DataFrame,
                            component_map: dict,
                            means_original: np.ndarray) -> list:
    total_users = len(df)
    output = []

    for comp_idx, archetype in component_map.items():
        aid   = archetype["id"]
        group = df[df["dominant_archetype"] == aid]
        count = len(group)

        empirical_centroid = (
            group[OCEAN_COLS].mean().round(4).tolist()
            if count > 0 else archetype["centroid"]
        )
        gmm_centroid = means_original[comp_idx].round(4).tolist()

        prob_col = f"prob_{aid}"
        avg_prob = float(df[prob_col].mean().round(4)) if prob_col in df else 0.0

        rating_stats = {}
        if count > 0:
            rating_stats = {
                "avg_rating":        round(group["avg_rating"].mean(), 3),
                "generosity_score":  round(group["generosity_score"].mean(), 3),
                "avg_review_length": round(group["avg_review_length"].mean(), 1),
                "avg_token_count":   round(group["avg_token_count"].mean(), 1),
            }

        top_users = (
            group.nlargest(5, "archetype_confidence")["user_id"].tolist()
            if count > 0 else []
        )

        output.append({
            "id":                  aid,
            "name":                archetype["name"],
            "description":         archetype["description"],
            "voice_traits":        archetype["voice_traits"],
            "gmm_centroid":        dict(zip(["O","C","E","A","N"], gmm_centroid)),
            "empirical_centroid":  dict(zip(["O","C","E","A","N"], empirical_centroid)),
            "target_centroid":     dict(zip(["O","C","E","A","N"], archetype["centroid"])),
            "avg_membership_prob": avg_prob,
            "user_count":          count,
            "user_pct":            round(count / total_users * 100, 2),
            "rating_stats":        rating_stats,
            "sample_user_ids":     top_users,
        })

    return output


# ── print distribution ────────────────────────────────────────────────────────

def print_distribution(archetypes_output: list):
    print("\n── Archetype Distribution (GMM — soft assignment) ──────")
    name_lookup = {a["id"]: a["name"] for a in ARCHETYPE_DEFINITIONS}
    for a in sorted(archetypes_output, key=lambda x: -x["user_count"]):
        bar  = "█" * int(a["user_pct"] / 2)
        prob = a["avg_membership_prob"]
        print(f"  {a['name']:<22} {a['user_count']:>6,} users "
              f"({a['user_pct']:>5.1f}%)  avg_prob={prob:.3f}  {bar}")
    print()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Archetype Clustering Pipeline  —  GMM")
    print("=" * 60)

    print("\nLoading user profiles...")
    df = load_profiles()

    print("\nFitting Gaussian Mixture Model...")
    gmm, scaler, probs, scaled_X = fit_gmm(df)

    print("\nNaming GMM components via cosine similarity...")
    component_map, means_original = name_components(gmm, scaler)
    for comp_idx, arch in sorted(component_map.items()):
        ocean_vec = means_original[comp_idx].round(3)
        print(f"  Component {comp_idx} → {arch['name']:<22}  "
              f"O={ocean_vec[0]:.3f} C={ocean_vec[1]:.3f} "
              f"E={ocean_vec[2]:.3f} A={ocean_vec[3]:.3f} N={ocean_vec[4]:.3f}")

    print("\nAssigning users (soft probabilities)...")
    df = assign_users(df, probs, component_map)

    # ── save user archetypes ──
    prob_cols = [f"prob_{component_map[i]['id']}" for i in range(N_ARCHETYPES)]
    save_cols = [
        "user_id", "source",
        "dominant_archetype", "secondary_archetype",
        "archetype_id", "archetype_name", "archetype_confidence",
        *prob_cols,
        *OCEAN_COLS,
        "avg_rating", "generosity_score", "avg_review_length",
    ]
    out_users = os.path.join(OUT_DIR, "user_archetypes.csv")
    df[save_cols].to_csv(out_users, index=False)
    print(f"✓ user_archetypes.csv saved → {len(df):,} users")

    # ── save archetypes.json ──
    archetypes_output = build_archetype_output(df, component_map, means_original)
    out_archetypes = os.path.join(OUT_DIR, "archetypes.json")
    with open(out_archetypes, "w") as f:
        json.dump(archetypes_output, f, indent=2)
    print(f"✓ archetypes.json saved → {len(archetypes_output)} archetypes")

    print_distribution(archetypes_output)

    # ── confidence stats ──
    avg_conf  = df["archetype_confidence"].mean()
    high_conf = (df["archetype_confidence"] > 0.90).sum()
    low_conf  = (df["archetype_confidence"] < 0.60).sum()
    print(f"  Avg dominant probability : {avg_conf:.4f}")
    print(f"  High confidence (>0.90)  : {high_conf:,} users "
          f"({high_conf/len(df)*100:.1f}%)")
    print(f"  Low confidence  (<0.60)  : {low_conf:,} users "
          f"({low_conf/len(df)*100:.1f}%)")

    # ── show 3 most ambiguous users (interesting blends) ──
    print(f"\n  Most ambiguous users (interesting personality blends):")
    sample = df.nsmallest(3, "archetype_confidence")[
        ["user_id", "dominant_archetype", "secondary_archetype",
         "archetype_confidence"]
    ]
    print(sample.to_string(index=False))

    print("\n── Day 2 complete ──────────────────────────────────────")
    print(f"   user_archetypes.csv : {out_users}")
    print(f"   archetypes.json     : {out_archetypes}")
    print("\n  Next → Day 3: OCEAN Engine")


if __name__ == "__main__":
    main()