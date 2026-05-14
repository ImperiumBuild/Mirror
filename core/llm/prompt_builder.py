"""
core/llm/prompt_builder.py
--------------------------
Builds the two-call prompt chain from persona + product context.

Updated: Call 1 now includes product context so the LLM knows
what aspects to focus on for this specific product.
"""

from __future__ import annotations
import json
import os
import re
from recommendations.models import Movie


def export_movies_to_json(file_path: str = "movies_dataset.json"):
    """
    Dumps ALL movies + enrichment fields into a JSON file
    for LLM consumption (batch context layer).
    """

    movies = Movie.objects.all()

    dataset = []

    for m in movies:
        dataset.append({
            "id": m.id,
            "tmdb_id": m.tmdb_id,
            "title": m.title,
            "region": m.region,
            "category": m.category,
            "overview": m.overview,
            "vote_average": m.vote_average,
            "popularity": m.popularity,
            "emotional_tone": m.emotional_tone,
            "top_praises": m.top_praises,
            "top_critics": getattr(m, "top_critics", []),  # safe fallback
            "poster_path": m.poster_path,
        })

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"[EXPORT DONE] {len(dataset)} movies written to {file_path}")

    return file_path



def load_movies_from_json(file_path: str = "movies_dataset.json"):
    """
    Loads movie dataset for LLM batch processing.
    This avoids hitting DB during inference.
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found. Run export first.")

    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)
    








def build_reasoning_prompt(
    profile:         dict,
    category:        str,
    product_context: dict | None = None,
) -> str:
    """
    Call 1 — Persona + product context reasoning.
    """
    ocean        = profile.get("ocean", {})
    voice        = profile.get("voice_profile", {})
    archetype    = profile.get("dominant_archetype", "reviewer")
    secondary    = profile.get("secondary_archetype", "")
    calibration  = profile.get("rating_calibration", {})
    style_tags   = profile.get("style_tags", [])
    pairwise_tags = profile.get("pairwise_tags", [])
    summary      = profile.get("profile_summary", "")

    ocean_desc = _ocean_to_traits(ocean)

    voice_desc = (
        f"Review length: {voice.get('length', 'medium')}. "
        f"Tone: {voice.get('tone', 'balanced')}. "
        f"Rating tendency: {voice.get('rating_bias', 'balanced')}. "
        f"Focuses on: {voice.get('focus', 'overall experience')}. "
        f"Handles negatives by: {voice.get('negatives', 'mentioning them fairly')}."
    )

    all_tags = list(set(style_tags + pairwise_tags))
    tags_str = ", ".join(all_tags) if all_tags else "balanced style"

    # product context section
    context_section = ""
    if product_context and product_context.get("name"):
        focus_areas  = product_context.get("common_focus_areas", [])
        avg_rating   = product_context.get("avg_public_rating", "")
        description  = product_context.get("description", "")
        sample_revs  = product_context.get("sample_reviews", [])

        focus_str = ", ".join(focus_areas) if focus_areas else "general experience"

        sample_str = ""
        if sample_revs:
            samples = "\n".join(
                f'  - "{r[:120]}"' for r in sample_revs[:3])
            sample_str = f"\nSample public reviews:\n{samples}"

        context_section = f"""
PRODUCT CONTEXT — {product_context.get('name', '')}:
- Public avg rating: {avg_rating}/5
- What reviewers typically focus on: {focus_str}
- Product description: {description[:300]}{sample_str}

When analysing how this user would review this product, consider:
1. Which of the common focus areas align with their personality?
2. What would THEY specifically notice given their traits?
3. How would their rating tendency interact with the public avg rating?"""

    prompt = f"""You are a behavioural analyst specialising in consumer psychology.

Analyse this user's reviewer personality and describe exactly how they would write a {category} review.

USER PROFILE:
- Archetype: {archetype.title()} (with {secondary} tendencies)
- Personality traits: {ocean_desc}
- Writing style: {voice_desc}
- Style signals: {tags_str}
- Rating calibration: {calibration.get('harsh_or_generous', 'balanced')} (avg {calibration.get('avg_given_rating', 3.0):.1f} stars)
- Profile summary: {summary}
{context_section}

TASK:
Provide a concise behavioural analysis covering:
1. TONE — how would they sound?
2. LENGTH — how long would their review be?
3. FOCUS — what specific aspects of THIS product would they focus on?
4. NEGATIVES — how would they handle flaws?
5. VOCABULARY — what kind of language would they use?
6. RATING STYLE — generous or demanding?

Be specific to THIS user reviewing THIS product. Under 200 words."""

    return prompt


def build_review_prompt(
    reasoning:        str,
    product_name:     str,
    category:         str,
    predicted_rating: int,
    optional_note:    str | None = None,
    sample_reviews:   list[str] | None = None,
    product_context:  dict | None = None,
) -> str:
    """
    Call 2 — Review generation with persona + product context.
    """
    note_section = ""
    if optional_note and optional_note.strip():
        note_section = f"\nThe user wants to mention: {optional_note.strip()}"

    samples_section = ""
    avg_words       = 30

    if sample_reviews:
        word_counts = [len(s.split()) for s in sample_reviews if s.strip()]
        if word_counts:
            avg_words = int(sum(word_counts) / len(word_counts))

        samples = "\n".join(
            f'SAMPLE {i+1}: "{s}"'
            for i, s in enumerate(sample_reviews[:3]))
        samples_section = f"""
ACTUAL WRITING SAMPLES FROM THIS USER:
{samples}

CRITICAL — match these samples exactly:
- Average word count in their samples: {avg_words} words. Stay within 10 words of this.
- If they use abbreviations (fr, cus, sha, btw, imo, ASAP), use them
- If they use exclamation marks, use them the same way
- If they write casually or in Nigerian English, mirror that exactly
- Copy their energy — do NOT produce a polished version of their writing"""

    # product focus from context
    product_focus_section = ""
    if product_context and product_context.get("common_focus_areas"):
        focus_areas = product_context["common_focus_areas"][:3]
        product_focus_section = (
            f"\nFor {product_name}, reviewers typically focus on: "
            f"{', '.join(focus_areas)}. Address what's relevant.")

    prompt = f"""Write a review in this person's voice. Sound EXACTLY like them.
{samples_section}

BEHAVIOURAL ANALYSIS:
{reasoning}
{product_focus_section}

REVIEW TO WRITE:
- Item: {product_name}
- Category: {category}
- Rating: {predicted_rating}/5 stars{note_section}

ABSOLUTE RULES:
1. Hard limit: {avg_words + 10} words maximum
2. Match their vocabulary — casual Nigerian English if that's their style
3. No formal words like "delivers", "utility", "seamlessly", "overall experience"
4. No multiple paragraphs if their samples are 1-2 sentences
5. No star rating mention in the text
6. Write ONLY the review. Nothing else.

Review:"""

    return prompt


def build_recommendation_prompt(
    profile:    dict,
    category:   str,
    candidates: list[dict],
) -> str:
    ocean       = profile.get("ocean", {})
    archetype   = profile.get("dominant_archetype", "reviewer")
    calibration = profile.get("rating_calibration", {})
    voice       = profile.get("voice_profile", {})
    ocean_desc  = _ocean_to_traits(ocean)
    tendency    = calibration.get("harsh_or_generous", "balanced")
    focus       = voice.get("focus", "overall experience")

    candidates_str = "\n".join([
        f"{i+1}. {c.get('title', 'Unknown')} "
        f"(avg rating: {c.get('avg_rating', 'N/A')})"
        for i, c in enumerate(candidates[:10])
    ])

    prompt = f"""You are a personal recommendation assistant. Recommend {category} this person would GENUINELY ENJOY.

USER PERSONALITY:
- Type: {archetype.title()}
- Traits: {ocean_desc}
- Rating tendency: {tendency} rater
- What they care about most: {focus}

HOW TO WRITE THE REASONING:
Pattern: "You [specific trait] — [how this item delivers on that], confirmed by [evidence]."

Good examples:
- "You prefer things that work without fuss — this delivers exactly that, confirmed by its 4.8 rating."
- "You care about reliability over features — this has been consistently praised for stability."

Bad examples (never):
- "Your critical nature will enjoy finding flaws."
- "As a demanding reviewer, this gives you much to analyse."

CANDIDATE {category.upper()} OPTIONS:
{candidates_str}

Recommend things they would ENJOY and rate highly. Harsh rater = needs consistently excellent items.

Return ONLY a JSON array:
[
  {{
    "rank": 1,
    "title": "exact title from candidates",
    "confidence": 0.85,
    "reasoning": "You [trait] — [delivery], confirmed by [evidence]."
  }}
]

Valid JSON only."""

    return prompt


def _ocean_to_traits(ocean: dict[str, float]) -> str:
    traits = []
    o, c, e, a, n = (
        ocean.get("O", 0.5), ocean.get("C", 0.5),
        ocean.get("E", 0.5), ocean.get("A", 0.5),
        ocean.get("N", 0.5),
    )
    traits.append("curious and open" if o >= 0.6 else
                  "practical and direct" if o <= 0.4 else "balanced thinker")
    traits.append("thorough and structured" if c >= 0.6 else
                  "spontaneous" if c <= 0.4 else "moderately organised")
    traits.append("expressive and energetic" if e >= 0.6 else
                  "measured and reserved" if e <= 0.4 else "situationally expressive")
    traits.append("generous and forgiving" if a >= 0.6 else
                  "demanding and critical" if a <= 0.4 else "fair and balanced")
    traits.append("emotionally reactive to problems" if n >= 0.6 else
                  "calm and even-keeled" if n <= 0.4 else "notices issues without dwelling")
    return ", ".join(traits)

import json

def build_movie_recommendation_prompt(
    profile: dict,
    movies: list[dict],
    n: int = 5,
) -> str:
    """
    Builds a context-rich prompt for movie recommendations using enriched DB metadata.
    """

    ocean = profile.get("ocean", {})
    archetype = profile.get("dominant_archetype", "reviewer")
    calibration = profile.get("rating_calibration", {})
    voice = profile.get("voice_profile", {})

    ocean_desc = _ocean_to_traits(ocean)
    tendency = calibration.get("harsh_or_generous", "balanced")
    focus = voice.get("focus", "overall experience")

    # Build compact movie context (VERY IMPORTANT for token control)
    movie_context = []

    for m in movies:
        movie_context.append({
            "title": m.get("title"),
            "region": m.get("region"),
            "category": m.get("category"),
            "overview": m.get("overview", "")[:300],
            "emotional_tone": m.get("emotional_tone"),
            "vote_average": m.get("vote_average"),
            "popularity": m.get("popularity"),
            "top_praises": m.get("top_praises", [])[:5],
            "top_critics": m.get("top_critics", [])[:5],
        })

    prompt = f"""
You are a world-class movie recommendation engine (Netflix-level personalization).

USER PROFILE:
- Type: {archetype.title()}
- Traits: {ocean_desc}
- Rating tendency: {tendency}
- What they care about most: {focus}

MOVIE DATABASE CONTEXT:
You are ONLY allowed to recommend movies from the dataset below.
Each movie includes real user sentiment signals (praises + critics).

MOVIES:
{json.dumps(movie_context, indent=2)}

TASK:
Recommend {n} movies that best match the user's personality and taste.

RULES:
- ONLY use movies from the provided dataset
- Use top_praises and top_critics as ground truth sentiment signals
- Use emotional_tone + overview to understand vibe
- Avoid recommending low-rated or heavily criticized movies unless user profile strongly matches them
- Prioritize alignment over popularity
- Be precise and personalized (this is not generic recommendation)
- Use mostly nollywood movies and few hollywood if they match the profile better

OUTPUT FORMAT (STRICT JSON ONLY):
[
  {{
    "rank": 1,
    "title": "movie title",
    "confidence": 0.92,
    "reasoning": "You [trait] — this matches because [praise/critic/tone-based explanation]."
  }}
]

Return ONLY valid JSON. No extra text.
""".strip()

    return prompt

def build_freeform_recommendation_prompt(
    profile:      dict,
    category:     str,
    sub_category: str | None = None,
    n:            int = 5,
) -> str:
    ocean       = profile.get("ocean", {})
    archetype   = profile.get("dominant_archetype", "reviewer")
    calibration = profile.get("rating_calibration", {})
    voice       = profile.get("voice_profile", {})
    ocean_desc  = _ocean_to_traits(ocean)
    tendency    = calibration.get("harsh_or_generous", "balanced")
    focus       = voice.get("focus", "overall experience")

    if category.lower() == "movies":
        export_movies_to_json()  # run this once to create the JSON dataset for LLM context
        movies_batch = load_movies_from_json()
        prompt = build_movie_recommendation_prompt(profile, movies_batch, n=5)
        return prompt
    cat_label = f"{category} — {sub_category}" if sub_category else category

    prompt = f"""You are a personal recommendation assistant with deep knowledge of popular {category}.

USER PERSONALITY:
- Type: {archetype.title()}
- Traits: {ocean_desc}
- Rating tendency: {tendency} rater
- What they care about most: {focus}

TASK:
Recommend {n} real, well-known {cat_label} that this specific person would genuinely enjoy.

RULES:
- Only recommend items you are 100% certain exist
- For apps: only recommend apps available on Google Play Store
- For products: only recommend products available on Jumia Nigeria
- Match items to the user's personality — a harsh rater needs consistently excellent items
- Write reasoning as: "You [trait] — [how this item delivers on that]"
- Do NOT recommend obscure items nobody has heard of
- Prioritise popular, well-reviewed items that match their personality

Return ONLY a JSON array:
[
  {{
    "rank": 1,
    "title": "exact app/product name",
    "confidence": 0.90,
    "reasoning": "You [trait] — [how this delivers on that]."
  }}
]

Valid JSON only. Nothing else."""

    return prompt

def _parse_recommendations(response: str) -> list[dict]:
    """
    Parses the JSON response from the recommendation prompt.
    Falls back gracefully if the LLM returns malformed JSON.
    """
    # strip markdown code fences if present
    cleaned = re.sub(r"```(?:json)?", "", response).strip()
    cleaned = cleaned.rstrip("`").strip()

    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            return data
        # sometimes wrapped in a key
        if isinstance(data, dict):
            for key in ("recommendations", "results", "items"):
                if key in data and isinstance(data[key], list):
                    return data[key]
    except json.JSONDecodeError:
        pass

    # fallback — extract anything that looks like a ranked item
    fallback = []
    lines    = response.split("\n")
    for line in lines:
        line = line.strip()
        if line and line[0].isdigit() and "." in line:
            title = line.split(".", 1)[-1].strip()
            if title:
                fallback.append({
                    "rank":      len(fallback) + 1,
                    "title":     title,
                    "confidence": 0.5,
                    "reasoning": "Recommended based on your profile.",
                })
    return fallback[:5]

# Add this function to core/llm/prompt_builder.py

def build_product_recommendation_prompt(
    profile:      dict,
    n:            int = 5,
) -> str:
    """
    Asks LLM to generate Nigerian-relevant product names
    based on user persona. Results are then searched on Jumia.
    """
    ocean       = profile.get("ocean", {})
    archetype   = profile.get("dominant_archetype", "reviewer")
    calibration = profile.get("rating_calibration", {})
    voice       = profile.get("voice_profile", {})
    ocean_desc  = _ocean_to_traits(ocean)
    tendency    = calibration.get("harsh_or_generous", "balanced")
    focus       = voice.get("focus", "overall experience")
    style_tags  = profile.get("style_tags", [])

    prompt = f"""You are a personal shopping assistant for Nigerian consumers.

USER PERSONALITY:
- Reviewer type: {archetype.title()}
- Traits: {ocean_desc}
- Rating tendency: {tendency} rater
- What they care about: {focus}
- Style signals: {", ".join(style_tags[:3]) if style_tags else "balanced"}

TASK:
Recommend {n} specific products for Jumia Nigeria that this person would genuinely want to buy.
For each, provide a SHORT search term (2-4 words) and a personalized reasoning.

RULES:
- Think about what products match their personality and Nigerian lifestyle
- A harsh rater needs reliable, well-known brands — not cheap generics
- A generous rater appreciates value and novelty
- Focus on everyday Nigerian needs: tech accessories, home items, personal care, electronics
- Search terms MUST be SHORT (2-4 words) to work well on Jumia search
- Write reasoning as: "You [trait] — [how this item delivers on that]"
- Do NOT recommend luxury items over ₦200,000 unless personality strongly suggests it
- Do NOT recommend food, services, or digital products

Return ONLY a JSON array of objects:
[
  {{
    "term": "oraimo wireless earbuds",
    "reasoning": "You value reliability and long-lasting quality — Oraimo is Nigeria's most trusted brand for durable tech accessories."
  }}
]

Valid JSON array only. Nothing else."""

    return prompt