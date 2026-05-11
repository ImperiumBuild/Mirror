"""
core/llm/prompt_builder.py
--------------------------
Builds the two-call prompt chain from persona + product context.

Updated: Call 1 now includes product context so the LLM knows
what aspects to focus on for this specific product.
"""

from __future__ import annotations


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