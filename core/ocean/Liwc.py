"""
liwc.py
-------
Layer 2 of the Train My LLM flow.

When a user picks their preferred review from a pair during training,
this module analyses the chosen review text and extracts a linguistic
fingerprint. This is independent of what the user says about themselves
— it captures revealed preference from the text they chose.

The analysis is inspired by LIWC (Linguistic Inquiry and Word Count)
but implemented as a lightweight proxy using curated word lists.

Usage:
    from core.ocean.liwc import analyse_chosen_reviews, ocean_from_liwc

    # user picked these reviews during Layer 2
    chosen_texts = [
        "Absolutely loved it! Best purchase I've made all year...",
        "The product works as described. Delivery was fast.",
    ]

    liwc_features = analyse_chosen_reviews(chosen_texts)
    ocean_adjustments = ocean_from_liwc(liwc_features)
    # {"O": +0.05, "C": -0.02, "E": +0.12, "A": +0.08, "N": -0.06}
"""

from __future__ import annotations
import re
import string
import math


# ── word lists ────────────────────────────────────────────────────────────────

# Positive affect words
POS_AFFECT = {
    "love", "loved", "amazing", "excellent", "great", "fantastic",
    "wonderful", "perfect", "awesome", "brilliant", "superb", "best",
    "beautiful", "enjoy", "enjoyed", "happy", "pleased", "delight",
    "delighted", "impressed", "outstanding", "incredible", "recommend",
    "worth", "solid", "quality", "gorgeous", "lovely", "joy", "glad",
    "thrilled", "grateful", "thankful", "excited", "fun", "nice",
}

# Negative affect words
NEG_AFFECT = {
    "terrible", "awful", "horrible", "worst", "bad", "poor", "useless",
    "broken", "waste", "disappointed", "disappointing", "defective",
    "cheap", "flimsy", "garbage", "junk", "scam", "fraud", "damaged",
    "missing", "wrong", "avoid", "regret", "annoying", "frustrated",
    "rubbish", "pathetic", "disgusting", "unacceptable", "misleading",
    "failure", "failed", "never", "returned", "refund", "overpriced",
}

# Certainty/absolutist words — high C and low O
CERTAINTY = {
    "definitely", "absolutely", "certainly", "clearly", "obviously",
    "always", "never", "completely", "totally", "exactly", "must",
    "without", "undoubtedly", "guaranteed", "every", "all", "perfect",
    "exactly", "precisely", "no doubt", "unquestionably",
}

# Hedging/tentative words — high O, lower C
HEDGE = {
    "maybe", "perhaps", "possibly", "might", "could", "seems", "appears",
    "somewhat", "kind", "sort", "fairly", "rather", "quite", "almost",
    "guess", "think", "believe", "probably", "generally", "usually",
    "typically", "often", "sometimes", "occasionally", "relatively",
}

# Analytical/cognitive words — high O, high C
ANALYTICAL = {
    "because", "therefore", "however", "although", "despite", "whereas",
    "consider", "analysis", "compare", "contrast", "reason", "evidence",
    "suggest", "indicate", "conclude", "evaluate", "assess", "determine",
    "specific", "detail", "particular", "accurate", "precise",
}

# Social/community words — high E, high A
SOCIAL = {
    "you", "your", "we", "our", "everyone", "people", "others",
    "anyone", "someone", "family", "friend", "customer", "buyer",
    "recommend", "share", "help", "helpful", "useful", "anyone",
}

# Self-reference words — high E (self-focused expressiveness)
SELF_REF = {"i", "me", "my", "mine", "myself", "i've", "i'm", "i'll"}

# Narrative/storytelling words — high O, high E
NARRATIVE = {
    "first", "then", "after", "before", "finally", "initially",
    "eventually", "when", "while", "during", "started", "began",
    "ended", "decided", "realized", "noticed", "found", "discovered",
    "tried", "bought", "ordered", "received", "opened",
}

# Warning/caution words — high N
WARNING = {
    "warning", "careful", "beware", "caution", "avoid", "don't",
    "cannot", "won't", "terrible", "danger", "problem", "issue",
    "concern", "worry", "risk", "unfortunately", "sadly", "disappointing",
}


# ── tokeniser ─────────────────────────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return [t for t in text.split() if t]


def get_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    sentences = re.split(r"[.!?]+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


# ── analyser ──────────────────────────────────────────────────────────────────

def analyse_text(text: str) -> dict:
    """
    Analyse a single review text and return LIWC-style feature counts.
    """
    tokens    = tokenize(text)
    sentences = get_sentences(text)
    n_tokens  = len(tokens)

    if n_tokens == 0:
        return {}

    token_set = set(tokens)

    def ratio(word_set: set) -> float:
        return sum(1 for t in tokens if t in word_set) / n_tokens

    exclamations = text.count("!")
    questions    = text.count("?")

    # type-token ratio (vocabulary richness, capped at 200 tokens)
    sample = tokens[:200]
    ttr    = len(set(sample)) / len(sample) if sample else 0.0

    avg_sentence_len = n_tokens / len(sentences) if sentences else 0.0

    return {
        "n_tokens":          n_tokens,
        "n_sentences":       len(sentences),
        "avg_sentence_len":  avg_sentence_len,
        "ttr":               round(ttr, 4),
        "pos_affect_ratio":  round(ratio(POS_AFFECT), 4),
        "neg_affect_ratio":  round(ratio(NEG_AFFECT), 4),
        "certainty_ratio":   round(ratio(CERTAINTY), 4),
        "hedge_ratio":       round(ratio(HEDGE), 4),
        "analytical_ratio":  round(ratio(ANALYTICAL), 4),
        "social_ratio":      round(ratio(SOCIAL), 4),
        "self_ref_ratio":    round(ratio(SELF_REF), 4),
        "narrative_ratio":   round(ratio(NARRATIVE), 4),
        "warning_ratio":     round(ratio(WARNING), 4),
        "exclamation_rate":  round(exclamations / n_tokens, 4),
        "question_rate":     round(questions / n_tokens, 4),
        "review_length":     len(text),
    }


def analyse_chosen_reviews(texts: list[str]) -> dict:
    """
    Aggregate LIWC features across all reviews the user chose
    during Layer 2 of training. Returns a single averaged feature dict.

    Args:
        texts: list of review text strings the user selected

    Returns:
        averaged LIWC feature dict
    """
    if not texts:
        return {}

    all_features = [analyse_text(t) for t in texts if t.strip()]
    all_features = [f for f in all_features if f]

    if not all_features:
        return {}

    # average all numeric features across chosen reviews
    keys = all_features[0].keys()
    aggregated = {}
    for key in keys:
        values = [f[key] for f in all_features if key in f]
        aggregated[key] = round(sum(values) / len(values), 4)

    return aggregated


# ── liwc → ocean adjustments ──────────────────────────────────────────────────

def ocean_from_liwc(features: dict) -> dict[str, float]:
    """
    Converts LIWC features into OCEAN dimension adjustments.
    These are SIGNED deltas to be added to the base OCEAN scores
    from Layer 1 (scorer.py).

    Research basis:
      O ↑ with: high TTR, long reviews, hedging, analytical, narrative
      C ↑ with: certainty, analytical, low sentence variance
      E ↑ with: exclamations, social words, self-reference, pos affect
      A ↑ with: pos affect, social words, low neg affect, low warning
      N ↑ with: neg affect, warning words, high exclamation (anxiety),
                 low pos affect
    """
    if not features:
        return {"O": 0.0, "C": 0.0, "E": 0.0, "A": 0.0, "N": 0.0}

    def scale(value: float, factor: float = 1.0) -> float:
        """Smooth scaling to keep adjustments small and bounded."""
        return round(math.tanh(value * factor) * 0.15, 4)

    ttr             = features.get("ttr", 0.5)
    avg_sent_len    = features.get("avg_sentence_len", 10)
    review_length   = features.get("review_length", 100)
    pos_affect      = features.get("pos_affect_ratio", 0)
    neg_affect      = features.get("neg_affect_ratio", 0)
    certainty       = features.get("certainty_ratio", 0)
    hedge           = features.get("hedge_ratio", 0)
    analytical      = features.get("analytical_ratio", 0)
    social          = features.get("social_ratio", 0)
    self_ref        = features.get("self_ref_ratio", 0)
    narrative       = features.get("narrative_ratio", 0)
    warning         = features.get("warning_ratio", 0)
    exclamation     = features.get("exclamation_rate", 0)

    # Openness: rich vocabulary, long thoughtful reviews, hedging, analytical
    O = (
        scale(ttr - 0.5, 3)           # above-average vocab richness
        + scale(avg_sent_len - 10, 0.5) # longer sentences
        + scale(hedge, 8)              # hedging = open-minded
        + scale(analytical, 8)         # analytical = intellectually open
        + scale(narrative, 5)          # storytelling = imaginative
        + scale(min(review_length / 500, 1) - 0.3, 2)  # long reviews
    )

    # Conscientiousness: certainty, analytical, structured
    C = (
        scale(certainty, 10)           # definitive language
        + scale(analytical, 8)         # structured thinking
        - scale(hedge, 5)              # hedging = less conscientious
    )

    # Extraversion: exclamations, social words, self-reference, positivity
    E = (
        scale(exclamation, 20)         # !! = expressive
        + scale(social, 8)             # talking to "you", "we"
        + scale(self_ref, 8)           # first person = self-expressive
        + scale(pos_affect, 6)         # positive = energetic
    )

    # Agreeableness: positive, social, low negativity, low warning
    A = (
        scale(pos_affect, 8)           # warm language
        + scale(social, 6)             # community-minded
        - scale(neg_affect, 8)         # negativity = disagreeable
        - scale(warning, 8)            # warnings = low trust
    )

    # Neuroticism: negative affect, warnings, low positivity
    N = (
        scale(neg_affect, 10)          # negative language
        + scale(warning, 10)           # warning others
        - scale(pos_affect, 6)         # positivity buffers N
        + scale(exclamation, 5)        # excited exclamations can signal anxiety
    )

    return {
        "O": round(O, 4),
        "C": round(C, 4),
        "E": round(E, 4),
        "A": round(A, 4),
        "N": round(N, 4),
    }


def tag_reasoning(features: dict) -> list[str]:
    """
    Returns a list of human-readable tags explaining what the chosen
    reviews reveal about the user's writing style.
    Used in the profile summary shown after training.

    e.g. ["tends toward detailed reviews",
          "uses direct, certain language",
          "focuses on the positive"]
    """
    tags = []

    if features.get("review_length", 0) > 300:
        tags.append("tends toward detailed reviews")
    elif features.get("review_length", 0) < 100:
        tags.append("prefers short, punchy reviews")

    if features.get("certainty_ratio", 0) > 0.02:
        tags.append("uses direct, certain language")
    elif features.get("hedge_ratio", 0) > 0.02:
        tags.append("writes with nuance and qualification")

    if features.get("pos_affect_ratio", 0) > features.get("neg_affect_ratio", 0) * 2:
        tags.append("focuses on the positive")
    elif features.get("neg_affect_ratio", 0) > features.get("pos_affect_ratio", 0):
        tags.append("doesn't shy away from calling out problems")

    if features.get("exclamation_rate", 0) > 0.02:
        tags.append("expressive and enthusiastic in tone")

    if features.get("analytical_ratio", 0) > 0.015:
        tags.append("analytical and structured in reasoning")

    if features.get("narrative_ratio", 0) > 0.03:
        tags.append("tells the story of their experience")

    if features.get("social_ratio", 0) > 0.04:
        tags.append("writes for the reader, not just themselves")

    return tags if tags else ["balanced writing style"]


# ── self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("liwc.py — self test")
    print("=" * 50)

    sample_reviews = [
        # Enthusiast-style
        (
            "Absolutely love this product! Best purchase I've made all year. "
            "The quality is amazing and it arrived so quickly. Highly recommend "
            "to everyone — you won't regret it!!!"
        ),
        # Critic-style
        (
            "Terrible quality. The product arrived damaged and customer service "
            "was completely useless. Avoid at all costs. I returned it immediately "
            "and warned my friends not to buy this garbage."
        ),
        # Storyteller-style
        (
            "I was initially skeptical when I ordered this, having had bad experiences "
            "before with similar products. But when it arrived, I was genuinely surprised. "
            "I've been using it for three weeks now and it's become part of my daily routine. "
            "I think what makes it stand out is the attention to detail in the design."
        ),
    ]

    for i, text in enumerate(sample_reviews, 1):
        features = analyse_text(text)
        adjustments = ocean_from_liwc(features)
        tags = tag_reasoning(features)

        print(f"\nSample {i}:")
        print(f"  Text preview: {text[:80]}...")
        print(f"  OCEAN adjustments: {adjustments}")
        print(f"  Style tags: {tags}")

    print("\n── Aggregate test (two reviews chosen) ──")
    agg = analyse_chosen_reviews(sample_reviews[:2])
    adj = ocean_from_liwc(agg)
    print(f"  Aggregated adjustments: {adj}")
    print(f"  Style tags: {tag_reasoning(agg)}")