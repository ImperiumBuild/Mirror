import json
import requests
from django.conf import settings
from recommendations.models import Movie

from core.llm import client

TMDB_API_KEY = settings.TMDB_API_KEY

client=client.get_client()
# =========================================================
# 1. FETCH REVIEWS FROM TMDB
# =========================================================
def fetch_reviews(tmdb_id, limit=20):
    url = f"https://api.themoviedb.org/3/movie/{tmdb_id}/reviews"

    try:
        response = requests.get(
            url,
            params={"api_key": TMDB_API_KEY, "language": "en-US", "page": 1},
            timeout=10
        )
        response.raise_for_status()

        results = response.json().get("results", [])

        cleaned = []
        for r in results[:limit]:
            content = (r.get("content") or "").strip()
            if content:
                cleaned.append(content[:400])  # keep it short for LLM

        return cleaned

    except Exception as e:
        print(f"[TMDB ERROR] {tmdb_id}: {e}")
        return []


# =========================================================
# 2. BUILD MOVIE INPUT PAYLOAD
# =========================================================
def build_movie_payload(movie):
    reviews = fetch_reviews(movie.tmdb_id)

    return {
        "id": movie.id,
        "title": movie.title,
        "overview": movie.overview or "",
        "reviews": reviews if reviews else None
    }


# =========================================================
# 3. BATCH GENERATOR
# =========================================================
def build_batches(batch_size=10):
    movies = Movie.objects.all()

    batch = []
    for movie in movies:
        batch.append(build_movie_payload(movie))

        if len(batch) == batch_size:
            yield batch
            batch = []

    if batch:
        yield batch


# =========================================================
# 4. LLM PROMPT (STRUCTURED OUTPUT)
# =========================================================
LLM_PROMPT = """
You are a movie metadata extraction system.

For each movie, extract:

1. top_praises (max 5 short phrases)
2. top_critics (max 5 short phrases)
3. category (action, romance, thriller, tragedy, adventure, mind_bending)
4. emotional_tone (positive, mixed, negative)

Rules:
- Use ONLY provided text
- If reviews exist → prioritize them
- If no reviews → use overview
- Do NOT hallucinate
- Keep outputs short and structured
- Return ONLY valid JSON array

Example format:
[
  {
    "id": 1,
    "top_praises": ["great visuals", "strong acting"],
    "top_critics": ["slow pacing"],
    "category": "thriller",
    "emotional_tone": "mixed"
  }
]

INPUT:
"""


# =========================================================
# 5. CALL LLM (OLLAMA OR ANY API)
# =========================================================
def call_llm(batch):
    try:
        payload = LLM_PROMPT + json.dumps(batch)

        client_response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=payload,
        )

        raw = client_response.text

        return json.loads(raw)

    except Exception as e:
        print(f"[LLM ERROR]: {e}")
        return []


# =========================================================
# 6. UPDATE DATABASE
# =========================================================
def update_movies(llm_output):
    for item in llm_output:
        try:
            movie = Movie.objects.get(id=item["id"])

            movie.top_praises = item.get("top_praises", [])
            movie.top_critics = item.get("top_critics", [])
            movie.emotional_tone = item.get("emotional_tone", "")
            movie.category = item.get("category", movie.category)

            movie.save()

            print(f"[UPDATED] {movie.title}")

        except Exception as e:
            print(f"[SKIP] {item.get('id')}: {e}")


# =========================================================
# 7. MAIN PIPELINE RUNNER
# =========================================================
def run_enrichment(batch_size=10):
    """
    Main entry point:
    - batches movies
    - sends to LLM
    - updates DB
    """

    total = 0

    for batch in build_batches(batch_size=batch_size):

        print(f"\n[PROCESSING BATCH] size={len(batch)}")

        llm_output = call_llm(batch)

        if llm_output:
            update_movies(llm_output)
            total += len(llm_output)

    print(f"\nDONE → Updated {total} movies")

def is_processed(movie):
    """
    A movie is considered processed if it already has enrichment data.
    You can tweak this logic later.
    """
    return bool(movie.top_praises) and bool(movie.emotional_tone)


def build_batches_cont(batch_size=10):
    """
    Only yields movies that are NOT yet processed.
    """
    movies = Movie.objects.all()

    batch = []

    for movie in movies:

        if is_processed(movie):
            continue  # SKIP already done

        batch.append(build_movie_payload(movie))

        if len(batch) == batch_size:
            yield batch
            batch = []

    if batch:
        yield batch


def run_enrichment_cont(batch_size=10):
    """
    CONTINUATION MODE:
    - skips already processed movies
    - resumes from where pipeline stopped
    - safe for quota limits
    """

    total = 0

    for batch in build_batches_cont(batch_size=batch_size):

        print(f"\n[CONTINUING BATCH] size={len(batch)}")

        llm_output = call_llm(batch)

        if not llm_output:
            print("[SKIP] Empty LLM response")
            continue

        update_movies(llm_output)
        total += len(llm_output)

        print(f"[BATCH DONE] processed={len(llm_output)}")

    print(f"\nDONE → Newly processed {total} movies")