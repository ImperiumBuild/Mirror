import requests

from recommendations.models import Movie
from django.conf import settings


TMDB_API_KEY = settings.TMDB_API_KEY

BASE_URL = "https://api.themoviedb.org/3"


GENRES = {
    "action": 28,
    "adventure": 12,
    "romance": 10749,
    "thriller": 53,
    "tragedy": 18,
}



def tmdb_discover(**params):
    url = f"{BASE_URL}/discover/movie"

    params["api_key"] = TMDB_API_KEY

    response = requests.get(url, params=params)
    response.raise_for_status()

    return response.json()


NOLLYWOOD_KEYWORD = 284361

def fetch_hollywood_movies(category, genre_id):

    response = requests.get(
        f"{BASE_URL}/discover/movie",
        params={
            "api_key": TMDB_API_KEY,
            "with_genres": genre_id,
            "sort_by": "popularity.desc",
            "vote_count.gte": 300,
            "page": 1,
        }
    )

    data = response.json()

    return data.get("results", [])[:10]

def fetch_nollywood_movies(category, genre_id, target=50):
    movies = []
    page = 1

    while len(movies) < target and page <= 5:
        data = tmdb_discover(
            with_genres=genre_id,
            with_origin_country="NG",
            sort_by="popularity.desc",
            page=page
        )

        results = data.get("results", [])
        movies.extend(results)

        page += 1

    unique = {m["id"]: m for m in movies}

    return list(unique.values())[:target]

def save_movies(movies, category, region):
    for movie in movies:
        try:
            Movie.objects.update_or_create(
                tmdb_id=movie["id"],
                defaults={
                    "title": movie.get("title"),
                    "overview": movie.get("overview", "") or "",
                    "poster_path": movie.get("poster_path"),  # may be None → ensure model allows null
                    "popularity": movie.get("popularity", 0),
                    "vote_average": movie.get("vote_average", 0),
                    "vote_count": movie.get("vote_count", 0),
                    "category": category,
                    "region": region,
                }
            )
        except Exception as e:
            print(f"Skipping {movie.get('title')} → {e}")

def fetch_movies():

    for category, genre_id in GENRES.items():

        hollywood = fetch_hollywood_movies(category, genre_id)

        save_movies(
            hollywood,
            category=category,
            region="hollywood"
        )

        nollywood = fetch_nollywood_movies(category, genre_id)

        save_movies(
            nollywood,
            category=category,
            region="nollywood"
        )

        print(f"{category} completed.")

    print("All movies fetched successfully.")