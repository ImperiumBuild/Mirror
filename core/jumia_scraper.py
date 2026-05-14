"""
core/jumia_scraper.py
---------------------
Scrapes Jumia Nigeria for product search results.
Returns product listings with title, price, image, link and rating.

Used by the recommendation engine to provide Nigerian-friendly
product recommendations with direct purchase links.

Usage:
    from core.jumia_scraper import search_jumia

    products = search_jumia("wireless earbuds", n=5)
    # [
    #   {
    #     "title":     "Oraimo FreePods 3 TWS...",
    #     "price":     "₦8,999",
    #     "image_url": "https://...",
    #     "url":       "https://www.jumia.com.ng/...",
    #     "rating":    4.2,
    #     "reviews":   128,
    #   },
    #   ...
    # ]
"""

from __future__ import annotations

import time
import random
import requests
from bs4 import BeautifulSoup

BASE_URL  = "https://www.jumia.com.ng"
SEARCH_URL = "https://www.jumia.com.ng/catalog/?q={query}&page={page}"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-GB,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


def search_jumia(
    query:   str,
    n:       int = 10,
    page:    int = 1,
    timeout: int = 10,
) -> list[dict]:
    """
    Searches Jumia Nigeria and returns product listings.

    Args:
        query:   Search term e.g. "wireless earbuds", "iPhone case"
        n:       Max number of results to return
        page:    Page number (default 1)
        timeout: Request timeout in seconds

    Returns:
        List of product dicts with title, price, image_url, url, rating, reviews
        Returns empty list if scraping fails.
    """
    url = SEARCH_URL.format(
        query=query.replace(" ", "+"),
        page=page,
    )

    try:
        response = requests.get(url, headers=HEADERS, timeout=timeout)
        if response.status_code != 200:
            print(f"  [Jumia] HTTP {response.status_code} for '{query}'")
            return []
    except requests.RequestException as e:
        print(f"  [Jumia] Request failed: {e}")
        return []

    soup     = BeautifulSoup(response.text, "html.parser")
    products = []

    # Jumia product cards are in article tags with class "prd"
    cards = soup.select("article.prd")

    if not cards:
        # fallback selector
        cards = soup.select("article[class*='prd']")

    for card in cards[:n]:
        try:
            product = _parse_card(card)
            if product:
                products.append(product)
        except Exception:
            continue

        if len(products) >= n:
            break

    print(f"  [Jumia] '{query}' → {len(products)} products found")
    return products


def _parse_card(card) -> dict | None:
    """Parse a single Jumia product card."""

    # title
    title_el = card.select_one("h3.name") or card.select_one(".name")
    if not title_el:
        return None
    title = title_el.get_text(strip=True)
    if not title:
        return None

    # price
    price_el = card.select_one("div.prc") or card.select_one(".prc")
    price    = price_el.get_text(strip=True) if price_el else "Price unavailable"

    # old price / discount
    old_price_el = card.select_one("div.old") or card.select_one(".old")
    old_price    = old_price_el.get_text(strip=True) if old_price_el else None

    # discount
    discount_el = card.select_one("div.bdg._dsct") or card.select_one("._dsct")
    discount    = discount_el.get_text(strip=True) if discount_el else None

    # image
    img_el    = card.select_one("img.img") or card.select_one("img[data-src]") or card.select_one("img")
    image_url = ""
    if img_el:
        image_url = (
            img_el.get("data-src")
            or img_el.get("src")
            or ""
        )

    # product URL
    link_el  = card.select_one("a.core") or card.select_one("a[href]")
    url      = ""
    if link_el:
        href = link_el.get("href", "")
        url  = BASE_URL + href if href.startswith("/") else href

    # rating
    rating_el = card.select_one("div.stars._s") or card.select_one(".stars")
    rating    = 0.0
    if rating_el:
        # rating is usually in the style width or text
        style = rating_el.get("style", "")
        if "width" in style:
            try:
                pct    = float(style.split("width:")[1].replace("%", "").strip())
                rating = round(pct / 20, 1)  # 100% = 5 stars
            except Exception:
                pass
        else:
            text = rating_el.get_text(strip=True)
            try:
                rating = float(text.split("out")[0].strip())
            except Exception:
                pass

    # review count
    reviews_el = card.select_one("div.rev") or card.select_one(".rev")
    reviews    = 0
    if reviews_el:
        text = reviews_el.get_text(strip=True).replace("(", "").replace(")", "")
        try:
            reviews = int(text)
        except Exception:
            pass

    return {
        "title":     title,
        "price":     price,
        "old_price": old_price,
        "discount":  discount,
        "image_url": image_url,
        "url":       url,
        "rating":    rating,
        "reviews":   reviews,
        "source":    "jumia",
    }


def get_product_details(product_url: str, timeout: int = 10) -> dict:
    """
    Fetches detailed info for a single Jumia product page.
    Used to get full description for LLM context.

    Args:
        product_url: Full Jumia product URL
        timeout:     Request timeout

    Returns:
        Dict with description, full specs, seller info
    """
    try:
        response = requests.get(
            product_url, headers=HEADERS, timeout=timeout)
        if response.status_code != 200:
            return {}
    except requests.RequestException:
        return {}

    soup = BeautifulSoup(response.text, "html.parser")

    # description
    desc_el = (
        soup.select_one("div.markup.-mhm.-pvl.-oxa.-sc") or
        soup.select_one("[class*='markup']") or
        soup.select_one(".pdp-desc")
    )
    description = desc_el.get_text(strip=True)[:500] if desc_el else ""

    # highlights/key features
    highlights = []
    for li in soup.select("ul.-blk li")[:5]:
        text = li.get_text(strip=True)
        if text:
            highlights.append(text)

    return {
        "description": description,
        "highlights":  highlights,
    }


# ── self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("jumia_scraper.py — self test")
    print("=" * 50)

    queries = ["wireless earbuds", "power bank", "phone case"]

    for query in queries:
        print(f"\n── {query} ──────────────────────────────────")
        products = search_jumia(query, n=3)
        for p in products:
            print(f"  {p['title'][:50]}")
            print(f"  Price: {p['price']}  Rating: {p['rating']}  Reviews: {p['reviews']}")
            print(f"  URL:   {p['url'][:60]}...")
            print(f"  Image: {p['image_url'][:60]}...")
            print()
        time.sleep(random.uniform(1, 2))