import json
import asyncio
from pathlib import Path

import httpx
import re
from urllib.parse import quote
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse


app = FastAPI(title="Bible Verse API", description="API for retrieving and caching Bible verses.", version="1.0.0")

# Define the path for storing cached verses
DATA_FILE = Path("verses_cache.json")

# In-memory cache structure:
# {
#   "John 3:16": {"data": <verse_data_dict>, "count": <int>},
#   ...
# }

# Use a lock to ensure thread-safe access to the cache and file
cache_lock = asyncio.Lock()

# Initialize the cache dictionary from file if it exists
if DATA_FILE.exists():
    try:
        with DATA_FILE.open("r", encoding="utf-8") as f:
            _cache = json.load(f)
    except json.JSONDecodeError:
        # If file is corrupted or empty, start with an empty cache
        _cache = {}
else:
    _cache = {}


def norm_ref(s: str) -> str:
    """Normaliza una referencia bíblica a una forma canónica.

    - Minúsculas
    - Trim y colapsa espacios
    - Quita espacios alrededor de ':'
    - Inserta espacio entre número y libro (p.ej., '1john' -> '1 john')
    - Inserta espacio entre libro y capítulo si falta (p.ej., 'john3:16' -> 'john 3:16')
    """
    s = s.strip().lower()
    s = re.sub(r"\s*:\s*", ":", s)
    s = re.sub(r"^(\d)\s*([a-z])", r"\1 \2", s)
    s = re.sub(r"([a-z])\s*(\d)", r"\1 \2", s)
    s = " ".join(s.split())
    return s


async def fetch_verse_from_api(reference: str) -> dict:
    """Fetch a verse from the bible-api.com service."""
    # Normaliza referencia y codifica para URL (manteniendo ':')
    normalized = norm_ref(reference)
    url_reference = quote(normalized, safe=":") 
    url = f"https://bible-api.com/{url_reference}"
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url)
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=503, detail=f"Error communicating with bible-api: {exc}")

    if response.status_code != 200:
        # bible-api returns 404 for invalid references
        if response.status_code == 404:
            raise HTTPException(status_code=404, detail="Verse not found. Check your reference (e.g., 'John 3:16').")
        else:
            raise HTTPException(status_code=response.status_code, detail="Unexpected error from bible-api.")

    try:
        verse_data = response.json()
    except json.JSONDecodeError:
        raise HTTPException(status_code=502, detail="Invalid response from bible-api.")
    return verse_data


async def save_cache_to_file() -> None:
    """Persist the in-memory cache to the JSON file."""
    try:
        with DATA_FILE.open("w", encoding="utf-8") as f:
            json.dump(_cache, f)
    except Exception as exc:
        # Log or handle file write error; raise internal server error if necessary
        raise HTTPException(status_code=500, detail=f"Error saving cache: {exc}")


@app.get("/verse")
async def get_verse(reference: str = Query(..., description="Verse reference (e.g., 'John 3:16')")):
    """Retrieve a verse by reference. Caches results and tracks popularity."""
    if not reference:
        raise HTTPException(status_code=400, detail="Reference parameter is required.")

    # Ensure consistent key format
    normalized_ref = norm_ref(reference)

    async with cache_lock:
        # Check if the verse exists in cache
        entry = _cache.get(normalized_ref)
        if entry:
            # Increment request count and return cached data
            entry["count"] += 1
            await save_cache_to_file()
            return JSONResponse(content=entry["data"]["text"])

        # If not cached, fetch from external API
        verse_data = await fetch_verse_from_api(normalized_ref)
        # Store in cache with initial count
        _cache[normalized_ref] = {"data": verse_data, "count": 1}
        await save_cache_to_file()
        return JSONResponse(content=verse_data)

 
@app.get("/top3")
async def get_top_three():
    """Return the top 3 most-requested verses along with their counts and data."""
    async with cache_lock:
        if not _cache:
            return JSONResponse(content="No hay registro de ninguna busqueda")
        # Sort by count in descending order and get top three
        sorted_entries = sorted(_cache.items(), key=lambda item: item[1]["count"], reverse=True)[:3]
        top_list = []
        for reference, entry in sorted_entries:
            top_list.append(
                {
                    "reference": reference,
                    "count": entry["count"],
                    "verse": entry["data"]["text"]
                }
            )
        return top_list


@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request, exc: HTTPException):
    """Customize error responses to ensure consistent JSON format."""
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.get("/")
async def root():
    """Root endpoint provides a simple welcome message and usage overview."""
    return {
        "message": (
            "Welcome to the Bible Verse API. Use /verse?reference=Book Chapter:Verse to get a verse, "
            "and /top3 to get the three most requested verses."
        )
    }


# Note: To run this API locally, use: uvicorn main:app --host 0.0.0.0 --port 8000
# Example curl "http://127.0.0.1:8000/verse?reference=John%203%3A16"  