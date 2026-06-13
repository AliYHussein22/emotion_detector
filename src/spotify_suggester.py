"""
src/spotify_suggester.py
────────────────────────
Uses the Spotify Web API (Client Credentials flow — no user login needed)
to search for mood-matched playlists.

Docs: https://developer.spotify.com/documentation/web-api
"""

from __future__ import annotations
import requests
import base64

# Emotion → search query mapping
MOOD_QUERIES: dict[str, str] = {
    "happy":    "feel good happy vibes",
    "sad":      "sad emotional healing",
    "angry":    "rage release intense workout",
    "fear":     "calm anxiety relief meditation",
    "surprise": "upbeat energetic pop",
    "disgust":  "cleanse refresh ambient",
    "neutral":  "lo-fi focus study",
}

_token_cache: dict = {}   # simple in-process cache


def _get_access_token(client_id: str, client_secret: str) -> str | None:
    """Fetch a Spotify Client Credentials token (cached for its lifetime)."""
    import time

    cache_key = (client_id, client_secret)
    cached     = _token_cache.get(cache_key)

    if cached and cached["expires_at"] > time.time() + 30:
        return cached["token"]

    credentials = base64.b64encode(
        f"{client_id}:{client_secret}".encode()
    ).decode()

    resp = requests.post(
        "https://accounts.spotify.com/api/token",
        headers={"Authorization": f"Basic {credentials}"},
        data={"grant_type": "client_credentials"},
        timeout=10,
    )

    if resp.status_code != 200:
        return None

    data  = resp.json()
    token = data["access_token"]
    _token_cache[cache_key] = {
        "token":      token,
        "expires_at": time.time() + data.get("expires_in", 3600),
    }
    return token


def get_playlist_suggestions(
    emotion: str,
    client_id: str,
    client_secret: str,
    limit: int = 4,
) -> list[dict]:
    """
    Return up to `limit` Spotify playlists that match the detected emotion.

    Each dict has:
        name        : str
        url         : str  – Spotify open link
        description : str
        image_url   : str | None
    """
    token = _get_access_token(client_id, client_secret)
    if not token:
        return []

    query = MOOD_QUERIES.get(emotion, emotion)

    resp = requests.get(
        "https://api.spotify.com/v1/search",
        headers={"Authorization": f"Bearer {token}"},
        params={
            "q":     query,
            "type":  "playlist",
            "limit": limit,
        },
        timeout=10,
    )

    if resp.status_code != 200:
        return []

    items = resp.json().get("playlists", {}).get("items", [])

    playlists = []
    for item in items:
        if item is None:
            continue
        images = item.get("images") or []
        playlists.append(
            {
                "name":        item.get("name", "Untitled"),
                "url":         item["external_urls"]["spotify"],
                "description": (item.get("description") or "").strip() or query.title(),
                "image_url":   images[0]["url"] if images else None,
            }
        )

    return playlists
