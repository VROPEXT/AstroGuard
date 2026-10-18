"""
Recuperation des donnees publiques JPL (gratuites, sans cle) :

 - CAD    : approches proches Terre-objet sur une periode
 - Sentry : objets a risque d'impact non nul (notre "verite terrain" ML)
 - SBDB   : elements orbitaux complets d'un objet

Tout est mis en cache dans data/ pour ne pas marteler l'API et pour
pouvoir rejouer hors-ligne.
"""

from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request

API = "https://ssd-api.jpl.nasa.gov"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
USER_AGENT = "AstroGuard/1.0 (moniteur de risque d'impact, usage perso)"


def _get(url: str, retries: int = 3) -> dict:
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=40) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"echec API apres {retries} tentatives : {last}")


def _cache_path(name: str) -> str:
    os.makedirs(DATA_DIR, exist_ok=True)
    return os.path.join(DATA_DIR, name)


def fetch_cad(date_min: str, date_max: str, dist_max: str = "0.05") -> list[dict]:
    """Approches proches entre deux dates (AAAA-MM-JJ), dist_max en UA."""
    q = urllib.parse.urlencode(
        {"date-min": date_min, "date-max": date_max, "dist-max": dist_max,
         "sort": "dist", "fullname": "true"}
    )
    d = _get(f"{API}/cad.api?{q}")
    fields = d.get("fields", [])
    rows = [dict(zip(fields, row)) for row in d.get("data", [])]
    with open(_cache_path("cad.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    return rows


def fetch_sentry() -> list[dict]:
    """Table Sentry complete : objets a probabilite d'impact non nulle."""
    d = _get(f"{API}/sentry.api")
    rows = d.get("data", [])
    with open(_cache_path("sentry.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    return rows


def fetch_orbit(des: str) -> dict | None:
    """Elements orbitaux + epoque d'un objet via SBDB. None si introuvable."""
    q = urllib.parse.urlencode({"sstr": des, "full-prec": "1"})
    try:
        d = _get(f"{API}/sbdb.api?{q}")
    except RuntimeError:
        return None
    orbit = d.get("orbit")
    if not orbit:
        return None
    els = {e["name"]: e["value"] for e in orbit.get("elements", [])}
    phys = {p["name"]: p["value"] for p in d.get("phys_par", [])}
    try:
        return {
            "des": des,
            "a": float(els["a"]),
            "e": float(els["e"]),
            "i": float(els["i"]),
            "om": float(els["om"]),
            "w": float(els["w"]),
            "ma": float(els["ma"]),
            "epoch": float(orbit["epoch"]),
            "H": float(phys.get("H")) if phys.get("H") else None,
            "diameter": float(phys.get("diameter")) if phys.get("diameter") else None,
            "moid": float(orbit["moid"]) if orbit.get("moid") else None,
        }
    except (KeyError, TypeError, ValueError):
        return None


def load_cached(name: str):
    p = _cache_path(name)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)
