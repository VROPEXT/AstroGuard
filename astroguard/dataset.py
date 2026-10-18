"""
Construction du jeu de donnees d'apprentissage a partir des donnees JPL.

Idee : le catalogue Sentry est produit par des simulations d'impact
couteuses. On apprend a un petit reseau a reperer, a partir de parametres
*bon marche* (magnitude, taille estimee, distance d'approche, vitesse),
quels geocroiseurs ont un profil "a surveiller" (presents dans Sentry)
vs "sans danger". C'est un pre-filtre rapide, pas un remplacant de Sentry.

Features par objet :
  0. H            : magnitude absolue (plus petit = plus gros/brillant)
  1. log10(dist)  : log de la distance d'approche minimale (UA)
  2. v_inf        : vitesse a l'infini (km/s)
  3. diametre_est : diametre estime depuis H (km)

Label : 1 si l'objet figure dans Sentry (risque non nul), sinon 0.
"""

from __future__ import annotations

import math

import numpy as np

from . import data

FEATURE_NAMES = ["H", "log10_dist_AU", "v_inf_km_s", "diametre_est_km"]


def diameter_from_h(h: float, albedo: float = 0.14) -> float:
    """Diametre (km) estime depuis la magnitude absolue H (albedo suppose)."""
    return 1329.0 / math.sqrt(albedo) * 10 ** (-0.2 * h)


def _row_features(h, dist, v_inf):
    return [h, math.log10(max(dist, 1e-6)), v_inf, diameter_from_h(h)]


def build(date_min="2010-01-01", date_max="2035-12-31", dist_max="0.2"):
    """Retourne (X, y, designations) a partir des feeds CAD + Sentry."""
    cad = data.fetch_cad(date_min, date_max, dist_max)
    sentry = data.fetch_sentry()
    sentry_des = {s["des"].strip() for s in sentry}

    # Un objet peut avoir plusieurs approches : on garde la plus proche.
    best: dict[str, dict] = {}
    for r in cad:
        des = r["des"].strip()
        try:
            dist = float(r["dist_min"])
            h = float(r["h"])
            v_inf = float(r["v_inf"])
        except (TypeError, ValueError):
            continue
        if des not in best or dist < best[des]["dist"]:
            best[des] = {"dist": dist, "h": h, "v_inf": v_inf}

    X, y, des_list = [], [], []
    for des, v in best.items():
        X.append(_row_features(v["h"], v["dist"], v["v_inf"]))
        y.append(1 if des in sentry_des else 0)
        des_list.append(des)

    return np.array(X, dtype=float), np.array(y, dtype=int), des_list
