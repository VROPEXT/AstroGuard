"""
Mecanique celeste : propagation d'orbites keplriennes et calcul des
approches proches de la Terre.

Tout est code a la main (pas de bibliotheque d'ephemerides) :
 - resolution de l'equation de Kepler par Newton-Raphson
 - passage elements orbitaux -> position heliocentrique (ecliptique J2000)
 - position de la Terre par les formules de Standish (JPL) basse precision
 - recherche du minimum de distance Terre-asteroide par echantillonnage
   puis raffinement (nombre d'or).

Unites : distances en UA, angles en radians (en interne), temps en jours
juliens (JD). La constante de Gauss donne le moyen mouvement directement
en rad/jour quand a est en UA, donc pas besoin de GM_soleil explicite.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Constante gravitationnelle de Gauss (UA^1.5 / jour). n = K / a^1.5
GAUSS_K = 0.01720209895
DEG = math.pi / 180.0
J2000 = 2451545.0  # JD du 1.5 janvier 2000


# --------------------------------------------------------------------------
# Equation de Kepler
# --------------------------------------------------------------------------
def solve_kepler(M: float, e: float, tol: float = 1e-12, itmax: int = 100) -> float:
    """Resout M = E - e*sin(E) pour l'anomalie excentrique E (Newton-Raphson).

    Valable pour les orbites elliptiques (0 <= e < 1), ce qui couvre tous
    les geocroiseurs. On ramene M dans [-pi, pi] et on part d'une estime
    robuste pour converger meme a forte excentricite.
    """
    M = (M + math.pi) % (2 * math.pi) - math.pi
    # Estime initiale : bonne meme pour e proche de 1.
    E = M + e * math.sin(M) / (1.0 - math.sin(M + e) + math.sin(M))
    for _ in range(itmax):
        f = E - e * math.sin(E) - M
        fp = 1.0 - e * math.cos(E)
        dE = f / fp
        E -= dE
        if abs(dE) < tol:
            break
    return E


@dataclass
class Orbit:
    """Jeu d'elements orbitaux keplriens a une epoque donnee.

    a  : demi-grand axe (UA)
    e  : excentricite
    i  : inclinaison (deg)
    om : longitude du noeud ascendant Omega (deg)
    w  : argument du perihelie omega (deg)
    ma : anomalie moyenne a l'epoque (deg)
    epoch : epoque des elements (JD)
    """

    a: float
    e: float
    i: float
    om: float
    w: float
    ma: float
    epoch: float

    def mean_motion(self) -> float:
        """Moyen mouvement n en rad/jour."""
        return GAUSS_K / (self.a ** 1.5)

    def position(self, jd: float) -> np.ndarray:
        """Position heliocentrique (ecliptique J2000) a l'instant jd, en UA."""
        n = self.mean_motion()
        M = self.ma * DEG + n * (jd - self.epoch)
        E = solve_kepler(M, self.e)

        # Coordonnees dans le plan orbital (x vers le perihelie).
        a = self.a
        x_orb = a * (math.cos(E) - self.e)
        y_orb = a * math.sqrt(1.0 - self.e * self.e) * math.sin(E)

        # Rotation plan orbital -> ecliptique : R_z(Om) R_x(i) R_z(w).
        cw, sw = math.cos(self.w * DEG), math.sin(self.w * DEG)
        ci, si = math.cos(self.i * DEG), math.sin(self.i * DEG)
        co, so = math.cos(self.om * DEG), math.sin(self.om * DEG)

        # Application directe de la matrice de rotation composee.
        x = (co * cw - so * sw * ci) * x_orb + (-co * sw - so * cw * ci) * y_orb
        y = (so * cw + co * sw * ci) * x_orb + (-so * sw + co * cw * ci) * y_orb
        z = (sw * si) * x_orb + (cw * si) * y_orb
        return np.array([x, y, z])


# --------------------------------------------------------------------------
# Position de la Terre (Standish, basse precision, ~arcminute)
# --------------------------------------------------------------------------
# Elements du barycentre Terre-Lune et leurs derivees (par siecle julien).
_EARTH = {
    "a": (1.00000261, 0.00000562),
    "e": (0.01671123, -0.00004392),
    "i": (-0.00001531, -0.01294668),
    "L": (100.46457166, 35999.37244981),   # longitude moyenne
    "wbar": (102.93768193, 0.32327364),     # longitude du perihelie
    "om": (0.0, 0.0),                        # noeud ~ 0 pour la Terre
}


def earth_position(jd: float) -> np.ndarray:
    """Position heliocentrique de la Terre (ecliptique J2000), en UA."""
    T = (jd - J2000) / 36525.0  # siecles juliens depuis J2000

    def val(k):
        base, rate = _EARTH[k]
        return base + rate * T

    a = val("a")
    e = val("e")
    i = val("i")
    L = val("L")
    wbar = val("wbar")
    om = val("om")

    w = wbar - om        # argument du perihelie
    M = L - wbar         # anomalie moyenne
    orb = Orbit(a=a, e=e, i=i, om=om, w=w, ma=M, epoch=jd)
    return orb.position(jd)


# --------------------------------------------------------------------------
# Approche proche Terre-asteroide
# --------------------------------------------------------------------------
def _distance(orbit: Orbit, jd: float) -> float:
    return float(np.linalg.norm(orbit.position(jd) - earth_position(jd)))


def close_approach(
    orbit: Orbit, jd_start: float, jd_end: float, step_days: float = 1.0
):
    """Cherche la distance minimale Terre-asteroide sur [jd_start, jd_end].

    1) echantillonnage grossier pas a pas pour trouver un creux,
    2) raffinement local par recherche du nombre d'or.

    Retourne (jd_min, distance_min_UA, vitesse_relative_km_s).
    """
    jds = np.arange(jd_start, jd_end + step_days, step_days)
    dists = np.array([_distance(orbit, jd) for jd in jds])
    k = int(np.argmin(dists))

    lo = jds[max(0, k - 1)]
    hi = jds[min(len(jds) - 1, k + 1)]

    # Recherche du nombre d'or dans [lo, hi].
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    c = hi - gr * (hi - lo)
    d = lo + gr * (hi - lo)
    for _ in range(60):
        if _distance(orbit, c) < _distance(orbit, d):
            hi = d
        else:
            lo = c
        c = hi - gr * (hi - lo)
        d = lo + gr * (hi - lo)
        if abs(hi - lo) < 1e-6:
            break
    jd_min = 0.5 * (lo + hi)
    dist_min = _distance(orbit, jd_min)

    # Vitesse relative par difference finie (UA/jour -> km/s).
    dt = 0.01
    p1 = orbit.position(jd_min - dt) - earth_position(jd_min - dt)
    p2 = orbit.position(jd_min + dt) - earth_position(jd_min + dt)
    v_au_day = np.linalg.norm(p2 - p1) / (2 * dt)
    v_km_s = v_au_day * 149597870.7 / 86400.0
    return jd_min, dist_min, v_km_s


def jd_now() -> float:
    """JD de l'instant present (UTC)."""
    import datetime as _dt

    now = _dt.datetime.now(_dt.timezone.utc)
    y, m = now.year, now.month
    day = (now.day + now.hour / 24.0 + now.minute / 1440.0 + now.second / 86400.0)
    if m <= 2:
        y -= 1
        m += 12
    A = y // 100
    B = 2 - A + A // 4
    return (
        math.floor(365.25 * (y + 4716))
        + math.floor(30.6001 * (m + 1))
        + day + B - 1524.5
    )
