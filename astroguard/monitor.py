"""
AstroGuard - execution quotidienne.

C'est le script lance automatiquement dans le cloud (GitHub Actions) :
 1. recupere les approches proches a venir (JPL CAD),
 2. attribue a chaque objet un score de risque avec le reseau maison,
 3. pour les mieux classes, re-propage l'orbite (moteur Kepler maison) pour
    confirmer date/distance d'approche independamment de la NASA,
 4. ecrit un rapport (Markdown + JSON) dans reports/.

Usage :  python -m astroguard.monitor [--days 180] [--top 15]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os

import numpy as np

from . import data, dataset
from .neural import NeuralNet
from .orbit import Orbit, close_approach, jd_now

ROOT = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(ROOT, "data", "model.json")
REPORT_DIR = os.path.join(ROOT, "reports")
LD_PER_AU = 389.17  # distances lunaires par UA


def jd_to_date(jd: float) -> str:
    jd = jd + 0.5
    Z = int(jd)
    F = jd - Z
    A = Z
    if Z >= 2299161:
        a = int((Z - 1867216.25) / 36524.25)
        A = Z + 1 + a - a // 4
    B = A + 1524
    C = int((B - 122.1) / 365.25)
    D = int(365.25 * C)
    E = int((B - D) / 30.6001)
    day = B - D - int(30.6001 * E) + F
    month = E - 1 if E < 14 else E - 13
    year = C - 4716 if month > 2 else C - 4715
    return f"{year:04d}-{month:02d}-{int(day):02d}"


def run(days: int = 180, top: int = 15) -> dict:
    today = dt.date.today()
    end = today + dt.timedelta(days=days)
    print(f"Recuperation des approches {today} -> {end} ...")
    cad = data.fetch_cad(today.isoformat(), end.isoformat(), dist_max="0.05")
    print(f"  {len(cad)} approches proches trouvees.")

    net = NeuralNet.load(MODEL_PATH)

    objs = []
    for r in cad:
        try:
            h = float(r["h"])
            dist = float(r["dist_min"])
            v_inf = float(r["v_inf"])
        except (TypeError, ValueError):
            continue
        feats = np.array([[h, math.log10(max(dist, 1e-6)), v_inf,
                           dataset.diameter_from_h(h)]])
        proba = float(net.predict_proba(feats)[0])
        objs.append({
            "des": r["des"].strip(),
            "fullname": r.get("fullname", r["des"]).strip(),
            "date_nasa": r.get("cd", ""),
            "dist_au": dist,
            "dist_ld": dist * LD_PER_AU,
            "v_inf": v_inf,
            "H": h,
            "diametre_est_m": dataset.diameter_from_h(h) * 1000.0,
            "risque": proba,
        })

    objs.sort(key=lambda o: o["risque"], reverse=True)

    # Verification orbitale maison pour les mieux classes.
    print(f"Verification orbitale (moteur maison) des {top} premiers ...")
    jd0 = jd_now()
    for o in objs[:top]:
        orb = data.fetch_orbit(o["des"])
        if not orb:
            o["check"] = None
            continue
        ob = Orbit(a=orb["a"], e=orb["e"], i=orb["i"], om=orb["om"],
                   w=orb["w"], ma=orb["ma"], epoch=orb["epoch"])
        jd_min, d_min, v_rel = close_approach(ob, jd0, jd0 + days, step_days=0.5)
        o["check"] = {
            "date_calc": jd_to_date(jd_min),
            "dist_calc_au": d_min,
            "dist_calc_ld": d_min * LD_PER_AU,
            "v_rel_km_s": v_rel,
            "moid_au": orb.get("moid"),
        }

    report = {
        "genere_le": dt.datetime.now(dt.timezone.utc).isoformat(),
        "fenetre_jours": days,
        "n_approches": len(objs),
        "objets": objs,
    }
    _write(report, today)
    return report


def _write(report: dict, today: dt.date):
    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(os.path.join(REPORT_DIR, "latest.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    lines = [
        f"# AstroGuard - rapport du {today.isoformat()}",
        "",
        f"*{report['n_approches']} approches a moins de 0,05 UA dans les "
        f"{report['fenetre_jours']} prochains jours.*",
        "",
        "Score de risque = sortie du reseau de neurones maison (0 a 1). "
        "Les objets marques sont a examiner en priorite ; ce n'est PAS un "
        "verdict d'impact, mais un pre-tri.",
        "",
        "| Risque | Objet | Date (NASA) | Distance | Diam. est. | v (km/s) |",
        "|-------:|-------|-------------|---------:|-----------:|---------:|",
    ]
    for o in report["objets"][:25]:
        flag = "**!**" if o["risque"] >= 0.5 else ""
        lines.append(
            f"| {o['risque']:.2f} {flag} | {o['fullname']} | {o['date_nasa']} "
            f"| {o['dist_ld']:.1f} LD | {o['diametre_est_m']:.0f} m "
            f"| {o['v_inf']:.1f} |"
        )

    lines += ["", "## Verification orbitale (moteur Kepler maison)", "",
              "Pour les objets les mieux classes, on recalcule l'approche a "
              "partir des seuls elements orbitaux, independamment de la NASA.", "",
              "| Objet | Date calc. | Dist. calc. | v_rel | MOID |",
              "|-------|-----------|------------:|------:|-----:|"]
    for o in report["objets"]:
        c = o.get("check")
        if not c:
            continue
        moid = f"{c['moid_au']:.4f} UA" if c.get("moid_au") is not None else "-"
        lines.append(
            f"| {o['fullname']} | {c['date_calc']} | {c['dist_calc_ld']:.1f} LD "
            f"| {c['v_rel_km_s']:.1f} km/s | {moid} |"
        )

    lines += ["", "---", "LD = distance lunaire (~384 400 km). Donnees : "
              "JPL/NASA (CAD, Sentry, SBDB). Calculs : AstroGuard."]

    with open(os.path.join(REPORT_DIR, f"rapport-{today.isoformat()}.md"),
              "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    with open(os.path.join(REPORT_DIR, "latest.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Rapport ecrit dans {REPORT_DIR}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=180)
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()
    rep = run(days=args.days, top=args.top)
    print(f"\nTop risques :")
    for o in rep["objets"][:5]:
        print(f"  {o['risque']:.2f}  {o['fullname']:<22} "
              f"{o['dist_ld']:.1f} LD  ~{o['diametre_est_m']:.0f} m")


if __name__ == "__main__":
    main()
