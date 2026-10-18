# 🛰️ AstroGuard

Une IA maison qui **surveille le risque d'impact des astéroïdes géocroiseurs**,
tourne toute seule dans le cloud (PC éteint), et fait de **vrais calculs** —
pas un LLM qui bricole du texte.

Deux cerveaux, tous les deux codés à la main en Python/numpy :

1. **Un moteur de mécanique céleste** (`orbit.py`) : résout l'équation de
   Kepler (Newton-Raphson) et propage l'orbite d'un astéroïde pour retrouver
   quand et à quelle distance il frôle la Terre. Validé contre les données
   NASA : il retrouve l'approche de `2025 UK9` au **31 oct. 2026** à 3 % près.

2. **Un réseau de neurones** (`neural.py`) : perceptron multicouche avec
   rétropropagation et Adam écrits de zéro (aucune lib de ML). Entraîné sur
   **32 000 vrais astéroïdes**, il pré-trie en « à surveiller » vs « sans
   danger » — **ROC-AUC 0,89**, rappel 83 %.

## À quoi ça sert

Les simulations d'impact officielles de la NASA (Sentry) sont lourdes.
AstroGuard apprend, à partir de paramètres **bon marché** (magnitude, taille,
distance et vitesse d'approche), à repérer vite les objets qui méritent une
analyse complète. C'est un **pré-filtre de sécurité** : il préfère lever une
fausse alerte que rater un objet dangereux. Ce n'est **pas** un verdict d'impact.

## Données (publiques, gratuites, sans clé)

- **CAD** — approches proches Terre-objet
- **Sentry** — objets à probabilité d'impact non nulle (vérité terrain)
- **SBDB** — éléments orbitaux complets

Source : API JPL/NASA (`ssd-api.jpl.nasa.gov`).

## Lancer en local

```bash
pip install -r requirements.txt
python -m astroguard.train      # (ré)entraîne le réseau -> data/model.json
python -m astroguard.monitor    # génère le rapport -> reports/latest.md
```

## Tourne tout seul dans le cloud

Le workflow [`.github/workflows/daily.yml`](.github/workflows/daily.yml) lance
le moniteur **chaque jour à 07h00 UTC** sur les serveurs de GitHub (gratuit,
PC éteint) et publie le rapport du jour dans [`reports/`](reports/).

## Structure

```
astroguard/
  orbit.py     moteur Kepler + approches proches
  neural.py    réseau de neurones (numpy, backprop maison)
  dataset.py   construction du jeu d'apprentissage (CAD + Sentry)
  train.py     entraînement + métriques
  data.py      accès aux API JPL/NASA
  monitor.py   run quotidien -> rapport
reports/       rapports générés (Markdown + JSON)
data/model.json  poids du réseau entraîné
```

*Données : NASA/JPL. Calculs : AstroGuard.*
