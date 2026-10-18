"""
Entrainement du reseau de neurones maison sur les donnees JPL.

Usage :  python -m astroguard.train
Sauvegarde le modele dans data/model.json et affiche les metriques
sur un jeu de test tenu a l'ecart.
"""

from __future__ import annotations

import os

import numpy as np

from . import dataset
from .neural import NeuralNet

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "model.json")


def _metrics(y_true, proba, threshold=0.5):
    pred = (proba >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    acc = (tp + tn) / len(y_true)
    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _auc(y_true, proba):
    """Aire sous la courbe ROC par la statistique de Mann-Whitney."""
    order = np.argsort(proba)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(proba) + 1)
    pos = y_true == 1
    n_pos = pos.sum()
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def main():
    print("1) Construction du jeu de donnees (CAD + Sentry)...")
    X, y, des = dataset.build()
    print(f"   {len(y)} objets, dont {int(y.sum())} 'a surveiller' (Sentry).")

    rng = np.random.default_rng(7)
    idx = rng.permutation(len(y))
    cut = int(0.8 * len(y))
    tr, te = idx[:cut], idx[cut:]

    print("2) Entrainement du reseau de neurones (numpy, backprop maison)...")
    net = NeuralNet([X.shape[1], 16, 8, 1], seed=42)
    net.train(X[tr], y[tr], epochs=400, lr=2e-3, batch=64, verbose=True)

    print("3) Evaluation sur le jeu de test...")
    proba = net.predict_proba(X[te])
    m = _metrics(y[te], proba)
    auc = _auc(y[te], proba)
    print(f"   exactitude={m['acc']:.3f}  precision={m['prec']:.3f}  "
          f"rappel={m['rec']:.3f}  F1={m['f1']:.3f}  ROC-AUC={auc:.3f}")
    print(f"   confusion : VP={m['tp']} FP={m['fp']} FN={m['fn']} VN={m['tn']}")

    net.save(MODEL_PATH)
    print(f"4) Modele sauvegarde : {MODEL_PATH}")


if __name__ == "__main__":
    main()
