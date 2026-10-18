"""
Reseau de neurones code entierement a la main en numpy : perceptron
multicouche (MLP) avec retropropagation.

Aucune bibliotheque de ML : toutes les derivees (backprop), l'optimiseur
(Adam) et la normalisation sont ecrits ici. Objectif : classifier un
geocroiseur en "a surveiller" vs "sans danger" a partir de parametres
bon marche (elements orbitaux, taille, MOID, vitesse), pour pre-filtrer
les objets qui meritent une analyse Sentry complete.
"""

from __future__ import annotations

import json

import numpy as np


def _relu(x):
    return np.maximum(0.0, x)


def _relu_grad(x):
    return (x > 0.0).astype(x.dtype)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


class NeuralNet:
    """MLP avec couches cachees ReLU et sortie sigmoide (classif binaire).

    Entrainement par descente de gradient mini-batch + Adam. La perte est
    l'entropie croisee binaire. Les entrees sont standardisees (moyenne 0,
    ecart-type 1) avec des statistiques apprises sur le jeu d'entrainement.
    """

    def __init__(self, layer_sizes: list[int], seed: int = 42):
        rng = np.random.default_rng(seed)
        self.layer_sizes = layer_sizes
        self.W, self.b = [], []
        for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            # Initialisation He pour les couches ReLU.
            self.W.append(rng.normal(0, np.sqrt(2.0 / n_in), (n_in, n_out)))
            self.b.append(np.zeros(n_out))
        self.mu = None
        self.sigma = None
        # Moments d'Adam.
        self._mW = [np.zeros_like(w) for w in self.W]
        self._vW = [np.zeros_like(w) for w in self.W]
        self._mb = [np.zeros_like(b) for b in self.b]
        self._vb = [np.zeros_like(b) for b in self.b]
        self._t = 0

    # -- normalisation -----------------------------------------------------
    def fit_scaler(self, X):
        self.mu = X.mean(axis=0)
        self.sigma = X.std(axis=0) + 1e-8

    def _scale(self, X):
        return (X - self.mu) / self.sigma

    # -- propagation avant -------------------------------------------------
    def _forward(self, Xs):
        acts = [Xs]
        pre = []
        a = Xs
        for li, (W, b) in enumerate(zip(self.W, self.b)):
            z = a @ W + b
            pre.append(z)
            a = _sigmoid(z) if li == len(self.W) - 1 else _relu(z)
            acts.append(a)
        return acts, pre

    def predict_proba(self, X):
        acts, _ = self._forward(self._scale(X))
        return acts[-1].ravel()

    def predict(self, X, threshold: float = 0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    # -- entrainement ------------------------------------------------------
    def train(self, X, y, *, epochs=300, lr=1e-3, batch=64,
              class_weight=None, seed=0, verbose=True):
        rng = np.random.default_rng(seed)
        y = y.reshape(-1, 1).astype(float)
        if self.mu is None:
            self.fit_scaler(X)
        Xs = self._scale(X)
        n = Xs.shape[0]

        # Ponderation des classes (jeu tres desequilibre : peu d'objets risque).
        if class_weight is None:
            pos = max(y.sum(), 1)
            neg = max(n - pos, 1)
            class_weight = {1: n / (2 * pos), 0: n / (2 * neg)}
        w_vec = np.where(y == 1, class_weight[1], class_weight[0])

        for ep in range(epochs):
            idx = rng.permutation(n)
            for s in range(0, n, batch):
                bi = idx[s:s + batch]
                self._step(Xs[bi], y[bi], w_vec[bi], lr)
            if verbose and (ep % 50 == 0 or ep == epochs - 1):
                p = self.predict_proba(X).reshape(-1, 1)
                loss = _bce(y, p, w_vec)
                print(f"  epoque {ep:4d}  perte={loss:.4f}")

    def _step(self, Xs, y, w, lr):
        acts, pre = self._forward(Xs)
        m = Xs.shape[0]
        # Gradient de la BCE + sigmoide : (p - y), pondere.
        delta = (acts[-1] - y) * w / m
        gW, gb = [None] * len(self.W), [None] * len(self.b)
        for li in reversed(range(len(self.W))):
            gW[li] = acts[li].T @ delta
            gb[li] = delta.sum(axis=0)
            if li > 0:
                delta = (delta @ self.W[li].T) * _relu_grad(pre[li - 1])
        self._adam(gW, gb, lr)

    def _adam(self, gW, gb, lr, b1=0.9, b2=0.999, eps=1e-8):
        self._t += 1
        for li in range(len(self.W)):
            self._mW[li] = b1 * self._mW[li] + (1 - b1) * gW[li]
            self._vW[li] = b2 * self._vW[li] + (1 - b2) * gW[li] ** 2
            mhat = self._mW[li] / (1 - b1 ** self._t)
            vhat = self._vW[li] / (1 - b2 ** self._t)
            self.W[li] -= lr * mhat / (np.sqrt(vhat) + eps)

            self._mb[li] = b1 * self._mb[li] + (1 - b1) * gb[li]
            self._vb[li] = b2 * self._vb[li] + (1 - b2) * gb[li] ** 2
            mhatb = self._mb[li] / (1 - b1 ** self._t)
            vhatb = self._vb[li] / (1 - b2 ** self._t)
            self.b[li] -= lr * mhatb / (np.sqrt(vhatb) + eps)

    # -- persistance -------------------------------------------------------
    def save(self, path: str):
        obj = {
            "layer_sizes": self.layer_sizes,
            "W": [w.tolist() for w in self.W],
            "b": [b.tolist() for b in self.b],
            "mu": self.mu.tolist(),
            "sigma": self.sigma.tolist(),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f)

    @classmethod
    def load(cls, path: str) -> "NeuralNet":
        with open(path, encoding="utf-8") as f:
            obj = json.load(f)
        net = cls(obj["layer_sizes"])
        net.W = [np.array(w) for w in obj["W"]]
        net.b = [np.array(b) for b in obj["b"]]
        net.mu = np.array(obj["mu"])
        net.sigma = np.array(obj["sigma"])
        return net


def _bce(y, p, w):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(np.mean(w * -(y * np.log(p) + (1 - y) * np.log(1 - p))))
