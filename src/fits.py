"""Power-law fits of loss (or a benchmark) against compute, and local slopes."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import optimize


@dataclass
class PowerLawFit:
    """L(C) = E + A * C**(-alpha).  E is the irreducible loss (0 for the pure power law)."""
    E: float
    A: float
    alpha: float
    n: int
    rmse_log: float
    irreducible: bool

    def predict(self, C):
        C = np.asarray(C, dtype=float)
        return self.E + self.A * C ** (-self.alpha)

    def elasticity(self, C):
        """d log L / d log C at C. For a pure power law this is -alpha; with E>0 it shrinks toward 0."""
        C = np.asarray(C, dtype=float)
        L = self.predict(C)
        return -self.alpha * (self.A * C ** (-self.alpha)) / L

    def label(self):
        base = f"L = {self.E:.2f} + {self.A:.3g}·C^-{self.alpha:.3f}" if self.irreducible else f"L = {self.A:.3g}·C^-{self.alpha:.3f}"
        return f"{base}  (n={self.n})"


def fit_power_law(C, L, irreducible: bool = False) -> PowerLawFit | None:
    C = np.asarray(C, dtype=float); L = np.asarray(L, dtype=float)
    ok = np.isfinite(C) & np.isfinite(L) & (C > 0) & (L > 0)
    C, L = C[ok], L[ok]
    need = 3 if irreducible else 2
    if len(C) < need:
        return None
    logC, logL = np.log(C), np.log(L)
    slope, intercept = np.polyfit(logC, logL, 1)
    A0, alpha0 = float(np.exp(intercept)), float(-slope)
    if not irreducible:
        pred = intercept + slope * logC
        return PowerLawFit(0.0, A0, alpha0, len(C), float(np.sqrt(np.mean((logL - pred) ** 2))), False)

    def resid(p):
        E, logA, alpha = p
        return np.log(np.maximum(E + np.exp(logA) * C ** (-alpha), 1e-9)) - logL

    p0 = [0.5 * L.min(), np.log(A0), max(alpha0, 0.01)]
    r = optimize.least_squares(resid, p0, bounds=([0, -50, 0], [L.min(), 50, 2]))
    E, logA, alpha = r.x
    return PowerLawFit(float(E), float(np.exp(logA)), float(alpha), len(C),
                       float(np.sqrt(np.mean(r.fun ** 2))), True)


def fit_experiments(df: pd.DataFrame, y: str = "loss", irreducible: bool = False) -> PowerLawFit | None:
    """Fit on non-final runs only, so the final run can be treated as an out-of-sample point."""
    ex = df[~df["is_final"]].dropna(subset=["flops", y])
    return fit_power_law(ex["flops"], ex[y], irreducible=irreducible)


def final_residuals(df: pd.DataFrame, fit: PowerLawFit, y: str = "loss") -> pd.DataFrame:
    """Actual minus predicted for each final run, in y units and in log units."""
    fin = df[df["is_final"]].dropna(subset=["flops", y]).copy()
    if fit is None or fin.empty:
        return fin.iloc[0:0]
    fin["predicted"] = fit.predict(fin["flops"])
    fin["residual"] = fin[y] - fin["predicted"]
    fin["log_residual"] = np.log(fin[y]) - np.log(fin["predicted"])
    fin["elasticity_at_final"] = fit.elasticity(fin["flops"])
    return fin[["run_id", "flops", y, "predicted", "residual", "log_residual", "elasticity_at_final"]]


def compute_multiplier(fit_old: PowerLawFit, fit_new: PowerLawFit, C: float) -> float:
    """How much more compute the old curve needs to match the new curve's loss at compute C."""
    target = fit_new.predict(C)
    f = lambda logc: fit_old.predict(np.exp(logc)) - target
    lo, hi = np.log(C), np.log(C) + np.log(1e6)
    if f(lo) <= 0:
        return 1.0
    return float(np.exp(optimize.brentq(f, lo, hi)) / C)


def stated_fit(E: float, A: float, alpha: float, n: int = 0) -> PowerLawFit:
    """Wrap a lab's own published law L = E + A * C**-alpha so it can be drawn like ours."""
    return PowerLawFit(E=E, A=A, alpha=alpha, n=n, rmse_log=0.0, irreducible=E > 0)
