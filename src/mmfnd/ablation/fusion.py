"""Validation-fitted expert calibration and reliability-weighted log opinion pool.

Probability orientation is always P(real). Labels: 0=fake, 1=real.
A2/A3 use raw, ungated experts to isolate their contribution; see protocol.md.
"""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit

EPS = 1e-7
CONFIGS = {
    "A0": (False, False, False, False),
    "A1": (False, False, False, False),
    "A2": (True, False, False, False),
    "A3": (False, True, False, False),
    "A4": (True, True, False, False),
    "A5": (True, True, True, False),
    "A6": (True, True, True, True),
}


def dtf(real, fake, para_max=10., para_min=10.):
    """Manuscript count formula; equal defaults preserve the stated sign."""
    real, fake = np.asarray(real, float), np.asarray(fake, float)
    if np.any(real < 0) or np.any(fake < 0):
        raise ValueError("Domain counts cannot be negative")
    return (real-fake) * (expit(np.maximum(real, fake)-para_max)
                          - expit(np.minimum(real, fake)-para_min))


def safe_logit(p):
    return logit(np.clip(p, EPS, 1-EPS))


def nll(y, p):
    p = np.clip(p, EPS, 1-EPS)
    return float(-np.mean(y*np.log(p)+(1-y)*np.log1p(-p)))


@dataclass
class Calibrator:
    slope: float = 1.
    offset: float = 0.

    def fit(self, x, y):
        # Positive slope: temperature/offset fit without reversing class semantics.
        if len(x) < 2 or len(np.unique(y)) < 2:
            return self
        def objective(v):
            return nll(y, expit(np.exp(v[0])*x+v[1]))
        opt = minimize(objective, [0., 0.], method="L-BFGS-B",
                       bounds=[(-5., 5.), (-10., 10.)])
        if not opt.success or not np.isfinite(opt.fun):
            raise RuntimeError(f"Calibration failed: {opt.message}")
        self.slope, self.offset = float(np.exp(opt.x[0])), float(opt.x[1])
        return self

    def predict(self, x):
        return expit(self.slope*x+self.offset)


class Fusion:
    def __init__(self, config):
        self.config = config
        self.domain, self.evidence, self.calibrated, self.gated = CONFIGS[config]
        self.calibrators = [Calibrator() for _ in range(3)]
        self.gate = (10., .5, .5, 10.)  # n0, eta, similarity tau, gamma
        self.fitted = False

    def raw(self, frame):
        mm = frame["early_logit" if self.config == "A0" else "hier_logit"].to_numpy(float)
        dom = frame.dtf.to_numpy(float)
        mass = (frame.p_sup+frame.p_ref).to_numpy(float)
        ev = np.divide(frame.p_sup.to_numpy(float), mass,
                       out=np.full(len(frame), .5), where=mass > EPS)
        available = np.column_stack([np.ones(len(frame), bool),
                                      frame.domain_n.to_numpy() > 0,
                                      frame.evidence_available.to_numpy(bool) & (mass > EPS)])
        return np.column_stack([mm, dom, safe_logit(ev)]), available

    def fit(self, validation):
        if set(validation.split) != {"val"}:
            raise ValueError("Fusion.fit accepts validation rows only")
        x, available = self.raw(validation)
        y = validation.label.to_numpy(int)
        if self.calibrated:
            for k, calibrator in enumerate(self.calibrators):
                mask = available[:, k]
                calibrator.fit(x[mask, k], y[mask])
        self.fitted = True
        if self.gated:
            # Base weights stay equal in A4/A5/A6, isolating calibration and gates.
            candidates = [(n, e, t, g) for n in (5., 10., 20.) for e in (.1, .5)
                          for t in (.3, .5, .7) for g in (5., 10.)]
            losses = []
            for gate in candidates:
                self.gate = gate
                losses.append(nll(y, self.predict(validation)))
            self.gate = candidates[int(np.argmin(losses))]
        return self

    def predict(self, frame, disable_domain_gate=False, disable_evidence_gate=False):
        if not self.fitted:
            raise RuntimeError("Fit on validation data before prediction")
        x, available = self.raw(frame)
        p = np.column_stack([c.predict(x[:, k]) for k, c in enumerate(self.calibrators)])
        p[~available] = .5
        weights = np.tile([1., float(self.domain), float(self.evidence)], (len(frame), 1))
        # Missing evidence has zero weight even in ungated variants: exact fallback.
        weights *= available
        if self.gated:
            n0, eta, tau, gamma = self.gate
            if not disable_domain_gate:
                weights[:, 1] *= expit(eta*(frame.domain_n.to_numpy()-n0))
            if not disable_evidence_gate:
                weights[:, 2] *= expit(gamma*(frame.s_max.to_numpy()-tau))*(1-frame.p_nei.to_numpy())
        weights /= weights.sum(axis=1, keepdims=True)
        return expit((weights*safe_logit(p)).sum(axis=1))

    def state(self):
        return {"config": self.config, "gate": self.gate,
                "calibrators": [vars(c) for c in self.calibrators],
                "base_weights": "equal", "threshold": .5}
