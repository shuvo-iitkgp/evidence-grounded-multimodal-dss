"""Classification, calibration, review and paired inference metrics."""
import numpy as np
from scipy.stats import binomtest, rankdata
from sklearn.metrics import (accuracy_score, average_precision_score,
    balanced_accuracy_score, brier_score_loss, f1_score, precision_score,
    recall_score, roc_auc_score)
from .fusion import nll

LAMBDAS = (.1, .5, 1., 10., 50.)
PAIRS = [("A1", "A0"), ("A2", "A1"), ("A3", "A1"), ("A4", "A2"),
         ("A4", "A3"), ("A5", "A4"), ("A6", "A5")]


def bootstrap_metrics(y, p):
    """Vectorized metrics used inside resampling, including tied prediction scores."""
    predicted = p >= .5
    positive = y == 0
    tp = np.sum(positive & ~predicted)
    tn = np.sum(~positive & predicted)
    errors = np.sum(predicted != y)
    f0 = 2*tp/(2*tp+errors) if 2*tp+errors else 0.
    f1 = 2*tn/(2*tn+errors) if 2*tn+errors else 0.
    npos, nneg = positive.sum(), (~positive).sum()
    score = 1-p
    auc, ap = np.nan,np.nan
    if npos and nneg:
        auc = (rankdata(score)[positive].sum()-npos*(npos+1)/2)/(npos*nneg)
        order = np.argsort(-score,kind="stable")
        ends = np.r_[np.flatnonzero(np.diff(score[order])),len(y)-1]
        true_pos = np.cumsum(positive[order])[ends]
        ap = np.sum(np.diff(np.r_[0,true_pos])/npos * true_pos/(ends+1))
    confidence = np.maximum(p,1-p)
    bins = np.minimum((confidence*15).astype(int),14)
    counts = np.bincount(bins,minlength=15)
    conf = np.bincount(bins,weights=confidence,minlength=15)
    correct = np.bincount(bins,weights=(predicted == y).astype(float),minlength=15)
    ece = np.abs(conf-correct)[counts > 0].sum()/len(y)
    return {"macro_f1":(f0+f1)/2,"roc_auc":auc,"pr_auc":ap,"ece":ece,
            "brier":np.mean((p-y)**2),"nll":nll(y,p)}


def reliability(y, p, bins=15):
    """Equal-width confidence ECE (not positive-class probability ECE)."""
    confidence = np.maximum(p, 1-p)
    correct = (p >= .5) == y
    idx = np.minimum((confidence*bins).astype(int), bins-1)
    rows = []
    for b in range(bins):
        mask = idx == b
        if mask.any():
            rows.append({"bin": b, "n": int(mask.sum()),
                         "confidence": float(confidence[mask].mean()),
                         "accuracy": float(correct[mask].mean())})
    return rows


def ranking(p):
    cmc = (1-np.clip(p, 1e-7, 1-1e-7))/np.clip(p, 1e-7, 1-1e-7)
    cmpu = 1/((cmc-1)**2+1e-7)
    return np.argsort(-cmpu, kind="stable")


def review_errors(y, p):
    errors = ((p >= .5) != y).astype(int)
    return np.r_[errors.sum(), errors.sum()-np.cumsum(errors[ranking(p)])]


def choose_review_fraction(y, p, penalty):
    costs = np.arange(len(y)+1)+penalty*review_errors(y, p)
    return float(np.argmin(costs)/len(y))


def metrics(y, p):
    y, p = np.asarray(y, int), np.asarray(p, float)
    if not len(y):
        return {}
    pred = (p >= .5).astype(int)
    errors = pred != y
    # Operational positive class is fake, despite stored labels being 0=fake.
    fake = y == 0
    fn = fake & (pred == 1)
    fp = (~fake) & (pred == 0)
    two_classes = len(np.unique(y)) == 2
    rel = reliability(y, p)
    result = {
        "n": len(y), "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, labels=[0, 1], average="macro", zero_division=0),
        "roc_auc": roc_auc_score(fake, 1-p) if two_classes else np.nan,
        "pr_auc": average_precision_score(fake, 1-p) if two_classes else np.nan,
        "balanced_accuracy": balanced_accuracy_score(y, pred) if two_classes else np.nan,
        "precision": precision_score(y, pred, pos_label=0, zero_division=0),
        "recall": recall_score(y, pred, pos_label=0, zero_division=0),
        "fpr": fp.sum()/(~fake).sum() if (~fake).any() else np.nan,
        "fnr": fn.sum()/fake.sum() if fake.any() else np.nan,
        "ece": sum(r["n"]*abs(r["accuracy"]-r["confidence"]) for r in rel)/len(y),
        "brier": brier_score_loss(y, p), "nll": nll(y, p),
    }
    order = ranking(p)
    residual = review_errors(y, p)
    # Discrete mean selective risk over retained sizes n..1; 0-coverage excluded.
    result["aurc"] = np.mean(residual[:-1]/np.arange(len(y), 0, -1))
    for frac in (.05, .1, .2):
        k = int(np.ceil(frac*len(y)))
        suffix = str(int(100*frac))
        result["error_capture_"+suffix] = errors[order[:k]].sum()/errors.sum() if errors.any() else np.nan
        result["fn_capture_"+suffix] = fn[order[:k]].sum()/fn.sum() if fn.any() else np.nan
        result["residual_error_"+suffix] = residual[k]/len(y)
    return result


def paired(y, a, b, draws=2000, seed=2026, clusters=None):
    """Stratified paired bootstrap within one seed, plus exact McNemar.

    Never pool repeated seed predictions as independent test observations.
    Bootstrap intervals describe test-sampling uncertainty conditional on training.
    """
    y = np.asarray(y, int)
    ma, mb = metrics(y, a), metrics(y, b)
    names = ("macro_f1", "pr_auc", "roc_auc", "ece", "brier", "nll")
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(y == label) for label in np.unique(y)]
    deltas = {key: [] for key in names}
    cluster_groups = None
    if clusters is not None:
        clusters = np.asarray(clusters)
        cluster_groups = [np.flatnonzero(clusters == c) for c in np.unique(clusters)]
    for _ in range(draws):
        if cluster_groups is None:
            idx = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        else:
            idx = np.concatenate([cluster_groups[j] for j in rng.integers(len(cluster_groups),size=len(cluster_groups))])
        xa, xb = bootstrap_metrics(y[idx], a[idx]), bootstrap_metrics(y[idx], b[idx])
        for key in names:
            deltas[key].append(xa[key]-xb[key])
    rows = []
    for key in names:
        values = np.asarray(deltas[key])
        finite = values[np.isfinite(values)]
        lo, hi = np.quantile(finite, [.025, .975]) if len(finite) else (np.nan, np.nan)
        rows.append({"metric": key, "difference": ma[key]-mb[key], "ci_low": lo, "ci_high": hi})
    ca, cb = (a >= .5) == y, (b >= .5) == y
    wins, losses = int((ca & ~cb).sum()), int((~ca & cb).sum())
    pvalue = binomtest(wins, wins+losses, .5).pvalue if wins+losses else 1.
    return rows, {"wins": wins, "losses": losses, "p_value": pvalue,
                  "accuracy_difference": float(ca.mean()-cb.mean())}


def holm(pvalues):
    pvalues = np.asarray(pvalues)
    order = np.argsort(pvalues)
    adjusted = np.minimum(1., np.maximum.accumulate(pvalues[order]*(len(order)-np.arange(len(order)))))
    out = np.empty(len(order))
    out[order] = adjusted
    return out
