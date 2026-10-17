"""Gate, score and assign candidate objects to targets (NumPy/SciPy only, no models).

For candidate c and target t:
  1. Gates (all must pass, else the pair is disallowed):
       Grounding DINO score for t's noun ≥ dino_threshold          — is it that kind of object?
       c's best SigLIP target phrase has t's noun                   — does SigLIP agree on the noun?
       c's best negative does not beat its best target phrase by     — is it clearly something else?
         more than neg_ratio×
       bare nouns only: SigLIP ≥ plain_siglip_threshold and         — both models must be surer when
                        Grounding DINO ≥ plain_dino_threshold          there is no attribute to check
     SigLIP's absolute scores are not comparable across phrases (a real fork scores ~0.9, a
     real white cup seen in a gripper ~0.02), so for targets with attributes SigLIP is only
     used relatively: noun agreement here, and choosing between instances in step 3. For bare
     nouns ("fork") SigLIP is the only second opinion on Grounding DINO, so it must also
     agree in absolute terms; this is what rejects a gripper finger boxed as "fork".
  2. Composite = w·log(DINO) + (1−w)·log(SigLIP), i.e. a weighted product: both must be high.
     w is low for targets with attributes (SigLIP judges the attribute) and high for bare
     nouns (SigLIP has nothing extra to judge, and is unreliable on tiny crops).
  3. Hungarian matching on the composite: each target gets at most one candidate, each
     candidate at most one target, maximizing the total. Disallowed pairs are a large
     finite negative and any match on one is discarded as "not found".
  4. Margins: how much better the match is than the runner-up candidate for that target,
     and than sibling phrases ("white cup" vs "gray cup") on the same crop. Small margins
     mark the match "uncertain" rather than confident.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from .parser import Target

BIG = 1e6
EPS = 1e-6


@dataclass
class AssignConfig:
    dino_threshold: float = 0.2
    plain_siglip_threshold: float = 0.05    # applies only to targets without attributes
    plain_dino_threshold: float = 0.3       # bare nouns: Grounding DINO carries the decision (w_plain), so it must be surer
    w_attr: float = 0.3         # Grounding DINO weight for targets with attributes
    w_plain: float = 0.8        # Grounding DINO weight for bare nouns
    neg_ratio: float = 1.0      # reject a crop if a negative phrase beats its best target by this factor
    min_ratio: float = 1.5      # margins below this ratio (in score space) → "uncertain"


@dataclass
class TargetMatch:
    target_id: int
    candidate: int | None       # index into the candidate list, None = not found
    score: float | None         # composite (log space)
    margin: float               # log ratio vs the runner-up candidate (inf if none)
    sibling_margin: float       # log ratio vs best sibling phrase on this crop (inf if none)
    status: str                 # "confident" | "uncertain" | "not_found"


@dataclass
class AssignResult:
    matches: list[TargetMatch]
    allowed: np.ndarray                     # (C, T) bool
    composite: np.ndarray                   # (C, T) log score (−inf where disallowed)
    rejected: np.ndarray                    # (C,) bool — a negative phrase clearly won
    extra: dict[int, list[int]] = field(default_factory=dict)   # open-ended target → more candidates

    def candidate_to_target(self) -> dict[int, int]:
        out = {m.candidate: m.target_id for m in self.matches if m.candidate is not None}
        for tid, cands in self.extra.items():
            out.update({c: tid for c in cands})
        return out


def assign(
    targets: list[Target],
    dino: np.ndarray,           # (C, T) Grounding DINO score of each candidate for each target's noun
    siglip: np.ndarray,         # (C, T) SigLIP score of each candidate for each target's phrase
    siglip_neg: np.ndarray,     # (C, K) SigLIP score for each negative phrase
    cfg: AssignConfig = AssignConfig(),
) -> AssignResult:
    C, T = dino.shape
    if siglip_neg.shape[1]:
        # neg_ratio > 1 would forgive near-ties, but on this data correct rejections (a bowl
        # close-up: "yellow bowl" 0.80 vs "yellow cup" 0.53) are often closer than wrong ones,
        # and a missed object is cheaper than a wrong one (tracking fills misses), so 1.0.
        rejected = siglip_neg.max(axis=1) > cfg.neg_ratio * siglip.max(axis=1, initial=0.0)
    else:
        rejected = np.zeros(C, dtype=bool)

    nouns = np.array([t.noun for t in targets])
    has_attrs = np.array([bool(t.attributes) for t in targets])
    best_noun = nouns[siglip.argmax(axis=1)] if C else np.array([], dtype=nouns.dtype)
    noun_agrees = best_noun[:, None] == nouns[None, :]
    siglip_ok = has_attrs[None, :] | (siglip >= cfg.plain_siglip_threshold)
    dino_ok = dino >= np.where(has_attrs, cfg.dino_threshold, cfg.plain_dino_threshold)[None, :]

    allowed = dino_ok & noun_agrees & siglip_ok & ~rejected[:, None]
    w = np.where(has_attrs, cfg.w_attr, cfg.w_plain)
    composite = w * np.log(np.maximum(dino, EPS)) + (1 - w) * np.log(np.maximum(siglip, EPS))
    composite = np.where(allowed, composite, -np.inf)

    rows, cols = linear_sum_assignment(np.where(allowed, composite, -BIG), maximize=True) if C else ([], [])
    chosen = {t: c for c, t in zip(rows, cols) if allowed[c, t]}

    min_log = np.log(cfg.min_ratio)
    matches = []
    for t, target in enumerate(targets):
        c = chosen.get(t)
        if c is None:
            matches.append(TargetMatch(target.id, None, None, np.inf, np.inf, "not_found"))
            continue
        others = np.delete(composite[:, t], c)
        margin = composite[c, t] - others.max() if np.isfinite(others).any() else np.inf
        siblings = [s for s, o in enumerate(targets) if o.noun == target.noun and o.phrase != target.phrase]
        sibling_margin = (
            np.log(siglip[c, t] + EPS) - np.log(siglip[c, siblings].max() + EPS) if siblings else np.inf
        )
        status = "confident" if min(margin, sibling_margin) >= min_log else "uncertain"
        matches.append(TargetMatch(target.id, int(c), float(composite[c, t]), float(margin),
                                   float(sibling_margin), status))

    # Open-ended targets ("all cups") also take every unclaimed candidate they score best for.
    extra: dict[int, list[int]] = {}
    claimed = set(chosen.values())
    for c in range(C):
        if c in claimed or not allowed[c].any():
            continue
        t = int(np.argmax(composite[c]))
        if targets[t].open_ended:
            extra.setdefault(targets[t].id, []).append(c)

    return AssignResult(matches=matches, allowed=allowed, composite=composite, rejected=rejected, extra=extra)
