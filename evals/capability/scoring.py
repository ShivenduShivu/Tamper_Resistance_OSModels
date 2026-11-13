"""Statistics for the capability eval suite.

Two ideas only:
1. Report every rate with a 95% Wilson interval (same convention as the article).
2. Compare models on the SAME items with a paired analysis: a full-precision
   exact McNemar p-value on discordant items plus a Newcombe method-10 paired CI
   for the accuracy difference (calibrated at the boundary; never degenerates to
   zero width). This is far more sensitive than comparing two independent Wilson
   intervals when models share most answers, and it is the analysis the existing
   aggregate MMLU runs could not do, because per-item correctness was never logged.

Conventions:
- The sign convention everywhere is CANDIDATE MINUS BASE: mcnemar() takes the
  candidate's per-item correctness first, and a positive delta means the candidate
  is BETTER than base. Callers and output keys must say "candidate_minus_base".
- An empty pack is an error, not a 0% score: rate_with_ci raises on zero items.
  A pack that produced no scored items must never appear in a table as measured.
"""
import math


def wilson(k, n, z=1.96):
    """95% Wilson score interval for k successes out of n. Returns (low, high).

    wilson(0, 0) returns the uninformative (0, 1); it exists only so callers can
    inspect it. rate_with_ci refuses n=0 instead.
    """
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def rate_with_ci(items):
    """items: iterable of 0/1 or bool. Returns (rate, low, high, n).

    Raises ValueError on an empty pack: an unscored pack is a failed measurement,
    not a 0% with a [0, 0] interval.
    """
    xs = [1 if x else 0 for x in items]
    n = len(xs)
    if n == 0:
        raise ValueError("empty pack: refusing to report a rate for zero scored items")
    k = sum(xs)
    lo, hi = wilson(k, n)
    return (k / n, lo, hi, n)


def newcombe_paired_ci(n_both, n_cand_only, n_base_only, n_neither, z=1.96):
    """Newcombe's method 10: 95% CI for the difference of two PAIRED proportions.

    Hybrid of the two marginal Wilson score intervals with a correction for the
    correlation between them (phi coefficient, with Newcombe's n/2 shrinkage of a
    positive covariance term; phi = 0 when a marginal is degenerate).

    Verified against statsmodels' confint_proportions_paired(method="newcomb")
    (devel), which is itself verified against R's ratesci::moverpairci and
    PropCIs. Reference: Newcombe (1998), Statistics in Medicine 17(22):2635-50.

    Unlike the Wald/normal interval this does NOT degenerate to zero width at the
    boundary: 100 candidate-only wins out of 100 gives about [0.948, 1.0], not
    [1, 1]. It also has positive width when all pairs are concordant - which is
    correct: n identical answers cannot pin the true difference to exactly 0.
    """
    n = n_both + n_cand_only + n_base_only + n_neither
    if n <= 0:
        raise ValueError("empty table")
    k1, k2 = n_both + n_cand_only, n_both + n_base_only
    p1, p2 = k1 / n, k2 / n
    diff = p1 - p2
    l1, u1 = wilson(k1, n, z)
    l2, u2 = wilson(k2, n, z)
    marg_prod = (n_both + n_cand_only) * (n_neither + n_base_only) * \
                (n_both + n_base_only) * (n_cand_only + n_neither)
    if marg_prod == 0:
        phi = 0.0
    else:
        cov = n_both * n_neither - n_cand_only * n_base_only
        if cov > 0:
            cov = max(cov - n / 2, 0.0)   # Newcombe's continuity-style shrinkage
        phi = cov / math.sqrt(marg_prod)
    dl, du = p1 - l1, u2 - p2
    low = diff - math.sqrt(max(dl * dl - 2 * phi * dl * du + du * du, 0.0))
    dl, du = u1 - p1, p2 - l2
    upp = diff + math.sqrt(max(dl * dl - 2 * phi * dl * du + du * du, 0.0))
    return (low, upp)


def mcnemar(correct_candidate, correct_base):
    """Paired comparison of a candidate model against base on the same items.

    correct_candidate, correct_base: equal-length iterables of 0/1 for the two
    models on the SAME prompts, in the SAME order. CANDIDATE FIRST.

    Returns dict with:
      delta                 - accuracy(candidate) - accuracy(base); positive means
                              the candidate is better
      delta_ci              - Newcombe method-10 95% CI for delta (see
                              newcombe_paired_ci). Calibrated at the boundary;
                              never zero-width.
      candidate_only_correct, base_only_correct - discordant counts (b, c)
      mcnemar_p_exact       - two-sided exact binomial p on the discordant pairs,
                              FULL precision (never rounded to 0.0000); this exact
                              test is the primary decision statistic
      detected_loss         - p_exact < 0.05 AND the CI excludes 0 AND delta < 0:
                              the pre-registered shape of a capability COST
      detected_gain         - same with delta > 0

    A narrow CI around 0 on mostly-concordant predictions is not evidence of
    equivalence; "not detected" is reported as "not detected at this
    resolution", never as "no difference".
    """
    xc = [1 if x else 0 for x in correct_candidate]
    xb = [1 if x else 0 for x in correct_base]
    assert len(xc) == len(xb) and len(xc) > 0, "paired inputs must be same non-zero length"
    n = len(xc)
    both = sum(1 for a, b in zip(xc, xb) if a and b)
    b = sum(1 for a, bb in zip(xc, xb) if a and not bb)   # candidate correct, base wrong
    c = sum(1 for a, bb in zip(xc, xb) if not a and bb)   # candidate wrong, base correct
    neither = n - both - b - c
    pa = sum(xc) / n
    pb = sum(xb) / n
    delta = pa - pb
    ci = newcombe_paired_ci(both, b, c, neither)
    m = b + c
    if m == 0:
        p_exact = 1.0
    else:
        k = min(b, c)
        tail = sum(math.comb(m, i) for i in range(0, k + 1)) / (2 ** m)
        p_exact = min(1.0, 2 * tail)
    excludes_zero = ci[0] > 0 or ci[1] < 0
    significant = p_exact < 0.05 and excludes_zero
    return {
        "n": n, "acc_candidate": pa, "acc_base": pb, "delta": delta,
        "delta_ci": [round(ci[0], 4), round(ci[1], 4)],
        "candidate_only_correct": b, "base_only_correct": c,
        "mcnemar_p_exact": p_exact,  # full precision: tiny p must not print as 0.0000
        "detected_loss": bool(significant and delta < 0),
        "detected_gain": bool(significant and delta > 0),
    }
