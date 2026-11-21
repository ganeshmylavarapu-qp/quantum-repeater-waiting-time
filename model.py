"""Chain and star workload provisioning and the original repeater waiting prescription."""
import itertools
import math
import random
from functools import lru_cache
import numpy as np


def edges(n, topology='chain'):
    if topology == 'star':
        return tuple((0, i) for i in range(1, n))
    if topology == 'chain':
        return tuple((i, i + 1) for i in range(1, n))
    raise ValueError('Use chain or star topology.')


def paths(n, topology='chain'):
    if topology == 'star':
        return [(s, t) if s == 0 else (s, 0, t)
                for s, t in itertools.combinations(range(n), 2)]
    edges(n, topology)
    return [tuple(range(s, t + 1)) for s in range(1, n + 1) for t in range(s + 1, n + 1)]


def demands(n, model, topology='chain'):
    if n < 2 or model not in (0, 1, 2):
        raise ValueError('Use N >= 2 and model 0, 1 or 2.')
    return tuple(0 if model == 0 else n * (n - 1) // 2 if model == 1 else (n - 1 if topology == 'star' else i * (n - i))
                 for i, _ in edges(n, topology))


def single_probability(distance, alpha=.21, detector=.95, rho=.9):
    return detector**2 * rho**2 * 10**(-alpha * distance / 10)


@lru_cache(maxsize=8192)
def binomial_tail(m, p, k):
    """Stable P(Binomial(m,p)>=k), including impossible demands and endpoints."""
    if int(m) != m or m < 0 or int(k) != k or not 0 <= p <= 1:
        raise ValueError('m and k must be integers; m >= 0 and 0 <= p <= 1.')
    m, k = int(m), int(k)
    if k <= 0:
        return 1.
    if k > m or p == 0:
        return 0.
    if p == 1:
        return 1.
    # Sum the small tail directly; use expm1 for the complementary tail.
    if k > m * p:
        x = np.arange(k, m + 1, dtype=float)
        first = (math.lgamma(m + 1) - math.lgamma(k + 1) - math.lgamma(m - k + 1)
                 + k * math.log(p) + (m - k) * math.log1p(-p))
        ratios = np.log((m - x[:-1]) / (x[:-1] + 1)) + math.log(p) - math.log1p(-p)
        logs = first + np.concatenate(([0.], np.cumsum(ratios)))
        peak = float(np.max(logs))
        result = math.exp(peak) * float(np.exp(logs - peak).sum())
    else:
        x = np.arange(k - 1, dtype=float)
        ratios = np.log((m - x) / (x + 1)) + math.log(p) - math.log1p(-p)
        logs = m * math.log1p(-p) + np.concatenate(([0.], np.cumsum(ratios)))
        peak = float(np.max(logs))
        log_cdf = peak + math.log(float(np.exp(logs - peak).sum()))
        result = -math.expm1(min(0., log_cdf))
    return float(np.clip(result, 0, 1))


def provisioning(n, model, m, p, topology='chain'):
    ks = demands(n, model, topology)
    bs = [binomial_tail(m, p, k) for k in ks]
    log_q = sum(math.log(b) for b in bs) if all(bs) else -math.inf
    return math.exp(log_q), ks, bs, log_q


def exposure(order, n, topology='chain'):
    """Same per-repeater counters and segment exposure as the previous app.

    Direct paths use no swap resource. Ordered requests increment their
    intermediate repeater counters. This is not a new global serial-time model.
    """
    counters = {}
    by_path = {}
    for path in order:
        waits = {r: counters.get(r, 0) for r in path[1:-1]}
        for r in waits:
            counters[r] = waits[r] + 1
        segments, last, total = len(path) - 1, 0, 0
        for tick in sorted(set(waits.values())):
            total += segments * (tick - last)
            segments -= sum(w == tick for w in waits.values())
            last = tick
        by_path[path] = (total, (len(path) - 1) * max(waits.values(), default=0))
    return np.asarray([by_path[p] for p in paths(n, topology)], dtype=float)


@lru_cache(maxsize=16)
def schedules(n, max_exact=10000, samples=300, seed=7, topology='chain'):
    ps = paths(n, topology)
    direct = [p for p in ps if len(p) == 2]
    conflict = [p for p in ps if len(p) > 2]
    exact = len(conflict) <= 8 and math.factorial(len(conflict)) <= max_exact
    if topology == 'star':
        # Every leaf-to-leaf request visits the same hub and has the same
        # path length. Orders permute the same set of hub waiting counts.
        orders, exact = [conflict], True
    elif exact:
        orders = itertools.permutations(conflict)
    else:
        rng = random.Random(seed)
        candidates = [tuple(conflict), tuple(reversed(conflict)),
                      tuple(sorted(conflict, key=len)), tuple(sorted(conflict, key=len, reverse=True))]
        for _ in range(samples):
            trial = conflict.copy()
            rng.shuffle(trial)
            candidates.append(tuple(trial))
        orders = dict.fromkeys(candidates)
    ordered = [tuple(direct) + tuple(order) for order in orders]
    return tuple(ordered), np.asarray([exposure(order, n, topology) for order in ordered]), exact


def fidelities(n, model, m, distance, deltas, T=200., alpha=.21, d1=.95,
               d2=.95, rho=.9, mem=.95, fiber=1.5, policy=0,
               max_exact=10000, samples=300, seed=7, topology='chain'):
    p = single_probability(distance, alpha, d1, rho)
    q, ks, bs, log_q = provisioning(n, model, m, p, topology)
    orders, exposures, exact = schedules(n, max_exact, samples, seed, topology)
    lengths = np.asarray([len(path) - 1 for path in paths(n, topology)])
    t0 = distance * fiber / 300.  # km -> one-way full-link propagation in ms
    swap = np.ones(len(lengths)) if model == 0 else (mem * d2)**(2 * (lengths - 1))
    dt = np.atleast_1d(deltas).astype(float)
    best, worst, best_ix, worst_ix = [], [], [], []
    for delta in dt:
        decay = np.exp(-(lengths[None, :] * t0 + exposures[:, :, policy] * delta) / T)
        values = .5 + .5 * q * decay * swap[None, :]
        averages = values.mean(axis=1)
        high, low = int(np.argmax(averages)), int(np.argmin(averages))
        best.append(averages[high]); worst.append(averages[low])
        best_ix.append(high); worst_ix.append(low)
    return dict(best=np.array(best), worst=np.array(worst), best_ix=best_ix, worst_ix=worst_ix,
                q=q, log_q=log_q, k=ks, b=bs, p=p, swap=swap, t0=t0,
                orders=orders, exposures=exposures, exact=exact)
