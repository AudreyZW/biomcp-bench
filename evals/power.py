"""How many tasks the live comparison actually needs.

The first live run used 25 tasks and reached p=0.210 on its strongest
comparison. That is not a result, and the useful question is whether 25 was
ever going to be enough. This module answers it before spending hours of
model time, and the answer shapes the task set.

McNemar's test looks only at discordant pairs -- tasks where the two variants
disagreed -- and asks whether the split between them is more lopsided than a
coin. So the power calculation has two inputs:

* `discordant_rate`: how often two variants disagree at all. Measured at 16/25
  for A->D and 4/25 for the adjacent steps.
* `favour`: among those disagreements, the share going the predicted way.
  Measured at 11/16 for A->D and 3/4 for adjacent steps.

Both are estimated from one small run, so they are rough. The point is the
order of magnitude: whether the answer is "a few more tasks" or "four times as
many".

The figures here are computed from the exact test, not the normal
approximation, and come out noticeably larger: the approximation puts the
mechanism comparisons at ~29 discordant pairs where the exact calculation needs
35, because at 29 pairs the real power is 0.71. Sizing a study off the
approximation would have left it underpowered while looking adequate.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

from stats import binomial_cdf, exact_mcnemar

ALPHA = 0.05
TARGET_POWER = 0.8


def _tail_probability(n: int, k: int, p: float) -> float:
    """P(X <= k) under Binomial(n, p), reusing the stable CDF from `stats`."""
    return binomial_cdf(n, k, p)


@cache
def rejection_threshold(discordant: int, alpha: float = ALPHA) -> int:
    """Largest k for which a split as lopsided as (k, discordant-k) is significant.

    Returns -1 when no split of this many pairs can clear `alpha` -- which is
    the case below six pairs, since the most extreme result available is
    2*P(X<=0) = 2/2^n and that only drops under 0.05 at n=6.
    """
    # The p-value rises with k, so binary search for the last significant one.
    low, high, threshold = 0, discordant // 2, -1
    while low <= high:
        middle = (low + high) // 2
        if exact_mcnemar(middle, discordant - middle) < alpha:
            threshold = middle
            low = middle + 1
        else:
            high = middle - 1
    return threshold


@cache
def power_at(discordant: int, favour: float, alpha: float = ALPHA) -> float:
    """Probability that `discordant` pairs would reach significance.

    A split is significant exactly when its smaller side is at or below
    `rejection_threshold`, so power is the probability of landing in either
    tail under the true rate. Computing the threshold once rather than
    p-testing every split keeps this linear in `discordant`; the enumerating
    version made the test suite take a minute and a half.
    """
    if discordant <= 0:
        return 0.0
    threshold = rejection_threshold(discordant, alpha)
    if threshold < 0:
        return 0.0
    # Reject in either tail: at most `threshold` favouring one side, or at
    # least `discordant - threshold` favouring the other.
    lower = _tail_probability(discordant, threshold, favour)
    upper = 1.0 - _tail_probability(discordant, discordant - threshold - 1, favour)
    return min(1.0, lower + upper)


def discordant_pairs_needed(
    favour: float, alpha: float = ALPHA, target_power: float = TARGET_POWER, cap: int = 2000
) -> int | None:
    """Pair count from which power stays at or above `target_power`.

    Not the first count that reaches it. Power is *not* monotonic in the pair
    count: the exact test's rejection region widens only at discrete steps, so
    power sawtooths. At favour=0.75 it first touches 0.80 at 30 pairs, then
    falls back to 0.77 at 31, 0.74 at 32 and 0.79 at 34 before settling above
    from 35 on.

    So the first crossing is not a safe size to plan around -- a study sized at
    30 pairs whose discordant count landed on 34 would be underpowered while
    appearing adequate. This returns the floor beyond which that cannot happen,
    found by scanning down from `cap` for the last insufficient count.
    """
    if favour <= 0.5:
        return None  # no effect to detect
    if power_at(cap, favour, alpha) < target_power:
        return None
    for discordant in range(cap, 0, -1):
        if power_at(discordant, favour, alpha) < target_power:
            return discordant + 1
    return 1


@dataclass(frozen=True)
class Requirement:
    comparison: str
    discordant_rate: float
    favour: float
    discordant_needed: int | None
    tasks_needed: int | None

    def describe(self) -> str:
        if self.tasks_needed is None:
            return f"{self.comparison}: no detectable effect in the observed direction"
        return (
            f"{self.comparison}: needs ~{self.discordant_needed} discordant pairs "
            f"=> ~{self.tasks_needed} tasks"
        )


def tasks_needed(
    comparison: str,
    gained: int,
    lost: int,
    total_tasks: int,
    alpha: float = ALPHA,
    target_power: float = TARGET_POWER,
) -> Requirement:
    """Translate an observed comparison into a required task count."""
    discordant = gained + lost
    rate = discordant / total_tasks if total_tasks else 0.0
    favour = gained / discordant if discordant else 0.0
    needed = discordant_pairs_needed(favour, alpha, target_power)
    tasks = None
    if needed is not None and rate > 0:
        tasks = int(-(-needed // rate))  # ceiling division
    return Requirement(comparison, rate, favour, needed, tasks)
