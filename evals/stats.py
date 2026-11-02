"""Paired significance testing for the live comparisons.

Every variant runs the same tasks against the same model, so the comparisons
are paired and McNemar's test is the right one: it looks only at the tasks
where two variants disagreed, and ignores the ones they both got right or both
got wrong.

This exists because the first live run produced 9, 11, 13, 15 out of 25 across
the four variants -- a perfectly monotonic trend that is not significant at any
step. A benchmark that printed the trend without the p-value would be making a
claim its own data does not support, which is exactly the failure this project
is about.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, fsum, inf, lgamma, log, log1p


@dataclass(frozen=True)
class Comparison:
    gained: int  # tasks the second variant solved and the first did not
    lost: int  # tasks the first solved and the second did not
    p_value: float

    @property
    def discordant(self) -> int:
        return self.gained + self.lost

    @property
    def net(self) -> int:
        return self.gained - self.lost

    def verdict(self, alpha: float = 0.05) -> str:
        if self.discordant == 0:
            return "identical"
        if self.p_value < alpha:
            return "significant"
        return "not significant"


def log_binomial_pmf(n: int, k: int, p: float = 0.5) -> float:
    """log P(X = k) for X ~ Binomial(n, p), via log-gamma.

    Log space because the direct form overflows: `comb(2000, 1000)` is a
    600-digit integer, and multiplying it by a float raises. The power analysis
    searches up to a couple of thousand pairs, so this has to stay finite there.
    """
    if k < 0 or k > n:
        return -inf
    if p <= 0.0:
        return 0.0 if k == 0 else -inf
    if p >= 1.0:
        return 0.0 if k == n else -inf
    log_coefficient = lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)
    return log_coefficient + k * log(p) + (n - k) * log1p(-p)


def binomial_cdf(n: int, k: int, p: float = 0.5) -> float:
    """P(X <= k), summed in log space and clamped to [0, 1]."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    total = fsum(exp(log_binomial_pmf(n, i, p)) for i in range(k + 1))
    return min(1.0, max(0.0, total))


def exact_mcnemar(gained: int, lost: int) -> float:
    """Two-sided exact binomial p-value on the discordant pairs.

    The exact binomial rather than the chi-square approximation, because the
    discordant counts here are often single digits and the approximation is not
    trustworthy that low.
    """
    n = gained + lost
    if n == 0:
        return 1.0
    return min(1.0, 2 * binomial_cdf(n, min(gained, lost)))


def compare(first: dict[str, bool], second: dict[str, bool]) -> Comparison:
    """Compare two variants' per-task outcomes over their shared tasks."""
    shared = sorted(set(first) & set(second))
    gained = sum(1 for t in shared if not first[t] and second[t])
    lost = sum(1 for t in shared if first[t] and not second[t])
    return Comparison(gained=gained, lost=lost, p_value=exact_mcnemar(gained, lost))


def outcomes_by_variant(rows: list[dict]) -> dict[str, dict[str, bool]]:
    table: dict[str, dict[str, bool]] = {}
    for row in rows:
        table.setdefault(row["variant"], {})[row["task"]] = bool(row.get("derivable"))
    return table


def paired_report(rows: list[dict], pairs: list[tuple[str, str]]) -> list[str]:
    """Lines describing each adjacent comparison, with its p-value."""
    table = outcomes_by_variant(rows)
    lines = [
        f"{'comparison':<14}{'gained':>7}{'lost':>6}{'net':>5}{'p':>8}  verdict",
        "-" * 60,
    ]
    for earlier, later in pairs:
        if earlier not in table or later not in table:
            continue
        result = compare(table[earlier], table[later])
        lines.append(
            f"{earlier + ' -> ' + later:<14}{result.gained:>7}{result.lost:>6}"
            f"{result.net:>+5}{result.p_value:>8.3f}  {result.verdict()}"
        )
    return lines
