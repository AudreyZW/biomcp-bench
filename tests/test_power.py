"""Tests for the power analysis that sizes the task set.

This decided how many tasks to generate, so it should be checked against
values that can be reasoned about independently: McNemar's exact test is a
two-sided binomial at p=0.5 on the discordant pairs, so six unanimous flips is
the smallest significant result, and power rises monotonically with the number
of discordant pairs.
"""

from __future__ import annotations

import pytest

from power import discordant_pairs_needed, power_at, tasks_needed


class TestPowerAt:
    def test_no_pairs_has_no_power(self):
        assert power_at(0, 0.8) == 0.0

    def test_too_few_pairs_cannot_reach_significance(self):
        # Fewer than six discordant pairs cannot produce p < 0.05 at all, so
        # power is zero no matter how large the true effect.
        for pairs in range(1, 6):
            assert power_at(pairs, 0.99) == 0.0

    def test_six_pairs_can_just_reach_significance(self):
        assert power_at(6, 0.99) > 0.9

    def test_trends_up_with_more_pairs(self):
        values = [power_at(n, 0.75) for n in (10, 20, 40, 80)]
        assert values == sorted(values)

    def test_is_not_monotonic_in_pair_count(self):
        """The exact test's rejection region widens in discrete steps, so power
        sawtooths. This is why `discordant_pairs_needed` scans rather than
        binary searching -- a binary search here returns a count that is not the
        smallest sufficient one."""
        assert power_at(35, 0.75) > power_at(36, 0.75)

    def test_rises_with_a_larger_effect(self):
        values = [power_at(40, favour) for favour in (0.55, 0.65, 0.75, 0.85)]
        assert values == sorted(values)

    def test_no_effect_gives_power_near_alpha(self):
        # With favour=0.5 the test should reject about alpha of the time.
        assert power_at(60, 0.5) < 0.1

    def test_is_a_probability(self):
        for pairs in (1, 5, 20, 100):
            for favour in (0.5, 0.7, 0.9):
                assert 0.0 <= power_at(pairs, favour) <= 1.0


class TestDiscordantPairsNeeded:
    def test_no_effect_is_undetectable(self):
        assert discordant_pairs_needed(0.5) is None
        assert discordant_pairs_needed(0.3) is None

    def test_a_large_effect_needs_few_pairs(self):
        assert discordant_pairs_needed(0.95) <= 10

    def test_a_small_effect_needs_many(self):
        small = discordant_pairs_needed(0.55)
        large = discordant_pairs_needed(0.85)
        assert small > large

    def test_result_is_a_floor_not_a_first_crossing(self):
        """Power never drops back below target at or above the returned count,
        and the count immediately below it is insufficient."""
        for favour in (0.6, 0.7, 0.75, 0.8):
            needed = discordant_pairs_needed(favour)
            assert power_at(needed - 1, favour) < 0.8
            assert all(power_at(n, favour) >= 0.8 for n in range(needed, needed + 40))

    def test_floor_is_above_the_first_crossing_when_power_sawtooths(self):
        first_crossing = next(n for n in range(1, 200) if power_at(n, 0.75) >= 0.8)
        floor = discordant_pairs_needed(0.75)
        assert first_crossing == 30
        assert floor == 35  # power dips below 0.8 again at 31, 32 and 34


class TestTasksNeeded:
    def test_scales_by_the_discordant_rate(self):
        """Two comparisons with the same split but different disagreement
        rates need different task counts."""
        often = tasks_needed("often", 8, 2, 20)  # 50% discordant
        rarely = tasks_needed("rarely", 8, 2, 200)  # 5% discordant
        assert rarely.tasks_needed > often.tasks_needed

    def test_reproduces_the_observed_sizing(self):
        # The numbers that set this project's task count. Computed from the
        # exact test, which is more demanding than the normal approximation.
        assert tasks_needed("A->D", 11, 5, 25).tasks_needed == pytest.approx(96, abs=2)
        assert tasks_needed("C->D", 3, 1, 25).tasks_needed == pytest.approx(219, abs=3)

    def test_sizing_is_conservative_not_optimistic(self):
        """Sized off the exact test and the monotone floor, so the figure is
        larger than either the normal approximation or a first-crossing scan
        would give. Both of those would have under-sized this study."""
        assert tasks_needed("C->D", 3, 1, 25).discordant_needed == 35

    def test_wrong_direction_is_undetectable(self):
        result = tasks_needed("backwards", 2, 8, 25)
        assert result.tasks_needed is None
        assert "no detectable effect" in result.describe()

    def test_no_disagreement_is_undetectable(self):
        assert tasks_needed("identical", 0, 0, 25).tasks_needed is None

    def test_describe_mentions_both_figures(self):
        text = tasks_needed("A->D", 11, 5, 25).describe()
        assert "discordant pairs" in text and "tasks" in text
