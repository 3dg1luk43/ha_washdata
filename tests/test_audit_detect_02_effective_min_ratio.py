"""DETECT-02 leftover: `effective_option_values` reported the wrong min duration ratio.

Since DETECT-02 the detector config's `min_duration_ratio` is the finish-deferral
ratio (0.8), not the matcher's Stage-1 bound. `effective_option_values` still read
it for `profile_match_min_duration_ratio`, so an unset key was reported as 0.8
while the matcher ran DEFAULT_PROFILE_MATCH_MIN_DURATION_RATIO (0.10):
suggestion "current" values and the equivalence filter were wrong for that key
on every fresh entry.
"""

from __future__ import annotations

import pytest

from custom_components.ha_washdata.const import (
    CONF_PROFILE_MATCH_MIN_DURATION_RATIO,
    DEFAULT_PROFILE_MATCH_MIN_DURATION_RATIO,
)
from custom_components.ha_washdata.detector_config import effective_option_values


@pytest.mark.parametrize("device_type", ["dishwasher", "washing_machine"])
def test_unset_min_ratio_reports_what_the_matcher_runs(device_type):
    eff = effective_option_values({}, device_type)
    # The manager builds its ProfileStore with this default for an unset key.
    assert eff[CONF_PROFILE_MATCH_MIN_DURATION_RATIO] == pytest.approx(
        DEFAULT_PROFILE_MATCH_MIN_DURATION_RATIO
    )
    assert eff[CONF_PROFILE_MATCH_MIN_DURATION_RATIO] == pytest.approx(0.10)


def test_a_set_min_ratio_is_reported_as_set():
    eff = effective_option_values({CONF_PROFILE_MATCH_MIN_DURATION_RATIO: 0.3}, "dishwasher")
    assert eff[CONF_PROFILE_MATCH_MIN_DURATION_RATIO] == pytest.approx(0.3)
