"""Audit MATCH-CORE-10: the per-device Stage-1 min-ratio map is gone, nothing moved.

``DEFAULT_PROFILE_MATCH_MIN_DURATION_RATIO_BY_DEVICE = {dishwasher: 0.10}`` equalled
the scalar default since it was added, and five sites looked it up (the manager
twice, the effective-options report, the suggestion engine's corrective default and
the shared-settings clamp). It is deleted with its plumbing; every site reads
``DEFAULT_PROFILE_MATCH_MIN_DURATION_RATIO``. The per-device check below passes on
the old code too: it is the proof that nothing changed. (What the effective-options
report shows for the key is DETECT-02's, tested in
test_audit_detect_02_effective_min_ratio.py.)
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from custom_components.ha_washdata import const as C
from custom_components.ha_washdata.manager import WashDataManager

_TYPES = sorted(C.DEVICE_TYPES)


def test_the_map_no_longer_exists():
    assert not hasattr(C, "DEFAULT_PROFILE_MATCH_MIN_DURATION_RATIO_BY_DEVICE")


class _Entry:
    def __init__(self, device_type: str) -> None:
        self.data = {"power_sensor": "sensor.p", "name": "t", "device_type": device_type}
        self.options: dict = {}
        self.title = "t"
        self.entry_id = "e"
        self.domain = C.DOMAIN

    def async_on_unload(self, *_a, **_k):
        return None

    def add_update_listener(self, *_a, **_k):
        return lambda: None


@pytest.mark.parametrize("device_type", _TYPES)
def test_every_device_type_still_resolves_the_shipped_default(device_type):
    assert C.DEFAULT_PROFILE_MATCH_MIN_DURATION_RATIO == pytest.approx(0.10)
    mgr = WashDataManager(MagicMock(), _Entry(device_type))
    assert mgr.profile_store._min_duration_ratio == pytest.approx(0.10)  # noqa: SLF001
    # A shared bundle's raised floor is clamped back to the same default.
    out = C.sanitize_shared_settings({C.CONF_PROFILE_MATCH_MIN_DURATION_RATIO: 0.6})
    assert out[C.CONF_PROFILE_MATCH_MIN_DURATION_RATIO] == pytest.approx(0.10)
