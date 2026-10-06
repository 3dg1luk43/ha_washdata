"""`training_stop_threshold`: the threshold on-device training cleans cycles with.

Shared by `async_run_training` and `devtools/ml_energy_gate_eval.py`. A stored
+inf (or an oversized JSON integer) must fall through to the next key: with an
infinite threshold no finite reading is active and the energy model gets no rows.
"""

from __future__ import annotations

import pytest

from custom_components.ha_washdata.const import CONF_MIN_POWER, CONF_STOP_THRESHOLD_W
from custom_components.ha_washdata.ml.training_task import training_stop_threshold


@pytest.mark.parametrize(
    ("merged", "expected"),
    [
        ({CONF_STOP_THRESHOLD_W: 3.5, CONF_MIN_POWER: 8.0}, 3.5),
        ({CONF_STOP_THRESHOLD_W: None, CONF_MIN_POWER: 8.0}, 8.0),
        ({CONF_STOP_THRESHOLD_W: 0, CONF_MIN_POWER: "x"}, 2.0),
        ({CONF_STOP_THRESHOLD_W: float("inf"), CONF_MIN_POWER: 8.0}, 8.0),
        ({CONF_STOP_THRESHOLD_W: "Infinity"}, 2.0),
        ({CONF_STOP_THRESHOLD_W: float("nan"), CONF_MIN_POWER: 8.0}, 8.0),
        ({CONF_STOP_THRESHOLD_W: 10**400, CONF_MIN_POWER: 8.0}, 8.0),
        ({}, 2.0),
    ],
)
def test_training_stop_threshold(merged, expected):
    assert training_stop_threshold(merged) == expected
