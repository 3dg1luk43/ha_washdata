# WashData - Home Assistant integration for appliance cycle monitoring via smart plugs.
# Copyright (C) 2026 Lukas Bandura
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
"""Auto-labelling needs a decisive margin, not just confidence (item 310).

The absolute match score is a weak predictor of being right (AUC 0.625); the
gap to the runner-up is a strong one (0.792). Labelling is the asymmetric
decision - a wrong label reshapes avg_duration and every future estimate of that
programme, while a missed one only asks the user - so it gates on the margin.

Deliberately a SEPARATE constant from MATCH_AMBIGUITY_MARGIN: that one also
reaches the detector as `_match_ambiguous` and gates Smart Termination, so
widening it there would defer cycle ends and undo item 306.
"""
from custom_components.ha_washdata.const import (
    MATCH_AMBIGUITY_MARGIN,
    MATCH_LABEL_MIN_MARGIN,
)


def test_label_margin_is_wider_than_the_termination_margin():
    """The whole point of the split: labelling is stricter than termination."""
    assert MATCH_LABEL_MIN_MARGIN > MATCH_AMBIGUITY_MARGIN


def test_termination_margin_unchanged():
    """Item 306 reduced end lag; widening this would give it back."""
    assert MATCH_AMBIGUITY_MARGIN == 0.05


def test_label_margin_matches_the_measured_operating_point():
    """0.08 is the argmax of (right - 2 x wrong) over 606 completed folds, and
    was independently selected by grouped CV in all five held-out device groups.

    Measured coverage/precision/wrong-labels at the shipped kernel:
        0.05 -> 84.3% / 86.3% / 69
        0.08 -> 77.3% / 88.7% / 52
        0.10 -> 72.2% / 90.0% / 43
    """
    assert MATCH_LABEL_MIN_MARGIN == 0.08


def test_gate_logic_rejects_a_crowded_field():
    """A confident match that is not clear of the field must not be recorded."""
    learning_confidence = 0.6

    def would_label(confidence: float, margin: float | None) -> bool:
        margin_ok = margin is None or float(margin) >= MATCH_LABEL_MIN_MARGIN
        return confidence >= learning_confidence and margin_ok

    # Confident and decisive -> labelled.
    assert would_label(0.90, 0.30) is True
    # Confident but the runner-up is right behind -> not labelled.
    assert would_label(0.90, 0.02) is False
    # Right at the boundary -> labelled.
    assert would_label(0.90, MATCH_LABEL_MIN_MARGIN) is True
    # Not confident enough, however decisive.
    assert would_label(0.10, 0.90) is False
    # No margin information (single candidate) must not block labelling.
    assert would_label(0.90, None) is True
