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
"""The ONE place a CycleDetectorConfig is built from an entry's options.

The manager used to build it twice - once in its constructor, once field by field
in the options reload - and the copies drifted (register item 351). Every replay
harness then hand-rolled a third copy with its own fallbacks: `end_gate_eval`
defaulted `min_off_gap` to 480 s for every device (a dishwasher runs 3600 s) and
read a key no option is stored under, so its washer end-lag figure was 12.2 min
where the configuration users run measures 16.2 (audit DETECT-12 / F2).

Pure: no Home Assistant objects, only mappings, so the manager, the Playground
fallback and every devtools harness call the same function.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import fields
from typing import Any

from .const import (
    CONF_ANTI_CREASE_FINALIZE_RATIO,
    CONF_ANTI_WRINKLE_ENABLED,
    CONF_ANTI_WRINKLE_EXIT_POWER,
    CONF_ANTI_WRINKLE_IDLE_TIMEOUT,
    CONF_ANTI_WRINKLE_MAX_DURATION,
    CONF_ANTI_WRINKLE_MAX_POWER,
    CONF_COMPLETION_MIN_SECONDS,
    CONF_CURVE_PREROLL_SECONDS,
    CONF_DELAY_CONFIRM_SECONDS,
    CONF_DELAY_START_DETECT_ENABLED,
    CONF_DELAY_TIMEOUT_HOURS,
    CONF_DISHWASHER_END_SPIKE_QUIET_RELEASE,
    CONF_END_ENERGY_THRESHOLD,
    CONF_INTERRUPTED_MIN_SECONDS,
    CONF_MIN_OFF_GAP,
    CONF_MIN_POWER,
    CONF_OFF_DELAY,
    CONF_POWER_OFF_DELAY,
    CONF_POWER_OFF_THRESHOLD_W,
    CONF_PROFILE_MATCH_INTERVAL,
    CONF_PROFILE_MATCH_MIN_DURATION_RATIO,
    CONF_PROFILE_MATCH_THRESHOLD,
    CONF_SMART_TERMINATION_DURATION_RATIO,
    CONF_NO_UPDATE_ACTIVE_TIMEOUT,
    CONF_PROFILE_MATCH_MAX_DURATION_RATIO,
    CONF_WATCHDOG_INTERVAL,
    CONF_START_DURATION_THRESHOLD,
    CONF_START_ENERGY_THRESHOLD,
    CONF_START_THRESHOLD_W,
    CONF_STOP_THRESHOLD_W,
    DEFAULT_ANTI_CREASE_FINALIZE_RATIO,
    DEFAULT_ANTI_WRINKLE_ENABLED,
    DEFAULT_ANTI_WRINKLE_EXIT_POWER,
    DEFAULT_ANTI_WRINKLE_IDLE_TIMEOUT,
    DEFAULT_ANTI_WRINKLE_MAX_DURATION,
    DEFAULT_ANTI_WRINKLE_MAX_POWER,
    DEFAULT_COMPLETION_MIN_SECONDS,
    DEFAULT_CURVE_PREROLL_SECONDS,
    DEFAULT_DELAY_CONFIRM_SECONDS,
    DEFAULT_DELAY_START_DETECT_ENABLED,
    DEFAULT_DELAY_TIMEOUT_HOURS,
    DEFAULT_END_ENERGY_THRESHOLD,
    DEFAULT_INTERRUPTED_MIN_SECONDS,
    DEFAULT_MIN_POWER,
    DEFAULT_POWER_OFF_DELAY,
    DEFAULT_POWER_OFF_THRESHOLD_W,
    DEFAULT_DEFER_FINISH_RATIO,
    DEFAULT_PROFILE_MATCH_INTERVAL,
    DEFAULT_PROFILE_MATCH_MIN_DURATION_RATIO,
    DEFAULT_PROFILE_MATCH_THRESHOLD,
    DEFAULT_NO_UPDATE_ACTIVE_TIMEOUT,
    DEFAULT_NO_UPDATE_ACTIVE_TIMEOUT_BY_DEVICE,
    DEFAULT_PROFILE_MATCH_MAX_DURATION_RATIO,
    DEFAULT_START_ENERGY_THRESHOLDS_BY_DEVICE,
    DEVICE_COMPLETION_THRESHOLDS,
    DISHWASHER_END_SPIKE_QUIET_RELEASE_SECONDS,
    TERMINAL_DROP_DEFAULT_ON_DEVICE_TYPES,
    TERMINAL_DROP_EARLINESS_RATIO,
    TERMINAL_DROP_MIN_CLEAN_CYCLES,
    TERMINAL_DROP_MIN_PEAK_RATIO,
    TERMINAL_DROP_MIN_QUIET_SPAN_S,
    TERMINAL_DROP_PEAK_FAMILIAR_TOL,
    resolve_min_off_gap_default,
    resolve_off_delay_default,
    resolve_smart_termination_duration_ratio_default,
    resolve_start_duration_default,
    resolve_watchdog_interval_default,
)
from .cycle_detector import CycleDetectorConfig
from .ml.engine import ml_models_enabled


def _finite(value: Any, default: float) -> float:
    """``value`` as a finite float, else ``default``."""
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return number if math.isfinite(number) else default


def default_start_threshold_w(min_power: float) -> float:
    """Start threshold when the user has not set one: a little above min_power."""
    return float(min_power) + max(1.0, 0.1 * float(min_power))


def default_stop_threshold_w(min_power: float) -> float:
    """Stop threshold when the user has not set one: 60 % of min_power."""
    return float(min_power) * 0.6


def build_detector_config(
    options: Mapping[str, Any] | None,
    data: Mapping[str, Any] | None,
    device_type: str,
) -> CycleDetectorConfig:
    """Build the detector configuration an entry with these options runs.

    ``min_power`` and ``off_delay`` fall back to ``data`` (a fresh entry keeps its
    structural keys there until the first settings save, #450); everything else is
    an option with a device-resolved default.
    """
    opts: Mapping[str, Any] = options or {}
    dat: Mapping[str, Any] = data or {}

    def opt(key: str, default: Any) -> Any:
        """The option, or the default when it is missing, null or not a usable number.

        A junk value used to raise straight out of `float()`/`int()`: setup failed
        for the whole entry (item 279, audit PLATFORM-13) and every Playground
        replay errored (PLAYGROUND-10). Bools pass through untouched.
        """
        value = opts.get(key, default)
        if value is None:
            return default
        if isinstance(default, bool) or isinstance(value, bool):
            return value
        if isinstance(default, (int, float)):
            try:
                number = float(value)
            except (TypeError, ValueError, OverflowError):
                return default
            return number if math.isfinite(number) else default
        return value

    min_power = float(
        opt(CONF_MIN_POWER, _finite(dat.get(CONF_MIN_POWER), DEFAULT_MIN_POWER))
    )
    off_delay = int(
        opt(
            CONF_OFF_DELAY,
            _finite(dat.get(CONF_OFF_DELAY), resolve_off_delay_default(device_type)),
        )
    )
    return CycleDetectorConfig(
        min_power=min_power,
        off_delay=off_delay,
        # Never above the device's own completion floor: the flat 150 s overrode a
        # pump's 5 s, so every pump run under 150 s was stored `interrupted` and
        # dropped from cadence learning, suggestions and baselines (DETECT-07). For
        # every other device type the completion floor is >= 300 s, so 150 stands.
        interrupted_min_seconds=int(
            opt(
                CONF_INTERRUPTED_MIN_SECONDS,
                min(
                    DEFAULT_INTERRUPTED_MIN_SECONDS,
                    DEVICE_COMPLETION_THRESHOLDS.get(
                        device_type, DEFAULT_INTERRUPTED_MIN_SECONDS
                    ),
                ),
            )
        ),
        completion_min_seconds=int(
            opt(
                CONF_COMPLETION_MIN_SECONDS,
                DEVICE_COMPLETION_THRESHOLDS.get(device_type, DEFAULT_COMPLETION_MIN_SECONDS),
            )
        ),
        start_duration_threshold=float(
            opt(CONF_START_DURATION_THRESHOLD, resolve_start_duration_default(device_type))
        ),
        min_off_gap=int(opt(CONF_MIN_OFF_GAP, resolve_min_off_gap_default(device_type))),
        start_energy_threshold=float(
            opt(
                CONF_START_ENERGY_THRESHOLD,
                DEFAULT_START_ENERGY_THRESHOLDS_BY_DEVICE.get(device_type, 0.2),
            )
        ),
        end_energy_threshold=float(opt(CONF_END_ENERGY_THRESHOLD, DEFAULT_END_ENERGY_THRESHOLD)),
        start_threshold_w=float(
            opt(CONF_START_THRESHOLD_W, default_start_threshold_w(min_power))
        ),
        stop_threshold_w=float(opt(CONF_STOP_THRESHOLD_W, default_stop_threshold_w(min_power))),
        power_off_threshold_w=float(
            opt(CONF_POWER_OFF_THRESHOLD_W, DEFAULT_POWER_OFF_THRESHOLD_W)
        ),
        power_off_delay=float(opt(CONF_POWER_OFF_DELAY, DEFAULT_POWER_OFF_DELAY)),
        # The finish-deferral ratio - NOT the matcher's Stage-1 bound, which used to
        # be fed in here through the shared field name (audit DETECT-02).
        min_duration_ratio=DEFAULT_DEFER_FINISH_RATIO,
        match_interval=int(opt(CONF_PROFILE_MATCH_INTERVAL, DEFAULT_PROFILE_MATCH_INTERVAL)),
        # `profile_match_threshold` gates Smart Termination's confidence check; the
        # default is the value that used to be hard-coded there.
        match_confidence_threshold=float(
            opt(CONF_PROFILE_MATCH_THRESHOLD, DEFAULT_PROFILE_MATCH_THRESHOLD)
        ),
        anti_wrinkle_enabled=bool(opt(CONF_ANTI_WRINKLE_ENABLED, DEFAULT_ANTI_WRINKLE_ENABLED)),
        anti_wrinkle_max_power=float(
            opt(CONF_ANTI_WRINKLE_MAX_POWER, DEFAULT_ANTI_WRINKLE_MAX_POWER)
        ),
        anti_wrinkle_max_duration=float(
            opt(CONF_ANTI_WRINKLE_MAX_DURATION, DEFAULT_ANTI_WRINKLE_MAX_DURATION)
        ),
        anti_wrinkle_exit_power=float(
            opt(CONF_ANTI_WRINKLE_EXIT_POWER, DEFAULT_ANTI_WRINKLE_EXIT_POWER)
        ),
        anti_wrinkle_idle_timeout=float(
            opt(CONF_ANTI_WRINKLE_IDLE_TIMEOUT, DEFAULT_ANTI_WRINKLE_IDLE_TIMEOUT)
        ),
        dishwasher_end_spike_quiet_release=float(
            opt(CONF_DISHWASHER_END_SPIKE_QUIET_RELEASE, DISHWASHER_END_SPIKE_QUIET_RELEASE_SECONDS)
        ),
        # #393: resolved here (not in the gate) so the field always carries a real
        # float - playground.effective_settings() skips a None-valued field.
        smart_termination_duration_ratio=float(
            opt(
                CONF_SMART_TERMINATION_DURATION_RATIO,
                resolve_smart_termination_duration_ratio_default(device_type),
            )
        ),
        anti_crease_finalize_ratio=float(
            opt(CONF_ANTI_CREASE_FINALIZE_RATIO, DEFAULT_ANTI_CREASE_FINALIZE_RATIO)
        ),
        curve_preroll_seconds=float(opt(CONF_CURVE_PREROLL_SECONDS, DEFAULT_CURVE_PREROLL_SECONDS)),
        delay_detect_enabled=bool(
            opt(CONF_DELAY_START_DETECT_ENABLED, DEFAULT_DELAY_START_DETECT_ENABLED)
        ),
        delay_confirm_seconds=float(opt(CONF_DELAY_CONFIRM_SECONDS, DEFAULT_DELAY_CONFIRM_SECONDS)),
        delay_timeout_seconds=float(opt(CONF_DELAY_TIMEOUT_HOURS, DEFAULT_DELAY_TIMEOUT_HOURS))
        * 3600.0,
        # #378: without this every non-washing-machine device ran the
        # washing-machine detection path.
        device_type=device_type,
    )


def terminal_drop_enabled(
    device_type: str | None, options: Mapping[str, Any] | None
) -> bool:
    """Whether the terminal-drop fast finalize runs for this device (audit ML-08).

    Always for ``TERMINAL_DROP_DEFAULT_ON_DEVICE_TYPES`` (dishwashers): it is pure
    statistics, not a model, so the "Apply smart models" toggle does not gate it
    there. Every other type keeps it behind that toggle, as before. The manager's
    provider and the Playground replay both ask here, so a replay finalizes where
    live would.
    """
    if device_type in TERMINAL_DROP_DEFAULT_ON_DEVICE_TYPES:
        return True
    return ml_models_enabled(options)


def terminal_drop_may_fire(
    device_type: str | None,
    options: Mapping[str, Any] | None,
    detector: Any,
    *,
    pinned: bool = False,
) -> bool:
    """Whether a wired terminal-drop provider may fire at this moment.

    The ML-toggle path is unchanged: it may always fire. The default-on dishwasher
    path (toggle off) fires only while the detector holds a committed program
    whose match is not ambiguous (it also refused a #364 prefix-ambiguous one
    until that flag was removed in 0.5.8); ``pinned`` (a program
    the user picked by hand, which the detector is never told is "committed")
    counts as committed. Ungated, a new
    programme at a familiar power with an early pause split a real cycle (one
    "Quick wash" on 213 replayed dishwasher cycles, end_gate_eval --loo
    --all-formats); gated, that split is gone and nothing else moved, while the
    synthetic plug-pull keeps 173 of 187 fires (cuts at 15/30/50% of 95 cycles,
    median close 4.5 min after the cut either way, 99.8 min without).
    """
    if ml_models_enabled(options):
        return True
    if device_type not in TERMINAL_DROP_DEFAULT_ON_DEVICE_TYPES:
        return False
    return bool(
        (pinned or getattr(detector, "_match_committed", False))
        and getattr(detector, "_matched_profile", None)
        and not getattr(detector, "_match_ambiguous", False)
    )


def terminal_drop_baseline_for(
    cycles: Any, stop_threshold_w: float
) -> tuple[float | None, tuple[float, float] | None]:
    """``profile_store.terminal_drop_baseline`` with the shipped constants."""
    from .profile_store import terminal_drop_baseline  # pylint: disable=import-outside-toplevel

    return terminal_drop_baseline(
        cycles,
        float(stop_threshold_w),
        TERMINAL_DROP_MIN_QUIET_SPAN_S,
        TERMINAL_DROP_MIN_CLEAN_CYCLES,
    )


def terminal_drop_fires(
    points: list[tuple[float, float]],
    baseline: tuple[float | None, tuple[float, float] | None],
    stop_threshold_w: float,
) -> bool:
    """``profile_store.is_terminal_drop`` with the shipped constants.

    One call for the manager's provider and the Playground's, so the two cannot
    drift on the ratios. ``baseline`` is :func:`terminal_drop_baseline_for`'s.
    """
    from .profile_store import is_terminal_drop  # pylint: disable=import-outside-toplevel

    earliest, peak_range = baseline
    return is_terminal_drop(
        points,
        earliest,
        peak_range,
        float(stop_threshold_w),
        TERMINAL_DROP_EARLINESS_RATIO,
        TERMINAL_DROP_MIN_PEAK_RATIO,
        TERMINAL_DROP_PEAK_FAMILIAR_TOL,
    )


def apply_detector_config(target: CycleDetectorConfig, source: CycleDetectorConfig) -> None:
    """Copy every field of ``source`` onto ``target`` in place.

    The options reload keeps the live detector (and its config object, which other
    code holds a reference to) and updates it field by field; doing that with a
    hand-written list is how the reload copy drifted from the constructor's.
    """
    for field in fields(CycleDetectorConfig):
        setattr(target, field.name, getattr(source, field.name))


def effective_option_values(
    options: Mapping[str, Any] | None, device_type: str
) -> dict[str, Any]:
    """The value each suggestible key ACTUALLY runs with, set or not.

    A fresh entry has ``options == {}`` (#450), and the suggestion gates compared
    against ``None``: every noise gate was skipped for an unset key, so users were
    told to set ``off_delay`` 180 and ``watchdog`` 30 - the values they were
    already running (14% of all prompts; audit SUGGEST-10).
    """
    opts: Mapping[str, Any] = options or {}
    cfg = build_detector_config(opts, opts, device_type)

    def opt(key: str, default: Any) -> Any:
        value = opts.get(key)
        return default if value is None else value

    return {
        CONF_MIN_POWER: cfg.min_power,
        CONF_OFF_DELAY: cfg.off_delay,
        CONF_MIN_OFF_GAP: cfg.min_off_gap,
        CONF_COMPLETION_MIN_SECONDS: cfg.completion_min_seconds,
        CONF_START_THRESHOLD_W: cfg.start_threshold_w,
        CONF_STOP_THRESHOLD_W: cfg.stop_threshold_w,
        CONF_END_ENERGY_THRESHOLD: cfg.end_energy_threshold,
        CONF_POWER_OFF_THRESHOLD_W: cfg.power_off_threshold_w,
        CONF_PROFILE_MATCH_INTERVAL: cfg.match_interval,
        # The matcher's Stage-1 bound, resolved as the manager resolves it for the
        # ProfileStore - NOT `cfg.min_duration_ratio`, which since DETECT-02 is the
        # detector's finish-deferral ratio (0.8) and reported 0.8 for an unset key.
        CONF_PROFILE_MATCH_MIN_DURATION_RATIO: opt(
            CONF_PROFILE_MATCH_MIN_DURATION_RATIO,
            DEFAULT_PROFILE_MATCH_MIN_DURATION_RATIO,
        ),
        CONF_PROFILE_MATCH_MAX_DURATION_RATIO: opt(
            CONF_PROFILE_MATCH_MAX_DURATION_RATIO, DEFAULT_PROFILE_MATCH_MAX_DURATION_RATIO
        ),
        CONF_WATCHDOG_INTERVAL: opt(
            CONF_WATCHDOG_INTERVAL, resolve_watchdog_interval_default(device_type)
        ),
        CONF_NO_UPDATE_ACTIVE_TIMEOUT: opt(
            CONF_NO_UPDATE_ACTIVE_TIMEOUT,
            DEFAULT_NO_UPDATE_ACTIVE_TIMEOUT_BY_DEVICE.get(
                device_type, DEFAULT_NO_UPDATE_ACTIVE_TIMEOUT
            ),
        ),
    }


def inverted_threshold_pair(
    before: Mapping[str, Any] | None,
    after: Mapping[str, Any] | None,
    device_type: str,
) -> tuple[float, float] | None:
    """``(start, stop)`` when a write leaves ``stop_threshold_w`` at or above
    ``start_threshold_w``, else None (register item 515).

    ``before`` and ``after`` are the merged data + options around the write, read
    the way the detector reads them (an unset threshold follows ``min_power``).
    Only a write that CHANGES the effective pair is held to it: a config that
    already runs inverted (a contributed washer export: stop 6.0 W above start
    2.3 W) keeps working as today until the pair itself is edited.
    """
    after_values = effective_option_values(after, device_type)
    start = float(after_values[CONF_START_THRESHOLD_W])
    stop = float(after_values[CONF_STOP_THRESHOLD_W])
    if stop < start:
        return None
    before_values = effective_option_values(before, device_type)
    if (
        float(before_values[CONF_START_THRESHOLD_W]),
        float(before_values[CONF_STOP_THRESHOLD_W]),
    ) == (start, stop):
        return None
    return start, stop
