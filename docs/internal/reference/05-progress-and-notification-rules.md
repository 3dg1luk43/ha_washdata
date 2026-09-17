# Technical Reference: `progress.py` + `notification_rules.py`

**WashData Home Assistant Integration — Internal Engineering Reference**
Generated: 2026-07-18 | Branch: 0.5.1

---

## Overview

Both modules are "pure" (no HA touches, executor-safe). They were extracted verbatim from `manager.py` to serve as a **single source of truth** consumed identically by:

- `manager.WashDataManager` (live integration) — thin instance-method wrappers
- `playground.SimRunner` (headless what-if replay) — direct calls

The byte-identical guarantee is enforced by a golden before/after snapshot plus the progress/phase/ML/energy test suite. Any arithmetic change in these files changes both live behaviour and Playground replay simultaneously.

---

## File: `custom_components/ha_washdata/progress.py`

**Module docstring** (line 17-30): Single source of truth for cycle-progress math. Pure given a `ProfileStore` (read-only), options mapping, and a replayed `(timestamp, power)` trace.

### Module-level constant

```python
PROJECTION_MIN_PROGRESS = 3.0   # progress.py:56
```

Progress must reach 3% before an energy projection is returned.

### Imported `const.py` constants

| Name | Value | Location in const.py |
|---|---|---|
| `ML_PROGRESS_BLEND_WEIGHT` | `0.5` | line 844 |
| `CYCLE_OVERRUN_ANOMALY_RATIO` | `1.5` | line 290 |
| `DEVICE_SMOOTHING_THRESHOLDS` | dict (see below) | lines 622-631 |

**`DEVICE_SMOOTHING_THRESHOLDS`** (percentage points backward-progress before heavy damping):

| Device type | Threshold |
|---|---|
| washing_machine | 5.0 |
| dryer | 3.0 |
| washer_dryer | 5.0 |
| dishwasher | 5.0 |
| air_fryer | 2.0 |
| bread_maker | 5.0 |
| pump | 2.0 |
| generic | 3.0 |

---

### `ProgressResult` dataclass (line 63-72)

```python
@dataclass
class ProgressResult:
    progress: float           # final progress % (0-99 live, 100 only at end)
    smoothed: float           # EMA-smoothed value carried to next tick (prev_smoothed)
    remaining: float          # seconds remaining
    total: float              # duration_so_far + remaining
    phase_progress: float | None  # raw pre-EMA phase estimate (diagnostic only)
    source: str               # "phase" | "linear" | "phase_blend"
```

---

### `profile_end_expectation()` (line 74-106)

**Signature:**
```python
def profile_end_expectation(
    store: Any,
    profile_name: str,
    expected_duration: float,
    cache: EndExpCache = None,
) -> tuple[dict[str, float] | None, EndExpCache]:
```

**Purpose:** Supplies per-profile median statistics (duration, peak power, energy) needed by ML feature extractors. Cached per profile so ENDING-state calls don't re-decompress history on every power reading.

**Logic:**
1. If `cache` is not None and `cache[0] == profile_name`, reuse `cache[1]` (skip decompression).
2. Otherwise: scan `store.get_past_cycles()`, filter to `profile_name`, decompress power data for each, take last 20 matching cycles, call `ml.feature_extraction.profile_expectation(points_list[-20:])`.
3. If `expected_duration > 0`, override `expectation["duration"]` with the authoritative matched value.
4. Returns `(expectation_dict | None, updated_cache)`.

**Cache type:** `tuple[str, dict[str, float]] | None` aliased as `EndExpCache`.

**Consumers:**
- `manager._profile_end_expectation()` (line 3683) — instance method, threads its own `_ml_end_expectation_cache`
- `playground.SimRunner._end_exp_fn()` (line 918) — threads `self.endexp_cache[0]`

---

### `ml_progress_percent()` (line 112-164)

**Signature:**
```python
def ml_progress_percent(
    store: Any,
    options: Any,
    matched_duration: float,
    trace: list[tuple[datetime, float]],
    profile_name: str,
    end_expectation_fn: EndExpFn,
    logger: logging.Logger | None = None,
) -> float | None:
```

**Purpose:** Returns the on-device `remaining_time` regressor's completion estimate as a percentage (0-99), or `None`. This is the ML contribution to the progress blend.

**Gate sequence (any failure -> return None):**
1. `ml_models_enabled(options)` — per-device opt-in (`CONF_ENABLE_ML_MODELS`), default off.
2. `profile_name` must be non-empty, not in `{"off", "detecting...", "restored..."}`, and present in `store.get_profiles()`.
3. `resolve_regressor("remaining_time", store)` — returns `(predict_fn, src)`. `predict_fn is None` when no on-device trained spec has been promoted (no shipped baseline for this capability — inert until training runs).
4. `trace` must have >= 4 points.
5. `end_expectation_fn(profile_name, matched_duration)` must return non-None.

**Math:**
```
t0 = trace[0][0]
pts = [(t - t0).total_seconds(), power]   # convert to offset format
feat = progress_features(pts, expectation)  # ml.feature_extraction
frac = predict_fn(feat)                     # standardized_linear regressor output
return clamp(frac, 0.0, 0.99) * 100.0
```

**Output range:** 0.0 to 99.0 (percentage). `math.isfinite` check on raw output before clamping. Never raises (broad `except Exception` -> returns `None`).

**Manager wrapper:** `manager._ml_progress_percent()` (line 3795) — thin call passing `self.detector.get_power_trace()`, `self._current_program`, `self._profile_end_expectation`.

---

### `ml_energy_total()` (line 167-221)

**Signature:**
```python
def ml_energy_total(
    store: Any,
    options: Any,
    matched_duration: float,
    trace: list[tuple[datetime, float]],
    profile_name: str,
    end_expectation_fn: EndExpFn,
    logger: logging.Logger | None = None,
) -> float | None:
```

**Purpose:** Predicts total cycle energy (Wh) using the on-device `total_energy` regressor, or `None`. Used as the preferred path in `projected_energy()` before the time-based fallback.

**Gates:** Identical to `ml_progress_percent()` — `ml_models_enabled`, profile validity, `resolve_regressor("total_energy", ...)`, min 4 trace points, non-None expectation.

**Math:**
```
feat = progress_features(pts, expectation)
frac = predict_fn(feat)           # energy-completion fraction [0, 1]
if not isfinite(frac) or frac < 0.05: return None   # floor gate
energy_so_far = cumulative_energy_wh(pts)[-1]       # integrated Wh so far
if energy_so_far <= 0.0: return None
total = energy_so_far / clamp(frac, 0.05, 1.0)
return max(total, energy_so_far)  # can never be less than already consumed
```

**Key design note:** `frac < 0.05` floor prevents a near-zero prediction from producing an astronomically large projection. The `max(total, energy_so_far)` at the end enforces a hard lower bound. Never raises.

---

### `estimate_phase_progress()` (line 224-448)

**Signature:**
```python
def estimate_phase_progress(
    store: Any,
    current_power_data: list[tuple[datetime, float]] | list[tuple[str, float]],
    current_duration: float,
    profile_name: str,
    logger: logging.Logger | None = None,
) -> tuple[float, float] | None:
```

**Purpose:** Sliding-window envelope match — scans the profile's statistical envelope and returns `(progress_pct, variance_watts)`, or `None` on failure.

**Algorithm:**

1. **Fetch envelope** via `store.get_envelope(profile_name)`. Fields: `min`, `max`, `avg`, `std` arrays (either `[[t, y], ...]` new format or `[y, ...]` legacy), `time_grid`, `target_duration`, `cycle_count`, `sampling_rates`.

2. **Extract current window:**
   ```
   window_duration = min(60.0, target_duration * 0.25)
   current_time = current_offsets[-1]
   window_start_time = max(0, current_time - window_duration)
   current_window_values = values[offsets >= window_start_time]
   ```
   Requires >= 3 samples in window; returns `None` otherwise.

3. **Sweep the time grid** (for each `i` in `time_grid`):
   - Slice envelope arrays at `[i : i + len(current_window)]`
   - Interpolate to match lengths if needed (np.interp on normalized [0,1] axis)
   - `within_bounds = all(current >= min*0.8 AND current <= max*1.2)`  (20% tolerance)
   - `bounds_score = mean(current >= min AND current <= max)`  (strict bounds fraction)
   - Correlation: `np.corrcoef(current_window, avg_window)[0,1]` if both std > 0, else 0.0
   - `mae = mean(|current - avg|)`
   - `max_power = max(max(avg_window), max(current_window), 1.0)`
   - `mae_normalized = 1 - min(mae / max_power, 1.0)`
   - **Base score formula:**
     ```
     score = 0.4 * max(corr, 0) + 0.3 * mae_normalized + 0.3 * bounds_score
     ```
   - **Time-proximity penalty:**
     ```
     time_diff = |time_window_start - current_duration|
     time_penalty = min(1.0, time_diff / (target_duration * 0.3))
     score = score * (1.0 - 0.4 * time_penalty)
     ```
   - Track `best_score`, `best_progress = (time_window_start / target_duration) * 100`, `best_time_window_start`.

4. **Threshold check:** `best_score < 0.4` -> return `None`.

5. **Variance extraction:** From std array at the best-matching position.

6. **Final clip:** `best_progress = clamp(best_progress, 0.0, 99.0)`.

7. **Returns:** `(best_progress, best_variance_watts)`.

**Manager wrapper:** `manager._estimate_phase_progress()` (line 5870) — thin call with `store=self.profile_store`, current trace, `duration_so_far`, `self._current_program`.

**Gate in manager** (line 5690): Only called when `len(trace) >= 10` and `self._current_program != "detecting..."`.

---

### `_compute_progress_base()` (line 451-549)

**Signature:**
```python
def _compute_progress_base(
    device_type: str,
    matched_duration: float,
    duration_so_far: float,
    prev_smoothed: float,
    phase_result: tuple[float, float] | None,
    ml_pct: float | None,
    logger: logging.Logger | None = None,
) -> ProgressResult | None:
```

**Purpose:** The golden-locked EMA + monotonicity + ML blend + back-calculation core. Private; called only by `compute_progress()`. Returns `None` when `matched_duration <= 0`.

#### Branch A: Phase-aware path (when `phase_result is not None`)

```
phase_progress, phase_variance = phase_result

# --- ML blend (pre-EMA) ---
if ml_pct is not None:
    w = ML_PROGRESS_BLEND_WEIGHT   # 0.5
    phase_progress = (1 - w) * phase_progress + w * ml_pct
    # i.e. phase_progress = 0.5 * phase_progress + 0.5 * ml_pct

# --- EMA smoothing ---
if prev_smoothed == 0.0:
    smoothed = phase_progress    # cold start: no smoothing
else:
    # Select alpha by phase variance
    alpha = 0.20                 # default
    if phase_variance > 100.0:
        alpha = 0.05             # high-variance: very slow tracking
    elif phase_variance > 50.0:
        alpha = 0.10             # medium-variance: moderate tracking

    smoothing_threshold = DEVICE_SMOOTHING_THRESHOLDS.get(device_type, 5.0)
    if phase_progress < prev_smoothed - smoothing_threshold:
        # Backward-progress drop: heavy damping
        smoothed = prev_smoothed * 0.95 + phase_progress * 0.05
    else:
        # Normal forward movement: standard EMA
        smoothed = prev_smoothed * (1 - alpha) + phase_progress * alpha

smoothed = min(99.0, smoothed)
progress = smoothed

remaining = matched_duration * (1 - progress / 100)
remaining = max(0.0, remaining)
total = duration_so_far + remaining

source = "phase"
```

**Key properties:**
- ML blend happens BEFORE EMA, so the ML signal is smoothed along with the phase signal.
- Three alpha tiers: variance < 50W -> alpha=0.20, 50-100W -> alpha=0.10, > 100W -> alpha=0.05.
- Backward-drop damping: EMA locked to 95/5 when phase drops more than `smoothing_threshold` pp below current.
- Hard cap: progress never exceeds 99% live.

#### Branch B: Linear fallback (when `phase_result is None`)

```
remaining = max(matched_duration - duration_so_far, 0.0)
progress = (duration_so_far / matched_duration) * 100

# --- ML blend ---
if ml_pct is not None:
    w = ML_PROGRESS_BLEND_WEIGHT   # 0.5
    progress = (1 - w) * progress + w * ml_pct
    remaining = max(matched_duration * (1 - progress / 100), 0.0)

# --- EMA smoothing ---
if prev_smoothed > 0:
    smoothed = prev_smoothed * 0.9 + progress * 0.1   # alpha fixed at 0.1
else:
    smoothed = progress

progress = clamp(smoothed, 0.0, 100.0)
remaining = max(matched_duration * (1 - progress / 100), 0.0)
total = duration_so_far + remaining

source = "linear"
```

**Difference from phase branch:** alpha is fixed at 0.1 (no variance-adaptive selection); `prev_smoothed` test is `> 0` not `== 0`; progress cap is 100.0 not 99.0 (but in practice never exceeds 100 since time caps at matched_duration); `phase_progress` field of `ProgressResult` is `None`.

---

### `compute_progress()` (line 552-604)

**Signature:**
```python
def compute_progress(
    device_type: str,
    matched_duration: float,
    duration_so_far: float,
    prev_smoothed: float,
    phase_result: tuple[float, float] | None,
    ml_pct: float | None,
    logger: logging.Logger | None = None,
    phase_remaining_s: float | None = None,
) -> ProgressResult | None:
```

**Purpose:** Public entry point. Calls `_compute_progress_base()` for the proven EMA+ML+monotonicity core, then optionally applies the **phase-resolved ETA blend** when `phase_remaining_s` is provided.

**Byte-identical guarantee:** When `phase_remaining_s is None` (the default), the result is exactly what `_compute_progress_base` returns — no change for existing callers. The golden snapshot and all prior tests continue to pass unchanged.

#### Phase-resolved ETA blend (when `phase_remaining_s` is not None):

```
base = _compute_progress_base(...)     # proven EMA core

f = clamp(base.progress / 100, 0.0, 1.0)   # completion fraction from EMA

# Weighted blend: lean on phase budget early, proven estimator late
blended_remaining = max(0.0, (1 - f) * phase_remaining_s + f * base.remaining)

total = duration_so_far + blended_remaining
if total > 0:
    progress = clamp(duration_so_far / total * 100, 0.0, 99.0)
else:
    progress = base.progress

return ProgressResult(progress, base.smoothed, blended_remaining, total,
                      base.phase_progress, "phase_blend")
```

**Design rationale (from docstring):** "lean on the phase budget early (when the base under/over-estimates most) and on the proven estimator late (near completion). `base.progress` supplies the blend weight so the result inherits the base's EMA smoothing rather than jittering per tick."

**Guard:** `phase_remaining_s` is rejected if not `math.isfinite(pr)` or `pr < 0`. Falls back to `base` silently on `TypeError`/`ValueError`.

**`source` field:** `"phase_blend"` when the ETA blend was applied (vs `"phase"` or `"linear"` from base).

**Manager call site** (line 5711-5720):
```python
result = progress_mod.compute_progress(
    self.device_type,
    float(self._matched_profile_duration),
    duration_so_far,
    self._smoothed_progress,
    phase_result,
    ml_pct,
    self._logger,
    phase_remaining_s=phase_remaining_s,
)
```

`phase_remaining_s` is populated from `profile_store.phase_remaining(trace, duration_so_far, device_type)` — gated on `phase_matching_enabled(options, device_type)` and `len(trace) >= 10` and `current_program not in ("detecting...", "off", None)`. Any failure leaves `phase_remaining_s = None`.

**Throttle:** `_update_remaining_only` throttles to one call per 5 seconds (line 5657-5661). `net_elapsed_seconds` (wall-clock minus user-paused time) is used for `duration_so_far`.

---

### `current_phase()` (line 607-635)

**Signature:**
```python
def current_phase(
    store: Any,
    state: str,
    current_program: str | None,
    cycle_progress: float,
) -> str | None:
```

**Purpose:** Returns the human-readable current phase label from the profile's configured ranges, indexed by the **ML-blended, EMA-smoothed progress fraction**. Never raises.

**Logic:**
```
if state not in {STATE_RUNNING, STATE_PAUSED, STATE_ENDING}: return None
if not profile or profile in {"off", "detecting...", "restored...", "none", "unknown"}: return None
ranges = store.get_profile_phase_ranges(profile)
if not ranges: return None
nominal = max(r["end"] for r in ranges)  # nominal end of last phase (seconds)
if nominal <= 0: return None
frac = clamp(cycle_progress / 100, 0.0, 1.0)
return store.check_phase_match(profile, frac * nominal)
```

**Key insight:** The phase lookup uses `frac * nominal` (progress fraction scaled to the profile's nominal time span), NOT raw elapsed seconds. This means overrunning or underrunning cycles still produce a correct phase label — if a cycle is 20% further through than expected, it still gets the phase that corresponds to 20% of the profile's timespan beyond where a typical cycle would be.

**Manager wrapper:** `manager._current_phase_from_progress()` (line 1379) — passes `self.detector.state`, `self._current_program`, `self._cycle_progress`.

---

### `projected_energy()` (line 638-678)

**Signature:**
```python
def projected_energy(
    store: Any,
    options: Any,
    matched_duration: float,
    trace: list[tuple[datetime, float]],
    current_program: str | None,
    cycle_progress: float,
    energy_so_far: float,
    price: float | None,
    end_expectation_fn: EndExpFn,
    logger: logging.Logger | None = None,
    cost_so_far: float | None = None,
) -> tuple[float | None, float | None]:
```

**Purpose:** Returns `(projected_wh, projected_cost)` for the running cycle. Both are `None` if progress is too low or no energy has accumulated. Never raises.

`cost_so_far` (#426) is the dynamic-tariff cost already incurred. When given, only `projected_wh - energy_so_far` is charged at `price`, so a cycle that ran through a cheap window is not retroactively repriced at the rate it happens to be in now; `None` keeps the flat `projected_wh/1000 × price`. The Playground passes the stored `price_timeline` through the same parameter, so the replay cannot diverge from the live estimator.

**Logic:**
```
if progress < PROJECTION_MIN_PROGRESS or energy_so_far <= 0: return None, None
# PROJECTION_MIN_PROGRESS = 3.0

# Preferred path: on-device regressor
projected_wh = ml_energy_total(store, options, matched_duration, trace, ...)

# Fallback: time-based linear projection
if projected_wh is None:
    projected_wh = energy_so_far / (progress / 100)

projected_wh = max(projected_wh, energy_so_far)   # hard lower bound

# Cost calculation
try:
    price_val = float(price)
except (TypeError, ValueError):
    price_val = None
cost = (projected_wh / 1000.0) * price_val if price_val is not None else None
# price=0.0 yields cost=0.0 (free/zero tariff), NOT None

return projected_wh, cost
```

**Fallback formula:** `projected_wh = energy_so_far / (progress / 100)` — purely time-proportional, assumes energy accumulates linearly.

**Manager call site:** `manager._update_projected_energy()` (line 5591) passes `self._current_program`, `self._cycle_progress`, `self._cycle_energy_wh`, `self._current_price_kwh`.

**Surfaced on:** `sensor.py`'s `projected_energy_kwh` and `projected_cost` state attributes; cleared when state leaves RUNNING/PAUSED/ENDING.

---

### `cycle_anomaly()` (line 681-694)

**Signature:**
```python
def cycle_anomaly(
    matched_duration: float,
    duration_so_far: float,
) -> tuple[float, str]:
```

**Purpose:** Computes the soft runtime overrun signal. Never raises; returns `(0.0, "none")` on any error.

**Formula:**
```
expected = float(matched_duration or 0)
if expected <= 0 or duration_so_far <= 0: return 0.0, "none"
ratio = duration_so_far / expected
anomaly = "overrun" if ratio >= CYCLE_OVERRUN_ANOMALY_RATIO else "none"
return ratio, anomaly
# CYCLE_OVERRUN_ANOMALY_RATIO = 1.5
```

**Semantics:** `ratio = 1.0` means exactly at expected duration; `ratio = 1.5` (the threshold) means the cycle has run 50% longer than the matched profile's expected duration. At this point `anomaly = "overrun"`.

**IMPORTANT:** This is a **visible-only** signal, never a notification. It surfaces as the `cycle_anomaly`/`overrun_ratio` attributes on the state sensor entity. It is distinct from the 300% "zombie-kill" hard limit (a separate CycleDetector timeout).

**Underrun** (`CYCLE_UNDERRUN_ANOMALY_RATIO = 0.55`) is NOT computed here — it is a post-cycle-end signal computed in `manager._async_process_cycle_end`. Only overrun is live.

**Manager wrapper:** `manager._update_cycle_anomaly()` (line 5629) — called after every `compute_progress()` call.

---

## File: `custom_components/ha_washdata/notification_rules.py`

**Module docstring** (line 17-31): Pure notification decision predicates. Extracted verbatim from `manager.py`. No HA touches; executor-safe. Delivery stays in the manager.

### Imported `const.py` constants

| Name | Value (conf key string) | Location |
|---|---|---|
| `CONF_NOTIFY_QUIET_START_HOUR` | `"notify_quiet_start_hour"` | line 177 |
| `CONF_NOTIFY_QUIET_END_HOUR` | `"notify_quiet_end_hour"` | line 178 |

No numeric thresholds from const.py — all thresholds are either derived from options or passed as arguments.

---

### `quiet_hours_bounds()` (line 41-64)

**Signature:**
```python
def quiet_hours_bounds(options: Any) -> tuple[int, int] | None:
```

**Purpose:** Extracts and validates the quiet-hours window from options. Returns `(start_hour, end_hour)` integers, or `None` when the feature is effectively off.

**Logic:**
```
raw_start = options.get(CONF_NOTIFY_QUIET_START_HOUR)
raw_end   = options.get(CONF_NOTIFY_QUIET_END_HOUR)

if either is None: return None
if either is bool: return None   # explicit bool-rejection (True==1, False==0 would silently coerce)
start = int(raw_start); end = int(raw_end)
if start != raw_start or end != raw_end: return None   # fractional float (1.5) rejected
if not (0 <= start <= 23) or not (0 <= end <= 23): return None
if start == end: return None   # zero-length window = feature off
return start, end
```

**Edge cases explicitly handled:**
- `bool` values (`True`/`False`) are rejected before `int()` conversion to prevent `True == 1` coercion.
- Fractional floats (e.g. `22.5`) are rejected via `int(raw) != raw` check.
- `start == end` is treated as "feature off" to avoid "always quiet" ambiguity.

**Manager wrapper:** `manager._quiet_hours_bounds()` (line 4462) — calls `notif_rules.quiet_hours_bounds(self.config_entry.options)`.

---

### `in_quiet_hours()` (line 67-83)

**Signature:**
```python
def in_quiet_hours(bounds: tuple[int, int] | None, when: datetime) -> bool:
```

**Purpose:** Returns `True` when `when` falls within the quiet window. Supports midnight-crossing windows.

**Logic:**
```
if bounds is None: return False
start, end = bounds
hour = when.hour

if start < end:
    # Same-day window: e.g. start=1, end=6 covers hours 1,2,3,4,5
    return start <= hour < end
else:
    # Wrap-around window: e.g. start=22, end=7 covers 22,23,0,1,2,3,4,5,6
    return hour >= start or hour < end
```

**End hour is exclusive** at hour granularity. Window `(22, 7)` covers 22:00 through 06:59; hour 7 is NOT quiet.

**`when` parameter contract:** The manager passes `dt_util.now()` (always HA-timezone-aware); the Playground passes the replay timestamp. The function reads only `when.hour`, so timezone is the caller's responsibility.

**Manager wrappers:**
- `manager._in_quiet_hours(when=None)` (line 4469) — defaults `when` to `dt_util.now()`
- Used at line 4847 in notification delivery gating

---

### `seconds_until_quiet_end()` (line 86-102)

**Signature:**
```python
def seconds_until_quiet_end(bounds: tuple[int, int] | None, when: datetime) -> float:
```

**Purpose:** Computes how long until quiet hours end, for scheduling a deferred notification.

**Logic:**
```
if bounds is None: return 0.0
if not in_quiet_hours(bounds, when): return 0.0

_start, end = bounds
target = when.replace(hour=end, minute=0, second=0, microsecond=0)   # today at end:00
if target <= when:
    # Wrap-around: end hour is earlier today -> it lands tomorrow
    target = target + timedelta(days=1)
return max(0.0, (target - when).total_seconds())
```

**Example:** If quiet window is `(22, 7)` and it's currently 23:30, `end = 7`, `target = today@07:00` which is `<= now`, so `target += 1 day`, returning seconds until tomorrow 07:00.

**Returns 0.0** (not None) when feature is off or not currently in quiet hours, so callers can use it unconditionally as a sleep/schedule offset.

**Manager wrapper:** `manager._seconds_until_quiet_end(when=None)` (line 4480). Used at line 4515 to schedule a deferred completion notification: `asyncio.sleep(delay)` where `delay = self._seconds_until_quiet_end()`.

---

### `milestone_crossed()` (line 105-133)

**Signature:**
```python
def milestone_crossed(prev_count: int, cur_count: int, milestones: Any) -> int | None:
```

**Purpose:** Returns the single highest milestone value crossed in the range `(prev_count, cur_count]`, or `None` if none was crossed. Firing exactly one notification when multiple milestones pass in one step (returns highest).

**Crossing condition:** `prev_count < m <= cur_count`

**Logic:**
```
if not milestones or isinstance(milestones, (str, bytes)): return None
try:
    iterator = list(milestones)
except TypeError:
    return None

crossed = None
for raw in iterator:
    if isinstance(raw, bool): continue    # reject True/False
    try:
        m = int(raw)
    except (TypeError, ValueError):
        continue
    if m != raw: continue      # reject fractional floats
    if m <= 0: continue        # must be positive
    if prev_count < m <= cur_count:
        if crossed is None or m > crossed:
            crossed = m
return crossed
```

**Type safety:** Same bool-rejection and fractional-float-rejection as `quiet_hours_bounds`. String milestones (`"50"`) are also rejected via `m != raw` (int != str).

**Returns largest crossed milestone** so one notification fires for the most significant event. If cycle count jumps from 47 to 53 and milestones are `[50, 100]`, returns `50`.

**Manager call site** (line 4614): `self._milestone_crossed(prev_count, cur_count, milestones)` where `milestones = options.get(CONF_NOTIFY_CYCLE_MILESTONES, [])`.

---

### `should_notify_pre_completion()` (line 136-156)

**Signature:**
```python
def should_notify_pre_completion(
    notify_before_end_minutes: float,
    already_notified: bool,
    time_remaining: float | None,
    cycle_progress: float,
    match_ambiguous: bool,
) -> bool:
```

**Purpose:** The one-shot "almost done" pre-completion gate. Returns `True` only when ALL of:

```
notify_before_end_minutes > 0          # feature configured
and not already_notified               # fire at most once per cycle
and time_remaining is not None         # model has a live estimate
and time_remaining <= (notify_before_end_minutes * 60)  # within lead window (seconds)
and cycle_progress < 100               # cycle not yet complete
and not match_ambiguous                # profile match must be unambiguous
```

**`match_ambiguous` guard:** Prevents the pre-completion notification from firing when two profiles are nearly indistinguishable (difference in top-1 vs top-2 score < `MATCH_AMBIGUITY_MARGIN`). Without this guard, a cycle matched to the wrong profile with a wildly off remaining estimate could fire a false "almost done" alert.

**One-shot semantics:** The manager sets `self._pre_completion_notified = True` once this fires; `already_notified` is that flag. The flag is cleared on new cycle start.

**Manager call site** (line 5551):
```python
if notif_rules.should_notify_pre_completion(
    notify_before_end_minutes=...,
    already_notified=self._pre_completion_notified,
    time_remaining=self._time_remaining,
    cycle_progress=self._cycle_progress,
    match_ambiguous=self._match_ambiguous,
):
```

**Playground call site** (line 1102): Same call, marking an event in the per-5s simulation series as `"notification_would_fire"`.

---

## Byte-Identical Guarantee: manager.py vs playground.py

Both callers follow the same pattern:

| Step | manager.py (live) | playground.py (SimRunner) |
|---|---|---|
| Get envelope match | `_estimate_phase_progress(trace, elapsed, program)` | `estimate_phase_progress(store, power_data, dur, program)` |
| Get ML % | `_ml_progress_percent(trace, program)` | `ml_progress_percent(store, options, matched_dur, trace, ...)` |
| Get phase remaining | `profile_store.phase_remaining(...)` gated on `phase_matching_enabled` | Same call in SimRunner |
| Run progress core | `compute_progress(device_type, matched_dur, elapsed, prev_smoothed, phase_result, ml_pct, logger, phase_remaining_s=...)` | Identical call (line 1084) |
| Current phase | `current_phase(store, state, program, progress)` | Same call (line 1092) |
| Projected energy | `projected_energy(store, options, matched_dur, trace, program, progress, energy, price, expfn)` | Same call (line 1095) |
| Pre-completion gate | `should_notify_pre_completion(...)` | Same call (line 1102) |

The manager's wrappers (`_estimate_phase_progress`, `_ml_progress_percent`, `_profile_end_expectation`) exist solely for per-call caching, logging context, and test mockability. They do not change the arithmetic.

---

## CLAUDE.md vs Code Discrepancies

1. **`ml_energy_total` not mentioned by name in CLAUDE.md.** CLAUDE.md lists `estimate_phase_progress`, `ml_progress_percent`, `compute_progress`, `current_phase`, `projected_energy`, `cycle_anomaly` as the module's functions. `ml_energy_total` (a direct parallel to `ml_progress_percent` for energy) and `_compute_progress_base` (the private core split out from `compute_progress`) and `profile_end_expectation` are undocumented in CLAUDE.md. These three functions exist and are called in production.

2. **`compute_progress` phase-resolved ETA blend.** CLAUDE.md says `compute_progress` is "the blend+EMA+back-calc" without mentioning the `phase_remaining_s` keyword argument and the `"phase_blend"` source mode added in Phase 4-5 of the phase-segmented matching feature (commits `2f96176` / `e712b86`). The docstring is accurate; CLAUDE.md's function description is stale.

3. **`ProgressResult.source` field.** CLAUDE.md does not document the `"phase_blend"` source value — only `"phase"` and `"linear"` are implied.

4. **`projected_energy` fallback formula.** CLAUDE.md says "energy_so_far ÷ blended-progress" which is correct for the fallback. The regressor-first preferred path is not described in CLAUDE.md.

5. **`CYCLE_UNDERRUN_ANOMALY_RATIO = 0.55`** exists in `const.py` (line 295) but `cycle_anomaly()` only computes the overrun signal. Underrun is post-cycle-end only, in `_async_process_cycle_end`. CLAUDE.md does not describe the underrun path explicitly, but the code comment at line 292-295 of `const.py` does.

---

## Quick-Reference: All Formulas

### ML Blend (pre-EMA, both branches)
```
blended = (1 - 0.5) * phase_or_linear_progress + 0.5 * ml_pct
```
`ML_PROGRESS_BLEND_WEIGHT = 0.5`

### Phase-branch EMA (post-blend)
```
# Normal forward:
smoothed = prev * (1 - alpha) + blended * alpha
# where alpha: variance > 100W -> 0.05; 50-100W -> 0.10; < 50W -> 0.20

# Backward-drop (blended < prev - threshold):
smoothed = prev * 0.95 + blended * 0.05
```

### Linear-fallback EMA (post-blend)
```
smoothed = prev * 0.9 + blended * 0.1   # fixed alpha 0.1
```

### Back-calculation (both branches)
```
remaining = matched_duration * (1 - smoothed / 100)
total = duration_so_far + remaining
```

### Phase-resolved ETA blend (layer above EMA)
```
f = clamp(base.progress / 100, 0, 1)
remaining = (1 - f) * phase_remaining_s + f * base.remaining
progress = clamp(duration_so_far / (duration_so_far + remaining) * 100, 0, 99)
```

### Overrun ratio + anomaly
```
ratio = duration_so_far / matched_duration
anomaly = "overrun" if ratio >= 1.5 else "none"
```

### Projected energy (fallback)
```
projected_wh = energy_so_far / (progress / 100)
projected_wh = max(projected_wh, energy_so_far)
cost = (projected_wh / 1000) * price_per_kwh
```

### Phase label indexing
```
frac = clamp(cycle_progress / 100, 0, 1)
label = check_phase_match(profile, frac * profile_nominal_end_seconds)
```

### Quiet-hours window test
```
# Non-wrap (start < end):
quiet = start <= hour < end
# Wrap-around (start > end, e.g. 22->7):
quiet = hour >= start or hour < end
```

### Milestone crossing
```
crossed = max(m for m in milestones if prev_count < m <= cur_count)
```

### Pre-completion gate
```
fire = (lead_minutes > 0) AND (not already_fired) AND (remaining <= lead_minutes * 60)
       AND (progress < 100) AND (not match_ambiguous)
```
