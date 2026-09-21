# 0.5.7 milestone plan

Eight issues. Evidence below is reproduced from the reporters' own exports, replayed through the
real `CycleDetector` (see "Evidence" per item). Four of the eight are one small change each; the
end-of-cycle cluster (#424 / #427 / #445) is three *distinct* root causes that were previously
treated as one.

Order of work: A (end-of-cycle cluster) -> B (suggestion engine) -> C (panel/UX) -> D (notifications).
A and B are coupled: B is what configures users into A.

---

## A. End-of-cycle cluster: #424, #427, #445

All three reporters describe "the appliance is finished but WashData keeps the cycle open".
0.5.6 shipped `_gate_cadence` (cap p95 at `5 x median`) for #424/#427; **it fixed the original
#427 trace and nothing else**, because the delay has three independent sources and 0.5.6 addressed
only part of one.

Replaying the reporters' stored traces through the shipped 0.5.6 detector reproduces each report
within ~1 minute of the live timings, so the diagnoses below are measured, not inferred.

### A1. Synthetic keepalives train the cadence estimator (primary cause of #424)

`process_reading(..., synthetic=True)` marks watchdog keepalives, and its own docstring says
"**Nothing consumes it yet**". In particular `_update_cadence(dt)` runs for synthetic readings
too. Once a publish-on-change plug falls silent, every interval in `_recent_dts` is a keepalive
spaced at the watchdog cadence, so **both** p95 *and* the median collapse onto the injection
spacing. `_gate_cadence = min(p95, 5 * median)` therefore never binds, and
`_dynamic_pause_threshold` / `_dynamic_end_threshold` (`3 x gate_cadence`) grow with the
integration's own injections - the self-feeding loop 0.5.6 set out to kill, entered from the other
side.

Measured on #424's 2026-09-17 v0.5.6 cycle (`ed421c54164c`, Beko DDN38530DX): the gate climbs
`192 s -> 530 s` purely on injected readings, taking `pause_thr` to 1590 s and `end_thr` to 1605 s,
and PAUSED alone lasts 1060 s.

**Fix:** skip `_update_cadence` for `synthetic=True` readings. The cadence estimator must only
learn what the *sensor* did. One-line guard at [cycle_detector.py:1194](../../custom_components/ha_washdata/cycle_detector.py#L1194);
update the `process_reading` docstring (it currently advertises the flag as unused) and register
item 260.

**Measured effect** (reporter's own trace, real detector): finish moves from **33.8 min -> 17.9 min**
after the physical end. Stored duration unchanged (14377 s), termination reason unchanged (`smart`).

### A2. The low-power keepalive is gated on stall timeouts, not on the watchdog cadence (primary cause of #427, contributory in #445)

In `_watchdog_check_stuck_cycle`, once `is_waiting_low_power()` holds, a 0 W keepalive is injected
only when `time_since_real_update > no_update_active_timeout` (branch 3a) or
`time_since_any_update > off_delay` (branch 3b). Both are *stall-detection* timeouts sized at
`p95_cadence * 20`; neither has anything to do with how fast the end accumulator should advance.
A publish-on-change plug going quiet at standby is the exact condition the keepalive exists for,
and it is precisely then that nothing is injected for minutes.

Measured on #427's 2026-09-17 v0.5.6 cycle (`60fcbdc82f9a`, AEG L8FE74485,
`no_update_active_timeout = 387 s`, `watchdog_interval = 30 s`): the accumulator freezes twice,
383 s before PAUSED and ~390 s inside ENDING - ~13 of the reported ~20 minutes is nothing but
"no reading arrived, so no gate was evaluated".

**Fix:** once `is_waiting_low_power()` and not `_verified_pause`, inject on the **watchdog
cadence** rather than on `no_update_active_timeout` / `off_delay`. This cannot end a cycle early:
`_time_below_threshold` accumulates wall-clock `dt` either way, so injecting more often changes
only the *latency of noticing* a crossing, never the accumulated total. Branch 2 (the staleness
force-end) and the `_verified_pause` exclusion stay exactly as they are.

**Measured effect:** finish moves from **13.2 min -> 8.2 min** after the last active reading, and
8.2 min is `effective_off_delay = max(off_delay 480, min_off_gap 480)` - i.e. exactly what the
reporter configured and expected ("I would expect WashData to leave Paused after roughly
8 minutes").

A1 and A2 are independent: A1 alone does nothing for #427, A2 alone does nothing for #424,
and neither regresses the other case (verified both ways).

### A3. `min_off_gap` silently overrides a lowered `off_delay` (#445, and TRON4R's 60-min tail in #424)

`effective_off_delay = max(off_delay, min_off_gap)` ([cycle_detector.py:2081](../../custom_components/ha_washdata/cycle_detector.py#L2081)).
`min_off_gap` defaults per device type (`washing_machine: 480`, `dishwasher: 3600`) and is not
set in either reporter's options, so:

- **#445** (Miele W6546): user set `off_delay = 180`, effective wait is **480 s**. Three of the four
  force-stops happened 364 s, 374 s and ~541 s after 0 W - i.e. the user gave up before the wait
  they never knew they had. "off_delay elapsed, nothing happened" is literally true and invisible.
- **#424 / TRON4R** and **ifurnadjiev's Samsung**: `off_delay = 180`, `min_off_gap` unset ->
  **3600 s**. That is TRON4R's constant, programme-length-independent 60.0-minute tail, and it is
  why lowering `off_delay` from 1780 to 1327 changed nothing.

**Fix (two parts):**
1. Surface it. When `min_off_gap > off_delay`, the Settings UI must say so on the `off_delay`
   control ("effective end wait: 480 s, raised by Minimum Off Gap") - a `_t()` key plus the
   derived value; no new setting.
2. Honour an explicit lowering. If the user has *explicitly* set `off_delay` below the device
   default `min_off_gap` and has not themselves set `min_off_gap`, the device-type prior is stale
   evidence and must not win. Treat the unset `min_off_gap` as `min(default, off_delay)` in the
   config builder ([manager.py:880](../../custom_components/ha_washdata/manager.py#L880)), not in
   the detector - so the Playground and `effective_settings()` report the same number the detector
   uses.

Part 2 changes end-detection timing, so it is gated on the validation in A5.

### A4. Standby above `stop_threshold_w` makes the cycle unendable (#445, cause #1 - reporter is right)

The Miele idles at 3.2-3.5 W after a programme ends; `stop_threshold_w` is 2.56 W. Idle power sits
permanently *above* the stop threshold, `_time_below_threshold` never accumulates, and the cycle
cannot end until the appliance is physically switched off. Confirmed in the export: every stored
cycle's trace ends at 3.2-3.5 W.

`_is_standby_band_stuck` already exists for exactly this, but it only fires past
`STANDBY_BAND_MIN_RATIO = 2.0` x expected duration with a >=10 min flat plateau - far too late to
help here (2x of a 91-min programme is 3 hours).

**Fix:** this is primarily B's problem - the suggestion engine produced `stop_threshold_w = 2.56`
from "minimum active power (3.2 W)" where 3.2 W *is* the idle level (see B2). Detector-side, the
proposal is to let the standby-band finalize fire at `1.0x` expected (not 2.0x) **when a profile is
matched and the plateau is below `STANDBY_BAND_MAX_FRACTION` of peak**, keeping the 2.0x gate for
unmatched cycles. Shorten-only and already trims the plateau before storing. Needs A5 validation.

### A5. Validation required before any of A1-A4 merges

Per the standing rule on detection-breaking changes:

- `./run_tests.sh --slow` (real-data replays) must stay green.
- `devtools/dtw_ab_eval.py` top-1 accuracy before/after must not regress.
- Replay the four reporter exports (Beko, AEG, Miele, Samsung) through the harness and record
  finish-time deltas in the register.
- Add `tests/test_issue_424_427_445_end_latency.py`: three fixtures (silent-plug dishwasher,
  silent-plug washer, standby-above-stop washer) asserting the finish happens within
  `effective_off_delay + one watchdog tick` of the last active reading.

### A6. Remaining, deliberately not fixed in 0.5.7

After A1+A2 the Beko still finishes ~18 min after the physical end. That residue is:
225 s (1.3 W standby above a 0.96 W stop threshold) + 212 s (first keepalive) + ~424 s (pause gate
sized from a *genuinely* sparse real cadence during the drying phase) + 318 s (the 300 s dishwasher
smart-termination debounce). Capping `_dynamic_pause_threshold` / `_dynamic_end_threshold` at a
fraction of `effective_off_delay` is the obvious next lever - a gate longer than the off delay is
incoherent, and the 0.5.6 docstring already records a case where it reached 1455 s against a 480 s
off delay. **Deferred to 0.5.8**: it needs its own A/B across the corpus, and A1+A2 already take
both reporters from "20-34 min late" to "the wait I configured".

---

## B. Suggestion engine feeds users into A: #445 (second report), contributory to #424/#427

The #445 reporter noticed this themselves: the suggestions push `off_delay` to 2033 s, which would
make their own bug far worse. They attributed it to their force-stops. That is not the cause.

### B1. `_resumed_low_runs` bills an entire intermittent wash phase as one pause

`_suggest_off_delay_from_pauses` measures "intra-cycle pauses" as spans below
`active_thr = max(stop_threshold_w, 0.02 * peak)`, closed only when the appliance then draws
active power for **120 contiguous seconds** (`_MIN_RESUME_ACTIVE_S`); any dip resets the
accumulator ("absorb the blip").

A Miele wash phase at 10 s sampling alternates ~110 W tumble for 20-60 s with 3.4 W pauses of
10-30 s. It never reaches 120 contiguous active seconds, so the *first* dip opens a low run that
stays open until a long heating/spin block finally arrives - 2000 s later. Inspecting the reported
2070 s "pause" in cycle `61fd5fcc54d2` shows 211 samples with a **maximum of 497.9 W inside it**.
It is not a pause; it is the wash.

Reproduced exactly: 30 pauses, p95 1973 s -> `off_delay` 2033 s, byte-identical to the export.

The active-floor constant is *not* the driver - I swept `0.02*peak`, `stop_threshold`,
`start_threshold`, `0.25*median_active`, `0.5*median_active` and the Miele p95 stays 1944-1976 s at
every one of them. The resume rule is the defect.

**What the number should be:** the detector resets `_time_below_threshold` on *any* reading
`>= stop_threshold_w`, so the only statistic `off_delay` can legitimately be sized from is the
longest **contiguous span below `stop_threshold_w` that then resumed**. Measured on the exports:

| device | current suggestion | genuine resumed sub-stop spans |
|---|---|---|
| #445 Miele | `off_delay` 2033 s | 69 sub-threshold readings, all single-sample (~10-30 s) |
| #427 AEG | `off_delay` 1150 s, `min_off_gap` 1150 s | **413 sub-threshold readings, none resumed** - zero real pauses |
| #424 Samsung | `off_delay` 267 s | 12 resumed, p95 75 s |

So on two of the three the correct answer is "no measured pause; keep the floor", and the shipped
heuristic returns 4-11x too large. This is the mechanism that configures users into A2/A3.

**Fix:** measure the pause as the contiguous sub-`stop_threshold_w` span, keep
`_resumed_low_runs`' active-threshold logic only to decide *whether* the appliance resumed, and
fall back to the cadence heuristic when fewer than 3 genuine spans exist. Shared by
`_suggest_off_delay_from_pauses` and `_scored_pauses`, so both move together.

### B2. `stop_threshold_w` derived from "minimum active power" when active == idle

#445's `stop_threshold_w = 2.56` came from "Based on minimum active power (3.2 W) observed in last
cycle" - but 3.2 W is also this appliance's idle level, so the derived stop threshold sits *below*
idle and the cycle can never end (A4). The current suggestion set also wants
`start_threshold_w = 3.26 W`, inside the same idle band, which would make every power-on look like
a cycle start.

**Fix:** when the p05 "lowest active power" is within a small margin of the trailing (post-cycle)
power level, the two are indistinguishable by level and the suggestion must be withheld with an
explicit reason ("this appliance's idle and lowest active power are the same, thresholds cannot be
derived from level alone") rather than emitting a value that cannot work. Surface it as an
advisory, not a silent skip.

### B3. Force-stopped cycles are not excluded from suggestion statistics

`select_clean_cycles` drops `status == "force_stopped"`, but the "Force cycle end" button goes
through `CycleDetector.user_stop()`, which stores `status="completed"`,
`termination_reason="user"`. So user-terminated cycles pass the clean filter. Confirmed in the
#445 export: three cycles with `termination_reason: "user"` counted as clean, exclusion map empty.

This is not what produced the 2033 s suggestion (B1 is), but the reporter is right that it is
wrong, and it also means `min_power` / `sampling_interval` / `completion_min_seconds` learn from
runs the user cut short.

**Fix:** add `termination_reason == "user"` to the `force_stopped` exclusion bucket in
`select_clean_cycles`, with its own reason code so the UI can say why.

---

## C. Panel: #443, #444, #442

### C1. #443 - every re-render scrolls to the top

`_render()` does `this._container.innerHTML = this._buildHtml()`
([ha-washdata-panel.js:3983](../../custom_components/ha_washdata/www/ha-washdata-panel.js#L3983)).
The scroll container `.wd-main { overflow-y: auto }` lives *inside* `_container`, so the swap
destroys it and the new one starts at `scrollTop: 0`. Focus is already preserved across the swap
(`_syncModalFocus`); scroll is not. Not Playground-specific - it hits any tab whose controls
trigger a re-render while scrolled, the Playground just re-renders on every input.

**Fix:** snapshot `scrollTop`/`scrollLeft` of `.wd-main` and of any open `.wd-modal` before the
`innerHTML` write and restore after `_wire()`, same shape as the existing focus preservation.
One central change in `_render()`. Add a Playwright spec: scroll the Playground to the bottom,
change a setting, assert `scrollTop` is unchanged.

### C2. #444 - profile card titles unreadable on light themes

`button.wd-attn-card, button.wd-profile-card { appearance: none; font: inherit; ... }`
([ha-washdata-panel.js:967](../../custom_components/ha_washdata/www/ha-washdata-panel.js#L967))
resets `font` but **not `color`**, and `.wd-profile-name` sets no color of its own. A `<button>`
therefore falls back to the UA `buttontext` system color, which follows the *browser/OS* color
scheme rather than the HA theme. The panel declares no `color-scheme` anywhere. That is exactly the
reporter's setup: Catppuccin **Auto** Latte Macchiato serving the light variant while the OS is
dark. It is the only rule in the panel that turns a card into a `<button>`, which is why only
these titles are affected and every explicitly-themed element next to them renders fine.

**Fix:** add `color: inherit` to that rule, and set `color-scheme` on `:host` so UA-derived colors
track the HA theme rather than the OS. The reporter also reports the `PROFILE (n)` heading and the
intro paragraph as unreadable; both use `var(--secondary-text-color)`, so **verify against a real
light theme before assuming they share this cause** - add a light-theme Playwright contrast check
over the Profiles tab rather than guessing.

### C3. #442 - option writes that bypass the settings changelog

Reporter's analysis is correct and is confirmed. `async_record_settings_changes` is called from
exactly two sites ([ws_api.py:1765](../../custom_components/ha_washdata/ws_api.py#L1765),
[ws_api.py:2452](../../custom_components/ha_washdata/ws_api.py#L2452)). There are **four**
bypassing writers, not the three the reporter found:

| writer | line | recorded |
|---|---|---|
| `ws_set_options` | 1786 | yes |
| `ws_store_download_device` (`include_settings`) | 1189 | no |
| `ws_import_config` (`entry_options`) | 3501 | no |
| store device-package apply | **3728** | no (not in the report) |
| `ws_apply_suggestions` ("Apply all") | 3907 | no (reproduced) |

**Fix:** the shared helper the reporter proposes, `_record_option_changes(hass, entry, updates)`,
applied at all four sites, recording the diff **before** `async_update_entry` schedules the reload
that rebuilds the store. Tag the entries with their source so history can show "applied from
suggestions" vs "imported" vs "manual". Test: apply suggestions, assert every changed key appears
in `settings_changelog` and reverts cleanly per setting.

---

## D. Notifications: #446, #417

### D1. #446 - iOS Live Activity is never ended

Reporter's diagnosis is correct and confirmed in the code. `_on_cycle_end` calls
`_clear_live_progress_notification(clear_services=False)`, which deliberately skips the
service-level `clear_notification`, and the finish notification instead carries `"activity": "end"`
([manager.py:6182](../../custom_components/ha_washdata/manager.py#L6182)). The HA companion API has
no `activity` key: an activity starts with `live_update: true`, updates by tag, and **ends only via
`clear_notification` with the same tag**. So nothing ever ends it, and the card sits frozen on the
lock screen until Apple's ~8 h expiry. The reporter confirmed a hand-sent `clear_notification`
cleared it.

The reporter is also right that `clear_services=True` is not the fix: `_live_notification_tag` *is*
`_lifecycle_tag` ([manager.py:646](../../custom_components/ha_washdata/manager.py#L646)), so a clear
sent first would dismiss-and-recreate the finish card - the flicker the current code avoids.

**Fix:** give the Live Activity its own tag (`{lifecycle}_live`) so the two surfaces are
independent, then at cycle end (a) send the finish notification on the lifecycle tag as today and
(b) send `clear_notification` on the live tag. Ordering: finish first, clear second, so the lock
screen never goes empty between them. Keep `activity: "end"` out of the payload - it is dead weight
and it documented a behaviour that never existed. Gate the clear on `_live_activity_started` so
devices that never started one get no stray call.

Also covers Liquidmasl's #417 follow-up ("the live notification never goes to done, counts up to
10 hours at 100%"). The second half of that comment - clearing on door-open - is a new feature
request, not part of this fix; split it into its own issue rather than growing #446.

### D2. #417 - already shipped, close it

The FR itself is implemented in 0.5.6: `_apply_live_notification_prefs` sets `silent: True` and
`push: {"interruption-level": "passive"}` on live *refreshes* only, gated on
`_live_activity_started` so the activity-starting push stays audible
([manager.py:7283](../../custom_components/ha_washdata/manager.py#L7283)).
`DEFAULT_NOTIFY_LIVE_SILENT = True`, and both `silent` and `push` are in `_MOBILE_ONLY_EXTRA_KEYS`
so they never reach non-mobile targets. The only open comment on the issue is the Live Activity
end, i.e. #446.

**Action:** no code change. Close #417 referencing #446, once #446 ships.

---

## Sequencing

1. **B1 + B3** (suggestion engine) - pure statistics, no detection risk, and it stops new users
   being configured into A. Ship first.
2. **A1 + A2** - the two measured end-latency fixes, behind the A5 validation gate. Biggest
   user-visible win: #424 33.8 -> 17.9 min, #427 13.2 -> 8.2 min.
3. **A3 + B2 + A4** - threshold/`min_off_gap` coherence. A3 part 1 (disclosure) is free and can go
   with (1).
4. **C1, C2, C3** - independent, parallelisable, each small.
5. **D1**, then close #417.

## Housekeeping

- Register (`docs/internal/INTEGRATION_REFERENCE.md` §7): new `[CODE]` items for A1 (and correct
  item 260, which records the synthetic flag as having no consumer), A2, A3, B1, B3, C1, C2, C3,
  D1; mark the #424/#427 entries as partially-fixed-in-0.5.6 with the measured residue rather than
  `[FIXED]`.
- `CHANGELOG.md`: re-cut the 0.5.7 `### TL;DR` as a whole once the set is known.
- After any `www/*.js` edit: `node devtools/build_panel.mjs` and commit the artifacts in the same
  commit.
