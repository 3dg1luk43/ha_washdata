# Open-issue triage & implementation plan

Point-in-time triage of the open GitHub backlog. Generated 2026-08-11.

Scope: every open issue **except #215** (additional-sensors FR, deferred by maintainer) **and #251**
(water-consumption FR, WIP). 24 issues covered.

Each issue carries a **verdict**, an **effort bucket**, and a concrete **approach sketch**. Effort buckets:

- **XS** trivial, under ~half a day
- **S** a few hours to ~1 day
- **M** 1-2 days
- **L** several days, higher regression surface
- **XL** week+, multi-subsystem

Verdicts: **FIX** (real bug, fix it), **IMPLEMENT** (FR that fits, build it),
**REFUSE** (does not fit a passive power monitor / not viable for 8000+ users),
**NEEDS-INFO** (act only after the reporter supplies missing data), **DONE** (already shipped).

> Note: no em-dash characters are used in this repo; hyphens and parentheses only.

---

## 1. Master list, sorted by effort

| # | Title (short) | Verdict | Effort | Ready fork/PR? |
|---|---|---|---|---|
| 328 | Blocking call at panel startup | FIX (code already on 0.5.4) | XS (release only) | fixed in `11dc122` |
| 359 | Panel replaces header, sidebar trap on tablet | FIX | XS | no |
| 346 | Two DEBUG lines for silent Smart-Termination | IMPLEMENT | XS | fork `cec295b`/`c7b8b6e` |
| 355 | Review counter says N, list is empty | FIX | XS-S | no |
| 367 | Make `power_profile_interval_min` configurable | IMPLEMENT | XS-S | offered |
| 348 | Verified-pause release divides by `avg_duration` | FIX | S | fork `3eb06b5` |
| 354 | Tile card ignores entity Display Precision | FIX | S | no |
| 362 | Stale "needs review" count after merge/split + DE plural typo | FIX | S | no |
| 347 | Live-progress notification: sticky + clickAction | IMPLEMENT | S | offered |
| 363 | Misses unchanged reports (`state_reported`) | FIX | S | part of #329 fork |
| 331 | Relabel resolves feedback (residual only) | DONE core / S residual | S | merged PR #336 |
| 369 | Inconsistent UTC offsets in stored timestamps | FIX (hygiene) + NEEDS-INFO (display) | S | no |
| 339 | Anti-wrinkle stuck when publish-on-change sensor goes silent | FIX | S-M | PR #340 open |
| 329 | Power sensor lags at finish (mislabeled `done`) | FIX + relabel | M | fork v0.5.3.7/.9 |
| 350 | Envelope alignment drops timestamps (per-sample not per-second) | FIX | M | fork `3400029` |
| 366 | Trim clock mode collapses cycle to 0s (DATA LOSS) | FIX | M | no |
| 343 | Auto-tune re-suggests thresholds that break anti-crease | IMPLEMENT | M | no |
| 342 | Door sensor support for dishwashers that auto-open | IMPLEMENT | M | no |
| 353 | Consume external "programme" value from smart appliance | IMPLEMENT | M | no |
| 364 | Smart-term splits wash at shorter prefix profile (cf #288) | FIX | L | no (needs export) |
| 334 | Water-level variants indistinguishable, no fill role | IMPLEMENT (`accepted`) | L | no (needs export) |
| 344 | Import historical power data (CSV / recorder) | NEEDS-INFO then phased | L-XL | no |
| 368 | Remote start of washers/dryers | REFUSE | n/a | no |
| 352 | "Summer break" notice | non-actionable | n/a | no |

**Ready-to-review fork/PR work** (much cheaper than the effort bucket implies, it is review not authoring):
#328 (`11dc122`), #339 (PR #340), #346, #348, #350, #329 (all kdjkdjkdj forks), #347/#367 (reporters offered PRs).

---

## 2. Status corrections to make right now (no code)

These are label/state fixes that keep the tracker honest:

1. **#329 is mislabeled `done` but is NOT fixed.** The two commits shipped under it (`66de4b7` retire dead
   `running_dead_zone`, `ef5e83b` chart-tail overlay) address a cosmetic freeze and a dead knob, not the
   detection lag. The real root cause (missing `async_track_state_report_event`) is unaddressed and still
   reproduced by 4 users on 0.5.3. **Remove `done`, keep `accepted`.**
2. **#328 must stay open until 0.5.4 ships.** The blocking-call fix exists in code on the `0.5.4` branch
   (`11dc122`) but `manifest.json` still reads `0.5.3` and the branch is not merged to `main`. Every 0.5.3
   user still hits the `ws_api.py` manifest read on restart, so "me too" reports will keep arriving until
   released. Bump the manifest, merge, release, then close.
3. **#362 headline premise is wrong.** Merge is NOT gone: it is a multi-select action in the Cycles tab
   (`ha_washdata/apply_merge` WS + `cyc-merge` button). Reframe the issue to the two real bugs it surfaced
   (stale review count, DE plural) and the discoverability ask. There is intentionally no `merge_cycles`
   HA service.
4. **#331 core is genuinely DONE** (PR #336, merged `b0a5ea7`, in 0.5.3). Only a small UX residual remains
   (legacy cycles with no pending feedback keep the red dot after relabel). Keep `done`; optionally split
   the residual.
5. **Issues needing `accepted` before their fork PRs can land** (contribution flow gates on it): #339, #348,
   #350, #363, #364, #366, #354, #355, #369.

---

## 3. Recommended work batches (coupling matters)

Several issues share a root cause or call site and should land as one change set, not piecemeal.

### Batch A - "Silent / publish-on-change power sensor" (highest user impact)
Zigbee, MQTT, Tasmota and Shelly plugs that publish on change (a very large slice of 8000+ installs)
all suffer variants of the same gap: no callback arrives when power is flat, so timers that only advance
inside `process_reading` never fire.

- **#363 + #329 are the same fix.** Both are the missing `async_track_state_report_event` subscription on
  the power entity. #363 is the "unchanged re-report is invisible" symptom; #329 is the "stuck-high value
  for the full reporting interval, cycle finishes ~15 min late" symptom. Land one opt-in second listener
  routed into `_async_power_changed`, plus the optional staleness-guarded heartbeat from the fork.
  Gate opt-in (default off) because report events shift `p95_dt` and the derived dynamic pause/end
  thresholds on existing installs.
- **#339 (anti-wrinkle stuck)** is the same failure class but in the anti-wrinkle tail, where BOTH detector
  timers are stopped. Its fix (drive an idle keepalive from the state-expiry timer, gated on
  `_last_real_reading_time`) is orthogonal to the `state_reported` fix and can land independently via
  PR #340, but validate the two together so keepalive cadence does not double-count.

Combined effort ~M. Test gate: detection/termination suites, not `dtw_ab_eval`.

### Batch B - "Verified-pause / Smart-Termination envelope math" (kdjkdjkdj fork pair)
Same call site (`manager.py:1234` release <-> `analysis.verify_profile_alignment_worker`), two independent
defects, neither subsumes the other. On a profile hit by both, fixing only one still hangs.

- **#348** the 0.95 release compares `mapped_time` (capped at envelope span) against `avg_duration`
  (a differently-derived trimmed mean), so the ceiling `span/avg_duration` can drop below 0.95 and the
  release becomes arithmetically unreachable. Fix: divide by the envelope's own span. (S)
- **#350** the live trace is fed to the aligner with timestamps discarded, so position advances per sample
  not per second; a sparse 0 W tail crawls ~30x slower than the clock and waits out the 4 h cap. Fix:
  resample the live trace onto the envelope's time step before DTW. (M) Open design choice for the
  maintainer: linear interp (matches build side) vs zero-order hold (more faithful to publish-on-change).
- **#346** two DEBUG lines at these exact decision points. Ship it in this batch so the next late-finish
  report is diagnosable. (XS)
- **#364** the harder cousin (see its section) can be worked after A/B tooling is warm, but keep it separate.

Note: #350 lives on the pause-release path, NOT the Stage 1-5 matcher, so the "sample-count beats
time-aware features" result (which is about matching scoring) does not apply here, and `dtw_ab_eval` is not
the validator. Standard termination/replay suites are.

### Batch C - "Review / feedback queue consistency"
The `pending_feedback` map and the panel's review surface drift apart in three ways:

- **#355** ML review stamps `ml_review.reviewed_at` but never resolves the pending feedback, so the header
  counter (`len(get_pending_feedback())`) counts a cycle the list hides (`needsReview` short-circuits on
  `isReviewed`). Counter=N, list empty.
- **#362 (bug 1)** `apply_merge_interactive` / `apply_split_interactive` remove cycles but never prune their
  `pending_feedback`, so the badge sticks after a merge.
- **#331 residual** manual relabel of a cycle with no pending feedback does not stamp `reviewed_at`, so the
  red dot persists with no resolve buttons.

The durable fix is single-source-of-truth on the backend: whenever a cycle is removed, reviewed, or
manually labeled, reconcile `pending_feedback` (call the existing `prune_orphaned_feedback()` from the
mutation paths, and have `ws_set_ml_review` resolve pending feedback like the label path does). Then make
the header counter use the same predicate as the list. Plus a one-shot reconcile so existing stuck entries
clear without the JSON workaround. Combined effort ~S-M. Also fix the DE plural typo (`de.json:714`
"Zykluss" -> "Zyklen") and note the `{s}` suffix pluralization hack is a systemic i18n smell affecting
many languages (out of scope to fully fix here).

### Batch D - "Timestamp / trim" (data-loss priority)
- **#366** is silent, permanent data loss (clock-mode trim collapses a cycle to 0s, no undo) and trim is a
  headline feature. Prioritize. Two defects: destructive backend with no min-window floor and no backup,
  plus a fragile local-clock->offset conversion.
- **#369** inconsistent stored UTC offsets are the upstream trigger for #366's offset mismatch. Normalize
  every timestamp write to canonical UTC (`dt_util.as_utc(...).isoformat()`), optionally with a v12 storage
  migration to canonicalize existing cycles (idempotent, instant-preserving).

Fix together. The #369 "wrong relative time" display symptom appears misdiagnosed (the list shows
time-since-**start**, which was correct in the reporter's screenshot); the actionable part is the storage
hygiene. Reply to clarify the display and proceed with normalization.

---

## 4. Per-issue detail

Ordered within each verdict group by ascending effort.

### 4.1 FIX (bugs)

#### #328 - Blocking call at panel startup  [XS, code done on 0.5.4]
`get_cache_buster()` did sync `scandir`/`getmtime` on the loop, and `ws_api.py` did a module-level
`manifest.json` read. Both are fixed in code: scandir offload landed in 0.5.3 (`7f1d0ed`); the manifest read
was removed and the version cached from HA's loaded integration in `11dc122` (on the `0.5.4` branch only).
A full sweep found no other import-time file I/O offenders. **Action:** bump `manifest.json` to 0.5.4, merge
`0.5.4` -> `main`, release; then close. Optional: a regression test asserting no bare `get_cache_buster()`
or module-level `read_text` in the setup path, and confirm `async_unregister_panel` is wired into
`async_unload_entry` to close out the secondary "Overwriting panel" warning.

#### #359 - Panel replaces header, sidebar inaccessible on tablet  [XS]
The panel is a full-page custom panel and renders its own header. It already has a burger that dispatches
HA `hass-toggle-menu`, but it is CSS-gated to `@media (max-width:870px)` only. On a wide tablet with the
mobile-app "Always hide sidebar" setting, the viewport is >870px so the burger never shows and the only
escape (the HA header menu) is what the panel replaced: a trap. **Fix:** show the burger whenever
`this._hass?.dockedSidebar === 'always_hidden'` regardless of width (the panel re-renders on hass change, so
it toggles live). Keep the existing media-query rule for the narrow case. Validate with a boot E2E (stray
backtick blanks the panel).

#### #355 - Review counter says N, list empty  [XS-S]
A cycle can carry both an unanswered `pending_feedback` (`user_response: null`) and `ml_review.reviewed_at`.
The header counter uses `len(get_pending_feedback())` (counts it); the list `needsReview()` short-circuits on
`isReviewed` (hides it). Backend confirms `ws_set_ml_review` never resolves pending feedback, unlike the
label path. **Fix (prefer backend):** have `ws_set_ml_review` resolve pending feedback when the review
confirms the profile, and make the counter use the same predicate as the list. Add a one-shot reconcile for
existing stuck entries. Distinct from #331. Reproduced by 2 users. See Batch C.

#### #348 - Verified-pause release divides by `avg_duration`  [S, fork `3eb06b5`]
`mapped_time / avg_dur > 0.95` at `manager.py:1234`, but `mapped_time` is capped at the envelope span
(`target_duration`, the median-pick member) while `avg_duration` is an outlier-trimmed mean. Force-ended
cycles inflate the mean, dropping the ceiling `span/avg_duration` below 0.95 over time: self-reinforcing,
spreads across the base, ends in a 4 h deferral cap. **Fix:** use the envelope's own span as the
denominator (ceiling 1.0 by construction); keep `avg_duration` in the debug line as the diagnostic. Fork
adds the first-ever tests for this branch. Land with #350.

#### #354 - Tile card ignores Display Precision  [S]
`ha-washdata-card.js:474` prints the raw `.state` (e.g. `16.2833... min`) and unconditionally appends "min".
No `formatEntityState`/`display_precision` anywhere in the card. **Fix:** format via
`this._hass.formatEntityState(stateObj)` (respects unit + display precision + duration device_class), or read
`hass.entities[id].display_precision` and `toFixed`. Reading the entity's own unit also fixes the hardcoded
"min" label. Optional companion: add `suggested_display_precision=0` to the three duration sensors in
`sensor.py`; delete the dead `attr.time_remaining` fallback. All card strings via `_getTranslation`.

#### #362 - Stale review count after merge/split + DE plural  [S]
Merge exists (discoverability gap, reframe the headline). Real bugs: (1) `apply_merge_interactive` and
`apply_split_interactive` never prune consumed cycles' `pending_feedback`, so the badge (raw
`get_pending_feedback` count) disagrees with the list; `prune_orphaned_feedback()` already exists and does
exactly the right thing, it just is not called from the mutation paths. (2) `de.json:714` renders "Zykluss"
because the English `{s}` suffix hack cannot pluralize German. **Fix:** prune in merge/split (mirror
`delete_cycle`), harden the counter to exclude orphans, correct the DE string (manual translation), optional
discoverability affordance. See Batch C.

#### #363 - Misses unchanged `state_reported` reports  [S, same fix as #329]
`manager.py:1772` uses `async_track_state_change_event` only, so a plug re-reporting the same value fires
`EVENT_STATE_REPORTED` (which WashData never subscribes to) and the reading is invisible; a sub-threshold
plateau never advances `_time_below_threshold`, so the cycle never ends. Verified `async_track_state_report_event`
exists in the HA baseline and its event carries `new_state` (no `old_state`), which the existing handler
already tolerates. Entity-filtered, passes the existing throttle, mutually exclusive with `state_changed`
so no double-count. **Fix:** add the second listener into `_async_power_changed`, tear it down on
reload/unload, add a report-shaped-event regression test. See Batch A.

#### #369 - Inconsistent stored UTC offsets  [S hygiene + NEEDS-INFO on display]
Cycles persist mixed `+02:00`/`+00:00` offsets (even mixed within one straddle cycle) because timestamps come
from UTC-aware state readings in some paths and `dt_util.now()` (local-aware) fallbacks in others. All
instants are correct (no naive datetimes exist). The "vor 2 Stunden" symptom looks like a start-vs-end
misread (the list renders start time, which was correct). **Fix:** normalize writes to canonical UTC; optional
v12 migration to canonicalize existing cycles (idempotent, instant-preserving). Reply to clarify the display,
do not block the hygiene fix. See Batch D.

#### #339 - Anti-wrinkle stuck when sensor goes silent  [S-M, PR #340 open]
On a publish-on-change plug, after the last tumble pulse the meter sends one 0 W and goes silent; the
anti-wrinkle idle-timeout and 2 h cap live inside `process_reading` and the watchdog is stopped for the whole
tail, so nothing advances the state: pinned in `anti_wrinkle` for hours. Confirmed by 2 users. PR #340's
reworked approach drives a keepalive from the state-expiry timer (the one timer that keeps ticking), gated on
`_last_real_reading_time` so the idle timer does not advance at half real-time, and exempts real tumble
pulses from the throttle. **Action:** review and accept PR #340; verify keepalive cadence vs the 60 s tick.
See Batch A.

#### #329 - Power sensor lags ~15 min at finish  [M, remove `done`, fork v0.5.3.7/.9]
Same root cause as #363. The watchdog's high-power branch re-injects the stale high reading to keep the cycle
alive, so a device stuck at 8 W / 70 W keeps "running" until a genuinely different value arrives at the max
reporting interval (the reported 15-min lag). Reproduced across Zigbee/MQTT/Tasmota/Shelly by 4 users, with a
tested fork fix. **Fix:** the Batch A `state_reported` subscription (opt-in, default off, to avoid shifting
`p95_dt`-derived thresholds), plus the optional heartbeat fallback. Remove the wrong `done` label.

#### #350 - Envelope alignment drops timestamps  [M, fork `3400029`]
`profile_store.py:5245` drops the trace timestamps, then the aligner warps sample-indexed data against a
time-gridded reference and returns `mapped_time = envelope_time_grid[mapped_idx]`, so position advances one
grid step per sample regardless of wall-clock. Control probe: 12 samples of 1900 W mapped identically to 12
samples of 0 W. **Fix:** pass the trace offsets through and resample onto the envelope's step (same linear
interp the build side uses), bounded ~2x envelope length, before coarse-align + DTW. Must preserve the
opposite guard (a genuinely partly-finished running cycle must still map short). Land with #348. Maintainer
decision: interp vs ZOH. See Batch B.

#### #366 - Trim clock mode collapses cycle to 0s  [M, DATA LOSS - prioritize]
Two compounding defects: (1) `trim_cycle_power_data` overwrites `power_data` in place with no min-window floor
and no backup, so a <=1-sample window yields `duration=0/energy=0` permanently, and re-trims operate on the
already-shrunk data; the WS `_trim_task` lacks the `end<=start` guard the legacy service has. (2) The frontend
`_clockToOffset` interprets typed wall-clock in the browser's local tz, wraps negatives with `+86400` and
clamps to `[0, full]`; combined with #369's offset mismatch it maps a reasonable clock window onto a
near-empty offset range. **Fix:** backend reject/no-op sub-2-sample or sub-MIN trims with a surfaced error,
and/or stash a one-slot pre-trim backup for undo; frontend treat empty input as "no change", disable Apply
below a threshold, derive offsets from the graph handles. Preserve the #330 overnight `+86400` behavior.
Regression test a clock-mode trim on a `+00:00` cycle. See Batch D.

#### #364 - Smart-term splits wash at shorter prefix profile (cf #288)  [L, needs export]
The #288 prefix guard (`_match_prefix_ambiguous`) has three leaks that reproduce both reported Miele cases:
(1) it needs a longer profile to exist in the candidate pool, so an **untrained** longer program is
uncatchable; (2) the guard gates the longer candidate on `shape_score >= 0.40` against its **full** envelope,
but a partial trace of a longer program inherently scores low, so the very candidate it should catch is
skipped; (3) the 1.5 duration ratio sits right at the boundary for the reported pair. **Fix (asymmetric,
shorten-only):** (a) score the longer candidate's **prefix** (trace vs its envelope truncated to
`current_duration`) instead of the full-envelope shape score, and (b) for the untrained case add a
power/envelope-conformance guard in the ENDING block, or a bounded post-termination merge-back if high power
resumes shortly. High regression surface (this path was tuned by #43/#288/#296/#311): must A/B via
`dtw_ab_eval` and the replay suites, and must never re-hang cycles. **Request the case-2 cycle export
(`25ebafe290f7`) first** to reproduce and validate.

### 4.2 IMPLEMENT (feature requests that fit)

#### #346 - Two DEBUG lines for silent Smart-Termination  [XS, fork]
Pure logging, no behavior change, no user-facing strings (so no translation burden). Add a debug line at the
verified-pause release (both the >0.95 release and the <=0.95 stay, plus the `avg_dur==0` skip) and at the
ENDING blocked fast-path (throttled to reason-change via a `_last_smart_term_block_reason` field, suppressed
while `_expected_duration==0`). Keep the release condition byte-identical. Ship with Batch B.

#### #367 - Configurable `power_profile_interval_min`  [XS-S, offered]
`sensor.py:753` hardcodes the 15-min bucket even though `get_profile_power_profile(interval_s=...)` already
accepts the parameter. **Implement:** add `CONF_POWER_PROFILE_INTERVAL_MIN` (minutes, default 15), read it in
the sensor, pass `interval_min*60`, report it in the attribute. Clamp to >=1 min. Additive, no migration,
read-time only (no detection impact). Panel setting + `_t()` keys (manual translations). Unblocks external
planners (EMHASS, tibber).

#### #347 - Live-progress notification sticky + clickAction  [S, offered]
Options on the existing notification, not a new type, so it does not violate the no-new-notification-types
rule. `sticky` is already in `_MOBILE_ONLY_EXTRA_KEYS`; `clickAction` must be added there (else the live
allow-list filters it out). **Implement:** `CONF_NOTIFY_LIVE_STICKY` (bool, default False) and
`CONF_NOTIFY_LIVE_CLICK_ACTION` (str, default ""), applied to both the waiting and progress `extra_vars` in
`_check_live_progress_notification`. Defaults reproduce today's behavior byte-for-byte. Panel settings under
Notifications, `_t()` keys, manual translations. No migration.

#### #343 - Auto-tune re-suggests anti-crease-breaking thresholds  [M]
Threshold suggestions anchor to the lowest active power **including** the ~3 W anti-crease baseline
(`suggestion_engine.py` p05/min_active), so they propose stops just above that baseline (exactly what breaks
end-detection), and there is no rejection memory so dismissed suggestions regenerate every labeled cycle.
Confirmed multi-user pain; over-prompting is an explicit repo anti-pattern. **Implement the durable combo:**
(1) a per-setting **lock** persisted in the store (`locked_settings`), filtered in
`learning._apply_suggestions_and_notify` and the WS apply/clear paths, with a lock icon on each suggestion
card; plus (2) **anti-crease sample exclusion**, gated on `anti_wrinkle_enabled` + eligible device type,
dropping post-main-load samples below `anti_wrinkle_max_power` before deriving stop/start/end_energy (reuse
the detector's own classification to stay consistent). Item 3 (cadence-bound off_delay) is lower value and
riskier: defer. The lock alone already kills the nag and is brand-agnostic. (Register item 40 about
`_SUGGESTION_KEYS` is now resolved and separate.)

#### #342 - Door sensor support for auto-open dishwashers  [M, partial today]
Fits the passive monitor perfectly (interprets an existing optional sensor, no appliance control). Today a
door-open during a cycle sets a sticky user-pause, so an AirDry machine that pops its door at the end gets
stuck in `user_paused` (Clean state is only entered if the door is **closed** at end). Workarounds exist
(clear the door-sensor field, or a `delay_on` template into External End Trigger) but they overrun and cannot
tell a long genuine mid-cycle pause from the end. **Implement:** `CONF_DOOR_OPENS_AT_END` (bool) +
`CONF_DOOR_END_DWELL_SECONDS` (default 60); in `_handle_door_sensor_change`, when enabled and RUNNING/ENDING,
start a dwell timer (mirror the existing `async_call_later` timers) instead of the immediate pause; if the
door stays open past the dwell, finalize via the same terminal path as `user_stop` (trimming the trailing
near-zero tail). Opt-in, dishwasher/washer-scoped. Fix the related trailing-0 W trim-on-force-end gap
alongside. `strings.json`/panel `_t()`, manual translations.

#### #353 - Consume external "programme" value from a smart appliance  [M]
Fits (still passive: reads another entity, feeds the existing manual-program override). ~70% of the mechanism
already exists: `set_manual_program`/`clear_manual_program`, matching is skipped when a manual program is
active, and `current_program` is already a sensor attribute. **Implement:** `CONF_PROGRAM_ENTITY` + a
per-device value->profile mapping table (heterogeneous program strings across integrations make a global
mapping impossible), a state listener (mirror the door/external-trigger listener) that drives
`set_manual_program`/`clear_manual_program` on change with unknown/unavailable fallback to power-based
matching, relaxed running-state/profile-exists guards for this path, and a mapping UI. Keep Stage 1-5
matching as the authoritative fallback so non-smart devices never regress. Surface the source (external vs
power) on the sensor; no new notification type. Consider auto-suggesting mappings from value<->matched-profile
co-occurrence.

#### #334 - Water-level variants indistinguishable (fill role)  [L, `accepted`, needs export]
The reporter's code reading is right but their fix location (add fill weight to `phase_match`) is wrong:
`phase_match` only feeds the opt-in ETA blend, not which program wins. The real disambiguator is Stage-5
`_stage5_pick_member`, which scores members on whole-cycle duration x mean-power x peak (diluting the fill
signal) and, critically, only runs when a **group** wins (the reporter's two variants were ungrouped, so it
never fired). Prior A/B (real data, N=41): a fill role dropped top-1 by 10pp on real EU front-loaders (fill is
noise there) but gained 20pp on controlled top-loader water-level data; a fill-separability gate cleanly
separates the two regimes (real families 0.14-0.68, synthetic 2.27). **Implement, narrowly:** add `ROLE_FILL`
to `phase_segmenter.py` (isolate the leading low-power run, do not merge-fold it), which auto-populates a fill
`RoleStat` in `envelope["phase_profile"]`; add a fill-agreement factor in `_stage5_pick_member` **only when
the group's fill is separable** (inert/byte-identical otherwise). Requires the user to group the variants
first (nudge via `suggest_profile_groups`). Never touch global Stage 1-4. **Request the reporter's real
top-loader export** to confirm the mechanism holds on their hardware before shipping. A/B via `dtw_ab_eval`.

### 4.3 NEEDS-INFO then build

#### #344 - Import historical power data (CSV / recorder)  [L-XL, phased]
Fits the product (offline analysis of power data, no control) and the primitives exist: `_recorder_power`
(recorder reads, executor-offloaded, already used for chart overlay), the Playground `SimRunner` (real
`CycleDetector` replay), and `reference_cycles` (a home for backfilled cycles that feeds envelopes/matcher but
is excluded from lifetime/energy stats). The genuinely new work is: raw-stream segmentation into candidate
cycles (the existing reprocess path only re-matches already-segmented cycles), a review/label UI, and
task-registry-backed background replay. The blocker for the median user is HA's 10-day default full-res
recorder retention (older data is hourly averages and will mis-detect); the CSV path is the broadly useful
one. **Action:** ask the reporter for the exact CSV column schema and confirm target = `reference_cycles`.
Then build phased: CSV ingest + chunked background segment-detection first, review UI second. Guard on
sampling interval (reuse `energy_gap_threshold_s`); enforce CSV size limits; never inflate lifetime stats.

### 4.4 REFUSE

#### #368 - Remote start of washers/dryers  [REFUSE, redirect]
Recommend refusing for a passive power monitor serving 8000+ heterogeneous users:
1. Autonomously energizing a heating appliance unattended on a timer is a safety/liability step-change from
   the existing user-initiated, immediate, symmetric pause/resume switch control.
2. It only works on the subset of appliances that resume-on-power (many require a physical re-press): fragile
   across the base.
3. The finish-by use case is already fully achievable with a stock HA automation plus the appliance's own
   delay-start (the reporter already does this): it belongs in HA's automation layer, not baked into a
   monitor.
4. The "intelligent interrupt to finish exactly on time" phase is not reliably deliverable.

**Constructive redirect:** document the HA automation recipe
(`on_at = target_finish - learned_avg_duration`, using the durations WashData already exposes, then
`switch.turn_on`). If any concession is wanted, keep it read-only: expose a "suggested power-on time to
finish by X" as a sensor attribute/service that returns a timestamp, leaving the switching to the user's
automation. Confirm the maintainer's direction before building even that.

### 4.5 Done / non-actionable

- **#331** core DONE (PR #336, 0.5.3). Optional S residual: on manual relabel of a cycle with no pending
  feedback, also stamp `ml_review.reviewed_at` so the red dot clears (or adjust `needsReview`); add the UI
  hint the reporter requested. Fold into Batch C or split a new issue.
- **#352** "Summer break" maintainer notice, not an issue. Close/ignore.

---

## 5. Comprehensiveness of reports

The backlog is unusually well triaged (community members, especially kdjkdjkdj, have supplied code-level
root causes, exports, and tested fork fixes). Reports that are NOT fully actionable as-is:

- **#344** under-specified: no CSV schema, no answer to the recorder-retention problem, no
  segment->review->persist design. NEEDS-INFO.
- **#364** missing the case-2 cycle export (`25ebafe290f7`) needed to reproduce and A/B any fix.
- **#334** missing the reporter's real top-loader export needed to confirm fill is a clean discriminator on
  their hardware (fill is regime-dependent).
- **#369** headline "wrong relative time" symptom appears misdiagnosed (start-vs-end); ask for a
  clock-next-to-list screenshot if it persists. The storage-hygiene half is actionable now.
- **#366** the exact trigger needs the cycle's stored `start_time`/offset, but the destructive-backend half
  is provable and fixable without it.

Everything else has enough detail to act on directly.

---

## 6. Suggested execution order

1. **Ship 0.5.4** (closes #328; bump manifest, merge, release). Relabel #329 (drop `done`).
2. **Batch A** (silent-sensor: #363 + #329 + accept PR #340 for #339). Highest user impact.
3. **Batch D** (#366 data loss + #369 hygiene).
4. **Quick wins in parallel:** #359, #346, #367, #347, #354.
5. **Batch B** (#348 + #350, with #346 debug lines).
6. **Batch C** (#355 + #362 + #331 residual: one review-queue consistency pass).
7. **Features:** #343, then #342, then #353.
8. **Hard/risky:** #364 (after export), #334 (after export, A/B gated).
9. **#344** after the reporter supplies the CSV schema; phased build.
10. **Reply and refuse #368** with the automation redirect. Close #352.

---

## 7. Implementation progress (0.5.4)

Updated as each issue is handled. Format: `[status] #N - note (commit)`.
Status: DONE (implemented + tested + committed), WIP, DEFERRED (needs-info), REFUSED.

- [DONE] #328 - fix already in code (`11dc122`); reverified complete, added AST regression guard
  `tests/test_issue_328_startup_blocking.py` (proven to catch both blocking-call regressions), bumped
  `manifest.json` 0.5.3 -> 0.5.4 so the fix reaches users. Changelog entry already present. (`047e962`)
- [DONE] #363 + #329 - same root cause. Added `_subscribe_power_sensor()` registering both
  `state_changed` and `state_reported` on the power sensor (always-on: corrective, not risky - throttle
  bounds density, pure on-change plugs get no extra events). New `tests/test_issue_363_state_reported.py`.
  Fast suite 1364 green. Register item 62. NOTE for maintainer: remove the wrong `done` label from #329.
  Heartbeat/synthetic-injection fallback intentionally NOT added (higher risk, marginal benefit once the
  plug's real reports are consumed); left as a possible follow-up.
