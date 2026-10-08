# The test box - a real Home Assistant to run WashData in

A disposable Home Assistant container, on its own network, with the working tree
bind-mounted in. It exists because of a specific and repeated failure: the unit
suite builds its Home Assistant with `MagicMock()`, a MagicMock says yes to
everything, and so **code that Home Assistant rejects outright can pass every
test**. That is not theoretical. `title: None` made every `clear_notification`
WashData sent fail validation on the service bus for months, while the ten tests
covering that feature passed, because they asserted the shape of the payload they
had just built rather than whether HA would take it.

The box is the other end of that: nothing here is stubbed. Real config flow, real
entity registry, real storage migrations, real WebSocket API, real notify
platform schemas, real service bus. **On its first end-to-end run it found a real
bug** (importing a configuration silently rebound the device to the exporter's
power sensor - register item 317).

It is a complement, not a replacement. Detection maths, matching accuracy and
progress estimation belong in `run_tests.sh` (time frozen; `--slow` replays the
recorded `cycle_data/` corpus) and the `devtools/*_eval.py` harnesses. What
belongs here is everything that crosses the boundary into Home Assistant.

**The Home Assistant version floats.** `docker-compose.yml` pins
`home-assistant:stable`, so the box runs whatever stable was last pulled (2026.9.3
when this was written), not a fixed release. Read it with
`./hactl.py ws get_config | grep '"version"'` and quote it when citing a box run.

## Quick start

```bash
cd devtools/testbox
./up.sh --fresh        # ~40 s: container, config, owner user, long-lived token
./smoke.sh             # ~12 min: one full cycle end to end, then the assertions
./down.sh              # stop (add --wipe to delete config/ too)
```

`./smoke.sh` is the whole point in one command: it creates a device through the
real config flow, seeds it from a real export, replays a recorded 76 min washing
machine cycle at 60x, waits for the cycle to close, and then runs the 22 checks
in `assert_run.py` that no mocked Home Assistant can make (see *What smoke.sh
proves* below). Most of the ~12 min is the wait for the cycle to close (see
*Current known state*).

**The default seed data is private.** Without `--export`, `smoke.sh` seeds from
the maintainer's own exports under the gitignored `cycle_data/me/`
(`washdata_export_01KXGA3C.json` for a washing machine, `washdata_export_01KDMTAA.json`
for `--type dishwasher`), so on any other checkout it stops at
`export not found`. Pass your own WashData export instead:
`./smoke.sh --fresh --export /path/to/washdata_export.json [--cycle N]`. It needs
at least one stored cycle with its power trace; `--cycle` picks which one is
replayed (default 0), and its profiles seed the matcher.

## Paths

Everything lives under `devtools/testbox/`:

| Path | What it is |
|---|---|
| `docker-compose.yml` | the container: image, port, mounts, healthcheck |
| `up.sh` | start + onboard. `--fresh` wipes `config/` first |
| `down.sh` | stop. `--wipe` deletes `config/` |
| `smoke.sh` | the end-to-end run: setup -> import -> replay -> assert |
| `hactl.py` | the driver: REST, WebSocket, state pushing, replay, log reading |
| `assert_run.py` | the acceptance checks, run by `smoke.sh` step 7 |
| `check_notify_actions.sh` | the notification-ACTION delivery path, ~1 min, standalone |
| `check_unload_confirm.sh` | unload confirmation without a door sensor (#451): the button entity, the service and the confirmation-entity listener, ~3 min, standalone |
| `support/configuration.yaml` | baseline HA config, copied into `config/` on first start |
| `config/.compressed_export.json` | the time-rescaled export `smoke.sh` seeds from |
| `support/custom_components/testbox_notify/` | notify platform that records every payload |
| `config/` | **generated, gitignored.** The HA config dir: `.storage`, logs, DB |
| `config/.testbox_token` | long-lived access token, written by `up.sh` |
| `config/notify_capture.jsonl` | one JSON object per delivered notification |
| `config/home-assistant.log` | the box's log, at DEBUG for `ha_washdata` |

Mounted in from the repo, read-only:

| Container path | Host path |
|---|---|
| `/config/custom_components/ha_washdata` | `custom_components/ha_washdata` (the working tree) |
| `/config/custom_components/testbox_notify` | `devtools/testbox/support/custom_components/testbox_notify` |

Because the integration is bind-mounted, **the box always runs the code you are
editing** - there is no copy step and nothing to forget. Python is loaded once at
startup, so after editing the integration:

```bash
./hactl.py restart     # restart + wait for HA *and* for WashData to finish setup
```

- UI: <http://127.0.0.1:8321> - user `testbox`, password `testbox-pw-0123`
- Bound to 127.0.0.1 only, on its own compose network: the box cannot see the
  real Home Assistant, the real MQTT broker or any real appliance.

## Driving it by hand

`hactl.py` needs the repo venv (`aiohttp`); `$REPO/.venv/bin/python3 hactl.py`,
or just `./hactl.py` if the venv is active.

```bash
./hactl.py states sensor.test_dishwasher      # every WashData entity + value
./hactl.py state sensor.test_dishwasher_state
./hactl.py set sensor.washdata_power 1200     # push a power reading
./hactl.py entry-id                           # the config entry id
./hactl.py ws ha_washdata/get_profiles entry_id=<id>
./hactl.py ws ha_washdata/get_options entry_id=<id>
./hactl.py set-options <id> off_delay=10 notify_live_chronometer=true
./hactl.py compress-export <export.json> <out.json> --speedup 60
./hactl.py import <id> <out.json>
./hactl.py replay <export.json> --cycle 3 --speedup 120 --tail 45
./hactl.py notifications                      # what actually reached a platform
./hactl.py errors                             # ERROR/WARNING/MultipleInvalid lines
```

`key=value` arguments are JSON-parsed when possible, so
`notify_live_services='["notify.mobile_app_testbox"]'` and `off_delay=10` behave
as written, and anything unparseable stays a string. Every `ha_washdata/*`
WebSocket command is reachable through `ws`, which makes this the fastest way to
exercise the panel's backend without the panel.

### Notification capture

`support/custom_components/testbox_notify/` is a real legacy notify platform.
`configuration.yaml` registers it twice, and the names are deliberate:

- **`notify.mobile_app_testbox`** - WashData's `_is_mobile_notify_service`
  matches the `mobile_app_` prefix, and *only* such targets get live updates,
  tags, chronometers and the other mobile-only keys. A differently named target
  would skip exactly the paths most worth testing.
- **`notify.plain_testbox`** - a non-mobile target, so "live updates must not go
  to non-mobile services" is observable instead of assumed.

Each delivered notification is appended to `config/notify_capture.jsonl` with its
full `data` dict, so the tag, `live_update`, `progress`, `chronometer` and the
`clear_notification` markers can all be asserted after the fact. A payload that
HA rejects never gets there - which is the entire point.

### The action path

WashData delivers a notification two ways: through the notify services, and
through a user-supplied *action* - a YAML script run with the notification's
variables bound. Every unit module that touches the second one mocks
`script_helper.Script`, so nothing proved an action delivers anything;
`./check_notify_actions.sh` does, in about a minute, by configuring a real action
with **no** notify services so anything that arrives can only have come through
it. That check is what found item 323: the templates were being delivered as
literal `{{ device }}` text, because the sequence never went through
`cv.SCRIPT_SCHEMA`.

### Time compression

A 76 min cycle replayed at 60x takes 76 s, which is the only reason a full run
fits in minutes. Three things have to move together, and getting any of them
wrong looks like a bug in the integration:

1. **The timing options.** The detector's gates are wall-clock, so `smoke.sh`
   divides `off_delay`, `min_off_gap`, `sampling_interval` and
   `watchdog_interval` by roughly the same factor. Everything else stays at the
   shipped default - the point is to exercise production behaviour, not a
   bespoke configuration.
2. **The seeded history.** `hactl.py compress-export` rescales the export's
   clock by the same factor (cycle durations and sample offsets, the profiles'
   duration statistics, the envelopes' time grids; shapes untouched). Without
   it the matcher compares a 76 s replay against a 1720 s learned profile, reads
   the cycle as 7% complete, and correctly refuses to end it -
   `Smart Termination not applied (duration_not_reached)`.
3. **The appliance type.** Dishwashers carry a hardcoded 30 min floor
   (`DISHWASHER_MIN_CYCLE_DURATION_S`, not an option), so a compressed
   dishwasher cycle can never reach its end gate: the detector logs
   `Deferring dishwasher cycle end: elapsed 388s < minimum 1800s` forever. The
   default is therefore a washing machine; `--type dishwasher` needs
   `--speedup 4` or lower and takes ~35 min.

   **Raising the speedup is self-defeating, not merely ineffective**, and this
   is the real ceiling (register item 357). Item 331 made that floor drop to the
   matched profile's own length, so in principle a compressed dishwasher should
   pass it - but only *while it is matched*. Measured at 60x: the profiles
   compress correctly to ~137 s, the cycle replays in ~144 s, and it stays
   matchable until Stage 1's `max_duration_ratio` (1.8) rejects it at ~247 s.
   The floor then holds the cycle open to 1800 s, which pushes elapsed to 7.3x
   the profile, which is what killed the match in the first place. Deadlock:
   `No profile match candidates ... duration ratio filter`, then
   `elapsed 1003s < minimum 1800s`, forever. A 60x run takes 16 min and still
   fails 2 of 19 checks (no cycle recorded).

   So the floor is only relative for a cycle that stays matched, and a
   compressed one cannot. Making it relative in the UNMATCHED case is not
   possible - there is no expected duration to be relative to - and making it an
   option would add a tunable that exists purely for tests. Use `--speedup 4`.
   This is a compressed-replay artifact only: a real dishwasher runs past 30 min
   anyway, so nothing here affects production.

Two consequences worth knowing:

- Absolute minute counts from a box run mean nothing. Behaviour does: a cycle
  starts, matches, progresses, ends once, and is stored once.
- `notify_live_interval_seconds` is floored at 30 in the manager, so at 60x one
  live update covers ~30 min of appliance time.

Three more wall-clock quantities do not scale and change behaviour, so a box run
cannot judge them (measured in the 0.5.7 review campaign, register item 389).
None is a bug; each has a way to test it that does not compress time:

- **The start-energy gate** is in Wh, so at 60x it takes 60x more appliance time
  to fill. A cycle with a low-power prelude is recorded starting ~8 min late
  (start offset +9 min against +1 min typical); its end is unaffected.
  *Test it* by replay: `devtools/start_gate_eval.py` feeds a raw history (a
  diagnostics dump's 24 h `power_trace`, a History CSV download, or a recorder
  database) through the real detector in real time and reports missed, late and
  phantom starts per gate value. In the box, `./hactl.py set-options <id>
  start_energy_threshold=<default / speedup>` removes the artefact, since the
  energy a compressed prelude delivers shrinks by the same factor.
- **`STANDBY_BAND_WINDOW_S`** (600 s) is a constant, not an option, so it cannot
  be scaled: at 60x it is 10 h of appliance time and the standby-band finalize can
  never fire. *Test it* at 2x (20 min of appliance time): `./smoke.sh` writes
  60x timings in step 3, so after it run `./hactl.py set-options <id>
  off_delay=<v> min_off_gap=<v> sampling_interval=<v> watchdog_interval=<v>
  profile_match_interval=<v> completion_min_seconds=<v> interrupted_min_seconds=<v>`
  with each `<v>` the shipped value halved, then `./hactl.py replay <export> --cycle N --speedup 2` on a cycle whose
  tail holds a flat standby above `stop_threshold_w`
  (`tests/test_issue_445_standby_above_stop.py` has the shape). By replay, a
  stored trace that ends in such a standby goes through the finalize in real time
  in `devtools/end_gate_eval.py --loo --all-formats`.
- **Back-to-back washes merge.** A 7 min gap between two washes becomes 7 s,
  shorter than the 10 s `min_off_gap` step 3 writes, so the box keeps as one cycle
  what a real-time replay splits in two. *Test it* by replay:
  `devtools/min_off_gap_eval.py` replays every trace at each candidate
  `min_off_gap` and runs a merge probe (one cycle replayed twice, the user's
  shortest real gap between loads apart), and
  `devtools/start_gate_eval.py` reports `merged` against the stored records of a
  raw history. In the box, 2x with the halved options above keeps every gap and
  every option in proportion.

After the trace ends the replay pushes 0 W once and stops. Home Assistant drops
unchanged states, so repeated 0 W produces no further events - which is exactly
the report-on-change plug behaviour behind #424/#427, and means the **watchdog is
the only thing left driving the detector**. That is deliberate: the box closes
its cycles through the same path a silent plug does.

## What smoke.sh proves

Each of these is unreachable from a mocked Home Assistant:

1. The config flow creates a working entry, and platforms create their entities.
2. A real export imports through the real storage migration, and profiles appear.
3. `ws_set_options` persists, reloads the entry and takes effect.
4. A replayed cycle is detected, matched and stored exactly once.
5. Notifications are *delivered*: start, live updates, pre-complete, finish.
6. Live updates reach mobile targets only, on their own dedicated tag.
7. The dismissals arrive - the lifecycle hand-over and the live-activity end.
   This is the #446 failure mode, and it is the check the unit suite cannot make.
8. No delivered notification carries a null title.
9. Nothing was rejected by a schema, no `ha_washdata` ERROR, no traceback.

`assert_run.py` exits non-zero on any failure, so it can gate a release.

## Current known state

**`smoke.sh` does not restart a box that is already up, so it can assert against
stale code.** It only calls `up.sh` when the container is unreachable; Python is
loaded once at startup, so a box left running from before your edit runs the OLD
module while the bind mount shows the new source. The failure reads as a code
bug: a newly added payload key is simply absent, and `docker exec ... grep` finds
it in the file. Run `./hactl.py restart` (or `./smoke.sh --fresh`) after editing
the integration and before trusting a run. Cost of learning this: one full
12-minute run and a real feature reported as broken when it worked (#454).

**Run `./smoke.sh --fresh` between comparison runs.** `setup-device` reuses an
entry by name and `--type X` names the device `Test X`, so a washing-machine run
followed by a dishwasher run leaves **both** devices in the box, driven from the
same power sensor. That cross-talk surfaces as unrelated assertion failures -
two devices' live notifications land on two tags, so "live updates share one
dedicated tag" fails for reasons that have nothing to do with the code.

`./smoke.sh` reports **22 of 22** on a clean box at 60x for a washing machine.
(It used to report 19 of 20: the seeded export's own `notify.mobile_app_s24`
target does not exist in the box and raised `ServiceNotFound`. Step 3 now points
every notify option at the box's own `testbox` targets.) `--type dishwasher`
previously seeded the **washing-machine** export regardless of type - so the
dishwasher path was being exercised against washing-machine profiles and the
run matched programmes like "30 deg / 2:09 / 800rpm". The export now follows
`--type`, and a dishwasher run takes ~30 min at any speedup for the reasons in
*Time compression* above. `./check_notify_actions.sh` passes. The whole notification lifecycle is proven
against a real service bus: start, a live update on its own tag, mobile-only
routing, the lifecycle hand-over dismissal, the live-activity end, the finished
alert delivered *before* that end, titles on every content notification, and a
log with no rejected call, no `ha_washdata` ERROR and no traceback.

One number is unaccounted for: the cycle closes about **7.5 min of wall clock**
after the trace ends, while the configured gates are 10 s. At 60x that is 7.5 h
of appliance time, so it is not obviously wrong, and the detector also detours
`running -> paused -> ending` on the way. It has not been traced to a specific
gate. If a real-time "cycle ends late" report ever needs reproducing, start
there. Register item 320.

## Limits

- **Timing is compressed**, so it cannot judge cycle-end accuracy or ETA
  convergence in minutes. Those live in `run_tests.sh --slow` and
  `devtools/end_gate_eval.py --loo`.
- **One appliance, one trace per run.** Matching accuracy over the corpus stays
  in `devtools/eval.py`, which drives the shipped matcher.
- **The companion app is not here.** The box proves Home Assistant accepted and
  delivered the payload; whether iOS then renders a Live Activity from it is
  still only verifiable on a phone.
- `mobile_app:` is loaded so the `mobile_app_notification_action` event exists,
  but no device is registered, so mobile action round-trips are not covered.
