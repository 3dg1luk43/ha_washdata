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
progress estimation belong in `run_tests.sh`, where time can be frozen and 606
recorded cycles can be replayed in 30 seconds. What belongs here is everything
that crosses the boundary into Home Assistant.

## Quick start

```bash
cd devtools/testbox
./up.sh --fresh        # ~40 s: container, config, owner user, long-lived token
./smoke.sh             # ~5 min: one full cycle end to end, then the assertions
./down.sh              # stop (add --wipe to delete config/ too)
```

`./smoke.sh` is the whole point in one command: it creates a device through the
real config flow, seeds it from a real export, replays a recorded 76 min washing
machine cycle at 60x, waits for the cycle to close, and then runs 15 checks that
no mocked Home Assistant can make (see *What smoke.sh proves* below).

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

Two consequences worth knowing:

- Absolute minute counts from a box run mean nothing. Behaviour does: a cycle
  starts, matches, progresses, ends once, and is stored once.
- `notify_live_interval_seconds` is floored at 30 in the manager, so at 60x one
  live update covers ~30 min of appliance time.

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

`./smoke.sh` reports **20 of 20**, and `./check_notify_actions.sh` passes. The whole notification lifecycle is proven
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
  `devtools/dtw_ab_eval.py`.
- **One appliance, one trace per run.** Matching accuracy over the corpus stays
  in the benchmark suite.
- **The companion app is not here.** The box proves Home Assistant accepted and
  delivered the payload; whether iOS then renders a Live Activity from it is
  still only verifiable on a phone.
- `mobile_app:` is loaded so the `mobile_app_notification_action` event exists,
  but no device is registered, so mobile action round-trips are not covered.
