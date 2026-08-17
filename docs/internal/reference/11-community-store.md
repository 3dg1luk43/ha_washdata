# WashData Community Store — Technical Reference

**Files covered:** `store.py` (471 lines), `store_client.py` (858 lines), `store_account.py` (187 lines), plus cross-references in `ws_api.py`, `profile_store.py`, `const.py`, and `setup_advisor.py`. Documentation from `docs/STORE.md` and the four design specs in `docs/superpowers/specs/`.

> Content-corrected 2026-08-17 for 0.5.4 (running_dead_zone no longer shareable; allow-list is 20 keys). Line anchors may still reflect the original 2026-07-18 / 0.5.1 pass.

---

## 1. Architecture Overview

The store subsystem is split across three modules with strict separation of concerns:

| Module | Role |
|---|---|
| `store_account.py` | Integration-wide (device-agnostic) persistent state: online toggle, GitHub account credentials, user preferences. Lives in a dedicated HA `Store` file (`ha_washdata_online`). |
| `store_client.py` | Async Firestore REST client. Public reads (no auth), authed writes via Firebase ID token exchange. No new dependency -- uses HA's shared aiohttp session. |
| `store.py` | `StoreBridge` orchestration class: gating checks, provenance derivation, LTTB downsampling, import/share/adopt flow against `ProfileStore`. No network code of its own. |

The online toggle, GitHub account, and preferences are **integration-wide** (one per HA install, stored in `hass.data[f"{DOMAIN}_online_cfg"]`). Brand/model declarations are **per-device** (`entry.options[CONF_STORE_BRAND]` / `CONF_STORE_MODEL`). The `StoreBridge` instance is lazily created per `WashDataManager` (`manager.store_bridge`, `manager.py:355-359`).

---

## 2. Firestore Data Model

Flat collections; parent-ID references; `*_lc` lowercase copies for case-insensitive matching. Schema v2.

### `brands/{brand_lc}`
Fields: `brand`, `brand_lc` (doc id), `status` (`approved|pending|removed`), `createdByUid`, `createdByName` (nullable, consent-gated), `createdAt`.

### `devices/{deviceId}`
`deviceId` = `{applianceType}__{brand_lc}__{model_lc}` (normalized; see `store_client.device_id()`). Fields: `applianceType`, `brand`, `brand_lc`, `model`, `model_lc`, `status`, `createdByUid`, `createdByName`, `createdAt`, `favoriteCount`, `confirmCount`, `manualUrl` (nullable, https only), `settings` (optional, shareable-settings allow-list dict).

### `profiles/{profileId}`
`profileId` = `{deviceId}__{program_lc}`. Fields: `deviceId`, `applianceType` (denorm), `program`, `program_lc`, `description`, `status`, `createdByUid`, `createdAt`, plus Stage 2 additions: `phases: [{name, start, end}]`, `phaseSourceCycleId`, `phasesSchemaVersion`.

### `cycles/{cycleId}` (content-hash ID)
Fields: `profileId`, `deviceId`, `brand_lc`, `program_lc`, `applianceType` (denorm), `uploaderUid`, `uploaderName`, `status`, `rejectionReason`, `trace: {points: [{o,w}], sampleIntervalSec}`, `stats: {duration, peak_w, mean_w, signature[, energy_wh]}`, `cycleSchemaVersion: 1`, `createdAt`, `downloads`, `commentCount`, `confirmCount`, `qc`. Subcollections: `ratings/{uid}`, `comments`.

### `config/site`
Global admin-tunable config. Key field: `confirmThreshold` (default 5, world-readable). Also carries `maintenance` flag.

### `analytics/totals` + `analytics/daily_{YYYYMMDD}`
Unauthenticated increment-only counters (`downloads` field). Written by `bump_analytics()` on every download/adopt action.

---

## 3. Community Catalog Auto-Promote Mechanism

A device entry starts `status="pending"` and is publicly visible. Any signed-in GitHub user may confirm it once. The batched write creates `devices/{id}/confirmations/{uid}` (create-only, uid-keyed) **and** increments `confirmCount` in the same transaction (`store_client.py:746-788`). The `existsAfter/!exists` Firestore rule makes inflation impossible: a second confirmation attempt from the same uid is rejected server-side.

After each confirmation, `confirm_device` reads the current `confirmCount` and compares it to `config/site.confirmThreshold` (fetched live, falls back to 5). When the threshold is met, a second batched write flips `status` to `approved`. This is **best-effort client-side**: the Firestore rule is the real security boundary (only allows `pending→approved` when `confirmCount >= threshold`).

**Threshold:** 5 confirmations, stored in `config/site.confirmThreshold`, admin-tunable with no rules redeploy. The integration reads it via `StoreClient.get_config()` (`store_client.py:317-326`).

---

## 4. Deterministic IDs and Normalization

All IDs must match byte-for-byte between the integration and the store's `lib/ids.js`.

### `normalize_token(s)` — `store_client.py:64-68`
```
lowercase -> NFKD normalization -> collapse non-alphanumeric runs to '-' -> trim '-'
```

### `device_id(appliance_type, brand, model)` — `store_client.py:71-72`
`normalize_token(appliance_type) + "__" + normalize_token(brand) + "__" + normalize_token(model)`

### `profile_id(dev_id, program)` — `store_client.py:75-76`
`dev_id + "__" + normalize_token(program)`

### `brand_id(brand)` — `store_client.py:79-80`
`brand.lower()` (no NFKD normalization — note subtle difference from `normalize_token`).

### `trace_hash(profile_id, pts)` — `store_client.py:149-161`
Deterministic SHA-256 content hash scoped to the profile ID. Offsets rounded to whole seconds; watts to 1 decimal. Makes share idempotent: re-uploading an identical trace collides on the same document ID and the `currentDocument: {exists: false}` precondition refuses the create silently.

### Appliance type mapping
`store.py:46` defines `_STORE_APPLIANCE_TYPE = {"washing_machine": "washer"}`. HA's `washing_machine` maps to `washer` in the catalog. All other types (`dryer`, `dishwasher`, `washer_dryer`) pass through unchanged. The `store_appliance_type(device_type)` helper (`store.py:49-50`) is used by every upload/search path.

---

## 5. `StoreClient` — Public Methods

All methods live in `store_client.StoreClient`. Unless noted, failures return `None`/empty and are logged; no method raises into the event loop.

### Auth
- **`ensure_id_token(refresh_token: str) -> str | None`** (`store_client.py:197-230`): Exchanges the stored refresh token for a Firebase ID token via `https://securetoken.googleapis.com/v1/token`. Caches the token for ~1h, keyed to the specific refresh token so a reconnect doesn't serve a stale token. Returns `None` on failure and sets `self._last_error`.

### Public reads (no auth)

**Read cache (0.5.3):** `list_brands`, `search_devices`, and `get_config` cache their responses in an in-memory `_read_cache` (`key -> (expiry, value)`) — catalog reads for `_CATALOG_CACHE_TTL_S` (900s), config for `_CONFIG_CACHE_TTL_S` (3600s). The catalog is public and slow-changing, yet the panel re-queries it on every Settings/Store tab open; on the store's free-tier read budget the brand-list query was the single largest read source. The `StoreClient` is one long-lived instance per manager, so the cache survives panel reloads. Cached rows are the *unfiltered* query result — the `q`/`model_query` prefix filter is applied in memory after a cache hit, so one entry serves every prefix. Only successful reads are cached (a failed `get_config` returns `{}` uncached). A successful brand/device create in `_commit_create` calls `_invalidate_catalog_cache()` (drops `brands:`/`devices:` keys) so a freshly-contributed entry appears immediately. Tested in `tests/test_store_client.py` (`test_list_brands_caches_across_calls`, `test_search_devices_caches_per_key`, `test_get_config_caches_and_never_caches_failure`, `test_catalog_cache_invalidated_on_create`, `test_read_cache_get_put_and_expiry`).

- **`list_brands(q, include_pending, page_size) -> list[dict]`** (`store_client.py`): Firestore query on `brands` collection, ordered by `brand_lc` ascending. Client-side prefix filter when `q` is provided. Default `page_size=60`. Cached (see above).
- **`search_devices(brand, appliance_type, model_query, include_pending, page_size) -> list[dict]`** (`store_client.py`): Ordered by `favoriteCount DESC`. Client-side `model_lc.startswith(model_query)` filter. Default `page_size=60`, `include_pending=False`. Cached per `(brand, appliance_type, include_pending, page_size)` key.
- **`get_device(device_id) -> dict | None`** (`store_client.py`): Direct GET on `devices/{device_id}`. Not cached (single-doc, user-gated).
- **`get_config() -> dict`** (`store_client.py`): GET on `config/site`. Returns `{}` on failure. Cached for 1h on success.
- **`get_profiles(dev_id, include_pending, page_size) -> list[dict]`** (`store_client.py:369-379`): Ordered by `createdAt DESC`. Default `page_size=100`, `include_pending=False`.
- **`device_profiles(brand, model, appliance_type) -> dict`** (`store_client.py:381-386`): Resolves `device_id` from brand/model/type and returns `{device_id, items}` with `include_pending=True`. Used by Share dialog picker.
- **`get_cycles(prof_id, include_pending, page_size) -> list[dict]`** (`store_client.py:417-451`): Ordered by `createdAt DESC`. Default `page_size=50`, `include_pending=True`. Attaches rating summary to each cycle. Rate-limited via `asyncio.Semaphore(_RATING_FANOUT_LIMIT=8)`.
- **`get_cycle(cycle_id) -> dict | None`** (`store_client.py:453-465`): Direct GET on `cycles/{cycle_id}`. Calls `_with_decoded_trace` to attach `importable` list.
- **`get_device_bundle(dev_id, include_pending) -> dict`** (`store_client.py:388-415`): Full device package: device doc (including `settings`) + profiles list, each hydrated with their cycles. Bounded by `asyncio.Semaphore(_BUNDLE_HYDRATE_LIMIT=4)` to cap concurrent profile-cycle fan-out.
- **`get_device_quality(device_id) -> dict`** (`store_client.py:361-363`): Aggregation query (`count` + `avg`) over `devices/{id}/ratings`. Returns `{avg: float|None, count: int}`.
- **`cycle_rating(cycle_id) -> dict`** (`store_client.py:365-367`): Same aggregation over `cycles/{id}/ratings`.

### Authed writes
- **`upload_reference_cycle(refresh_token, uid, uploader_name, meta, points, stats, qc, return_status=False) -> str | None | dict`** (`store_client.py:533-672`): Creates brand/device/profile docs (create-if-missing, idempotent) then creates the reference cycle. Cycle ID is `trace_hash(profile_id, pts)`. All fields validated before any write. With `return_status=True` returns `{id, created}` to distinguish new vs already-existing.
- **`upload_device_bundle(refresh_token, uid, uploader_name, device_meta, items) -> dict`** (`store_client.py:674-725`): Calls `upload_reference_cycle` per item with shared device metadata. Returns `{ok, cycle_ids, created, duplicates, errors}`.
- **`confirm_device(refresh_token, uid, device_id) -> dict | None`** (`store_client.py:746-788`): Batched write: create `confirmations/{uid}` + increment `confirmCount`. Best-effort auto-promote if threshold reached. Returns `{confirmed, confirmCount, status}` or `None` on failure.
- **`rate_device(refresh_token, uid, device_id, rating) -> bool`** (`store_client.py:790-805`): Upserts `devices/{id}/ratings/{uid}`. Rating must be 1-5.
- **`bump_downloads(cycle_ids) -> None`** (`store_client.py:807-827`): Unauthenticated `+1` to `downloads` on each listed cycle. Chunked at 400 per commit (Firestore 500-write limit). Never raises.
- **`bump_analytics(field, n) -> None`** (`store_client.py:830-858`): Unauthenticated `+n` to both `analytics/totals.{field}` and `analytics/daily_{YYYYMMDD}.{field}`.

### Internal helpers
- **`_run_query(sq, parent) -> list[dict]`**: Posts a Firestore `runQuery` structured query.
- **`_commit_create(id_token, path, fields, server_ts_field) -> bool`**: Create-if-missing commit with server timestamp. 409/ALREADY_EXISTS treated as success.
- **`_commit_create_ex(...)` -> `tuple[bool, bool]`**: Same but returns `(ok, created)`.
- **`_commit(id_token, writes) -> tuple[bool, str]`**: Raw batched commit.
- **`_with_decoded_trace(cycle) -> dict`**: Attaches `importable = [[offset, watts], ...]` when `cycleSchemaVersion` is in `SUPPORTED_CYCLE_SCHEMA_VERSIONS = {1}`.
- **`_rating_agg(parent_path) -> dict`**: Firestore aggregation query (count + avg).
- **`last_error() -> str | None`**: Short reason for the last failed write, used by `StoreBridge` to surface UI error messages.

### Firestore REST encoding
- **`pack_points(pairs) -> list[dict]`**: `[[o,w]] -> [{o,w}]` (Firestore cannot store nested arrays).
- **`unpack_points(points) -> list[list[float]]`**: Inverse.
- **`_encode(v) -> dict`** / **`_decode(v) -> Any`**: Firestore typed-value encode/decode for the REST API.

---

## 6. `StoreBridge` — Public Methods

All live at `store.StoreBridge`. All are no-op-safe (return `{"error": ...}` instead of raising). Callers must gate on `online_features_enabled()` first; the WS layer handles this.

### Account / status (global)
- **`status() -> dict`** (`store.py:165-166`): Returns `{enabled, connected, uid, name}`. Reads from `store_account`.
- **`connect(refresh_token, uid, name) -> dict`** (`store.py:168-173`): Validates the refresh token by doing one exchange, then persists via `store_account.async_set_account`. Returns `{connected, uid, name}` or `{"error": "token_invalid"}`.
- **`disconnect() -> dict`** (`store.py:175-177`): Clears the stored account via `store_account.async_clear_account`. Returns `{connected: False}`.

### Catalog browse (reads, no auth required)
- **`list_brands(query, include_pending) -> list[dict]`** (`store.py:181-182`): Passthrough to `StoreClient.list_brands`.
- **`search_devices(brand, appliance_type, model_query, include_pending) -> list[dict]`** (`store.py:184-190`): Passthrough to `StoreClient.search_devices`.
- **`get_profiles(device_id) -> list[dict]`** (`store.py:192-193`): Passthrough.
- **`device_profiles(brand, model, appliance_type) -> dict`** (`store.py:195-198`): Maps HA appliance type first, then calls `StoreClient.device_profiles`.
- **`get_cycles(profile_id) -> list[dict]`** (`store.py:200-201`): Passthrough.
- **`get_device_quality(device_id) -> dict`** (`store.py:203-204`): Passthrough.

### Community actions (authed writes)
- **`confirm_device(device_id) -> dict`** (`store.py:208-213`): Requires connected account. Calls `StoreClient.confirm_device`.
- **`rate_device(device_id, rating) -> dict`** (`store.py:215-220`): Requires connected account.

### Import / share
- **`import_cycle(cycle_id, target_profile, new_profile_name) -> dict`** (`store.py:224-251`): Downloads a store cycle by ID, validates, calls `ProfileStore.add_reference_cycle`. Bumps `downloads` counter and `analytics/totals.downloads`. Returns `{profile, cycle_id}` or `{"error": ...}`.
- **`share_cycle(local_cycle_id, program, brand, model, appliance_type, sample_interval_sec, description) -> dict`** (`store.py:254-288`): Looks up cycle in both `past_cycles` and `reference_cycles`. Computes stats via `_cycle_upload_stats`. Runs LTTB downsampling in executor. Calls `StoreClient.upload_reference_cycle`. Returns `{store_cycle_id}` or `{"error": ...}`.
- **`share_device(brand, model, appliance_type, items, include_phases, settings) -> dict`** (`store.py:290-363`): Full device bundle upload. Resolves each `local_cycle_id` to trace + stats. Offloads per-cycle LTTB in executor. Attaches phase ranges (in absolute seconds) to items whose program is in `include_phases`. Attaches `settings` dict to `device_meta`. Calls `StoreClient.upload_device_bundle`. Returns `{ok, cycle_ids, created, duplicates, errors}`.
- **`download_device(device_id_, device_type) -> dict`** (`store.py:365-431`): Downloads full device bundle. Idempotent: skips any cycle whose `meta.source == "store:<id>"` is already in local `reference_cycles`. Calls `ProfileStore.add_reference_cycle` for each new cycle. Applies phase ranges via `_apply_phases`. Bumps `downloads` and analytics only for newly-imported cycles. Returns `{profiles_adopted, cycles_imported, phases_applied, settings}`.

### Internal
- **`_apply_phases(program, phases, device_type) -> bool`** (`store.py:433-471`): Validates phase list, calls `ProfileStore.async_set_profile_phase_ranges`. Reconciles unknown labels into `custom_phases` via `async_create_custom_phase`. Never raises; returns `True` when a non-empty map was applied.

### Free functions in `store.py`
- **`online_features_enabled(hass) -> bool`** (`store.py:38-40`): Global gate check, delegates to `store_account.online_enabled`.
- **`store_appliance_type(device_type) -> str`** (`store.py:49-50`): Maps `washing_machine` to `washer`.
- **`derive_qc(cycle) -> int`** (`store.py:53-69`): Returns `QC_RECORDING` (1), `QC_EDITED` (2), or `QC_MANUAL` (3) based on `meta.source == "recorder"`, `"original_samples" in meta`, or `meta.edited`. Recording takes precedence over edited.
- **`_downsample(points, max_n=10000) -> list`** (`store.py:72-122`): LTTB (Largest Triangle Three Buckets) O(N) downsampler. Preserves peaks and troughs over nearest-index subsampling.
- **`_cycle_upload_stats(cyc, pts) -> dict`** (`store.py:125-147`): Computes `{duration, peak_w, mean_w, signature}`, optionally `energy_wh` when `> 0`. Shared by `share_cycle` and `share_device`.

---

## 7. `store_account.py` — Integration-wide Persistent State

Stored in `hass.data[f"{DOMAIN}_online_cfg"]` (loaded once at setup, serialized to `ha_washdata_online` HA Store file, schema version 1). Load is serialized under a double-checked lock to handle concurrent config entry setups.

Default state:
```python
{"online_enabled": False, "account": {}, "migrated": False, "prefs": {"show_contributor": True}}
```

### Public functions
- **`async_load(hass) -> None`** (`store_account.py:61-92`): Load-once from disk. Double-checked lock.
- **`online_enabled(hass) -> bool`** (`store_account.py:106-108`): Global online features gate.
- **`async_set_online(hass, on: bool) -> None`** (`store_account.py:111-119`): Enables/disables. **Disabling clears the stored refresh token** so a disabled install never leaves a live credential on disk.
- **`get_prefs(hass) -> dict`** (`store_account.py:122-126`): Returns integration-wide store preferences with defaults filled in.
- **`get_pref(hass, key) -> Any`** (`store_account.py:129-131`): Single pref accessor.
- **`async_set_prefs(hass, patch) -> dict`** (`store_account.py:134-145`): Merge-update only known keys. Returns updated prefs.
- **`migration_done(hass) -> bool`** (`store_account.py:148-150`): One-time per-device → global migration flag.
- **`async_mark_migrated(hass) -> None`** (`store_account.py:153-155`): Stamps `migrated=True`.
- **`get_account(hass) -> dict`** (`store_account.py:159-162`): Full account including refresh token. For internal use only.
- **`get_identity(hass) -> dict`** (`store_account.py:165-168`): Safe UI view: `{connected, uid, name}`. **Never includes the refresh token.**
- **`async_set_account(hass, account) -> None`** (`store_account.py:171-181`): Replaces stored account entirely. `None` values are dropped.
- **`async_clear_account(hass) -> None`** (`store_account.py:184-186`): Sets `account = {}`.

Current preferences: `show_contributor` (bool, default `True`) — whether contributor attribution is shown in pickers.

---

## 8. WebSocket API Surface

All handlers live in `ws_api.py`. Every handler is wrapped in `_guard` for RBAC. Store write handlers require `edit` access; browse/read handlers require only `read` access (see `_READ_LEVEL_COMMANDS` at `ws_api.py:466-473`). All handlers gate on `online_features_enabled` via `_store_ctx()` (`ws_api.py:678-688`): they return `{"disabled": True}` instead of an error when the global toggle is off.

### Global store settings (no `entry_id` semantics for the online toggle)
| WS type | Handler | Requires |
|---|---|---|
| `ha_washdata/store_set_online` | `ws_store_set_online` | edit |
| `ha_washdata/store_set_prefs` | `ws_store_set_prefs` | edit |

### Account management
| WS type | Handler |
|---|---|
| `ha_washdata/store_status` | `ws_store_status` |
| `ha_washdata/store_connect` | `ws_store_connect` (params: `refresh_token`, `uid`, `name?`) |
| `ha_washdata/store_disconnect` | `ws_store_disconnect` |

### Browse / read (no auth needed)
| WS type | Handler |
|---|---|
| `ha_washdata/store_list_brands` | `ws_store_list_brands` (params: `query?`, `include_pending?`) |
| `ha_washdata/store_search_devices` | `ws_store_search_devices` (params: `brand?`, `appliance_type?`, `model_query?`, `include_pending?`) |
| `ha_washdata/store_get_profiles` | `ws_store_get_profiles` (params: `device_id`) |
| `ha_washdata/store_get_device_profiles` | `ws_store_get_device_profiles` (params: `brand`, `model`, `appliance_type`) |
| `ha_washdata/store_get_cycles` | `ws_store_get_cycles` (params: `profile_id`) |
| `ha_washdata/store_get_device_quality` | `ws_store_get_device_quality` (params: `device_id`) |

### Community actions (authed writes)
| WS type | Handler |
|---|---|
| `ha_washdata/store_confirm_device` | `ws_store_confirm_device` (params: `device_id`) |
| `ha_washdata/store_rate_device` | `ws_store_rate_device` (params: `device_id`, `rating: 1-5`) |

### Single-cycle import/upload
| WS type | Handler |
|---|---|
| `ha_washdata/store_import_cycle` | `ws_store_import_cycle` (params: `cycle_id`, `target_profile?`, `new_profile_name?`) |
| `ha_washdata/store_upload_cycle` | `ws_store_upload_cycle` (params: `local_cycle_id`, `program`, `sample_interval_sec?`, `description?`) — brand/model come from `entry.options` |

### Device bundle (Stage 1 complete)
| WS type | Handler |
|---|---|
| `ha_washdata/store_upload_device` | `ws_store_upload_device` (params: `items: [{local_cycle_id, program}]`, `include_phases?: [str]`, `include_settings?: bool`) |
| `ha_washdata/store_download_device` | `ws_store_download_device` (params: `device_id`, `include_settings?: bool`) |
| `ha_washdata/get_shareable_cycles` | `ws_get_shareable_cycles` — returns `{items, phase_programs, all_programs}` for the share-device tree |

`ws_store_upload_device` (`ws_api.py:933-961`): reads brand/model from `entry.options`; if `include_settings`, filters `entry.options` to `SHAREABLE_SETTING_KEYS` (numeric, non-bool only) before passing to `StoreBridge.share_device`.

`ws_store_download_device` (`ws_api.py:965-998`): after `StoreBridge.download_device`, if `include_settings` and bundle carries settings, filters to `SHAREABLE_SETTING_KEYS` (same numeric guard), applies to `entry.options` via `async_update_entry`, and reports `settings_applied` count.

---

## 9. Device Package Sharing — Bundle Contents and Semantics

### What travels in a device bundle
```
Device bundle
  device_meta: {applianceType, brand, model}
  settings:    {allow-listed detection/matching keys, numeric values only}  [optional]
  items: [
    {
      program,
      points,          # LTTB-downsampled trace (<=10000 pts), [[offset_s, watts]]
      stats:           {duration, peak_w, mean_w, signature, [energy_wh]}
      qc:              1|2|3  (provenance code)
      sampleIntervalSec
      phases:          [{name, start_s, end_s}]  [optional, per-program]
      phaseSourceCycleId                          [optional]
    }
  ]
```

Phase units are **absolute seconds**. On download, phase ranges replace the local profile's phase ranges (not merged). The rationale: phases are authored against a specific cycle's timeline; runtime adaptation to local cycle-length variance is handled by the progress-fraction phase indexer in `manager._current_phase_from_progress`.

### Shareable settings allow-list
Defined in `const.py` as `SHAREABLE_SETTING_KEYS` (20 keys). Contains only recognition/matching thresholds: `min_power`, `off_delay`, `start_threshold_w`, `stop_threshold_w`, `start_duration_threshold`, `start_energy_threshold`, `completion_min_seconds`, `min_off_gap`, `end_energy_threshold`, `power_off_threshold_w`, `power_off_delay`, `profile_match_threshold`, `profile_unmatch_threshold`, `profile_match_interval`, `profile_match_min/max_duration_ratio`, `profile_duration_tolerance`, `duration_tolerance`, `auto_label_confidence`, `learning_confidence`. (`running_dead_zone` was removed from the allow-list in 0.5.3 when the setting was retired.) **Excluded by design:** entity IDs, notify services, energy price, sampling cadence, EMA smoothing, housekeeping timers, anti-wrinkle, drain, plug-robustness. All values are plain numbers (no PII, no HA topology leakage).

Settings filtering is applied **twice** (defense in depth): once in the WS handler at upload (`ws_api.py:953-955`) and once inside `StoreClient.upload_reference_cycle` at the Firestore boundary (`store_client.py:614-620`). On download, filtered again in `ws_store_download_device` (`ws_api.py:985-993`).

### Adopt semantics
- **Device bundle download**: merge/upsert. Profiles present in the bundle replace their local counterparts' reference cycles and phases. Local profiles **not** in the bundle are untouched. Real `past_cycles` are never touched.
- **Idempotent**: cycles with `meta.source == "store:<id>"` already in local `reference_cycles` are skipped (`store.py:396-397`). Re-downloading does not inflate download counters for already-imported cycles.

---

## 10. Reference Cycle Isolation

Imported cycles live in **`reference_cycles`** (a separate top-level key in the profile store's JSON), never in `past_cycles`. This is a structural guarantee, not a filter.

Consequences:
- `lifetime_energy_wh`, `lifetime_cycle_count` are accumulated only by `manager._async_process_cycle_end`, never by `add_reference_cycle` — imported energy never shows up in HA long-term statistics.
- All energy/cost/count/recency/trend consumers read `past_cycles` exclusively, so all are automatically immune.
- `_rebuild_envelope_sync` uses `past_cycles + reference_cycles` for **shape input** (so a profile seeded purely by imports still gets a matchable envelope), but **usage stats** (`avg_energy`, `cycle_count`, cost) from **real cycles only**.
- `get_shareable_cycles()` (`profile_store.py:1391-1419`) intentionally excludes `reference_cycles` — you cannot re-share what you downloaded.

`add_reference_cycle()` (`profile_store.py:1422-1485`): validates trace (at least 2 finite-coordinate points, positive duration), re-bases to offset 0, stamps with import time, forces `status="completed"` and `ml_review.golden=True`, creates a minimal profile entry if absent, then calls `_add_cycle_data` targeting `reference_cycles`, rebuilds the envelope, and persists.

---

## 11. Privacy and Safety

### What is uploaded
- Power trace: watt values + second offsets (after LTTB downsampling to <=10000 points).
- Program name (user-assigned string).
- Duration, peak watt, mean watt, energy (if known), signature dict.
- QC provenance code (integer 1-3, obfuscated in the store; mapping never in public docs).
- Uploader name: only the GitHub display name, and only when the user ticks the consent checkbox.

### What is never uploaded
- Entity IDs, HA URL, device serial numbers, home address, appliance age.
- `past_cycles` data (only golden/recorded cycles from the local `past_cycles` are shareable; `reference_cycles` are excluded from re-sharing).
- `lifetime_energy_wh` or any usage statistics.
- The refresh token: it travels only over the authenticated HA WebSocket; never logged, never in HA events (enforced by `diagnostics._SENSITIVE_KEYS`).
- Detection settings outside the `SHAREABLE_SETTING_KEYS` allow-list.

### What browse/download transmits
Anonymous Firestore REST reads. Brand, model, and appliance type are sent as query parameters to filter results. No account, no credentials.

### Credential storage
The Firebase refresh token is stored in `ha_washdata_online` (HA `Store`, plaintext JSON at rest — documented accepted risk, same as HA's own credentials). Disabling online features **immediately clears the stored token** (`store_account.py:116-118`). Explicit disconnect does the same.

### Firebase API key
Public by design (identifies the project; access control is entirely in Firestore rules). Restricted to Identity Toolkit + Token Service + Cloud Firestore APIs. The Spark free-tier plan means abuse can only exhaust quota (service pauses; no billing).

---

## 12. Staged Implementation Status

### Stage 1 — Device bundle (profiles + selected cycles): **IMPLEMENTED**
- `StoreBridge.share_device()` and `StoreBridge.download_device()` exist and are tested.
- WS handlers `ws_store_upload_device` and `ws_store_download_device` registered.
- `ws_get_shareable_cycles` provides the panel's share-tree source data.
- Phase bundling (Stage 2) is included in Stage 1's implementation.
- Settings bundling (Stage 3) is also included.

### Stage 2 — Phase bundling: **IMPLEMENTED** (shipped with Stage 1)
- `include_phases` parameter in `share_device` and `ws_store_upload_device`.
- Phase attachment in `upload_reference_cycle` (`store_client.py:631-645`).
- `_apply_phases` in `StoreBridge.download_device`.

### Stage 3 — Settings bundling: **IMPLEMENTED** (shipped with Stage 1)
- `SHAREABLE_SETTING_KEYS` defined.
- Settings filtered and attached at upload; filtered and applied at download.

### Stage 4 — Web device configurator: **NOT YET** (web repo work; spec in `2026-07-15-store-device-sharing-design.md`)

### Stage 5 — Contribution (PR) workflow: **NOT YET** (design only; `contributions` collection not yet present)

### Adoption guidance system: **IMPLEMENTED** (`setup_advisor.py` + `ws_get_setup_status`)
- `compute_setup_phase()` is a pure function driven by profile state, coverage gaps, and store-adoption status.
- Phase 1c (Store Download Verification) depends on `reference_cycles` presence.

---

## 13. Code vs. Docs Discrepancies

### STORE.md § 2 "Getting started" — step 3-4 is inaccurate
STORE.md instructs: *"Go to the Advanced tab > Click the gear icon in the top area of the Advanced tab."* The gear is a **header-level button**, not specific to the Advanced tab. The online features toggle is inside the gear's **"Online & Community"** pane, which is only visible to admin users (RBAC). The step should read "click the gear icon in the panel header."

### STORE.md § 2 step 5 — brand/model declaration is not in the gear
STORE.md says *"Use the Brand and Model pickers to declare which appliance you own"* in the context of the gear. The panel's own `_htmlOnlineSettings()` comment (`www/ha-washdata-panel.js`) explicitly states: *"Appliance brand/model are NOT here - they are per-device settings under Basic > Device info."* These are `CONF_STORE_BRAND` / `CONF_STORE_MODEL` in `entry.options`, set in the Settings tab's Basic section, not in the gear.

### Integration design spec (2026-07-14) § 6 — account storage location changed
The spec described persisting `store_account` inside the `profile_store` key. The implementation uses a **separate global HA `Store`** (`ha_washdata_online`), which is cleaner (device-agnostic). The migration path (`__init__.py:311-346`) hoists any legacy per-device account from `profile_store.get_store_account()` into the global store on first load.

### Integration design spec storage version reference outdated
The spec (`2026-07-14-washdata-store-integration-design.md`) mentions "storage v9 -> v10" for adding `reference_cycles`. Current `STORAGE_VERSION` in `const.py` is **11**. The migration history diverged; `reference_cycles` was added at some earlier step (default-initialized in `profile_store.py:943`).

### Community catalog spec — cycle count denormalization
The catalog design spec describes `profileCount` and `cycleCount` as denormalized running totals. The implementation comment (`store_client.py:669-671`) notes these are **calculated via COUNT aggregation** on the store side, not maintained as running totals by the integration — an earlier approach of incrementing them was abandoned when the rule denied the update.

### Adoption guidance spec open questions — now answered
The `2026-07-17-adoption-guidance-system-design.md` had three open questions with answers appended inline. Phase 2→3 transition snooze is 14 days (with a "never show again" option); Phase 1c advances on any non-ambiguous match; Phase 3 tuning items can be snoozed or permanently dismissed.

---

## 14. Key Constants (`const.py`)

| Constant | Value | Purpose |
|---|---|---|
| `STORE_PROJECT_ID` | `"washdata-store"` | Firestore project identifier |
| `STORE_API_KEY` | `"AIzaSy..."` | Firebase web API key (public) |
| `STORE_WEB_ORIGIN` | `"https://3dg1luk43.github.io/washdata-store"` | Trusted postMessage origin for GitHub connect |
| `DEFAULT_ENABLE_ONLINE_FEATURES` | `False` | Master gate default |
| `SUPPORTED_CYCLE_SCHEMA_VERSIONS` | `{1}` | Accepted trace format versions |
| `QC_RECORDING` / `QC_EDITED` / `QC_MANUAL` | `1` / `2` / `3` | Provenance codes |
| `SHAREABLE_SETTING_KEYS` | (20-key tuple) | Allow-list for settings bundling |

---

## 15. Test Coverage

| File | Coverage |
|---|---|
| `tests/test_store_account.py` | Load/save round-trip, migration |
| `tests/test_store_bridge.py` | All `StoreBridge` methods including share_device with phases, settings, download idempotency, phase application |
| `tests/test_store_client.py` | Token exchange/cache, normalize/id parity, Firestore decode, upload shape, confirm/rate, bundle hydration, idempotency, phase/settings attachment |
| `tests/test_store_provenance.py` | `derive_qc` for all three provenance codes, recorder precedence over edited |

Tests use mocked HTTP sessions; no live Firestore calls. The store test suite is in the fast category (no `slow` marks).
