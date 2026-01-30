# Changelog

All notable changes to HA WashData will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2026-01-12


**Major Architectural Rewrite ("vNext")**

This release marks a complete re-engineering of the HA WashData core, transitioning from simple heuristics to a rigorous signal processing pipeline and robust state machine. While the version number is minor, this is effectively a new engine under the hood.

**Verification Status**: Comprehensive logic review and validation of Detection, Matching, Switching, and Prediction logic completed (Jan 2026).


### Core Architecture: Signal Processing & State Machine
- **New Signal Processing Engine** (`signal_processing.py`):
  - **Dt-Aware Integration**: Replaced simple averaging with trapezoidal Riemann sum integration (`integrate_wh`) that respects variable sampling intervals.
  - **Robust Smoothing**: Implemented `robust_smooth`, a hybrid algorithm combining a Median Filter (spike rejection) with a Time-Aware Exponential Moving Average (EMA) for clean trend detection.
  - **Adaptive Resampling**: New primitives (`resample_adaptive`, `resample_uniform`) handle irregular sensor updates and enforce strict gap handling (no interpolation across large gaps).
  - **Idle Baseline Learning**: Automatically learns the device's "true zero" using Median Absolute Deviation (MAD), removing the need for manual calibration.

- **Finite State Machine (FSM)**:
  - Replaced binary ON/OFF logic with a formal FSM: `OFF` → `STARTING` → `RUNNING` ↔ `PAUSED` → `ENDING` → `OFF`.
  - **Dt-Aware Gating**: Start/End detection now uses accumulated time/energy gates (e.g., "energy since idle > X Wh") rather than sample counts, making it immune to sensor update frequency.
  - **Smart Pausing**: Distinguishes between "End of Cycle" and "Mid-Cycle Pause" using dynamic thresholds derived from the sensor's sampling cadence (`_p95_dt`).

### Storage v2 & Migration
- **Profile Store v2** (`profile_store.py`):
  - **New Schema**: Introduced a versioned storage schema (v2) optimized for performance.
  - **Trace Compression**: Historical power traces are now compressed using relative time deltas, significantly reducing disk usage.
  - **Robust Migration**: Included a designated `WashDataStore` engine that automatically upgrades v1 data to v2 without data loss, preserving user labels and corrections.

### ✨ functionality & Features
- **Configurable Sampling Interval**: New "Sampling Interval" setting allows users to throttle high-frequency sensors (e.g., 1s updates) to reduce CPU load.
- **Precision Configuration**: Configuration flow now uses **Text Box** inputs for all numeric thresholds, offering precise control over parameters like `start_energy_threshold` (Wh) and `drop_ratio`.
- **Smart Resume**: "Resurrection" logic restores the exact cycle state (including sub-state) after a Home Assistant restart.
- **Auto-Labeling**: Increased default confidence threshold to **0.75** (from 0.70) to leverage the improved accuracy of the new engine.
- **Diagnostic Sensors**: Added dynamic diagnostic sensors for each profile (e.g., `sensor.washdata_..._profile_cotton_count`) showing the total cycle count properly.
- **Statistics**: Added "Total Energy" column to the Profile Statistics table, showing the cumulative energy consumed for each profile.

### 🛠️ Technical Improvements
- **Logging**: Added more granular `termination_reason` logging (e.g., `smart`, `timeout`, `force_stopped`) to `cycle_detector` and `profile_store`.
- **Timezone Robustness**: Complete refactor to use timezone-aware datetimes (`dt_util.now()`) exclusively, permanently fixing "offset-naive/offset-aware" comparison errors.
- **Strict Typing**: Codebase now strictly adheres to type hinting, with extensive use of `TypeAlias` and `dataclass` for internal structures.
- **Performance**: Optimized `last_match_details` sensor attribute to exclude large raw data arrays, preventing Home Assistant state update bloat.
- **Serialization**: Fixed `MatchResult` JSON serialization issues that were blocking sensor updates.

### 🐛 Bug Fixes
- **Premature Termination & Dishwasher Logic**: Major robustness improvements for dishwashers.
  - Implemented "Verified Pause" logic to prevent early termination during long drying phases.
  - Added "End Spike Wait Period": Dishwashers now wait up to 5 extra minutes after expected duration to capture final pump-out spikes.
  - Increased Smart Termination duration ratio to **0.99** for dishwashers to be more conservative.
- **Ghost Cycles**: Enhanced filtering of short "noise" events.
  - Implemented `completion_min_seconds` logic to ignore brief spikes.
  - Refined "Ghost Cycle Suppressor" with a **20-min "Suspicious Window"** to only suppress short cycles if they occur shortly after a previous cycle, preventing false positives on legitimate quick consecutive runs.
- **Profile Learning**: Fixed a feedback loop where trimming silent tails caused profiles to shrink over time.
  - Now preserves the trailing silence (and end spikes) for all natural cycle completions, ensuring the full duration is learned.
- **Start/End Flutter**: Start debounce and End repeat counts are now configurable and backed by robust accumulators, eliminating false starts/ends.
- **Cycle Detector**: Adjusted duration validation logic to strict 90%-125% window for completion.
- **Translations**: Fixed "intl string context variable not provided" errors in logs by properly passing placeholders to translation engine.
- **Debug Sensors**: Fixed "Top Candidates" sensor showing "None" due to missing data propagation.
- **Code Quality**: Addressed various linting issues (indentation, whitespace, unused arguments).
- **Crash Fixes**: Resolved `UnboundLocalError` and specific edge-case crashes in `profile_store.py` during migration.
- **Critical Fix (Runtime Matching)**: Fixed an issue where runtime profile matching was blocking the event loop and skipping DTW; now uses the full async pipeline.
- **Legacy Data Repair**: Added automatic reconstruction of missing `time_grid` in old profile envelopes to prevent errors.
- **Validation**: Fixed missing `dtw_bandwidth` key in `strings.json` causing config flow validation errors.
- **Maintenance Safety**: Fixed aggressive cleanup logic that was deleting empty/new profiles (pending training); these are now safely preserved.
- **Test Suite**: Fixed verification tests for Smart Termination and Profile Store matching.

### ⚠️ Deprecations
- **Legacy Logic**: Removed "consecutive samples" based detection in favor of time-aware accumulators.
- **Sliders**: Removed slider inputs in config flow in favor of precise text inputs.

## [0.3.2] - 2026-01-02

### Added
- **Manual Control**: "Force End Cycle" button to manually terminate stuck cycles (treats as "Completed" and saves data).
- **Reliability**: "Cycle Resurrection" logic to restore active cycle state after Home Assistant restarts.
- **Reliability**: "Smart Resume" estimation to provide progress updates during "detecting..." phase based on historical averages.
- **Smart Cycle Extension**: New feature to prevent premature cycle termination during long low-power phases (e.g., dishwasher drying).
- **Statistics**: Energy (kWh) estimates added to Profile Statistics table.
- **UI**: Added legends to profile graphs and scaled them up by 50% for better readability.
- **Config Flow**: Split "Manage Data" into "Manage Cycles" and "Manage Profiles" menus for better usability.
- **Config Flow**: Added ability to manually edit profile "Average Duration" for tuning Smart Extension.

### Fixed
- **Icons**: Fixed missing icon for "Cycle Program" select entity (now dynamically adapts to device type).
- **Frontend**: Added missing "min" unit to "Time Remaining" display on card.
- **Logic**: Improved filtering of "Ghost Cycles" (extremely short noise events).
- **Bug**: Fixed JSON serialization error preventing status updates in some cases.
- **Bug**: Ensure stats are immediately rebuilt after merging cycles.
- **Translation Keys**: Corrected missing labels for "Smart Extension Threshold" and other advanced settings.

## [0.3.1] - 2024-12-31

### Added
- **Manual Duration for Profiles**: Users can now specify a manual "Baseline Duration" when creating profiles, useful for setting up profiles without historical data (e.g., "Eco Mode - 180 mins").
- **Onboarding First Profile Step**: New users are now prompted to optionally create their first profile immediately after setting up the device, streamlining the initial experience.
- **Automatic Recalculation**: Deleting a cycle now automatically triggers a recalculation of the associated profile's statistical envelope, ensuring accurate estimates even after cleaning up bad data.

### Fixed
- **Translations**: Fixed missing text labels in the "Create Profile" modal and Onboarding flow.
- **Configuration Flow**: Resolved `AttributeError: _get_schema` in the initial setup step.
- **Mocking Issues**: Improved test verification process for Config Flow.
- **Manual Override**: Fixed issue where unselecting a manual profile while idle would not clear the program sensor.

## [0.3.0] - 2024-12-31

This release marks a significant milestone for HA WashData, introducing intelligent profile-based cycle detection, a dedicated dashboard card, a completely rewritten configuration experience, and major improvements to cycle detection and time estimation.

### Added

#### Custom Dashboard Card
**Brand new in v0.3.0!** A native-feeling Lovelace card designed specifically for washing machines, dryers, and dishwashers.
- **Compact Tile Design**: A sleek 1x6 row layout that fits perfectly with standard Home Assistant tile cards
- **Dynamic Styling**: Configure a custom **Active Icon Color** (e.g., Green or Blue) that lights up only when the appliance is running
- **Program Display**: Directly shows the detected or selected program (e.g., "Cotton 60°C") via the new **Program Entity** selector
- **Smart Details**: Toggle between "Time Remaining" (estimated) and "Progress %" as the primary status indicator
- **Shadow DOM Implementation**: Isolated styling to prevent conflicts with other Home Assistant components

#### Intelligent Profile Matching System
New cycle detection capabilities powered by NumPy:
- **Profile-Based Detection**: Learns appliance cycle patterns and uses shape correlation matching (MAE, correlation, peak similarity) to identify cycles in real-time
- **Predictive Cycle Ending**: Short-circuits the `off_delay` wait when a cycle matches with high confidence (>90%) and is >98% complete, reducing unnecessary wait times by up to 30 seconds
- **Confidence Boosting**: Adds a 20% score boost to profile matches with exceptionally high shape correlation (>0.85)
- **Smart Time Prediction**: Detects high-variance phases (e.g., heating) and "locks" the time estimate to prevent erratic jumps during unstable phases
- **Cycle Extension Logic**: Automatically extends cycles when profile matching indicates the appliance is still running, preventing premature cycle end detection
- **Sub-State Reporting**: Displays detailed cycle phases (e.g., "Running (Heating)", "Running (Cotton 60°C - 75%)") for better visibility
- **Profile Persistence**: Detected cycle profile names are now persisted and restored across Home Assistant restarts

#### Program Selection Entity
- New `select.<name>_program_select` entity for manual program overrides and system teaching
- Allows users to manually select the active program, helping the system learn and improve detection accuracy

#### Enhanced Configuration Wizard
The configuration flow has been rebuilt from the ground up to be friendlier and more organized:
- **Two-Step Wizard**: 
  - **Step 1 (Basic)**: Essential settings (Device Type, Power Sensor) and **Notifications** are now properly grouped here
  - **Step 2 (Advanced)**: Accessible via the "Edit Advanced Settings" checkbox, containing fine-tuning options for power thresholds and timeouts
- **Smart Suggestions**: The "Apply Suggested Values" feature is now integrated into the Advanced step, helping you easily adopt values learned by the engine
- **Reactive Synchronization**: All profile/cycle modifications (create/delete/rename/label) trigger updates to keep the `select` entity in sync
- **Precision UI**: Replaced sliders with precise text-based box inputs for all configuration parameters
- **Start Duration Threshold**: Now configurable in advanced settings (previously hard-coded)

#### Advanced Cycle Detection
New logic to handle tricky appliances and prevent false detections:
- **Start Debounce Filtering**: Configurable debounce period to ignore brief power spikes before confirming cycle start
- **Running Dead Zone**: A new setting to ignore power dips during the first few seconds of a cycle (useful for machines that pause shortly after starting)
- **End Repeat Count**: Requires the "Off" condition to be met multiple times consecutively before finishing a cycle, preventing false cycle ends during long pauses/soaking
- **Ghost Cycle Prevention**: Added `completion_min_seconds` to filter out short "noise" cycles from being recorded as completed
- **Device Type Configuration**: Support for multiple appliance types (washing machine, dryer, dishwasher, coffee machine) with device-type-aware progress smoothing thresholds

#### Smoother Estimation Engine
- **EMA Smoothing**: Implemented Exponential Moving Average smoothing for progress and time-remaining sensors, eliminating the "jumping" behavior seen in previous versions
- **Monotonic Progress**: The progress percentage is now (almost) strictly enforced to never go backwards, ensuring a consistent countdown experience
- **Smoothed Progress Initialization**: `_smoothed_progress` is now properly initialized in `__init__` to avoid runtime errors
- **Smart Phase Detection**: High-variance phases are detected and handled separately to prevent estimate instability

#### Pre-Completion Notifications
- Configurable alerts (`notify_before_end_minutes`) before estimated cycle end
- Helps users prepare for cycle completion without constant monitoring

#### Enhanced Testing Infrastructure
- **Data-Driven Tests**: New test suite `tests/test_real_data.py` replays real-world CSV/JSON cycle data
- **Manager Tests**: New `tests/test_manager.py` for comprehensive manager functionality testing
- **Profile Store Tests**: New `tests/test_profile_store.py` for storage and matching validation
- **Restart Persistence Tests**: New `tests/repro/test_restart_persistence.py` to verify state recovery
- **Cycle Detector Improvements**: Enhanced `tests/test_cycle_detector.py` with new test cases
- **Conftest Utilities**: Added `tests/conftest.py` with shared test fixtures

#### Development & DevOps
- **GitHub Actions Workflows**: Added `hassfest.yml` and `validate.yml` for automated validation
- **Enhanced Mock Tooling**: `devtools/mqtt_mock_socket.py` now supports:
  - `--speedup X`: Compresses time for faster testing
  - `--variability Y`: Adds realistic duration variance (default 0.15) for shape matching validation
  - `--fault [DROPOUT|GLITCH|STUCK|INCOMPLETE]`: Injects anomalies for resilience testing
- **Secrets Template**: Added `devtools/secrets.py.template` for easier development setup

#### Documentation & Assets
- **Enhanced README**: Completely rewritten with detailed configuration options, examples, and troubleshooting
- **Updated IMPLEMENTATION.md**: Reflects new architecture with NumPy-powered matching and profile persistence
- **Improved TESTING.md**: Enhanced verification guide with new test scenarios
- **Screenshot Assets**: Added screenshots in `img/` directory:
  - `integration-controls.png`
  - `integration-diagnostics.png`
  - `integration-profiles.png`
  - `integration-sensors.png`
  - `integration-settings.png`
- **GitHub Funding**: Added `.github/FUNDING.yml` for sponsor support

### Changed

#### Core System Improvements
- **Manifest Dependencies**: Added `lovelace` and `http` to `after_dependencies` to ensure reliable card loading
- **NumPy Requirement**: Added `numpy` to requirements for advanced shape correlation matching
- **Service Definitions**: Enhanced `services.yaml` with new export and configuration options
- **Profile Store Refactoring**: Complete rewrite for improved type safety, compression, and NumPy-powered matching
- **Manager Enhancements**: 
  - Better state machine handling with reactive synchronization
  - Improved notification system
  - Enhanced progress tracking with device-type-aware smoothing
  - Power sensor change protection (blocked when cycle is active)
- **Cycle Detector Evolution**: 
  - More robust state transitions
  - Better handling of edge cases
  - Enhanced logging for debugging

#### Configuration & Localization
- **Translation Updates**: Full translation support for new wizard steps and advanced settings in both `strings.json` and `translations/en.json`
- **Configuration Validation**: Improved validation and error handling in config flow
- **Settings Migration**: Automatic migration of existing settings to new format

#### Code Quality & Maintenance
- **Fixed Indentation Issues**: Corrected inconsistent indentation throughout codebase
- **Removed Trailing Whitespace**: Cleaned up formatting issues
- **Removed Unused Variables**: Eliminated unused `manager` variable from `async_step_settings`
- **Removed Unused Imports**: Cleaned up `MagicMock` and `STATE_OFF` imports from test files
- **Improved Error Handling**: Added descriptive debug logging for exception handling
- **Fixed Redundant Code**: Removed duplicate `device_type` assignment in cycle data
- **Memory Leak Prevention**: Fixed event listener accumulation in dashboard card
- **Performance Optimization**: Implemented result caching for profile matcher to avoid redundant calls

### Removed

- **Deprecated Auto-Maintenance Switch**: Removed standalone `auto_maintenance` switch entity (now a backend setting)

### Fixed

- **README Path Inconsistency**: Corrected card path documentation from `/ha_washdata/card.js` to `/ha_washdata/ha-washdata-card.js`
- **Card Editor Domain Support**: Added "select" entity domain to program_entity selector in dashboard card editor
- **End Condition Counter**: Fixed potential infinite increment issue when power stays low for extended periods
- **Start Duration Threshold**: Removed unconditional override in initial setup to allow user customization
- **Cycle Interruption Handling**: Better detection and classification of interrupted, force-stopped, and resumed cycles
- **Profile Match Extension**: Added confidence check (≥70%) to prevent spurious cycle extensions
- **Config Flow Import**: Removed duplicate import of `CONF_AUTO_MERGE_GAP_SECONDS`
- **Power Sensor Change**: Now properly blocked when a cycle is active to prevent data inconsistency

### Security

- All code changes have been validated through CodeQL security scanning
- No new vulnerabilities introduced

### Migration Guide

- **Automatic Migration**: Your existing settings will be migrated automatically to the new format
- **Card Setup**: After updating, look for the "WashData Card" in the dashboard card picker
- **Select Entity**: A new `select.<name>_program_select` entity will be created automatically
- **Deprecated Switch**: The `auto_maintenance` switch entity will be removed; this is now a backend setting

### Breaking Changes

None. This release is fully backward compatible with v0.2.x configurations.

---

## [0.2.x] - Previous Releases

See git history for details on previous releases.

