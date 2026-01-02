# Changelog

All notable changes to HA WashData will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.2] - 2026-01-02

### Added
- **Smart Cycle Extension**: New feature to prevent premature cycle termination during long low-power phases (e.g., dishwasher drying).
  - Uses historical profile data to enforce a minimum cycle duration (default 95% of average).
  - Configurable via "Smart Extension Threshold" in Advanced Settings.
- **Dynamic Configuration UI**: 
  - Switched numerical inputs to text boxes for better precision.
  - Added strict upper/lower bounds checks for safer configuration.

### Fixed
- **JSON Serialization Error**: Fixed a bug where `samples_recorded` caused API errors due to being a bounded method instead of a property.
- **Translation Keys**: Corrected missing labels for "Smart Extension Threshold" and other advanced settings in the configuration flow.
- **Config Flow UX**: Improved layout and step organization for advanced settings.

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

