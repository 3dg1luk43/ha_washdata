# Panel Navigation Map

Auto-generated 2026-08-20 from `www/ha-washdata-panel.js` (12011 lines, 282 methods).
Regenerate: `node devtools/gen_panel_map.mjs`

Each entry: **Method name** — line number (method size in lines).
Methods >100 lines are flagged ⚠ and summarised in the table at the bottom.

---

## Lifecycle & Setup

- **constructor** — L1758 (175 lines) ⚠
- **connectedCallback** — L1953 (28 lines)
- **disconnectedCallback** — L1981 (18 lines)
- **_boot** — L1999 (31 lines)
- **_setupSubscriptions** — L2030 (41 lines)
- **_fetchPanelLang** — L2411 (19 lines)
- **_loadPanelLang** — L2430 (15 lines)
- **_loadPanelTranslations** — L2445 (8 lines)
- **_startPoll** — L2453 (1 lines)
- **_stopPoll** — L2454 (4 lines)
- **_applyPanelConfig** — L3394 (18 lines)

## Background Task Registry

- **_onTaskEvent** — L2071 (17 lines)
- **_pgAdoptTask** — L2088 (22 lines)
- **_pgAdoptExisting** — L2110 (7 lines)
- **_settleTaskCallback** — L2138 (15 lines)
- **_autoSettleAdopted** — L2153 (18 lines)
- **_kickAndTrack** — L2171 (56 lines)
- **_finalizeTaskError** — L2227 (13 lines)
- **_pollTaskGeneric** — L2240 (23 lines)
- **_deviceName** — L2263 (5 lines)
- **_taskActionLabel** — L2268 (15 lines)
- **_fmtEta** — L2283 (9 lines)
- **_exclNote** — L2292 (10 lines)
- **_htmlTaskPills** — L2302 (30 lines)
- **_updateTaskPills** — L2332 (8 lines)
- **_addProvisionalTask** — L2340 (14 lines)
- **_onTrackedTaskProgress** — L2354 (10 lines)
- **_pgFinishTask** — L2364 (25 lines)
- **_pgPollTask** — L2389 (15 lines)
- **_deviceTypeLabel** — L3358 (7 lines)
- **_deviceTypeOpts** — L3365 (6 lines)

## i18n / Translations

- **_panelTransUrl** — L2404 (7 lines)
- **_localize** — L3316 (7 lines)
- **_tLookup** — L3323 (8 lines)
- **_t** — L3331 (17 lines)

## WebSocket + Data Fetching

- **_ws** — L2458 (2 lines)
- **_fetchAll** — L2460 (112 lines) ⚠
- **_fetchCycles** — L2572 (19 lines)
- **_loadMoreCycles** — L2591 (12 lines)
- **_ensureStatusPhases** — L2603 (12 lines)
- **_fetchSettingsChangelog** — L2615 (15 lines)
- **_loadMlIndex** — L2825 (16 lines)
- **_loadMlSettings** — L2841 (13 lines)
- **_loadMlTrainingStatus** — L2854 (11 lines)
- **_fetchCycleProfileEnv** — L2865 (18 lines)
- **_fetchSuggestions** — L2883 (12 lines)
- **_fetchProfiles** — L2895 (14 lines)
- **_ensureProfileEnvs** — L2909 (13 lines)
- **_fetchProfileGroups** — L2922 (9 lines)
- **_selectDevice** — L2931 (66 lines)
- **_refreshDeviceBar** — L2997 (19 lines)
- **_refreshLogDrawer** — L3016 (16 lines)
- **_refreshLogFilterOptions** — L3032 (11 lines)
- **_fetchTabData** — L3043 (150 lines) ⚠
- **_fetchToolsData** — L3193 (11 lines)
- **_fetchMaintenance** — L3204 (10 lines)
- **_fetchLogs** — L3214 (12 lines)
- **_refreshLogViews** — L3284 (11 lines)
- **_syncLogFilters** — L3295 (7 lines)
- **_fetchRecState** — L3302 (4 lines)
- **_fetchFeedbacks** — L3306 (4 lines)
- **_fetchPhases** — L3310 (6 lines)
- **_loadShareProfiles** — L4994 (22 lines)
- **_loadDeviceAutomations** — L5055 (25 lines)
- **_loadStoreStatus** — L7619 (10 lines)
- **_ensureStoreConnectListener** — L7652 (60 lines)

## Undo / Optimistic Delete

- **_registerUndo** — L2630 (7 lines)
- **_undoDelete** — L2637 (11 lines)
- **_commitDelete** — L2648 (30 lines)
- **_flushPendingDeletes** — L2688 (6 lines)
- **_deleteCyclesWithUndo** — L2694 (50 lines)
- **_deleteProfileWithUndo** — L2744 (31 lines)

## Navigation & Routing

- **_dispatchSetupCta** — L3910 (42 lines)
- **_reloadSetupStatus** — L3952 (15 lines)
- **_navigate** — L5080 (9 lines)
- **_newAutomationFromEvent** — L5089 (32 lines)
- **_pref** — L7879 (7 lines)
- **_setPref** — L7886 (6 lines)

## Core Render Pipeline

- **_htmlPgRecentRuns** — L2117 (21 lines)
- **_htmlLogFilters** — L3269 (15 lines)
- **_applyFontScale** — L3387 (7 lines)
- **_render** — L3474 (25 lines)
- **_htmlHeader** — L3596 (41 lines)
- **_htmlBody** — L3637 (32 lines)
- **_htmlDeviceBar** — L3669 (25 lines)
- **_htmlStatus** — L3694 (148 lines) ⚠
- **_htmlSetupCard** — L3842 (68 lines)
- **_htmlPhaseTimeline** — L3967 (28 lines)
- **_htmlRecordingWidget** — L3995 (34 lines)
- **_htmlHistory** — L4029 (206 lines) ⚠
- **_htmlProfiles** — L4340 (74 lines)
- **_htmlProfileGroupModal** — L4414 (39 lines)
- **_htmlSettings** — L4489 (130 lines) ⚠
- **_htmlSettingsHistory** — L4619 (32 lines)
- **_htmlAutomations** — L5016 (39 lines)
- **_htmlSettingsSection** — L5191 (44 lines)
- **_htmlSettingsSearch** — L5235 (28 lines)
- **_htmlSettingsSugOnly** — L5277 (29 lines)
- **_htmlMlTab** — L5306 (32 lines)
- **_htmlMlStatusSection** — L5338 (38 lines)
- **_htmlMlLearnedSection** — L5376 (35 lines)
- **_htmlMatchingTuningCard** — L5447 (56 lines)
- **_htmlPgControlPanel** — L5653 (61 lines)
- **_htmlPlayground** — L5832 (73 lines)
- **_htmlPgDrawer** — L5905 (21 lines)
- **_htmlPgParamRows** — L5926 (94 lines)
- **_htmlPgAlerts** — L6020 (29 lines)
- **_htmlPgHistoryMode** — L6068 (71 lines)
- **_htmlPgBatchBar** — L6139 (11 lines)
- **_htmlPgSweepMode** — L6200 (19 lines)
- **_htmlPgSweepResult** — L6219 (32 lines)
- **_htmlPgStrip** — L6337 (16 lines)
- **_htmlPgAnalysis** — L6353 (71 lines)
- **_htmlPhases** — L7137 (30 lines)
- **_htmlDiagnostics** — L7167 (47 lines)
- **_htmlMaintenance** — L7225 (73 lines)
- **_htmlPanel** — L7298 (21 lines)
- **_htmlPanelPrefs** — L7325 (45 lines)
- **_htmlPanelSettings** — L7370 (23 lines)
- **_htmlPanelAccess** — L7393 (32 lines)
- **_htmlStore** — L7425 (22 lines)
- **_htmlStoreCrumbs** — L7447 (17 lines)
- **_htmlStoreLoading** — L7464 (4 lines)
- **_htmlStoreBrands** — L7468 (23 lines)
- **_htmlStoreDevice** — L7491 (20 lines)
- **_htmlStoreProfile** — L7511 (32 lines)
- **_htmlGearModal** — L7561 (20 lines)
- **_htmlOnlineSettings** — L7581 (28 lines)
- **_htmlStorePrefs** — L7609 (10 lines)
- **_htmlLogDrawer** — L7729 (22 lines)
- **_htmlModal** — L8104 (137 lines) ⚠
- **_htmlShareDeviceModal** — L8241 (85 lines)
- **_htmlSelectionTree** — L8412 (64 lines)
- **_htmlExportSelectModal** — L8476 (18 lines)
- **_htmlImportWizardModal** — L8494 (74 lines)
- **_htmlCycleModal** — L8568 (221 lines) ⚠
- **_htmlProfilePanel** — L8789 (168 lines) ⚠
- **_htmlCompareModal** — L9021 (32 lines)

## Settings Form & Persistence

- **_snapshotCycleReviewForm** — L3555 (18 lines)
- **_wizInitSel** — L8353 (15 lines)
- **_snapshotFormToPending** — L11720 (44 lines)
- **_conflictKeysForOpts** — L11764 (9 lines)
- **_conflictKeysFromOpts** — L11776 (6 lines)
- **_cascadeConflictFix** — L11864 (37 lines)
- **_saveSettings** — L11901 (111 lines) ⚠

## Community Store

- **_storeApplianceType** — L4830 (7 lines)
- **_storeDeviceDeclared** — L4837 (9 lines)
- **_shareableByProgram** — L4846 (29 lines)
- **_storeSparkline** — L7543 (18 lines)
- **_storeSearch** — L7629 (23 lines)

## Playground (Simulation)

- **_pgOverrideFields** — L5503 (40 lines)
- **_pgFieldVal** — L5543 (26 lines)
- **_pgFetchSettings** — L5569 (20 lines)
- **_pgCurrentValues** — L5589 (13 lines)
- **_pgStagedVal** — L5602 (6 lines)
- **_pgSetStaged** — L5608 (6 lines)
- **_pgClearStaged** — L5614 (8 lines)
- **_pgChangedKeys** — L5622 (15 lines)
- **_pgSameVal** — L5637 (11 lines)
- **_pgIsPublishable** — L5648 (5 lines)
- **_pgApplyPresetValues** — L5714 (11 lines)
- **_pgSavePreset** — L5725 (21 lines)
- **_pgDeletePreset** — L5746 (19 lines)
- **_pgLoadLive** — L5765 (15 lines)
- **_pgLoadSuggested** — L5780 (27 lines)
- **_pgPublishOne** — L5807 (25 lines)
- **_pgAlertLabel** — L6049 (19 lines)
- **_pgUpdateBatchBar** — L6150 (13 lines)
- **_pgRunHistory** — L6163 (27 lines)
- **_pgSweepObjectives** — L6190 (10 lines)
- **_pgRunSweep2** — L6251 (35 lines)
- **_pgApplyToSettings** — L6286 (30 lines)
- **_pgApplySweepValue** — L6316 (21 lines)
- **_pgLoad** — L6424 (77 lines)
- **_pgCancelRun** — L6501 (11 lines)
- **_pgSelectCycle** — L6512 (19 lines)
- **_pgLoadDetail** — L6531 (42 lines)
- **_pgRerunDetail** — L6573 (13 lines)
- **_pgMapState** — L6586 (9 lines)
- **_pgSeriesAt** — L6595 (9 lines)
- **_pgStateSegsFromSeries** — L6604 (14 lines)
- **_pgDrawCanvas** — L6618 (386 lines) ⚠
- **_pgEventMeta** — L7004 (19 lines)
- **_pgEventDescription** — L7023 (17 lines)
- **_pgUpdateParamInput** — L7040 (15 lines)
- **_pgUpdateStripAt** — L7055 (40 lines)
- **_pgIsUnknownCmd** — L7095 (7 lines)
- **_pgInterpPower** — L7102 (15 lines)
- **_pgTrapEnergy** — L7117 (14 lines)

## ML Insights

- **_mlQualityChip** — L5411 (22 lines)
- **_mlTrendBadge** — L5433 (14 lines)

## Canvas Drawing

- **_drawProfileSparklines** — L4312 (28 lines)
- **_drawGroupCanvas** — L4453 (18 lines)
- **_drawPlaygroundCanvases** — L7131 (6 lines)
- **_drawCurves** — L7751 (98 lines)
- **_drawModalCanvas** — L7849 (14 lines)
- **_redrawCanvas** — L7863 (16 lines)
- **_drawStatusCurve** — L7892 (41 lines)
- **_drawCycleEditor** — L8957 (64 lines)
- **_drawCompareCanvas** — L9053 (35 lines)
- **_drawProfileEnvelope** — L9088 (11 lines)
- **_drawPhaseEditor** — L9099 (15 lines)
- **_drawSpaghetti** — L9114 (24 lines)
- **_wireCycleCanvas** — L10076 (39 lines)
- **_wirePhaseCanvas** — L10138 (39 lines)

## Event Wiring

- **_wire** — L9138 (851 lines) ⚠
- **_wireSplitSegments** — L10115 (8 lines)
- **_wirePhaseInputs** — L10123 (15 lines)
- **_wireCleanup** — L10186 (37 lines)

## Action Dispatch

- **_onAction** — L10223 (907 lines) ⚠

## Modal Action Dispatch

- **_onModalAction** — L11130 (590 lines) ⚠

## Utilities & Helpers

- **hass** — L1933 (17 lines)
- **panel** — L1950 (1 lines)
- **narrow** — L1951 (2 lines)
- **_isActiveEntry** — L2678 (10 lines)
- **_onKeydown** — L2775 (50 lines)
- **_logComponents** — L3226 (5 lines)
- **_logDevices** — L3231 (5 lines)
- **_filteredLogRecords** — L3236 (14 lines)
- **_logLinesHtml** — L3250 (19 lines)
- **_stateColor** — L3348 (5 lines)
- **_stateLabel** — L3353 (5 lines)
- **_deviceOpts** — L3371 (16 lines)
- **_isAdmin** — L3412 (1 lines)
- **_curPerm** — L3413 (1 lines)
- **_canEdit** — L3414 (1 lines)
- **_canFull** — L3415 (4 lines)
- **_onlineEnabled** — L3419 (4 lines)
- **_visibleTabIds** — L3423 (19 lines)
- **_busyRun** — L3442 (8 lines)
- **_closeCycleDetail** — L3450 (24 lines)
- **_resizeLogsPage** — L3499 (11 lines)
- **_syncModalFocus** — L3510 (36 lines)
- **_renderPreservingFormEdits** — L3546 (9 lines)
- **_buildHtml** — L3573 (23 lines)
- **_trendIcon** — L4235 (6 lines)
- **_profileCardHtml** — L4241 (71 lines)
- **_settingsLevel** — L4471 (7 lines)
- **_settingFieldVisible** — L4478 (6 lines)
- **_secHasBasicFields** — L4484 (5 lines)
- **_renderField** — L4651 (69 lines)
- **_renderStorePicker** — L4720 (15 lines)
- **_statusTag** — L4735 (14 lines)
- **_renderBrandPicker** — L4749 (22 lines)
- **_renderModelPicker** — L4771 (59 lines)
- **_catalogEntryKey** — L4875 (11 lines)
- **_ensureCatalogEntry** — L4886 (10 lines)
- **_catalogEntryFor** — L4896 (9 lines)
- **_loadCatalogEntry** — L4905 (32 lines)
- **_refreshComboAfterLoad** — L4937 (12 lines)
- **_ensureCatalogList** — L4949 (19 lines)
- **_loadCatalogBrands** — L4968 (12 lines)
- **_loadCatalogDevices** — L4980 (14 lines)
- **_convertLegacyActions** — L5121 (70 lines)
- **_mlSugKeys** — L5263 (14 lines)
- **_maintLabel** — L7214 (11 lines)
- **_levelSelect** — L7319 (6 lines)
- **_saveStoreOptions** — L7712 (17 lines)
- **_attachHover** — L7933 (34 lines)
- **_onGraphHover** — L7967 (13 lines)
- **_onGraphHoverInner** — L7980 (54 lines)
- **_showGraphTip** — L8034 (13 lines)
- **_hideGraphTip** — L8047 (7 lines)
- **_positionTip** — L8054 (22 lines)
- **_syncSpagRowHighlight** — L8076 (12 lines)
- **_showToast** — L8088 (10 lines)
- **_profileOptions** — L8098 (6 lines)
- **_wizCatOrder** — L8326 (6 lines)
- **_wizCatLabel** — L8332 (21 lines)
- **_wizSelectionPayload** — L8368 (15 lines)
- **_wizGroupIds** — L8383 (7 lines)
- **_wizCatState** — L8390 (22 lines)
- **_syncTrimInputs** — L9989 (15 lines)
- **_snapTrimBounds** — L10004 (23 lines)
- **_offsetToClock** — L10027 (6 lines)
- **_clockToOffset** — L10033 (22 lines)
- **_trimInputToOffset** — L10055 (8 lines)
- **_toggleSplit** — L10063 (13 lines)
- **_syncPhaseInputs** — L10177 (9 lines)
- **_conflictCountForOpts** — L11773 (3 lines)
- **_readSettingsFormValues** — L11782 (16 lines)
- **_liveValidateSettings** — L11798 (66 lines)

---

## Oversized Methods (>100 lines)

| Method | Line | Size | Group |
|--------|------|------|-------|
| `_onAction` | 10223 | 907 | Action Dispatch |
| `_wire` | 9138 | 851 | Event Wiring |
| `_onModalAction` | 11130 | 590 | Modal Action Dispatch |
| `_pgDrawCanvas` | 6618 | 386 | Playground (Simulation) |
| `_htmlCycleModal` | 8568 | 221 | Core Render Pipeline |
| `_htmlHistory` | 4029 | 206 | Core Render Pipeline |
| `constructor` | 1758 | 175 | Lifecycle & Setup |
| `_htmlProfilePanel` | 8789 | 168 | Core Render Pipeline |
| `_fetchTabData` | 3043 | 150 | WebSocket + Data Fetching |
| `_htmlStatus` | 3694 | 148 | Core Render Pipeline |
| `_htmlModal` | 8104 | 137 | Core Render Pipeline |
| `_htmlSettings` | 4489 | 130 | Core Render Pipeline |
| `_fetchAll` | 2460 | 112 | WebSocket + Data Fetching |
| `_saveSettings` | 11901 | 111 | Settings Form & Persistence |
