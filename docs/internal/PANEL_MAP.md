# Panel Navigation Map

Auto-generated 2026-08-20 from `www/ha-washdata-panel.js` (12277 lines, 297 methods).
Regenerate: `node devtools/gen_panel_map.mjs`

Each entry: **Method name** — line number (method size in lines).
Methods >100 lines are flagged ⚠ and summarised in the table at the bottom.

---

## Lifecycle & Setup

- **constructor** — L1779 (178 lines) ⚠
- **connectedCallback** — L1977 (28 lines)
- **disconnectedCallback** — L2005 (18 lines)
- **_boot** — L2023 (31 lines)
- **_setupSubscriptions** — L2054 (41 lines)
- **_fetchPanelLang** — L2435 (19 lines)
- **_loadPanelLang** — L2454 (15 lines)
- **_loadPanelTranslations** — L2469 (8 lines)
- **_startPoll** — L2477 (1 lines)
- **_stopPoll** — L2478 (4 lines)
- **_applyPanelConfig** — L3433 (18 lines)

## Background Task Registry

- **_onTaskEvent** — L2095 (17 lines)
- **_pgAdoptTask** — L2112 (22 lines)
- **_pgAdoptExisting** — L2134 (7 lines)
- **_settleTaskCallback** — L2162 (15 lines)
- **_autoSettleAdopted** — L2177 (18 lines)
- **_kickAndTrack** — L2195 (56 lines)
- **_finalizeTaskError** — L2251 (13 lines)
- **_pollTaskGeneric** — L2264 (23 lines)
- **_deviceName** — L2287 (5 lines)
- **_taskActionLabel** — L2292 (15 lines)
- **_fmtEta** — L2307 (9 lines)
- **_exclNote** — L2316 (10 lines)
- **_htmlTaskPills** — L2326 (30 lines)
- **_updateTaskPills** — L2356 (8 lines)
- **_addProvisionalTask** — L2364 (14 lines)
- **_onTrackedTaskProgress** — L2378 (10 lines)
- **_pgFinishTask** — L2388 (25 lines)
- **_pgPollTask** — L2413 (15 lines)
- **_deviceTypeLabel** — L3397 (7 lines)
- **_deviceTypeOpts** — L3404 (6 lines)

## i18n / Translations

- **_panelTransUrl** — L2428 (7 lines)
- **_localize** — L3355 (7 lines)
- **_tLookup** — L3362 (8 lines)
- **_t** — L3370 (17 lines)

## WebSocket + Data Fetching

- **_ws** — L2482 (2 lines)
- **_fetchAll** — L2484 (119 lines) ⚠
- **_fetchCycles** — L2603 (19 lines)
- **_loadMoreCycles** — L2622 (12 lines)
- **_ensureStatusPhases** — L2634 (12 lines)
- **_fetchSettingsChangelog** — L2646 (15 lines)
- **_loadMlIndex** — L2856 (16 lines)
- **_loadMlSettings** — L2872 (13 lines)
- **_loadMlTrainingStatus** — L2885 (11 lines)
- **_fetchCycleProfileEnv** — L2896 (18 lines)
- **_fetchSuggestions** — L2914 (12 lines)
- **_fetchProfiles** — L2926 (14 lines)
- **_ensureProfileEnvs** — L2940 (13 lines)
- **_fetchProfileGroups** — L2953 (9 lines)
- **_selectDevice** — L2962 (72 lines)
- **_refreshDeviceBar** — L3034 (19 lines)
- **_refreshLogDrawer** — L3053 (16 lines)
- **_refreshLogFilterOptions** — L3069 (11 lines)
- **_fetchTabData** — L3080 (152 lines) ⚠
- **_fetchToolsData** — L3232 (11 lines)
- **_fetchMaintenance** — L3243 (10 lines)
- **_fetchLogs** — L3253 (12 lines)
- **_refreshLogViews** — L3323 (11 lines)
- **_syncLogFilters** — L3334 (7 lines)
- **_fetchRecState** — L3341 (4 lines)
- **_fetchFeedbacks** — L3345 (4 lines)
- **_fetchPhases** — L3349 (6 lines)
- **_loadShareProfiles** — L5089 (22 lines)
- **_loadDeviceAutomations** — L5150 (25 lines)
- **_loadStoreStatus** — L7753 (15 lines)
- **_ensureStoreConnectListener** — L7829 (62 lines)

## Undo / Optimistic Delete

- **_registerUndo** — L2661 (7 lines)
- **_undoDelete** — L2668 (11 lines)
- **_commitDelete** — L2679 (30 lines)
- **_flushPendingDeletes** — L2719 (6 lines)
- **_deleteCyclesWithUndo** — L2725 (50 lines)
- **_deleteProfileWithUndo** — L2775 (31 lines)

## Navigation & Routing

- **_dispatchSetupCta** — L3949 (42 lines)
- **_reloadSetupStatus** — L3991 (15 lines)
- **_navigate** — L5175 (9 lines)
- **_newAutomationFromEvent** — L5184 (32 lines)
- **_pref** — L8058 (7 lines)
- **_setPref** — L8065 (6 lines)

## Core Render Pipeline

- **_htmlPgRecentRuns** — L2141 (21 lines)
- **_htmlLogFilters** — L3308 (15 lines)
- **_applyFontScale** — L3426 (7 lines)
- **_render** — L3513 (25 lines)
- **_htmlHeader** — L3635 (41 lines)
- **_htmlBody** — L3676 (32 lines)
- **_htmlDeviceBar** — L3708 (25 lines)
- **_htmlStatus** — L3733 (148 lines) ⚠
- **_htmlSetupCard** — L3881 (68 lines)
- **_htmlPhaseTimeline** — L4006 (28 lines)
- **_htmlRecordingWidget** — L4034 (34 lines)
- **_htmlHistory** — L4068 (206 lines) ⚠
- **_htmlProfiles** — L4379 (74 lines)
- **_htmlProfileGroupModal** — L4453 (39 lines)
- **_htmlSettings** — L4528 (130 lines) ⚠
- **_htmlSettingsHistory** — L4658 (32 lines)
- **_htmlAutomations** — L5111 (39 lines)
- **_htmlSettingsSection** — L5286 (44 lines)
- **_htmlSettingsSearch** — L5330 (28 lines)
- **_htmlSettingsSugOnly** — L5372 (29 lines)
- **_htmlMlTab** — L5401 (32 lines)
- **_htmlMlStatusSection** — L5433 (38 lines)
- **_htmlMlLearnedSection** — L5471 (35 lines)
- **_htmlMatchingTuningCard** — L5542 (56 lines)
- **_htmlPgControlPanel** — L5748 (61 lines)
- **_htmlPlayground** — L5927 (73 lines)
- **_htmlPgDrawer** — L6000 (21 lines)
- **_htmlPgParamRows** — L6021 (94 lines)
- **_htmlPgAlerts** — L6115 (29 lines)
- **_htmlPgHistoryMode** — L6163 (71 lines)
- **_htmlPgBatchBar** — L6234 (11 lines)
- **_htmlPgSweepMode** — L6295 (19 lines)
- **_htmlPgSweepResult** — L6314 (32 lines)
- **_htmlPgStrip** — L6432 (16 lines)
- **_htmlPgAnalysis** — L6448 (71 lines)
- **_htmlPhases** — L7232 (30 lines)
- **_htmlDiagnostics** — L7262 (47 lines)
- **_htmlMaintenance** — L7320 (73 lines)
- **_htmlPanel** — L7393 (21 lines)
- **_htmlPanelPrefs** — L7420 (45 lines)
- **_htmlPanelSettings** — L7465 (23 lines)
- **_htmlPanelAccess** — L7488 (32 lines)
- **_htmlStore** — L7520 (22 lines)
- **_htmlStoreCrumbs** — L7542 (17 lines)
- **_htmlStoreLoading** — L7559 (4 lines)
- **_htmlStoreBrands** — L7563 (53 lines)
- **_htmlStoreDevice** — L7616 (20 lines)
- **_htmlStoreProfile** — L7636 (32 lines)
- **_htmlGearModal** — L7686 (20 lines)
- **_htmlOnlineSettings** — L7706 (28 lines)
- **_htmlStorePrefs** — L7734 (19 lines)
- **_htmlLogDrawer** — L7908 (22 lines)
- **_htmlModal** — L8283 (137 lines) ⚠
- **_htmlShareDeviceModal** — L8420 (85 lines)
- **_htmlSelectionTree** — L8591 (64 lines)
- **_htmlExportSelectModal** — L8655 (18 lines)
- **_htmlImportWizardModal** — L8673 (74 lines)
- **_htmlCycleModal** — L8747 (221 lines) ⚠
- **_htmlProfilePanel** — L8968 (168 lines) ⚠
- **_htmlCompareModal** — L9200 (32 lines)

## Settings Form & Persistence

- **_snapshotCycleReviewForm** — L3594 (18 lines)
- **_wizInitSel** — L8532 (15 lines)
- **_snapshotFormToPending** — L11986 (44 lines)
- **_conflictKeysForOpts** — L12030 (9 lines)
- **_conflictKeysFromOpts** — L12042 (6 lines)
- **_cascadeConflictFix** — L12130 (37 lines)
- **_saveSettings** — L12167 (111 lines) ⚠

## Community Store

- **_storeApplianceType** — L4869 (7 lines)
- **_storeDeviceDeclared** — L4876 (9 lines)
- **_shareableByProgram** — L4885 (29 lines)
- **_storeSparkline** — L7668 (18 lines)
- **_storeBrandScope** — L7768 (4 lines)
- **_storeSearch** — L7772 (35 lines)
- **_storeItemHasContent** — L7822 (7 lines)

## Playground (Simulation)

- **_pgOverrideFields** — L5598 (40 lines)
- **_pgFieldVal** — L5638 (26 lines)
- **_pgFetchSettings** — L5664 (20 lines)
- **_pgCurrentValues** — L5684 (13 lines)
- **_pgStagedVal** — L5697 (6 lines)
- **_pgSetStaged** — L5703 (6 lines)
- **_pgClearStaged** — L5709 (8 lines)
- **_pgChangedKeys** — L5717 (15 lines)
- **_pgSameVal** — L5732 (11 lines)
- **_pgIsPublishable** — L5743 (5 lines)
- **_pgApplyPresetValues** — L5809 (11 lines)
- **_pgSavePreset** — L5820 (21 lines)
- **_pgDeletePreset** — L5841 (19 lines)
- **_pgLoadLive** — L5860 (15 lines)
- **_pgLoadSuggested** — L5875 (27 lines)
- **_pgPublishOne** — L5902 (25 lines)
- **_pgAlertLabel** — L6144 (19 lines)
- **_pgUpdateBatchBar** — L6245 (13 lines)
- **_pgRunHistory** — L6258 (27 lines)
- **_pgSweepObjectives** — L6285 (10 lines)
- **_pgRunSweep2** — L6346 (35 lines)
- **_pgApplyToSettings** — L6381 (30 lines)
- **_pgApplySweepValue** — L6411 (21 lines)
- **_pgLoad** — L6519 (77 lines)
- **_pgCancelRun** — L6596 (11 lines)
- **_pgSelectCycle** — L6607 (19 lines)
- **_pgLoadDetail** — L6626 (42 lines)
- **_pgRerunDetail** — L6668 (13 lines)
- **_pgMapState** — L6681 (9 lines)
- **_pgSeriesAt** — L6690 (9 lines)
- **_pgStateSegsFromSeries** — L6699 (14 lines)
- **_pgDrawCanvas** — L6713 (386 lines) ⚠
- **_pgEventMeta** — L7099 (19 lines)
- **_pgEventDescription** — L7118 (17 lines)
- **_pgUpdateParamInput** — L7135 (15 lines)
- **_pgUpdateStripAt** — L7150 (40 lines)
- **_pgIsUnknownCmd** — L7190 (7 lines)
- **_pgInterpPower** — L7197 (15 lines)
- **_pgTrapEnergy** — L7212 (14 lines)

## ML Insights

- **_mlQualityChip** — L5506 (22 lines)
- **_mlTrendBadge** — L5528 (14 lines)

## Canvas Drawing

- **_drawProfileSparklines** — L4351 (28 lines)
- **_drawGroupCanvas** — L4492 (18 lines)
- **_drawPlaygroundCanvases** — L7226 (6 lines)
- **_drawCurves** — L7930 (98 lines)
- **_drawModalCanvas** — L8028 (14 lines)
- **_redrawCanvas** — L8042 (16 lines)
- **_drawStatusCurve** — L8071 (41 lines)
- **_drawCycleEditor** — L9136 (64 lines)
- **_drawCompareCanvas** — L9232 (35 lines)
- **_drawProfileEnvelope** — L9267 (11 lines)
- **_drawPhaseEditor** — L9278 (15 lines)
- **_drawSpaghetti** — L9293 (24 lines)
- **_wireCycleCanvas** — L10255 (39 lines)
- **_wirePhaseCanvas** — L10317 (39 lines)

## Event Wiring

- **_wire** — L9317 (851 lines) ⚠
- **_wireSplitSegments** — L10294 (8 lines)
- **_wirePhaseInputs** — L10302 (15 lines)
- **_wireCleanup** — L10365 (37 lines)

## Action Dispatch

- **_onAction** — L10402 (446 lines) ⚠
- **_onActSuggestions** — L10848 (59 lines)
- **_onActMl** — L10907 (34 lines)
- **_onActStore** — L10941 (271 lines) ⚠
- **_onActAuto** — L11212 (45 lines)
- **_onActMaintenance** — L11257 (46 lines)
- **_onActPlayground** — L11303 (70 lines)

## Modal Action Dispatch

- **_onModalAction** — L11373 (201 lines) ⚠
- **_onMActImport** — L11574 (174 lines) ⚠
- **_onMActStoreShare** — L11748 (72 lines)
- **_onMActCycleDetail** — L11820 (92 lines)
- **_onMActProfilePanel** — L11912 (74 lines)

## Utilities & Helpers

- **hass** — L1957 (17 lines)
- **panel** — L1974 (1 lines)
- **narrow** — L1975 (2 lines)
- **_isActiveEntry** — L2709 (10 lines)
- **_onKeydown** — L2806 (50 lines)
- **_logComponents** — L3265 (5 lines)
- **_logDevices** — L3270 (5 lines)
- **_filteredLogRecords** — L3275 (14 lines)
- **_logLinesHtml** — L3289 (19 lines)
- **_stateColor** — L3387 (5 lines)
- **_stateLabel** — L3392 (5 lines)
- **_deviceOpts** — L3410 (16 lines)
- **_isAdmin** — L3451 (1 lines)
- **_curPerm** — L3452 (1 lines)
- **_canEdit** — L3453 (1 lines)
- **_canFull** — L3454 (4 lines)
- **_onlineEnabled** — L3458 (4 lines)
- **_visibleTabIds** — L3462 (19 lines)
- **_busyRun** — L3481 (8 lines)
- **_closeCycleDetail** — L3489 (24 lines)
- **_resizeLogsPage** — L3538 (11 lines)
- **_syncModalFocus** — L3549 (36 lines)
- **_renderPreservingFormEdits** — L3585 (9 lines)
- **_buildHtml** — L3612 (23 lines)
- **_trendIcon** — L4274 (6 lines)
- **_profileCardHtml** — L4280 (71 lines)
- **_settingsLevel** — L4510 (7 lines)
- **_settingFieldVisible** — L4517 (6 lines)
- **_secHasBasicFields** — L4523 (5 lines)
- **_renderField** — L4690 (69 lines)
- **_renderStorePicker** — L4759 (15 lines)
- **_statusTag** — L4774 (14 lines)
- **_renderBrandPicker** — L4788 (22 lines)
- **_renderModelPicker** — L4810 (59 lines)
- **_catalogEntryKey** — L4914 (11 lines)
- **_ensureCatalogEntry** — L4925 (10 lines)
- **_catalogEntryFor** — L4935 (9 lines)
- **_loadCatalogEntry** — L4944 (32 lines)
- **_refreshComboAfterLoad** — L4976 (14 lines)
- **_ensureCatalogList** — L4990 (22 lines)
- **_ensureBrandCandidates** — L5012 (20 lines)
- **_mergeBrandCandidates** — L5032 (11 lines)
- **_loadCatalogBrands** — L5043 (32 lines)
- **_loadCatalogDevices** — L5075 (14 lines)
- **_convertLegacyActions** — L5216 (70 lines)
- **_mlSugKeys** — L5358 (14 lines)
- **_maintLabel** — L7309 (11 lines)
- **_levelSelect** — L7414 (6 lines)
- **_sortStoreDevices** — L7807 (15 lines)
- **_saveStoreOptions** — L7891 (17 lines)
- **_attachHover** — L8112 (34 lines)
- **_onGraphHover** — L8146 (13 lines)
- **_onGraphHoverInner** — L8159 (54 lines)
- **_showGraphTip** — L8213 (13 lines)
- **_hideGraphTip** — L8226 (7 lines)
- **_positionTip** — L8233 (22 lines)
- **_syncSpagRowHighlight** — L8255 (12 lines)
- **_showToast** — L8267 (10 lines)
- **_profileOptions** — L8277 (6 lines)
- **_wizCatOrder** — L8505 (6 lines)
- **_wizCatLabel** — L8511 (21 lines)
- **_wizSelectionPayload** — L8547 (15 lines)
- **_wizGroupIds** — L8562 (7 lines)
- **_wizCatState** — L8569 (22 lines)
- **_syncTrimInputs** — L10168 (15 lines)
- **_snapTrimBounds** — L10183 (23 lines)
- **_offsetToClock** — L10206 (6 lines)
- **_clockToOffset** — L10212 (22 lines)
- **_trimInputToOffset** — L10234 (8 lines)
- **_toggleSplit** — L10242 (13 lines)
- **_syncPhaseInputs** — L10356 (9 lines)
- **_conflictCountForOpts** — L12039 (3 lines)
- **_readSettingsFormValues** — L12048 (16 lines)
- **_liveValidateSettings** — L12064 (66 lines)

---

## Oversized Methods (>100 lines)

| Method | Line | Size | Group |
|--------|------|------|-------|
| `_wire` | 9317 | 851 | Event Wiring |
| `_onAction` | 10402 | 446 | Action Dispatch |
| `_pgDrawCanvas` | 6713 | 386 | Playground (Simulation) |
| `_onActStore` | 10941 | 271 | Action Dispatch |
| `_htmlCycleModal` | 8747 | 221 | Core Render Pipeline |
| `_htmlHistory` | 4068 | 206 | Core Render Pipeline |
| `_onModalAction` | 11373 | 201 | Modal Action Dispatch |
| `constructor` | 1779 | 178 | Lifecycle & Setup |
| `_onMActImport` | 11574 | 174 | Modal Action Dispatch |
| `_htmlProfilePanel` | 8968 | 168 | Core Render Pipeline |
| `_fetchTabData` | 3080 | 152 | WebSocket + Data Fetching |
| `_htmlStatus` | 3733 | 148 | Core Render Pipeline |
| `_htmlModal` | 8283 | 137 | Core Render Pipeline |
| `_htmlSettings` | 4528 | 130 | Core Render Pipeline |
| `_fetchAll` | 2484 | 119 | WebSocket + Data Fetching |
| `_saveSettings` | 12167 | 111 | Settings Form & Persistence |
