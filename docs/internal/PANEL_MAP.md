# Panel Navigation Map

Auto-generated 2026-10-02 from `www/ha-washdata-panel.js` (14744 lines, 350 methods).
Regenerate: `node devtools/gen_panel_map.mjs`

Each entry: **Method name** — line number (method size in lines).
Methods >100 lines are flagged ⚠ and summarised in the table at the bottom.

---

## Lifecycle & Setup

- **constructor** — L2090 (208 lines) ⚠
- **connectedCallback** — L2318 (55 lines)
- **disconnectedCallback** — L2385 (28 lines)
- **_boot** — L2413 (40 lines)
- **_setupSubscriptions** — L2453 (41 lines)
- **_fetchPanelLang** — L2859 (19 lines)
- **_loadPanelLang** — L2878 (15 lines)
- **_loadPanelTranslations** — L2893 (8 lines)
- **_startPoll** — L2901 (1 lines)
- **_stopPoll** — L2902 (4 lines)
- **_applyPanelConfig** — L4026 (18 lines)

## Background Task Registry

- **_onTaskEvent** — L2494 (23 lines)
- **_pgAdoptTask** — L2517 (22 lines)
- **_pgAdoptExisting** — L2539 (7 lines)
- **_settleTaskCallback** — L2567 (15 lines)
- **_autoSettleAdopted** — L2582 (18 lines)
- **_kickAndTrack** — L2600 (56 lines)
- **_finalizeTaskError** — L2656 (13 lines)
- **_pollTaskGeneric** — L2669 (23 lines)
- **_deviceName** — L2692 (5 lines)
- **_taskActionLabel** — L2697 (17 lines)
- **_fmtEta** — L2714 (9 lines)
- **_exclNote** — L2723 (10 lines)
- **_htmlTaskPills** — L2733 (30 lines)
- **_updateTaskPills** — L2763 (8 lines)
- **_addProvisionalTask** — L2771 (14 lines)
- **_onTrackedTaskProgress** — L2785 (10 lines)
- **_pgFinishTask** — L2795 (26 lines)
- **_pgPollTask** — L2837 (15 lines)
- **_deviceTypeLabel** — L3944 (11 lines)
- **_deviceTypeOpts** — L3955 (18 lines)

## i18n / Translations

- **_panelTransUrl** — L2852 (7 lines)
- **_localize** — L3902 (7 lines)
- **_tLookup** — L3909 (8 lines)
- **_t** — L3917 (17 lines)

## WebSocket + Data Fetching

- **_ws** — L2906 (2 lines)
- **_fetchAll** — L2908 (136 lines) ⚠
- **_fetchCycles** — L3044 (21 lines)
- **_loadMoreCycles** — L3065 (12 lines)
- **_ensureStatusPhases** — L3077 (12 lines)
- **_fetchSettingsChangelog** — L3089 (15 lines)
- **_loadMlIndex** — L3299 (17 lines)
- **_loadMlSettings** — L3316 (14 lines)
- **_loadMlTrainingStatus** — L3330 (11 lines)
- **_fetchCycleProfileEnv** — L3341 (18 lines)
- **_fetchSuggestions** — L3359 (13 lines)
- **_fetchProfiles** — L3372 (15 lines)
- **_ensureProfileEnvs** — L3387 (13 lines)
- **_fetchProfileGroups** — L3400 (17 lines)
- **_selectDevice** — L3476 (74 lines)
- **_refreshDeviceBar** — L3563 (19 lines)
- **_refreshLogDrawer** — L3582 (16 lines)
- **_refreshLogFilterOptions** — L3598 (11 lines)
- **_fetchTabData** — L3609 (170 lines) ⚠
- **_fetchToolsData** — L3779 (11 lines)
- **_fetchMaintenance** — L3790 (10 lines)
- **_fetchLogs** — L3800 (12 lines)
- **_refreshLogViews** — L3870 (11 lines)
- **_syncLogFilters** — L3881 (7 lines)
- **_fetchRecState** — L3888 (4 lines)
- **_fetchFeedbacks** — L3892 (4 lines)
- **_fetchPhases** — L3896 (6 lines)
- **_ensureStoreTypeDevices** — L6040 (23 lines)
- **_loadShareProfiles** — L6077 (22 lines)
- **_loadDeviceAutomations** — L6138 (25 lines)
- **_loadStoreStatus** — L8880 (15 lines)
- **_ensureStoreConnectListener** — L9001 (103 lines) ⚠

## Undo / Optimistic Delete

- **_registerUndo** — L3104 (7 lines)
- **_undoDelete** — L3111 (11 lines)
- **_commitDelete** — L3122 (30 lines)
- **_flushPendingDeletes** — L3162 (6 lines)
- **_deleteCyclesWithUndo** — L3168 (50 lines)
- **_deleteProfileWithUndo** — L3218 (31 lines)

## Navigation & Routing

- **_dispatchSetupCta** — L4768 (42 lines)
- **_reloadSetupStatus** — L4810 (15 lines)
- **_navigate** — L6163 (9 lines)
- **_newAutomationFromEvent** — L6172 (32 lines)
- **_pref** — L9306 (7 lines)
- **_setPref** — L9313 (6 lines)

## Core Render Pipeline

- **_htmlPgRecentRuns** — L2546 (21 lines)
- **_htmlStaleChip** — L3550 (8 lines)
- **_htmlLogFilters** — L3855 (15 lines)
- **_applyFontScale** — L4019 (7 lines)
- **_render** — L4240 (72 lines)
- **_htmlHeader** — L4412 (41 lines)
- **_htmlBody** — L4453 (40 lines)
- **_htmlDeviceBar** — L4493 (26 lines)
- **_htmlStatus** — L4519 (181 lines) ⚠
- **_htmlSetupCard** — L4700 (68 lines)
- **_htmlPhaseTimeline** — L4825 (28 lines)
- **_htmlRecordingWidget** — L4853 (34 lines)
- **_htmlHistory** — L4887 (224 lines) ⚠
- **_htmlProfiles** — L5255 (74 lines)
- **_htmlProfileGroupModal** — L5329 (39 lines)
- **_htmlSettings** — L5404 (132 lines) ⚠
- **_htmlSettingsHistory** — L5536 (32 lines)
- **_htmlAutomations** — L6099 (39 lines)
- **_htmlSettingsSection** — L6274 (44 lines)
- **_htmlSettingsSearch** — L6318 (28 lines)
- **_htmlSettingsSugOnly** — L6400 (29 lines)
- **_htmlMlTab** — L6429 (32 lines)
- **_htmlMlStatusSection** — L6461 (38 lines)
- **_htmlMlLearnedSection** — L6499 (35 lines)
- **_htmlMatchingTuningCard** — L6570 (56 lines)
- **_htmlPgControlPanel** — L6802 (61 lines)
- **_htmlPlayground** — L6981 (73 lines)
- **_htmlPgDrawer** — L7054 (21 lines)
- **_htmlPgParamRows** — L7075 (94 lines)
- **_htmlPgAlerts** — L7169 (29 lines)
- **_htmlPgHistoryMode** — L7217 (71 lines)
- **_htmlPgBatchBar** — L7288 (11 lines)
- **_htmlPgSweepMode** — L7349 (19 lines)
- **_htmlPgSweepResult** — L7368 (32 lines)
- **_htmlPgStrip** — L7486 (16 lines)
- **_htmlPgAnalysis** — L7502 (71 lines)
- **_htmlPhases** — L8292 (30 lines)
- **_htmlDiagnostics** — L8322 (54 lines)
- **_htmlMaintenance** — L8387 (118 lines) ⚠
- **_htmlPanel** — L8505 (21 lines)
- **_htmlPanelPrefs** — L8532 (45 lines)
- **_htmlPanelSettings** — L8577 (23 lines)
- **_htmlPanelAccess** — L8600 (32 lines)
- **_htmlStore** — L8632 (22 lines)
- **_htmlStoreCrumbs** — L8654 (17 lines)
- **_htmlStoreLoading** — L8671 (4 lines)
- **_htmlStoreBrands** — L8675 (56 lines)
- **_htmlStoreDevice** — L8743 (20 lines)
- **_htmlStoreProfile** — L8763 (32 lines)
- **_htmlGearModal** — L8813 (20 lines)
- **_htmlOnlineSettings** — L8833 (28 lines)
- **_htmlStorePrefs** — L8861 (19 lines)
- **_htmlLogDrawer** — L9134 (22 lines)
- **_htmlModal** — L9845 (138 lines) ⚠
- **_htmlShareDeviceModal** — L9983 (85 lines)
- **_htmlSelectionTree** — L10154 (64 lines)
- **_htmlExportSelectModal** — L10218 (18 lines)
- **_htmlImportWizardModal** — L10236 (80 lines)
- **_htmlHistoryImportModal** — L10335 (144 lines) ⚠
- **_htmlCycleModal** — L10492 (243 lines) ⚠
- **_htmlProfilePanel** — L10735 (168 lines) ⚠
- **_htmlCompareModal** — L10974 (32 lines)

## Settings Form & Persistence

- **_snapshotCycleReviewForm** — L4371 (18 lines)
- **_wizInitSel** — L10095 (15 lines)
- **_snapshotFormToPending** — L14335 (54 lines)
- **_conflictKeysForOpts** — L14405 (12 lines)
- **_conflictKeysFromOpts** — L14425 (7 lines)
- **_cascadeConflictFix** — L14543 (52 lines)
- **_saveSettings** — L14608 (137 lines) ⚠

## Community Store

- **_storeApplianceType** — L5760 (7 lines)
- **_storeDeviceDeclared** — L5767 (9 lines)
- **_shareableByProgram** — L5776 (33 lines)
- **_storeSearchHtml** — L8731 (12 lines)
- **_storeSparkline** — L8795 (18 lines)
- **_storeBrandScope** — L8895 (8 lines)
- **_storeOpenDevice** — L8923 (14 lines)
- **_storeOpenModel** — L8937 (7 lines)
- **_storeSearch** — L8944 (35 lines)
- **_storeItemHasContent** — L8994 (7 lines)

## Playground (Simulation)

- **_pgOverrideFields** — L6626 (42 lines)
- **_pgFieldVal** — L6668 (26 lines)
- **_pgFetchSettings** — L6694 (18 lines)
- **_pgApplySuggestions** — L6712 (11 lines)
- **_pgFetchSuggestions** — L6723 (15 lines)
- **_pgCurrentValues** — L6738 (13 lines)
- **_pgStagedVal** — L6751 (6 lines)
- **_pgSetStaged** — L6757 (6 lines)
- **_pgClearStaged** — L6763 (8 lines)
- **_pgChangedKeys** — L6771 (15 lines)
- **_pgSameVal** — L6786 (11 lines)
- **_pgIsPublishable** — L6797 (5 lines)
- **_pgApplyPresetValues** — L6863 (11 lines)
- **_pgSavePreset** — L6874 (21 lines)
- **_pgDeletePreset** — L6895 (19 lines)
- **_pgLoadLive** — L6914 (15 lines)
- **_pgLoadSuggested** — L6929 (27 lines)
- **_pgPublishOne** — L6956 (25 lines)
- **_pgAlertLabel** — L7198 (19 lines)
- **_pgUpdateBatchBar** — L7299 (13 lines)
- **_pgRunHistory** — L7312 (27 lines)
- **_pgSweepObjectives** — L7339 (10 lines)
- **_pgRunSweep2** — L7400 (35 lines)
- **_pgApplyToSettings** — L7435 (30 lines)
- **_pgApplySweepValue** — L7465 (21 lines)
- **_pgLoad** — L7573 (77 lines)
- **_pgCancelRun** — L7650 (11 lines)
- **_pgSelectCycle** — L7661 (19 lines)
- **_pgLoadDetail** — L7680 (42 lines)
- **_pgRerunDetail** — L7722 (13 lines)
- **_pgMapState** — L7735 (9 lines)
- **_pgSeriesAt** — L7744 (9 lines)
- **_pgStateSegsFromSeries** — L7753 (14 lines)
- **_pgDrawCanvas** — L7767 (392 lines) ⚠
- **_pgEventMeta** — L8159 (19 lines)
- **_pgEventDescription** — L8178 (17 lines)
- **_pgUpdateParamInput** — L8195 (15 lines)
- **_pgUpdateStripAt** — L8210 (40 lines)
- **_pgIsUnknownCmd** — L8250 (7 lines)
- **_pgInterpPower** — L8257 (15 lines)
- **_pgTrapEnergy** — L8272 (14 lines)

## ML Insights

- **_mlQualityChip** — L6534 (22 lines)
- **_mlTrendBadge** — L6556 (14 lines)

## Canvas Drawing

- **_drawProfileSparklines** — L5242 (13 lines)
- **_drawGroupCanvas** — L5368 (18 lines)
- **_drawPlaygroundCanvases** — L8286 (6 lines)
- **_drawCurves** — L9156 (120 lines) ⚠
- **_drawModalCanvas** — L9276 (14 lines)
- **_redrawCanvas** — L9290 (16 lines)
- **_drawStatusCurve** — L9319 (58 lines)
- **_drawHistorySparklines** — L10479 (13 lines)
- **_drawCycleEditor** — L10903 (71 lines)
- **_drawCompareCanvas** — L11006 (35 lines)
- **_drawProfileEnvelope** — L11041 (11 lines)
- **_drawPhaseEditor** — L11052 (15 lines)
- **_drawSpaghetti** — L11067 (24 lines)
- **_wireCycleCanvas** — L12276 (64 lines)
- **_wirePhaseCanvas** — L12363 (49 lines)

## Event Wiring

- **_wire** — L11091 (1098 lines) ⚠
- **_wireSplitSegments** — L12340 (8 lines)
- **_wirePhaseInputs** — L12348 (15 lines)
- **_wireCleanup** — L12421 (46 lines)

## Action Dispatch

- **_onAction** — L12467 (468 lines) ⚠
- **_onActSuggestions** — L12935 (60 lines)
- **_onActMl** — L12995 (34 lines)
- **_onActStore** — L13029 (275 lines) ⚠
- **_onActAuto** — L13304 (45 lines)
- **_onActMaintenance** — L13349 (58 lines)
- **_onActPlayground** — L13407 (70 lines)

## Modal Action Dispatch

- **_onModalAction** — L13477 (210 lines) ⚠
- **_onMActHistoryImport** — L13792 (102 lines) ⚠
- **_onMActImport** — L13894 (176 lines) ⚠
- **_onMActStoreShare** — L14070 (72 lines)
- **_onMActCycleDetail** — L14142 (92 lines)
- **_onMActProfilePanel** — L14234 (101 lines) ⚠

## Utilities & Helpers

- **hass** — L2298 (17 lines)
- **panel** — L2315 (1 lines)
- **narrow** — L2316 (2 lines)
- **_syncPanelHeight** — L2373 (12 lines)
- **_awaitTask** — L2821 (16 lines)
- **_isActiveEntry** — L3152 (10 lines)
- **_onKeydown** — L3249 (50 lines)
- **_deepLinkToken** — L3417 (11 lines)
- **_deepLinkIdx** — L3428 (18 lines)
- **_rememberDevice** — L3446 (9 lines)
- **_onOwnPath** — L3455 (11 lines)
- **_onLocationChanged** — L3466 (10 lines)
- **_refreshStaleChip** — L3558 (5 lines)
- **_logComponents** — L3812 (5 lines)
- **_logDevices** — L3817 (5 lines)
- **_filteredLogRecords** — L3822 (14 lines)
- **_logLinesHtml** — L3836 (19 lines)
- **_stateColor** — L3934 (5 lines)
- **_stateLabel** — L3939 (5 lines)
- **_deviceOpts** — L3973 (26 lines)
- **_ownDeviceId** — L3999 (20 lines)
- **_isAdmin** — L4044 (1 lines)
- **_curPerm** — L4045 (1 lines)
- **_canEdit** — L4046 (1 lines)
- **_canFull** — L4047 (4 lines)
- **_onlineEnabled** — L4051 (4 lines)
- **_visibleTabIds** — L4055 (19 lines)
- **_busyRun** — L4074 (8 lines)
- **_closeCycleDetail** — L4082 (64 lines)
- **_eachScroller** — L4146 (15 lines)
- **_captureScroll** — L4161 (10 lines)
- **_scrollChromeOnly** — L4171 (9 lines)
- **_restoreScroll** — L4180 (23 lines)
- **_navKey** — L4203 (15 lines)
- **_modalNavKey** — L4218 (22 lines)
- **_resizeLogsPage** — L4312 (11 lines)
- **_syncModalFocus** — L4323 (39 lines)
- **_renderPreservingFormEdits** — L4362 (9 lines)
- **_buildHtml** — L4389 (23 lines)
- **_trendIcon** — L5111 (6 lines)
- **_profileCardHtml** — L5117 (107 lines) ⚠
- **_paintSparkline** — L5224 (18 lines)
- **_settingsLevel** — L5386 (7 lines)
- **_settingFieldVisible** — L5393 (6 lines)
- **_secHasBasicFields** — L5399 (5 lines)
- **_renderField** — L5568 (82 lines)
- **_renderStorePicker** — L5650 (15 lines)
- **_statusTag** — L5665 (14 lines)
- **_renderBrandPicker** — L5679 (22 lines)
- **_renderModelPicker** — L5701 (59 lines)
- **_editedOpts** — L5809 (7 lines)
- **_catalogEntryKey** — L5816 (11 lines)
- **_ensureCatalogEntry** — L5827 (10 lines)
- **_catalogEntryFor** — L5837 (9 lines)
- **_loadCatalogEntry** — L5846 (32 lines)
- **_refreshComboAfterLoad** — L5878 (32 lines)
- **_ensureCatalogList** — L5910 (32 lines)
- **_ensureBrandCandidates** — L5942 (20 lines)
- **_mergeBrandCandidates** — L5962 (11 lines)
- **_loadCatalogBrands** — L5973 (36 lines)
- **_syncStoreSearchCandidates** — L6009 (21 lines)
- **_dropModelCandidates** — L6030 (10 lines)
- **_loadCatalogDevices** — L6063 (14 lines)
- **_convertLegacyActions** — L6204 (70 lines)
- **_mlSugKeys** — L6346 (14 lines)
- **_mlSugKeysFrom** — L6360 (15 lines)
- **_sugCountsForDevice** — L6375 (25 lines)
- **_maintLabel** — L8376 (11 lines)
- **_levelSelect** — L8526 (6 lines)
- **_renderKeepingStoreQFocus** — L8903 (20 lines)
- **_sortStoreDevices** — L8979 (15 lines)
- **_saveStoreOptions** — L9104 (30 lines)
- **_attachGraphGestures** — L9377 (100 lines)
- **_attachTouchOwnershipGuard** — L9477 (9 lines)
- **_onAxisHandle** — L9486 (13 lines)
- **_onGraphPinch** — L9499 (38 lines)
- **_setCanvasViewport** — L9537 (17 lines)
- **_zoomCanvasAbout** — L9554 (13 lines)
- **_resetCanvasZoom** — L9567 (11 lines)
- **_ensureZoomReset** — L9578 (20 lines)
- **_syncZoomReset** — L9598 (7 lines)
- **_onGraphHover** — L9605 (14 lines)
- **_onGraphHoverInner** — L9619 (79 lines)
- **_showGraphTip** — L9698 (38 lines)
- **_pinGraphTip** — L9736 (12 lines)
- **_maybeDismissGraphTip** — L9748 (7 lines)
- **_unpinGraphTip** — L9755 (9 lines)
- **_hideGraphTip** — L9764 (12 lines)
- **_positionTip** — L9776 (28 lines)
- **_positionComboDrop** — L9804 (13 lines)
- **_syncSpagRowHighlight** — L9817 (12 lines)
- **_showToast** — L9829 (10 lines)
- **_profileOptions** — L9839 (6 lines)
- **_wizCatOrder** — L10068 (6 lines)
- **_wizCatLabel** — L10074 (21 lines)
- **_wizSelectionPayload** — L10110 (15 lines)
- **_wizGroupIds** — L10125 (7 lines)
- **_wizCatState** — L10132 (22 lines)
- **_histSkipReason** — L10316 (10 lines)
- **_histSegReason** — L10326 (9 lines)
- **_syncTrimInputs** — L12189 (15 lines)
- **_snapTrimBounds** — L12204 (23 lines)
- **_offsetToClock** — L12227 (6 lines)
- **_clockToOffset** — L12233 (22 lines)
- **_trimInputToOffset** — L12255 (8 lines)
- **_toggleSplit** — L12263 (13 lines)
- **_syncPhaseInputs** — L12412 (9 lines)
- **_histUpload** — L13687 (21 lines)
- **_histStartScan** — L13708 (18 lines)
- **_histAdopt** — L13726 (13 lines)
- **_histTaskFinished** — L13739 (53 lines)
- **_collectCheckboxlist** — L14389 (16 lines)
- **_conflictCtx** — L14417 (5 lines)
- **_conflictCountForOpts** — L14422 (3 lines)
- **_readSettingsFormValues** — L14432 (26 lines)
- **_liveValidateSettings** — L14458 (85 lines)
- **_changedOptions** — L14595 (13 lines)

---

## Oversized Methods (>100 lines)

| Method | Line | Size | Group |
|--------|------|------|-------|
| `_wire` | 11091 | 1098 | Event Wiring |
| `_onAction` | 12467 | 468 | Action Dispatch |
| `_pgDrawCanvas` | 7767 | 392 | Playground (Simulation) |
| `_onActStore` | 13029 | 275 | Action Dispatch |
| `_htmlCycleModal` | 10492 | 243 | Core Render Pipeline |
| `_htmlHistory` | 4887 | 224 | Core Render Pipeline |
| `_onModalAction` | 13477 | 210 | Modal Action Dispatch |
| `constructor` | 2090 | 208 | Lifecycle & Setup |
| `_htmlStatus` | 4519 | 181 | Core Render Pipeline |
| `_onMActImport` | 13894 | 176 | Modal Action Dispatch |
| `_fetchTabData` | 3609 | 170 | WebSocket + Data Fetching |
| `_htmlProfilePanel` | 10735 | 168 | Core Render Pipeline |
| `_htmlHistoryImportModal` | 10335 | 144 | Core Render Pipeline |
| `_htmlModal` | 9845 | 138 | Core Render Pipeline |
| `_saveSettings` | 14608 | 137 | Settings Form & Persistence |
| `_fetchAll` | 2908 | 136 | WebSocket + Data Fetching |
| `_htmlSettings` | 5404 | 132 | Core Render Pipeline |
| `_drawCurves` | 9156 | 120 | Canvas Drawing |
| `_htmlMaintenance` | 8387 | 118 | Core Render Pipeline |
| `_profileCardHtml` | 5117 | 107 | Utilities & Helpers |
| `_ensureStoreConnectListener` | 9001 | 103 | WebSocket + Data Fetching |
| `_onMActHistoryImport` | 13792 | 102 | Modal Action Dispatch |
| `_onMActProfilePanel` | 14234 | 101 | Modal Action Dispatch |
