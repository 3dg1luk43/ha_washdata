#!/usr/bin/env node
/**
 * Generate docs/internal/PANEL_MAP.md from the panel source.
 *
 * Extracts every class method / getter / setter from ha-washdata-panel.js,
 * groups them by subsystem prefix, reports line numbers and method sizes,
 * and flags oversized methods (>100 lines).
 *
 * Usage: node devtools/gen_panel_map.mjs
 */

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PANEL = path.join(__dirname, '../custom_components/ha_washdata/www/ha-washdata-panel.js');
const OUT   = path.join(__dirname, '../docs/internal/PANEL_MAP.md');

const src = fs.readFileSync(PANEL, 'utf8').split('\n');
const total = src.length;

// Match a method definition at exactly 2-space indent inside the class body.
const METHOD_RE = /^  ((?:async |static |get |set )*)?([a-z_A-Z][a-zA-Z0-9_]*)\s*\(/;
// JS keywords and built-in names that are not method definitions.
const JS_KEYWORDS = new Set([
  'for','while','if','else','switch','case','catch','try','do','return','throw',
  'yield','await','const','let','var','function','class','import','export','default',
  'new','delete','typeof','instanceof','in','of','break','continue','debugger','with',
  'void','null','undefined','true','false',
  'String','Number','Boolean','Array','Object','Promise','Error','Map','Set',
  'WeakMap','WeakSet','Symbol','Proxy','Reflect','JSON','Math','Date','RegExp',
  'parseInt','parseFloat','isNaN','isFinite','decodeURIComponent','encodeURIComponent',
]);

const methods = [];
for (let i = 0; i < src.length; i++) {
  const m = METHOD_RE.exec(src[i]);
  if (!m) continue;
  const name = m[2];
  if (JS_KEYWORDS.has(name)) continue;
  // Skip object/CSS property `key: value`.
  const rest = src[i].slice(src[i].indexOf(name) + name.length).trim();
  if (rest.startsWith(':')) continue;
  methods.push({ line: i + 1, name, mods: (m[1] || '').trim() });
}

// Method size = distance to the next method (or end of file).
for (let i = 0; i < methods.length; i++) {
  methods[i].size = (methods[i + 1] ? methods[i + 1].line : total + 1) - methods[i].line;
}

// ── Subsystem groupings (checked in order; first match wins) ─────────────────
const GROUPS = [
  { id: 'lifecycle', label: 'Lifecycle & Setup',
    test: n => ['constructor','connectedCallback','disconnectedCallback','_boot','_setupSubscriptions','_startPoll','_stopPoll'].includes(n)
           || n.startsWith('_applyPanel') || n.startsWith('_loadPanel') || n.startsWith('_fetchPanel') },
  { id: 'tasks', label: 'Background Task Registry',
    test: n => n.startsWith('_kickAndTrack') || n.startsWith('_pgAdopt') || n.startsWith('_pgFinish')
           || n.startsWith('_pgPoll') || n.startsWith('_onTask') || n.startsWith('_settleTask')
           || n.startsWith('_autoSettle') || n.startsWith('_finalizeTask') || n.startsWith('_pollTask')
           || n.startsWith('_addProvisional') || n.startsWith('_onTracked') || n.startsWith('_taskAction')
           || n.startsWith('_fmtEta') || n.startsWith('_exclNote') || n.startsWith('_htmlTask')
           || n.startsWith('_updateTask') || n.startsWith('_deviceName') || n.startsWith('_deviceType') },
  { id: 'i18n', label: 'i18n / Translations',
    test: n => n === '_t' || n === '_tLookup' || n === '_localize'
           || n.startsWith('_panelTransUrl') || n.startsWith('_fetchPanelLang')
           || n.startsWith('_loadPanelLang') || n.startsWith('_loadPanelTrans') },
  { id: 'ws', label: 'WebSocket + Data Fetching',
    test: n => n === '_ws' || n.startsWith('_fetch') || n.startsWith('_loadMore')
           || n.startsWith('_ensureStatus') || n.startsWith('_ensureProfile')
           || n.startsWith('_loadMl') || n.startsWith('_loadStore') || n.startsWith('_loadShare')
           || n.startsWith('_ensureStore') || n.startsWith('_loadDevice') || n.startsWith('_loadRbac')
           || n.startsWith('_fetchAll') || n.startsWith('_selectDevice')
           || n.startsWith('_refreshDevice') || n.startsWith('_refreshLog') || n.startsWith('_syncLog') },
  { id: 'undo', label: 'Undo / Optimistic Delete',
    test: n => n.startsWith('_registerUndo') || n.startsWith('_undoDelete')
           || n.startsWith('_commitDelete') || n.startsWith('_flushPending')
           || n.startsWith('_deleteCycles') || n.startsWith('_deleteProfile') },
  { id: 'navigation', label: 'Navigation & Routing',
    test: n => n.startsWith('_navigate') || n.startsWith('_newAutomat') || n.startsWith('_dispatchSetup')
           || n.startsWith('_pref') || n.startsWith('_setPref') || n.startsWith('_reloadSetup') },
  { id: 'render', label: 'Core Render Pipeline',
    test: n => n === '_render' || n.startsWith('_scheduleRender') || n.startsWith('_html')
           || n.startsWith('_renderTab') || n.startsWith('_updateFontScale') || n.startsWith('_applyFontScale') },
  { id: 'settings', label: 'Settings Form & Persistence',
    test: n => n.startsWith('_saveSettings') || n.startsWith('_snapshot') || n.startsWith('_applyPending')
           || n.startsWith('_cascade') || n.startsWith('_conflictKeys') || n.startsWith('_settingsConflict')
           || n.startsWith('_wizInit') || n.startsWith('_applyExport') || n.startsWith('_applyImport') },
  { id: 'store', label: 'Community Store',
    test: n => n.startsWith('_store') || n.startsWith('_storeSearch') || n.startsWith('_storeApp')
           || n.startsWith('_shareable') || n.startsWith('_loadShareProf') || n.startsWith('_shareableBy') },
  { id: 'playground', label: 'Playground (Simulation)',
    test: n => n.startsWith('_pg') },
  { id: 'ml', label: 'ML Insights',
    test: n => n.startsWith('_mlQuality') || n.startsWith('_mlTrend') || n.startsWith('_htmlMl') },
  { id: 'canvas', label: 'Canvas Drawing',
    test: n => n.endsWith('Canvas') || n.endsWith('Graph') || n.endsWith('Chart')
           || n.startsWith('_draw') || n.startsWith('_paintCycle') },
  { id: 'wire', label: 'Event Wiring',
    test: n => n.startsWith('_wire') },
  { id: 'action', label: 'Action Dispatch',
    test: n => n === '_onAction' || n.startsWith('_onAct') },
  { id: 'modal', label: 'Modal Action Dispatch',
    test: n => n === '_onModalAction' || n.startsWith('_onMAct') || n.startsWith('_onMact') },
  { id: 'utils', label: 'Utilities & Helpers', test: () => true },
];

function groupFor(name) {
  for (const g of GROUPS) if (g.test(name)) return g.id;
  return 'utils';
}

const byGroup = Object.fromEntries(GROUPS.map(g => [g.id, []]));
for (const m of methods) byGroup[groupFor(m.name)].push(m);

const OVERSIZE = 100;
const oversized = methods.filter(m => m.size > OVERSIZE).sort((a, b) => b.size - a.size);

const now = new Date().toISOString().slice(0, 10);
const out = [
  `# Panel Navigation Map`,
  ``,
  `Auto-generated ${now} from \`www/ha-washdata-panel.js\` (${total} lines, ${methods.length} methods).`,
  `Regenerate: \`node devtools/gen_panel_map.mjs\``,
  ``,
  `Each entry: **Method name** — line number (method size in lines).`,
  `Methods >${OVERSIZE} lines are flagged ⚠ and summarised in the table at the bottom.`,
  ``,
  `---`,
  ``,
];

for (const g of GROUPS) {
  const members = byGroup[g.id];
  if (!members.length) continue;
  out.push(`## ${g.label}`);
  out.push(``);
  for (const m of members) {
    out.push(`- **${m.name}** — L${m.line} (${m.size} lines)${m.size > OVERSIZE ? ' ⚠' : ''}`);
  }
  out.push(``);
}

out.push(`---`, ``, `## Oversized Methods (>${OVERSIZE} lines)`, ``,
  `| Method | Line | Size | Group |`, `|--------|------|------|-------|`);
for (const m of oversized) {
  const g = GROUPS.find(g => g.id === groupFor(m.name));
  out.push(`| \`${m.name}\` | ${m.line} | ${m.size} | ${g?.label || '?'} |`);
}
out.push(``);

fs.writeFileSync(OUT, out.join('\n'));
console.log(`Written: ${OUT}`);
console.log(`  ${methods.length} methods, ${oversized.length} oversized (>${OVERSIZE} lines)`);
