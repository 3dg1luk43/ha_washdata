/**
 * Issue #504: text and knobs on a primary-coloured surface ignored the theme.
 *
 * A theme with a light accent (white --primary-color on a dark theme) made every
 * active chip, primary button, entity pill and checked toggle white-on-white,
 * because they painted a hardcoded white. HA's convention for text on
 * --primary-color is --text-primary-color, so the panel's --wd-on-primary follows
 * it; the status fills follow HA's --ha-color-on-*-loud tokens.
 *
 * Two properties are locked here:
 *  1. Under default theme variables nothing moves: every colour is what it was
 *     before the fix (with the variables unset, as in this fixture, and with HA's
 *     own default declarations).
 *  2. Under a light accent the text takes --text-primary-color, the toggle tracks
 *     take --switch-checked-track-color (else the primary colour) and the knobs the
 *     on-primary colour. The knob deliberately ignores --switch-checked-button-color:
 *     legacy themes set it to the accent, which hid the knob in the track (31444485).
 */

import { test, expect } from '@playwright/test';
import type { Page } from '@playwright/test';
import { bootPanel, clickTab } from '../helpers/panel';
import options from '../fixtures/mock-data/options.json';

const WHITE = [255, 255, 255, 1];
const PRIMARY = [3, 169, 244, 1]; // the fixture's --primary-color

// Probe markup built from the panel's own class names: the shadow root's stylesheet
// styles them exactly as it styles the real controls, including states (active,
// checked) that this fixture never renders on its own.
const PROBES = `
  <button class="wd-chip active" data-p="chip">x</button>
  <button class="wd-btn wd-btn-primary" data-p="btn">x</button>
  <button class="wd-sec-btn active" data-p="sec">x</button>
  <button class="wd-pg-pub" data-p="pub">x</button>
  <div class="wd-pillbox"><span class="wd-pill" data-p="pill"><span class="wd-pill-txt">x</span><button class="wd-pill-x" data-p="pillx">x</button></span></div>
  <span class="wd-switch"><input type="checkbox" checked><span class="wd-switch-slider" data-p="switch"></span></span>
  <span class="wd-toggle-track"><input type="checkbox" checked><span class="wd-toggle-knob" data-p="toggle"></span></span>
  <button class="wd-btn wd-btn-danger" data-p="danger">x</button>
  <span class="wd-dbadge rec" data-p="rec">x</span>
  <button class="wd-sug-use" data-p="suguse">x</button>
  <div class="wd-toast-success" data-p="tsuccess">x</div>
  <div class="wd-toast-error" data-p="terror">x</div>
  <div class="wd-toast-warning" data-p="twarning">x</div>
  <div class="wd-toast-info" data-p="tinfo">x</div>
  <span data-p="resolve"></span>
`;

// :hover rules cannot be held reliably on the touch project, so their declared
// value is resolved on a probe in the same shadow root instead.
const HOVER_RULES: Array<[string, string]> = [
  ['.wd-addbtn:hover', 'color'],
  // The shorthand: a longhand of a shorthand holding var() reads back empty.
  ['.wd-pill-x:hover', 'background'],
  ['.wd-auto-del:hover', 'color'],
  ['.wd-conflict-fix:hover', 'color'],
];

type Colours = Record<string, string>;

async function readColours(page: Page): Promise<Colours> {
  return page.evaluate(
    ({ html, hover }: { html: string; hover: Array<[string, string]> }) => {
      const sr = document.getElementById('wd-panel')!.shadowRoot!;
      // Created fresh per read, so no transition is mid-flight.
      const box = document.createElement('div');
      box.style.cssText = 'position:fixed;left:0;top:0;z-index:99999';
      box.innerHTML = html;
      sr.appendChild(box);
      const q = (p: string) => box.querySelector(`[data-p="${p}"]`) as HTMLElement;
      const out: Record<string, string> = {};
      for (const p of ['chip', 'btn', 'sec', 'pub', 'pill', 'pillx', 'danger', 'rec', 'suguse',
        'tsuccess', 'terror', 'twarning', 'tinfo']) {
        out[`${p}.color`] = getComputedStyle(q(p)).color;
      }
      out['pillx.bg'] = getComputedStyle(q('pillx')).backgroundColor;
      out['switch.track'] = getComputedStyle(q('switch')).backgroundColor;
      out['switch.knob'] = getComputedStyle(q('switch'), '::before').backgroundColor;
      out['toggle.track'] = getComputedStyle(q('toggle')).backgroundColor;
      out['toggle.knob'] = getComputedStyle(q('toggle'), '::after').backgroundColor;

      const rules: CSSStyleRule[] = [];
      for (const sheet of Array.from(sr.styleSheets)) {
        for (const r of Array.from(sheet.cssRules)) if (r instanceof CSSStyleRule) rules.push(r);
      }
      const probe = q('resolve');
      for (const [sel, prop] of hover) {
        const rule = rules.find((r) => r.selectorText === sel);
        const declared = rule ? rule.style.getPropertyValue(prop) : '';
        probe.style.setProperty('color', declared || 'rgb(1, 2, 3)');
        out[sel] = declared ? getComputedStyle(probe).color : `no ${prop} in rule ${sel}`;
      }
      box.remove();
      return out;
    },
    { html: PROBES, hover: HOVER_RULES },
  );
}

// rgb()/rgba() and the color(srgb ...) form color-mix() serializes to, as 0-255 + alpha.
function rgba(c: string): number[] {
  let m = c.match(/^rgba?\(([^)]+)\)$/);
  if (m) {
    const p = m[1].split(/[\s,/]+/).filter(Boolean).map(Number);
    return [p[0], p[1], p[2], p.length > 3 ? +p[3].toFixed(3) : 1];
  }
  m = c.match(/^color\(srgb ([^)]+)\)$/);
  if (m) {
    const p = m[1].split(/[\s/]+/).filter(Boolean).map(Number);
    return [...p.slice(0, 3).map((x) => Math.round(x * 255)), p.length > 3 ? +p[3].toFixed(3) : 1];
  }
  throw new Error(`unparseable colour: ${c}`);
}

function expectColours(got: Colours, want: Record<string, number[]>) {
  const actual = Object.fromEntries(Object.keys(want).map((k) => [k, got[k] && rgba(got[k])]));
  expect(actual).toEqual(want);
}

// Pre-#504 values under the fixture's default variables.
const DEFAULT_WANT: Record<string, number[]> = {
  'chip.color': WHITE, 'btn.color': WHITE, 'sec.color': WHITE, 'pub.color': WHITE,
  'pill.color': WHITE, 'pillx.color': WHITE, 'pillx.bg': [255, 255, 255, 0.25],
  'switch.track': PRIMARY, 'switch.knob': WHITE,
  'toggle.track': PRIMARY, 'toggle.knob': WHITE,
  'danger.color': WHITE, 'rec.color': WHITE, 'suguse.color': WHITE,
  'tsuccess.color': WHITE, 'terror.color': WHITE, 'twarning.color': WHITE, 'tinfo.color': WHITE,
  '.wd-addbtn:hover': WHITE, '.wd-pill-x:hover': [255, 255, 255, 0.45],
  '.wd-auto-del:hover': WHITE, '.wd-conflict-fix:hover': WHITE,
};

// HA's own declarations for the variables the fix reads (frontend 20260930).
const HA_DEFAULTS =
  '--text-primary-color: #fff; --ha-color-on-danger-loud: #fff; ' +
  '--ha-color-on-warning-loud: #fff; --ha-color-on-success-loud: #fff;';

async function openNotificationSettings(page: Page) {
  await bootPanel(page, {
    'ha_washdata/get_options': {
      options: { ...options, notify_finish_services: ['notify.mobile_app_phone'] }, defaults: {},
    },
  }, {}, { settingsLevel: 'advanced' });
  await clickTab(page, 'settings');
  await page.locator('button[data-sec="notifications"]').first().click();
  await expect(page.locator('.wd-pillbox[data-opt="notify_finish_services"] .wd-pill')).toBeVisible({ timeout: 8_000 });
}

// The real rendered controls, so the probes cannot drift from the markup.
async function readRendered(page: Page): Promise<Colours> {
  return page.evaluate(() => {
    const sr = document.getElementById('wd-panel')!.shadowRoot!;
    const el = (s: string) => {
      const e = sr.querySelector(s);
      if (!e) throw new Error(`not rendered: ${s}`);
      return e;
    };
    const knob = el('#wd-settings-level-chk + .wd-toggle-knob');
    return {
      'save.color': getComputedStyle(el('#wd-settings-save')).color,
      'sec.color': getComputedStyle(el('.wd-sec-btn.active')).color,
      'pill.color': getComputedStyle(el('.wd-pillbox .wd-pill')).color,
      'pillx.color': getComputedStyle(el('.wd-pillbox .wd-pill-x')).color,
      'level.track': getComputedStyle(knob).backgroundColor,
      'level.knob': getComputedStyle(knob, '::after').backgroundColor,
    };
  });
}

for (const [name, decl] of [['unset', ''], ['HA default declarations', HA_DEFAULTS]]) {
  test(`default theme (${name}): every on-primary and status colour is unchanged`, async ({ page }) => {
    await page.goto('/');
    if (decl) await page.addStyleTag({ content: `:root, ha-washdata-panel { ${decl} }` });
    await openNotificationSettings(page);

    expectColours(await readColours(page), DEFAULT_WANT);
    expectColours(await readRendered(page), {
      'save.color': WHITE, 'sec.color': WHITE, 'pill.color': WHITE, 'pillx.color': WHITE,
      'level.track': PRIMARY, 'level.knob': WHITE,
    });
  });
}

test('a light accent theme: text and knobs take --text-primary-color, tracks the switch variable', async ({ page }) => {
  const DARK = [28, 28, 28, 1];
  const TRACK = [224, 224, 224, 1];
  await page.goto('/');
  // Set on the panel host, as a theme applied to the panel would be. The button
  // colour is the accent, the legacy-theme value the knob must not follow.
  await page.addStyleTag({
    content:
      'ha-washdata-panel { --primary-color: #ffffff; --text-primary-color: #1c1c1c; ' +
      '--switch-checked-track-color: #e0e0e0; --switch-checked-button-color: #ffffff; ' +
      '--ha-color-on-danger-loud: #101010; --ha-color-on-warning-loud: #202020; ' +
      '--ha-color-on-success-loud: #303030; }',
  });
  await openNotificationSettings(page);

  const ERR = [16, 16, 16, 1];
  const WARN = [32, 32, 32, 1];
  const OK = [48, 48, 48, 1];
  expectColours(await readColours(page), {
    'chip.color': DARK, 'btn.color': DARK, 'sec.color': DARK, 'pub.color': DARK,
    'pill.color': DARK, 'pillx.color': DARK, 'pillx.bg': [28, 28, 28, 0.25],
    'switch.track': TRACK, 'switch.knob': DARK,
    'toggle.track': TRACK, 'toggle.knob': DARK,
    'danger.color': ERR, 'rec.color': ERR, 'suguse.color': WARN,
    'tsuccess.color': OK, 'terror.color': ERR, 'twarning.color': WARN,
    // No HA on-info token: the info toast keeps its white.
    'tinfo.color': WHITE,
    '.wd-addbtn:hover': DARK, '.wd-pill-x:hover': [28, 28, 28, 0.45],
    '.wd-auto-del:hover': ERR, '.wd-conflict-fix:hover': ERR,
  });
  expectColours(await readRendered(page), {
    'save.color': DARK, 'sec.color': DARK, 'pill.color': DARK, 'pillx.color': DARK,
    'level.track': TRACK, 'level.knob': DARK,
  });

  // No --switch-checked-track-color (HA itself no longer declares it): both tracks
  // fall back to the primary colour, and the knob still reads on it.
  await page.addStyleTag({ content: 'ha-washdata-panel { --switch-checked-track-color: initial; }' });
  const got = await readColours(page);
  expectColours(got, {
    'switch.track': WHITE, 'switch.knob': DARK, 'toggle.track': WHITE, 'toggle.knob': DARK,
  });
});
