/**
 * #530: the running cycle does not look like the program picked by hand.
 *
 * The backend's background check sets `program_mismatch` on the device (the pick,
 * the program it looks like, the margin). The Status card says so next to the
 * program selector; the pick itself stays selected. Display only. The
 * Automations tab offers the matching integration trigger.
 */

import { test, expect } from '@playwright/test';
import { bootPanel, clickTab } from '../helpers/panel';
import deviceRunning from '../fixtures/mock-data/device-running.json';

const ENTRY = 'test-entry-001';
const HASS = {
  devices: {
    wd: { id: 'wd-device', identifiers: [['ha_washdata', ENTRY]], config_entries: [ENTRY] },
  },
};

function running(fields: Record<string, unknown>) {
  const dev = JSON.parse(JSON.stringify(deviceRunning));
  Object.assign(dev.devices[0], fields);
  return dev;
}

const MISMATCH = { selected: 'Cotton 40°C', looks_like: 'Eco 60°C', margin: 0.31 };

test.beforeEach(async ({ page }) => {
  await page.goto('/');
});

test('a hand-picked program the cycle does not look like gets a badge', async ({ page }) => {
  await bootPanel(page, { 'ha_washdata/get_devices': running({
    manual_program: true, program_mismatch: MISMATCH,
  }) });
  const badge = page.locator('[data-testid="program-mismatch"]');
  await expect(badge).toBeVisible({ timeout: 8_000 });
  await expect(badge).toContainText('Looks more like Eco 60°C');
  // The tip names the pick and says nothing was changed.
  const tip = page.locator('.wd-prog-ctl .wd-tip').last();
  await expect(tip).toHaveAttribute('aria-label', /does not look like Cotton 40°C/);
  await expect(tip).toHaveAttribute('aria-label', /Nothing is changed/);
  // The pick stays selected and is still marked as the user's.
  await expect(page.locator('#wd-status-prog')).toHaveValue('Cotton 40°C');
  await expect(page.locator('.wd-prog-tag.manual')).toBeVisible();
});

test('no badge without a mismatch, or on an auto-detected program', async ({ page }) => {
  await bootPanel(page, { 'ha_washdata/get_devices': running({ manual_program: true }) });
  await expect(page.locator('#wd-status-prog')).toHaveValue('Cotton 40°C', { timeout: 8_000 });
  await expect(page.locator('[data-testid="program-mismatch"]')).toHaveCount(0);

  await page.goto('/');
  await bootPanel(page, { 'ha_washdata/get_devices': running({
    manual_program: false, program_mismatch: MISMATCH,
  }) });
  await expect(page.locator('#wd-status-prog')).toHaveValue('Cotton 40°C', { timeout: 8_000 });
  await expect(page.locator('[data-testid="program-mismatch"]')).toHaveCount(0);
});

test('no badge once the cycle is over', async ({ page }) => {
  await bootPanel(page, { 'ha_washdata/get_devices': running({
    detector_state: 'off', manual_program: true, program_mismatch: MISMATCH,
  }) });
  await expect(page.locator('#wd-status-prog')).toBeVisible({ timeout: 8_000 });
  await expect(page.locator('[data-testid="program-mismatch"]')).toHaveCount(0);
});

test('a slower device list sent before a program change does not bring the badge back', async ({ page }) => {
  await bootPanel(page, { 'ha_washdata/get_devices': running({ manual_program: true }) });
  await expect(page.locator('#wd-status-prog')).toHaveValue('Cotton 40°C', { timeout: 8_000 });
  const stale = running({ manual_program: true, program_mismatch: MISMATCH });
  const fresh = running({ manual_program: true });
  const shown = await page.evaluate(async ({ stale, fresh }) => {
    const el = document.querySelector('ha-washdata-panel') as any;
    const real = el._ws.bind(el);
    let release: () => void = () => {};
    const slow = new Promise<void>((r) => { release = r; });
    let n = 0;
    el._ws = (msg: any) => {
      if (msg.type !== 'ha_washdata/get_devices') return real(msg);
      n += 1;
      return n === 1 ? slow.then(() => stale) : Promise.resolve(fresh);
    };
    const older = el._fetchAll();   // a poll in flight when the program changed
    await el._fetchAll();           // the refresh after the change
    release();
    await older;
    return el._devices[el._selIdx].program_mismatch ?? null;
  }, { stale, fresh });
  expect(shown).toBeNull();
  await expect(page.locator('[data-testid="program-mismatch"]')).toHaveCount(0);
});

test('a slower expected curve for the previous program never shows under the new one', async ({ page }) => {
  await bootPanel(page, { 'ha_washdata/get_devices': running({ manual_program: true }) });
  await expect(page.locator('#wd-status-prog')).toHaveValue('Cotton 40°C', { timeout: 8_000 });
  const eco = running({ manual_program: true, current_program: 'Eco 60°C' });
  const shown = await page.evaluate(async ({ eco }) => {
    const el = document.querySelector('ha-washdata-panel') as any;
    const real = el._ws.bind(el);
    let release: () => void = () => {};
    const slow = new Promise<void>((r) => { release = r; });
    let asked: () => void = () => {};
    const cottonAsked = new Promise<void>((r) => { asked = r; });
    let devices: any = null;
    el._ws = (msg: any) => {
      if (msg.type === 'ha_washdata/get_devices' && devices) return Promise.resolve(devices);
      if (msg.type !== 'ha_washdata/get_profile_envelope') return real(msg);
      const answer = { envelope: { program: msg.profile_name } };
      if (msg.profile_name !== 'Cotton 40°C') return Promise.resolve(answer);
      asked();
      return slow.then(() => answer);
    };
    el._statusEnv = null; el._statusEnvName = null; el._statusEnvWant = null;
    const older = el._fetchAll();   // Cotton's curve in flight
    await cottonAsked;
    devices = eco;                  // the program changed to Eco
    await el._fetchAll();
    release();
    await older;
    await el._fetchAll();           // a later poll
    return { name: el._statusEnvName, env: el._statusEnv && el._statusEnv.program };
  }, { eco });
  expect(shown).toEqual({ name: 'Eco 60°C', env: 'Eco 60°C' });
});

// Two appliances running a program of the same name.
function twoRunning(fields: Record<string, unknown> = {}) {
  const two = running({ manual_program: true, ...fields });
  two.devices.push({ ...two.devices[0], entry_id: 'test-entry-002', title: 'Dryer' });
  return two;
}

test('the expected curve and phases are kept per appliance, not per program name', async ({ page }) => {
  await bootPanel(page, { 'ha_washdata/get_devices': running({ manual_program: true }) });
  await expect(page.locator('#wd-status-prog')).toHaveValue('Cotton 40°C', { timeout: 8_000 });
  const got = await page.evaluate(async ({ two }) => {
    const el = document.querySelector('ha-washdata-panel') as any;
    const real = el._ws.bind(el);
    const asked: string[] = [];
    el._ws = (msg: any) => {
      if (msg.type === 'ha_washdata/get_devices') return Promise.resolve(two);
      if (msg.type === 'ha_washdata/get_profile_envelope') {
        asked.push(msg.entry_id);
        return Promise.resolve({ envelope: { entry: msg.entry_id } });
      }
      if (msg.type === 'ha_washdata/get_profile_phases') {
        return Promise.resolve({ phases: [{ name: msg.entry_id, start: 0, end: 60 }] });
      }
      return real(msg);
    };
    el._statusEnv = null; el._statusEnvName = null; el._statusEnvEid = null; el._statusEnvWant = null;
    el._statusPhases = []; el._statusPhasesName = null; el._statusPhasesEid = null;
    await el._fetchAll();
    el._selIdx = 1;     // another appliance shown without a device switch (a ?device= link)
    await el._fetchAll();
    return { asked, env: el._statusEnv && el._statusEnv.entry, phase: el._statusPhases[0]?.name };
  }, { two: twoRunning() });
  expect(got).toEqual({
    asked: ['test-entry-001', 'test-entry-002'], env: 'test-entry-002', phase: 'test-entry-002',
  });
});

test('a failed expected-curve request is asked again on the next poll', async ({ page }) => {
  await bootPanel(page, { 'ha_washdata/get_devices': running({ manual_program: true }) });
  await expect(page.locator('#wd-status-prog')).toHaveValue('Cotton 40°C', { timeout: 8_000 });
  const got = await page.evaluate(async () => {
    const el = document.querySelector('ha-washdata-panel') as any;
    const real = el._ws.bind(el);
    let n = 0;
    el._ws = (msg: any) => {
      if (msg.type !== 'ha_washdata/get_profile_envelope') return real(msg);
      n += 1;
      return n === 1 ? Promise.reject(new Error('timeout')) : Promise.resolve({ envelope: null });
    };
    el._statusEnv = null; el._statusEnvName = null; el._statusEnvEid = null; el._statusEnvWant = null;
    await el._fetchAll();
    const afterFailure = { asked: n, name: el._statusEnvName };
    await el._fetchAll();   // asked again
    await el._fetchAll();   // an answer without an envelope is kept
    return { afterFailure, asked: n, name: el._statusEnvName };
  });
  expect(got).toEqual({ afterFailure: { asked: 1, name: null }, asked: 2, name: 'Cotton 40°C' });
});

test('a slower phase answer for a superseded program never replaces the current one', async ({ page }) => {
  await bootPanel(page, { 'ha_washdata/get_devices': running({ manual_program: true }) });
  await expect(page.locator('#wd-status-prog')).toHaveValue('Cotton 40°C', { timeout: 8_000 });
  const got = await page.evaluate(async () => {
    const el = document.querySelector('ha-washdata-panel') as any;
    const real = el._ws.bind(el);
    const releases: Array<() => void> = [];
    let failCotton = false;
    el._ws = (msg: any) => {
      if (msg.type !== 'ha_washdata/get_profile_phases') return real(msg);
      const answer = { phases: [{ name: msg.profile_name, start: 0, end: 60 }] };
      if (msg.profile_name !== 'Cotton 40°C') return Promise.resolve(answer);
      return new Promise((resolve, reject) => {
        releases.push(() => (failCotton ? reject(new Error('lost')) : resolve(answer)));
      });
    };
    const out: string[] = [];
    for (const fail of [false, true]) {
      failCotton = fail;
      el._statusPhases = []; el._statusPhasesName = null; el._statusPhasesEid = null;
      const older = el._ensureStatusPhases('test-entry-001', 'Cotton 40°C');
      await el._ensureStatusPhases('test-entry-001', 'Eco 60°C');   // the program changed
      releases.shift()!();
      await older;
      out.push(el._statusPhases.map((p: any) => p.name).join(','));
    }
    return out;
  });
  expect(got).toEqual(['Eco 60°C', 'Eco 60°C']);
});

test('the mismatch badge comes and goes in place while the program selector is open', async ({ page }) => {
  await bootPanel(page, { 'ha_washdata/get_devices': running({ manual_program: true }) });
  const sel = page.locator('#wd-status-prog');
  await expect(sel).toHaveValue('Cotton 40°C', { timeout: 8_000 });
  await sel.focus();
  const seen = await page.evaluate(async ({ on, off }) => {
    const el = document.querySelector('ha-washdata-panel') as any;
    const sr = el.shadowRoot;
    const real = el._ws.bind(el);
    let list = on;
    el._ws = (msg: any) => (msg.type === 'ha_washdata/get_devices' ? Promise.resolve(list) : real(msg));
    const before = sr.getElementById('wd-status-prog');
    const state = () => ({
      badge: sr.querySelectorAll('[data-testid="program-mismatch"]').length,
      same: sr.getElementById('wd-status-prog') === before,
      focused: sr.activeElement === before,
    });
    await el._fetchAll();
    const shown = state();
    list = off;
    await el._fetchAll();
    return { shown, cleared: state() };
  }, {
    on: running({ manual_program: true, program_mismatch: MISMATCH }),
    off: running({ manual_program: true }),
  });
  expect(seen).toEqual({
    shown: { badge: 1, same: true, focused: true },
    cleared: { badge: 0, same: true, focused: true },
  });
  await expect(page.locator('[data-testid="program-mismatch"]')).toHaveCount(0);
});

test('a slower power history for the previous appliance never lands under the new one', async ({ page }) => {
  await bootPanel(page, { 'ha_washdata/get_devices': running({ manual_program: true }) });
  await expect(page.locator('#wd-status-prog')).toHaveValue('Cotton 40°C', { timeout: 8_000 });
  const got = await page.evaluate(async ({ two }) => {
    const el = document.querySelector('ha-washdata-panel') as any;
    const real = el._ws.bind(el);
    let release: () => void = () => {};
    const slow = new Promise<void>((r) => { release = r; });
    let asked: () => void = () => {};
    const washerAsked = new Promise<void>((r) => { asked = r; });
    el._ws = (msg: any) => {
      if (msg.type === 'ha_washdata/get_devices') return Promise.resolve(two);
      if (msg.type !== 'ha_washdata/get_power_history') return real(msg);
      const answer = { live: [], raw: [], cycle_active: true, cycle_elapsed_s: 0, of: msg.entry_id };
      if (msg.entry_id !== 'test-entry-001') return Promise.resolve(answer);
      asked();
      return slow.then(() => answer);
    };
    let paints = 0;
    const finish = el._finishFetch.bind(el);
    el._finishFetch = (...a: unknown[]) => { paints += 1; return finish(...a); };
    el._selIdx = 0;
    const older = el._fetchAll();   // the washer's history waits on the recorder
    await washerAsked;
    el._selIdx = 1;                 // the user is on the dryer now
    await el._fetchAll();
    const paintsBefore = paints;
    release();
    await older;
    return { of: el._powerData.of, paintsBefore, paints };
  }, { two: twoRunning() });
  expect(got).toEqual({ of: 'test-entry-002', paintsBefore: 1, paints: 1 });
});

test('a superseded refresh leaves the first paint to the newer one', async ({ page }) => {
  await bootPanel(page, { 'ha_washdata/get_devices': running({ manual_program: true }) });
  await expect(page.locator('#wd-status-prog')).toHaveValue('Cotton 40°C', { timeout: 8_000 });
  const fresh = running({ manual_program: true });
  const loading = await page.evaluate(async ({ fresh }) => {
    const el = document.querySelector('ha-washdata-panel') as any;
    const real = el._ws.bind(el);
    let releaseList: () => void = () => {};
    const slowList = new Promise<void>((r) => { releaseList = r; });
    let releaseData: () => void = () => {};
    const slowData = new Promise<void>((r) => { releaseData = r; });
    let n = 0;
    el._ws = (msg: any) => {
      if (msg.type === 'ha_washdata/get_power_history') return slowData.then(() => real(msg));
      if (msg.type !== 'ha_washdata/get_devices') return real(msg);
      n += 1;
      return n === 1 ? slowList.then(() => fresh) : Promise.resolve(fresh);
    };
    el._loading = true;               // a first load
    const older = el._fetchAll();
    const newer = el._fetchAll();     // its list is in, its live data is not
    await new Promise((r) => setTimeout(r, 50));
    releaseList();
    await older;
    const whileNewerLoads = el._loading;
    releaseData();
    await newer;
    return { whileNewerLoads, after: el._loading };
  }, { fresh });
  expect(loading).toEqual({ whileNewerLoads: true, after: false });
});

test('a stale device-list error after a newer list is not a lost connection', async ({ page }) => {
  await bootPanel(page, { 'ha_washdata/get_devices': running({ manual_program: true }) });
  await expect(page.locator('#wd-status-prog')).toHaveValue('Cotton 40°C', { timeout: 8_000 });
  const fresh = running({ manual_program: true });
  const failed = await page.evaluate(async ({ fresh }) => {
    const el = document.querySelector('ha-washdata-panel') as any;
    const real = el._ws.bind(el);
    let release: () => void = () => {};
    const slow = new Promise<void>((r) => { release = r; });
    let n = 0;
    el._ws = (msg: any) => {
      if (msg.type !== 'ha_washdata/get_devices') return real(msg);
      n += 1;
      return n === 1
        ? slow.then(() => { throw new Error('connection lost'); })
        : Promise.resolve(fresh);
    };
    const older = el._fetchAll();
    await el._fetchAll();
    release();
    await older;
    return el._fetchFailed;
  }, { fresh });
  expect(failed).toBe(false);
  await expect(page.locator('#wd-status-prog')).toHaveValue('Cotton 40°C');
});

test('the program it looks like renders as text, never as markup', async ({ page }) => {
  await bootPanel(page, { 'ha_washdata/get_devices': running({
    manual_program: true,
    program_mismatch: { ...MISMATCH, looks_like: '<img src=x id=wd-xss>' },
  }) });
  await expect(page.locator('[data-testid="program-mismatch"]'))
    .toContainText('<img src=x id=wd-xss>', { timeout: 8_000 });
  await expect(page.locator('#wd-xss')).toHaveCount(0);
});

test('the Automations tab starts an automation from the trigger', async ({ page }) => {
  await bootPanel(page, {
    'ha_washdata/blueprints/list': { blueprints: [] },
    'config/label_registry/list': [],
    'config/category_registry/list': [],
  }, HASS);
  await page.evaluate(() => {
    const el: any = document.querySelector('ha-washdata-panel');
    (window as any).__api = [];
    el._hass.callApi = (method: string, path: string, body: unknown) => {
      (window as any).__api.push({ method, path, body });
      return Promise.resolve({});
    };
  });
  await clickTab(page, 'automations');
  await page.locator('[data-action="auto-toggle-triggers"]').click();
  const item = page.locator('[data-action="auto-new-trigger"][data-trigger="program_mismatch"]');
  await expect(item).toHaveText('Running program differs from the selected one');
  await item.click();
  await expect.poll(() => page.evaluate(() => (window as any).__api.length)).toBeGreaterThan(0);
  const [call] = await page.evaluate(() => (window as any).__api);
  expect(call.body.triggers).toEqual([
    { trigger: 'ha_washdata.program_mismatch', target: { device_id: ['wd-device'] }, options: {} },
  ]);
});

test('a history read slower than the poll still lands and paints', async ({ page }) => {
  await bootPanel(page, { 'ha_washdata/get_devices': running({ manual_program: true }) });
  await expect(page.locator('#wd-status-prog')).toHaveValue('Cotton 40°C', { timeout: 8_000 });
  const got = await page.evaluate(async () => {
    const el = document.querySelector('ha-washdata-panel') as any;
    const real = el._ws.bind(el);
    const gates: Array<() => void> = [];
    const asked: Array<() => void> = [];
    const askedP = [0, 1, 2].map((i) => new Promise<void>((r) => { asked[i] = r; }));
    let n = 0;
    el._ws = (msg: any) => {
      if (msg.type !== 'ha_washdata/get_power_history') return real(msg);
      const i = n++;
      const open = new Promise<void>((r) => { gates[i] = r; });
      asked[i]();
      return open.then(() => ({ live: [], raw: [], cycle_active: true, cycle_elapsed_s: 0, n: i }));
    };
    let paints = 0;
    const finish = el._finishFetch.bind(el);
    el._finishFetch = (...a: unknown[]) => { paints += 1; return finish(...a); };
    const first = el._fetchAll();
    await askedP[0];
    const second = el._fetchAll();   // the next poll's list is in, its read is not
    await askedP[1];
    gates[0]();
    await first;
    const landed = { n: el._powerData.n, paints };
    const third = el._fetchAll();
    await askedP[2];
    gates[2]();
    await third;
    gates[1]();                       // older than what already landed
    await second;
    return { landed, after: el._powerData.n, paints };
  });
  expect(got).toEqual({ landed: { n: 0, paints: 1 }, after: 2, paints: 2 });
});

test('a refresh whose appliance changed under it does not claim the phase slot', async ({ page }) => {
  await bootPanel(page, { 'ha_washdata/get_devices': running({ manual_program: true }) });
  await expect(page.locator('#wd-status-prog')).toHaveValue('Cotton 40°C', { timeout: 8_000 });
  const calls = await page.evaluate(async ({ two }) => {
    const el = document.querySelector('ha-washdata-panel') as any;
    const real = el._ws.bind(el);
    let release: () => void = () => {};
    const slow = new Promise<void>((r) => { release = r; });
    let asked: () => void = () => {};
    const envAsked = new Promise<void>((r) => { asked = r; });
    el._ws = (msg: any) => {
      if (msg.type === 'ha_washdata/get_devices') return Promise.resolve(two);
      if (msg.type === 'ha_washdata/get_profile_envelope' && msg.entry_id === 'test-entry-001') {
        asked();
        return slow.then(() => real(msg));
      }
      return real(msg);
    };
    const seen: string[] = [];
    const ensure = el._ensureStatusPhases.bind(el);
    el._ensureStatusPhases = (eid: string, prog: string) => { seen.push(eid); return ensure(eid, prog); };
    el._statusEnvEid = null; el._statusEnvName = null; el._statusEnvWant = null;
    el._selIdx = 0;
    const older = el._fetchAll();
    await envAsked;
    el._selIdx = 1;                   // the user is on the dryer now
    release();
    await older;
    return seen;
  }, { two: twoRunning() });
  expect(calls).not.toContain('test-entry-001');
});
