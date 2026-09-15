/**
 * Scoped-registry race: the card must survive Home Assistant replacing
 * window.customElements after this module already registered its elements
 * (issue #384, upstream home-assistant/frontend#53890).
 *
 * With `lovelace: resource_mode: yaml` the resource collection is read-only, so
 * the integration registers the card through add_extra_js_url() and the browser
 * starts it as an independent module import, racing HA's app bundle. The app
 * entry's first import is @webcomponents/scoped-custom-element-registry, which
 * REPLACES window.customElements with a shim that answers get()/whenDefined()
 * from its own map - so a card that won the race lost its registration, silently:
 * the module ran, window.customCards listed the card, and the only symptom was
 * create-element-base.ts's 2 s timeout ("Custom element not found").
 *
 * The fixture loads the genuine polyfill (served by serve.mjs from
 * node_modules) in either order via ?registry=, so both the failing and the
 * healthy production order are pinned here.
 */

import { test, expect } from '@playwright/test';

const RUN: Record<string, unknown> = {
  states: {
    'sensor.wm_state': { state: 'running', attributes: { sub_state: 'Running (Rinsing)', current_program_guess: 'Cotton 40' } },
    'sensor.wm_cycle_progress': { state: '64', attributes: {} },
    'sensor.wm_time_remaining': { state: '16', attributes: { unit_of_measurement: 'min' } },
  },
  entities: {
    'sensor.wm_state': { device_id: 'd1', platform: 'ha_washdata', translation_key: 'washer_state' },
    'sensor.wm_cycle_progress': { device_id: 'd1', platform: 'ha_washdata', translation_key: 'cycle_progress' },
    'sensor.wm_time_remaining': { device_id: 'd1', platform: 'ha_washdata', translation_key: 'time_remaining' },
  },
  devices: { d1: { name: 'Washing Machine' } },
};

// What the dashboard itself does before it renders a custom card
// (create-element-base.ts): look the tag up in the *current* registry, then wait
// on whenDefined for at most 2 s. Both have to succeed, so both are asserted.
async function dashboardCanCreate(page: any) {
  return page.evaluate(async () => {
    const defined = !!customElements.get('ha-washdata-card');
    const whenDefined = await Promise.race([
      customElements.whenDefined('ha-washdata-card').then(() => true),
      new Promise((resolve) => setTimeout(() => resolve(false), 2000)),
    ]);
    return { defined, whenDefined };
  });
}

for (const registry of ['after-card', 'before-card'] as const) {
  const when =
    registry === 'after-card'
      ? 'card module wins the race (extra-module path)'
      : 'polyfill installs first (UI-registry path)';

  test(`registry swap - ${when}: the card stays defined and renders`, async ({ page }) => {
    const pageErrors: string[] = [];
    page.on('pageerror', (err) => pageErrors.push(err.message));

    await page.goto(`/card.html?registry=${registry}`);
    await page.waitForFunction(() => (window as any).__ready === true, { timeout: 10_000 });

    // The definition made at module scope is visible before the swap either way.
    expect(await page.evaluate(() => (window as any).__afterCardImport)).toBe(true);

    // The self-heal runs on a 200 ms poll, so give it room; before the fix this
    // stayed undefined forever in the after-card order.
    await page.waitForFunction(() => !!customElements.get('ha-washdata-card'), { timeout: 5_000 });
    expect(await dashboardCanCreate(page)).toEqual({ defined: true, whenDefined: true });
    expect(await page.evaluate(() => !!customElements.get('ha-washdata-card-editor'))).toBe(true);

    // Re-registering must be rejected: exactly one registry may see the tag. A
    // second define succeeding is the signature of the orphaned-definition bug.
    const redefine = await page.evaluate(() => {
      try {
        customElements.define('ha-washdata-card', class extends HTMLElement {});
        return 'accepted';
      } catch {
        return 'rejected';
      }
    });
    expect(redefine).toBe('rejected');

    // Registered once in the picker, and it actually renders through the shim.
    expect(await page.evaluate(() => (window as any).customCards.filter((c: any) => c.type === 'ha-washdata-card').length)).toBe(1);
    await page.evaluate((d) => (window as any).__mountCard({ entity: 'sensor.wm_state', layout: 'tile' }, d), RUN);
    const vm = await page.evaluate(() => {
      const sr = (window as any).__card.shadowRoot;
      return { title: sr.getElementById('title').textContent, bar: sr.getElementById('barfill').style.width };
    });
    expect(vm.title).toBe('Washing Machine');
    expect(vm.bar).toBe('64%');

    // The original failure was silent; a noisy fix would be its own regression.
    expect(pageErrors).toEqual([]);
  });
}


// The editor is defined in the same loop as the card, but wdDefineElements
// catches per tag, so a define that throws for the editor alone is swallowed.
// The self-heal poll used to re-run only while the CARD was missing, so that
// failure was permanent and Lovelace fell back to the YAML editor for the rest
// of the session.
//
// No ?registry= here on purpose: with the polyfill loaded, its swap re-defines
// BOTH tags as a side effect and recovers the editor whether or not the poll
// checks for it, which hides the defect. Plain native registry, no swap.
test('a failed editor define is retried, not left undefined', async ({ page }) => {
  await page.addInitScript(() => {
    let failuresLeft = 1;
    const original = customElements.define.bind(customElements);
    customElements.define = (name: string, ctor: any, options?: any) => {
      if (name === 'ha-washdata-card-editor' && failuresLeft > 0) {
        failuresLeft -= 1;
        throw new Error('simulated one-off editor define failure');
      }
      return original(name, ctor, options);
    };
  });

  await page.goto('/card.html');
  await page.waitForFunction(() => (window as any).__ready === true, { timeout: 10_000 });

  // The card is unaffected: only the editor's define threw.
  expect(await page.evaluate(() => !!customElements.get('ha-washdata-card'))).toBe(true);

  // The poll has to come back for the editor. It runs every 200 ms; before the
  // fix this stayed undefined until the 30 s deadline.
  await page.waitForFunction(
    () => !!customElements.get('ha-washdata-card-editor'),
    { timeout: 5_000 },
  );
});
