/**
 * Discussion #461: custom maintenance tasks, one-click "Log done", device-type
 * presets, and the pointer to the external Maintenance Supporter integration.
 */

import { test, expect } from '@playwright/test';
import { bootPanel, clickTab, assertWsCalled, assertWsNotCalled } from '../helpers/panel';

const TASK = {
  id: 'custom_aaaa000001', name: 'Water filter', cycles: 40, days: 180,
  since: '2026-09-01T00:00:00+00:00', since_cycle_count: 10,
};

const DISHWASHER_LOG = {
  log: [
    { id: 'm1', date: '2026-09-20T09:00:00+00:00', event_type: 'custom_gone', task_name: 'Old reminder', notes: '' },
  ],
  due: ['salt'],
  event_types: ['salt', 'rinse_aid', 'filter_clean', 'descale', 'other'],
  reminders: { salt: 30, rinse_aid: 40, filter_clean: 50 },
  cycles_since: { salt: 31, rinse_aid: 31, filter_clean: 41 },
  custom_tasks: [TASK],
  status: [
    { id: 'salt', custom: false, name: null, cycles_interval: 30, days_interval: 0, cycles_since: 31, days_since: 20, due: true },
    { id: 'rinse_aid', custom: false, name: null, cycles_interval: 40, days_interval: 0, cycles_since: 31, days_since: 20, due: false },
    { id: 'filter_clean', custom: false, name: null, cycles_interval: 50, days_interval: 0, cycles_since: 41, days_since: null, due: false },
    { id: TASK.id, custom: true, name: 'Water filter', cycles_interval: 40, days_interval: 180, cycles_since: 5, days_since: 34, due: false },
  ],
  lifetime_cycle_count: 41,
  limits: { tasks_max: 20, name_max: 60, cycles_max: 100000, days_max: 3650 },
};

const WRITES = {
  'ha_washdata/get_maintenance_log': DISHWASHER_LOG,
  'ha_washdata/add_maintenance_task': { success: true, task: { ...TASK, id: 'custom_bbbb000002', name: 'Door seal', cycles: 0, days: 30 } },
  'ha_washdata/update_maintenance_task': { success: true, task: TASK },
  'ha_washdata/delete_maintenance_task': { success: true },
  'ha_washdata/add_maintenance_event': { success: true, event: { id: 'm2', date: '2026-10-05T09:00:00+00:00', event_type: 'salt', notes: '', cycle_count_at_log: 41 } },
};

async function openMaintenance(page) {
  await clickTab(page, 'history');
  const sub = page.locator('[data-hsub="maintenance"]').first();
  await expect(sub).toBeVisible({ timeout: 8_000 });
  await sub.click();
  await expect(page.locator('button[data-action="maint-save-reminders"]')).toBeVisible({ timeout: 8_000 });
}

test.beforeEach(async ({ page }) => {
  await page.goto('/');
  await bootPanel(page, WRITES);
});

test('a due task is logged done in one click', async ({ page }) => {
  await openMaintenance(page);
  const banner = page.locator('.wd-maint-due');
  await expect(banner).toContainText('Maintenance due: Refill salt');
  await banner.locator('button[data-action="maint-log-done"][data-mtype="salt"]').click();
  const calls = await assertWsCalled(page, 'ha_washdata/add_maintenance_event');
  expect(calls[0]).toMatchObject({ event_type: 'salt' });
  await expect(page.locator('.wd-toast', { hasText: 'Logged as done: Refill salt' }).first()).toBeVisible({ timeout: 5_000 });
});

test('device-type presets show as editor rows', async ({ page }) => {
  await openMaintenance(page);
  await expect(page.locator('[data-maint-rem="salt"]')).toHaveValue('30');
  await expect(page.locator('[data-maint-rem="rinse_aid"]')).toHaveValue('40');
  // A washer-only task is not offered on a dishwasher.
  await expect(page.locator('[data-maint-rem="drum_clean"]')).toHaveCount(0);
});

test('custom tasks show progress by cycles and by days', async ({ page }) => {
  await openMaintenance(page);
  const row = page.locator(`[data-mstatus="${TASK.id}"]`);
  await expect(row).toContainText('Water filter');
  await expect(row).toContainText('5 / 40 cycles');
  await expect(row).toContainText('34 / 180 days');
  // A removed task's log entry still reads, by the name it was logged with.
  await expect(page.locator('.wd-card', { hasText: 'Maintenance Log' }).first()).toContainText('Old reminder');
  // Custom tasks can be logged from the add-event form too.
  await expect(page.locator(`#wd-maint-type option[value="${TASK.id}"]`)).toHaveText('Water filter');
});

test('add a custom task', async ({ page }) => {
  await openMaintenance(page);
  await page.locator('#wd-mtask-new-name').fill('Door seal');
  await page.locator('#wd-mtask-new-days').fill('30');
  await page.locator('button[data-action="maint-task-add"]').click();
  const calls = await assertWsCalled(page, 'ha_washdata/add_maintenance_task');
  expect(calls[0]).toMatchObject({ name: 'Door seal', days: 30 });
  expect(calls[0]).not.toHaveProperty('cycles');
  await expect(page.locator('.wd-toast', { hasText: 'Task added' }).first()).toBeVisible({ timeout: 5_000 });
});

test('a custom task needs a name', async ({ page }) => {
  await openMaintenance(page);
  await page.locator('button[data-action="maint-task-add"]').click();
  await expect(page.locator('.wd-toast', { hasText: 'Give the task a name' }).first()).toBeVisible({ timeout: 5_000 });
  await assertWsNotCalled(page, 'ha_washdata/add_maintenance_task');
});

test('rename a custom task', async ({ page }) => {
  await openMaintenance(page);
  await page.locator(`[data-mtask-name="${TASK.id}"]`).fill('Fridge water filter');
  await page.locator('button[data-action="maint-save-reminders"]').click();
  const calls = await assertWsCalled(page, 'ha_washdata/update_maintenance_task');
  expect(calls[0]).toMatchObject({ task_id: TASK.id, name: 'Fridge water filter' });
  // Only what changed is sent; the built-in rows were untouched, so no reload.
  expect(calls[0]).not.toHaveProperty('cycles');
  await assertWsNotCalled(page, 'ha_washdata/set_options');
});

test('a changed built-in interval is still saved through set_options', async ({ page }) => {
  await openMaintenance(page);
  await page.locator('[data-maint-rem="salt"]').fill('0');
  await page.locator('button[data-action="maint-save-reminders"]').click();
  const calls = await assertWsCalled(page, 'ha_washdata/set_options');
  expect((calls[0] as any).options.maintenance_reminder_cycles).toMatchObject({ salt: 0, rinse_aid: 40, filter_clean: 50 });
});

test('remove a custom task', async ({ page }) => {
  await openMaintenance(page);
  await page.locator(`button[data-action="maint-task-remove"][data-mtask="${TASK.id}"]`).click();
  const modal = page.locator('.wd-modal');
  await expect(modal).toContainText('Water filter');
  await modal.locator('button[data-maction="ok"]').click();
  const calls = await assertWsCalled(page, 'ha_washdata/delete_maintenance_task');
  expect(calls[0]).toMatchObject({ task_id: TASK.id });
});

test('custom task names are shown as typed, never as markup', async ({ page }) => {
  const evil = '<img src=x onerror="window.__pwned=1">';
  await page.evaluate((name) => {
    const w = window as any;
    const log = JSON.parse(JSON.stringify(w.__ws_handlers['ha_washdata/get_maintenance_log']));
    log.custom_tasks[0].name = name;
    log.status[3].name = name;
    w.__ws_handlers['ha_washdata/get_maintenance_log'] = log;
  }, evil);
  await openMaintenance(page);
  await expect(page.locator(`[data-mstatus="${TASK.id}"]`)).toContainText(evil);
  await expect(page.locator(`[data-mtask-name="${TASK.id}"]`)).toHaveValue(evil);
  expect(await page.evaluate(() => (window as any).__pwned)).toBeUndefined();
});

test('links to the Maintenance Supporter integration', async ({ page }) => {
  await openMaintenance(page);
  const card = page.locator('.wd-maint-supporter');
  await expect(card).toContainText('Maintenance Supporter');
  await expect(card).toContainText('cycle counter');
  const link = card.locator('a');
  await expect(link).toHaveAttribute('href', 'https://github.com/iluebbe/maintenance_supporter');
  await expect(link).toHaveAttribute('target', '_blank');
  await expect(link).toHaveAttribute('rel', /noopener/);
});
