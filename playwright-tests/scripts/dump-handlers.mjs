// Print the E2E suite's default WS mock responses as JSON.
//
// Loads helpers/ws-handlers.ts through Playwright's own TypeScript loader (the
// same transform the specs go through, so no extra toolchain) and writes
//   {"handlers": {"ha_washdata/<command>": <response>, ...}, "functions": [...]}
// to stdout. Function handlers cannot be serialised; they are listed by name.
//
// Consumed by tests/test_e2e_mock_contract.py, which validates each response
// against the backend's ws_schema contract.
//
//   node playwright-tests/scripts/dump-handlers.mjs

import { createRequire } from 'node:module';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
const { transform } = require('playwright/lib/common');

const mod = await transform.requireOrImport(path.join(here, '..', 'helpers', 'ws-handlers.ts'));
const handlers = {};
const functions = [];
for (const [command, value] of Object.entries(mod.DEFAULT_HANDLERS)) {
  if (typeof value === 'function') functions.push(command);
  else handlers[command] = value;
}
process.stdout.write(JSON.stringify({ handlers, functions }));
