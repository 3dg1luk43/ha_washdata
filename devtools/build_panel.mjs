#!/usr/bin/env node
/**
 * Build the shipped, minified panel/card assets from their readable sources.
 *
 * Why this exists: the panel source is a large, heavily commented single file
 * (that is deliberate -- it is what makes the thing maintainable). Users should
 * not have to download the comments. So the readable file stays the source of
 * truth in git, and this script produces a `.min.js` sibling next to it.
 *
 * The safety contract, which the integration and the test suite both enforce:
 *
 *   - build-manifest.json records the SHA-256 of every source file at build time.
 *   - frontend.py serves a `.min.js` ONLY when that recorded hash still matches
 *     the source on disk; otherwise it silently falls back to the readable
 *     source. So a forgotten rebuild can never ship stale code -- worst case
 *     users get the (correct, larger) unminified file.
 *   - tests/test_panel_build.py fails when an artifact is stale, so a forgotten
 *     rebuild is caught in CI rather than in production.
 *   - The E2E suite can run against the built artifacts (PANEL_BUILD=min), so a
 *     minified bundle that does not actually work cannot be released.
 *
 * Usage:
 *   node build_panel.mjs                 # build artifacts + manifest
 *   node build_panel.mjs --check         # verify artifacts are current; exit 1 if not
 *   node build_panel.mjs --check --www D # verify a different www/ (the pre-commit hook
 *                                        # points this at the STAGED snapshot, so a
 *                                        # rebuild that was never staged still fails)
 */

import { createHash } from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import * as esbuild from 'esbuild';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
// `--www <dir>` verifies an alternative asset directory. Only --check honours it: a
// build always writes the real tree. The pre-commit hook uses it to check the staged
// blobs rather than the working tree, which is the only way to catch the common miss -
// source rebuilt and staged, artifact rebuilt but left unstaged.
const wwwFlag = process.argv.indexOf('--www');
const WWW = wwwFlag !== -1 && process.argv[wwwFlag + 1]
  ? path.resolve(process.argv[wwwFlag + 1])
  : path.join(__dirname, '..', 'custom_components', 'ha_washdata', 'www');
const MANIFEST = path.join(WWW, 'build-manifest.json');

// Assets to minify. `source` is the readable file kept in git; `artifact` is the
// generated file the integration prefers to serve when it is current.
const TARGETS = [
  { source: 'ha-washdata-panel.js', artifact: 'ha-washdata-panel.min.js' },
  { source: 'ha-washdata-card.js', artifact: 'ha-washdata-card.min.js' },
];

// esbuild strips `//` comments, which would drop the AGPL notice from the shipped
// file. The licence has to travel with the distributed artifact, so it is
// re-attached as a banner (and points a reader at the readable source).
const LICENSE_BANNER = [
  '/*!',
  ' * WashData - Home Assistant integration for appliance cycle monitoring via smart plugs.',
  ' * Copyright (C) 2026 Lukas Bandura',
  ' * SPDX-License-Identifier: AGPL-3.0-or-later',
  ' *',
  ' * This file is a MINIFIED BUILD. The corresponding readable source ships in the',
  ' * same directory and is the licensed, preferred form for modification.',
  ' * This program comes with ABSOLUTELY NO WARRANTY. See the GNU Affero General',
  ' * Public License for details: <https://www.gnu.org/licenses/>.',
  ' */',
].join('\n');

const sha256 = (buf) => createHash('sha256').update(buf).digest('hex');

async function minify(sourcePath) {
  const code = fs.readFileSync(sourcePath, 'utf8');
  // target: 'esnext' means "minify only, never transpile". Downlevelling could
  // change semantics of code that is already validated by the test suite, and the
  // panel only has to run in the browsers Home Assistant's own frontend supports.
  const out = await esbuild.transform(code, {
    minify: true,
    target: 'esnext',
    format: 'esm',
    // Property/method names are never mangled by esbuild, which is what makes
    // this safe for a class with 270+ methods reached via `this.`.
    legalComments: 'none',
    sourcemap: false,
  });
  for (const w of out.warnings) {
    console.warn(`  ! esbuild warning: ${w.text}`);
  }
  return `${LICENSE_BANNER}\n${out.code}`;
}

function readManifest() {
  try {
    return JSON.parse(fs.readFileSync(MANIFEST, 'utf8'));
  } catch {
    return null;
  }
}

/** Is every artifact present and built from exactly the current sources? */
function verify(manifest) {
  const problems = [];
  if (!manifest || !manifest.assets) return ['build-manifest.json is missing or unreadable'];

  for (const t of TARGETS) {
    const entry = manifest.assets[t.source];
    if (!entry) {
      problems.push(`${t.source}: no manifest entry (never built)`);
      continue;
    }
    const srcPath = path.join(WWW, t.source);
    const artPath = path.join(WWW, t.artifact);
    if (!fs.existsSync(artPath)) {
      problems.push(`${t.artifact}: artifact missing`);
      continue;
    }
    const srcHash = sha256(fs.readFileSync(srcPath));
    if (srcHash !== entry.source_sha256) {
      problems.push(`${t.source}: source changed since last build (rebuild required)`);
    }
    const artHash = sha256(fs.readFileSync(artPath));
    if (artHash !== entry.artifact_sha256) {
      problems.push(`${t.artifact}: artifact was modified by hand (rebuild required)`);
    }
  }
  return problems;
}

async function build() {
  const assets = {};
  for (const t of TARGETS) {
    const srcPath = path.join(WWW, t.source);
    const artPath = path.join(WWW, t.artifact);
    if (!fs.existsSync(srcPath)) {
      console.error(`ERROR: source not found: ${srcPath}`);
      process.exitCode = 1;
      return;
    }
    const srcBuf = fs.readFileSync(srcPath);
    const code = await minify(srcPath);
    fs.writeFileSync(artPath, code);

    const artBuf = fs.readFileSync(artPath);
    assets[t.source] = {
      artifact: t.artifact,
      source_sha256: sha256(srcBuf),
      artifact_sha256: sha256(artBuf),
      source_bytes: srcBuf.length,
      artifact_bytes: artBuf.length,
    };
    const pct = (100 * (1 - artBuf.length / srcBuf.length)).toFixed(1);
    console.log(
      `  ${t.source} -> ${t.artifact}  ` +
        `${(srcBuf.length / 1024).toFixed(1)} KB -> ${(artBuf.length / 1024).toFixed(1)} KB (-${pct}%)`,
    );
  }

  const manifest = {
    // Informational only -- correctness is decided by the hashes below.
    generator: `esbuild ${esbuild.version}`,
    assets,
  };
  fs.writeFileSync(MANIFEST, `${JSON.stringify(manifest, null, 2)}\n`);
  console.log(`  manifest: ${path.relative(process.cwd(), MANIFEST)}`);
}

const checkOnly = process.argv.includes('--check');

if (checkOnly) {
  const problems = verify(readManifest());
  if (problems.length) {
    console.error('Panel build is NOT current:');
    for (const p of problems) console.error(`  - ${p}`);
    console.error('\nRun:  node devtools/build_panel.mjs');
    process.exit(1);
  }
  console.log('Panel build is current.');
} else {
  await build();
  const problems = verify(readManifest());
  if (problems.length) {
    console.error('Post-build verification FAILED:');
    for (const p of problems) console.error(`  - ${p}`);
    process.exit(1);
  }
  console.log('Build verified.');
}
