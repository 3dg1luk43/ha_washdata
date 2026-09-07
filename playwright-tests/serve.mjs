/**
 * Minimal static server for Playwright fixtures.
 * Serves fixtures/ as root; resolves panel.js from the source tree.
 * Eliminates the need for symlinks or copying the panel JS file.
 */
import http from 'http';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const FIXTURES_DIR = path.join(__dirname, 'fixtures');
// Set PANEL_BUILD=min to run the suite against the minified build artifacts
// instead of the readable sources. This is what makes a broken minified bundle
// impossible to ship: the same 340 tests have to pass against the exact bytes
// users will download.
const WWW = path.join(__dirname, '../custom_components/ha_washdata/www');
const USE_MIN = process.env.PANEL_BUILD === 'min';
const PANEL_SRC = path.join(
  WWW,
  USE_MIN ? 'ha-washdata-panel.min.js' : 'ha-washdata-panel.js',
);
const CARD_SRC = path.join(
  WWW,
  USE_MIN ? 'ha-washdata-card.min.js' : 'ha-washdata-card.js',
);
// Per-language panel translations are served straight from the integration's
// translations/panel/ directory (one {lang}.json per language), matching how
// the integration registers them at runtime.
const TRANSLATIONS_DIR = path.join(
  __dirname,
  '../custom_components/ha_washdata/translations/panel',
);
// The real @webcomponents/scoped-custom-element-registry build that the Home
// Assistant frontend loads as the first import of its app entry. Served so the
// card fixture can reproduce either load order against the genuine polyfill
// instead of a hand-written stand-in (issue #384).
const SCOPED_REGISTRY_SRC = path.join(
  __dirname,
  'node_modules/@webcomponents/scoped-custom-element-registry',
  'scoped-custom-element-registry.min.js',
);
// playwright.config.ts passes PORT via the webServer.env option so the two
// build modes (readable=4567, minified=4568) never share a server instance.
const PORT = parseInt(process.env.PORT ?? '4567', 10);

const MIME = {
  '.html': 'text/html',
  '.js': 'application/javascript',
  '.json': 'application/json',
  '.css': 'text/css',
};

const server = http.createServer((req, res) => {
  const url = new URL(req.url, `http://localhost:${PORT}`);
  let filePath;

  if (url.pathname === '/panel.js') {
    filePath = PANEL_SRC;
  } else if (url.pathname === '/card.js') {
    filePath = CARD_SRC;
  } else if (url.pathname === '/scoped-custom-element-registry.js') {
    filePath = SCOPED_REGISTRY_SRC;
  } else if (url.pathname.startsWith('/ha_washdata/panel-translations/')) {
    // /ha_washdata/panel-translations/{lang}.json -> translations/panel/{lang}.json
    const name = path.basename(url.pathname);
    filePath = path.join(TRANSLATIONS_DIR, name);
  } else {
    // Serve from fixtures/; default to index.html
    const rel = url.pathname === '/' ? 'index.html' : url.pathname.slice(1);
    filePath = path.join(FIXTURES_DIR, rel);
  }

  fs.readFile(filePath, (err, data) => {
    if (err) {
      res.writeHead(404, { 'Content-Type': 'text/plain' });
      res.end(`Not found: ${filePath}`);
      return;
    }
    const ext = path.extname(filePath);
    res.writeHead(200, {
      'Content-Type': MIME[ext] || 'application/octet-stream',
      'Access-Control-Allow-Origin': '*',
    });
    res.end(data);
  });
});

server.listen(PORT, () => {
  console.log(
    `WashData E2E fixture server listening on http://localhost:${PORT} ` +
      `(serving ${USE_MIN ? 'MINIFIED build artifacts' : 'readable sources'})`,
  );
});
