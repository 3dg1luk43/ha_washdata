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
const PANEL_SRC = path.join(
  __dirname,
  '../custom_components/ha_washdata/www/ha-washdata-panel.js',
);
const TRANSLATIONS_SRC = path.join(
  __dirname,
  '../custom_components/ha_washdata/www/panel-translations.json',
);
const PORT = 4567;

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
  } else if (url.pathname === '/ha_washdata/panel-translations.json') {
    filePath = TRANSLATIONS_SRC;
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
  console.log(`WashData E2E fixture server listening on http://localhost:${PORT}`);
});
