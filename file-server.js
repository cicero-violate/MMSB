const http = require('http');
const fs = require('fs');
const path = require('path');

const PORT = 8888;
const ROOT_DIR = __dirname;

const URL_PREFIX = '/mmsb';  // Change if your public path changes
const FULL_URL = URL_PREFIX.endsWith('/') ? URL_PREFIX : URL_PREFIX + '/';

const MIME_TYPES = {
  '.html': 'text/html',
  '.css': 'text/css',
  '.js': 'text/javascript',
  '.json': 'application/json',
  '.md': 'text/markdown',
  '.txt': 'text/plain',
  '.rs': 'text/plain',
  '.toml': 'text/plain',
  '.yaml': 'text/yaml',
  '.yml': 'text/yaml',
  '.sh': 'text/x-shellscript',
  '.py': 'text/x-python',
  '.pdf': 'application/pdf',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.gif': 'image/gif',
  '.svg': 'image/svg+xml',
};

function getContentType(filePath) {
  const ext = path.extname(filePath).toLowerCase();
  return MIME_TYPES[ext] || 'application/octet-stream';
}

function generateDirectoryListing(dirPath, urlPath) {
  const files = fs.readdirSync(dirPath);
  const parentPath = path.dirname(urlPath);
    const parentHref = parentPath === '.' ? URL_PREFIX : URL_PREFIX + parentPath.replace(/\\/g, '/');
    const parent = urlPath === '/' ? '' : `<tr><td><a href="${parentHref}">../</a></td><td>-</td><td>-</td></tr>`;
  
  const items = files.map(file => {
    const fullPath = path.join(dirPath, file);
    const stat = fs.statSync(fullPath);
    const isDir = stat.isDirectory();
    const href = URL_PREFIX + path.join(urlPath, file).replace(/\\/g, '/');
const size = isDir ? '-' : `${(stat.size / 1024).toFixed(2)} KB`;
    const modified = stat.mtime.toISOString().slice(0, 19).replace('T', ' ');
    
    return `<tr>
      <td><a href="${href}">${file}${isDir ? '/' : ''}</a></td>
      <td>${size}</td>
      <td>${modified}</td>
    </tr>`;
  }).join('\n');

  return `<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Index of ${urlPath}</title>
  <style>
    body { font-family: monospace; margin: 20px; background: #1e1e1e; color: #d4d4d4; }
    h1 { color: #4ec9b0; }
    table { border-collapse: collapse; width: 100%; margin-top: 20px; }
    th, td { text-align: left; padding: 8px; border-bottom: 1px solid #3e3e3e; }
    th { background: #2d2d2d; color: #4ec9b0; }
    a { color: #569cd6; text-decoration: none; }
    a:hover { text-decoration: underline; }
    tr:hover { background: #2d2d2d; }
  </style>
</head>
<body>
  <h1>Index of ${urlPath}</h1>
  <table>
    <thead>
      <tr><th>Name</th><th>Size</th><th>Modified</th></tr>
    </thead>
    <tbody>
      ${parent}
      ${items}
    </tbody>
  </table>
</body>
</html>`;
}

const server = http.createServer((req, res) => {
  let urlPath = decodeURIComponent(req.url);
  if (urlPath.includes('..')) {
    res.writeHead(403);
    res.end('Forbidden');
    return;
  }

  const filePath = path.join(ROOT_DIR, urlPath);

  fs.stat(filePath, (err, stats) => {
    if (err) {
      res.writeHead(404);
      res.end('Not Found');
      return;
    }

    if (stats.isDirectory()) {
      const indexPath = path.join(filePath, 'index.html');
      if (fs.existsSync(indexPath)) {
        fs.readFile(indexPath, (err, data) => {
          if (err) {
            res.writeHead(500);
            res.end('Internal Server Error');
            return;
          }
          res.writeHead(200, { 'Content-Type': 'text/html' });
          res.end(data);
        });
      } else {
        const html = generateDirectoryListing(filePath, urlPath);
        res.writeHead(200, { 'Content-Type': 'text/html' });
        res.end(html);
      }
    } else {
      fs.readFile(filePath, (err, data) => {
        if (err) {
          res.writeHead(500);
          res.end('Internal Server Error');
          return;
        }
        res.writeHead(200, { 
          'Content-Type': getContentType(filePath),
          'Cache-Control': 'no-cache, no-store, must-revalidate',
          'Pragma': 'no-cache',
          'Expires': '0'
        });
        res.end(data);
      });
    }
  });
});

server.listen(PORT, '127.0.0.1', () => {
  console.log(`MMSB file server running on http://127.0.0.1:${PORT}`);
});
