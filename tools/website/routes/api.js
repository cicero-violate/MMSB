/**
 * API Response Formatter
 * Formats file server responses in JSON, HTML, or text format
 */

const path = require('path');
const fs = require('fs');

/**
 * Format directory listing response
 * @param {Array<Object>} files - File array with {name, stat, isDir}
 * @param {Object} params - Query parameters
 * @param {string} urlPath - Current URL path
 * @param {Object} pagination - Pagination info (optional)
 * @returns {Object} Formatted response with {content, contentType}
 */
function formatDirectoryResponse(files, params, urlPath, pagination = null) {
  const format = params.format || 'html';

  switch (format) {
    case 'json':
      return formatDirectoryJSON(files, params, urlPath, pagination);
    case 'text':
      return formatDirectoryText(files, params, urlPath);
    case 'html':
    default:
      return formatDirectoryHTML(files, params, urlPath, pagination);
  }
}

/**
 * Format directory listing as JSON
 */
function formatDirectoryJSON(files, params, urlPath, pagination) {
  const URL_PREFIX = '/mmsb';
  
  const fileObjects = files.map(file => {
    const obj = {
      name: file.name,
      type: file.isDir ? 'directory' : 'file',
      path: path.join(urlPath, file.name).replace(/\\/g, '/'),
      url: URL_PREFIX + path.join(urlPath, file.name).replace(/\\/g, '/')
    };

    if (!file.isDir) {
      obj.size = file.stat.size;
      obj.sizeFormatted = formatSize(file.stat.size);
      obj.extension = path.extname(file.name).toLowerCase();
    }

    obj.modified = file.stat.mtime.toISOString();
    obj.modifiedFormatted = formatDate(file.stat.mtime);

    return obj;
  });

  const response = {
    path: urlPath,
    files: fileObjects
  };

  if (pagination) {
    response.pagination = pagination;
  }

  const content = params.pretty 
    ? JSON.stringify(response, null, 2)
    : JSON.stringify(response);

  return {
    content: content,
    contentType: 'application/json'
  };
}

/**
 * Format directory listing as plain text
 */
function formatDirectoryText(files, params, urlPath) {
  let lines = [`Directory: ${urlPath}`, ''];

  // Header
  lines.push('TYPE'.padEnd(12) + 'SIZE'.padEnd(15) + 'MODIFIED'.padEnd(22) + 'NAME');
  lines.push('-'.repeat(80));

  // Files
  for (const file of files) {
    const type = file.isDir ? '[DIR]' : '[FILE]';
    const size = file.isDir ? '-' : formatSize(file.stat.size);
    const modified = formatDate(file.stat.mtime);
    const name = file.name;

    lines.push(
      type.padEnd(12) + 
      size.padEnd(15) + 
      modified.padEnd(22) + 
      name
    );
  }

  return {
    content: lines.join('\n'),
    contentType: 'text/plain'
  };
}

/**
 * Format directory listing as HTML
 */
function formatDirectoryHTML(files, params, urlPath, pagination) {
  const URL_PREFIX = '/mmsb';
  const parentPath = path.dirname(urlPath);
  const parentHref = parentPath === '.' ? URL_PREFIX : URL_PREFIX + parentPath.replace(/\\/g, '/');
  const parent = urlPath === '/' ? '' : `<tr><td><a href="${parentHref}">../</a></td><td>-</td><td>-</td></tr>`;
  
  const items = files.map(file => {
    const href = URL_PREFIX + path.join(urlPath, file.name).replace(/\\/g, '/');
    const size = file.isDir ? '-' : formatSize(file.stat.size);
    const modified = formatDate(file.stat.mtime);
    
    return `<tr>
      <td><a href="${href}">${file.name}${file.isDir ? '/' : ''}</a></td>
      <td>${size}</td>
      <td>${modified}</td>
    </tr>`;
  }).join('\n');

  // Build query info section
  let queryInfo = '';
  if (params.ext || params.type || params.search || params.pattern) {
    const filters = [];
    if (params.ext) filters.push(`Extensions: ${params.ext.join(', ')}`);
    if (params.type) filters.push(`Types: ${params.type.join(', ')}`);
    if (params.search) filters.push(`Search: "${params.search}"`);
    if (params.pattern) filters.push(`Pattern: "${params.pattern}"`);
    
    queryInfo = `<div class="query-info">
      <strong>Active Filters:</strong> ${filters.join(' | ')}
    </div>`;
  }

  // Build pagination section
  let paginationInfo = '';
  if (pagination && pagination.limit) {
    paginationInfo = `<div class="pagination">
      Showing ${pagination.offset + 1}-${pagination.offset + pagination.returned} of ${pagination.total} items
      ${pagination.hasMore ? '(more available)' : ''}
    </div>`;
  }

  return {
    content: `<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Index of ${urlPath}</title>
  <style>
    body { font-family: monospace; margin: 20px; background: #1e1e1e; color: #d4d4d4; }
    h1 { color: #4ec9b0; }
    .query-info { 
      margin: 10px 0; 
      padding: 10px; 
      background: #2d2d2d; 
      border-left: 3px solid #4ec9b0;
      color: #dcdcaa;
    }
    .pagination {
      margin: 10px 0;
      padding: 8px;
      background: #2d2d2d;
      color: #9cdcfe;
    }
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
  ${queryInfo}
  ${paginationInfo}
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
</html>`,
    contentType: 'text/html'
  };
}

/**
 * Format file metadata response (JSON only)
 */
function formatMetadataResponse(filePath, stat, params) {
  const metadata = {
    name: path.basename(filePath),
    path: filePath,
    size: stat.size,
    sizeFormatted: formatSize(stat.size),
    modified: stat.mtime.toISOString(),
    modifiedFormatted: formatDate(stat.mtime),
    created: stat.birthtime.toISOString(),
    accessed: stat.atime.toISOString(),
    isDirectory: stat.isDirectory(),
    isFile: stat.isFile(),
  };

  if (stat.isFile()) {
    metadata.extension = path.extname(filePath).toLowerCase();
  }

  const content = params.pretty 
    ? JSON.stringify(metadata, null, 2)
    : JSON.stringify(metadata);

  return {
    content: content,
    contentType: 'application/json'
  };
}

/**
 * Format statistics response (JSON only)
 */
function formatStatsResponse(files, params, urlPath) {
  const stats = {
    path: urlPath,
    totalFiles: 0,
    totalDirs: 0,
    totalSize: 0,
    fileTypes: {}
  };

  for (const file of files) {
    if (file.isDir) {
      stats.totalDirs++;
    } else {
      stats.totalFiles++;
      stats.totalSize += file.stat.size;
      
      const ext = path.extname(file.name).toLowerCase() || '.none';
      stats.fileTypes[ext] = (stats.fileTypes[ext] || 0) + 1;
    }
  }

  stats.totalSizeFormatted = formatSize(stats.totalSize);

  const content = params.pretty 
    ? JSON.stringify(stats, null, 2)
    : JSON.stringify(stats);

  return {
    content: content,
    contentType: 'application/json'
  };
}

/**
 * Format file preview response
 */
function formatPreviewResponse(filePath, params) {
  const content = fs.readFileSync(filePath, 'utf8');
  const lines = content.split('\n');
  const previewLines = params.lines || 20;
  const truncated = lines.slice(0, previewLines);

  const response = {
    path: filePath,
    totalLines: lines.length,
    previewLines: previewLines,
    truncated: lines.length > previewLines,
    content: truncated.join('\n')
  };

  const jsonContent = params.pretty 
    ? JSON.stringify(response, null, 2)
    : JSON.stringify(response);

  return {
    content: jsonContent,
    contentType: 'application/json'
  };
}

/**
 * Format error response
 */
function formatErrorResponse(statusCode, message, format = 'json') {
  const error = {
    error: true,
    statusCode: statusCode,
    message: message
  };

  if (format === 'json') {
    return {
      content: JSON.stringify(error, null, 2),
      contentType: 'application/json'
    };
  } else if (format === 'text') {
    return {
      content: `ERROR ${statusCode}: ${message}`,
      contentType: 'text/plain'
    };
  } else {
    return {
      content: `<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>Error ${statusCode}</title>
  <style>
    body { font-family: monospace; margin: 40px; background: #1e1e1e; color: #d4d4d4; }
    h1 { color: #f48771; }
  </style>
</head>
<body>
  <h1>Error ${statusCode}</h1>
  <p>${message}</p>
</body>
</html>`,
      contentType: 'text/html'
    };
  }
}

/**
 * Format file size in human-readable format
 */
function formatSize(bytes) {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

/**
 * Format date in readable format
 */
function formatDate(date) {
  return date.toISOString().slice(0, 19).replace('T', ' ');
}

module.exports = {
  formatDirectoryResponse,
  formatMetadataResponse,
  formatStatsResponse,
  formatPreviewResponse,
  formatErrorResponse,
  formatSize,
  formatDate,
};
