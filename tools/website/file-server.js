/**
 * MMSB Enhanced File Server
 * Advanced query-based file serving with filtering, sorting, and multiple output formats
 */

const http = require('http');
const fs = require('fs');
const path = require('path');

// Load configuration
const config = require('./config/server-config.json');

// Import middleware
const { parseQueryParams, validateQueryParams } = require('./middleware/query-parser');
const { filterFiles, getFileStats } = require('./middleware/filter');
const { sortFiles, paginateFiles } = require('./middleware/sort');
const { gatherFilesRecursive, gatherFilesShallow } = require('./middleware/recursive');
const { ServerError, ErrorTypes, handleError, validateRequest, logRequest } = require('./middleware/error-handler');
const { handleApplyPatch } = require('./routes/apply-patch');
const { handleInstructionBuilder } = require('./routes/instruction-builder');
const { handleSaveInstruction } = require('./routes/save-instruction');
const { handleListInstructions, handleGetInstruction } = require('./routes/list-instructions');
const { handleApplyInstruction } = require('./routes/apply-instruction');
const { 
  formatDirectoryResponse, 
  formatMetadataResponse, 
  formatStatsResponse,
  formatPreviewResponse,
  formatRecursiveResponse,
  formatErrorResponse
} = require('./routes/api');

// Server configuration
const PORT = config.server.port || 8888;
const HOST = config.server.host || '127.0.0.1';
const URL_PREFIX = config.server.urlPrefix || '/mmsb';
const ROOT_DIR = path.resolve(__dirname, '../..');

// MIME types for direct file serving
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
  '.jl': 'text/plain',
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

/**
 * Add cache-busting headers to response
 */
function addCacheHeaders(headers) {
  if (config.cache.noCache) {
    headers['Cache-Control'] = 'no-cache, no-store, must-revalidate';
    headers['Pragma'] = 'no-cache';
    headers['Expires'] = '0';
  } else if (config.cache.maxAge) {
    headers['Cache-Control'] = `max-age=${config.cache.maxAge}`;
  }
  return headers;
}

/**
 * Main request handler
 */
const server = http.createServer(async (req, res) => {
  const startTime = Date.now();

  try {
    // Parse query parameters
    const params = parseQueryParams(req.url);
    
    // Handle apply-patch endpoint
    if (params.path === '/apply-patch' || params.path === `${URL_PREFIX}/apply-patch`) {
      return handleApplyPatch(params, res);
    }
    
    // Handle instruction-builder endpoint
    if (params.path === '/build-patch' || params.path === `${URL_PREFIX}/build-patch`) {
      return handleInstructionBuilder(params, res);
    }
    
    // Handle save-instruction endpoint
    if (params.path === '/save-instruction' || params.path === `${URL_PREFIX}/save-instruction`) {
      return handleSaveInstruction(params, res);
    }
    
    // Handle list-instructions endpoint
    if (params.path === '/list-instructions' || params.path === `${URL_PREFIX}/list-instructions`) {
      return handleListInstructions(params, res);
    }
    
    // Handle get-instruction endpoint
    if (params.path === '/instruction' || params.path === `${URL_PREFIX}/instruction`) {
      return handleGetInstruction(params, res);
    }
    
    // Handle apply-instruction endpoint
    if (params.path === '/apply-instruction' || params.path === `${URL_PREFIX}/apply-instruction`) {
      return handleApplyInstruction(params, res);
    }
    
    // Strip URL prefix to get file path
    let urlPath = params.path;
    if (urlPath.startsWith(URL_PREFIX)) {
      urlPath = urlPath.substring(URL_PREFIX.length);
    }
    
    // Normalize path
    urlPath = urlPath || '/';
    params.path = urlPath;

    // Validate request
    validateRequest(params);

    // Resolve file system path
    const filePath = path.join(ROOT_DIR, urlPath);

    // Check if path exists
    if (!fs.existsSync(filePath)) {
      throw new ServerError(ErrorTypes.NOT_FOUND, 'File or directory not found');
    }

    const stat = fs.statSync(filePath);

    // Handle directory requests
    if (stat.isDirectory()) {
      await handleDirectoryRequest(filePath, urlPath, params, res);
    } 
    // Handle file requests
    else {
      await handleFileRequest(filePath, urlPath, params, res, stat);
    }

    // Log successful request
    const duration = Date.now() - startTime;
    if (config.logging.logRequests) {
      logRequest(req.method, req.url, 200);
      console.log(`  → Completed in ${duration}ms`);
    }

  } catch (err) {
    if (config.logging.logErrors) {
      console.error('Request failed:', err.message);
    }
    
    const params = parseQueryParams(req.url);
    handleError(err, res, params.format);
  }
});

/**
 * Handle directory listing requests
*/
async function handleDirectoryRequest(dirPath, urlPath, params, res) {
  // Only serve index.html if no query parameters are present
  const hasQueryParams = params.format || params.ext || params.search || 
                         params.sort || params.stats || params.recursive ||
                         params.limit || params.pattern || params.type;
  
  if (!hasQueryParams) {
    const indexPath = path.join(dirPath, 'index.html');
    if (fs.existsSync(indexPath)) {
      const data = fs.readFileSync(indexPath);
      const headers = addCacheHeaders({ 'Content-Type': 'text/html' });
      res.writeHead(200, headers);
      res.end(data);
      return;
    }
  }

  // Force JSON format for stats requests
  if (params.stats) {
    params.format = 'json';
  }

  // Gather files (recursive or shallow)
  let files;
  if (params.recursive) {
    files = gatherFilesRecursive(dirPath, params, urlPath);
    
    // For recursive queries, return special format
    const response = formatRecursiveResponse(files, params, urlPath);
    const headers = addCacheHeaders({ 'Content-Type': response.contentType });
    res.writeHead(200, headers);
    res.end(response.content);
    return;
  } else {
    files = gatherFilesShallow(dirPath, urlPath);
  }

  // Handle stats-only request
  if (params.stats) {
    const response = formatStatsResponse(files, params, urlPath);
    const headers = addCacheHeaders({ 'Content-Type': response.contentType });
    res.writeHead(200, headers);
    res.end(response.content);
    return;
  }

  // Apply filters
  let filtered = filterFiles(files, params);

  // Apply sorting
  let sorted = sortFiles(filtered, params);

  // Apply pagination
  const paginated = paginateFiles(sorted, params);

  // Format response
  const response = formatDirectoryResponse(
    paginated.files, 
    params, 
    urlPath, 
    paginated.pagination
  );

  const headers = addCacheHeaders({ 'Content-Type': response.contentType });
  res.writeHead(200, headers);
  res.end(response.content);
}

/**
 * Handle file requests
 */
async function handleFileRequest(filePath, urlPath, params, res, stat) {
  // Force JSON format for metadata requests
  if (params.metadata) {
    params.format = 'json';
  }

  // Handle metadata-only request
  if (params.metadata) {
    const response = formatMetadataResponse(filePath, stat, params);
    const headers = addCacheHeaders({ 'Content-Type': response.contentType });
    res.writeHead(200, headers);
    res.end(response.content);
    return;
  }

  // Handle preview request (text files only)
  if (params.preview) {
    try {
      const response = formatPreviewResponse(filePath, params);
      const headers = addCacheHeaders({ 'Content-Type': response.contentType });
      res.writeHead(200, headers);
      res.end(response.content);
      return;
    } catch (err) {
      throw new ServerError(ErrorTypes.BAD_REQUEST, 'Cannot preview binary or non-text files');
    }
  }

  // Serve file content
  const data = fs.readFileSync(filePath);
  const contentType = getContentType(filePath);
  
  const headers = {
    'Content-Type': contentType,
  };

  res.writeHead(200, addCacheHeaders(headers));
  res.end(data);
}

/**
 * Start server
 */
server.listen(PORT, HOST, () => {
  console.log('='.repeat(60));
  console.log('MMSB Enhanced File Server');
  console.log('='.repeat(60));
  console.log(`Server:    http://${HOST}:${PORT}${URL_PREFIX}`);
  console.log(`Root Dir:  ${ROOT_DIR}`);
  console.log(`Config:    ./config/server-config.json`);
  console.log('='.repeat(60));
  console.log('Features enabled:');
  console.log(`  ✓ Query filtering (ext, type, search, pattern)`);
  console.log(`  ✓ Sorting (name, size, modified, type)`);
  console.log(`  ✓ Multiple formats (JSON, HTML, text)`);
  console.log(`  ✓ Pagination (limit/offset, page/pagesize)`);
  console.log(`  ${config.features.recursive.enabled ? '✓' : '✗'} Recursive listing (depth: ${config.features.recursive.maxDepth})`);
  console.log(`  ${config.features.preview.enabled ? '✓' : '✗'} Content preview`);
  console.log(`  ✓ Metadata queries`);
  console.log(`  ✓ Statistics aggregation`);
  console.log('='.repeat(60));
  console.log('Documentation: ./README.md');
  console.log('Example: ' + `http://${HOST}:${PORT}${URL_PREFIX}/src?ext=.rs&sort=modified&format=json`);
  console.log('='.repeat(60));
});

// Handle shutdown gracefully
process.on('SIGINT', () => {
  console.log('\nShutting down server...');
  server.close(() => {
    console.log('Server stopped');
    process.exit(0);
  });
});
