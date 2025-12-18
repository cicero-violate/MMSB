/**
 * Query Parameter Parser Middleware
 * Extracts and validates URL query parameters for file server operations
 */

const url = require('url');

/**
 * Parse and validate query parameters from request URL
 * @param {string} requestUrl - The full request URL
 * @returns {Object} Parsed query parameters with defaults
 */
function parseQueryParams(requestUrl) {
  const parsedUrl = url.parse(requestUrl, true);
  const query = parsedUrl.query;

  return {
    // Path parameters
    path: parsedUrl.pathname,

    // Filter parameters
    ext: parseExtensions(query.ext),
    type: parseTypes(query.type),

    // Sort parameters
    sort: parseSortField(query.sort),
    order: parseSortOrder(query.order),

    // Format parameters
    format: parseFormat(query.format),
    pretty: parseBoolean(query.pretty, false),

    // Pagination parameters
    limit: parseLimit(query.limit),
    offset: parseOffset(query.offset),
    page: parseInt(query.page) || null,
    pagesize: parseInt(query.pagesize) || null,

    // Search parameters
    search: query.search || null,
    pattern: query.pattern || null,

    // Recursive parameters
    recursive: parseBoolean(query.recursive, false),
    depth: parseDepth(query.depth),

    // Metadata parameters
    metadata: parseBoolean(query.metadata, false),
    stats: parseBoolean(query.stats, false),

    // Apply-patch parameters
    patch: query.patch || null,
    workdir: query.workdir || null,
    encoding: query.encoding || null,

    // Preview parameters
    preview: parseBoolean(query.preview, false),
    lines: parseInt(query.lines) || 20,
  };
}

/**
 * Parse file extensions from query parameter
 * @param {string} extParam - Extension parameter (e.g., ".rs" or "rs,toml")
 * @returns {Array<string>|null} Array of extensions or null
 */
function parseExtensions(extParam) {
  if (!extParam) return null;

  const exts = extParam.split(',').map(e => {
    e = e.trim();
    return e.startsWith('.') ? e : '.' + e;
  });

  return exts.length > 0 ? exts : null;
}

/**
 * Parse file types from query parameter
 * @param {string} typeParam - Type parameter (e.g., "rust,toml,markdown")
 * @returns {Array<string>|null} Array of types or null
 */
function parseTypes(typeParam) {
  if (!typeParam) return null;

  const types = typeParam.split(',').map(t => t.trim().toLowerCase());
  return types.length > 0 ? types : null;
}

/**
 * Parse sort field with validation
 * @param {string} sortParam - Sort field (name, size, modified)
 * @returns {string} Validated sort field
 */
function parseSortField(sortParam) {
  const validFields = ['name', 'size', 'modified', 'type'];
  const field = (sortParam || 'name').toLowerCase();
  return validFields.includes(field) ? field : 'name';
}

/**
 * Parse sort order with validation
 * @param {string} orderParam - Sort order (asc, desc)
 * @returns {string} Validated sort order
 */
function parseSortOrder(orderParam) {
  const order = (orderParam || 'asc').toLowerCase();
  return order === 'desc' ? 'desc' : 'asc';
}

/**
 * Parse output format with validation
 * @param {string} formatParam - Format (html, json, text)
 * @returns {string} Validated format
 */
function parseFormat(formatParam) {
  const validFormats = ['html', 'json', 'text'];
  const format = (formatParam || 'html').toLowerCase();
  return validFormats.includes(format) ? format : 'html';
}

/**
 * Parse boolean parameter
 * @param {string} param - Boolean parameter value
 * @param {boolean} defaultValue - Default value if not specified
 * @returns {boolean} Parsed boolean value
 */
function parseBoolean(param, defaultValue) {
  if (param === undefined || param === null) return defaultValue;
  const val = param.toString().toLowerCase();
  return val === 'true' || val === '1' || val === 'yes';
}

/**
 * Parse limit parameter with bounds
 * @param {string} limitParam - Limit value
 * @returns {number|null} Validated limit or null for no limit
 */
function parseLimit(limitParam) {
  if (!limitParam) return null;
  const limit = parseInt(limitParam);
  if (isNaN(limit) || limit <= 0) return null;
  return Math.min(limit, 10000); // Max 10000 items
}

/**
 * Parse offset parameter
 * @param {string} offsetParam - Offset value
 * @returns {number} Validated offset
 */
function parseOffset(offsetParam) {
  if (!offsetParam) return 0;
  const offset = parseInt(offsetParam);
  return isNaN(offset) || offset < 0 ? 0 : offset;
}

/**
 * Parse recursive depth parameter
 * @param {string} depthParam - Depth value
 * @returns {number|null} Validated depth or null for unlimited
 */
function parseDepth(depthParam) {
  if (!depthParam) return null;
  const depth = parseInt(depthParam);
  if (isNaN(depth) || depth <= 0) return null;
  return Math.min(depth, 10); // Max depth of 10
}

/**
 * Validate query parameters for conflicts or invalid combinations
 * @param {Object} params - Parsed query parameters
 * @returns {Object} Validation result with { valid: boolean, errors: Array<string> }
 */
function validateQueryParams(params) {
  const errors = [];

  // Check for conflicting pagination parameters
  if (params.page !== null && params.offset > 0) {
    errors.push('Cannot use both "page" and "offset" parameters');
  }

  // Check for conflicting format parameters
  if (params.metadata && params.format !== 'json') {
    errors.push('Metadata queries require format=json');
  }

  if (params.stats && params.format !== 'json') {
    errors.push('Statistics queries require format=json');
  }

  // Check for invalid preview combinations
  if (params.preview && params.metadata) {
    errors.push('Cannot use both "preview" and "metadata" parameters');
  }

  return {
    valid: errors.length === 0,
    errors: errors
  };
}

/**
 * Map type names to file extensions
 */
const TYPE_TO_EXTENSIONS = {
  'rust': ['.rs'],
  'javascript': ['.js'],
  'typescript': ['.ts'],
  'python': ['.py'],
  'julia': ['.jl'],
  'markdown': ['.md'],
  'config': ['.toml', '.yaml', '.yml', '.json'],
  'shell': ['.sh', '.bash'],
  'text': ['.txt', '.log'],
  'image': ['.png', '.jpg', '.jpeg', '.gif', '.svg'],
  'document': ['.pdf', '.html'],
};

/**
 * Convert type names to extension list
 * @param {Array<string>} types - Array of type names
 * @returns {Array<string>} Array of extensions
 */
function typesToExtensions(types) {
  if (!types) return null;

  const extensions = new Set();
  for (const type of types) {
    const exts = TYPE_TO_EXTENSIONS[type];
    if (exts) {
      exts.forEach(ext => extensions.add(ext));
    }
  }

  return extensions.size > 0 ? Array.from(extensions) : null;
}

module.exports = {
  parseQueryParams,
  validateQueryParams,
  typesToExtensions,
  TYPE_TO_EXTENSIONS,
};
