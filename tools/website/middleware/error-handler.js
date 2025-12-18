/**
 * Error Handler Middleware
 * Standardizes error responses and logging
 */

const { formatErrorResponse } = require('../routes/api');

/**
 * Error types
 */
const ErrorTypes = {
  BAD_REQUEST: 'BAD_REQUEST',
  FORBIDDEN: 'FORBIDDEN',
  NOT_FOUND: 'NOT_FOUND',
  INTERNAL_ERROR: 'INTERNAL_ERROR',
  VALIDATION_ERROR: 'VALIDATION_ERROR',
};

/**
 * Create standardized error object
 */
class ServerError extends Error {
  constructor(type, message, details = null) {
    super(message);
    this.type = type;
    this.details = details;
    this.statusCode = getStatusCode(type);
  }
}

/**
 * Map error type to HTTP status code
 */
function getStatusCode(errorType) {
  const statusMap = {
    [ErrorTypes.BAD_REQUEST]: 400,
    [ErrorTypes.FORBIDDEN]: 403,
    [ErrorTypes.NOT_FOUND]: 404,
    [ErrorTypes.INTERNAL_ERROR]: 500,
    [ErrorTypes.VALIDATION_ERROR]: 400,
  };
  return statusMap[errorType] || 500;
}

/**
 * Handle error and send response
 */
function handleError(err, res, format = 'json') {
  let statusCode = 500;
  let message = 'Internal Server Error';
  let details = null;

  if (err instanceof ServerError) {
    statusCode = err.statusCode;
    message = err.message;
    details = err.details;
  } else if (err.code === 'ENOENT') {
    statusCode = 404;
    message = 'File or directory not found';
  } else if (err.code === 'EACCES') {
    statusCode = 403;
    message = 'Access denied';
  } else {
    // Log unexpected errors
    console.error('Unexpected error:', err);
  }

  const response = formatErrorResponse(statusCode, message, format);
  
  res.writeHead(statusCode, { 'Content-Type': response.contentType });
  res.end(response.content);

  // Log error
  logError(statusCode, message, details);
}

/**
 * Log error to console
 */
function logError(statusCode, message, details) {
  const timestamp = new Date().toISOString();
  console.error(`[${timestamp}] ERROR ${statusCode}: ${message}`);
  if (details) {
    console.error('Details:', details);
  }
}

/**
 * Validate query parameters
 */
function validateRequest(params) {
  const errors = [];

  // Check for path traversal
  if (params.path && params.path.includes('..')) {
    errors.push('Path traversal not allowed');
  }

  // Validate pagination
  if (params.page !== null && params.offset > 0) {
    errors.push('Cannot use both "page" and "offset" parameters');
  }

  // Validate format requirements
  if (params.metadata && params.format !== 'json') {
    errors.push('Metadata queries require format=json');
  }

  if (params.stats && params.format !== 'json') {
    errors.push('Statistics queries require format=json');
  }

  // Validate preview
  if (params.preview && params.metadata) {
    errors.push('Cannot use both "preview" and "metadata" parameters');
  }

  // Validate recursive depth
  if (params.recursive && params.depth !== null && params.depth > 10) {
    errors.push('Maximum recursive depth is 10');
  }

  // Validate limit
  if (params.limit !== null && params.limit > 10000) {
    errors.push('Maximum limit is 10000');
  }

  if (errors.length > 0) {
    throw new ServerError(
      ErrorTypes.VALIDATION_ERROR,
      'Invalid query parameters',
      errors
    );
  }

  return true;
}

/**
 * Log request
 */
function logRequest(method, url, statusCode) {
  const timestamp = new Date().toISOString();
  console.log(`[${timestamp}] ${method} ${url} → ${statusCode}`);
}

module.exports = {
  ServerError,
  ErrorTypes,
  handleError,
  validateRequest,
  logRequest,
};
