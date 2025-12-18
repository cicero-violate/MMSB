/**
 * Sort Middleware
 * Sorts files based on name, size, modified date, or type
 */

const path = require('path');

/**
 * Sort files based on query parameters
 * @param {Array<Object>} files - Array of file objects with {name, stat, isDir}
 * @param {Object} params - Query parameters with sort field and order
 * @returns {Array<Object>} Sorted file array
 */
function sortFiles(files, params) {
  const sortField = params.sort || 'name';
  const sortOrder = params.order || 'asc';

  // Create a copy to avoid mutating original
  const sorted = [...files];

  // Always sort directories first, then apply requested sort
  sorted.sort((a, b) => {
    // Directories always come first
    if (a.isDir && !b.isDir) return -1;
    if (!a.isDir && b.isDir) return 1;

    // Both are same type (both dirs or both files), apply sort
    let comparison = 0;

    switch (sortField) {
      case 'name':
        comparison = compareByName(a, b);
        break;
      case 'size':
        comparison = compareBySize(a, b);
        break;
      case 'modified':
        comparison = compareByModified(a, b);
        break;
      case 'type':
        comparison = compareByType(a, b);
        break;
      default:
        comparison = compareByName(a, b);
    }

    // Apply sort order
    return sortOrder === 'desc' ? -comparison : comparison;
  });

  return sorted;
}

/**
 * Compare files by name (case-insensitive)
 * @param {Object} a - First file
 * @param {Object} b - Second file
 * @returns {number} Comparison result
 */
function compareByName(a, b) {
  return a.name.toLowerCase().localeCompare(b.name.toLowerCase());
}

/**
 * Compare files by size
 * @param {Object} a - First file
 * @param {Object} b - Second file
 * @returns {number} Comparison result
 */
function compareBySize(a, b) {
  // Directories have no meaningful size comparison
  if (a.isDir && b.isDir) return 0;
  
  const sizeA = a.isDir ? 0 : a.stat.size;
  const sizeB = b.isDir ? 0 : b.stat.size;
  
  return sizeA - sizeB;
}

/**
 * Compare files by modification time
 * @param {Object} a - First file
 * @param {Object} b - Second file
 * @returns {number} Comparison result
 */
function compareByModified(a, b) {
  const timeA = a.stat.mtime.getTime();
  const timeB = b.stat.mtime.getTime();
  
  return timeA - timeB;
}

/**
 * Compare files by type (file extension)
 * @param {Object} a - First file
 * @param {Object} b - Second file
 * @returns {number} Comparison result
 */
function compareByType(a, b) {
  // Directories have no extension
  if (a.isDir && b.isDir) return 0;
  
  const extA = a.isDir ? '' : path.extname(a.name).toLowerCase();
  const extB = b.isDir ? '' : path.extname(b.name).toLowerCase();
  
  const extCompare = extA.localeCompare(extB);
  
  // If extensions are the same, sort by name
  return extCompare !== 0 ? extCompare : compareByName(a, b);
}

/**
 * Apply pagination to sorted files
 * @param {Array<Object>} files - Sorted file array
 * @param {Object} params - Query parameters with pagination options
 * @returns {Object} Paginated result with { files, pagination }
 */
function paginateFiles(files, params) {
  const total = files.length;
  
  // Calculate offset and limit
  let offset = params.offset || 0;
  let limit = params.limit;

  // Handle page-based pagination
  if (params.page !== null && params.pagesize) {
    const page = Math.max(1, params.page);
    const pagesize = params.pagesize;
    offset = (page - 1) * pagesize;
    limit = pagesize;
  }

  // No pagination if limit not specified
  if (!limit) {
    return {
      files: files,
      pagination: {
        total: total,
        offset: 0,
        limit: null,
        hasMore: false
      }
    };
  }

  // Apply pagination
  const end = Math.min(offset + limit, total);
  const paginatedFiles = files.slice(offset, end);

  return {
    files: paginatedFiles,
    pagination: {
      total: total,
      offset: offset,
      limit: limit,
      returned: paginatedFiles.length,
      hasMore: end < total
    }
  };
}

module.exports = {
  sortFiles,
  paginateFiles,
  compareByName,
  compareBySize,
  compareByModified,
  compareByType,
};
