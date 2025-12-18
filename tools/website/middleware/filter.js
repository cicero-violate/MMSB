/**
 * File Filter Middleware
 * Filters files based on extension, type, and search patterns
 */

const path = require('path');
const { typesToExtensions } = require('./query-parser');

/**
 * Filter files based on query parameters
 * @param {Array<Object>} files - Array of file objects with {name, stat, isDir}
 * @param {Object} params - Query parameters from parser
 * @returns {Array<Object>} Filtered file array
 */
function filterFiles(files, params) {
  let filtered = files;

  // Apply extension filter
  if (params.ext && params.ext.length > 0) {
    filtered = filterByExtensions(filtered, params.ext);
  }

  // Apply type filter (converts types to extensions)
  if (params.type && params.type.length > 0) {
    const typeExts = typesToExtensions(params.type);
    if (typeExts) {
      filtered = filterByExtensions(filtered, typeExts);
    }
  }

  // Apply search filter (filename contains search term)
  if (params.search) {
    filtered = filterBySearch(filtered, params.search);
  }

  // Apply pattern filter (glob-style matching)
  if (params.pattern) {
    filtered = filterByPattern(filtered, params.pattern);
  }

  return filtered;
}

/**
 * Filter files by file extensions
 * @param {Array<Object>} files - File array
 * @param {Array<string>} extensions - Array of extensions (e.g., ['.rs', '.toml'])
 * @returns {Array<Object>} Filtered files
 */
function filterByExtensions(files, extensions) {
  const extSet = new Set(extensions.map(e => e.toLowerCase()));
  
  return files.filter(file => {
    // Always include directories
    if (file.isDir) return true;
    
    const ext = path.extname(file.name).toLowerCase();
    return extSet.has(ext);
  });
}

/**
 * Filter files by search term (case-insensitive substring match)
 * @param {Array<Object>} files - File array
 * @param {string} searchTerm - Search term
 * @returns {Array<Object>} Filtered files
 */
function filterBySearch(files, searchTerm) {
  const term = searchTerm.toLowerCase();
  
  return files.filter(file => {
    return file.name.toLowerCase().includes(term);
  });
}

/**
 * Filter files by glob pattern
 * @param {Array<Object>} files - File array
 * @param {string} pattern - Glob pattern (e.g., "*.rs", "test_*")
 * @returns {Array<Object>} Filtered files
 */
function filterByPattern(files, pattern) {
  const regex = globToRegex(pattern);
  
  return files.filter(file => {
    return regex.test(file.name);
  });
}

/**
 * Convert glob pattern to RegExp
 * Supports * (any characters) and ? (single character)
 * @param {string} pattern - Glob pattern
 * @returns {RegExp} Regular expression
 */
function globToRegex(pattern) {
  // Escape special regex characters except * and ?
  let regexStr = pattern
    .replace(/[.+^${}()|[\]\\]/g, '\\$&')
    .replace(/\*/g, '.*')
    .replace(/\?/g, '.');
  
  return new RegExp('^' + regexStr + '$', 'i');
}

/**
 * Get file statistics for filtered results
 * @param {Array<Object>} files - File array
 * @returns {Object} Statistics object
 */
function getFileStats(files) {
  const stats = {
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

  return stats;
}

module.exports = {
  filterFiles,
  filterByExtensions,
  filterBySearch,
  filterByPattern,
  getFileStats,
  globToRegex,
};
