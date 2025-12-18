/**
 * Recursive Directory Traversal Middleware
 * Handles depth-limited recursive directory listing
 */

const fs = require('fs');
const path = require('path');

/**
 * Recursively gather files from directory tree
 * @param {string} dirPath - Root directory path
 * @param {Object} params - Query parameters with recursive and depth options
 * @param {string} basePath - Base path for relative URLs (optional)
 * @returns {Array<Object>} Array of file objects with relative paths
 */
function gatherFilesRecursive(dirPath, params, basePath = '') {
  if (!params.recursive) {
    // Non-recursive: just return immediate children
    return gatherFilesShallow(dirPath, basePath);
  }

  const maxDepth = params.depth || Infinity;
  const files = [];

  function traverse(currentPath, relativePath, currentDepth) {
    // Check depth limit
    if (currentDepth > maxDepth) {
      return;
    }

    try {
      const entries = fs.readdirSync(currentPath);

      for (const entry of entries) {
        const fullPath = path.join(currentPath, entry);
        const relPath = path.join(relativePath, entry);

        try {
          const stat = fs.statSync(fullPath);
          const isDir = stat.isDirectory();

          files.push({
            name: entry,
            path: relPath.replace(/\\/g, '/'),
            fullPath: fullPath,
            stat: stat,
            isDir: isDir,
            depth: currentDepth
          });

          // Recurse into subdirectories
          if (isDir && currentDepth < maxDepth) {
            traverse(fullPath, relPath, currentDepth + 1);
          }
        } catch (err) {
          // Skip files we can't stat
          continue;
        }
      }
    } catch (err) {
      // Skip directories we can't read
      return;
    }
  }

  traverse(dirPath, basePath, 0);
  return files;
}

/**
 * Gather files from single directory (non-recursive)
 * @param {string} dirPath - Directory path
 * @param {string} basePath - Base path for relative URLs
 * @returns {Array<Object>} Array of file objects
 */
function gatherFilesShallow(dirPath, basePath = '') {
  const files = [];

  try {
    const entries = fs.readdirSync(dirPath);

    for (const entry of entries) {
      const fullPath = path.join(dirPath, entry);
      const relPath = path.join(basePath, entry);

      try {
        const stat = fs.statSync(fullPath);

        files.push({
          name: entry,
          path: relPath.replace(/\\/g, '/'),
          fullPath: fullPath,
          stat: stat,
          isDir: stat.isDirectory(),
          depth: 0
        });
      } catch (err) {
        // Skip files we can't stat
        continue;
      }
    }
  } catch (err) {
    // Return empty array if can't read directory
    return [];
  }

  return files;
}

/**
 * Build tree structure from flat file list
 * @param {Array<Object>} files - Flat array of file objects
 * @returns {Object} Tree structure
 */
function buildTreeStructure(files) {
  const tree = {
    name: '/',
    type: 'directory',
    children: []
  };

  const pathMap = new Map();
  pathMap.set('', tree);

  // Sort by path to ensure parents are processed before children
  const sorted = [...files].sort((a, b) => 
    a.path.localeCompare(b.path)
  );

  for (const file of sorted) {
    const parentPath = path.dirname(file.path);
    const parent = pathMap.get(parentPath === '.' ? '' : parentPath);

    if (!parent) continue;

    const node = {
      name: file.name,
      type: file.isDir ? 'directory' : 'file',
      path: file.path
    };

    if (!file.isDir) {
      node.size = file.stat.size;
      node.modified = file.stat.mtime.toISOString();
    }

    if (file.isDir) {
      node.children = [];
      pathMap.set(file.path, node);
    }

    parent.children.push(node);
  }

  return tree;
}

/**
 * Get directory statistics for recursive listing
 * @param {Array<Object>} files - File array
 * @returns {Object} Recursive statistics
 */
function getRecursiveStats(files) {
  const stats = {
    totalFiles: 0,
    totalDirs: 0,
    totalSize: 0,
    maxDepth: 0,
    fileTypes: {},
    depthDistribution: {}
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

    stats.maxDepth = Math.max(stats.maxDepth, file.depth);
    stats.depthDistribution[file.depth] = (stats.depthDistribution[file.depth] || 0) + 1;
  }

  return stats;
}

/**
 * Format recursive listing for display
 * @param {Array<Object>} files - File array with depth info
 * @returns {string} Formatted text output
 */
function formatRecursiveTree(files) {
  const lines = [];

  // Group by directory
  const byPath = new Map();
  for (const file of files) {
    const dir = path.dirname(file.path) || '/';
    if (!byPath.has(dir)) {
      byPath.set(dir, []);
    }
    byPath.get(dir).push(file);
  }

  // Sort directories
  const sortedDirs = Array.from(byPath.keys()).sort();

  for (const dir of sortedDirs) {
    const dirFiles = byPath.get(dir);
    const depth = dir === '/' ? 0 : dir.split('/').length;
    const indent = '  '.repeat(depth);

    if (dir !== '/') {
      lines.push(`${indent}${dir}/`);
    }

    for (const file of dirFiles) {
      const fileIndent = '  '.repeat(depth + 1);
      const marker = file.isDir ? '[DIR]' : '[FILE]';
      lines.push(`${fileIndent}${marker} ${file.name}`);
    }
  }

  return lines.join('\n');
}

module.exports = {
  gatherFilesRecursive,
  gatherFilesShallow,
  buildTreeStructure,
  getRecursiveStats,
  formatRecursiveTree,
};
