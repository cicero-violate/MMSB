/**
 * Save Instruction Endpoint
 * Persists builder-generated patches as reusable instructions
 */

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

const INSTRUCTIONS_DIR = '/home/cicero-arch-omen/ai_sandbox/codex-agent/codex_sse/server-tools/MMSB/instructions';

/**
 * Save built patch as instruction
 */
function handleSaveInstruction(params, res) {
  try {
    // Validate required params
    if (!params.id) {
      return sendError(res, 400, 'Missing required parameter: id');
    }
    if (!params.file) {
      return sendError(res, 400, 'Missing required parameter: file');
    }

    // Build instruction directory
    const instructionDir = path.join(INSTRUCTIONS_DIR, params.id);
    if (fs.existsSync(instructionDir)) {
      return sendError(res, 400, `Instruction ${params.id} already exists`);
    }

    // Create directory structure
    fs.mkdirSync(instructionDir, { recursive: true });

    // Build patch based on params (same logic as instruction-builder)
    const patch = buildPatchFromParams(params);

    // Save patch.diff
    fs.writeFileSync(
      path.join(instructionDir, 'patch.diff'),
      patch
    );

    // Save target.json
    const target = {
      workdir: params.workdir || '',
      files: [params.file]
    };
    fs.writeFileSync(
      path.join(instructionDir, 'target.json'),
      JSON.stringify(target, null, 2)
    );

    // Save meta.json
    const meta = {
      created: new Date().toISOString(),
      template: params.template || 'direct',
      purpose: params.purpose || 'User-generated patch',
      risk: params.risk || 'unknown',
      author: params.author || 'llm'
    };
    fs.writeFileSync(
      path.join(instructionDir, 'meta.json'),
      JSON.stringify(meta, null, 2)
    );

    // Update index.json
    updateInstructionIndex(params.id, params.file, meta);

    sendSuccess(res, {
      success: true,
      message: `Instruction ${params.id} saved successfully`,
      location: instructionDir,
      files_created: ['patch.diff', 'target.json', 'meta.json']
    });

  } catch (err) {
    sendError(res, 500, `Failed to save instruction: ${err.message}`);
  }
}

/**
 * Build patch from params (same templates as instruction-builder)
 */
function buildPatchFromParams(params) {
  if (params.template === 'simple_replace') {
    return `*** Begin Patch
*** Update File: ${params.file}
@@
 ${params.context_before || ''}
-${params.old_line}
+${params.new_line}
 ${params.context_after || ''}
*** End Patch`;
  } else if (params.template === 'add_cache_headers') {
    return `*** Begin Patch
*** Update File: ${params.file}
@@ ${params.function || ''}
 ${params.context_before || ''}
-${params.old_line}
+const headers = addCacheHeaders(${params.headers_object || '{ "Content-Type": "text/html" }'});
+res.writeHead(200, headers);
 ${params.context_after || ''}
*** End Patch`;
  } else {
    // Direct mode
    const contextLines = [
      params.context_1,
      params.context_2,
      params.context_3
    ].filter(c => c).map(c => ` ${c}`).join('\n');
    
    const contextAfter = [
      params.context_4,
      params.context_5,
      params.context_6
    ].filter(c => c).map(c => ` ${c}`).join('\n');
    
    return `*** Begin Patch
*** Update File: ${params.file}
@@ ${params.function || ''}
${contextLines}
-${params.line_before}
+${params.line_after}
${contextAfter}
*** End Patch`;
  }
}

/**
 * Update instruction index
 */
function updateInstructionIndex(id, file, meta) {
  const indexPath = path.join(INSTRUCTIONS_DIR, 'index.json');
  
  let index = {};
  if (fs.existsSync(indexPath)) {
    index = JSON.parse(fs.readFileSync(indexPath, 'utf-8'));
  }

  index[id] = {
    summary: meta.purpose,
    files: [file],
    risk: meta.risk,
    created: meta.created,
    template: meta.template
  };

  fs.writeFileSync(indexPath, JSON.stringify(index, null, 2));
}

function sendSuccess(res, data) {
  res.writeHead(200, {
    'Content-Type': 'application/json',
    'Cache-Control': 'no-cache, no-store, must-revalidate',
    'Pragma': 'no-cache',
    'Expires': '0'
  });
  res.end(JSON.stringify(data, null, 2));
}

function sendError(res, statusCode, message) {
  res.writeHead(statusCode, {
    'Content-Type': 'application/json',
    'Cache-Control': 'no-cache, no-store, must-revalidate',
    'Pragma': 'no-cache',
    'Expires': '0'
  });
  res.end(JSON.stringify({
    success: false,
    error: message
  }, null, 2));
}

module.exports = { handleSaveInstruction };
