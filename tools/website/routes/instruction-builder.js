/**
 * Instruction Builder Endpoint
 * Builds patches from GET parameters (no encoding needed)
 */

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');
const os = require('os');

const APPLY_PATCH_BIN = '/home/cicero-arch-omen/ai_sandbox/tools/apply-patch/target/release/apply_patch';

/**
 * Build and apply patch from GET parameters
 */
function handleInstructionBuilder(params, res) {
  try {
    // Determine build method
    if (params.template) {
      return applyTemplate(params, res);
    } else if (params.file && params.line_before && params.line_after) {
      return applySimpleReplace(params, res);
    } else {
      return sendError(res, 400, 'Must provide either template or (file, line_before, line_after)');
    }
  } catch (err) {
    sendError(res, 500, `Internal error: ${err.message}`);
  }
}

/**
 * Simple line replacement
 */
function applySimpleReplace(params, res) {
  const workdir = params.workdir || '/home/cicero-arch-omen/ai_sandbox/codex-agent/codex_sse/server-tools/MMSB';
  
  if (!fs.existsSync(workdir)) {
    return sendError(res, 400, `Working directory does not exist: ${workdir}`);
  }

  // Build patch
  const contextMarker = params.function || '';
  const patch = `*** Begin Patch
*** Update File: ${params.file}
${contextMarker ? `@@ ${contextMarker}` : '@@'}
 ${params.context_1 || ''}
 ${params.context_2 || ''}
 ${params.context_3 || ''}
-${params.line_before}
+${params.line_after}
 ${params.context_4 || ''}
 ${params.context_5 || ''}
 ${params.context_6 || ''}
*** End Patch`;

  return executePatch(patch, workdir, res, params);
}

/**
 * Template-based patching
 */
function applyTemplate(params, res) {
  const templates = {
    'add_cache_headers': (p) => `*** Begin Patch
*** Update File: ${p.file}
@@ ${p.function || ''}
 ${p.context_before || ''}
-${p.old_line}
+const headers = addCacheHeaders(${p.headers_object || '{ "Content-Type": "text/html" }'});
+res.writeHead(200, headers);
 ${p.context_after || ''}
*** End Patch`,

    'simple_replace': (p) => `*** Begin Patch
*** Update File: ${p.file}
@@
 ${p.context_before || ''}
-${p.old_line}
+${p.new_line}
 ${p.context_after || ''}
*** End Patch`,

    'add_function': (p) => `*** Begin Patch
*** Update File: ${p.file}
@@
 ${p.context_before || ''}
+${p.function_body}
 ${p.context_after || ''}
*** End Patch`
  };

  const templateFn = templates[params.template];
  if (!templateFn) {
    return sendError(res, 400, `Unknown template: ${params.template}. Available: ${Object.keys(templates).join(', ')}`);
  }

  const patch = templateFn(params);
  const workdir = params.workdir || '/home/cicero-arch-omen/ai_sandbox/codex-agent/codex_sse/server-tools/MMSB';
  
  return executePatch(patch, workdir, res, params);
}

/**
 * Execute patch
 */
function executePatch(patch, workdir, res, params) {
  const tmpFile = path.join(os.tmpdir(), `patch_${Date.now()}.txt`);
  fs.writeFileSync(tmpFile, patch);

  try {
    const result = execSync(
      `cd "${workdir}" && "${APPLY_PATCH_BIN}" < "${tmpFile}"`,
      { 
        encoding: 'utf-8',
        maxBuffer: 10 * 1024 * 1024
      }
    );

    fs.unlinkSync(tmpFile);

    sendSuccess(res, {
      success: true,
      message: 'Patch applied successfully',
      output: result.trim(),
      workdir: workdir,
      patch_preview: patch.split('\n').slice(0, 10).join('\n') + '...'
    });

  } catch (execError) {
    if (fs.existsSync(tmpFile)) {
      fs.unlinkSync(tmpFile);
    }

    sendError(res, 400, `Patch failed: ${execError.message}`, execError.stderr);
  }
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

function sendError(res, statusCode, message, details = null) {
  res.writeHead(statusCode, {
    'Content-Type': 'application/json',
    'Cache-Control': 'no-cache, no-store, must-revalidate',
    'Pragma': 'no-cache',
    'Expires': '0'
  });
  res.end(JSON.stringify({
    success: false,
    error: message,
    details: details
  }, null, 2));
}

module.exports = { handleInstructionBuilder };
