/**
 * Apply Patch Endpoint
 * Executes apply_patch tool via GET request with patch content
 */

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');
const os = require('os');

const APPLY_PATCH_BIN = '/home/cicero-arch-omen/ai_sandbox/tools/apply-patch/target/release/apply_patch';

/**
 * Handle apply-patch GET request
 * Query params: patch (base64 or raw), workdir (optional)
 */
function handleApplyPatch(params, res) {
  try {
    // Validate patch parameter
    if (!params.patch) {
      return sendError(res, 400, 'Missing required parameter: patch');
    }

    // Decode patch if base64
    let patchContent = params.patch;
    if (params.encoding === 'base64') {
      patchContent = Buffer.from(params.patch, 'base64').toString('utf-8');
    }

    // Validate patch format
    if (!patchContent.includes('*** Begin Patch')) {
      return sendError(res, 400, 'Invalid patch format: must start with "*** Begin Patch"');
    }

    // Determine working directory
    const workdir = params.workdir || '/home/cicero-arch-omen/ai_sandbox/codex-agent/codex_sse/server-tools/MMSB';
    
    // Validate workdir exists
    if (!fs.existsSync(workdir)) {
      return sendError(res, 400, `Working directory does not exist: ${workdir}`);
    }

    // Create temporary file for patch
    const tmpFile = path.join(os.tmpdir(), `patch_${Date.now()}.txt`);
    fs.writeFileSync(tmpFile, patchContent);

    try {
      // Execute apply_patch
      const result = execSync(
        `cd "${workdir}" && "${APPLY_PATCH_BIN}" < "${tmpFile}"`,
        { 
          encoding: 'utf-8',
          maxBuffer: 10 * 1024 * 1024 // 10MB
        }
      );

      // Clean up temp file
      fs.unlinkSync(tmpFile);

      // Return success
      sendSuccess(res, {
        success: true,
        message: 'Patch applied successfully',
        output: result.trim(),
        workdir: workdir
      });

    } catch (execError) {
      // Clean up temp file
      if (fs.existsSync(tmpFile)) {
        fs.unlinkSync(tmpFile);
      }

      // Return execution error
      sendError(res, 400, `Patch failed: ${execError.message}`, execError.stderr);
    }

  } catch (err) {
    sendError(res, 500, `Internal error: ${err.message}`);
  }
}

function sendSuccess(res, data) {
  const response = JSON.stringify(data, null, 2);
  res.writeHead(200, {
    'Content-Type': 'application/json',
    'Cache-Control': 'no-cache, no-store, must-revalidate',
    'Pragma': 'no-cache',
    'Expires': '0'
  });
  res.end(response);
}

function sendError(res, statusCode, message, details = null) {
  const response = {
    success: false,
    error: message,
    details: details
  };
  res.writeHead(statusCode, {
    'Content-Type': 'application/json',
    'Cache-Control': 'no-cache, no-store, must-revalidate',
    'Pragma': 'no-cache',
    'Expires': '0'
  });
  res.end(JSON.stringify(response, null, 2));
}

module.exports = { handleApplyPatch };
