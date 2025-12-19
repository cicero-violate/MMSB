/**
 * Apply Instruction Endpoint
 * Executes saved instruction by ID
 */

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');
const os = require('os');

const APPLY_PATCH_BIN = '/home/cicero-arch-omen/ai_sandbox/tools/apply-patch/target/release/apply_patch';
const INSTRUCTIONS_DIR = '/home/cicero-arch-omen/ai_sandbox/codex-agent/codex_sse/server-tools/MMSB/instructions';
const MMSB_ROOT = '/home/cicero-arch-omen/ai_sandbox/codex-agent/codex_sse/server-tools/MMSB';

/**
 * Apply instruction by ID
 */
function handleApplyInstruction(params, res) {
  try {
    if (!params.id) {
      return sendError(res, 400, 'Missing required parameter: id');
    }

    const instructionDir = path.join(INSTRUCTIONS_DIR, params.id);
    if (!fs.existsSync(instructionDir)) {
      return sendError(res, 404, `Instruction ${params.id} not found`);
    }

    // Load instruction files
    const patchPath = path.join(instructionDir, 'patch.diff');
    const targetPath = path.join(instructionDir, 'target.json');
    const metaPath = path.join(instructionDir, 'meta.json');

    if (!fs.existsSync(patchPath)) {
      return sendError(res, 500, `Instruction ${params.id} missing patch.diff`);
    }

    const patch = fs.readFileSync(patchPath, 'utf-8');
    const target = fs.existsSync(targetPath) 
      ? JSON.parse(fs.readFileSync(targetPath, 'utf-8'))
      : { workdir: 'tools/website' };
    const meta = fs.existsSync(metaPath)
      ? JSON.parse(fs.readFileSync(metaPath, 'utf-8'))
      : {};

    // Determine workdir
    const workdir = path.join(MMSB_ROOT, target.workdir || '');
    if (!fs.existsSync(workdir)) {
      return sendError(res, 400, `Working directory does not exist: ${workdir}`);
    }

    // Execute patch
    const tmpFile = path.join(os.tmpdir(), `instruction_${params.id}_${Date.now()}.txt`);
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
        instruction: params.id,
        message: 'Instruction applied successfully',
        output: result.trim(),
        workdir: workdir,
        meta: meta
      });

    } catch (execError) {
      if (fs.existsSync(tmpFile)) {
        fs.unlinkSync(tmpFile);
      }

      sendError(res, 400, `Instruction failed: ${execError.message}`, execError.stderr);
    }

  } catch (err) {
    sendError(res, 500, `Internal error: ${err.message}`);
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

module.exports = { handleApplyInstruction };
