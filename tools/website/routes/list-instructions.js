/**
 * List Instructions Endpoint
 * Returns index of all saved instructions
 */

const fs = require('fs');
const path = require('path');

const INSTRUCTIONS_DIR = '/home/cicero-arch-omen/ai_sandbox/codex-agent/codex_sse/server-tools/MMSB/instructions';

/**
 * List all instructions
 */
function handleListInstructions(params, res) {
  try {
    const indexPath = path.join(INSTRUCTIONS_DIR, 'index.json');
    
    if (!fs.existsSync(indexPath)) {
      return sendSuccess(res, {
        instructions: {},
        count: 0,
        message: 'No instructions found'
      });
    }

    const index = JSON.parse(fs.readFileSync(indexPath, 'utf-8'));
    
    sendSuccess(res, {
      instructions: index,
      count: Object.keys(index).length
    });

  } catch (err) {
    sendError(res, 500, `Failed to list instructions: ${err.message}`);
  }
}

/**
 * Get single instruction details
 */
function handleGetInstruction(params, res) {
  try {
    if (!params.id) {
      return sendError(res, 400, 'Missing required parameter: id');
    }

    const instructionDir = path.join(INSTRUCTIONS_DIR, params.id);
    if (!fs.existsSync(instructionDir)) {
      return sendError(res, 404, `Instruction ${params.id} not found`);
    }

    // Read all instruction files
    const patch = fs.readFileSync(path.join(instructionDir, 'patch.diff'), 'utf-8');
    const target = JSON.parse(fs.readFileSync(path.join(instructionDir, 'target.json'), 'utf-8'));
    const meta = JSON.parse(fs.readFileSync(path.join(instructionDir, 'meta.json'), 'utf-8'));

    sendSuccess(res, {
      id: params.id,
      patch: patch,
      target: target,
      meta: meta
    });

  } catch (err) {
    sendError(res, 500, `Failed to get instruction: ${err.message}`);
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

module.exports = { handleListInstructions, handleGetInstruction };
