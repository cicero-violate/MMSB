/**
 * Task Manager Endpoints
 * Track LLM tasks and associate them with instructions
 */

const fs = require('fs');
const path = require('path');

const TASKS_DIR = '/home/cicero-arch-omen/ai_sandbox/codex-agent/codex_sse/server-tools/MMSB/tasks';

/**
 * Create new task
 */
function handleCreateTask(params, res) {
  try {
    if (!params.task_id) {
      return sendError(res, 400, 'Missing required parameter: task_id');
    }

    const taskDir = path.join(TASKS_DIR, params.task_id);
    if (fs.existsSync(taskDir)) {
      return sendError(res, 400, `Task ${params.task_id} already exists`);
    }

    fs.mkdirSync(taskDir, { recursive: true });

    const task = {
      id: params.task_id,
      description: params.description || '',
      status: 'created',
      created: new Date().toISOString(),
      instructions: [],
      context_queries: [],
      result: null
    };

    fs.writeFileSync(
      path.join(taskDir, 'task.json'),
      JSON.stringify(task, null, 2)
    );

    updateTaskIndex(params.task_id, task);

    sendSuccess(res, {
      success: true,
      task_id: params.task_id,
      message: 'Task created'
    });

  } catch (err) {
    sendError(res, 500, `Failed to create task: ${err.message}`);
  }
}

/**
 * Update task with context query
 */
function handleAddContext(params, res) {
  try {
    if (!params.task_id || !params.query_url) {
      return sendError(res, 400, 'Missing required parameters: task_id, query_url');
    }

    const taskPath = path.join(TASKS_DIR, params.task_id, 'task.json');
    if (!fs.existsSync(taskPath)) {
      return sendError(res, 404, `Task ${params.task_id} not found`);
    }

    const task = JSON.parse(fs.readFileSync(taskPath, 'utf-8'));
    task.context_queries.push({
      url: params.query_url,
      timestamp: new Date().toISOString()
    });

    fs.writeFileSync(taskPath, JSON.stringify(task, null, 2));

    sendSuccess(res, {
      success: true,
      context_count: task.context_queries.length
    });

  } catch (err) {
    sendError(res, 500, `Failed to add context: ${err.message}`);
  }
}

/**
 * Associate instruction with task
 */
function handleAddInstruction(params, res) {
  try {
    if (!params.task_id || !params.instruction_id) {
      return sendError(res, 400, 'Missing required parameters: task_id, instruction_id');
    }

    const taskPath = path.join(TASKS_DIR, params.task_id, 'task.json');
    if (!fs.existsSync(taskPath)) {
      return sendError(res, 404, `Task ${params.task_id} not found`);
    }

    const task = JSON.parse(fs.readFileSync(taskPath, 'utf-8'));
    task.instructions.push({
      id: params.instruction_id,
      timestamp: new Date().toISOString(),
      action: params.action || 'created'
    });
    task.status = 'in_progress';

    fs.writeFileSync(taskPath, JSON.stringify(task, null, 2));

    sendSuccess(res, {
      success: true,
      instruction_count: task.instructions.length
    });

  } catch (err) {
    sendError(res, 500, `Failed to add instruction: ${err.message}`);
  }
}

/**
 * Complete task
 */
function handleCompleteTask(params, res) {
  try {
    if (!params.task_id) {
      return sendError(res, 400, 'Missing required parameter: task_id');
    }

    const taskPath = path.join(TASKS_DIR, params.task_id, 'task.json');
    if (!fs.existsSync(taskPath)) {
      return sendError(res, 404, `Task ${params.task_id} not found`);
    }

    const task = JSON.parse(fs.readFileSync(taskPath, 'utf-8'));
    task.status = params.status || 'completed';
    task.result = params.result || 'success';
    task.completed = new Date().toISOString();

    fs.writeFileSync(taskPath, JSON.stringify(task, null, 2));
    updateTaskIndex(params.task_id, task);

    sendSuccess(res, {
      success: true,
      task_id: params.task_id,
      status: task.status
    });

  } catch (err) {
    sendError(res, 500, `Failed to complete task: ${err.message}`);
  }
}

/**
 * List all tasks
 */
function handleListTasks(params, res) {
  try {
    const indexPath = path.join(TASKS_DIR, 'index.json');
    
    if (!fs.existsSync(indexPath)) {
      return sendSuccess(res, {
        tasks: {},
        count: 0
      });
    }

    const index = JSON.parse(fs.readFileSync(indexPath, 'utf-8'));
    sendSuccess(res, {
      tasks: index,
      count: Object.keys(index).length
    });

  } catch (err) {
    sendError(res, 500, `Failed to list tasks: ${err.message}`);
  }
}

/**
 * Get task details
 */
function handleGetTask(params, res) {
  try {
    if (!params.task_id) {
      return sendError(res, 400, 'Missing required parameter: task_id');
    }

    const taskPath = path.join(TASKS_DIR, params.task_id, 'task.json');
    if (!fs.existsSync(taskPath)) {
      return sendError(res, 404, `Task ${params.task_id} not found`);
    }

    const task = JSON.parse(fs.readFileSync(taskPath, 'utf-8'));
    sendSuccess(res, task);

  } catch (err) {
    sendError(res, 500, `Failed to get task: ${err.message}`);
  }
}

/**
 * Update task index
 */
function updateTaskIndex(taskId, task) {
  const indexPath = path.join(TASKS_DIR, 'index.json');
  
  let index = {};
  if (fs.existsSync(indexPath)) {
    index = JSON.parse(fs.readFileSync(indexPath, 'utf-8'));
  }

  index[taskId] = {
    description: task.description,
    status: task.status,
    created: task.created,
    completed: task.completed || null,
    instruction_count: task.instructions ? task.instructions.length : 0
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

module.exports = {
  handleCreateTask,
  handleAddContext,
  handleAddInstruction,
  handleCompleteTask,
  handleListTasks,
  handleGetTask
};
