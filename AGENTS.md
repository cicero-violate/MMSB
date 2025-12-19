# MMSB Agent Instructions

You are working with the MMSB (Memory-Mapped Shared Bus) project. This file provides instructions for agent workflows.

## Core Principle

**Execute GET requests—don't just describe them.** When you see workflow steps, use web_fetch to call each URL.

## Workflow Endpoints

Base URL: `https://cheese-server.duckdns.org/mmsb`

### Task Management

1. **Create task**: `/create-task?task_id=TASK_XXX&description=...&_t=<timestamp>`
2. **Track context**: `/add-context?task_id=TASK_XXX&query_url=...&_t=<timestamp>`
3. **Link instruction**: `/add-instruction?task_id=TASK_XXX&instruction_id=...&action=created&_t=<timestamp>`
4. **Complete task**: `/complete-task?task_id=TASK_XXX&status=completed&result=success&_t=<timestamp>`

### Instruction System

1. **List instructions**: `/list-instructions?_t=<timestamp>` - Check for existing solutions
2. **Get instruction**: `/instruction?id=XXX&_t=<timestamp>` - Review details
3. **Build patch**: `/build-patch?template=simple_replace&file=X&old_line=A&new_line=B&context_before=C&context_after=D&_t=<timestamp>`
4. **Save instruction**: `/save-instruction?id=XXX&<build_params>&purpose=...&_t=<timestamp>`
5. **Apply instruction**: `/apply-instruction?id=XXX&_t=<timestamp>`

### Templates

**simple_replace**: `?template=simple_replace&file=X&old_line=A&new_line=B&context_before=C&context_after=D`
**add_cache_headers**: `?template=add_cache_headers&file=X&function=fname&old_line=A&context_before=C&context_after=D`
**Direct mode**: `?file=X&line_before=A&line_after=B&context_1=C1&context_2=C2&context_3=C3`

## Standard Workflow

### For Creating New Files

1. Create task: `TASK_CREATE_<filename>`
2. Check existing instructions: `/list-instructions`
3. Build patch with Add File syntax or empty line_before
4. Save as instruction: `CREATE_<filename>`
5. Apply instruction
6. Complete task

### For Modifying Files

1. Create task: `TASK_MODIFY_<area>`
2. Gather context (2-3 queries):
   - Read current file: `/mmsb/path/to/file?_t=<timestamp>`
   - Track query: `/add-context?task_id=...&query_url=...`
3. Check existing instructions: `/list-instructions`
4. If exists: reuse via `/apply-instruction?id=...`
5. If new: build → save → apply
6. Complete task

## Critical Rules

1. **Always generate fresh timestamps**: Use current milliseconds (Date.now() in JS)
2. **URL-encode all parameters**: Spaces = `%20`, newlines = `%0A`
3. **Check response.success**: Read error messages if false, adjust and retry
4. **Track every action**: Use task endpoints to log queries and instructions
5. **Reuse proven solutions**: Always check `/list-instructions` first

## Example: Create Test3.jl

Execute these GET requests in sequence:

```
1. https://cheese-server.duckdns.org/mmsb/create-task?task_id=TASK_CREATE_TEST3&description=Create%20Test3.jl&_t=1734567890000

2. https://cheese-server.duckdns.org/mmsb/list-instructions?_t=1734567890001

3. https://cheese-server.duckdns.org/mmsb/save-instruction?id=CREATE_TEST3_JL&file=Test3.jl&line_before=&line_after=module%20Test3%0Ausing%20Test%0A%23%20TODO%3A%20tests%0Aend&purpose=Create%20Test3.jl&_t=1734567890002

4. https://cheese-server.duckdns.org/mmsb/apply-instruction?id=CREATE_TEST3_JL&_t=1734567890003

5. https://cheese-server.duckdns.org/mmsb/add-instruction?task_id=TASK_CREATE_TEST3&instruction_id=CREATE_TEST3_JL&action=applied&_t=1734567890004

6. https://cheese-server.duckdns.org/mmsb/complete-task?task_id=TASK_CREATE_TEST3&status=completed&result=success&_t=1734567890005
```

## Directory Structure

```
instructions/     # Saved patches (reusable)
  FIX_001/
    patch.diff
    target.json
    meta.json
  index.json

tasks/           # Session logs (auditability)
  TASK_001/
    task.json
  index.json
```

## Response Verification

After each GET, check JSON response:
- `{"success": true}` → proceed
- `{"success": false, "error": "..."}` → read error, adjust, retry

Never proceed after an error without fixing the cause.
