# ChatGPT Actions Agent Setup

## Quick Start

### 1. Configure ChatGPT Actions

In your ChatGPT Actions configuration:

**System Prompt:** Paste the entire contents of `../AGENTS.md`

**API Configuration:** Point to your OpenAPI schema at:
```
/home/cicero-arch-omen/ai_sandbox/chatgpt-website/chatgpt_actions/openapi.schema
```

### 2. Invoke Agent

**User Message:** Just type the role name:
```
Diagnostics Agent
```

That's it. The agent will:
1. Read its role playbook
2. Read DAG_PRIORITY.md and TASK_LOG.md
3. Execute the next task immediately
4. Write results to files
5. Report completion

## Available Roles

- `Diagnostics Agent` - Run tests, capture logs, update DIAGNOSTICS.md
- `Planning Agent` - Update DAG and schedule priorities
- `Rust Core Agent` - Build Rust code, sync libraries
- `Julia Agent` - Implement features, run tests

## Why This Works

**Problem:** ChatGPT's alignment training makes it conversational and hesitant.

**Solution:** 
- System prompt contains execution protocol
- System prompt shows forbidden behaviors (asking permission)
- System prompt shows correct behaviors (execute then report)
- User prompt is minimal (just role name)
- First instruction is "execute tools WITHOUT responding first"

**Key insight:** The system prompt explicitly tells the model that the first tokens in the response should be a tool call, not text.

## Troubleshooting

**If agent still asks permission:**
- Verify AGENTS.md is in system prompt field, not user message
- Check that OpenAPI schema is loaded correctly
- Ensure role name matches exactly (case-sensitive)

**If agent says it can't find files:**
- Verify `cwd` parameter in tool calls
- Check OpenAPI schema workspaceRoot setting

**If agent executes but doesn't write files:**
- Check apply_patch format in system prompt
- Verify authorization matrix permissions
- Look for tool call errors in response
