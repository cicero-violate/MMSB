## `update_plan`

You can create and update plans by using **code blocks with `plan` language tag**. The proxy will automatically extract and execute them.

**Format:**

```plan
EXPLANATION: Optional explanation of plan changes
1. [in_progress] Step description (5-7 words)
2. [pending] Next step description
3. [pending] Final step description
```

**The code block will be transformed into an update_plan function call.** Do NOT use ChatGPT's native `call_tool` function.

**Rules:**
- Each step description: 5-7 words maximum
- Status markers: `[pending]`, `[in_progress]`, `[completed]`
- Always have exactly ONE `[in_progress]` step at a time
- When updating: output a new code block with updated statuses
- Optional: Start with `EXPLANATION:` line to describe why the plan changed

**Example:**
```plan
EXPLANATION: Started directory scan
1. [completed] Scan directory contents  
2. [in_progress] Create summary file
3. [pending] Verify summary accuracy
```
