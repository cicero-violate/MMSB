# Step 5/6: Apply Instruction

**EXECUTE to apply the patch:**
```
https://cheese-server.duckdns.org/mmsb/apply-instruction?id=<instruction_id>&_t=<Date.now()>
```

**Check response:**
- `{"success": true}` → Continue
- `{"success": false, "error": "..."}` → Read error, fix context, rebuild

**Track application:**
```
https://cheese-server.duckdns.org/mmsb/add-instruction?task_id=<your_task_id>&instruction_id=<instruction_id>&action=applied&_t=<Date.now()>
```

**Next instruction:**
```
https://cheese-server.duckdns.org/mmsb/INSTRUCTION_06_COMPLETE.md?_t=<Date.now()>
```
