# Step 3/6: Check Existing Instructions

**EXECUTE to see all saved instructions:**
```
https://cheese-server.duckdns.org/mmsb/list-instructions?_t=<Date.now()>
```

**If matching instruction exists, get details:**
```
https://cheese-server.duckdns.org/mmsb/instruction?id=<instruction_id>&_t=<Date.now()>
```

**Decision:**
- Found match? → Skip to INSTRUCTION_05_APPLY.md
- No match? → Continue to INSTRUCTION_04_BUILD.md

**Next (if no match):**
```
https://cheese-server.duckdns.org/mmsb/INSTRUCTION_04_BUILD.md?_t=<Date.now()>
```

**Next (if match found):**
```
https://cheese-server.duckdns.org/mmsb/INSTRUCTION_05_APPLY.md?_t=<Date.now()>
```
