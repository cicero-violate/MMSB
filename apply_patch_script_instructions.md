# apply_patch Script — LLM Instructions

This document defines **the only correct way** for an LLM (ChatGPT) to submit
changes to this repository.

The system is **patch-driven**, **append-only**, and **deterministic**.

Chat messages are NOT authoritative.
Only patches are.

---

## 1) Mental Model (MANDATORY)

Think of this repository as:

- A ledger
- A flight recorder
- An append-only log

You **do not edit files directly**.
You **emit patch files**.

The human operator applies them via:

```bash
./apply_patch_script.sh
```

or preflights them via:

```bash
./apply_patch_script.sh --check
```

---

## 2) What You (the LLM) Are Allowed to Send

You may only send:

- One or more `*.patch` files
- Each patch must be **self-contained**
- Each patch must use the `apply_patch` format

You must **never**:

- Describe changes without a patch
- Ask the human to “edit this manually”
- Assume prior patches applied unless confirmed

---

## 3) Patch File Naming Convention

Use **simple, explicit names**:

```
txt.patch
txt2.patch
txt3.patch
```

or

```
01-phase6-complete.patch
02-add-adapter-surface.patch
```

Order matters.
Each patch is applied **independently**.

---

## 4) Patch Format (STRICT)

Every patch MUST follow this structure exactly:

```
*** Begin Patch
*** Add File: <relative/path>
+file contents
*** End Patch
```

or

```
*** Begin Patch
*** Update File: <relative/path>
@@
-old lines
+new lines
*** End Patch
```

Rules:

- Paths are **relative only**
- Never use absolute paths
- Never omit the action header (Add / Update / Delete)
- Never reorder existing content unless explicitly instructed

---

## 5) Append-Only Rule (CRITICAL)

Unless explicitly told otherwise:

- You may **only append**
- You may **not rewrite history**
- You may **not clean up**
- You may **not normalize language**

If context does not match:

> Let the patch FAIL.

Do NOT weaken context to “make it apply”.

---

## 6) How to Send Patches in Chat

When responding, you must:

1. Clearly label each patch
2. Put each patch in its own code block
3. Do NOT mix explanation inside the patch

### Example

```
Here is txt.patch:

```patch
*** Begin Patch
*** Update File: README.md
@@
+New appended line
*** End Patch
```

Here is txt2.patch:

```patch
*** Begin Patch
*** Add File: notes.txt
+Some content
*** End Patch
```
```

---

## 7) Preflight Is Authoritative

The human will run:

```bash
./apply_patch_script.sh --check
```

Outcomes mean:

- **WOULD APPLY** → patch is valid
- **WOULD FAIL** → patch is obsolete or incorrect
- **ALREADY APPLIED** → patch is redundant

If your patch fails:

- Do NOT argue
- Do NOT explain around it
- Regenerate the patch against current state

---

## 8) Multiple Patches

If a change requires multiple patches:

- Split them cleanly
- One concern per patch
- Never assume ordering unless stated

The script will tell the human exactly what succeeded or failed.

---

## 9) Authority Boundary

The LLM:

- Proposes patches
- Never applies patches
- Never assumes state

The human:

- Runs preflight
- Applies patches
- Decides what is kept

---

## 10) Final Rule

If you are unsure:

> Emit a patch anyway and let it fail.

Failure is **signal**, not error.

Truth is preferred over convenience.

---

End of instructions.
