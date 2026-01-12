## Shell Intent Codeblocks

Shell codeblocks are used to **declare shell intent**, not to execute commands.

The system will extract shell codeblocks and convert them into
`shell.intent.v1` JSON deltas. **No execution is implied.**

---

## Format (Required)

Use a fenced codeblock with a shell language tag:

```sh
command goes here
````

or

```bash
command goes here
```

Both are accepted and normalized to `sh`.

---

## Critical Rules (Invariants)

* Shell codeblocks **do NOT execute**
* Shell codeblocks **must contain only shell text**
* Do NOT include explanations, comments, or narrative inside shell blocks
* All explanatory text must be **outside** the codeblock
* Shell codeblocks are **pure intent artifacts**

Violating these rules breaks intent extraction.

---

## Ordering

* Shell codeblocks are processed **in message order**
* Each block becomes a separate `shell.intent.v1` delta
* Ordering is semantic and must be preserved

---

## Execution (Explicit, External)

Shell commands are executed **only** by a dedicated runner
(e.g. `shell_runner`) that consumes intent deltas.

Execution is:

* opt-in
* explicit
* recorded via `shell.execution.v1`
* never triggered by message parsing

---

## Correct Examples

Single intent:

```sh
ls -la
```

Multiple intents in one message (allowed):

```sh
cargo build
```

```sh
cargo test
```

These produce **two intent deltas**, not execution.

---

## Incorrect Examples (Forbidden)

❌ Narrative inside shell block:

```sh
# now we list files
ls -la
```

❌ Assuming execution:

```sh
ls -la   # this will run
```

❌ Using shell blocks as scripts:

```sh
for f in *.rs; do
  echo $f
done
```

(Loops and scripts belong in execution artifacts, not intent.)

---

## Mental Model

> A shell codeblock is a **declarative request**, not an action.

Intent → Runner → Execution
never
Intent → Side effects

---

## Summary

* Shell codeblocks declare **what should happen**
* JSON deltas record **what was meant**
* Execution is **separate, explicit, and auditable**
